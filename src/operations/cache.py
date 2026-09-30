"""Disposable, content-addressed cache for fetched light-curve segments.

The cache is a performance optimization only.  A hit is used solely to answer
the segment callback of :meth:`SegmentedFetchCheckpoint.fetch_missing`; the
checkpoint still normalizes, hashes, and commits its own evidence, so cache
content never becomes authority merely because a file exists.  Every entry is
verified (object SHA-256, header identity, Parquet schema identity, row
accounting, and per-object locus ownership and order) before reuse, and a
failing entry is dropped and re-fetched.  Deleting the whole cache at any time
loses no scientific authority.

Confinement: the root must be a real directory that neither equals nor
overlaps production data, the Sentinel V2 cache path (which must stay
absent), canary/recovery evidence, migration audits, or any root of the
capability using it (checkpoints, staging, journals, locks, evidence).  On
shire the only acceptable root is :data:`PROPOSED_ARNOR_CACHE_ROOT`, exactly;
it is never created implicitly and is not populated by this release.
Internal directories and files are opened without following symlinks and
their resolved containment is re-proven at every write.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.cli_profiles import (
    MIDDLE_EARTH_CACHE_ROOT,
    MIDDLE_EARTH_CANARY_ROOT,
    MIDDLE_EARTH_DATA_ROOT,
    MIDDLE_EARTH_MIGRATION_AUDITS_ROOT,
    MIDDLE_EARTH_PROJECT_ROOT,
)

from .fetch_checkpoint import (
    FetchCheckpointBinding,
    FetchObjectResult,
    _parquet_payload,
    _read_parquet,
)


SEGMENT_CACHE_SCHEMA = "v3.fetch-segment-cache.v1"
PROPOSED_ARNOR_CACHE_ROOT = Path("/astro/store/shire/ANTARES/work/cache/fetch-segments-v1")
_FORBIDDEN_ROOTS = (
    MIDDLE_EARTH_DATA_ROOT,
    MIDDLE_EARTH_CACHE_ROOT,
    MIDDLE_EARTH_CANARY_ROOT,
    MIDDLE_EARTH_MIGRATION_AUDITS_ROOT,
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


class SegmentCacheRefused(RuntimeError):
    """The requested cache location would violate authority boundaries."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _real_directory(path: Path, parent: Path) -> Path:
    """Create or re-prove one internal directory directly beneath ``parent``."""
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    observed = os.lstat(path)
    if not stat.S_ISDIR(observed.st_mode) or path.resolve(strict=True) != parent / path.name:
        raise SegmentCacheRefused(f"Cache directory {path} is not a real contained directory.")
    return path


class SegmentCache:
    """Content-addressed segment store: ``index/<key>`` -> ``objects/<sha256>``."""

    def __init__(
        self,
        root: Path,
        *,
        forbidden_roots: Sequence[Path] = _FORBIDDEN_ROOTS,
    ) -> None:
        lexical = Path(os.path.abspath(os.fspath(root)))
        if lexical.is_symlink() or not lexical.is_dir():
            raise SegmentCacheRefused("Cache root must be an existing real directory.")
        resolved = lexical.resolve(strict=True)
        on_shire = _overlaps(resolved, MIDDLE_EARTH_PROJECT_ROOT) or _overlaps(
            lexical, MIDDLE_EARTH_PROJECT_ROOT
        )
        if on_shire and (lexical != PROPOSED_ARNOR_CACHE_ROOT or resolved != lexical):
            raise SegmentCacheRefused(
                f"The only acceptable ANTARES cache root is {PROPOSED_ARNOR_CACHE_ROOT}."
            )
        self.root = resolved
        self.require_disjoint(forbidden_roots)
        self.objects = _real_directory(resolved / "objects", resolved)
        self.index = _real_directory(resolved / "index", resolved)
        self.tmp = _real_directory(resolved / "tmp", resolved)
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "rejected": 0, "stored": 0}

    def require_disjoint(self, protected: Iterable[Path]) -> None:
        """Refuse if the cache equals or nests with any authority root."""
        for path in protected:
            forbidden = Path(path).expanduser().resolve(strict=False)
            if _overlaps(self.root, forbidden):
                raise SegmentCacheRefused(
                    f"Cache root {self.root} overlaps protected root {forbidden}."
                )

    @staticmethod
    def segment_key(binding: FetchCheckpointBinding, locus_ids: Sequence[str]) -> str:
        """Bind a segment to every acquisition identity except the run root name."""
        identity = binding.as_dict()
        identity.pop("run_id", None)
        return _sha256(
            _canonical(
                {
                    "schema_version": SEGMENT_CACHE_SCHEMA,
                    "binding": identity,
                    "locus_ids": [str(value) for value in locus_ids],
                }
            )
        )

    def _count(self, name: str) -> None:
        with self._lock:
            self.stats[name] += 1

    def _shard(self, parent: Path, digest: str, *, create: bool = True) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SegmentCacheRefused("Cache identity is not a SHA-256 digest.")
        if not create:
            return parent / digest[:2] / digest
        return _real_directory(parent / digest[:2], parent) / digest

    def _read(self, path: Path) -> bytes:
        if path.parent.resolve(strict=True) != self.root / path.parent.parent.name / path.parent.name:
            raise SegmentCacheRefused("Cache entry escaped its directory.")
        descriptor = os.open(str(path), os.O_RDONLY | _NOFOLLOW)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise SegmentCacheRefused("Cache entry is not a regular file.")
            chunks = []
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    return b"".join(chunks)
                chunks.append(block)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        """Write via ``tmp/`` and rename, re-proving containment at the boundary."""
        _real_directory(self.tmp, self.root)
        expected_parent = _real_directory(path.parent, path.parent.parent)
        if expected_parent.resolve(strict=True).parent.parent != self.root:
            raise SegmentCacheRefused("Cache write escaped the cache root.")
        temporary = self.tmp / f"{uuid.uuid4().hex}.tmp"
        descriptor = os.open(
            str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600
        )
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if path.is_symlink():
            raise SegmentCacheRefused("Cache entry is a symlink substitution.")
        os.replace(temporary, path)

    def _reject(self, key: str) -> None:
        self._count("rejected")
        try:
            path = self._shard(self.index, key, create=False)
            if path.is_file() and not path.is_symlink():
                path.unlink()
        except (FileNotFoundError, SegmentCacheRefused):
            pass

    def get(self, key: str, locus_ids: Sequence[str]) -> Optional[Tuple[FetchObjectResult, ...]]:
        try:
            index_path = self._shard(self.index, key, create=False)
        except SegmentCacheRefused:
            self._count("rejected")
            return None
        if not index_path.exists() and not index_path.is_symlink():
            self._count("misses")
            return None
        try:
            digest = self._read(index_path).decode("ascii").strip()
            payload = self._read(self._shard(self.objects, digest, create=False))
            if _sha256(payload) != digest:
                raise ValueError("object digest mismatch")
            header_bytes, parquet = payload.split(b"\n", 1)
            header = json.loads(header_bytes)
            objects = header["objects"]
            if (
                header.get("schema_version") != SEGMENT_CACHE_SCHEMA
                or header.get("key") != key
                or _sha256(parquet) != header.get("alerts_sha256")
                or [item["locus_id"] for item in objects] != [str(v) for v in locus_ids]
            ):
                raise ValueError("header identity mismatch")
            alerts = _read_parquet(parquet, expected_schema_sha256=header["schema_sha256"])
            if sum(int(item["alert_rows"]) for item in objects) != len(alerts):
                raise ValueError("row accounting mismatch")
            results: List[FetchObjectResult] = []
            cursor = 0
            for item in objects:
                rows = int(item["alert_rows"])
                if rows < 0:
                    raise ValueError("negative row count")
                frame = None
                if rows:
                    frame = alerts.iloc[cursor : cursor + rows].reset_index(drop=True)
                    owners = frame["locus_id"] if "locus_id" in frame.columns else None
                    if owners is None or owners.isna().any() or not owners.astype(str).eq(
                        str(item["locus_id"])
                    ).all():
                        raise ValueError("alert rows are not owned by their object")
                cursor += rows
                results.append(
                    FetchObjectResult(
                        str(item["locus_id"]),
                        frame,
                        retry_count=int(item["retry_count"]),
                        retry_exception_types=tuple(item["retry_exception_types"]),
                    )
                )
        except Exception:
            self._reject(key)
            return None
        self._count("hits")
        return tuple(results)

    def put(self, key: str, results: Sequence[FetchObjectResult]) -> str:
        frames = []
        objects = []
        for value in results:
            rows = 0
            if value.alerts is not None and not value.alerts.empty:
                frame = value.alerts.copy(deep=True)
                if "locus_id" not in frame.columns:
                    frame["locus_id"] = value.locus_id
                frames.append(frame)
                rows = len(frame)
            objects.append(
                {
                    "locus_id": value.locus_id,
                    "alert_rows": rows,
                    "retry_count": value.retry_count,
                    "retry_exception_types": list(value.retry_exception_types),
                }
            )
        alerts = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame({"locus_id": pd.Series(dtype="object")})
        )
        parquet, schema_sha = _parquet_payload(alerts)
        header = {
            "schema_version": SEGMENT_CACHE_SCHEMA,
            "key": key,
            "objects": objects,
            "alerts_sha256": _sha256(parquet),
            "schema_sha256": schema_sha,
        }
        payload = _canonical(header) + b"\n" + parquet
        digest = _sha256(payload)
        self._atomic_write(self._shard(self.objects, digest), payload)
        self._atomic_write(self._shard(self.index, key), (digest + "\n").encode("ascii"))
        self._count("stored")
        return digest

    def wrap(
        self,
        binding: FetchCheckpointBinding,
        fetch_segment: Callable[[Tuple[str, ...]], Any],
    ) -> Callable[[Tuple[str, ...]], Tuple[FetchObjectResult, ...]]:
        """Return a segment callback that serves verified hits, else fetches and stores."""

        def cached(locus_ids: Tuple[str, ...]) -> Tuple[FetchObjectResult, ...]:
            key = self.segment_key(binding, locus_ids)
            hit = self.get(key, locus_ids)
            if hit is not None:
                return hit
            fetched = tuple(fetch_segment(locus_ids))
            self.put(key, fetched)
            return fetched

        return cached

    def statistics(self) -> Mapping[str, int]:
        with self._lock:
            return dict(self.stats)


__all__ = [
    "PROPOSED_ARNOR_CACHE_ROOT",
    "SEGMENT_CACHE_SCHEMA",
    "SegmentCache",
    "SegmentCacheRefused",
]
