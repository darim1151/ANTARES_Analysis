"""Run-local, non-authoritative checkpoints for expensive ANTARES fetches.

This module deliberately knows nothing about production publication.  A
checkpoint is rooted below one sealed live-read capability, contains no path to
the production data or cache namespaces, and never calls ANTARES itself.  The
caller supplies a segment callback, which makes the implementation fully
offline-testable and keeps the existing provider transport policy in charge of
network access.

Durability is represented by immutable, content-addressed Parquet blobs and a
canonical segment receipt committed *after* its blob.  A final completion
record is committed only after every deterministic contiguous segment has been
reopened and verified.  Files left in ``tmp`` and unreferenced valid blobs are
non-committed operational debris; they are ignored and never deleted here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import pandas as pd

CHECKPOINT_SCHEMA_VERSION = "phase6.segmented-fetch-checkpoint.v1"
SEGMENT_SCHEMA_VERSION = "phase6.segmented-fetch-segment.v1"
COMPLETION_SCHEMA_VERSION = "phase6.segmented-fetch-complete.v1"
CHECKPOINT_DIRECTORY = Path("checkpoints") / "live-fetch-v1"
MAX_SEGMENT_OBJECTS = 1024
MAX_SEGMENT_ALERT_ROWS = 2_000_000
MAX_SEGMENT_MEMORY_BYTES = 512 * 1024 * 1024
MAX_SEGMENT_PARQUET_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_BLOB_NAME = re.compile(r"^[0-9a-f]{64}\.parquet$")
_SEGMENT_NAME = re.compile(
    r"^segment-(?P<index>[0-9]{8})-(?P<start>[0-9]{12})-(?P<stop>[0-9]{12})\.commit\.json$"
)


class FetchCheckpointError(RuntimeError):
    """Base failure for run-local fetch checkpoint operations."""


class FetchCheckpointBindingError(FetchCheckpointError):
    """The requested checkpoint does not match its sealed run/query identity."""


class FetchCheckpointCorrupt(FetchCheckpointError):
    """Committed checkpoint evidence is malformed or fails integrity checks."""


class FetchCheckpointAmbiguous(FetchCheckpointCorrupt):
    """Multiple or unexpected durable states prevent deterministic reuse."""


class FetchCheckpointIncomplete(FetchCheckpointError):
    """An operation requiring the final completion marker was requested early."""


class FetchCheckpointFetchError(FetchCheckpointError):
    """The caller's segment fetch did not return one complete exact result set."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FetchCheckpointError("Checkpoint evidence is not canonical JSON.") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identifier_sha256(values: Sequence[str]) -> str:
    """Match the live provider's newline-delimited ordered-identity digest."""
    digest = hashlib.sha256()
    for value in values:
        text = str(value)
        if not text or "\n" in text or "\r" in text:
            raise FetchCheckpointBindingError(
                "Checkpoint locus identities must be non-empty single-line strings."
            )
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_hash(value: object, label: str, pattern: re.Pattern[str]) -> str:
    text = str(value)
    if pattern.fullmatch(text) is None:
        raise FetchCheckpointBindingError(f"{label} is not a canonical digest.")
    return text


def _require_component(value: object, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or len(Path(text).parts) != 1
    ):
        raise FetchCheckpointBindingError(f"{label} must be one safe path component.")
    return text


def _lexical_child(root: Path, relative: Path) -> Path:
    """Return a contained lexical child without resolving an in-tree symlink."""
    root = Path(root)
    relative = Path(relative)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise FetchCheckpointBindingError("Checkpoint path is not a safe relative path.")
    canonical_root = root.resolve(strict=True)
    candidate = Path(os.path.abspath(os.fspath(canonical_root / relative)))
    try:
        candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise FetchCheckpointBindingError(
            "Checkpoint path escaped its sealed run root."
        ) from exc
    return candidate


@dataclass(frozen=True)
class FetchCheckpointBinding:
    """Immutable identity to which every segment and completion record is bound."""

    run_id: str
    release_sha: str
    configuration_sha256: str
    target_date_utc: str
    mjd_min: float
    mjd_max: float
    provider_name: str
    provider_scenario: str
    provider_policy_sha256: str
    query_contract_sha256: str
    query_identity_sha256: str
    query_locus_order_sha256: str
    expected_objects: int
    segment_size: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_component(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "release_sha",
            _require_hash(self.release_sha, "release_sha", _HEX_40),
        )
        for field_name in (
            "configuration_sha256",
            "provider_policy_sha256",
            "query_contract_sha256",
            "query_identity_sha256",
            "query_locus_order_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_hash(getattr(self, field_name), field_name, _HEX_64),
            )
        try:
            parsed = date.fromisoformat(self.target_date_utc)
        except (TypeError, ValueError) as exc:
            raise FetchCheckpointBindingError(
                "target_date_utc must use canonical YYYY-MM-DD form."
            ) from exc
        if parsed.isoformat() != self.target_date_utc:
            raise FetchCheckpointBindingError(
                "target_date_utc must use canonical YYYY-MM-DD form."
            )
        for field_name in ("mjd_min", "mjd_max"):
            value = getattr(self, field_name)
            if isinstance(value, bool):
                raise FetchCheckpointBindingError("MJD bounds must be finite numbers.")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise FetchCheckpointBindingError(
                    "MJD bounds must be finite numbers."
                ) from exc
            if not math.isfinite(normalized):
                raise FetchCheckpointBindingError("MJD bounds must be finite numbers.")
            object.__setattr__(self, field_name, normalized)
        if self.mjd_min >= self.mjd_max:
            raise FetchCheckpointBindingError("mjd_min must be below mjd_max.")
        if self.provider_name != "live-antares":
            raise FetchCheckpointBindingError(
                "Segmented fetch checkpoints are restricted to live-antares."
            )
        if self.provider_scenario != "commissioning-v1":
            raise FetchCheckpointBindingError(
                "Unexpected live ANTARES provider scenario."
            )
        if type(self.expected_objects) is not int or self.expected_objects < 0:
            raise FetchCheckpointBindingError(
                "expected_objects must be a non-negative integer."
            )
        if (
            type(self.segment_size) is not int
            or self.segment_size <= 0
            or self.segment_size > MAX_SEGMENT_OBJECTS
        ):
            raise FetchCheckpointBindingError(
                f"segment_size must be between 1 and {MAX_SEGMENT_OBJECTS}."
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "release_sha": self.release_sha,
            "configuration_sha256": self.configuration_sha256,
            "target_date_utc": self.target_date_utc,
            "mjd_min": self.mjd_min,
            "mjd_max": self.mjd_max,
            "provider_name": self.provider_name,
            "provider_scenario": self.provider_scenario,
            "provider_policy_sha256": self.provider_policy_sha256,
            "query_contract_sha256": self.query_contract_sha256,
            "query_identity_sha256": self.query_identity_sha256,
            "query_locus_order_sha256": self.query_locus_order_sha256,
            "expected_objects": self.expected_objects,
            "segment_size": self.segment_size,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256(_canonical_json(self.as_dict()))


@dataclass(frozen=True, eq=False)
class FetchObjectResult:
    """One successfully fetched full-history result returned by the caller."""

    locus_id: str
    alerts: Optional[pd.DataFrame]
    retry_count: int = 0
    retry_exception_types: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        locus_id = str(self.locus_id)
        if not locus_id or "\n" in locus_id or "\r" in locus_id:
            raise ValueError("Fetched locus_id must be a non-empty single-line string.")
        object.__setattr__(self, "locus_id", locus_id)
        if self.alerts is not None and not isinstance(self.alerts, pd.DataFrame):
            raise TypeError("Fetched alerts must be a pandas DataFrame or null.")
        if type(self.retry_count) is not int or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer.")
        if any(
            not isinstance(value, str) or not value
            for value in self.retry_exception_types
        ):
            raise ValueError("Retry exception types must be non-empty strings.")
        normalized = tuple(sorted(set(self.retry_exception_types)))
        if self.retry_count == 0 and normalized:
            raise ValueError("Retry types require a positive retry count.")
        object.__setattr__(self, "retry_exception_types", normalized)


@dataclass(frozen=True)
class FetchSegmentPlan:
    index: int
    start: int
    stop: int
    locus_ids: Tuple[str, ...]

    @property
    def key(self) -> str:
        return f"segment-{self.index:08d}-{self.start:012d}-{self.stop:012d}"

    @property
    def commit_name(self) -> str:
        return self.key + ".commit.json"


@dataclass(frozen=True)
class FetchSegmentRecord:
    plan: FetchSegmentPlan
    receipt: Mapping[str, Any]
    receipt_sha256: str
    blob_path: Path


@dataclass(frozen=True, eq=False)
class ReopenedFetchSegment:
    plan: FetchSegmentPlan
    alerts: pd.DataFrame
    objects: Tuple[Mapping[str, Any], ...]
    receipt_sha256: str


@dataclass(frozen=True)
class FetchCheckpointCompletion:
    checkpoint_root: Path
    checkpoint_identity_sha256: str
    requested_objects: int
    completed_objects: int
    segment_count: int
    alert_rows: int
    retry_count: int
    retry_exception_types: Tuple[str, ...]
    completion_sha256: str
    reused_segments: int
    fetched_segments: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "checkpoint_root": str(self.checkpoint_root),
            "checkpoint_identity_sha256": self.checkpoint_identity_sha256,
            "requested_objects": self.requested_objects,
            "completed_objects": self.completed_objects,
            "failed_objects": 0,
            "full_locus_history_requests": self.requested_objects,
            "full_locus_history_completed": self.completed_objects,
            "segment_count": self.segment_count,
            "alert_rows": self.alert_rows,
            "retry_count": self.retry_count,
            "retry_exception_types": list(self.retry_exception_types),
            "completion_sha256": self.completion_sha256,
            "reused_segments": self.reused_segments,
            "fetched_segments": self.fetched_segments,
            "completed": True,
            "partial": False,
            "authoritative": False,
            "production_eligible": False,
        }


SegmentFetcher = Callable[[Tuple[str, ...]], Iterable[FetchObjectResult]]
CheckpointEventHook = Callable[[str, Mapping[str, Any]], None]


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FetchCheckpointCorrupt("Checkpoint JSON contains duplicate keys.")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except FetchCheckpointCorrupt:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FetchCheckpointCorrupt(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise FetchCheckpointCorrupt(f"{label} is not a JSON object.")
    if _canonical_json(value) != payload:
        raise FetchCheckpointCorrupt(f"{label} is not canonical JSON.")
    return value


def _directory_fd(path: Path) -> int:
    return os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = _directory_fd(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path, root: Path) -> None:
    canonical_root = Path(root).resolve(strict=True)
    candidate = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    try:
        relative = candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise FetchCheckpointBindingError(
            "Checkpoint directory escaped its sealed run root."
        ) from exc
    cursor = canonical_root
    for component in relative.parts:
        cursor = cursor / component
        try:
            cursor.mkdir(mode=0o700)
        except FileExistsError:
            observed = os.lstat(cursor)
            if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                raise FetchCheckpointCorrupt(
                    "Checkpoint path contains a symlink or non-directory."
                )
        descriptor = _directory_fd(cursor)
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISDIR(observed.st_mode):
                raise FetchCheckpointCorrupt("Checkpoint directory is unsafe.")
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(cursor.parent)


def _read_regular(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise FetchCheckpointCorrupt(f"Could not open {label} safely.") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FetchCheckpointCorrupt(f"{label} is not a regular file.")
        if before.st_size < 0 or before.st_size > maximum:
            raise FetchCheckpointCorrupt(f"{label} exceeds its bounded size.")
        chunks: List[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise FetchCheckpointCorrupt(f"{label} ended before its recorded size.")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise FetchCheckpointCorrupt(f"{label} grew while it was read.")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise FetchCheckpointCorrupt(f"{label} changed while it was read.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _temp_name(label: str) -> str:
    return f".tmp-{label}-{os.getpid()}-{secrets.token_hex(8)}"


def _write_temp_fsynced(directory: Path, payload: bytes, label: str) -> Path:
    name = _temp_name(label)
    directory_fd = _directory_fd(directory)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset : offset + 1024 * 1024])
            if written <= 0:
                raise OSError("Short checkpoint write.")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    _fsync_directory(directory)
    return directory / name


def _unlink_owned_temp(path: Path) -> None:
    directory_fd = _directory_fd(path.parent)
    try:
        try:
            os.unlink(path.name, dir_fd=directory_fd)
        except FileNotFoundError:
            return
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    _fsync_directory(path.parent)


def _commit_json_noreplace(
    path: Path,
    value: Mapping[str, Any],
    *,
    temporary_directory: Path,
) -> str:
    payload = _canonical_json(value)
    if len(payload) > MAX_JSON_BYTES:
        raise FetchCheckpointError("Checkpoint receipt exceeds its bounded size.")
    if path.exists() or path.is_symlink():
        observed = _read_regular(path, maximum=MAX_JSON_BYTES, label=path.name)
        _parse_canonical_json(observed, path.name)
        if observed != payload:
            raise FetchCheckpointAmbiguous(
                f"Immutable checkpoint record conflicts: {path.name}."
            )
        return _sha256(observed)

    # Every pre-commit byte lives in the dedicated temporary namespace.  A
    # process death therefore cannot leave a marker-shaped file beside durable
    # receipts; only the final no-replace hard link makes the record committed.
    temporary = _write_temp_fsynced(temporary_directory, payload, path.name)
    temporary_fd = _directory_fd(temporary_directory)
    parent_fd = _directory_fd(path.parent)
    try:
        try:
            os.link(
                temporary.name,
                path.name,
                src_dir_fd=temporary_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            observed = _read_regular(path, maximum=MAX_JSON_BYTES, label=path.name)
            _parse_canonical_json(observed, path.name)
            if observed != payload:
                raise FetchCheckpointAmbiguous(
                    f"Concurrent checkpoint record conflicts: {path.name}."
                )
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
        os.close(temporary_fd)
    _unlink_owned_temp(temporary)
    return _sha256(payload)


def _parquet_payload(frame: pd.DataFrame) -> Tuple[bytes, str]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
        schema_sha256 = _sha256(table.schema.serialize().to_pybytes())
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            version="2.6",
            data_page_version="1.0",
            compression="snappy",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=65536,
        )
        payload = sink.getvalue().to_pybytes()
    except Exception as exc:
        raise FetchCheckpointFetchError(
            f"Segment Parquet construction failed ({type(exc).__name__})."
        ) from exc
    if len(payload) > MAX_SEGMENT_PARQUET_BYTES:
        raise FetchCheckpointFetchError("Segment Parquet exceeds its bounded size.")
    return payload, schema_sha256


def _read_parquet(payload: bytes, *, expected_schema_sha256: str) -> pd.DataFrame:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(pa.BufferReader(payload))
        observed_schema = _sha256(table.schema.serialize().to_pybytes())
        if observed_schema != expected_schema_sha256:
            raise FetchCheckpointCorrupt("Segment Parquet schema identity disagrees.")
        return table.to_pandas()
    except FetchCheckpointCorrupt:
        raise
    except Exception as exc:
        raise FetchCheckpointCorrupt(
            f"Segment Parquet could not be reopened ({type(exc).__name__})."
        ) from exc


class SegmentedFetchCheckpoint:
    """Durable segment ledger beneath one sealed live-read run root."""

    def __init__(self, capability: object, binding: FetchCheckpointBinding) -> None:
        # Import lazily to avoid making this operational helper part of the
        # live-provider import graph.
        from .live_antares import LiveAntaresReadCapability

        if type(capability) is not LiveAntaresReadCapability:
            raise FetchCheckpointBindingError(
                "A sealed LiveAntaresReadCapability is required."
            )
        if not isinstance(binding, FetchCheckpointBinding):
            raise FetchCheckpointBindingError("A FetchCheckpointBinding is required.")
        if (
            capability.run_id != binding.run_id
            or capability.release_sha != binding.release_sha
            or capability.target_date_utc != binding.target_date_utc
        ):
            raise FetchCheckpointBindingError(
                "Checkpoint binding disagrees with the sealed live-read capability."
            )
        run_root = Path(capability.run_root)
        observed = os.lstat(run_root)
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise FetchCheckpointBindingError("Sealed run root is missing or unsafe.")
        self.capability = capability
        self.read_only = False
        self.binding = binding
        self.run_root = run_root.resolve(strict=True)
        self.root = _lexical_child(self.run_root, CHECKPOINT_DIRECTORY)
        self.blobs = _lexical_child(
            self.run_root, CHECKPOINT_DIRECTORY / "blobs"
        )
        self.segments = _lexical_child(
            self.run_root, CHECKPOINT_DIRECTORY / "segments"
        )
        self.tmp = _lexical_child(self.run_root, CHECKPOINT_DIRECTORY / "tmp")
        for directory in (self.root, self.blobs, self.segments, self.tmp):
            _ensure_private_directory(directory, self.run_root)
        self.header_path = _lexical_child(
            self.run_root, CHECKPOINT_DIRECTORY / "checkpoint.json"
        )
        self.complete_path = _lexical_child(
            self.run_root, CHECKPOINT_DIRECTORY / "fetch-complete.json"
        )
        self._open_or_create_header()

    @classmethod
    def open(
        cls, capability: object, binding: FetchCheckpointBinding
    ) -> "SegmentedFetchCheckpoint":
        return cls(capability, binding)

    @classmethod
    def open_read_only(
        cls, run_root: Path, binding: FetchCheckpointBinding
    ) -> "SegmentedFetchCheckpoint":
        """Open existing evidence without granting live-read or write authority.

        Cross-release authorization belongs to the caller's recovery contract;
        this reader still requires the checkpoint's exact originating binding.
        It never creates, chmods, fsyncs, repairs, or upgrades source entries.
        """
        if not isinstance(binding, FetchCheckpointBinding):
            raise FetchCheckpointBindingError("A FetchCheckpointBinding is required.")
        root = Path(os.path.abspath(os.fspath(run_root)))
        if root.name != binding.run_id or root.resolve(strict=True) != root:
            raise FetchCheckpointBindingError("Read-only run root identity is unsafe.")
        instance = cls.__new__(cls)
        instance.capability = None
        instance.read_only = True
        instance.binding = binding
        instance.run_root = root
        instance.root = root / CHECKPOINT_DIRECTORY
        instance.blobs = instance.root / "blobs"
        instance.segments = instance.root / "segments"
        instance.tmp = instance.root / "tmp"
        instance.header_path = instance.root / "checkpoint.json"
        instance.complete_path = instance.root / "fetch-complete.json"
        instance._check_read_only_directories()
        instance._open_or_create_header(allow_create=False)
        return instance

    def _check_read_only_directories(self) -> None:
        for path in (
            self.run_root,
            self.root.parent,
            self.root,
            self.blobs,
            self.segments,
            self.tmp,
        ):
            observed = os.lstat(path)
            if (
                not stat.S_ISDIR(observed.st_mode)
                or path.resolve(strict=True) != path
                or stat.S_IMODE(observed.st_mode) != 0o700
            ):
                raise FetchCheckpointCorrupt("Read-only checkpoint directory is unsafe.")
        if any(self.tmp.iterdir()):
            raise FetchCheckpointAmbiguous("Read-only checkpoint has temporary residue.")

    def _header(self) -> Dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_identity_sha256": self.binding.identity_sha256,
            "binding": self.binding.as_dict(),
            "layout": {
                "blob_directory": "blobs",
                "segment_directory": "segments",
                "temporary_directory": "tmp",
                "completion_record": "fetch-complete.json",
                "segment_commit_marker": "*.commit.json",
            },
            "role": "operational-fetch-checkpoint",
            "authoritative": False,
            "production_eligible": False,
            "publication_capability": False,
            "cache": False,
            "secret_material_recorded": False,
        }

    def _open_or_create_header(self, *, allow_create: bool = True) -> None:
        if self.read_only:
            allow_create = False
        expected = self._header()
        if self.header_path.exists() or self.header_path.is_symlink():
            payload = _read_regular(
                self.header_path, maximum=MAX_JSON_BYTES, label="checkpoint header"
            )
            observed = _parse_canonical_json(payload, "checkpoint header")
            if observed != expected:
                raise FetchCheckpointBindingError(
                    "Existing checkpoint belongs to another sealed identity."
                )
            return
        if not allow_create:
            raise FetchCheckpointCorrupt(
                "Immutable checkpoint header disappeared after open."
            )
        if (
            any(self.blobs.iterdir())
            or any(self.segments.iterdir())
            or self.complete_path.exists()
            or self.complete_path.is_symlink()
        ):
            raise FetchCheckpointAmbiguous(
                "Checkpoint content exists without its immutable header."
            )
        expected_root_entries = {"blobs", "segments", "tmp"}
        if any(entry.name not in expected_root_entries for entry in self.root.iterdir()):
            raise FetchCheckpointAmbiguous(
                "Checkpoint root contains content without its immutable header."
            )
        _commit_json_noreplace(
            self.header_path,
            expected,
            temporary_directory=self.tmp,
        )

    def _plans(self, ordered_locus_ids: Sequence[str]) -> Tuple[FetchSegmentPlan, ...]:
        ids = tuple(str(value) for value in ordered_locus_ids)
        if len(ids) != self.binding.expected_objects:
            raise FetchCheckpointBindingError(
                "Ordered query result count disagrees with checkpoint binding."
            )
        if len(set(ids)) != len(ids):
            raise FetchCheckpointBindingError(
                "Ordered query result contains duplicate locus identities."
            )
        if _identifier_sha256(ids) != self.binding.query_locus_order_sha256:
            raise FetchCheckpointBindingError(
                "Ordered query result digest disagrees with checkpoint binding."
            )
        plans = []
        for index, start in enumerate(range(0, len(ids), self.binding.segment_size)):
            stop = min(start + self.binding.segment_size, len(ids))
            plans.append(FetchSegmentPlan(index, start, stop, ids[start:stop]))
        return tuple(plans)

    def _validate_structure(self, plans: Sequence[FetchSegmentPlan]) -> None:
        if self.read_only:
            self._check_read_only_directories()
        else:
            for directory in (self.root, self.blobs, self.segments, self.tmp):
                _ensure_private_directory(directory, self.run_root)
        # Reprove the immutable binding on every public operation.  This also
        # fails closed if the header was removed after the object was opened.
        self._open_or_create_header(allow_create=False)
        allowed_root = {
            "blobs",
            "segments",
            "tmp",
            "checkpoint.json",
            "fetch-complete.json",
        }
        for entry in self.root.iterdir():
            observed = os.lstat(entry)
            if stat.S_ISLNK(observed.st_mode) or entry.name not in allowed_root:
                raise FetchCheckpointAmbiguous(
                    "Checkpoint root contains an unexpected or linked entry."
                )
        expected_markers = {plan.commit_name for plan in plans}
        for entry in self.segments.iterdir():
            observed = os.lstat(entry)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or _SEGMENT_NAME.fullmatch(entry.name) is None
                or entry.name not in expected_markers
            ):
                raise FetchCheckpointAmbiguous(
                    "Checkpoint contains an unexpected segment commit record."
                )
        for entry in self.blobs.iterdir():
            observed = os.lstat(entry)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or _BLOB_NAME.fullmatch(entry.name) is None
            ):
                raise FetchCheckpointAmbiguous(
                    "Checkpoint blob directory contains an unsafe entry."
                )
            payload = _read_regular(
                entry,
                maximum=MAX_SEGMENT_PARQUET_BYTES,
                label="checkpoint blob",
            )
            if _sha256(payload) + ".parquet" != entry.name:
                raise FetchCheckpointCorrupt(
                    "Content-addressed checkpoint blob hash disagrees."
                )
        for entry in self.tmp.iterdir():
            observed = os.lstat(entry)
            if (
                stat.S_ISLNK(observed.st_mode)
                or not stat.S_ISREG(observed.st_mode)
                or not entry.name.startswith(".tmp-")
            ):
                raise FetchCheckpointAmbiguous(
                    "Checkpoint temporary directory contains an unsafe entry."
                )

    def _store_blob(self, payload: bytes, digest: str) -> Path:
        if self.read_only:
            raise FetchCheckpointBindingError("Read-only checkpoints cannot store blobs.")
        path = _lexical_child(self.blobs, Path(digest + ".parquet"))
        if path.exists() or path.is_symlink():
            observed = _read_regular(
                path,
                maximum=MAX_SEGMENT_PARQUET_BYTES,
                label="checkpoint blob",
            )
            if len(observed) != len(payload) or _sha256(observed) != digest:
                raise FetchCheckpointAmbiguous(
                    "Existing content-addressed checkpoint blob conflicts."
                )
            return path
        temporary = _write_temp_fsynced(self.tmp, payload, "segment-parquet")
        tmp_fd = _directory_fd(self.tmp)
        blob_fd = _directory_fd(self.blobs)
        try:
            try:
                os.link(
                    temporary.name,
                    path.name,
                    src_dir_fd=tmp_fd,
                    dst_dir_fd=blob_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                observed = _read_regular(
                    path,
                    maximum=MAX_SEGMENT_PARQUET_BYTES,
                    label="checkpoint blob",
                )
                if len(observed) != len(payload) or _sha256(observed) != digest:
                    raise FetchCheckpointAmbiguous(
                        "Concurrent content-addressed blob conflicts."
                    )
            os.fsync(blob_fd)
        finally:
            os.close(blob_fd)
            os.close(tmp_fd)
        _unlink_owned_temp(temporary)
        return path

    def _normalize_results(
        self,
        plan: FetchSegmentPlan,
        returned: Iterable[FetchObjectResult],
    ) -> Tuple[pd.DataFrame, Tuple[Mapping[str, Any], ...]]:
        try:
            values = tuple(returned)
        except Exception as exc:
            raise FetchCheckpointFetchError(
                f"Segment callback iteration failed ({type(exc).__name__})."
            ) from exc
        if any(type(value) is not FetchObjectResult for value in values):
            raise FetchCheckpointFetchError(
                "Segment callback returned an invalid result type."
            )
        by_id: Dict[str, FetchObjectResult] = {}
        for value in values:
            if value.locus_id in by_id:
                raise FetchCheckpointFetchError(
                    "Segment callback returned a duplicate locus identity."
                )
            by_id[value.locus_id] = value
        if set(by_id) != set(plan.locus_ids) or len(values) != len(plan.locus_ids):
            raise FetchCheckpointFetchError(
                "Segment callback did not return the exact requested identities."
            )

        frames: List[pd.DataFrame] = []
        receipts: List[Mapping[str, Any]] = []
        memory_bytes = 0
        alert_rows = 0
        for relative, locus_id in enumerate(plan.locus_ids):
            value = by_id[locus_id]
            if value.alerts is None or value.alerts.empty:
                frame = None
                rows = 0
            else:
                frame = value.alerts.copy(deep=True)
                if "locus_id" in frame.columns:
                    observed_ids = frame["locus_id"]
                    if observed_ids.isna().any() or not observed_ids.astype(str).eq(
                        locus_id
                    ).all():
                        raise FetchCheckpointFetchError(
                            "Fetched alert rows disagree with their locus identity."
                        )
                else:
                    frame["locus_id"] = locus_id
                rows = len(frame)
                alert_rows += rows
                memory_bytes += int(frame.memory_usage(index=True, deep=True).sum())
                if alert_rows > MAX_SEGMENT_ALERT_ROWS:
                    raise FetchCheckpointFetchError(
                        "Segment alert rows exceed the bounded limit."
                    )
                if memory_bytes > MAX_SEGMENT_MEMORY_BYTES:
                    raise FetchCheckpointFetchError(
                        "Segment frame memory exceeds the bounded limit."
                    )
                frames.append(frame)
            receipts.append(
                {
                    "ordinal": plan.start + relative,
                    "locus_id": locus_id,
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
        return alerts, tuple(receipts)

    def _segment_receipt(
        self,
        plan: FetchSegmentPlan,
        objects: Sequence[Mapping[str, Any]],
        *,
        blob_path: Path,
        blob_payload: bytes,
        parquet_schema_sha256: str,
    ) -> Dict[str, Any]:
        retry_types = sorted(
            {
                item
                for value in objects
                for item in value["retry_exception_types"]
            }
        )
        return {
            "schema_version": SEGMENT_SCHEMA_VERSION,
            "checkpoint_identity_sha256": self.binding.identity_sha256,
            "segment": {
                "index": plan.index,
                "start": plan.start,
                "stop": plan.stop,
                "requested_objects": len(plan.locus_ids),
                "ordered_locus_ids": list(plan.locus_ids),
                "ordered_locus_id_sha256": _identifier_sha256(plan.locus_ids),
            },
            "objects": list(objects),
            "completed_objects": len(plan.locus_ids),
            "failed_objects": 0,
            "partial": False,
            "alert_rows": sum(int(value["alert_rows"]) for value in objects),
            "retry_count": sum(int(value["retry_count"]) for value in objects),
            "retry_exception_types": retry_types,
            "artifact": {
                "path": f"blobs/{blob_path.name}",
                "bytes": len(blob_payload),
                "sha256": _sha256(blob_payload),
                "parquet_schema_sha256": parquet_schema_sha256,
            },
            "authoritative": False,
            "production_eligible": False,
            "secret_material_recorded": False,
        }

    def _validate_object_receipts(
        self,
        plan: FetchSegmentPlan,
        objects: object,
    ) -> Tuple[Mapping[str, Any], ...]:
        if not isinstance(objects, list) or len(objects) != len(plan.locus_ids):
            raise FetchCheckpointCorrupt("Segment object receipts are incomplete.")
        expected_keys = {
            "ordinal",
            "locus_id",
            "alert_rows",
            "retry_count",
            "retry_exception_types",
        }
        normalized = []
        for relative, (expected_id, value) in enumerate(zip(plan.locus_ids, objects)):
            if not isinstance(value, dict) or set(value) != expected_keys:
                raise FetchCheckpointCorrupt("Segment object receipt fields are invalid.")
            retry_types = value.get("retry_exception_types")
            if (
                value.get("ordinal") != plan.start + relative
                or value.get("locus_id") != expected_id
                or type(value.get("alert_rows")) is not int
                or value["alert_rows"] < 0
                or type(value.get("retry_count")) is not int
                or value["retry_count"] < 0
                or not isinstance(retry_types, list)
                or retry_types != sorted(set(retry_types))
                or any(not isinstance(item, str) or not item for item in retry_types)
                or (value["retry_count"] == 0 and retry_types)
            ):
                raise FetchCheckpointCorrupt("Segment object receipt values are invalid.")
            normalized.append(value)
        return tuple(normalized)

    def _load_segment(
        self, plan: FetchSegmentPlan, *, reopen: bool
    ) -> Optional[Tuple[FetchSegmentRecord, Optional[ReopenedFetchSegment]]]:
        commit_path = _lexical_child(self.segments, Path(plan.commit_name))
        if not commit_path.exists() and not commit_path.is_symlink():
            return None
        payload = _read_regular(
            commit_path, maximum=MAX_JSON_BYTES, label="segment commit record"
        )
        receipt = _parse_canonical_json(payload, "segment commit record")
        expected_keys = {
            "schema_version",
            "checkpoint_identity_sha256",
            "segment",
            "objects",
            "completed_objects",
            "failed_objects",
            "partial",
            "alert_rows",
            "retry_count",
            "retry_exception_types",
            "artifact",
            "authoritative",
            "production_eligible",
            "secret_material_recorded",
        }
        segment = receipt.get("segment")
        expected_segment_keys = {
            "index",
            "start",
            "stop",
            "requested_objects",
            "ordered_locus_ids",
            "ordered_locus_id_sha256",
        }
        if (
            set(receipt) != expected_keys
            or receipt.get("schema_version") != SEGMENT_SCHEMA_VERSION
            or receipt.get("checkpoint_identity_sha256")
            != self.binding.identity_sha256
            or not isinstance(segment, dict)
            or set(segment) != expected_segment_keys
            or segment.get("index") != plan.index
            or segment.get("start") != plan.start
            or segment.get("stop") != plan.stop
            or segment.get("requested_objects") != len(plan.locus_ids)
            or segment.get("ordered_locus_ids") != list(plan.locus_ids)
            or segment.get("ordered_locus_id_sha256")
            != _identifier_sha256(plan.locus_ids)
            or receipt.get("completed_objects") != len(plan.locus_ids)
            or receipt.get("failed_objects") != 0
            or receipt.get("partial") is not False
            or receipt.get("authoritative") is not False
            or receipt.get("production_eligible") is not False
            or receipt.get("secret_material_recorded") is not False
        ):
            raise FetchCheckpointCorrupt("Segment receipt identity is invalid.")
        objects = self._validate_object_receipts(plan, receipt.get("objects"))
        artifact = receipt.get("artifact")
        artifact_keys = {"path", "bytes", "sha256", "parquet_schema_sha256"}
        if not isinstance(artifact, dict) or set(artifact) != artifact_keys:
            raise FetchCheckpointCorrupt("Segment artifact receipt is invalid.")
        digest = artifact.get("sha256")
        schema_digest = artifact.get("parquet_schema_sha256")
        if (
            not isinstance(digest, str)
            or _HEX_64.fullmatch(digest) is None
            or not isinstance(schema_digest, str)
            or _HEX_64.fullmatch(schema_digest) is None
            or artifact.get("path") != f"blobs/{digest}.parquet"
            or type(artifact.get("bytes")) is not int
            or artifact["bytes"] < 0
            or artifact["bytes"] > MAX_SEGMENT_PARQUET_BYTES
        ):
            raise FetchCheckpointCorrupt("Segment artifact identity is invalid.")
        blob_path = _lexical_child(self.root, Path(str(artifact["path"])))
        blob = _read_regular(
            blob_path,
            maximum=MAX_SEGMENT_PARQUET_BYTES,
            label="segment Parquet blob",
        )
        if len(blob) != artifact["bytes"] or _sha256(blob) != digest:
            raise FetchCheckpointCorrupt("Segment Parquet size or hash disagrees.")
        frame = _read_parquet(blob, expected_schema_sha256=schema_digest)
        expected_rows = sum(int(value["alert_rows"]) for value in objects)
        retry_count = sum(int(value["retry_count"]) for value in objects)
        retry_types = sorted(
            {
                item
                for value in objects
                for item in value["retry_exception_types"]
            }
        )
        if (
            receipt.get("alert_rows") != expected_rows
            or receipt.get("retry_count") != retry_count
            or receipt.get("retry_exception_types") != retry_types
            or len(frame) != expected_rows
            or "locus_id" not in frame.columns
        ):
            raise FetchCheckpointCorrupt("Segment counts or retry evidence disagree.")
        expected_row_ids = [
            str(value["locus_id"])
            for value in objects
            for _ in range(int(value["alert_rows"]))
        ]
        if frame["locus_id"].isna().any() or (
            frame["locus_id"].astype(str).tolist() != expected_row_ids
        ):
            raise FetchCheckpointCorrupt("Segment alert ordering or identity disagrees.")
        record = FetchSegmentRecord(plan, receipt, _sha256(payload), blob_path)
        reopened = (
            ReopenedFetchSegment(plan, frame, objects, _sha256(payload))
            if reopen
            else None
        )
        return record, reopened

    def _load_existing(
        self, plans: Sequence[FetchSegmentPlan]
    ) -> Dict[int, FetchSegmentRecord]:
        self._validate_structure(plans)
        records: Dict[int, FetchSegmentRecord] = {}
        for plan in plans:
            loaded = self._load_segment(plan, reopen=False)
            if loaded is not None:
                records[plan.index] = loaded[0]
        return records

    def _completion_document(
        self, records: Mapping[int, FetchSegmentRecord]
    ) -> Dict[str, Any]:
        ordered = [records[index] for index in sorted(records)]
        retry_types = sorted(
            {
                item
                for record in ordered
                for item in record.receipt["retry_exception_types"]
            }
        )
        segment_identities = [
            {
                "index": record.plan.index,
                "start": record.plan.start,
                "stop": record.plan.stop,
                "receipt_sha256": record.receipt_sha256,
                "blob_sha256": record.receipt["artifact"]["sha256"],
            }
            for record in ordered
        ]
        segments_sha256 = _sha256(
            _canonical_json({"segments": segment_identities})
        )
        return {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "checkpoint_identity_sha256": self.binding.identity_sha256,
            "requested_objects": self.binding.expected_objects,
            "completed_objects": sum(
                int(record.receipt["completed_objects"]) for record in ordered
            ),
            "failed_objects": 0,
            "segment_count": len(ordered),
            "segments": segment_identities,
            "segments_sha256": segments_sha256,
            "alert_rows": sum(int(record.receipt["alert_rows"]) for record in ordered),
            "retry_count": sum(int(record.receipt["retry_count"]) for record in ordered),
            "retry_exception_types": retry_types,
            "coverage": "exact-contiguous-query-order",
            "completed": True,
            "partial": False,
            "authoritative": False,
            "production_eligible": False,
            "publication_capability": False,
            "secret_material_recorded": False,
        }

    def _validate_or_commit_completion(
        self,
        plans: Sequence[FetchSegmentPlan],
        records: Mapping[int, FetchSegmentRecord],
        *,
        event_hook: Optional[CheckpointEventHook],
    ) -> Tuple[Mapping[str, Any], str]:
        if set(records) != {plan.index for plan in plans}:
            if self.complete_path.exists() or self.complete_path.is_symlink():
                raise FetchCheckpointCorrupt(
                    "Fetch-complete marker exists without exact segment coverage."
                )
            raise FetchCheckpointIncomplete("Fetch checkpoint is not complete.")
        expected_start = 0
        for plan in plans:
            if plan.start != expected_start or plan.stop <= plan.start:
                raise FetchCheckpointCorrupt("Segment coverage is not exactly contiguous.")
            expected_start = plan.stop
        if expected_start != self.binding.expected_objects:
            raise FetchCheckpointCorrupt("Segment coverage does not reach query completion.")
        document = self._completion_document(records)
        if (
            document["completed_objects"] != self.binding.expected_objects
            or document["failed_objects"] != 0
            or document["segment_count"] != len(plans)
        ):
            raise FetchCheckpointCorrupt("Fetch completion totals are contradictory.")
        if self.read_only:
            payload = _read_regular(
                self.complete_path, maximum=MAX_JSON_BYTES, label="fetch completion"
            )
            if _parse_canonical_json(payload, "fetch completion") != document:
                raise FetchCheckpointCorrupt("Read-only completion identity differs.")
            referenced = {
                record.receipt["artifact"]["sha256"] + ".parquet"
                for record in records.values()
            }
            if {path.name for path in self.blobs.iterdir()} != referenced:
                raise FetchCheckpointAmbiguous("Read-only checkpoint has extra blobs.")
            return document, _sha256(payload)
        if event_hook is not None:
            event_hook(
                "before_fetch_complete_commit",
                {
                    "segment_count": len(plans),
                    "completed_objects": self.binding.expected_objects,
                },
            )
        completion_sha = _commit_json_noreplace(
            self.complete_path,
            document,
            temporary_directory=self.tmp,
        )
        if event_hook is not None:
            event_hook(
                "after_fetch_complete_commit",
                {"completion_sha256": completion_sha},
            )
        return document, completion_sha

    def fetch_missing(
        self,
        ordered_locus_ids: Sequence[str],
        fetch_segment: SegmentFetcher,
        *,
        event_hook: Optional[CheckpointEventHook] = None,
    ) -> FetchCheckpointCompletion:
        """Reuse committed segments and fetch only deterministic missing segments."""
        if self.read_only:
            raise FetchCheckpointBindingError("Read-only checkpoints cannot invoke callbacks.")
        if not callable(fetch_segment):
            raise TypeError("fetch_segment must be callable.")
        plans = self._plans(ordered_locus_ids)
        records = self._load_existing(plans)
        initially_reused = len(records)

        if self.complete_path.exists() or self.complete_path.is_symlink():
            document, completion_sha = self._validate_or_commit_completion(
                plans, records, event_hook=None
            )
            return self._completion_result(
                document,
                completion_sha,
                reused_segments=initially_reused,
                fetched_segments=0,
            )

        fetched_segments = 0
        for plan in plans:
            if plan.index in records:
                continue
            try:
                returned = fetch_segment(plan.locus_ids)
                alerts, objects = self._normalize_results(plan, returned)
            except FetchCheckpointError:
                raise
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                raise FetchCheckpointFetchError(
                    f"Segment callback failed ({type(exc).__name__}); no segment committed."
                ) from exc
            payload, schema_sha = _parquet_payload(alerts)
            digest = _sha256(payload)
            blob_path = self._store_blob(payload, digest)
            if event_hook is not None:
                event_hook(
                    "after_segment_blob_committed",
                    {"segment_index": plan.index, "blob_sha256": digest},
                )
            receipt = self._segment_receipt(
                plan,
                objects,
                blob_path=blob_path,
                blob_payload=payload,
                parquet_schema_sha256=schema_sha,
            )
            if event_hook is not None:
                event_hook(
                    "before_segment_commit",
                    {"segment_index": plan.index, "start": plan.start, "stop": plan.stop},
                )
            commit_path = _lexical_child(self.segments, Path(plan.commit_name))
            _commit_json_noreplace(
                commit_path,
                receipt,
                temporary_directory=self.tmp,
            )
            if event_hook is not None:
                event_hook(
                    "after_segment_commit",
                    {"segment_index": plan.index, "start": plan.start, "stop": plan.stop},
                )
            loaded = self._load_segment(plan, reopen=False)
            if loaded is None:
                raise FetchCheckpointCorrupt("Committed segment disappeared during reproof.")
            records[plan.index] = loaded[0]
            fetched_segments += 1

        document, completion_sha = self._validate_or_commit_completion(
            plans, records, event_hook=event_hook
        )
        return self._completion_result(
            document,
            completion_sha,
            reused_segments=initially_reused,
            fetched_segments=fetched_segments,
        )

    def _completion_result(
        self,
        document: Mapping[str, Any],
        completion_sha: str,
        *,
        reused_segments: int,
        fetched_segments: int,
    ) -> FetchCheckpointCompletion:
        return FetchCheckpointCompletion(
            checkpoint_root=self.root,
            checkpoint_identity_sha256=self.binding.identity_sha256,
            requested_objects=int(document["requested_objects"]),
            completed_objects=int(document["completed_objects"]),
            segment_count=int(document["segment_count"]),
            alert_rows=int(document["alert_rows"]),
            retry_count=int(document["retry_count"]),
            retry_exception_types=tuple(document["retry_exception_types"]),
            completion_sha256=completion_sha,
            reused_segments=reused_segments,
            fetched_segments=fetched_segments,
        )

    def inspect_complete(
        self, ordered_locus_ids: Sequence[str]
    ) -> FetchCheckpointCompletion:
        """Validate a complete checkpoint without invoking a fetch callback."""
        plans = self._plans(ordered_locus_ids)
        records = self._load_existing(plans)
        if not (self.complete_path.exists() or self.complete_path.is_symlink()):
            raise FetchCheckpointIncomplete("Fetch-complete marker is absent.")
        document, completion_sha = self._validate_or_commit_completion(
            plans, records, event_hook=None
        )
        return self._completion_result(
            document,
            completion_sha,
            reused_segments=len(records),
            fetched_segments=0,
        )

    def iter_segments(
        self, ordered_locus_ids: Sequence[str]
    ) -> Iterator[ReopenedFetchSegment]:
        """Yield verified completed data one bounded segment at a time."""
        self.inspect_complete(ordered_locus_ids)
        plans = self._plans(ordered_locus_ids)
        for plan in plans:
            loaded = self._load_segment(plan, reopen=True)
            if loaded is None or loaded[1] is None:
                raise FetchCheckpointCorrupt("Completed segment is missing during reopen.")
            yield loaded[1]

    def iter_objects(
        self, ordered_locus_ids: Sequence[str]
    ) -> Iterator[FetchObjectResult]:
        """Reconstruct verified per-locus results in sealed query order."""
        for segment in self.iter_segments(ordered_locus_ids):
            cursor = 0
            for value in segment.objects:
                rows = int(value["alert_rows"])
                frame = (
                    segment.alerts.iloc[cursor : cursor + rows].copy(deep=True)
                    if rows
                    else None
                )
                cursor += rows
                yield FetchObjectResult(
                    str(value["locus_id"]),
                    frame,
                    retry_count=int(value["retry_count"]),
                    retry_exception_types=tuple(value["retry_exception_types"]),
                )
            if cursor != len(segment.alerts):
                raise FetchCheckpointCorrupt(
                    "Segment reconstruction did not consume every alert row."
                )

    def reconstruct_alerts(self, ordered_locus_ids: Sequence[str]) -> pd.DataFrame:
        """Convenience full-frame reconstruction; streaming callers use iter_segments."""
        frames = [segment.alerts for segment in self.iter_segments(ordered_locus_ids)]
        return (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame({"locus_id": pd.Series(dtype="object")})
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "COMPLETION_SCHEMA_VERSION",
    "CheckpointEventHook",
    "FetchCheckpointAmbiguous",
    "FetchCheckpointBinding",
    "FetchCheckpointBindingError",
    "FetchCheckpointCompletion",
    "FetchCheckpointCorrupt",
    "FetchCheckpointError",
    "FetchCheckpointFetchError",
    "FetchCheckpointIncomplete",
    "FetchObjectResult",
    "FetchSegmentPlan",
    "FetchSegmentRecord",
    "ReopenedFetchSegment",
    "SegmentFetcher",
    "SegmentedFetchCheckpoint",
]
