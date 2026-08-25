"""Conservative single-writer lock contract for local temporary fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .storage import DevelopmentWriteCapability, OPERATIONS_DIRECTORY, contained_path


class LockUnavailable(RuntimeError):
    pass


class LockOwnershipError(RuntimeError):
    pass


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Lock timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def lock_identity(target_identity: str) -> str:
    digest = hashlib.sha256(target_identity.encode("utf-8")).hexdigest()[:24]
    return f"writer-{digest}.lock"


def _directory_identity(path: Path) -> Tuple[int, int]:
    if path.is_symlink() or not path.is_dir():
        raise LockOwnershipError("Writer lock directory is missing or unsafe.")
    observed = path.stat()
    return observed.st_dev, observed.st_ino


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_private_directory(path: Path, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=False)
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise LockOwnershipError("Writer lock parent escaped its capability.") from exc
    cursor = resolved_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            cursor.mkdir(mode=0o700)
        except FileExistsError:
            if cursor.is_symlink() or not cursor.is_dir():
                raise LockOwnershipError(
                    "Writer lock parent is missing, unsafe, or not a directory."
                )
        descriptor = os.open(
            str(cursor),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISDIR(observed.st_mode):
                raise LockOwnershipError("Writer lock parent is not a directory.")
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(cursor.parent)


def _write_metadata_fsynced(path: Path, payload: str) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        encoded = payload.encode("utf-8")
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            written = stream.write(encoded)
            if written != len(encoded):
                raise OSError("Short writer-lock metadata write.")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class LockInspection:
    exists: bool
    ambiguous: bool
    stale_candidate: bool
    metadata: Optional[Mapping[str, Any]]


class WriterLock:
    """Atomic-directory lock that never auto-steals stale/ambiguous locks."""

    def __init__(
        self,
        capability: DevelopmentWriteCapability,
        target_identity: str,
        run_id: str,
        *,
        owner: Optional[str] = None,
        hostname: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> None:
        self.capability = capability
        self.target_identity = target_identity
        self.run_id = run_id
        self.owner = owner or str(os.getuid())
        self.hostname = hostname or socket.gethostname()
        self.pid = os.getpid() if pid is None else int(pid)
        self.token = uuid.uuid4().hex
        self.path = contained_path(
            capability.root,
            Path(OPERATIONS_DIRECTORY) / "locks" / lock_identity(target_identity),
        )
        self.metadata_path = self.path / "owner.json"
        self._held = False
        self._directory_identity: Optional[Tuple[int, int]] = None

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self, *, at: Optional[datetime] = None) -> Mapping[str, Any]:
        timestamp = at or datetime.now(timezone.utc)
        _make_private_directory(self.path.parent, self.capability.root)
        try:
            self.path.mkdir(mode=0o700)
        except FileExistsError as exc:
            inspection = self.inspect()
            raise LockUnavailable(
                "Writer lock already exists; it will not be stolen automatically. "
                f"ambiguous={inspection.ambiguous}"
            ) from exc
        self._directory_identity = _directory_identity(self.path)
        descriptor = os.open(
            str(self.path),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)
        metadata = {
            "schema_version": "1.0",
            "target_identity": self.target_identity,
            "run_id": self.run_id,
            "owner": self.owner,
            "hostname": self.hostname,
            "pid": self.pid,
            "acquired_at_utc": _iso(timestamp),
            "ownership_token": self.token,
        }
        try:
            _write_metadata_fsynced(
                self.metadata_path,
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
            _fsync_directory(self.path)
            _fsync_directory(self.path.parent)
        except Exception:
            # A failed metadata write may have left ambiguous state. Never
            # recursively clean or steal it automatically.
            raise
        self._held = True
        return metadata

    def inspect(
        self,
        *,
        now: Optional[datetime] = None,
        stale_after_seconds: Optional[float] = None,
    ) -> LockInspection:
        if not self.path.exists() and not self.path.is_symlink():
            return LockInspection(False, False, False, None)
        if self.path.is_symlink() or not self.path.is_dir() or self.metadata_path.is_symlink():
            return LockInspection(True, True, False, None)
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return LockInspection(True, True, False, None)
        if not isinstance(metadata, dict):
            return LockInspection(True, True, False, None)
        stale_candidate = False
        if stale_after_seconds is not None:
            try:
                acquired = datetime.fromisoformat(str(metadata["acquired_at_utc"]))
                current = now or datetime.now(timezone.utc)
                stale_candidate = (current - acquired).total_seconds() >= stale_after_seconds
            except (KeyError, TypeError, ValueError):
                return LockInspection(True, True, False, metadata)
        return LockInspection(True, False, stale_candidate, metadata)

    def assert_owned(self) -> Mapping[str, Any]:
        """Re-prove the on-disk lock, not merely this object's memory flag."""
        if not self._held or self._directory_identity is None:
            raise LockOwnershipError("This lock object does not own the writer lock.")
        if _directory_identity(self.path) != self._directory_identity:
            raise LockOwnershipError("Writer lock directory identity changed.")
        inspection = self.inspect()
        metadata = inspection.metadata
        if inspection.ambiguous or metadata is None:
            raise LockOwnershipError("Writer lock ownership metadata is ambiguous.")
        expected = {
            "target_identity": self.target_identity,
            "run_id": self.run_id,
            "owner": self.owner,
            "hostname": self.hostname,
            "pid": self.pid,
            "ownership_token": self.token,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise LockOwnershipError("Writer lock is owned by another operation.")
        if set(self.path.iterdir()) != {self.metadata_path}:
            raise LockOwnershipError("Writer lock directory contents are ambiguous.")
        return metadata

    def release(self) -> None:
        self.assert_owned()
        self.metadata_path.unlink()
        _fsync_directory(self.path)
        self.path.rmdir()
        _fsync_directory(self.path.parent)
        self._held = False
        self._directory_identity = None
