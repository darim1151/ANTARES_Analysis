"""Conservative single-writer lock contract for sealed write capabilities."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import socket
import stat
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from .storage import (
    DevelopmentWriteCapability,
    SyntheticWriteCapability,
    contained_path,
)


class LockUnavailable(RuntimeError):
    pass


class LockOwnershipError(RuntimeError):
    pass


WriteCapability = Union[DevelopmentWriteCapability, SyntheticWriteCapability]
LOCK_SCHEMA_VERSION = "2.0"


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("Lock timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def lock_identity(target_identity: str) -> str:
    digest = hashlib.sha256(target_identity.encode("utf-8")).hexdigest()[:24]
    return f"writer-{digest}.lock"


def _canonical_target(capability: WriteCapability, target_identity: str) -> Path:
    """Resolve one target beneath the capability's publication boundary."""
    target = Path(target_identity)
    published_root = capability.published_root.resolve(strict=False)
    if not target.is_absolute():
        return contained_path(published_root, target)
    resolved = target.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(published_root)
    except ValueError as exc:
        raise LockOwnershipError(
            "Writer lock target escaped its publication root."
        ) from exc
    if not relative.parts:
        raise LockOwnershipError(
            "Writer lock target must be beneath its publication root."
        )
    return resolved


def _effective_user(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _linux_process_start_identity(pid: int) -> Optional[str]:
    """Return boot-scoped Linux process identity using procfs start ticks."""
    try:
        stat_text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        closing = stat_text.rfind(")")
        if closing < 0:
            return None
        fields_after_name = stat_text[closing + 2 :].split()
        # Field 22 is process start time; the first item above is field 3.
        start_ticks = fields_after_name[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        if not start_ticks.isdigit() or not boot_id:
            return None
        return f"linux:{boot_id}:{start_ticks}"
    except (IndexError, OSError, UnicodeError):
        return None


def _process_start_identity(pid: int, hostname: str) -> str:
    """Return a PID-reuse-resistant process identity when the OS exposes one."""
    linux_identity = _linux_process_start_identity(pid)
    if linux_identity is not None:
        return linux_identity
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        started = " ".join(completed.stdout.split())
        if completed.returncode == 0 and started:
            return f"ps:{hostname}:{pid}:{started}"
    except (OSError, subprocess.SubprocessError):
        pass
    # Explicit test PIDs and already-exited processes still receive a stable,
    # conservative identity. Such metadata can never prove a live process.
    return f"unavailable:{hostname}:{pid}"


def _optional_text(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string or null.")
    return value


def _artifact_hash_mapping(value: Optional[Mapping[str, str]]) -> Dict[str, str]:
    if value is None:
        return {}
    normalized: Dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Artifact-hash names must be non-empty strings.")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Artifact hashes must be non-empty strings.")
        normalized[name] = digest
    return {name: normalized[name] for name in sorted(normalized)}


def _valid_lock_metadata(metadata: Mapping[str, Any]) -> bool:
    required = {
        "schema_version",
        "target_identity",
        "canonical_target",
        "canonical_root",
        "transaction_id",
        "run_id",
        "release_sha",
        "plan_hash",
        "config_hash",
        "artifact_hashes",
        "owner",
        "effective_user",
        "effective_uid",
        "hostname",
        "pid",
        "process_start_identity",
        "acquired_at_utc",
        "ownership_token",
    }
    if (
        set(metadata) != required
        or metadata.get("schema_version") != LOCK_SCHEMA_VERSION
    ):
        return False
    required_text = required - {
        "release_sha",
        "plan_hash",
        "config_hash",
        "artifact_hashes",
        "effective_uid",
        "pid",
    }
    if any(
        not isinstance(metadata.get(field), str) or not metadata.get(field)
        for field in required_text
    ):
        return False
    if any(
        metadata.get(field) is not None
        and (not isinstance(metadata.get(field), str) or not metadata.get(field))
        for field in ("release_sha", "plan_hash", "config_hash")
    ):
        return False
    hashes = metadata.get("artifact_hashes")
    if not isinstance(hashes, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(digest, str)
        or not digest
        for name, digest in hashes.items()
    ):
        return False
    if (
        type(metadata.get("effective_uid")) is not int
        or type(metadata.get("pid")) is not int
    ):
        return False
    try:
        acquired = datetime.fromisoformat(str(metadata["acquired_at_utc"]))
    except (TypeError, ValueError):
        return False
    return acquired.tzinfo is not None


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


def _replace_metadata_fsynced(path: Path, payload: str) -> None:
    """Atomically replace owned metadata, preserving a fail-closed outcome."""
    parent_fd = os.open(
        str(path.parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    temporary = f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        try:
            os.fchmod(descriptor, 0o600)
            encoded = payload.encode("utf-8")
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("Short writer-lock metadata update.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        created = False
        os.fsync(parent_fd)
    finally:
        if created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


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
        capability: WriteCapability,
        target_identity: str,
        run_id: str,
        *,
        owner: Optional[str] = None,
        hostname: Optional[str] = None,
        pid: Optional[int] = None,
        transaction_id: Optional[str] = None,
        release_sha: Optional[str] = None,
        plan_hash: Optional[str] = None,
        config_hash: Optional[str] = None,
        artifact_hashes: Optional[Mapping[str, str]] = None,
        effective_user: Optional[str] = None,
        effective_uid: Optional[int] = None,
        process_start_identity: Optional[str] = None,
    ) -> None:
        if not isinstance(
            capability, (DevelopmentWriteCapability, SyntheticWriteCapability)
        ):
            raise TypeError("WriterLock requires a sealed write capability.")
        if not isinstance(target_identity, str) or not target_identity:
            raise ValueError("Writer lock target identity must be non-empty.")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("Writer lock run id must be non-empty.")
        self.capability = capability
        self.target_identity = target_identity
        self.run_id = run_id
        self.owner = owner or str(os.getuid())
        self.hostname = hostname or socket.gethostname()
        self.pid = os.getpid() if pid is None else int(pid)
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("Writer lock owner must be non-empty.")
        if not isinstance(self.hostname, str) or not self.hostname:
            raise ValueError("Writer lock hostname must be non-empty.")
        if self.pid <= 0:
            raise ValueError("Writer lock PID must be positive.")
        self.transaction_id = transaction_id or run_id
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("Writer lock transaction id must be non-empty.")
        self.release_sha = _optional_text(release_sha, "release_sha")
        self.plan_hash = _optional_text(plan_hash, "plan_hash")
        self.config_hash = _optional_text(config_hash, "config_hash")
        self.artifact_hashes = _artifact_hash_mapping(artifact_hashes)
        uid = os.geteuid() if effective_uid is None else int(effective_uid)
        if uid < 0:
            raise ValueError("Writer lock effective UID must be non-negative.")
        self.effective_uid = uid
        self.effective_user = effective_user or _effective_user(uid)
        if not isinstance(self.effective_user, str) or not self.effective_user:
            raise ValueError("Writer lock effective user must be non-empty.")
        self.process_start_identity = (
            _optional_text(process_start_identity, "process_start_identity")
            if process_start_identity is not None
            else _process_start_identity(self.pid, self.hostname)
        )
        self.token = uuid.uuid4().hex
        self.canonical_root = capability.root.resolve(strict=True)
        self.canonical_target = _canonical_target(capability, target_identity)
        self.path = contained_path(
            capability.lock_root,
            Path(lock_identity(target_identity)),
        )
        self.metadata_path = self.path / "owner.json"
        self._held = False
        self._directory_identity: Optional[Tuple[int, int]] = None
        self._owned_metadata: Optional[Mapping[str, Any]] = None

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
            "schema_version": LOCK_SCHEMA_VERSION,
            "target_identity": self.target_identity,
            "canonical_target": str(self.canonical_target),
            "canonical_root": str(self.canonical_root),
            "transaction_id": self.transaction_id,
            "run_id": self.run_id,
            "release_sha": self.release_sha,
            "plan_hash": self.plan_hash,
            "config_hash": self.config_hash,
            "artifact_hashes": self.artifact_hashes,
            "owner": self.owner,
            "effective_user": self.effective_user,
            "effective_uid": self.effective_uid,
            "hostname": self.hostname,
            "pid": self.pid,
            "process_start_identity": self.process_start_identity,
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
        self._owned_metadata = metadata
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
            metadata_stat = self.metadata_path.lstat()
        except OSError:
            return LockInspection(True, True, False, None)
        if not stat.S_ISREG(metadata_stat.st_mode):
            return LockInspection(True, True, False, None)
        try:
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return LockInspection(True, True, False, None)
        if not isinstance(metadata, dict) or not _valid_lock_metadata(metadata):
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
        if (
            not self._held
            or self._directory_identity is None
            or self._owned_metadata is None
        ):
            raise LockOwnershipError("This lock object does not own the writer lock.")
        if _directory_identity(self.path) != self._directory_identity:
            raise LockOwnershipError("Writer lock directory identity changed.")
        inspection = self.inspect()
        metadata = inspection.metadata
        if inspection.ambiguous or metadata is None:
            raise LockOwnershipError("Writer lock ownership metadata is ambiguous.")
        if metadata != self._owned_metadata:
            raise LockOwnershipError("Writer lock is owned by another operation.")
        if set(self.path.iterdir()) != {self.metadata_path}:
            raise LockOwnershipError("Writer lock directory contents are ambiguous.")
        return metadata

    def update_artifact_hashes(
        self, artifact_hashes: Mapping[str, str]
    ) -> Mapping[str, Any]:
        """Durably attach validated artifact digests while retaining ownership."""
        self.assert_owned()
        normalized = _artifact_hash_mapping(artifact_hashes)
        assert self._owned_metadata is not None
        updated = dict(self._owned_metadata)
        updated["artifact_hashes"] = normalized
        _replace_metadata_fsynced(
            self.metadata_path,
            json.dumps(updated, indent=2, sort_keys=True) + "\n",
        )
        self.artifact_hashes = normalized
        self._owned_metadata = updated
        return self.assert_owned()

    def release(self) -> None:
        self.assert_owned()
        self.metadata_path.unlink()
        _fsync_directory(self.path)
        self.path.rmdir()
        _fsync_directory(self.path.parent)
        self._held = False
        self._directory_identity = None
        self._owned_metadata = None
