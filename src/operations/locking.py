"""Conservative single-writer lock contract for local temporary fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

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

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self, *, at: Optional[datetime] = None) -> Mapping[str, Any]:
        timestamp = at or datetime.now(timezone.utc)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError as exc:
            inspection = self.inspect()
            raise LockUnavailable(
                "Writer lock already exists; it will not be stolen automatically. "
                f"ambiguous={inspection.ambiguous}"
            ) from exc
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
            self.metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            self.path.rmdir()
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

    def release(self) -> None:
        if not self._held:
            raise LockOwnershipError("This lock object does not own the writer lock.")
        inspection = self.inspect()
        metadata = inspection.metadata
        if inspection.ambiguous or metadata is None:
            raise LockOwnershipError("Writer lock ownership metadata is ambiguous.")
        if (
            metadata.get("run_id") != self.run_id
            or metadata.get("ownership_token") != self.token
            or metadata.get("target_identity") != self.target_identity
        ):
            raise LockOwnershipError("Writer lock is owned by another operation.")
        unexpected = [path for path in self.path.iterdir() if path != self.metadata_path]
        if unexpected:
            raise LockOwnershipError("Writer lock directory contains unexpected files.")
        self.metadata_path.unlink()
        self.path.rmdir()
        self._held = False
