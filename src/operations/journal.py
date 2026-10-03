"""Durable, deterministic transaction journals for future writer orchestration.

The journal is an operational record, not a write authorization.  Each update
is written to a new file in the journal's directory, fsynced, atomically
replaced, and followed by a parent-directory fsync.  Callers must still hold
the appropriate writer lock while updating a journal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from .state import ExecutionState, LEGAL_TRANSITIONS


JOURNAL_SCHEMA_VERSION = "1.0"
_JOURNAL_FIELDS = frozenset(
    {
        "schema_version",
        "revision",
        "descriptor",
        "state",
        "outcome",
        "published",
        "reconciliation_required",
        "created_at_utc",
        "updated_at_utc",
        "transitions",
        "artifacts",
        "validation",
        "publication",
        "durability",
        "reconciliation",
        "failure",
        "recovery",
    }
)


class JournalError(RuntimeError):
    """A journal operation could not be completed safely."""


class JournalCorrupt(JournalError):
    """A journal is malformed, inconsistent, or uses an unknown schema."""


class JournalOutcome(str, Enum):
    """Publication-aware durable outcome; never collapse post-commit failure."""

    ACTIVE = "ACTIVE"
    UNPUBLISHED_FAILURE = "UNPUBLISHED_FAILURE"
    PUBLISHED = "PUBLISHED"
    PUBLISHED_DURABILITY_UNCERTAIN = "PUBLISHED_DURABILITY_UNCERTAIN"
    PUBLISHED_RECONCILIATION_REQUIRED = "PUBLISHED_RECONCILIATION_REQUIRED"
    COMPLETE = "COMPLETE"


def _derive_outcome(
    state: ExecutionState,
    published: bool,
    reconciliation_required: bool,
    durability: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    failure: Mapping[str, Any],
) -> JournalOutcome:
    if state == ExecutionState.COMPLETE:
        return JournalOutcome.COMPLETE
    if not published:
        return (
            JournalOutcome.UNPUBLISHED_FAILURE
            if state == ExecutionState.FAILED
            else JournalOutcome.ACTIVE
        )
    durability_status = durability.get("status")
    if durability_status in {"uncertain", "unknown", "indeterminate", "failed"} or (
        failure.get("reason") == "published_durability_uncertain"
    ):
        return JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN
    reconciliation_status = reconciliation.get("status")
    if reconciliation_required or reconciliation_status in {"required", "failed"}:
        return JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED
    return JournalOutcome.PUBLISHED


def _iso(value: Optional[datetime] = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise JournalError("Journal timestamps must be timezone-aware.")
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _validate_timestamp(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise JournalCorrupt(f"Journal field {field_name!r} is not a timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise JournalCorrupt(
            f"Journal field {field_name!r} is not an ISO timestamp."
        ) from exc
    if parsed.tzinfo is None:
        raise JournalCorrupt(f"Journal field {field_name!r} lacks a timezone.")
    return value


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JournalError("Journal JSON cannot contain non-finite numbers.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            if not isinstance(key, str):
                raise JournalError("Journal mapping keys must be strings.")
            result[key] = _json_value(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise JournalError(
        f"Journal JSON cannot serialize values of type {type(value).__name__}."
    )


def _mapping(value: object, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise JournalCorrupt(f"Journal field {field_name!r} must be an object.")
    try:
        normalized = _json_value(value)
    except JournalError as exc:
        raise JournalCorrupt(f"Journal field {field_name!r} is invalid.") from exc
    if not isinstance(normalized, dict):
        raise JournalCorrupt(f"Journal field {field_name!r} must be an object.")
    return normalized


def _nonempty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JournalCorrupt(f"Journal field {field_name!r} must be non-empty.")
    return value


@dataclass(frozen=True)
class TransactionDescriptor:
    """Stable identity and all paths needed to inspect one transaction."""

    run_id: str
    operation: str
    target_identity: str
    target_path: str
    stage_path: str
    lock_path: str
    profile: str
    plan_id: Optional[str] = None
    release_sha: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "operation",
            "target_identity",
            "target_path",
            "stage_path",
            "lock_path",
            "profile",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ).strip():
                raise JournalError(f"Transaction descriptor {name} must be non-empty.")
        if self.plan_id is not None and not isinstance(self.plan_id, str):
            raise JournalError("Transaction descriptor plan_id must be a string or null.")
        if self.release_sha is not None and not isinstance(self.release_sha, str):
            raise JournalError(
                "Transaction descriptor release_sha must be a string or null."
            )
        _json_value(self.metadata)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "operation": self.operation,
            "target_identity": self.target_identity,
            "target_path": self.target_path,
            "stage_path": self.stage_path,
            "lock_path": self.lock_path,
            "profile": self.profile,
            "plan_id": self.plan_id,
            "release_sha": self.release_sha,
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> "TransactionDescriptor":
        data = _mapping(value, "descriptor")
        expected = {
            "run_id",
            "operation",
            "target_identity",
            "target_path",
            "stage_path",
            "lock_path",
            "profile",
            "plan_id",
            "release_sha",
            "metadata",
        }
        if set(data) != expected:
            raise JournalCorrupt("Transaction descriptor fields are incomplete or unknown.")
        try:
            return cls(
                run_id=_nonempty(data["run_id"], "descriptor.run_id"),
                operation=_nonempty(data["operation"], "descriptor.operation"),
                target_identity=_nonempty(
                    data["target_identity"], "descriptor.target_identity"
                ),
                target_path=_nonempty(data["target_path"], "descriptor.target_path"),
                stage_path=_nonempty(data["stage_path"], "descriptor.stage_path"),
                lock_path=_nonempty(data["lock_path"], "descriptor.lock_path"),
                profile=_nonempty(data["profile"], "descriptor.profile"),
                plan_id=data["plan_id"],
                release_sha=data["release_sha"],
                metadata=_mapping(data["metadata"], "descriptor.metadata"),
            )
        except JournalError as exc:
            raise JournalCorrupt("Transaction descriptor is invalid.") from exc


@dataclass(frozen=True)
class ArtifactIdentity:
    """Content and filesystem identity captured at a durable boundary."""

    name: str
    path: str
    device: int
    inode: int
    size: int
    mode: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise JournalError("Artifact identity requires a name.")
        if not self.path or not isinstance(self.path, str):
            raise JournalError("Artifact identity requires a path.")
        for field_name in ("device", "inode", "size", "mode"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise JournalError(
                    f"Artifact identity {field_name} must be a non-negative integer."
                )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise JournalError("Artifact identity requires a lowercase SHA-256 digest.")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactIdentity":
        data = _mapping(value, "artifact")
        expected = {"name", "path", "device", "inode", "size", "mode", "sha256"}
        if set(data) != expected:
            raise JournalCorrupt("Artifact identity fields are incomplete or unknown.")
        try:
            return cls(**data)
        except (JournalError, TypeError) as exc:
            raise JournalCorrupt("Artifact identity is invalid.") from exc

    @classmethod
    def from_path(cls, name: str, path: Path) -> "ArtifactIdentity":
        """Read and hash one stable regular file without following its final symlink."""
        path = Path(path)
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise JournalError(f"Artifact is not a regular file: {path}.")
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
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
                raise JournalError(f"Artifact changed while it was hashed: {path}.")
            return cls(
                name=name,
                path=str(path),
                device=after.st_dev,
                inode=after.st_ino,
                size=after.st_size,
                mode=stat.S_IMODE(after.st_mode),
                sha256=digest.hexdigest(),
            )
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class JournalTransition:
    sequence: int
    previous: ExecutionState
    current: ExecutionState
    at_utc: str
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "previous": self.previous.value,
            "current": self.current.value,
            "at_utc": self.at_utc,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: object) -> "JournalTransition":
        data = _mapping(value, "transition")
        if set(data) != {"sequence", "previous", "current", "at_utc", "reason"}:
            raise JournalCorrupt("Journal transition fields are incomplete or unknown.")
        if type(data["sequence"]) is not int or data["sequence"] <= 0:
            raise JournalCorrupt("Journal transition sequence must be positive.")
        try:
            previous = ExecutionState(data["previous"])
            current = ExecutionState(data["current"])
        except (TypeError, ValueError) as exc:
            raise JournalCorrupt("Journal transition uses an unknown state.") from exc
        reason = data["reason"]
        if reason is not None and not isinstance(reason, str):
            raise JournalCorrupt("Journal transition reason must be a string or null.")
        return cls(
            sequence=data["sequence"],
            previous=previous,
            current=current,
            at_utc=_validate_timestamp(data["at_utc"], "transition.at_utc"),
            reason=reason,
        )


@dataclass(frozen=True)
class JournalSnapshot:
    schema_version: str
    revision: int
    descriptor: TransactionDescriptor
    state: ExecutionState
    outcome: JournalOutcome
    published: bool
    reconciliation_required: bool
    created_at_utc: str
    updated_at_utc: str
    transitions: Tuple[JournalTransition, ...]
    artifacts: Mapping[str, ArtifactIdentity]
    validation: Mapping[str, Any]
    publication: Mapping[str, Any]
    durability: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    failure: Mapping[str, Any]
    recovery: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "descriptor": self.descriptor.as_dict(),
            "state": self.state.value,
            "outcome": self.outcome.value,
            "published": self.published,
            "reconciliation_required": self.reconciliation_required,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
            "transitions": [item.as_dict() for item in self.transitions],
            "artifacts": {
                name: self.artifacts[name].as_dict()
                for name in sorted(self.artifacts)
            },
            "validation": _json_value(self.validation),
            "publication": _json_value(self.publication),
            "durability": _json_value(self.durability),
            "reconciliation": _json_value(self.reconciliation),
            "failure": _json_value(self.failure),
            "recovery": _json_value(self.recovery),
        }

    @classmethod
    def from_dict(cls, value: object) -> "JournalSnapshot":
        data = _mapping(value, "journal")
        if set(data) != _JOURNAL_FIELDS:
            raise JournalCorrupt("Journal fields are incomplete or unknown.")
        if data["schema_version"] != JOURNAL_SCHEMA_VERSION:
            raise JournalCorrupt(
                f"Unsupported journal schema: {data['schema_version']!r}."
            )
        if type(data["revision"]) is not int or data["revision"] < 0:
            raise JournalCorrupt("Journal revision must be non-negative.")
        try:
            state = ExecutionState(data["state"])
        except (TypeError, ValueError) as exc:
            raise JournalCorrupt("Journal state is unknown.") from exc
        try:
            outcome = JournalOutcome(data["outcome"])
        except (TypeError, ValueError) as exc:
            raise JournalCorrupt("Journal outcome is unknown.") from exc
        if type(data["published"]) is not bool or type(
            data["reconciliation_required"]
        ) is not bool:
            raise JournalCorrupt("Journal publication flags must be booleans.")
        if not isinstance(data["transitions"], list):
            raise JournalCorrupt("Journal transitions must be an array.")
        transitions = tuple(
            JournalTransition.from_dict(item) for item in data["transitions"]
        )
        cursor = ExecutionState.PLANNED
        published = False
        for sequence, transition in enumerate(transitions, start=1):
            if transition.sequence != sequence or transition.previous != cursor:
                raise JournalCorrupt("Journal transition history is discontinuous.")
            if transition.current not in LEGAL_TRANSITIONS[cursor]:
                raise JournalCorrupt("Journal transition history is illegal.")
            cursor = transition.current
            if cursor == ExecutionState.PUBLISHED:
                published = True
        if cursor != state:
            raise JournalCorrupt("Journal state disagrees with transition history.")
        if published != data["published"]:
            raise JournalCorrupt("Journal published flag disagrees with its history.")
        expected_reconciliation = bool(state == ExecutionState.FAILED and published)
        if state == ExecutionState.COMPLETE:
            expected_reconciliation = False
        if data["reconciliation_required"] != expected_reconciliation:
            raise JournalCorrupt(
                "Journal reconciliation flag disagrees with its lifecycle state."
            )
        if data["revision"] < len(transitions):
            raise JournalCorrupt("Journal revision predates its transition history.")
        artifact_data = _mapping(data["artifacts"], "artifacts")
        artifacts: Dict[str, ArtifactIdentity] = {}
        for name, item in artifact_data.items():
            identity = ArtifactIdentity.from_dict(item)
            if identity.name != name:
                raise JournalCorrupt("Artifact mapping key disagrees with its identity.")
            artifacts[name] = identity
        failure = _mapping(data["failure"], "failure")
        if state == ExecutionState.FAILED and not isinstance(
            failure.get("reason"), str
        ):
            raise JournalCorrupt("A failed journal requires a failure reason.")
        if state != ExecutionState.FAILED and failure.get("reason") is not None:
            raise JournalCorrupt("A non-failed journal records a terminal failure.")
        validation = _mapping(data["validation"], "validation")
        publication = _mapping(data["publication"], "publication")
        durability = _mapping(data["durability"], "durability")
        reconciliation = _mapping(data["reconciliation"], "reconciliation")
        recovery = _mapping(data["recovery"], "recovery")
        derived_outcome = _derive_outcome(
            state,
            published,
            data["reconciliation_required"],
            durability,
            reconciliation,
            failure,
        )
        if outcome != derived_outcome:
            raise JournalCorrupt(
                "Journal outcome disagrees with publication-aware evidence."
            )
        return cls(
            schema_version=JOURNAL_SCHEMA_VERSION,
            revision=data["revision"],
            descriptor=TransactionDescriptor.from_dict(data["descriptor"]),
            state=state,
            outcome=outcome,
            published=published,
            reconciliation_required=data["reconciliation_required"],
            created_at_utc=_validate_timestamp(
                data["created_at_utc"], "created_at_utc"
            ),
            updated_at_utc=_validate_timestamp(
                data["updated_at_utc"], "updated_at_utc"
            ),
            transitions=transitions,
            artifacts=artifacts,
            validation=validation,
            publication=publication,
            durability=durability,
            reconciliation=reconciliation,
            failure=failure,
            recovery=recovery,
        )


def _encode(snapshot: JournalSnapshot) -> bytes:
    try:
        return (
            json.dumps(snapshot.as_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, JournalError) as exc:
        raise JournalError("Journal could not be serialized deterministically.") from exc


def _parent_fd(path: Path) -> Tuple[Path, int]:
    path = Path(path)
    if not path.name or path.name in {".", ".."}:
        raise JournalError("Journal path must name a file.")
    parent = path.parent
    if parent.is_symlink() or not parent.is_dir():
        raise JournalError("Journal parent must be an existing real directory.")
    try:
        descriptor = os.open(
            str(parent),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError as exc:
        raise JournalError("Journal parent could not be opened safely.") from exc
    return parent, descriptor


def _read_bytes(path: Path) -> bytes:
    _, directory_fd = _parent_fd(path)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            observed = os.fstat(descriptor)
            if not stat.S_ISREG(observed.st_mode):
                raise JournalCorrupt("Journal path is not a regular file.")
            if observed.st_size > 16 * 1024 * 1024:
                raise JournalCorrupt("Journal exceeds the 16 MiB safety limit.")
            chunks = []
            remaining = 16 * 1024 * 1024 + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > 16 * 1024 * 1024:
                raise JournalCorrupt("Journal exceeds the 16 MiB safety limit.")
            return payload
        finally:
            os.close(descriptor)
    except FileNotFoundError as exc:
        raise JournalError(f"Journal does not exist: {path}.") from exc
    except OSError as exc:
        raise JournalCorrupt("Journal could not be opened safely.") from exc
    finally:
        os.close(directory_fd)


def _decode(payload: bytes) -> JournalSnapshot:
    def unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        value: Dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise JournalCorrupt(f"Journal contains duplicate key {key!r}.")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=unique_object
        )
    except JournalCorrupt:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise JournalCorrupt("Journal is not valid UTF-8 JSON.") from exc
    return JournalSnapshot.from_dict(value)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Short journal write.")
        offset += written


def _atomic_write(path: Path, payload: bytes, *, create: bool) -> None:
    _, directory_fd = _parent_fd(path)
    temporary_name = f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary_created = False
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_fd,
        )
        temporary_created = True
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if create:
            try:
                os.link(
                    temporary_name,
                    path.name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise JournalError(f"Journal already exists: {path}.") from exc
            os.unlink(temporary_name, dir_fd=directory_fd)
            temporary_created = False
        else:
            try:
                observed = os.stat(
                    path.name, dir_fd=directory_fd, follow_symlinks=False
                )
            except FileNotFoundError as exc:
                raise JournalError(f"Journal disappeared before update: {path}.") from exc
            if not stat.S_ISREG(observed.st_mode):
                raise JournalError("Journal update target is not a regular file.")
            os.replace(
                temporary_name,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            temporary_created = False
        os.fsync(directory_fd)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _merged(current: Mapping[str, Any], update: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result = dict(_json_value(current))
    if update is not None:
        normalized = _json_value(update)
        if not isinstance(normalized, dict):
            raise JournalError("Journal section updates must be mappings.")
        result.update(normalized)
    return result


class TransactionJournal:
    """Loaded durable journal with optimistic revision checks."""

    def __init__(self, path: Path, snapshot: JournalSnapshot) -> None:
        self.path = Path(path)
        self._snapshot = snapshot

    @property
    def snapshot(self) -> JournalSnapshot:
        return self._snapshot

    @classmethod
    def create(
        cls,
        path: Path,
        descriptor: TransactionDescriptor,
        *,
        at: Optional[datetime] = None,
    ) -> "TransactionJournal":
        timestamp = _iso(at)
        snapshot = JournalSnapshot(
            schema_version=JOURNAL_SCHEMA_VERSION,
            revision=0,
            descriptor=descriptor,
            state=ExecutionState.PLANNED,
            outcome=JournalOutcome.ACTIVE,
            published=False,
            reconciliation_required=False,
            created_at_utc=timestamp,
            updated_at_utc=timestamp,
            transitions=(),
            artifacts={},
            validation={},
            publication={"attempted": False, "committed": False},
            durability={},
            reconciliation={},
            failure={},
            recovery={},
        )
        try:
            _atomic_write(Path(path), _encode(snapshot), create=True)
        except JournalError:
            raise
        except OSError as exc:
            raise JournalError("Journal creation failed durably.") from exc
        return cls(path, snapshot)

    @classmethod
    def load(cls, path: Path) -> "TransactionJournal":
        return cls(path, _decode(_read_bytes(Path(path))))

    def _current(self, expected_revision: Optional[int]) -> JournalSnapshot:
        current = _decode(_read_bytes(self.path))
        expected = self._snapshot.revision if expected_revision is None else expected_revision
        if current.descriptor.run_id != self._snapshot.descriptor.run_id:
            raise JournalError("Journal run identity changed on disk.")
        if current.revision != expected:
            raise JournalError(
                f"Journal revision conflict: expected {expected}, found {current.revision}."
            )
        return current

    def update(
        self,
        *,
        artifacts: Optional[Mapping[str, ArtifactIdentity]] = None,
        validation: Optional[Mapping[str, Any]] = None,
        publication: Optional[Mapping[str, Any]] = None,
        durability: Optional[Mapping[str, Any]] = None,
        reconciliation: Optional[Mapping[str, Any]] = None,
        failure: Optional[Mapping[str, Any]] = None,
        recovery: Optional[Mapping[str, Any]] = None,
        expected_revision: Optional[int] = None,
        at: Optional[datetime] = None,
    ) -> JournalSnapshot:
        current = self._current(expected_revision)
        if failure is not None and current.state != ExecutionState.FAILED:
            if "reason" in failure:
                raise JournalError(
                    "A terminal failure reason may only accompany a FAILED state."
                )
        artifact_values = dict(current.artifacts)
        if artifacts is not None:
            for name, identity in artifacts.items():
                if not isinstance(name, str) or not isinstance(
                    identity, ArtifactIdentity
                ):
                    raise JournalError(
                        "Artifact updates must map names to ArtifactIdentity values."
                    )
                if identity.name != name:
                    raise JournalError(
                        "Artifact update key disagrees with the identity name."
                    )
                artifact_values[name] = identity
        validation_values = _merged(current.validation, validation)
        publication_values = _merged(current.publication, publication)
        durability_values = _merged(current.durability, durability)
        reconciliation_values = _merged(current.reconciliation, reconciliation)
        failure_values = _merged(current.failure, failure)
        recovery_values = _merged(current.recovery, recovery)
        snapshot = JournalSnapshot(
            schema_version=JOURNAL_SCHEMA_VERSION,
            revision=current.revision + 1,
            descriptor=current.descriptor,
            state=current.state,
            outcome=_derive_outcome(
                current.state,
                current.published,
                current.reconciliation_required,
                durability_values,
                reconciliation_values,
                failure_values,
            ),
            published=current.published,
            reconciliation_required=current.reconciliation_required,
            created_at_utc=current.created_at_utc,
            updated_at_utc=_iso(at),
            transitions=current.transitions,
            artifacts=artifact_values,
            validation=validation_values,
            publication=publication_values,
            durability=durability_values,
            reconciliation=reconciliation_values,
            failure=failure_values,
            recovery=recovery_values,
        )
        try:
            _atomic_write(self.path, _encode(snapshot), create=False)
        except JournalError:
            raise
        except OSError as exc:
            raise JournalError("Journal update failed durably.") from exc
        self._snapshot = snapshot
        return snapshot

    def transition(
        self,
        target: Union[ExecutionState, str],
        *,
        reason: Optional[str] = None,
        artifacts: Optional[Mapping[str, ArtifactIdentity]] = None,
        validation: Optional[Mapping[str, Any]] = None,
        publication: Optional[Mapping[str, Any]] = None,
        durability: Optional[Mapping[str, Any]] = None,
        reconciliation: Optional[Mapping[str, Any]] = None,
        failure: Optional[Mapping[str, Any]] = None,
        recovery: Optional[Mapping[str, Any]] = None,
        expected_revision: Optional[int] = None,
        at: Optional[datetime] = None,
    ) -> JournalSnapshot:
        current = self._current(expected_revision)
        try:
            next_state = target if isinstance(target, ExecutionState) else ExecutionState(target)
        except (TypeError, ValueError) as exc:
            raise JournalError(f"Unknown journal state: {target!r}.") from exc
        if next_state not in LEGAL_TRANSITIONS[current.state]:
            raise JournalError(
                f"Illegal journal transition: {current.state.value} -> {next_state.value}."
            )
        timestamp = _iso(at)
        published = current.published or next_state == ExecutionState.PUBLISHED
        reconciliation_required = bool(
            next_state == ExecutionState.FAILED and published
        )
        if next_state == ExecutionState.COMPLETE:
            reconciliation_required = False
        artifact_values = dict(current.artifacts)
        if artifacts is not None:
            for name, identity in artifacts.items():
                if not isinstance(name, str) or not isinstance(
                    identity, ArtifactIdentity
                ) or identity.name != name:
                    raise JournalError("Artifact transition evidence is invalid.")
                artifact_values[name] = identity
        failure_values = _merged(current.failure, failure)
        if next_state == ExecutionState.FAILED:
            failure_values["reason"] = reason or str(
                failure_values.get("reason") or "unspecified_failure"
            )
            failure_values["at_utc"] = timestamp
        transition = JournalTransition(
            sequence=len(current.transitions) + 1,
            previous=current.state,
            current=next_state,
            at_utc=timestamp,
            reason=reason,
        )
        validation_values = _merged(current.validation, validation)
        publication_values = _merged(current.publication, publication)
        durability_values = _merged(current.durability, durability)
        reconciliation_values = _merged(current.reconciliation, reconciliation)
        recovery_values = _merged(current.recovery, recovery)
        snapshot = JournalSnapshot(
            schema_version=JOURNAL_SCHEMA_VERSION,
            revision=current.revision + 1,
            descriptor=current.descriptor,
            state=next_state,
            outcome=_derive_outcome(
                next_state,
                published,
                reconciliation_required,
                durability_values,
                reconciliation_values,
                failure_values,
            ),
            published=published,
            reconciliation_required=reconciliation_required,
            created_at_utc=current.created_at_utc,
            updated_at_utc=timestamp,
            transitions=current.transitions + (transition,),
            artifacts=artifact_values,
            validation=validation_values,
            publication=publication_values,
            durability=durability_values,
            reconciliation=reconciliation_values,
            failure=failure_values,
            recovery=recovery_values,
        )
        try:
            _atomic_write(self.path, _encode(snapshot), create=False)
        except JournalError:
            raise
        except OSError as exc:
            raise JournalError("Journal transition failed durably.") from exc
        self._snapshot = snapshot
        return snapshot
