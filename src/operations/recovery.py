"""Read-only inspection and fail-closed recovery classification.

This module deliberately exposes no deletion, lock-release, resume, or repair
operation.  A journal is evidence, not truth: classifications are derived from
the journal and independently observed target, stage, lock, artifact, symlink,
identity, checksum, and manifest state.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .journal import (
    ArtifactIdentity,
    JournalError,
    JournalSnapshot,
    TransactionJournal,
)
from .state import ExecutionState


REQUIRED_ARTIFACTS = ("loci.parquet", "alerts.parquet", "manifest.json")
PENDING_MANIFEST = ".manifest.pending"


class RecoveryDisposition(str, Enum):
    SAFE_TO_DISCARD = "SAFE_TO_DISCARD"
    SAFE_TO_RESUME = "SAFE_TO_RESUME"
    REQUIRES_REVALIDATION = "REQUIRES_REVALIDATION"
    REQUIRES_RECONCILIATION = "REQUIRES_RECONCILIATION"
    REQUIRES_OPERATOR_DECISION = "REQUIRES_OPERATOR_DECISION"
    MUST_NOT_AUTO_DELETE = "MUST_NOT_AUTO_DELETE"


@dataclass(frozen=True)
class ArtifactObservation:
    name: str
    path: str
    kind: str
    device: Optional[int] = None
    inode: Optional[int] = None
    size: Optional[int] = None
    mode: Optional[int] = None
    sha256: Optional[str] = None
    error: Optional[str] = None

    @property
    def regular(self) -> bool:
        return self.kind == "regular"

    def matches(self, expected: ArtifactIdentity) -> bool:
        return bool(
            self.regular
            and self.name == expected.name
            and self.device == expected.device
            and self.inode == expected.inode
            and self.size == expected.size
            and self.mode == expected.mode
            and self.sha256 == expected.sha256
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
            "mode": self.mode,
            "sha256": self.sha256,
            "error": self.error,
        }


@dataclass(frozen=True)
class DirectoryObservation:
    path: str
    kind: str
    device: Optional[int] = None
    inode: Optional[int] = None
    mode: Optional[int] = None
    entries: Tuple[str, ...] = ()
    unexpected_entries: Tuple[str, ...] = ()
    artifacts: Mapping[str, ArtifactObservation] = field(default_factory=dict)
    json_valid: Mapping[str, Optional[bool]] = field(default_factory=dict)
    json_summary: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def exists(self) -> bool:
        return self.kind != "absent"

    @property
    def safe_directory(self) -> bool:
        return self.kind == "directory"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "entries": list(self.entries),
            "unexpected_entries": list(self.unexpected_entries),
            "artifacts": {
                name: self.artifacts[name].as_dict()
                for name in sorted(self.artifacts)
            },
            "json_valid": dict(self.json_valid),
            "json_summary": {
                name: dict(self.json_summary[name])
                for name in sorted(self.json_summary)
            },
            "error": self.error,
        }


@dataclass(frozen=True)
class RecoveryAssessment:
    journal_path: str
    journal: Optional[JournalSnapshot]
    target: DirectoryObservation
    stage: DirectoryObservation
    lock: DirectoryObservation
    dispositions: Tuple[RecoveryDisposition, ...]
    summary: str
    disagreements: Tuple[str, ...]
    lock_owner_status: str
    read_only: bool = True

    @property
    def requires_operator(self) -> bool:
        return RecoveryDisposition.REQUIRES_OPERATOR_DECISION in self.dispositions

    def as_dict(self) -> Dict[str, Any]:
        return {
            "journal_path": self.journal_path,
            "journal": None if self.journal is None else self.journal.as_dict(),
            "target": self.target.as_dict(),
            "stage": self.stage.as_dict(),
            "lock": self.lock.as_dict(),
            "dispositions": [item.value for item in self.dispositions],
            "summary": self.summary,
            "disagreements": list(self.disagreements),
            "lock_owner_status": self.lock_owner_status,
            "read_only": True,
        }


def _observe_artifact(path: Path, name: str) -> ArtifactObservation:
    try:
        initial = path.lstat()
    except FileNotFoundError:
        return ArtifactObservation(name, str(path), "absent")
    except OSError as exc:
        return ArtifactObservation(
            name, str(path), "error", error=f"{type(exc).__name__}: {exc}"
        )
    if stat.S_ISLNK(initial.st_mode):
        return ArtifactObservation(
            name,
            str(path),
            "symlink",
            device=initial.st_dev,
            inode=initial.st_ino,
            mode=stat.S_IMODE(initial.st_mode),
        )
    if not stat.S_ISREG(initial.st_mode):
        return ArtifactObservation(
            name,
            str(path),
            "other",
            device=initial.st_dev,
            inode=initial.st_ino,
            size=initial.st_size,
            mode=stat.S_IMODE(initial.st_mode),
        )
    try:
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OSError("Observed artifact ceased to be regular.")
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        return ArtifactObservation(
            name, str(path), "error", error=f"{type(exc).__name__}: {exc}"
        )
    stable_before = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
    )
    stable_open = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if stable_before != stable_open or stable_open != stable_after:
        return ArtifactObservation(
            name,
            str(path),
            "changed",
            device=after.st_dev,
            inode=after.st_ino,
            size=after.st_size,
            mode=stat.S_IMODE(after.st_mode),
            error="Artifact identity changed during inspection.",
        )
    return ArtifactObservation(
        name=name,
        path=str(path),
        kind="regular",
        device=after.st_dev,
        inode=after.st_ino,
        size=after.st_size,
        mode=stat.S_IMODE(after.st_mode),
        sha256=digest.hexdigest(),
    )


def _read_json_object(
    path: Path,
    expected: ArtifactObservation,
) -> Tuple[bool, Mapping[str, Any]]:
    try:
        descriptor = os.open(
            str(path),
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > 16 * 1024 * 1024
                or before.st_dev != expected.device
                or before.st_ino != expected.inode
                or before.st_size != expected.size
                or stat.S_IMODE(before.st_mode) != expected.mode
            ):
                return False, {}
            digest = hashlib.sha256()
            chunks = []
            remaining = 16 * 1024 * 1024 + 1
            while remaining > 0:
                chunk = os.read(descriptor, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        payload = b"".join(chunks)
        if (
            len(payload) > 16 * 1024 * 1024
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or digest.hexdigest() != expected.sha256
        ):
            return False, {}
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False, {}
    if not isinstance(value, dict):
        return False, {}
    return True, value


def _json_summary(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if name == "owner.json":
        allowed = (
            "schema_version",
            "target_identity",
            "canonical_target",
            "transaction_id",
            "run_id",
            "owner",
            "effective_user",
            "effective_uid",
            "hostname",
            "pid",
            "process_start_identity",
            "acquired_at_utc",
        )
        return {key: value.get(key) for key in allowed}
    if name == "manifest.json":
        return {
            "date_utc": value.get("date_utc"),
            "status": value.get("status"),
            "actual_loci": value.get("actual_loci"),
            "alert_rows": value.get("alert_rows"),
        }
    return {}


def _observe_directory(
    path: Path,
    *,
    known_entries: Sequence[str],
    artifact_names: Sequence[str],
    json_names: Sequence[str],
) -> DirectoryObservation:
    path = Path(path)
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return DirectoryObservation(str(path), "absent")
    except OSError as exc:
        return DirectoryObservation(
            str(path), "error", error=f"{type(exc).__name__}: {exc}"
        )
    if stat.S_ISLNK(observed.st_mode):
        return DirectoryObservation(
            str(path),
            "symlink",
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=stat.S_IMODE(observed.st_mode),
        )
    if not stat.S_ISDIR(observed.st_mode):
        return DirectoryObservation(
            str(path),
            "other",
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=stat.S_IMODE(observed.st_mode),
        )
    try:
        entries = tuple(sorted(item.name for item in path.iterdir()))
    except OSError as exc:
        return DirectoryObservation(
            str(path),
            "error",
            device=observed.st_dev,
            inode=observed.st_ino,
            mode=stat.S_IMODE(observed.st_mode),
            error=f"{type(exc).__name__}: {exc}",
        )
    known = set(known_entries)
    artifacts = {
        name: _observe_artifact(path / name, name) for name in artifact_names
    }
    json_valid: Dict[str, Optional[bool]] = {}
    summaries: Dict[str, Mapping[str, Any]] = {}
    for name in json_names:
        artifact = artifacts.get(name) or _observe_artifact(path / name, name)
        if artifact.kind == "absent":
            json_valid[name] = None
        elif not artifact.regular:
            json_valid[name] = False
        else:
            valid, value = _read_json_object(path / name, artifact)
            json_valid[name] = valid
            if valid:
                summaries[name] = _json_summary(name, value)
    try:
        final = path.lstat()
    except OSError as exc:
        return DirectoryObservation(
            str(path), "changed", error=f"{type(exc).__name__}: {exc}"
        )
    if (
        not stat.S_ISDIR(final.st_mode)
        or (final.st_dev, final.st_ino) != (observed.st_dev, observed.st_ino)
    ):
        return DirectoryObservation(
            str(path),
            "changed",
            device=final.st_dev,
            inode=final.st_ino,
            mode=stat.S_IMODE(final.st_mode),
            error="Directory identity changed during inspection.",
        )
    return DirectoryObservation(
        path=str(path),
        kind="directory",
        device=observed.st_dev,
        inode=observed.st_ino,
        mode=stat.S_IMODE(observed.st_mode),
        entries=entries,
        unexpected_entries=tuple(sorted(set(entries) - known)),
        artifacts=artifacts,
        json_valid=json_valid,
        json_summary=summaries,
    )


def _absolute(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _unsafe_observation(
    label: str,
    observation: DirectoryObservation,
    disagreements: list[str],
) -> None:
    if observation.kind in {"symlink", "other", "changed", "error"}:
        disagreements.append(f"{label}_root_is_{observation.kind}")
    if observation.unexpected_entries:
        disagreements.append(f"{label}_contains_unexpected_entries")
    for name, artifact in observation.artifacts.items():
        if artifact.kind in {"symlink", "other", "changed", "error"}:
            disagreements.append(f"{label}_{name}_is_{artifact.kind}")


def _append_unique(
    values: list[RecoveryDisposition], disposition: RecoveryDisposition
) -> None:
    if disposition not in values:
        values.append(disposition)


def _lock_owner_status(lock: DirectoryObservation) -> str:
    """Classify liveness using host, PID, and start identity; never PID alone."""
    if not lock.exists:
        return "ABSENT"
    if (
        not lock.safe_directory
        or lock.json_valid.get("owner.json") is not True
    ):
        return "AMBIGUOUS"
    metadata = lock.json_summary.get("owner.json", {})
    hostname = metadata.get("hostname")
    pid = metadata.get("pid")
    recorded_start = metadata.get("process_start_identity")
    if (
        not isinstance(hostname, str)
        or type(pid) is not int
        or pid <= 0
        or not isinstance(recorded_start, str)
        or not recorded_start
    ):
        return "AMBIGUOUS"
    local_host = socket.gethostname().split(".", 1)[0].lower()
    if hostname.split(".", 1)[0].lower() != local_host:
        return "AMBIGUOUS"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "STALE"
    except PermissionError:
        return "AMBIGUOUS"
    except OSError:
        return "AMBIGUOUS"
    from .locking import _process_start_identity

    observed_start = _process_start_identity(pid, hostname)
    if observed_start.startswith("unavailable:"):
        return "AMBIGUOUS"
    return "ACTIVE" if observed_start == recorded_start else "STALE"


def inspect_recovery(
    journal_path: Path,
    *,
    target_path: Path,
    stage_path: Path,
    lock_path: Path,
) -> RecoveryAssessment:
    """Inspect durable evidence and classify it without changing any path."""
    target = _observe_directory(
        target_path,
        known_entries=REQUIRED_ARTIFACTS + (PENDING_MANIFEST,),
        artifact_names=REQUIRED_ARTIFACTS + (PENDING_MANIFEST,),
        json_names=("manifest.json",),
    )
    stage = _observe_directory(
        stage_path,
        known_entries=REQUIRED_ARTIFACTS,
        artifact_names=REQUIRED_ARTIFACTS,
        json_names=("manifest.json",),
    )
    lock = _observe_directory(
        lock_path,
        known_entries=("owner.json",),
        artifact_names=("owner.json",),
        json_names=("owner.json",),
    )
    disagreements: list[str] = []
    lock_owner_status = _lock_owner_status(lock)
    journal: Optional[JournalSnapshot]
    try:
        journal = TransactionJournal.load(journal_path).snapshot
    except JournalError as exc:
        journal = None
        disagreements.append(f"journal_invalid:{type(exc).__name__}:{exc}")
    try:
        temporary_journals = tuple(
            journal_path.parent.glob(f".{journal_path.name}.tmp-*")
        )
    except OSError:
        temporary_journals = (journal_path,)
    if temporary_journals:
        disagreements.append("orphan_journal_temporary_present")

    _unsafe_observation("target", target, disagreements)
    _unsafe_observation("stage", stage, disagreements)
    _unsafe_observation("lock", lock, disagreements)

    if journal is not None:
        descriptor_paths = {
            "target": journal.descriptor.target_path,
            "stage": journal.descriptor.stage_path,
            "lock": journal.descriptor.lock_path,
        }
        explicit_paths = {
            "target": str(target_path),
            "stage": str(stage_path),
            "lock": str(lock_path),
        }
        for name in descriptor_paths:
            if _absolute(descriptor_paths[name]) != _absolute(explicit_paths[name]):
                disagreements.append(f"{name}_path_disagrees_with_journal")

        attempted = journal.publication.get("attempted", False)
        committed = journal.publication.get("committed", False)
        if type(attempted) is not bool or type(committed) is not bool:
            disagreements.append("publication_flags_are_malformed")

        physical_by_name = {
            name: tuple(
                item
                for item in (target.artifacts.get(name), stage.artifacts.get(name))
                if item is not None and item.kind != "absent"
            )
            for name in REQUIRED_ARTIFACTS
        }
        for name, observations in physical_by_name.items():
            expected = journal.artifacts.get(name)
            if observations and expected is None:
                disagreements.append(f"journal_missing_{name}_identity")
            elif expected is not None:
                allowed_identity_paths = {
                    _absolute(str(Path(target_path) / name)),
                    _absolute(str(Path(stage_path) / name)),
                }
                if _absolute(expected.path) not in allowed_identity_paths:
                    disagreements.append(f"journal_{name}_path_is_out_of_scope")
                for observation in observations:
                    if not observation.matches(expected):
                        disagreements.append(f"{name}_identity_or_checksum_disagrees")

    target_complete = bool(
        target.safe_directory
        and all(target.artifacts[name].regular for name in REQUIRED_ARTIFACTS)
        and target.json_valid.get("manifest.json") is True
    )
    target_reserved = target.safe_directory
    target_has_content = bool(target.safe_directory and target.entries)
    stage_complete = bool(
        stage.safe_directory
        and all(stage.artifacts[name].regular for name in REQUIRED_ARTIFACTS)
        and stage.json_valid.get("manifest.json") is True
    )
    stage_has_content = bool(stage.safe_directory and stage.entries)
    lock_present = lock.exists

    if journal is not None:
        if target_complete and not journal.published:
            disagreements.append("visible_manifest_disagrees_with_unpublished_journal")
        if journal.published and not target_complete:
            disagreements.append("published_journal_lacks_complete_visible_target")
        if journal.state in {
            ExecutionState.STAGED,
            ExecutionState.VALIDATED,
        } and not stage_has_content:
            disagreements.append("journal_stage_state_lacks_stage_artifacts")
        if journal.state in {
            ExecutionState.LOCKED,
            ExecutionState.QUERYING,
            ExecutionState.FETCHING,
            ExecutionState.STAGED,
            ExecutionState.VALIDATED,
        } and not lock_present:
            disagreements.append("journal_lock_state_lacks_lock_artifact")

    dispositions: list[RecoveryDisposition] = []
    if disagreements or journal is None:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        if target.safe_directory and not target_complete:
            _append_unique(
                dispositions, RecoveryDisposition.REQUIRES_REVALIDATION
            )
        elif stage_complete:
            _append_unique(
                dispositions, RecoveryDisposition.REQUIRES_REVALIDATION
            )
        summary = "Durable evidence is malformed, unsafe, or disagrees; operator review required."
    elif target_complete and journal is not None:
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        pending_present = bool(
            target.artifacts[PENDING_MANIFEST].kind != "absent"
        )
        durability_status = journal.durability.get("status")
        durability_uncertain = durability_status in {
            "unknown",
            "uncertain",
            "failed",
            "indeterminate",
        } or journal.failure.get("reason") == "published_durability_uncertain"
        reconciliation_complete = bool(
            journal.state == ExecutionState.COMPLETE
            or journal.reconciliation.get("completed") is True
        )
        if (
            reconciliation_complete
            and not stage_has_content
            and not lock_present
            and not pending_present
        ):
            summary = "Committed science and completed reconciliation agree; published science is preserved."
        else:
            _append_unique(
                dispositions, RecoveryDisposition.REQUIRES_RECONCILIATION
            )
            if stage_has_content or lock_present or pending_present or durability_uncertain:
                _append_unique(
                    dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION
                )
            summary = "A committed partition is visible and must be preserved and reconciled."
    elif target_reserved:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        summary = (
            "A reserved or partial target exists without a complete commit marker; "
            "revalidation and operator review are required."
        )
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_REVALIDATION)
    elif journal is not None and journal.published:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        summary = "The journal records possible publication but no complete target is provable."
    elif stage_complete and journal is not None:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_REVALIDATION)
        if lock_present:
            _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
            _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        summary = "A complete unpublished stage exists and must be revalidated before reuse."
    elif stage_has_content and journal is not None:
        attempted = journal.publication.get("attempted", False)
        if (
            journal.state == ExecutionState.FAILED
            and attempted is False
            and not lock_present
        ):
            _append_unique(dispositions, RecoveryDisposition.SAFE_TO_DISCARD)
            summary = "Only journal-matched partial staging remains before publication."
        else:
            _append_unique(
                dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION
            )
            _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
            summary = "Partial staging cannot be safely classified for automatic action."
    elif lock_present:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        summary = "A lock artifact survives and must not be stolen automatically."
    elif journal is not None and journal.state in {
        ExecutionState.PLANNED,
        ExecutionState.PRECHECKED,
    }:
        _append_unique(dispositions, RecoveryDisposition.SAFE_TO_DISCARD)
        summary = (
            "Only an unpublished operational journal exists; this release does not "
            "auto-resume it."
        )
    elif (
        journal is not None
        and journal.state == ExecutionState.FAILED
        and journal.publication.get("attempted", False) is False
    ):
        _append_unique(dispositions, RecoveryDisposition.SAFE_TO_DISCARD)
        summary = "The failed pre-publication operation has no remaining artifacts."
    else:
        _append_unique(dispositions, RecoveryDisposition.REQUIRES_OPERATOR_DECISION)
        _append_unique(dispositions, RecoveryDisposition.MUST_NOT_AUTO_DELETE)
        summary = "Durable evidence is insufficient for an automatic recovery decision."

    return RecoveryAssessment(
        journal_path=str(journal_path),
        journal=journal,
        target=target,
        stage=stage,
        lock=lock,
        dispositions=tuple(dispositions),
        summary=summary,
        disagreements=tuple(sorted(set(disagreements))),
        lock_owner_status=lock_owner_status,
    )


class RecoveryInspector:
    """Small integration wrapper around the read-only inspection function."""

    @staticmethod
    def inspect(
        journal_path: Path,
        *,
        target_path: Path,
        stage_path: Path,
        lock_path: Path,
    ) -> RecoveryAssessment:
        return inspect_recovery(
            journal_path,
            target_path=target_path,
            stage_path=stage_path,
            lock_path=lock_path,
        )
