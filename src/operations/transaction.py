"""Guarded transactional publication contract for temporary fixtures only."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from .locking import LockOwnershipError, WriterLock
from .state import ExecutionState, ExecutionStateMachine, StateSnapshot
from .storage import DevelopmentWriteCapability, OPERATIONS_DIRECTORY, contained_path


class TransactionError(RuntimeError):
    pass


class _PublicationCommittedError(TransactionError):
    """The manifest committed or its remote outcome cannot be proven absent."""


@dataclass(frozen=True)
class _ArtifactSnapshot:
    """Identity and content proven at the validation boundary."""

    device: int
    inode: int
    size: int
    mode: int
    sha256: str


def _directory_identity(path: Path) -> Tuple[int, int]:
    if path.is_symlink() or not path.is_dir():
        raise TransactionError(f"Expected a real directory: {path}.")
    observed = path.stat()
    return observed.st_dev, observed.st_ino


def _artifact_snapshot(path: Path) -> _ArtifactSnapshot:
    """Hash one regular file through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    descriptor = os.open(str(path), flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise TransactionError(f"Expected a regular staged artifact: {path}.")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return _ArtifactSnapshot(
            device=observed.st_dev,
            inode=observed.st_ino,
            size=observed.st_size,
            mode=stat.S_IMODE(observed.st_mode),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _assert_artifact_snapshot(path: Path, expected: _ArtifactSnapshot) -> None:
    if _artifact_snapshot(path) != expected:
        raise TransactionError(f"Validated staged artifact changed: {path.name}.")


def _artifact_snapshot_at(directory_fd: int, name: str) -> _ArtifactSnapshot:
    """Hash one no-follow regular file relative to a pinned directory."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
        os, "O_CLOEXEC", 0
    )
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise TransactionError(f"Expected a regular artifact: {name}.")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        return _ArtifactSnapshot(
            device=observed.st_dev,
            inode=observed.st_ino,
            size=observed.st_size,
            mode=stat.S_IMODE(observed.st_mode),
            sha256=digest.hexdigest(),
        )
    finally:
        os.close(descriptor)


def _assert_artifact_snapshot_at(
    directory_fd: int,
    name: str,
    expected: _ArtifactSnapshot,
) -> None:
    if _artifact_snapshot_at(directory_fd, name) != expected:
        raise TransactionError(f"Validated artifact changed: {name}.")


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes using a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, payload: bytes) -> _ArtifactSnapshot:
    """Create one new staged artifact and persist its exact payload."""
    if not isinstance(payload, bytes):
        raise TypeError("Staged artifact payloads must be bytes.")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
        os, "O_NOFOLLOW", 0
    ) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(str(path), flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(f"Short staged-artifact write: {path}.")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return _artifact_snapshot(path)


def _ensure_directory_tree_fsynced(path: Path, root: Path) -> None:
    """Create a contained directory chain and persist every new entry."""
    path = path.resolve(strict=False)
    root = root.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TransactionError("Directory creation escaped the capability root.") from exc

    missing = []
    cursor = path
    while cursor != root and not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise TransactionError("Directory ancestor is missing, unsafe, or not a directory.")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
        descriptor = os.open(
            str(directory),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(directory.parent)
    if path.is_symlink() or not path.is_dir():
        raise TransactionError("Expected a real contained directory.")


def _assert_link_pair(
    source: Path,
    destination: Path,
    expected: _ArtifactSnapshot,
) -> None:
    _assert_artifact_snapshot(source, expected)
    _assert_artifact_snapshot(destination, expected)
    if not os.path.samefile(source, destination):
        raise TransactionError(
            f"Published artifact is not the validated staged inode: {source.name}."
        )


def _assert_link_pair_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    expected: _ArtifactSnapshot,
) -> None:
    source = _artifact_snapshot_at(source_fd, source_name)
    destination = _artifact_snapshot_at(destination_fd, destination_name)
    if source != expected or destination != expected:
        raise TransactionError(
            f"Published artifact does not match validation: {destination_name}."
        )


def _link_validated_artifact_at(
    source_fd: int,
    source_name: str,
    destination_fd: int,
    destination_name: str,
    expected: _ArtifactSnapshot,
) -> None:
    _assert_artifact_snapshot_at(source_fd, source_name, expected)
    os.link(
        source_name,
        destination_name,
        src_dir_fd=source_fd,
        dst_dir_fd=destination_fd,
        follow_symlinks=False,
    )
    _assert_link_pair_at(
        source_fd,
        source_name,
        destination_fd,
        destination_name,
        expected,
    )


def _directory_fd(path: Path) -> int:
    return os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )


def _fd_identity(descriptor: int) -> Tuple[int, int]:
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        raise TransactionError("Pinned publication descriptor is not a directory.")
    return observed.st_dev, observed.st_ino


def _close_descriptor_quietly(descriptor: int) -> None:
    """Close a read-only directory fd without changing commit classification."""
    try:
        os.close(descriptor)
    except OSError:
        pass


def _assert_target_entry(
    parent_fd: int,
    target_name: str,
    target_identity: Tuple[int, int],
) -> None:
    observed = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != target_identity
    ):
        raise TransactionError("Reserved publication target identity changed.")


def _manifest_link_committed_at(
    target_fd: int,
    expected: _ArtifactSnapshot,
) -> bool:
    try:
        committed = _artifact_snapshot_at(target_fd, "manifest.json")
    except (OSError, TransactionError):
        return False
    return committed == expected


_INDETERMINATE_LINK_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR,
        errno.EIO,
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETUNREACH", None),
    )
    if value is not None
)


def _link_outcome_indeterminate(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno in _INDETERMINATE_LINK_ERRNOS


def _publish_noreplace(
    stage: Path,
    target: Path,
    validated: Mapping[str, _ArtifactSnapshot],
) -> None:
    """Reserve a target atomically and publish hard-linked artifacts manifest-last.

    Arnor's NFSv4.2 mount rejects ``renameat2(RENAME_NOREPLACE)``.  An exclusive
    directory reservation prevents target replacement, while same-filesystem
    hard links preserve the exact validated staged inodes during commit.  The
    manifest link is the logical commit marker and is created only after both
    data links persist.
    """
    stage_fd = _directory_fd(stage)
    try:
        parent_fd = _directory_fd(target.parent)
    except Exception:
        _close_descriptor_quietly(stage_fd)
        raise
    target_fd: Optional[int] = None
    try:
        try:
            os.mkdir(target.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise TransactionError(
                "A published or reserved target already exists; overwrite refused."
            ) from exc
        target_fd = os.open(
            target.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        target_identity = _fd_identity(target_fd)
        try:
            os.fchmod(target_fd, 0o700)
            os.fsync(target_fd)
            os.fsync(parent_fd)
            _assert_target_entry(parent_fd, target.name, target_identity)
            for name in ("loci.parquet", "alerts.parquet"):
                _link_validated_artifact_at(
                    stage_fd,
                    name,
                    target_fd,
                    name,
                    validated[name],
                )
            os.fsync(target_fd)
            for name in ("loci.parquet", "alerts.parquet"):
                _assert_link_pair_at(
                    stage_fd, name, target_fd, name, validated[name]
                )
            _link_validated_artifact_at(
                stage_fd,
                "manifest.json",
                target_fd,
                ".manifest.pending",
                validated["manifest.json"],
            )
            os.fsync(target_fd)
            _assert_target_entry(parent_fd, target.name, target_identity)
            for name in ("loci.parquet", "alerts.parquet"):
                _assert_link_pair_at(
                    stage_fd, name, target_fd, name, validated[name]
                )
            _assert_link_pair_at(
                stage_fd,
                "manifest.json",
                target_fd,
                ".manifest.pending",
                validated["manifest.json"],
            )
        except Exception as exc:
            raise TransactionError(
                "Manifest-last publication failed after target reservation; "
                "manual recovery classification is required."
            ) from exc

        try:
            os.link(
                ".manifest.pending",
                "manifest.json",
                src_dir_fd=target_fd,
                dst_dir_fd=target_fd,
                follow_symlinks=False,
            )
        except Exception as exc:
            if _manifest_link_committed_at(
                target_fd, validated["manifest.json"]
            ) or _link_outcome_indeterminate(exc):
                raise _PublicationCommittedError(
                    "Manifest commit is visible or its remote outcome is ambiguous."
                ) from exc
            raise TransactionError(
                "Manifest-last publication failed after target reservation; "
                "manual recovery classification is required."
            ) from exc

        try:
            _assert_link_pair_at(
                target_fd,
                ".manifest.pending",
                target_fd,
                "manifest.json",
                validated["manifest.json"],
            )
            for name in ("loci.parquet", "alerts.parquet"):
                _assert_link_pair_at(
                    stage_fd, name, target_fd, name, validated[name]
                )
            _assert_target_entry(parent_fd, target.name, target_identity)
            os.fsync(target_fd)
            os.fsync(parent_fd)
            os.unlink(".manifest.pending", dir_fd=target_fd)
            os.fsync(target_fd)
        except Exception as exc:
            raise _PublicationCommittedError(
                "Manifest commit is visible but post-commit durability is uncertain."
            ) from exc
    finally:
        if target_fd is not None:
            _close_descriptor_quietly(target_fd)
        _close_descriptor_quietly(parent_fd)
        _close_descriptor_quietly(stage_fd)


def _cleanup_committed_stage(
    stage: Path,
    target: Path,
    validated: Mapping[str, _ArtifactSnapshot],
    stage_identity: Tuple[int, int],
) -> None:
    """Remove only the three proven source links after a committed publication."""
    required = ("loci.parquet", "alerts.parquet", "manifest.json")
    if (
        stage.is_symlink()
        or not stage.is_dir()
        or _directory_identity(stage) != stage_identity
    ):
        raise TransactionError("Committed stage is missing or unsafe.")
    unexpected = sorted(
        path.name for path in stage.iterdir() if path.name not in required
    )
    if unexpected:
        raise TransactionError(
            f"Committed stage contains unexpected artifacts: {unexpected}."
        )
    for name in required:
        source = stage / name
        destination = target / name
        _assert_link_pair(source, destination, validated[name])
    for name in required:
        (stage / name).unlink()
    _fsync_directory(stage)
    stage.rmdir()
    _fsync_directory(stage.parent)


@dataclass(frozen=True)
class QueryFetchEvidence:
    """Evidence that prevents failures from becoming valid empty science."""

    query_completed: bool
    fetch_completed: bool
    loci_rows: int
    alert_rows: int
    query_errors: Tuple[str, ...] = ()
    fetch_errors: Tuple[str, ...] = ()
    zero_row_proof: Optional[str] = None

    @property
    def clean(self) -> bool:
        return bool(
            self.query_completed
            and self.fetch_completed
            and not self.query_errors
            and not self.fetch_errors
            and self.loci_rows >= 0
            and self.alert_rows >= 0
        )

    @property
    def valid_zero_row(self) -> bool:
        return bool(
            self.clean
            and self.loci_rows == 0
            and self.alert_rows == 0
            and self.zero_row_proof == "completed_successful_query"
        )


class StagedValidator(Protocol):
    def __call__(self, stage_directory: Path) -> bool:
        ...


class PublicationTransaction:
    """Local proof of the future prepare/stage/validate/publish contract.

    Construction requires a capability that cannot target production paths and
    a held lock owned by the same run. No public constructor can enable Arnor.
    """

    REQUIRED_ARTIFACTS = ("loci.parquet", "alerts.parquet", "manifest.json")

    def __init__(
        self,
        capability: DevelopmentWriteCapability,
        writer_lock: WriterLock,
        target_relative: Path,
        run_id: str,
    ) -> None:
        if not writer_lock.held or writer_lock.run_id != run_id:
            raise TransactionError("A held writer lock owned by this run is required.")
        if writer_lock.capability != capability:
            raise TransactionError("Writer lock and transaction capabilities differ.")
        self.capability = capability
        self.writer_lock = writer_lock
        self.run_id = run_id
        self.target_relative = Path(target_relative)
        self.target = contained_path(capability.root, self.target_relative)
        if writer_lock.target_identity != self.target_relative.as_posix():
            raise TransactionError(
                "Writer lock identity does not match the publication target."
            )
        stage_identity = writer_lock.path.name.removesuffix(".lock")
        self.stage = contained_path(
            capability.root,
            Path(OPERATIONS_DIRECTORY) / "staging" / run_id / stage_identity,
        )
        self.state_machine = ExecutionStateMachine()
        self.evidence: Optional[QueryFetchEvidence] = None
        self._publication_attempted = False
        self._owned_stage_identity: Optional[Tuple[int, int]] = None
        self._owned_stage_artifacts: Dict[str, _ArtifactSnapshot] = {}
        self._validated_artifacts: Optional[Dict[str, _ArtifactSnapshot]] = None

    @property
    def state(self) -> ExecutionState:
        return self.state_machine.state

    def _fail(self, reason: str) -> None:
        if self.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
            self.state_machine.fail(reason)

    def prepare(self) -> StateSnapshot:
        if self.target.exists() or self.target.is_symlink():
            self._fail("published_target_already_exists")
            raise TransactionError("A published target already exists; overwrite refused.")
        self.state_machine.transition(ExecutionState.PRECHECKED)
        try:
            self.writer_lock.assert_owned()
        except LockOwnershipError as exc:
            self._fail("writer_lock_lost")
            raise TransactionError("Writer lock is no longer owned by this run.") from exc
        self.state_machine.transition(ExecutionState.LOCKED)
        ancestor = self.target.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if os.stat(ancestor).st_dev != os.stat(self.capability.root).st_dev:
            self._fail("staging_filesystem_mismatch")
            raise TransactionError(
                "Staging and publication targets must use the same filesystem."
            )
        if self.stage.exists() or self.stage.is_symlink():
            self._fail("staging_target_already_exists")
            raise TransactionError("Staging target already exists.")
        _ensure_directory_tree_fsynced(self.stage.parent, self.capability.root)
        try:
            self.stage.mkdir(mode=0o700)
        except FileExistsError as exc:
            self._fail("staging_target_already_exists")
            raise TransactionError("Staging target already exists.") from exc
        self._owned_stage_identity = _directory_identity(self.stage)
        descriptor = os.open(
            str(self.stage),
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.stage.parent)
        if os.stat(self.stage).st_dev != os.stat(self.capability.root).st_dev:
            self._fail("staging_filesystem_mismatch")
            raise TransactionError(
                "Staging and publication targets must use the same filesystem."
            )
        return self.state_machine.snapshot()

    def begin_query(self) -> StateSnapshot:
        return self.state_machine.transition(ExecutionState.QUERYING)

    def begin_fetch(self) -> StateSnapshot:
        return self.state_machine.transition(ExecutionState.FETCHING)

    def stage_artifacts(
        self,
        artifacts: Mapping[str, bytes],
        evidence: QueryFetchEvidence,
    ) -> StateSnapshot:
        if self.state != ExecutionState.FETCHING:
            raise TransactionError("Artifacts may only be staged after fetching.")
        if set(artifacts) != set(self.REQUIRED_ARTIFACTS):
            self._fail("staged_artifact_set_invalid")
            raise TransactionError("A complete staged nightly artifact set is required.")
        if not evidence.clean:
            self._fail("query_or_fetch_failed")
            raise TransactionError("Query/fetch failures cannot be staged as science.")
        if evidence.loci_rows == 0 and evidence.alert_rows == 0 and not evidence.valid_zero_row:
            self._fail("zero_row_evidence_invalid")
            raise TransactionError("Zero-row publication requires successful query evidence.")
        self.evidence = evidence
        try:
            for name in ("loci.parquet", "alerts.parquet"):
                self._owned_stage_artifacts[name] = _write_fsynced(
                    self.stage / name, artifacts[name]
                )
            # Manifest is deliberately staged last.
            self._owned_stage_artifacts["manifest.json"] = _write_fsynced(
                self.stage / "manifest.json", artifacts["manifest.json"]
            )
            _fsync_directory(self.stage)
        except Exception:
            self._fail("staging_write_failed")
            raise
        return self.state_machine.transition(ExecutionState.STAGED)

    def validate(self, validator: StagedValidator) -> StateSnapshot:
        if self.state != ExecutionState.STAGED or self.evidence is None:
            raise TransactionError("Only a complete staged transaction can be validated.")
        before: Dict[str, _ArtifactSnapshot] = {}
        for name in self.REQUIRED_ARTIFACTS:
            path = self.stage / name
            if path.is_symlink() or not path.is_file():
                self._fail("staged_artifact_missing_or_unsafe")
                raise TransactionError(f"Missing or unsafe staged artifact: {name}.")
            try:
                before[name] = _artifact_snapshot(path)
            except (OSError, TransactionError) as exc:
                self._fail("staged_artifact_missing_or_unsafe")
                raise TransactionError(
                    f"Missing or unsafe staged artifact: {name}."
                ) from exc
        try:
            manifest = json.loads((self.stage / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self._fail("staged_manifest_invalid")
            raise TransactionError("Staged manifest is invalid JSON.") from exc
        if not isinstance(manifest, dict):
            self._fail("staged_manifest_invalid")
            raise TransactionError("Staged manifest must be a JSON object.")
        if (
            manifest.get("actual_loci") != self.evidence.loci_rows
            or manifest.get("alert_rows") != self.evidence.alert_rows
        ):
            self._fail("staged_manifest_count_mismatch")
            raise TransactionError("Staged manifest counts do not match query evidence.")
        if self.evidence.valid_zero_row:
            validation = manifest.get("validation")
            validation = validation if isinstance(validation, dict) else {}
            if not (
                validation.get("query_completed_pass") is True
                and validation.get("query_fetch_clean") is True
                and validation.get("zero_row_schema_pass") is True
                and validation.get("append_ready") is True
            ):
                self._fail("zero_row_manifest_evidence_invalid")
                raise TransactionError("Zero-row manifest does not preserve success evidence.")
        try:
            passed = bool(validator(self.stage))
        except Exception:
            self._fail("staged_validation_error")
            raise
        if not passed:
            self._fail("staged_validation_failed")
            raise TransactionError("Staged validation failed; publication refused.")
        try:
            after = {
                name: _artifact_snapshot(self.stage / name)
                for name in self.REQUIRED_ARTIFACTS
            }
        except (OSError, TransactionError) as exc:
            self._fail("staged_artifact_changed_during_validation")
            raise TransactionError(
                "Staged artifacts changed during validation; publication refused."
            ) from exc
        if after != before:
            self._fail("staged_artifact_changed_during_validation")
            raise TransactionError(
                "Staged artifacts changed during validation; publication refused."
            )
        self._validated_artifacts = before
        return self.state_machine.transition(ExecutionState.VALIDATED)

    def publish(self) -> StateSnapshot:
        if self.state != ExecutionState.VALIDATED:
            raise TransactionError("Publication requires a validated stage.")
        try:
            self.writer_lock.assert_owned()
        except LockOwnershipError as exc:
            self._fail("writer_lock_lost")
            raise TransactionError("Writer lock was lost before publication.") from exc
        if self._validated_artifacts is None:
            self._fail("validated_artifact_snapshot_missing")
            raise TransactionError("Validated artifact identity is missing.")
        if (
            self._owned_stage_identity is None
            or _directory_identity(self.stage) != self._owned_stage_identity
        ):
            self._fail("staging_identity_changed")
            raise TransactionError("Staging directory identity changed.")
        try:
            for name, expected in self._validated_artifacts.items():
                _assert_artifact_snapshot(self.stage / name, expected)
        except (OSError, TransactionError) as exc:
            self._fail("validated_artifact_changed_before_publication")
            raise TransactionError(
                "A validated staged artifact changed before publication."
            ) from exc
        if self.target.exists() or self.target.is_symlink():
            self._fail("published_target_already_exists")
            raise TransactionError("A published target already exists; overwrite refused.")
        _ensure_directory_tree_fsynced(self.target.parent, self.capability.root)
        if os.stat(self.stage).st_dev != os.stat(self.target.parent).st_dev:
            self._fail("staging_filesystem_mismatch")
            raise TransactionError(
                "Atomic publication requires same-filesystem staging."
            )
        try:
            self.writer_lock.assert_owned()
        except LockOwnershipError as exc:
            self._fail("writer_lock_lost")
            raise TransactionError("Writer lock was lost before publication.") from exc
        self._publication_attempted = True
        try:
            _publish_noreplace(
                self.stage, self.target, self._validated_artifacts
            )
        except _PublicationCommittedError as exc:
            self.state_machine.transition(ExecutionState.PUBLISHED)
            self._fail("published_durability_uncertain")
            raise TransactionError(
                "Publication may be visible and requires operator recovery."
            ) from exc
        except Exception:
            self._fail("publication_before_manifest_failed")
            raise
        published = self.state_machine.transition(ExecutionState.PUBLISHED)
        try:
            _cleanup_committed_stage(
                self.stage,
                self.target,
                self._validated_artifacts,
                self._owned_stage_identity,
            )
        except Exception as exc:
            self._fail("published_stage_cleanup_failed")
            raise TransactionError(
                "Publication committed but staged-link cleanup requires recovery."
            ) from exc
        self._owned_stage_identity = None
        self._owned_stage_artifacts.clear()
        return published

    def release_writer_lock(self) -> None:
        if self.state != ExecutionState.PUBLISHED:
            raise TransactionError(
                "Writer lock release is ordered immediately after publication."
            )
        self.writer_lock.release()

    def begin_reconciliation(self) -> StateSnapshot:
        if self.writer_lock.held:
            raise TransactionError(
                "Release the nightly writer lock before derived reconciliation."
            )
        return self.state_machine.transition(ExecutionState.RECONCILING)

    def complete_reconciliation(self) -> StateSnapshot:
        return self.state_machine.transition(ExecutionState.COMPLETE)

    def fail_reconciliation(self, reason: str) -> StateSnapshot:
        if self.state not in {ExecutionState.PUBLISHED, ExecutionState.RECONCILING}:
            raise TransactionError("Reconciliation failure requires published science.")
        return self.state_machine.fail(reason)

    def abort(self, reason: str = "transaction_aborted") -> StateSnapshot:
        if self.state_machine.published:
            if self.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
                self.state_machine.fail(reason)
            return self.state_machine.snapshot()
        if not self._publication_attempted and self._owned_stage_identity is not None:
            try:
                owned_stage = (
                    not self.stage.is_symlink()
                    and self.stage.is_dir()
                    and _directory_identity(self.stage) == self._owned_stage_identity
                )
                entries = {path.name: path for path in self.stage.iterdir()} if owned_stage else {}
                owned_contents = owned_stage and set(entries) == set(
                    self._owned_stage_artifacts
                )
                if owned_contents:
                    for name, expected in self._owned_stage_artifacts.items():
                        _assert_artifact_snapshot(entries[name], expected)
                    for name in self._owned_stage_artifacts:
                        entries[name].unlink()
                    _fsync_directory(self.stage)
                    self.stage.rmdir()
                    _fsync_directory(self.stage.parent)
                    self._owned_stage_identity = None
                    self._owned_stage_artifacts.clear()
            except (OSError, TransactionError):
                # Ambiguous state is preserved for explicit recovery classification.
                pass
        if self.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
            self.state_machine.fail(reason)
        return self.state_machine.snapshot()
