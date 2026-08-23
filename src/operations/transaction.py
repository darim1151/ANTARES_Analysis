"""Guarded transactional publication contract for temporary fixtures only."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Tuple

from .locking import WriterLock
from .state import ExecutionState, ExecutionStateMachine, StateSnapshot
from .storage import DevelopmentWriteCapability, OPERATIONS_DIRECTORY, contained_path


class TransactionError(RuntimeError):
    pass


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
        stage_identity = writer_lock.path.name.removesuffix(".lock")
        self.stage = contained_path(
            capability.root,
            Path(OPERATIONS_DIRECTORY) / "staging" / run_id / stage_identity,
        )
        self.state_machine = ExecutionStateMachine()
        self.evidence: Optional[QueryFetchEvidence] = None

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
        if not self.writer_lock.held:
            self._fail("writer_lock_lost")
            raise TransactionError("Writer lock is no longer held.")
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
        self.stage.mkdir(parents=True)
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
                (self.stage / name).write_bytes(artifacts[name])
            # Manifest is deliberately staged last.
            (self.stage / "manifest.json").write_bytes(artifacts["manifest.json"])
        except Exception:
            self._fail("staging_write_failed")
            raise
        return self.state_machine.transition(ExecutionState.STAGED)

    def validate(self, validator: StagedValidator) -> StateSnapshot:
        if self.state != ExecutionState.STAGED or self.evidence is None:
            raise TransactionError("Only a complete staged transaction can be validated.")
        for name in self.REQUIRED_ARTIFACTS:
            path = self.stage / name
            if path.is_symlink() or not path.is_file():
                self._fail("staged_artifact_missing_or_unsafe")
                raise TransactionError(f"Missing or unsafe staged artifact: {name}.")
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
        return self.state_machine.transition(ExecutionState.VALIDATED)

    def publish(self) -> StateSnapshot:
        if self.state != ExecutionState.VALIDATED:
            raise TransactionError("Publication requires a validated stage.")
        if not self.writer_lock.held:
            self._fail("writer_lock_lost")
            raise TransactionError("Writer lock was lost before publication.")
        if self.target.exists() or self.target.is_symlink():
            self._fail("published_target_already_exists")
            raise TransactionError("A published target already exists; overwrite refused.")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(self.stage).st_dev != os.stat(self.target.parent).st_dev:
            self._fail("staging_filesystem_mismatch")
            raise TransactionError(
                "Atomic publication requires same-filesystem staging."
            )
        try:
            os.rename(self.stage, self.target)
        except Exception:
            self._fail("atomic_publication_failed")
            raise
        return self.state_machine.transition(ExecutionState.PUBLISHED)

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
        if self.stage.exists() and not self.stage.is_symlink():
            shutil.rmtree(self.stage)
        if self.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
            self.state_machine.fail(reason)
        return self.state_machine.snapshot()
