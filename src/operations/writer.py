"""Single transactional nightly-writer orchestration used by synthetic Phase 5.

The orchestration is production-shaped, while issuance is deliberately not:
this release accepts only a sealed :class:`SyntheticWriteCapability` and the
deterministic synthetic provider.  There is no production capability factory
or live-provider adapter to configure around that boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from .journal import (
    ArtifactIdentity,
    JournalError,
    JournalOutcome,
    TransactionDescriptor,
    TransactionJournal,
)
from .locking import LockUnavailable, WriterLock, lock_identity
from .report import Artifact, ExitCode, Issue, OperationReport
from .state import ExecutionState
from .storage import (
    SyntheticWriteCapability,
    contained_path,
)
from .transaction import PublicationTransaction, TransactionError


WRITER_CONTRACT_VERSION = "phase5-writer-v1"
EXPECTED_ARTIFACTS = ("loci.parquet", "alerts.parquet", "manifest.json")


class WriterError(RuntimeError):
    pass


class ProductionAuthorizationUnavailable(WriterError):
    """The absent production capability/provider boundary was requested."""


class ExistingValidTarget(WriterError):
    """An independently valid partition already occupies the no-clobber target."""


class InjectedWriterFailure(WriterError):
    def __init__(self, point: str) -> None:
        self.point = point
        super().__init__(f"Injected writer failure at {point}.")


class FailureInjector:
    """Deterministic one-shot exception injection for boundary qualification."""

    def __init__(self, *points: str) -> None:
        self._points = set(points)
        self.observed: list[str] = []

    def __call__(self, point: str, details: Mapping[str, Any]) -> None:
        del details
        self.observed.append(point)
        if point in self._points:
            self._points.remove(point)
            raise InjectedWriterFailure(point)


FaultHook = Callable[[str, Mapping[str, Any]], None]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise WriterError("Writer clocks must be timezone-aware.")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def _ensure_private_tree(path: Path, root: Path) -> None:
    """Create one capability-contained private tree with durable entries."""
    canonical_root = Path(root).resolve(strict=True)
    candidate = Path(path).resolve(strict=False)
    try:
        relative = candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise WriterError("Operational directory escaped the capability root.") from exc
    cursor = canonical_root
    for component in relative.parts:
        cursor = cursor / component
        try:
            cursor.mkdir(mode=0o700)
        except FileExistsError:
            if cursor.is_symlink() or not cursor.is_dir():
                raise WriterError("Operational directory is missing or unsafe.")
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
                raise WriterError("Operational path is not a directory.")
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(cursor.parent)


def _safe_component(value: str, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or len(Path(text).parts) != 1
    ):
        raise WriterError(f"{label} must be one safe path component.")
    return text


def nightly_target_relative(date_utc: str) -> Path:
    try:
        parsed = datetime.strptime(date_utc, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise WriterError("Night target must use canonical YYYY-MM-DD.") from exc
    if parsed.strftime("%Y-%m-%d") != date_utc:
        raise WriterError("Night target must use canonical YYYY-MM-DD.")
    return (
        Path("data")
        / "lsst_only"
        / "nightly"
        / parsed.strftime("%Y")
        / parsed.strftime("%m")
        / parsed.strftime("%d")
    )


@dataclass(frozen=True)
class NightExecutionSpec:
    transaction_id: str
    plan_id: str
    release_sha: str
    configuration_identity: str
    science_request: Any

    def __post_init__(self) -> None:
        _safe_component(self.transaction_id, "transaction_id")
        for label in ("plan_id", "release_sha", "configuration_identity"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise WriterError(f"{label} must be non-empty.")


@dataclass(frozen=True)
class ReconciliationResult:
    state: str
    path: str
    sha256: str
    idempotent_replay: bool


class SyntheticReconciler:
    """Idempotent synthetic derived promotion under its own writer lock."""

    def reconcile(
        self,
        capability: SyntheticWriteCapability,
        *,
        date_utc: str,
        published_artifacts: Mapping[str, ArtifactIdentity],
        transaction_id: str,
    ) -> ReconciliationResult:
        final_relative = Path("derived") / "nightly" / f"{date_utc}.json"
        final_path = contained_path(capability.published_root, final_relative)
        payload = _canonical_json(
            {
                "schema_version": "phase5.synthetic-reconciliation.v1",
                "date_utc": date_utc,
                "science_sha256": {
                    name: published_artifacts[name].sha256
                    for name in sorted(published_artifacts)
                },
            }
        ) + b"\n"
        expected_hash = hashlib.sha256(payload).hexdigest()
        if final_path.exists() or final_path.is_symlink():
            if final_path.is_symlink() or not final_path.is_file():
                raise WriterError("Existing derived target is unsafe.")
            observed = final_path.read_bytes()
            if observed != payload:
                raise WriterError("Existing derived target conflicts with science.")
            return ReconciliationResult(
                "complete", str(final_path), expected_hash, True
            )

        _ensure_private_tree(final_path.parent, capability.root)
        stage_root = contained_path(
            capability.staging_root,
            Path(transaction_id) / "reconciliation",
        )
        _ensure_private_tree(stage_root, capability.root)
        stage = stage_root / "nightly-summary.json"
        descriptor = os.open(
            str(stage),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("Short reconciliation write.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(stage_root)
        if hashlib.sha256(stage.read_bytes()).hexdigest() != expected_hash:
            raise WriterError("Derived staging validation failed.")
        try:
            os.link(stage, final_path, follow_symlinks=False)
        except FileExistsError:
            if final_path.is_symlink() or not final_path.is_file():
                raise WriterError("Late derived target is unsafe.")
            if final_path.read_bytes() != payload:
                raise WriterError("Late derived target conflicts with science.")
            idempotent = True
        else:
            idempotent = False
            _fsync_directory(final_path.parent)
        if final_path.read_bytes() != payload:
            raise WriterError("Promoted derived artifact failed reopening.")
        stage.unlink()
        _fsync_directory(stage_root)
        stage_root.rmdir()
        _fsync_directory(stage_root.parent)
        return ReconciliationResult(
            "complete", str(final_path), expected_hash, idempotent
        )


def _read_nofollow(path: Path) -> bytes:
    descriptor = os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise WriterError(f"Published artifact is not regular: {path.name}.")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def independent_reopen(
    target: Path,
    expected_science: Any,
) -> Mapping[str, ArtifactIdentity]:
    """Freshly reopen canonical paths and rerun the science validator."""
    from .science import reopen_and_validate_artifacts

    if target.is_symlink() or not target.is_dir():
        raise WriterError("Published target is missing or unsafe.")
    entries = {path.name for path in target.iterdir()}
    if entries != set(EXPECTED_ARTIFACTS):
        raise WriterError("Published target artifact set is not exact.")
    payloads = {
        name: _read_nofollow(target / name) for name in EXPECTED_ARTIFACTS
    }
    reopen_and_validate_artifacts(payloads, expected=expected_science)
    return {
        name: ArtifactIdentity.from_path(name, target / name)
        for name in EXPECTED_ARTIFACTS
    }


def production_ingest_refusal(
    date_utc: str,
    *,
    now: Callable[[], datetime] = _utc_now,
) -> OperationReport:
    """Fail before provider construction because production authority is absent."""
    started = now()
    nightly_target_relative(date_utc)
    finished = now()
    return OperationReport(
        operation="night.ingest",
        success=False,
        status="refused",
        started_at_utc=_iso(started),
        finished_at_utc=_iso(finished),
        elapsed_seconds=max(0.0, (finished - started).total_seconds()),
        exit_code=ExitCode.REFUSED,
        refusal_reasons=(
            Issue(
                "production_authorization_unavailable",
                "This release contains no production publication capability or live provider adapter.",
            ),
        ),
        next_actions=("Use `night plan`; production execution requires a later release.",),
        details={
            "target_night": date_utc,
            "provider_constructed": False,
            "writer_capability_issued": False,
            "configurable_bypass": False,
        },
        read_only=True,
    )


class TransactionalNightWriter:
    """Production-shaped coordinator reachable only with synthetic authority."""

    def __init__(
        self,
        capability: SyntheticWriteCapability,
        provider: Any,
        *,
        reconciler: Optional[SyntheticReconciler] = None,
        fault_hook: Optional[FaultHook] = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        from .science import SyntheticScienceProvider

        if type(capability) is not SyntheticWriteCapability:
            raise ProductionAuthorizationUnavailable(
                "Transactional execution requires a sealed synthetic capability."
            )
        if type(provider) is not SyntheticScienceProvider:
            raise ProductionAuthorizationUnavailable(
                "This release has no authorized live science-provider adapter."
            )
        self.capability = capability
        self.provider = provider
        self.reconciler = reconciler or SyntheticReconciler()
        self.fault_hook = fault_hook
        self.clock = clock

    def _checkpoint(self, point: str, **details: Any) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point, details)

    def _publication_event(
        self, event: str, details: Mapping[str, object]
    ) -> None:
        self._checkpoint(event, **dict(details))

    def _preflight(
        self,
        spec: NightExecutionSpec,
        target: Path,
        journal_path: Path,
        plan_hash: str,
        config_hash: str,
    ) -> Mapping[str, Any]:
        from src.history import mjd_to_utc_date
        from .science import NightScienceRequest

        if not isinstance(spec.science_request, NightScienceRequest):
            raise WriterError("Writer requires a NightScienceRequest.")
        if spec.science_request.lsst_only is not True:
            raise WriterError("Writer preflight requires LSST-only science.")
        if mjd_to_utc_date(spec.science_request.mjd_min) != (
            spec.science_request.date_utc
        ):
            raise WriterError("MJD lower bound and target UTC night disagree.")
        root = self.capability.root
        if root.is_symlink() or not root.is_dir():
            raise WriterError("Synthetic run root is missing or unsafe.")
        if root.stat().st_uid != os.geteuid():
            raise WriterError("Synthetic run root is not owned by the effective user.")
        for label, path in (
            ("published", self.capability.published_root),
            ("staging", self.capability.staging_root),
            ("locks", self.capability.lock_root),
            ("journals", self.capability.journal_root),
            ("evidence", self.capability.evidence_root),
        ):
            resolved = path.resolve(strict=False)
            try:
                resolved.relative_to(root.resolve(strict=True))
            except ValueError as exc:
                raise WriterError(f"{label} root escaped the synthetic run.") from exc
            if path.is_symlink():
                raise WriterError(f"{label} root must not be a symlink.")
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_dir():
                raise WriterError(
                    "Existing target is invalid or ambiguous; replacement is refused."
                )
            from .science import reopen_and_validate_artifacts

            try:
                entries = {path.name for path in target.iterdir()}
                if entries != set(EXPECTED_ARTIFACTS):
                    raise WriterError("Existing target artifact set is incomplete.")
                reopen_and_validate_artifacts(
                    {
                        name: _read_nofollow(target / name)
                        for name in EXPECTED_ARTIFACTS
                    }
                )
            except Exception as exc:
                raise WriterError(
                    "Existing target is invalid or ambiguous; replacement is refused."
                ) from exc
            raise ExistingValidTarget(
                "An independently valid target already exists; overwrite is refused."
            )
        if not TransactionJournal.load(journal_path).snapshot.descriptor.run_id == (
            spec.transaction_id
        ):
            raise WriterError("Durable journal identity failed preflight.")
        capacity = os.statvfs(root)
        free_bytes = int(capacity.f_bavail * capacity.f_frsize)
        free_inodes = int(capacity.f_favail)
        if free_bytes <= 0:
            raise WriterError("No writable filesystem bytes are available.")
        if capacity.f_files and free_inodes <= 0:
            raise WriterError("No writable filesystem inodes are available.")
        return {
            "contract_version": WRITER_CONTRACT_VERSION,
            "capability_environment": self.capability.environment,
            "canonical_run_root": str(root.resolve(strict=True)),
            "canonical_target": str(target),
            "same_device_root": root.stat().st_dev,
            "free_bytes": free_bytes,
            "free_inodes": free_inodes,
            "quota_visibility": "not_observed",
            "plan_hash": plan_hash,
            "configuration_hash": config_hash,
            "provider": self.provider.provider_name,
            "provider_scenario": self.provider.scenario.value,
            "production_authorization": False,
            "live_provider_available": False,
            "cache_created": False,
        }

    @staticmethod
    def _identities(stage: Path) -> Mapping[str, ArtifactIdentity]:
        return {
            name: ArtifactIdentity.from_path(name, stage / name)
            for name in EXPECTED_ARTIFACTS
        }

    @staticmethod
    def _hashes(
        identities: Mapping[str, ArtifactIdentity]
    ) -> Mapping[str, str]:
        return {
            name: identities[name].sha256 for name in sorted(identities)
        }

    def _record_failure(
        self,
        journal: TransactionJournal,
        *,
        error: BaseException,
        published: bool,
        publication_attempted: bool,
        target: Path,
    ) -> Tuple[str, ...]:
        """Best-effort durable failure record; physical evidence remains truth."""
        journal_errors = []
        durability_confirmed = bool(
            journal.snapshot.durability.get("status")
            == "client_fsync_confirmed"
        )
        if published:
            reason = (
                "published_reconciliation_required"
                if durability_confirmed
                else "published_durability_uncertain"
            )
        else:
            reason = type(error).__name__.lower()
        failure = {
            "reason": reason,
            "exception_type": type(error).__name__,
            "message": str(error),
            "publication_boundary_crossed": published,
        }
        try:
            if published and not journal.snapshot.published:
                journal.transition(
                    ExecutionState.PUBLISHED,
                    publication={
                        "attempted": True,
                        "committed": True,
                        "boundary": "manifest.json",
                        "physical_manifest_visible": (
                            (target / "manifest.json").is_file()
                            and not (target / "manifest.json").is_symlink()
                        ),
                    },
                    durability={
                        "status": (
                            "client_fsync_confirmed"
                            if durability_confirmed
                            else "uncertain"
                        )
                    },
                )
            if journal.snapshot.state not in {
                ExecutionState.FAILED,
                ExecutionState.COMPLETE,
            }:
                journal.transition(
                    ExecutionState.FAILED,
                    reason=reason,
                    publication={"attempted": publication_attempted},
                    durability=(
                        {
                            "status": (
                                "client_fsync_confirmed"
                                if durability_confirmed
                                else "uncertain"
                            )
                        }
                        if published
                        else None
                    ),
                    reconciliation=(
                        {"status": "required"} if published else None
                    ),
                    failure=failure,
                )
            elif journal.snapshot.state == ExecutionState.FAILED:
                journal.update(failure=failure)
        except Exception as journal_error:
            journal_errors.append(
                f"{type(journal_error).__name__}: {journal_error}"
            )
        return tuple(journal_errors)

    def execute(self, spec: NightExecutionSpec) -> OperationReport:
        """Run one synthetic transaction through the exact writer machinery."""
        from .science import (
            ProviderError,
            build_night_artifacts,
            reopen_and_validate_artifacts,
        )

        started = self.clock()
        request = spec.science_request
        target_relative = nightly_target_relative(request.date_utc)
        target = contained_path(self.capability.published_root, target_relative)
        plan_document = {
            "writer_contract": WRITER_CONTRACT_VERSION,
            "transaction_id": spec.transaction_id,
            "plan_id": spec.plan_id,
            "release_sha": spec.release_sha,
            "configuration_identity": spec.configuration_identity,
            "provider": self.provider.provider_name,
            "provider_scenario": self.provider.scenario.value,
            "science_request": {
                "date_utc": request.date_utc,
                "mjd_min": request.mjd_min,
                "mjd_max": request.mjd_max,
                "target_loci": request.target_loci,
                "query_tag": request.query_tag,
                "lsst_only": request.lsst_only,
            },
            "target_relative": target_relative.as_posix(),
        }
        plan_hash = _digest(plan_document)
        config_hash = _digest(
            {
                "configuration_identity": spec.configuration_identity,
                "capability_environment": self.capability.environment,
                "provider": self.provider.provider_name,
                "provider_scenario": self.provider.scenario.value,
            }
        )
        writer_lock = WriterLock(
            self.capability,
            target_relative.as_posix(),
            spec.transaction_id,
            transaction_id=spec.transaction_id,
            release_sha=spec.release_sha,
            plan_hash=plan_hash,
            config_hash=config_hash,
        )
        stage = contained_path(
            self.capability.staging_root,
            Path(spec.transaction_id)
            / writer_lock.path.name.removesuffix(".lock"),
        )
        journal_path = contained_path(
            self.capability.journal_root,
            Path(f"{_safe_component(spec.transaction_id, 'transaction_id')}.json"),
        )
        _ensure_private_tree(self.capability.journal_root, self.capability.root)
        if journal_path.exists() or journal_path.is_symlink():
            finished = self.clock()
            try:
                if journal_path.is_symlink():
                    raise JournalError("Existing journal path is a symlink.")
                existing = TransactionJournal.load(journal_path).snapshot
                existing_outcome = existing.outcome.value
                message = (
                    "Transaction identity already exists; duplicate execution is refused."
                )
                status = "duplicate_transaction"
                errors: Tuple[Issue, ...] = ()
                refusals: Tuple[Issue, ...] = (
                    Issue("duplicate_transaction_id", message),
                )
                exit_code = ExitCode.REFUSED
            except JournalError as error:
                existing_outcome = "UNKNOWN"
                status = "ambiguous_existing_journal"
                errors = (Issue(type(error).__name__, str(error)),)
                refusals = ()
                exit_code = ExitCode.OPERATIONAL_FAILURE
            return OperationReport(
                operation="night.synthetic_ingest",
                success=False,
                status=status,
                started_at_utc=_iso(started),
                finished_at_utc=_iso(finished),
                elapsed_seconds=max(0.0, (finished - started).total_seconds()),
                exit_code=exit_code,
                run_id=spec.transaction_id,
                errors=errors,
                refusal_reasons=refusals,
                artifacts=(
                    Artifact(
                        "transaction-journal",
                        "operational-state",
                        str(journal_path),
                        existing_outcome,
                    ),
                ),
                next_actions=(
                    "Inspect the existing transaction; never reuse its identity.",
                ),
                details={
                    "journal_outcome": existing_outcome,
                    "provider_invoked": False,
                    "duplicate_publication_attempted": False,
                    "production_authorized": False,
                },
                read_only=True,
            )
        descriptor = TransactionDescriptor(
            run_id=spec.transaction_id,
            operation="night.synthetic_ingest",
            target_identity=target_relative.as_posix(),
            target_path=str(target),
            stage_path=str(stage),
            lock_path=str(writer_lock.path),
            profile=f"synthetic:{self.capability.environment}",
            plan_id=spec.plan_id,
            release_sha=spec.release_sha,
            metadata={
                "schema_version": "phase5.transaction-descriptor.v1",
                "target_utc_night": request.date_utc,
                "capability_run_id": self.capability.run_id,
                "plan_hash": plan_hash,
                "configuration_identity": spec.configuration_identity,
                "configuration_hash": config_hash,
                "science_provider_identity": (
                    f"{self.provider.provider_name}:{self.provider.scenario.value}"
                ),
                "target_data_root": str(self.capability.published_root),
                "staging_root": str(self.capability.staging_root),
                "operations_root": str(self.capability.root / "control"),
                "journal_root": str(self.capability.journal_root),
                "evidence_root": str(self.capability.evidence_root),
                "cache_root": None,
                "expected_artifacts": list(EXPECTED_ARTIFACTS),
                "production_authorization_available": False,
            },
        )
        journal = TransactionJournal.create(
            journal_path, descriptor, at=started
        )
        transaction: Optional[PublicationTransaction] = None
        science_result: Any = None
        published_identities: Mapping[str, ArtifactIdentity] = {}
        derived: Optional[ReconciliationResult] = None
        reconciliation_lock: Optional[WriterLock] = None
        publication_attempted = False
        published = False
        cleanup_errors = []
        try:
            self._checkpoint("before_preflight")
            preflight = self._preflight(
                spec, target, journal_path, plan_hash, config_hash
            )
            journal.transition(
                ExecutionState.PRECHECKED,
                validation={"preflight": preflight, "passed": True},
            )
            self._checkpoint("after_preflight")
            self._checkpoint("before_lock")
            writer_lock.acquire(at=self.clock())
            journal.transition(
                ExecutionState.LOCKED,
                publication={"writer_lock_acquired": True},
            )
            self._checkpoint("after_lock")

            transaction = PublicationTransaction(
                self.capability,
                writer_lock,
                target_relative,
                spec.transaction_id,
                publication_event_hook=self._publication_event,
            )
            transaction.prepare()
            journal.transition(ExecutionState.QUERYING)
            transaction.begin_query()
            self._checkpoint("during_query")
            query_result = self.provider.query(request)
            query_result.require_completed()
            self._checkpoint("after_successful_query")

            journal.transition(ExecutionState.FETCHING)
            transaction.begin_fetch()
            self._checkpoint("during_fetch")
            science_result = self.provider.fetch(request, query_result)
            science_result.require_publishable()
            self._checkpoint("after_successful_fetch")
            artifacts = build_night_artifacts(science_result)
            reopen_and_validate_artifacts(artifacts, expected=science_result)

            self._checkpoint("during_staging")
            transaction.stage_artifacts(artifacts, science_result.evidence)
            staged_identities = self._identities(transaction.stage)
            journal.transition(
                ExecutionState.STAGED,
                artifacts=staged_identities,
                validation={
                    "query_evidence": science_result.query_evidence.as_dict(),
                    "fetch_evidence": science_result.fetch_evidence.as_dict(),
                    "science": dict(science_result.validation),
                },
            )
            self._checkpoint("after_staging")

            validation_result: Dict[str, Any] = {}

            def staged_validator(stage_directory: Path) -> bool:
                payloads = {
                    name: _read_nofollow(stage_directory / name)
                    for name in EXPECTED_ARTIFACTS
                }
                reopened = reopen_and_validate_artifacts(
                    payloads, expected=science_result
                )
                validation_result.update(
                    {
                        "passed": True,
                        "date_utc": reopened.manifest.get("date_utc"),
                        "actual_loci": len(reopened.loci),
                        "alert_rows": len(reopened.alerts),
                    }
                )
                return True

            self._checkpoint("during_validation")
            transaction.validate(staged_validator)
            validated_identities = self._identities(transaction.stage)
            if validated_identities != staged_identities:
                raise WriterError("Staged identity changed across validation.")
            writer_lock.update_artifact_hashes(
                self._hashes(validated_identities)
            )
            journal.transition(
                ExecutionState.VALIDATED,
                artifacts=validated_identities,
                validation={
                    "passed": True,
                    "result": validation_result,
                },
            )
            self._checkpoint("after_validation")

            self._checkpoint("during_precommit_reproof")
            reproof = transaction.precommit_reprove()
            if {
                name: reproof[name]["sha256"] for name in sorted(reproof)
            } != self._hashes(validated_identities):
                raise WriterError("Pre-commit reproof disagrees with the journal.")
            journal.update(
                validation={
                    "precommit_reproved": True,
                    "precommit_artifact_sha256": self._hashes(
                        validated_identities
                    ),
                }
            )
            self._checkpoint("after_precommit_reproof")
            publication_attempted = True
            journal.update(
                publication={
                    "attempted": True,
                    "committed": False,
                    "boundary": "manifest.json",
                    "strategy": "exclusive-directory-hardlink-manifest-last",
                },
                durability={"status": "pending"},
            )
            self._checkpoint("immediately_before_publication")
            transaction.publish()
            published = True
            journal.transition(
                ExecutionState.PUBLISHED,
                publication={
                    "attempted": True,
                    "committed": True,
                    "boundary_crossed": True,
                    "boundary": "manifest.json",
                },
                durability={"status": "pending_confirmation"},
                reconciliation={"status": "required"},
            )
            self._checkpoint("during_durability_confirmation")
            journal.update(
                durability={
                    "status": "client_fsync_confirmed",
                    "server_crash_survival_claimed": False,
                }
            )
            published_identities = independent_reopen(
                transaction.target, science_result
            )
            if self._hashes(published_identities) != self._hashes(
                validated_identities
            ):
                raise WriterError(
                    "Independent reopen differs from validated staged bytes."
                )
            journal.update(
                publication={
                    "independent_reopen_passed": True,
                    "published_artifact_sha256": self._hashes(
                        published_identities
                    ),
                }
            )

            self._checkpoint("before_unlock")
            transaction.release_writer_lock()
            journal.update(publication={"writer_lock_released": True})
            transaction.begin_reconciliation()

            reconciliation_identity = (
                Path("derived") / "nightly" / request.date_utc
            ).as_posix()
            reconciliation_lock = WriterLock(
                self.capability,
                reconciliation_identity,
                spec.transaction_id,
                transaction_id=f"{spec.transaction_id}:reconciliation",
                release_sha=spec.release_sha,
                plan_hash=plan_hash,
                config_hash=config_hash,
                artifact_hashes=self._hashes(published_identities),
            )
            reconciliation_lock.acquire(at=self.clock())
            journal.transition(
                ExecutionState.RECONCILING,
                reconciliation={
                    "status": "running",
                    "lock_path": str(reconciliation_lock.path),
                    "nightly_publication_independent": True,
                },
            )
            self._checkpoint("during_reconciliation")
            derived = self.reconciler.reconcile(
                self.capability,
                date_utc=request.date_utc,
                published_artifacts=published_identities,
                transaction_id=spec.transaction_id,
            )
            reconciliation_lock.release()
            journal.update(
                reconciliation={
                    "status": "validated",
                    "derived_path": derived.path,
                    "derived_sha256": derived.sha256,
                    "idempotent_replay": derived.idempotent_replay,
                    "lock_released": True,
                }
            )
            self._checkpoint("after_reconciliation_before_completion")
            transaction.complete_reconciliation()
            journal.transition(
                ExecutionState.COMPLETE,
                reconciliation={"status": "complete", "completed": True},
            )
            finished = self.clock()
            return OperationReport(
                operation="night.synthetic_ingest",
                success=True,
                status="complete",
                started_at_utc=_iso(started),
                finished_at_utc=_iso(finished),
                elapsed_seconds=max(0.0, (finished - started).total_seconds()),
                exit_code=ExitCode.SUCCESS,
                run_id=spec.transaction_id,
                counts={
                    "loci": int(science_result.evidence.loci_rows),
                    "alerts": int(science_result.evidence.alert_rows),
                    "published_artifacts": len(published_identities),
                },
                artifacts=tuple(
                    [
                        Artifact(
                            name,
                            "nightly-science",
                            str(transaction.target / name),
                            "published",
                        )
                        for name in EXPECTED_ARTIFACTS
                    ]
                    + [
                        Artifact(
                            "transaction-journal",
                            "operational-state",
                            str(journal_path),
                            "complete",
                        ),
                        Artifact(
                            "synthetic-reconciliation",
                            "derived",
                            derived.path,
                            "complete",
                        ),
                    ]
                ),
                details={
                    "journal_schema": journal.snapshot.schema_version,
                    "journal_outcome": journal.snapshot.outcome.value,
                    "plan_hash": plan_hash,
                    "configuration_hash": config_hash,
                    "provider": self.provider.provider_name,
                    "provider_scenario": self.provider.scenario.value,
                    "publication_boundary": "manifest.json",
                    "independent_reopen": True,
                    "reconciliation_independent": True,
                    "production_authorized": False,
                },
                read_only=False,
            )
        except BaseException as error:
            # Real process death (SIGKILL/os._exit) bypasses this path entirely;
            # subsequent inspection therefore depends only on durable evidence.
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            published = bool(
                published
                or (
                    transaction is not None
                    and transaction.state_machine.published
                )
            )
            cleanup_errors.extend(
                self._record_failure(
                    journal,
                    error=error,
                    published=published,
                    publication_attempted=publication_attempted,
                    target=target,
                )
            )
            if transaction is not None and not published:
                try:
                    transaction.abort("writer_failed_before_publication")
                except Exception as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if reconciliation_lock is not None and reconciliation_lock.held:
                try:
                    reconciliation_lock.release()
                except Exception as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if writer_lock.held:
                try:
                    writer_lock.release()
                except Exception as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            finished = self.clock()
            outcome = journal.snapshot.outcome
            status = outcome.value.lower()
            exit_code = (
                ExitCode.REFUSED
                if isinstance(
                    error,
                    (
                        ExistingValidTarget,
                        LockUnavailable,
                        ProductionAuthorizationUnavailable,
                    ),
                )
                else ExitCode.OPERATIONAL_FAILURE
            )
            if isinstance(error, ExistingValidTarget):
                status = "existing_valid_target"
            return OperationReport(
                operation="night.synthetic_ingest",
                success=False,
                status=status,
                started_at_utc=_iso(started),
                finished_at_utc=_iso(finished),
                elapsed_seconds=max(0.0, (finished - started).total_seconds()),
                exit_code=exit_code,
                run_id=spec.transaction_id,
                errors=(
                    Issue(
                        type(error).__name__,
                        str(error) or type(error).__name__,
                    ),
                ),
                warnings=tuple(
                    Issue("cleanup_or_journal_error", item)
                    for item in cleanup_errors
                ),
                artifacts=(
                    Artifact(
                        "transaction-journal",
                        "operational-state",
                        str(journal_path),
                        journal.snapshot.state.value,
                    ),
                ),
                next_actions=(
                    "Run read-only recovery inspection; do not delete ambiguous state.",
                ),
                details={
                    "journal_outcome": outcome.value,
                    "published": journal.snapshot.published,
                    "publication_attempted": publication_attempted,
                    "target": str(target),
                    "stage": str(stage),
                    "lock": str(writer_lock.path),
                    "production_authorized": False,
                },
                read_only=False,
            )


def execute_synthetic_night(
    capability: SyntheticWriteCapability,
    provider: Any,
    spec: NightExecutionSpec,
    *,
    reconciler: Optional[SyntheticReconciler] = None,
    fault_hook: Optional[FaultHook] = None,
    clock: Callable[[], datetime] = _utc_now,
) -> OperationReport:
    """Shared Python/Jupyter/canary API for the one writer implementation."""
    return TransactionalNightWriter(
        capability,
        provider,
        reconciler=reconciler,
        fault_hook=fault_hook,
        clock=clock,
    ).execute(spec)


__all__ = [
    "FailureInjector",
    "ExistingValidTarget",
    "InjectedWriterFailure",
    "NightExecutionSpec",
    "ProductionAuthorizationUnavailable",
    "ReconciliationResult",
    "SyntheticReconciler",
    "TransactionalNightWriter",
    "WriterError",
    "execute_synthetic_night",
    "independent_reopen",
    "nightly_target_relative",
    "production_ingest_refusal",
]
