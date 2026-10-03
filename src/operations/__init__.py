"""Reusable planning, recovery, and capability-guarded writer contracts.

The public planning API is safe for CLI and Jupyter use. Transaction and lock
implementations are restricted to explicit temporary/synthetic canary
capabilities; production writer activation and a live provider are absent.
"""

from .context import (
    NullLogger,
    OperationContext,
    context_from_environment,
    context_from_profile,
)
from .locking import (
    LockInspection,
    LockOwnershipError,
    LockUnavailable,
    WriterLock,
    lock_identity,
)
from .journal import (
    ArtifactIdentity,
    JournalCorrupt,
    JournalError,
    JournalOutcome,
    JournalSnapshot,
    TransactionDescriptor,
    TransactionJournal,
)
from .plan import plan_backfill, plan_night
from .report import (
    Artifact,
    Evidence,
    ExitCode,
    Issue,
    OperationReport,
)
from .state import (
    ExecutionState,
    ExecutionStateMachine,
    IllegalTransition,
    StateSnapshot,
)
from .storage import (
    ACCEPTED_ZERO_ROW_NIGHTS,
    DevelopmentWriteCapability,
    NightInspection,
    NightLocation,
    StorageContractError,
    StorageLayout,
    SyntheticWriteCapability,
    contained_path,
    validate_root_separation,
    valid_zero_row_evidence,
)
from .transaction import (
    PublicationTransaction,
    QueryFetchEvidence,
    TransactionError,
)
from .recovery import (
    RecoveryAssessment,
    RecoveryDisposition,
    RecoveryInspector,
    inspect_recovery,
)
from .writer import (
    FailureInjector,
    InjectedWriterFailure,
    NightExecutionSpec,
    ProductionAuthorizationUnavailable,
    SyntheticReconciler,
    TransactionalNightWriter,
    WriterError,
    execute_synthetic_night,
    independent_reopen,
    nightly_target_relative,
    production_ingest_refusal,
)


__all__ = [
    "ACCEPTED_ZERO_ROW_NIGHTS",
    "Artifact",
    "ArtifactIdentity",
    "DevelopmentWriteCapability",
    "Evidence",
    "ExecutionState",
    "ExecutionStateMachine",
    "ExitCode",
    "IllegalTransition",
    "Issue",
    "JournalCorrupt",
    "JournalError",
    "JournalOutcome",
    "JournalSnapshot",
    "LockInspection",
    "LockOwnershipError",
    "LockUnavailable",
    "NightInspection",
    "NightLocation",
    "NullLogger",
    "OperationContext",
    "OperationReport",
    "PublicationTransaction",
    "QueryFetchEvidence",
    "RecoveryAssessment",
    "RecoveryDisposition",
    "RecoveryInspector",
    "StateSnapshot",
    "StorageContractError",
    "StorageLayout",
    "SyntheticWriteCapability",
    "TransactionError",
    "WriterLock",
    "FailureInjector",
    "InjectedWriterFailure",
    "NightExecutionSpec",
    "ProductionAuthorizationUnavailable",
    "SyntheticReconciler",
    "TransactionDescriptor",
    "TransactionJournal",
    "TransactionalNightWriter",
    "WriterError",
    "contained_path",
    "context_from_environment",
    "context_from_profile",
    "execute_synthetic_night",
    "independent_reopen",
    "inspect_recovery",
    "lock_identity",
    "plan_backfill",
    "plan_night",
    "nightly_target_relative",
    "production_ingest_refusal",
    "validate_root_separation",
    "valid_zero_row_evidence",
]
