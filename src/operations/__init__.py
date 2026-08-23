"""Reusable read-only operations and guarded future-writer contracts.

The public planning API is safe for CLI and Jupyter use. Transaction and lock
implementations are restricted to explicit local temporary-root capabilities;
production writer activation is intentionally absent.
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
    contained_path,
    validate_root_separation,
    valid_zero_row_evidence,
)
from .transaction import (
    PublicationTransaction,
    QueryFetchEvidence,
    TransactionError,
)


__all__ = [
    "ACCEPTED_ZERO_ROW_NIGHTS",
    "Artifact",
    "DevelopmentWriteCapability",
    "Evidence",
    "ExecutionState",
    "ExecutionStateMachine",
    "ExitCode",
    "IllegalTransition",
    "Issue",
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
    "StateSnapshot",
    "StorageContractError",
    "StorageLayout",
    "TransactionError",
    "WriterLock",
    "contained_path",
    "context_from_environment",
    "context_from_profile",
    "lock_identity",
    "plan_backfill",
    "plan_night",
    "validate_root_separation",
    "valid_zero_row_evidence",
]
