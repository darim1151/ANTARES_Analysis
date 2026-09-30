"""Explicit, authorized, crash-consistent publication of one validated nightly candidate.

Candidate (recovery or backfill) bytes are never rewritten.  Authority is a
separate state: the published nightly ``manifest.json`` is derived from the
candidate manifest, records the candidate's hash and original chronology as
provenance, and carries publication chronology of its own.  Loci and alert
Parquet bytes are published unchanged.

Authority commit model
----------------------
One publication changes reader-visible products that POSIX cannot replace in
one operation: the nightly partition and the two cumulative Parquet files.
Canonical readers hold the NFS-resident publication authority lock in shared
mode for their complete logical read; the publisher holds that same object
exclusively for its complete lifecycle. The publisher also brackets the change
with one durable production marker,
``data/lsst_only/PUBLICATION_TRANSACTION_IN_PROGRESS.json`` (the *gate*):

1. Before the gate exists, production is byte-for-byte the authorized
   baseline: ``NOT_COMMITTED``.  All staging, planning and validation happen
   here, followed by a fresh full Sentinel V2 qualification.
2. The gate is hard-linked into production (``O_EXCL`` semantics) before any
   other production mutation. Readers are already excluded by the exclusive
   authority lock; after a crash releases that lock, the gate makes every
   interruption ``RECONCILIATION_REQUIRED`` and readers refuse. The journal,
   gate and staged material suffice to roll forward deterministically.
3. The nightly partition (manifest-last) and both cumulative files are
   installed idempotently and verified against the journal.
4. The durable removal of the gate is the single authority commit
   (``COMPLETE``).  Recovery recognizes ``COMPLETE`` from physical evidence
   even when the journal or the terminal record was not yet written.

This is not an atomic multi-file replace. The kernel lock provides live
reader/writer serialization; the gate and journal provide durable recovery
classification. Reader-visible outcomes are the previous generation, an
explicit refusal for unresolved recovery, or the complete new generation.
Classification is always derived from durable evidence, never from in-process
flags.

Only a sealed synthetic capability or the exact Control-token-qualified June
27 production capability is accepted.  The production issuer is deliberately
hard-bound to that single canary and cannot authorize another night.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import io
import json
import os
import re
import socket
import stat
import time
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .. import history
from ..authority import (
    AUTHORITY_LOCK_NAME,
    AuthorityLockError,
    AuthorityLockUnavailable,
    exclusive_authority_lock,
)
from .journal import (
    ArtifactIdentity,
    JournalError,
    JournalOutcome,
    TransactionDescriptor,
    TransactionJournal,
)
from .locking import LockUnavailable, WriterLock, lock_identity
from .state import ExecutionState
from .storage import (
    PRODUCTION_AUTHORITY_ROOT,
    PRODUCTION_CONTROL_ROOT,
    PRODUCTION_DATA_ROOT,
    PRODUCTION_EVIDENCE_ROOT,
    PRODUCTION_STAGE_ROOT,
    ProductionPublicationCapability,
    StorageContractError,
    SyntheticWriteCapability,
    _issue_production_publication_capability,
    contained_path,
)
from .transaction import (
    PublicationTransaction,
    QueryFetchEvidence,
    _artifact_snapshot,
    _ensure_directory_tree_fsynced,
    _fsync_directory,
    _publish_noreplace,
)
from .writer import (
    EXPECTED_ARTIFACTS,
    RECONCILIATION_LOCK_POLL_SECONDS,
    RECONCILIATION_LOCK_WAIT_SECONDS,
    SHARED_RECONCILIATION_LOCK_IDENTITY,
    ProductionAuthorizationUnavailable,
    WriterError,
    _ensure_private_tree,
    independent_reopen,
    nightly_target_relative,
)


AUTHORIZATION_SCHEMA = "v3.night-publication-authorization.v2"
AUTHORITY_SCHEMA = "v3.night-authority.v2"
PUBLICATION_RECORD_SCHEMA = "v3.night-publication-record.v2"
CANDIDATE_RECORD_SCHEMA = "v3.night-candidate-record.v1"
GATE_SCHEMA = "v3.publication-gate.v1"
PublicationWriteCapability = Union[
    SyntheticWriteCapability,
    ProductionPublicationCapability,
]
DESCRIPTOR_SCHEMA = "v3.publication-descriptor.v2"
PUBLICATION_OPERATION = "night.publish_candidate"
AUTHORIZED_OPERATION = "publish-night"
PUBLICATION_LOCK_NAME = AUTHORITY_LOCK_NAME
CANDIDATE_KINDS = frozenset({"offline-recovery", "backfill-night"})
TRANSACTION_ID_PATTERN = re.compile(
    r"^v3pub-(\d{4}-\d{2}-\d{2})-([0-9a-f]{12})-a([1-9][0-9]{0,5})$"
)
_HEX64 = frozenset("0123456789abcdef")
_CUMULATIVE_KEYS = ("loci_index", "nightly_summary")

FaultHook = Callable[[str, Mapping[str, Any]], None]


# ---------------------------------------------------------------------------
# Authority states and failure categories
# ---------------------------------------------------------------------------


class AuthorityState(str, Enum):
    """The only externally visible interpretations of one publication."""

    NOT_COMMITTED = "NOT_COMMITTED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    COMPLETE = "COMPLETE"
    CONTRADICTION = "CONTRADICTION"


class FailureCategory(str, Enum):
    LOCK_CONTENTION = "transient_lock_contention"
    INTERRUPTED_BEFORE_COMMIT = "retryable_interrupted_before_authority_transition"
    RECONCILIATION_INTERRUPTED = "retryable_interrupted_reconciliation"
    TRANSIENT_NETWORK = "retryable_network_or_transient_io"
    AUTHORIZATION_REFUSED = "permanent_authorization_refusal"
    FILESYSTEM_PERMANENT = "permanent_filesystem_or_permission"
    CANDIDATE_CORRUPTION = "candidate_corruption"
    SENTINEL_DRIFT = "sentinel_drift"
    PREDECESSOR_GAP = "predecessor_gap"
    JOURNAL_CONTRADICTION = "unrecoverable_journal_contradiction"
    UNCLASSIFIED = "unclassified_failure"


RETRYABLE_CATEGORIES = frozenset(
    {
        FailureCategory.LOCK_CONTENTION,
        FailureCategory.INTERRUPTED_BEFORE_COMMIT,
        FailureCategory.RECONCILIATION_INTERRUPTED,
        FailureCategory.TRANSIENT_NETWORK,
    }
)

_REFUSAL_CATEGORIES = {
    FailureCategory.AUTHORIZATION_REFUSED: (
        "authorization_absent", "authorization_invalid", "authorization_night_mismatch",
        "authorization_candidate_mismatch", "authorization_release_mismatch",
        "authorization_binding_mismatch", "authorization_expired",
        "authorization_nonce_reused", "already_authoritative",
        "production_publication_not_enabled",
    ),
    FailureCategory.CANDIDATE_CORRUPTION: (
        "candidate_missing", "candidate_incomplete", "candidate_record_invalid",
        "candidate_hash_mismatch", "candidate_validation_failed",
        "candidate_already_authoritative", "candidate_night_mismatch",
        "candidate_not_terminal_unpublished", "chronology_invalid", "date_invalid",
        "unsafe_evidence", "evidence_invalid",
    ),
    FailureCategory.SENTINEL_DRIFT: (
        "sentinel_drift", "mount_binding_drift", "production_root_mismatch",
        "runtime_linkage_mismatch", "cache_present", "cumulative_plan_mismatch",
        "cumulative_schema_drift", "cumulative_duplicate_night",
        "cumulative_extension_not_exact",
    ),
    FailureCategory.PREDECESSOR_GAP: ("predecessor_gap",),
    FailureCategory.JOURNAL_CONTRADICTION: (
        "journal_contradiction", "authority_contradiction", "path_contradiction",
        "transaction_residue", "cumulative_contradiction",
    ),
    FailureCategory.LOCK_CONTENTION: ("publication_lock_busy", "transaction_pending"),
    FailureCategory.RECONCILIATION_INTERRUPTED: ("authority_transition_pending",),
}
REFUSAL_CATEGORY = {
    code: category for category, codes in _REFUSAL_CATEGORIES.items() for code in codes
}

_PERMANENT_ERRNOS = frozenset(
    value
    for value in (
        errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOSPC,
        getattr(errno, "EDQUOT", None), errno.ENOTDIR, errno.EISDIR, errno.ELOOP,
        errno.EXDEV, errno.ENAMETOOLONG, errno.EMLINK, errno.ENOENT, errno.EEXIST,
        errno.ENOTEMPTY, getattr(errno, "ENOLCK", None),
    )
    if value is not None
)
_TRANSIENT_ERRNOS = frozenset(
    value
    for value in (
        errno.EINTR, errno.EAGAIN, errno.EIO, errno.EBUSY, getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None), getattr(errno, "ECONNRESET", None),
        getattr(errno, "ECONNREFUSED", None), getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "ENETUNREACH", None),
    )
    if value is not None
)


class PublicationError(WriterError):
    """Publication failed; physical evidence and the journal remain truth."""


class PublicationRefused(PublicationError):
    """A precondition or evidence check failed closed."""

    def __init__(
        self, code: str, message: str, category: Optional[FailureCategory] = None
    ) -> None:
        self.code = code
        self.category = category or REFUSAL_CATEGORY.get(code, FailureCategory.UNCLASSIFIED)
        super().__init__(f"{code}: {message}")


def classify_failure(
    error: BaseException, *, authority_transition_begun: bool = False
) -> FailureCategory:
    """Map one failure to an explicit category; never 'transient because OSError'.

    Wrapping layers (for example the transaction's manifest-last errors) are
    looked through: the first recognized exception in the cause chain decides.
    """
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        category = _classify_one(current, authority_transition_begun)
        if category is not FailureCategory.UNCLASSIFIED:
            return category
        current = current.__cause__ or current.__context__
    return FailureCategory.UNCLASSIFIED


def _classify_one(error: BaseException, authority_transition_begun: bool) -> FailureCategory:
    from .fetch_checkpoint import FetchCheckpointFetchError
    from .writer import InjectedWriterFailure

    if isinstance(error, PublicationRefused):
        return error.category
    if isinstance(error, (LockUnavailable, history.PublicationInProgress)):
        return FailureCategory.LOCK_CONTENTION
    interrupted = (
        FailureCategory.RECONCILIATION_INTERRUPTED
        if authority_transition_begun
        else FailureCategory.INTERRUPTED_BEFORE_COMMIT
    )
    if isinstance(error, InjectedWriterFailure):
        return interrupted
    if isinstance(error, (ConnectionError, TimeoutError)):
        return FailureCategory.TRANSIENT_NETWORK
    if isinstance(error, FetchCheckpointFetchError):
        cause = error.__cause__
        if isinstance(cause, (ConnectionError, TimeoutError)) or (
            isinstance(cause, OSError) and cause.errno in _TRANSIENT_ERRNOS
        ):
            return FailureCategory.TRANSIENT_NETWORK
        return FailureCategory.CANDIDATE_CORRUPTION
    issue = getattr(error, "issue", None)
    if issue is not None and getattr(issue, "retryable", False):
        return FailureCategory.TRANSIENT_NETWORK
    if isinstance(error, OSError):
        if error.errno in _TRANSIENT_ERRNOS:
            return (
                FailureCategory.RECONCILIATION_INTERRUPTED
                if authority_transition_begun
                else FailureCategory.TRANSIENT_NETWORK
            )
        # EACCES/EROFS/ENOSPC/structural errors, and OSErrors without an errno,
        # are never presumed transient.
        return FailureCategory.FILESYSTEM_PERMANENT
    if isinstance(error, (JournalError, StorageContractError)):
        return FailureCategory.JOURNAL_CONTRADICTION
    return FailureCategory.UNCLASSIFIED


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise PublicationError("Publication clocks must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PublicationRefused("chronology_invalid", f"{label} is missing.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PublicationRefused(
            "chronology_invalid", f"{label} is not ISO-8601."
        ) from exc
    if parsed.tzinfo is None:
        raise PublicationRefused("chronology_invalid", f"{label} lacks a timezone.")
    return parsed.astimezone(timezone.utc)


def _require_order(*pairs: Tuple[str, Any]) -> None:
    """Refuse unless every labelled timestamp is <= the next one."""
    parsed = [(label, _parse_utc(value, label)) for label, value in pairs]
    for (left_label, left), (right_label, right) in zip(parsed, parsed[1:]):
        if left > right:
            raise PublicationRefused(
                "chronology_invalid", f"{left_label} is later than {right_label}."
            )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _is_sha40(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _HEX64


def _canonical_date(value: Any, label: str) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise PublicationRefused("date_invalid", f"{label} is not YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise PublicationRefused("date_invalid", f"{label} is not YYYY-MM-DD.")
    return value


def _previous_night(date_utc: str) -> str:
    return (date.fromisoformat(date_utc) - timedelta(days=1)).isoformat()


def _read_regular(path: Path) -> bytes:
    descriptor = os.open(
        str(path),
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PublicationRefused("unsafe_evidence", f"{path.name} is not regular.")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Dict[str, Any]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PublicationRefused("evidence_invalid", f"Duplicate key in {path.name}.")
            result[key] = value
        return result

    try:
        value = json.loads(_read_regular(path).decode("utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationRefused("evidence_invalid", f"{path.name} is unreadable.") from exc
    if not isinstance(value, dict):
        raise PublicationRefused("evidence_invalid", f"{path.name} is not an object.")
    return value


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(_read_regular(path))


def _optional_sha256(path: Path) -> Optional[str]:
    if not path.exists() and not path.is_symlink():
        return None
    return _file_sha256(path)


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    """Create one durable evidence file; never replace existing evidence."""
    from .commissioning import _write_json_atomic

    if path.exists() or path.is_symlink():
        raise PublicationError(f"Evidence already exists: {path.name}.")
    _write_json_atomic(path, value)


def _write_new_file(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        str(path),
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(errno.EIO, f"Short write: {path.name}.")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_real_path(root: Path, path: Path) -> None:
    """Refuse symlink substitution anywhere between ``root`` and ``path``."""
    root = Path(root)
    if root.is_symlink():
        raise PublicationRefused("path_contradiction", f"{root} is a symlink.")
    try:
        relative = Path(path).relative_to(root)
    except ValueError as exc:
        raise PublicationRefused("path_contradiction", f"{path} escapes {root}.") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PublicationRefused("path_contradiction", f"{cursor} is a symlink.")


# ---------------------------------------------------------------------------
# Candidate state (never authoritative)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NightCandidate:
    """A validated, non-authoritative nightly candidate and its provenance."""

    kind: str
    date_utc: str
    candidate_dir: Path
    record_path: Path
    record_sha256: str
    release_sha: str
    artifacts: Mapping[str, Mapping[str, Any]]
    loci: int
    alerts: int
    validation_passed: bool
    authoritative: bool
    construction_completed_at_utc: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def artifact_sha256(self) -> Dict[str, str]:
        return {name: str(self.artifacts[name]["sha256"]) for name in EXPECTED_ARTIFACTS}

    @property
    def provenance_sha256(self) -> str:
        return _sha256_bytes(_canonical(dict(self.provenance)))


def _verify_candidate_artifacts(
    candidate_dir: Path, recorded: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    if candidate_dir.is_symlink() or not candidate_dir.is_dir():
        raise PublicationRefused("candidate_missing", "Candidate directory is unsafe.")
    entries = {path.name for path in candidate_dir.iterdir()}
    required = set(EXPECTED_ARTIFACTS)
    if not required <= entries:
        raise PublicationRefused("candidate_incomplete", "Candidate artifacts are missing.")
    if set(recorded) != required:
        raise PublicationRefused("candidate_record_invalid", "Recorded artifact set differs.")
    observed = {}
    for name in EXPECTED_ARTIFACTS:
        payload = _read_regular(candidate_dir / name)
        observed[name] = {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        expected = recorded[name]
        if (
            not isinstance(expected, Mapping)
            or expected.get("sha256") != observed[name]["sha256"]
            or expected.get("bytes") != observed[name]["bytes"]
        ):
            raise PublicationRefused(
                "candidate_hash_mismatch", f"Candidate {name} differs from its record."
            )
    return observed


def _candidate_manifest(candidate_dir: Path) -> Dict[str, Any]:
    manifest = _read_json(candidate_dir / "manifest.json")
    for key in ("authoritative", "publishable", "publication_authorized"):
        if manifest.get(key) not in (None, False):
            raise PublicationRefused(
                "candidate_already_authoritative",
                f"Candidate manifest declares {key}={manifest.get(key)!r}.",
            )
    if "authority" in manifest:
        raise PublicationRefused(
            "candidate_already_authoritative", "Candidate carries an authority block."
        )
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("append_ready") is not True:
        raise PublicationRefused("candidate_validation_failed", "Candidate is not append-ready.")
    return manifest


def load_offline_recovery_candidate(root: Path) -> NightCandidate:
    """Load a terminal ``RECOVERY_COMPLETE_UNPUBLISHED`` offline-recovery root."""
    from .offline_recovery import CONTRACT

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise PublicationRefused("candidate_missing", "Recovery root is missing or unsafe.")
    binding_path = root / "binding.json"
    binding = _read_json(binding_path)
    seal = _read_regular(root / "binding.sha256").decode("ascii").strip()
    if _file_sha256(binding_path) != seal:
        raise PublicationRefused("candidate_record_invalid", "Recovery binding seal differs.")
    final_path = root / "status" / "RECOVERY_FINAL.json"
    final = _read_json(final_path)
    counts = final.get("callback_and_network_counts")
    source_before = final.get("source_before_sha256")
    production_before = final.get("production_before_sha256")
    if not (
        binding.get("schema_version") == CONTRACT
        and final.get("schema_version") == CONTRACT
        and final.get("run_id") == binding.get("run_id") == root.name
        and final.get("binding_sha256") == seal
        and final.get("success") is True
        and final.get("status") == "RECOVERY_COMPLETE_UNPUBLISHED"
        and final.get("authoritative") is False
        and final.get("publishable") is False
        and final.get("publication_attempted") is False
        and binding.get("authoritative") is False
        and binding.get("publishable") is False
        and binding.get("publication_authorized") is False
        and isinstance(counts, dict)
        and counts
        and all(value == 0 for value in counts.values())
        and _is_hex64(source_before)
        and source_before == final.get("source_after_sha256")
        and source_before == binding.get("source_durable_identity")
        and _is_hex64(production_before)
        and production_before == final.get("production_after_sha256")
    ):
        raise PublicationRefused(
            "candidate_not_terminal_unpublished",
            "Recovery evidence is not a clean terminal unpublished success.",
        )
    validation = final.get("validation")
    if not isinstance(validation, dict) or validation.get("append_ready") is not True:
        raise PublicationRefused("candidate_validation_failed", "Recovery validation did not pass.")
    recorded = _read_json(root / "evidence" / "artifacts.json")
    if final.get("artifacts") != recorded:
        raise PublicationRefused("candidate_record_invalid", "Recovery artifact evidence differs.")
    candidate_dir = root / "candidate"
    artifacts = _verify_candidate_artifacts(candidate_dir, recorded)
    manifest = _candidate_manifest(candidate_dir)
    night = _canonical_date(binding.get("night"), "night")
    if manifest.get("date_utc") != night:
        raise PublicationRefused("candidate_night_mismatch", "Manifest night differs from binding.")
    fetch = final.get("fetch_checkpoint")
    fetch = fetch if isinstance(fetch, dict) else {}
    release_sha = binding.get("consumer_sha")
    if not _is_sha40(release_sha):
        raise PublicationRefused("candidate_record_invalid", "Recovery release identity is invalid.")
    finished = final.get("finished_at_utc")
    _parse_utc(finished, "recovery finished_at_utc")
    return NightCandidate(
        kind="offline-recovery",
        date_utc=night,
        candidate_dir=candidate_dir,
        record_path=final_path,
        record_sha256=_file_sha256(final_path),
        release_sha=release_sha,
        artifacts=artifacts,
        loci=int(manifest["actual_loci"]),
        alerts=int(manifest["alert_rows"]),
        validation_passed=True,
        authoritative=False,
        construction_completed_at_utc=finished,
        provenance={
            "recovery_root": str(root),
            "recovery_run_id": root.name,
            "recovery_contract": CONTRACT,
            "binding_sha256": seal,
            "source_root": binding.get("source_root"),
            "source_release_sha": binding.get("source_sha"),
            "source_durable_identity": source_before,
            "query_identity": binding.get("query_identity"),
            "fetch_identity": binding.get("fetch_identity"),
            "reused_segments": fetch.get("reused_segments"),
            "fetched_segments": fetch.get("fetched_segments"),
            "recovery_production_fingerprint": production_before,
            "recovery_finished_at_utc": finished,
        },
    )


def load_backfill_candidate(night_root: Path) -> NightCandidate:
    """Load a candidate committed by the backfill controller."""
    night_root = Path(night_root)
    candidate_dir = night_root / "candidate"
    record_path = candidate_dir / "candidate-record.json"
    if not record_path.is_file() or record_path.is_symlink():
        raise PublicationRefused("candidate_missing", "Candidate record is absent.")
    record = _read_json(record_path)
    if not (
        record.get("schema_version") == CANDIDATE_RECORD_SCHEMA
        and record.get("authoritative") is False
        and record.get("validation_passed") is True
        and _is_sha40(record.get("release_sha"))
    ):
        raise PublicationRefused("candidate_record_invalid", "Backfill candidate record is invalid.")
    night = _canonical_date(record.get("date_utc"), "date_utc")
    artifacts = _verify_candidate_artifacts(candidate_dir, record.get("artifacts", {}))
    manifest = _candidate_manifest(candidate_dir)
    if manifest.get("date_utc") != night:
        raise PublicationRefused("candidate_night_mismatch", "Manifest night differs from record.")
    if record.get("loci") != manifest.get("actual_loci") or record.get("alerts") != manifest.get("alert_rows"):
        raise PublicationRefused("candidate_record_invalid", "Candidate counts differ.")
    completed = record.get("constructed_at_utc")
    _parse_utc(completed, "constructed_at_utc")
    return NightCandidate(
        kind="backfill-night",
        date_utc=night,
        candidate_dir=candidate_dir,
        record_path=record_path,
        record_sha256=_file_sha256(record_path),
        release_sha=record["release_sha"],
        artifacts=artifacts,
        loci=int(record["loci"]),
        alerts=int(record["alerts"]),
        validation_passed=True,
        authoritative=False,
        construction_completed_at_utc=completed,
        provenance=dict(record.get("provenance", {})),
    )


# ---------------------------------------------------------------------------
# Separate, explicit authorization bound to the full Sentinel V2 context
# ---------------------------------------------------------------------------

_AUTH_FIELDS = (
    "schema_version",
    "operation",
    "date_utc",
    "predecessor_date_utc",
    "candidate_kind",
    "candidate_dir",
    "candidate_record_sha256",
    "candidate_provenance_sha256",
    "candidate_release_sha",
    "artifact_sha256",
    "publisher_release_sha",
    "production",
    "expected_cumulative_sha256",
    "nonce",
    "authorized_by",
    "authorized_at_utc",
    "expires_at_utc",
)
_PRODUCTION_FIELDS = frozenset(
    {
        "canonical_root",
        "mount_binding",
        "durable_fingerprint_sha256",
        "manifest_count",
        "cumulative_sha256",
        "predicates",
    }
)
_MOUNT_FIELDS = frozenset({"mount_point", "filesystem_type", "source"})
_PREDICATE_FIELDS = frozenset(
    {"target_path", "target_absent", "cache_path", "cache_absent", "transaction_artifacts"}
)


def production_binding_from_sentinel(sentinel: Mapping[str, Any]) -> Dict[str, Any]:
    """Project one Sentinel V2 capture onto its durable, authorizable identity.

    Runtime ``st_dev`` observations are deliberately excluded: devices are
    session-local and are re-checked live at the mutation boundary instead.
    """
    state = sentinel["durable_state"]
    predicates = sentinel["qualification_predicates"]
    return {
        "canonical_root": str(state["canonical_data_root"]),
        "mount_binding": {key: str(sentinel["mount_binding"][key]) for key in sorted(_MOUNT_FIELDS)},
        "durable_fingerprint_sha256": str(sentinel["durable_fingerprint_sha256"]),
        "manifest_count": int(state["manifest_count"]),
        "cumulative_sha256": {
            key: str(state["cumulative_artifact_hashes"][key]) for key in _CUMULATIVE_KEYS
        },
        "predicates": {
            "target_path": str(predicates["target_path"]),
            "target_absent": predicates["target_absent"],
            "cache_path": str(predicates["cache_path"]),
            "cache_absent": predicates["cache_absent"],
            "transaction_artifacts": list(predicates["transaction_artifacts"]),
        },
    }


def _validate_production_binding(value: Any, date_utc: str) -> Dict[str, Any]:
    def invalid(message: str) -> PublicationRefused:
        return PublicationRefused("authorization_invalid", f"Production binding: {message}")

    if not isinstance(value, Mapping) or set(value) != _PRODUCTION_FIELDS:
        raise invalid("fields differ.")
    normalized = json.loads(_canonical(dict(value)))
    root = normalized["canonical_root"]
    if not isinstance(root, str) or not Path(root).is_absolute() or ".." in Path(root).parts:
        raise invalid("canonical_root must be absolute.")
    mount = normalized["mount_binding"]
    if (
        not isinstance(mount, dict)
        or set(mount) != _MOUNT_FIELDS
        or not all(isinstance(item, str) and item for item in mount.values())
    ):
        raise invalid("mount binding is malformed.")
    if not _is_hex64(normalized["durable_fingerprint_sha256"]):
        raise invalid("durable fingerprint is malformed.")
    count = normalized["manifest_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise invalid("manifest_count is malformed.")
    cumulative = normalized["cumulative_sha256"]
    if (
        not isinstance(cumulative, dict)
        or set(cumulative) != set(_CUMULATIVE_KEYS)
        or not all(_is_hex64(item) for item in cumulative.values())
    ):
        raise invalid("cumulative baseline is malformed.")
    predicates = normalized["predicates"]
    expected_target = str(history.nightly_paths(Path(root), date_utc)["dir"])
    if (
        not isinstance(predicates, dict)
        or set(predicates) != _PREDICATE_FIELDS
        or predicates["target_path"] != expected_target
        or predicates["target_absent"] is not True
        or predicates["cache_absent"] is not True
        or predicates["transaction_artifacts"] != []
        or not isinstance(predicates["cache_path"], str)
        or not Path(predicates["cache_path"]).is_absolute()
    ):
        raise invalid("Sentinel predicates are not the qualified publication state.")
    return normalized


@dataclass(frozen=True)
class PublicationAuthorization:
    """Control's explicit permission to publish exactly one candidate once.

    Integrity is bound by :attr:`digest`.  The repository has no trusted
    signing primitive, so authenticity rests on filesystem ownership of the
    authorization and control roots; a signature is a Control requirement.
    """

    date_utc: str
    predecessor_date_utc: str
    candidate_kind: str
    candidate_dir: str
    candidate_record_sha256: str
    candidate_provenance_sha256: str
    candidate_release_sha: str
    artifact_sha256: Mapping[str, str]
    publisher_release_sha: str
    production: Mapping[str, Any]
    expected_cumulative_sha256: Mapping[str, str]
    nonce: str
    authorized_by: str
    authorized_at_utc: str
    expires_at_utc: str
    operation: str = AUTHORIZED_OPERATION
    schema_version: str = AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        _canonical_date(self.date_utc, "date_utc")
        _canonical_date(self.predecessor_date_utc, "predecessor_date_utc")
        if (
            self.schema_version != AUTHORIZATION_SCHEMA
            or self.operation != AUTHORIZED_OPERATION
            or self.predecessor_date_utc != _previous_night(self.date_utc)
            or self.candidate_kind not in CANDIDATE_KINDS
            or not isinstance(self.candidate_dir, str)
            or not Path(self.candidate_dir).is_absolute()
            or not _is_hex64(self.candidate_record_sha256)
            or not _is_hex64(self.candidate_provenance_sha256)
            or not _is_sha40(self.candidate_release_sha)
            or not _is_sha40(self.publisher_release_sha)
            or not isinstance(self.artifact_sha256, Mapping)
            or set(self.artifact_sha256) != set(EXPECTED_ARTIFACTS)
            or not all(_is_hex64(value) for value in self.artifact_sha256.values())
            or not isinstance(self.expected_cumulative_sha256, Mapping)
            or set(self.expected_cumulative_sha256) != set(_CUMULATIVE_KEYS)
            or not all(_is_hex64(value) for value in self.expected_cumulative_sha256.values())
            or not isinstance(self.nonce, str)
            or len(self.nonce) != 32
            or not set(self.nonce) <= _HEX64
            or not isinstance(self.authorized_by, str)
            or not self.authorized_by.strip()
        ):
            raise PublicationRefused("authorization_invalid", "Authorization is malformed.")
        _require_order(
            ("authorized_at_utc", self.authorized_at_utc),
            ("expires_at_utc", self.expires_at_utc),
        )
        if _parse_utc(self.authorized_at_utc, "a") == _parse_utc(self.expires_at_utc, "e"):
            raise PublicationRefused("authorization_invalid", "Authorization has no validity.")
        object.__setattr__(
            self, "production", _validate_production_binding(self.production, self.date_utc)
        )
        object.__setattr__(self, "artifact_sha256", dict(sorted(self.artifact_sha256.items())))
        object.__setattr__(
            self,
            "expected_cumulative_sha256",
            dict(sorted(self.expected_cumulative_sha256.items())),
        )

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(_canonical({name: getattr(self, name) for name in _AUTH_FIELDS}))

    @property
    def digest(self) -> str:
        return _sha256_bytes(_canonical(self.as_dict()))

    @property
    def baseline_production_fingerprint(self) -> str:
        return str(self.production["durable_fingerprint_sha256"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublicationAuthorization":
        if not isinstance(value, Mapping) or set(value) != set(_AUTH_FIELDS):
            raise PublicationRefused("authorization_invalid", "Authorization fields differ.")
        return cls(**{name: value[name] for name in _AUTH_FIELDS})


def authorize_publication(
    candidate: NightCandidate,
    *,
    production: Mapping[str, Any],
    predecessor_date_utc: str,
    publisher_release_sha: str,
    expected_cumulative_sha256: Mapping[str, str],
    authorized_by: str,
    authorized_at_utc: str,
    expires_at_utc: str,
    nonce: str,
) -> PublicationAuthorization:
    """Construct an authorization bound to exact candidate and production evidence."""
    return PublicationAuthorization(
        date_utc=candidate.date_utc,
        predecessor_date_utc=predecessor_date_utc,
        candidate_kind=candidate.kind,
        candidate_dir=str(Path(candidate.candidate_dir).resolve()),
        candidate_record_sha256=candidate.record_sha256,
        candidate_provenance_sha256=candidate.provenance_sha256,
        candidate_release_sha=candidate.release_sha,
        artifact_sha256=candidate.artifact_sha256(),
        publisher_release_sha=publisher_release_sha,
        production=production,
        expected_cumulative_sha256=expected_cumulative_sha256,
        nonce=nonce,
        authorized_by=authorized_by,
        authorized_at_utc=authorized_at_utc,
        expires_at_utc=expires_at_utc,
    )


def write_authorization(path: Path, authorization: PublicationAuthorization) -> None:
    _write_json_new(Path(path), authorization.as_dict())


def load_authorization(path: Path) -> PublicationAuthorization:
    return PublicationAuthorization.from_dict(_read_json(Path(path)))


# ---------------------------------------------------------------------------
# Authoritative manifest (new metadata; candidate bytes untouched)
# ---------------------------------------------------------------------------


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize exactly like :func:`build_night_artifacts`."""
    return (
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def candidate_chronology_defects(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    """Classify known candidate-manifest chronology defects without repairing them."""
    defects = []
    started = manifest.get("started_at_utc")
    finished = manifest.get("finished_at_utc")
    try:
        if _parse_utc(finished, "finished_at_utc") < _parse_utc(started, "started_at_utc"):
            defects.append("finished_at_utc_precedes_started_at_utc")
    except PublicationRefused:
        defects.append("chronology_unparseable")
    if finished == f"{manifest.get('date_utc')}T00:00:00+00:00":
        defects.append("finished_at_utc_is_night_start_request_placeholder")
    return tuple(defects)


def summary_source_manifest(
    candidate_manifest: Mapping[str, Any], candidate: NightCandidate
) -> Dict[str, Any]:
    """Return the manifest fields that feed authoritative cumulative products.

    The candidate's contradictory ``finished_at_utc`` never reaches authority:
    the trustworthy candidate completion time replaces it.  Nothing that the
    cumulative summary reads depends on the transaction, so the expected
    cumulative hashes can be planned (and authorized) before publication.
    """
    if candidate_manifest.get("date_utc") != candidate.date_utc:
        raise PublicationRefused("candidate_night_mismatch", "Manifest night differs.")
    _require_order(
        ("candidate query_request_started_at_utc", candidate_manifest.get("started_at_utc")),
        ("candidate_completed_at_utc", candidate.construction_completed_at_utc),
    )
    manifest = dict(candidate_manifest)
    manifest.update(
        {
            "authoritative": True,
            "publishable": True,
            "publication_authorized": True,
            "finished_at_utc": candidate.construction_completed_at_utc,
        }
    )
    return manifest


def build_authoritative_manifest(
    candidate_manifest_bytes: bytes,
    candidate: NightCandidate,
    authorization: PublicationAuthorization,
    *,
    transaction_id: str,
    publication_started_at_utc: str,
) -> bytes:
    """Derive the authoritative nightly manifest from the immutable candidate."""
    candidate_manifest = json.loads(candidate_manifest_bytes.decode("utf-8"))
    manifest = summary_source_manifest(candidate_manifest, candidate)
    _require_order(
        ("query_request_started_at_utc", candidate_manifest.get("started_at_utc")),
        ("candidate_completed_at_utc", candidate.construction_completed_at_utc),
        ("publication_authorized_at_utc", authorization.authorized_at_utc),
        ("publication_transaction_started_at_utc", publication_started_at_utc),
        ("authorization expires_at_utc", authorization.expires_at_utc),
    )
    original = {
        key: candidate_manifest.get(key)
        for key in ("started_at_utc", "ingested_at_utc", "finished_at_utc", "runtime_seconds")
    }
    manifest["authority"] = {
        "schema_version": AUTHORITY_SCHEMA,
        "state": "authoritative",
        "transaction_id": transaction_id,
        "authorization_sha256": authorization.digest,
        "authorized_by": authorization.authorized_by,
        "publisher_release_sha": authorization.publisher_release_sha,
        "commit_marker": {
            "kind": "publication-gate-removal",
            "gate_name": history.PUBLICATION_GATE_NAME,
            "committed_at_recorded_in": "publication-journal-and-terminal-record",
        },
        "predecessor": {
            "date_utc": authorization.predecessor_date_utc,
            "production_fingerprint": authorization.baseline_production_fingerprint,
        },
        "provenance_identities": {
            "scientific_artifact_contract": candidate_manifest.get(
                "offline_recovery_contract", candidate_manifest.get("schema_version")
            ),
            "candidate_manifest_schema": candidate_manifest.get("schema_version"),
            "candidate_execution_release_sha": candidate.release_sha,
            "publisher_release_sha": authorization.publisher_release_sha,
        },
        "candidate": {
            "kind": candidate.kind,
            "record_sha256": candidate.record_sha256,
            "release_sha": candidate.release_sha,
            "manifest_sha256": _sha256_bytes(candidate_manifest_bytes),
            "artifacts": {
                name: dict(candidate.artifacts[name]) for name in EXPECTED_ARTIFACTS
            },
            "provenance": dict(candidate.provenance),
        },
        "chronology": {
            "query_request_started_at_utc": candidate_manifest.get("started_at_utc"),
            "candidate_completed_at_utc": candidate.construction_completed_at_utc,
            "candidate_completed_at_utc_source": (
                "recovery_final_finished_at_utc"
                if candidate.kind == "offline-recovery"
                else "backfill_candidate_constructed_at_utc"
            ),
            "publication_authorized_at_utc": authorization.authorized_at_utc,
            "publication_transaction_started_at_utc": publication_started_at_utc,
            "candidate_manifest_original": original,
            "candidate_manifest_defects": list(candidate_chronology_defects(candidate_manifest)),
        },
    }
    return _manifest_bytes(manifest)


# ---------------------------------------------------------------------------
# Cumulative extension: planned in memory, staged, installed idempotently
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CumulativePlan:
    baseline_sha256: Mapping[str, Optional[str]]
    payloads: Mapping[str, bytes]
    expected_sha256: Mapping[str, str]
    modes: Mapping[str, int]
    prior_rows: Mapping[str, int]
    added_rows: Mapping[str, int]
    dtype_changes: Mapping[str, Any]
    arrow_schemas: Mapping[str, Any]
    schema_changes: Mapping[str, Any]

    def journal_view(self) -> Dict[str, Any]:
        return {
            "baseline_sha256": dict(self.baseline_sha256),
            "expected_sha256": dict(self.expected_sha256),
            "modes": {key: int(value) for key, value in self.modes.items()},
            "prior_rows": dict(self.prior_rows),
            "added_rows": dict(self.added_rows),
            "dtype_changes": dict(self.dtype_changes),
            "arrow_schemas": dict(self.arrow_schemas),
            "schema_changes": dict(self.schema_changes),
        }

    @property
    def schema_sha256(self) -> Dict[str, str]:
        return {
            key: _sha256_bytes(_canonical(self.arrow_schemas[key]))
            for key in _CUMULATIVE_KEYS
        }


def _arrow_schema(payload: bytes) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(io.BytesIO(payload))
    return [
        {"name": item.name, "type": str(item.type), "nullable": bool(item.nullable)}
        for item in schema
    ]


def _schema_change_allowed(before: str, after: str) -> bool:
    """Persisted type changes the canonical full rebuild itself produces."""
    return bool(
        before == after
        or before == "null"
        or (before.startswith("int") and after == "double")
    )


def _prior_rows_persisted_identically(
    before: bytes, after: bytes, column: str, night: str
) -> bool:
    """Arrow-level proof: every pre-existing row persists with identical values."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    old = pq.read_table(io.BytesIO(before))
    new = pq.read_table(io.BytesIO(after))
    return new.filter(pc.not_equal(new[column], night)).equals(old)


def _parquet_bytes(frame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def plan_cumulative_extension(
    data_root: Path,
    authoritative_manifest: Mapping[str, Any],
    loci_payload: bytes,
) -> CumulativePlan:
    """Plan an extension from one complete authority generation."""
    with history.authoritative_read(data_root):
        return _plan_cumulative_extension_locked(
            data_root, authoritative_manifest, loci_payload
        )


def _plan_cumulative_extension_locked(
    data_root: Path,
    authoritative_manifest: Mapping[str, Any],
    loci_payload: bytes,
) -> CumulativePlan:
    """Extend cumulative products by exactly one night and prove prior rows intact.

    The outer public function holds the shared/exclusive reentrant authority
    lock while these baseline bytes are hashed and parsed.
    """
    import pandas as pd

    night = authoritative_manifest["date_utc"]
    paths = history.cumulative_paths(data_root)
    baseline: Dict[str, Optional[str]] = {}
    baseline_payloads: Dict[str, Optional[bytes]] = {}
    modes: Dict[str, int] = {}
    for key in _CUMULATIVE_KEYS:
        path = paths[key]
        if path.exists() or path.is_symlink():
            payload = _read_regular(path)
            baseline[key] = _sha256_bytes(payload)
            baseline_payloads[key] = payload
            modes[key] = stat.S_IMODE(os.lstat(path).st_mode)
        else:
            baseline[key] = None
            baseline_payloads[key] = None
            modes[key] = 0o600
    old_index = (
        pd.read_parquet(io.BytesIO(baseline_payloads["loci_index"]))
        if baseline_payloads["loci_index"] is not None
        else pd.DataFrame(columns=history.CUMULATIVE_INDEX_COLUMNS)
    )
    old_summary = (
        pd.read_parquet(io.BytesIO(baseline_payloads["nightly_summary"]))
        if baseline_payloads["nightly_summary"] is not None
        else pd.DataFrame(
            columns=list(history._manifest_to_summary_row({"date_utc": None}).keys())
        )
    )
    if "night_date_utc" in old_index.columns and old_index["night_date_utc"].eq(night).any():
        raise PublicationRefused("cumulative_duplicate_night", "Loci index already holds the night.")
    if "date_utc" in old_summary.columns and old_summary["date_utc"].eq(night).any():
        raise PublicationRefused("cumulative_duplicate_night", "Summary already holds the night.")

    night_loci = pd.read_parquet(io.BytesIO(loci_payload))
    keep = [column for column in history.CUMULATIVE_INDEX_COLUMNS if column in night_loci.columns]
    added_index = night_loci[keep].copy()
    if len(old_index.columns) and list(old_index.columns) != keep:
        raise PublicationRefused(
            "cumulative_schema_drift", "Night loci columns differ from the cumulative index."
        )
    new_index = pd.concat([old_index, added_index], ignore_index=True, sort=False)
    subset = [c for c in ["night_date_utc", history.LOCUS_ID_COL] if c in new_index.columns]
    if subset:
        new_index = new_index.drop_duplicates(subset=subset, keep="last")
    if "night_mjd_min" in new_index.columns:
        new_index = new_index.sort_values(
            ["night_mjd_min", history.LOCUS_ID_COL], kind="mergesort"
        ).reset_index(drop=True)
    if len(new_index) != len(old_index) + len(added_index):
        raise PublicationRefused("cumulative_extension_not_exact", "Loci index lost or merged rows.")

    summary_row = history._manifest_to_summary_row(dict(authoritative_manifest))
    with warnings.catch_warnings():
        # pandas warns that all-null entries may later affect dtype inference;
        # the exact prior-row and persisted-schema checks below refuse drift.
        warnings.simplefilter("ignore", FutureWarning)
        new_summary = pd.concat(
            [old_summary, pd.DataFrame([summary_row])], ignore_index=True, sort=False
        )
    new_summary = new_summary.sort_values(["mjd_min", "date_utc"], kind="mergesort").reset_index(
        drop=True
    )

    dtype_changes = {}
    for label, old, new in (
        ("loci_index", old_index, new_index),
        ("nightly_summary", old_summary, new_summary),
    ):
        for column in old.columns:
            if len(old) and str(old[column].dtype) != str(new[column].dtype):
                dtype_changes[f"{label}.{column}"] = [
                    str(old[column].dtype), str(new[column].dtype)
                ]
    if int(new_index["night_date_utc"].eq(night).sum()) != int(
        authoritative_manifest["actual_loci"]
    ):
        raise PublicationRefused("cumulative_extension_not_exact", "Night row count differs.")

    payloads = {
        "loci_index": _parquet_bytes(new_index),
        "nightly_summary": _parquet_bytes(new_summary),
    }
    arrow_schemas: Dict[str, Any] = {}
    schema_changes: Dict[str, Any] = {}
    frames = {
        "loci_index": (old_index, new_index, "night_date_utc"),
        "nightly_summary": (old_summary, new_summary, "date_utc"),
    }
    for key in _CUMULATIVE_KEYS:
        after = _arrow_schema(payloads[key])
        before = (
            _arrow_schema(baseline_payloads[key])
            if baseline_payloads[key] is not None
            else None
        )
        arrow_schemas[key] = {"before": before, "after": after}
        if before is None:
            continue
        if [item["name"] for item in before] != [item["name"] for item in after]:
            raise PublicationRefused(
                "cumulative_schema_drift", f"{key} column order or set would change."
            )
        for old_field, new_field in zip(before, after):
            if old_field != new_field:
                if not _schema_change_allowed(old_field["type"], new_field["type"]):
                    raise PublicationRefused(
                        "cumulative_schema_drift",
                        f"{key}.{old_field['name']} would change persisted type "
                        f"{old_field['type']} -> {new_field['type']}.",
                    )
                schema_changes[f"{key}.{old_field['name']}"] = [old_field, new_field]
        old, new, column = frames[key]
        if not len(old):
            continue
        # Prior rows must keep exact values and order.  With an unchanged
        # persisted schema the Arrow comparison is decisive (it is exact for
        # nested struct/list columns); an allowed widening falls back to a
        # value comparison, and anything uncomparable fails closed.
        if before == after:
            exact = _prior_rows_persisted_identically(
                baseline_payloads[key], payloads[key], column, night
            )
        else:
            try:
                pd.testing.assert_frame_equal(
                    new[new[column] != night].reset_index(drop=True),
                    old.reset_index(drop=True),
                    check_exact=True,
                    check_dtype=False,
                )
                exact = True
            except (AssertionError, TypeError, ValueError):
                exact = False
        if not exact:
            raise PublicationRefused(
                "cumulative_extension_not_exact", f"Prior {key} rows would change."
            )
    return CumulativePlan(
        baseline_sha256=baseline,
        payloads=payloads,
        expected_sha256={key: _sha256_bytes(payloads[key]) for key in _CUMULATIVE_KEYS},
        modes=modes,
        prior_rows={"loci_index": len(old_index), "nightly_summary": len(old_summary)},
        added_rows={"loci_index": len(added_index), "nightly_summary": 1},
        dtype_changes=dtype_changes,
        arrow_schemas=arrow_schemas,
        schema_changes=schema_changes,
    )


# ---------------------------------------------------------------------------
# Durable publication gate: crash/recovery evidence, never the NFS mutex
# ---------------------------------------------------------------------------

_GATE_FIELDS = frozenset(
    {
        "schema_version",
        "state",
        "transaction_id",
        "date_utc",
        "predecessor_date_utc",
        "authorization_sha256",
        "authoritative_manifest_sha256",
        "baseline_cumulative_sha256",
        "expected_cumulative_sha256",
        "publisher_release_sha",
        "created_at_utc",
    }
)


def read_publication_gate(data_root: Path) -> Optional[Dict[str, Any]]:
    """Return the unresolved-transition marker, ``None``, or refuse if unsafe."""
    gate = history.publication_gate_path(data_root)
    if not gate.exists() and not gate.is_symlink():
        return None
    if gate.is_symlink() or not gate.is_file():
        raise PublicationRefused("authority_contradiction", "Publication gate is unsafe.")
    try:
        document = _read_json(gate)
    except PublicationRefused as exc:
        raise PublicationRefused(
            "authority_contradiction", "Publication gate is unreadable."
        ) from exc
    if (
        set(document) != _GATE_FIELDS
        or document.get("schema_version") != GATE_SCHEMA
        or document.get("state") != "authority_transition_in_progress"
        or not isinstance(document.get("transaction_id"), str)
        or TRANSACTION_ID_PATTERN.fullmatch(document["transaction_id"]) is None
    ):
        raise PublicationRefused("authority_contradiction", "Publication gate is malformed.")
    return document


def _gate_names(data_root: Path, transaction_id: str) -> bool:
    """Evidence-based: does the durable gate exist for this transaction?"""
    try:
        gate = read_publication_gate(data_root)
    except PublicationRefused:
        return False
    return gate is not None and gate.get("transaction_id") == transaction_id


# ---------------------------------------------------------------------------
# Production-wide publication exclusion
# ---------------------------------------------------------------------------

class PublicationAuthorityLock:
    """Own the universal exclusive authority lock, preflight to evidence.

    ``flock`` is released by the kernel when the owner dies, so a crash never
    strands exclusion; the durable gate and journal then force any new owner to
    resume that exact transaction or refuse.  Canonical readers acquire the
    same object in shared mode for their complete logical operation.
    """

    def __init__(
        self,
        capability: PublicationWriteCapability,
        *,
        wait_seconds: float = 0.0,
        poll_seconds: float = 0.05,
    ) -> None:
        self.capability = capability
        try:
            self.path = contained_path(
                capability.lock_root, Path(PUBLICATION_LOCK_NAME)
            )
        except StorageContractError as exc:
            raise PublicationRefused(
                "path_contradiction",
                f"Publication authority lock is unsafe: {exc}",
            ) from exc
        self.wait_seconds = float(wait_seconds)
        self.poll_seconds = float(poll_seconds)
        self._context: Any = None
        self._lease: Any = None

    def __enter__(self) -> "PublicationAuthorityLock":
        try:
            _ensure_private_tree(self.path.parent, self.capability.root)
        except (StorageContractError, WriterError) as exc:
            raise PublicationRefused(
                "path_contradiction",
                f"Publication authority lock is unsafe: {exc}",
            ) from exc
        self._context = exclusive_authority_lock(
            self.capability.published_root,
            lock_path=self.path,
            wait_seconds=self.wait_seconds,
            poll_seconds=self.poll_seconds,
            create=True,
        )
        try:
            self._lease = self._context.__enter__()
        except AuthorityLockUnavailable as exc:
            raise PublicationRefused(
                "publication_lock_busy",
                "A reader or another publisher owns the production authority lock.",
            ) from exc
        except AuthorityLockError as exc:
            raise PublicationRefused(
                "path_contradiction", f"Publication authority lock is unsafe: {exc}"
            ) from exc
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._context is None:
            return
        context = self._context
        self._context = None
        self._lease = None
        context.__exit__(*exc)


# ---------------------------------------------------------------------------
# Path re-derivation (journal paths are recorded, never trusted)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionPaths:
    journal: Path
    target_relative: Path
    target: Path
    stage: Path
    stage_parent: Path
    cumulative_stage: Path
    cumulative_staged: Mapping[str, Path]
    cumulative_targets: Mapping[str, Path]
    writer_lock: Path
    reconciliation_lock: Path
    gate: Path
    staged_gate: Path


def transaction_paths(
    capability: PublicationWriteCapability, date_utc: str, transaction_id: str
) -> TransactionPaths:
    """Re-derive every transaction path from trusted capability roots and identity."""
    try:
        return _transaction_paths(capability, date_utc, transaction_id)
    except StorageContractError as exc:
        raise PublicationRefused("path_contradiction", str(exc)) from exc


def _transaction_paths(
    capability: PublicationWriteCapability, date_utc: str, transaction_id: str
) -> TransactionPaths:
    match = TRANSACTION_ID_PATTERN.fullmatch(transaction_id)
    if match is None or match.group(1) != date_utc:
        raise PublicationRefused("path_contradiction", "Transaction identity is malformed.")
    target_relative = nightly_target_relative(date_utc)
    writer_name = lock_identity(target_relative.as_posix())
    stage_parent = contained_path(capability.staging_root, Path(transaction_id))
    cumulative_stage = contained_path(capability.staging_root, Path(f"{transaction_id}-cumulative"))
    cumulative = history.cumulative_paths(capability.published_root)
    return TransactionPaths(
        journal=contained_path(capability.journal_root, Path(f"{transaction_id}.json")),
        target_relative=target_relative,
        target=contained_path(capability.published_root, target_relative),
        stage=contained_path(
            capability.staging_root, Path(transaction_id) / writer_name.removesuffix(".lock")
        ),
        stage_parent=stage_parent,
        cumulative_stage=cumulative_stage,
        cumulative_staged={
            key: cumulative_stage / f"{key}.parquet" for key in _CUMULATIVE_KEYS
        },
        cumulative_targets={key: Path(cumulative[key]) for key in _CUMULATIVE_KEYS},
        writer_lock=contained_path(capability.lock_root, Path(writer_name)),
        reconciliation_lock=contained_path(
            capability.lock_root, Path(lock_identity(SHARED_RECONCILIATION_LOCK_IDENTITY))
        ),
        gate=Path(history.publication_gate_path(capability.published_root)),
        staged_gate=contained_path(capability.staging_root, Path(f"{transaction_id}.gate.json")),
    )


def _relative_views(capability: PublicationWriteCapability, paths: TransactionPaths) -> Dict[str, Any]:
    staging = capability.staging_root
    published = capability.published_root
    return {
        "staged_relative": {
            key: paths.cumulative_staged[key].relative_to(staging).as_posix()
            for key in _CUMULATIVE_KEYS
        },
        "target_relative": {
            key: paths.cumulative_targets[key].relative_to(published).as_posix()
            for key in _CUMULATIVE_KEYS
        },
        "gate_relative": paths.gate.relative_to(published).as_posix(),
    }


# ---------------------------------------------------------------------------
# Durable-evidence classification (shared by publisher, controller and CLI)
# ---------------------------------------------------------------------------


def _metadata(journal: TransactionJournal) -> Mapping[str, Any]:
    return journal.snapshot.descriptor.metadata


def _is_publication_journal(journal: TransactionJournal) -> bool:
    return journal.snapshot.descriptor.operation == PUBLICATION_OPERATION


def _unfinished(journal: TransactionJournal) -> bool:
    return journal.snapshot.outcome not in {
        JournalOutcome.COMPLETE,
        JournalOutcome.UNPUBLISHED_FAILURE,
    }


def load_transaction_journals(journal_root: Path) -> Tuple[TransactionJournal, ...]:
    """Load every transaction journal of every class; refuse on any corruption."""
    root = Path(journal_root)
    if not root.exists() and not root.is_symlink():
        return ()
    if root.is_symlink() or not root.is_dir():
        raise PublicationRefused("journal_contradiction", "Journal root is unsafe.")
    journals = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue  # atomic-write temporaries; never authoritative
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            raise PublicationRefused(
                "journal_contradiction", f"Unexpected journal-root entry: {path.name}."
            )
        try:
            journals.append(TransactionJournal.load(path))
        except JournalError as exc:
            raise PublicationRefused(
                "journal_contradiction", f"Journal {path.name} is unreadable: {exc}"
            ) from exc
    return tuple(journals)


def _physical_nights(data_root: Path) -> Tuple[str, ...]:
    """Every discoverable nightly manifest, irrespective of the gate (internal)."""
    nights = []
    root = history.survey_data_root(data_root) / "nightly"
    for path in history._manifest_paths(data_root):
        nights.append("-".join(path.parent.relative_to(root).parts))
    return tuple(sorted(nights))


def authoritative_nights(data_root: Path) -> Tuple[str, ...]:
    """Committed authoritative nights; refuses while a transition is unresolved."""
    with history.authoritative_read(data_root):
        return _physical_nights(data_root)


def classify_night_authority(
    data_root: Path, journal_root: Path, date_utc: str
) -> Dict[str, Any]:
    """Classify durable evidence while holding the shared authority lock."""
    with history.authority_read_lock(data_root):
        return _classify_night_authority_locked(data_root, journal_root, date_utc)


def _classify_night_authority_locked(
    data_root: Path, journal_root: Path, date_utc: str
) -> Dict[str, Any]:
    """Classify one night's authority purely from durable evidence.

    Precedence: an unresolved gate, then publication journals, then files.
    Manifest existence alone never yields ``COMPLETE`` for a V3 publication.
    """
    gate = read_publication_gate(data_root)
    journals = [
        journal
        for journal in load_transaction_journals(journal_root)
        if _is_publication_journal(journal)
        and _metadata(journal).get("target_utc_night") == date_utc
    ]
    target = Path(data_root) / nightly_target_relative(date_utc)
    manifest_path = target / "manifest.json"
    if manifest_path.is_symlink() or target.is_symlink():
        manifest_sha = "unsafe"
    else:
        manifest_sha = _file_sha256(manifest_path) if manifest_path.is_file() else None
    result: Dict[str, Any] = {
        "date_utc": date_utc,
        "gate_present": gate is not None,
        "gate_date_utc": gate.get("date_utc") if gate else None,
        "gate_transaction_id": gate.get("transaction_id") if gate else None,
        "manifest_present": manifest_sha is not None,
        "transaction_id": None,
        "authorization_sha256": None,
        "finalized": False,
        "legacy": False,
        "pending_pre_gate": False,
        "resulting_production_fingerprint": None,
    }

    def done(state: AuthorityState, journal: Optional[TransactionJournal] = None, **extra):
        if journal is not None:
            result["transaction_id"] = journal.snapshot.descriptor.run_id
            result["authorization_sha256"] = _metadata(journal).get("authorization_sha256")
            result["journal_outcome"] = journal.snapshot.outcome.value
        result.update(extra)
        result["state"] = state.value
        return result

    if gate is not None and gate.get("date_utc") == date_utc:
        owner = [
            journal for journal in journals
            if journal.snapshot.descriptor.run_id == gate["transaction_id"]
        ]
        if len(owner) != 1:
            return done(AuthorityState.CONTRADICTION, reason="gate_without_journal")
        return done(AuthorityState.RECONCILIATION_REQUIRED, owner[0])
    complete = [j for j in journals if j.snapshot.outcome == JournalOutcome.COMPLETE]
    committed = [j for j in journals if _unfinished(j) and j.snapshot.published]
    pending = [j for j in journals if _unfinished(j) and not j.snapshot.published]
    if len(complete) > 1 or (complete and committed):
        return done(AuthorityState.CONTRADICTION, reason="multiple_authorities")
    if complete:
        journal = complete[0]
        if manifest_sha != _metadata(journal).get("authoritative_manifest_sha256"):
            return done(AuthorityState.CONTRADICTION, journal, reason="manifest_differs_from_journal")
        return done(
            AuthorityState.COMPLETE,
            journal,
            finalized=True,
            resulting_production_fingerprint=journal.snapshot.reconciliation.get(
                "resulting_production_fingerprint"
            ),
        )
    if committed:
        journal = committed[-1]
        expected = _metadata(journal).get("cumulative", {}).get("expected_sha256", {})
        cumulative = history.cumulative_paths(data_root)
        if (
            gate is None
            and manifest_sha == _metadata(journal).get("authoritative_manifest_sha256")
            and all(
                _optional_sha256(Path(cumulative[key])) == expected.get(key)
                for key in _CUMULATIVE_KEYS
            )
        ):
            # Crash after the gate was removed but before journal COMPLETE.
            return done(AuthorityState.COMPLETE, journal, finalized=False)
        return done(AuthorityState.CONTRADICTION, journal, reason="transition_without_gate")
    if manifest_sha is not None:
        if manifest_sha == "unsafe":
            return done(AuthorityState.CONTRADICTION, reason="unsafe_partition")
        manifest = _read_json(manifest_path)
        if "authority" in manifest:
            return done(AuthorityState.CONTRADICTION, reason="authority_without_journal")
        return done(AuthorityState.COMPLETE, finalized=True, legacy=True)
    if target.exists() or target.is_symlink():
        return done(AuthorityState.CONTRADICTION, reason="reserved_target_without_gate")
    return done(
        AuthorityState.NOT_COMMITTED,
        pending[-1] if pending else None,
        pending_pre_gate=bool(pending),
    )


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationOutcome:
    status: str
    success: bool
    record: Mapping[str, Any]
    record_path: Optional[Path]

    @property
    def authority_state(self) -> Optional[str]:
        return self.record.get("authority_state")

    @property
    def retryable(self) -> bool:
        return bool(self.record.get("retryable"))


class NightPublisher:
    """Ordered single-writer publisher for validated nightly candidates."""

    def __init__(
        self,
        capability: PublicationWriteCapability,
        *,
        publisher_release_sha: str,
        cache_root: Path,
        mountinfo_lines: Optional[Sequence[str]] = None,
        clock: Callable[[], datetime] = _utc_now,
        fault_hook: Optional[FaultHook] = None,
        lock_wait_seconds: float = 0.0,
    ) -> None:
        if type(capability) not in {
            SyntheticWriteCapability,
            ProductionPublicationCapability,
        }:
            raise ProductionAuthorizationUnavailable(
                "Publisher requires an exact sealed publication capability."
            )
        if not _is_sha40(publisher_release_sha):
            raise PublicationError("Publisher release identity must be a full SHA.")
        if (
            type(capability) is ProductionPublicationCapability
            and publisher_release_sha != capability.publisher_release_sha
        ):
            raise ProductionAuthorizationUnavailable(
                "Production capability is bound to another publisher release."
            )
        self.capability = capability
        self.data_root = capability.published_root
        self.cache_root = Path(cache_root)
        self.publisher_release_sha = publisher_release_sha
        self.mountinfo_lines = mountinfo_lines
        self.clock = clock
        self.fault_hook = fault_hook
        self.lock_wait_seconds = lock_wait_seconds

    # -- observation -------------------------------------------------------

    def _checkpoint(self, point: str, **details: Any) -> None:
        if self.fault_hook is not None:
            self.fault_hook(point, details)

    def sentinel(self, date_utc: str) -> Mapping[str, Any]:
        from .commissioning import capture_production_sentinel

        return capture_production_sentinel(
            self.data_root, self.cache_root, date_utc, mountinfo_lines=self.mountinfo_lines
        )

    def _journals(self) -> Tuple[TransactionJournal, ...]:
        return load_transaction_journals(self.capability.journal_root)

    def _paths(self, date_utc: str, transaction_id: str) -> TransactionPaths:
        return transaction_paths(self.capability, date_utc, transaction_id)

    def authority_state(self, date_utc: str) -> Dict[str, Any]:
        return classify_night_authority(
            self.data_root, self.capability.journal_root, date_utc
        )

    def pending_authorization(self, date_utc: str) -> Optional[PublicationAuthorization]:
        """The authorization of the unresolved transaction for this night, if any."""
        state = self.authority_state(date_utc)
        if state["transaction_id"] is None or (
            state["state"] == AuthorityState.COMPLETE.value and state["finalized"]
        ):
            return None
        for journal in self._journals():
            if journal.snapshot.descriptor.run_id == state["transaction_id"]:
                authorization = PublicationAuthorization.from_dict(
                    _metadata(journal)["authorization"]
                )
                if authorization.digest != _metadata(journal).get("authorization_sha256"):
                    raise PublicationRefused(
                        "journal_contradiction", "Journal authorization digest differs."
                    )
                return authorization
        return None

    def _residue(self, own_transaction: Optional[str] = None) -> Tuple[str, ...]:
        allowed = set()
        if own_transaction is not None:
            allowed = {
                own_transaction,
                f"{own_transaction}-cumulative",
                f"{own_transaction}.gate.json",
            }
        found = []
        staging = self.capability.staging_root
        if staging.exists():
            found.extend(
                f"staging/{path.name}"
                for path in sorted(staging.iterdir())
                if path.name not in allowed
            )
        locks = self.capability.lock_root
        if locks.exists():
            for path in sorted(locks.iterdir()):
                if path.name == PUBLICATION_LOCK_NAME:
                    continue
                if own_transaction is not None and _lock_transaction(path) in {
                    own_transaction, f"{own_transaction}:reconciliation"
                }:
                    continue
                found.append(f"locks/{path.name}")
        return tuple(found)

    def authorization_inputs(self, candidate: NightCandidate) -> Dict[str, Any]:
        """Read-only: the Sentinel binding and cumulative plan Control must authorize."""
        if read_publication_gate(self.data_root) is not None:
            raise PublicationRefused(
                "authority_transition_pending", "A publication transition is unresolved."
            )
        payloads = {name: _read_regular(candidate.candidate_dir / name) for name in EXPECTED_ARTIFACTS}
        if {name: _sha256_bytes(payloads[name]) for name in EXPECTED_ARTIFACTS} != candidate.artifact_sha256():
            raise PublicationRefused("candidate_hash_mismatch", "Candidate bytes changed.")
        manifest = summary_source_manifest(
            json.loads(payloads["manifest.json"].decode("utf-8")), candidate
        )
        with history.authoritative_read(self.data_root):
            nights = _physical_nights(self.data_root)
            plan = plan_cumulative_extension(self.data_root, manifest, payloads["loci.parquet"])
            sentinel = self.sentinel(candidate.date_utc)
        production = production_binding_from_sentinel(sentinel)
        if production["cumulative_sha256"] != plan.baseline_sha256:
            raise PublicationRefused("sentinel_drift", "Production changed while planning.")
        return {
            "sentinel": sentinel,
            "production": production,
            "predecessor_date_utc": nights[-1] if nights else None,
            "expected_cumulative_sha256": dict(plan.expected_sha256),
            "cumulative_plan": plan.journal_view(),
        }

    # -- public API --------------------------------------------------------

    def publish(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
    ) -> PublicationOutcome:
        started_clock = self.clock()
        try:
            with PublicationAuthorityLock(
                self.capability, wait_seconds=self.lock_wait_seconds
            ):
                self._verify_authorization(candidate, authorization)
                return self._publish_locked(candidate, authorization, started_clock)
        except PublicationRefused as refusal:
            return self._refused(candidate, authorization, refusal)
        except Exception as error:
            return self._unexpected(candidate, authorization, error)

    # -- preconditions -----------------------------------------------------

    def _verify_authorization(
        self, candidate: NightCandidate, authorization: PublicationAuthorization
    ) -> None:
        if not isinstance(authorization, PublicationAuthorization):
            raise PublicationRefused("authorization_absent", "Explicit authorization is required.")
        if not candidate.validation_passed:
            raise PublicationRefused("candidate_validation_failed", "Candidate validation did not pass.")
        if candidate.authoritative:
            raise PublicationRefused("candidate_already_authoritative", "Candidate is authoritative.")
        checks = (
            (authorization.date_utc == candidate.date_utc, "authorization_night_mismatch"),
            (authorization.candidate_kind == candidate.kind, "authorization_candidate_mismatch"),
            (authorization.candidate_record_sha256 == candidate.record_sha256, "authorization_candidate_mismatch"),
            (authorization.candidate_dir == str(Path(candidate.candidate_dir).resolve()), "authorization_candidate_mismatch"),
            (authorization.candidate_provenance_sha256 == candidate.provenance_sha256, "authorization_candidate_mismatch"),
            (authorization.candidate_release_sha == candidate.release_sha, "authorization_release_mismatch"),
            (authorization.publisher_release_sha == self.publisher_release_sha, "authorization_release_mismatch"),
            (dict(authorization.artifact_sha256) == candidate.artifact_sha256(), "candidate_hash_mismatch"),
            (authorization.production["canonical_root"] == str(self.data_root), "production_root_mismatch"),
            (
                authorization.production["predicates"]["cache_path"] == str(self.cache_root),
                "authorization_binding_mismatch",
            ),
        )
        for passed, code in checks:
            if not passed:
                raise PublicationRefused(code, "Authorization does not bind this candidate.")
        # Re-hash on disk: candidate bytes must be exactly what was authorized.
        observed = _verify_candidate_artifacts(candidate.candidate_dir, candidate.artifacts)
        if {name: observed[name]["sha256"] for name in EXPECTED_ARTIFACTS} != dict(
            authorization.artifact_sha256
        ):
            raise PublicationRefused("candidate_hash_mismatch", "Candidate bytes changed.")
        if _file_sha256(candidate.record_path) != candidate.record_sha256:
            raise PublicationRefused("candidate_hash_mismatch", "Candidate record changed.")
        if type(self.capability) is ProductionPublicationCapability:
            binding = ProductionPublicationBinding(**dict(self.capability.binding))
            if (
                binding.digest != self.capability.binding_sha256
                or authorization.digest != self.capability.authorization_sha256
                or binding.authorization_sha256 != authorization.digest
                or binding.publisher_release_sha != self.publisher_release_sha
                or candidate.provenance.get("binding_sha256")
                != binding.candidate_binding_sha256
                or candidate.provenance_sha256
                != binding.candidate_provenance_sha256
                or production_authority_lock_identity()
                != dict(binding.authority_lock)
            ):
                raise PublicationRefused(
                    "authorization_binding_mismatch",
                    "Production capability no longer matches its exact binding.",
                )

    def _qualify_production(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        *,
        own_transaction: Optional[str] = None,
        baseline: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Fresh full Sentinel V2 qualification against the authorized binding."""
        if read_publication_gate(self.data_root) is not None:
            raise PublicationRefused(
                "authority_transition_pending", "A publication transition is unresolved."
            )
        nights = _physical_nights(self.data_root)
        if candidate.date_utc in nights:
            raise PublicationRefused("already_authoritative", "The night is already published.")
        latest = nights[-1] if nights else None
        if latest != authorization.predecessor_date_utc:
            raise PublicationRefused(
                "predecessor_gap",
                f"Latest authoritative night is {latest}; expected "
                f"{authorization.predecessor_date_utc}.",
            )
        residue = self._residue(own_transaction)
        if residue:
            raise PublicationRefused("transaction_residue", f"Residue present: {residue}.")
        from .commissioning import CommissioningError

        try:
            sentinel = self.sentinel(candidate.date_utc)
        except CommissioningError as exc:
            raise PublicationRefused(
                "mount_binding_drift" if "mount" in str(exc).lower() else "sentinel_drift",
                f"Fresh Sentinel V2 qualification failed: {exc}",
            ) from exc
        observed = production_binding_from_sentinel(sentinel)
        expected = authorization.production
        predicates = observed["predicates"]
        if observed["canonical_root"] != expected["canonical_root"]:
            raise PublicationRefused("production_root_mismatch", "Production root differs.")
        if observed["mount_binding"] != expected["mount_binding"]:
            raise PublicationRefused("mount_binding_drift", "Production mount binding differs.")
        if predicates["cache_absent"] is not True:
            raise PublicationRefused("cache_present", "Production cache predicate failed.")
        if predicates["transaction_artifacts"]:
            raise PublicationRefused(
                "transaction_residue",
                f"Production transaction artifacts: {predicates['transaction_artifacts']}.",
            )
        if predicates["target_absent"] is not True:
            raise PublicationRefused("already_authoritative", "Production target exists.")
        if predicates != expected["predicates"]:
            raise PublicationRefused("authorization_binding_mismatch", "Sentinel predicates differ.")
        for key in ("durable_fingerprint_sha256", "manifest_count", "cumulative_sha256"):
            if observed[key] != expected[key]:
                raise PublicationRefused(
                    "sentinel_drift", f"Production {key} differs from the authorized baseline."
                )
        if baseline is not None and (
            sentinel["durable_fingerprint_sha256"] != baseline["durable_fingerprint_sha256"]
        ):
            raise PublicationRefused("sentinel_drift", "Production changed during staging.")
        runtime = sentinel["runtime_observation"]
        root_device = runtime["directory_devices"]["data_root"]["device"]
        devices = {row["device"] for row in runtime["directory_devices"].values()}
        devices |= {row["device"] for row in runtime["manifest_devices"]}
        devices |= {row["device"] for row in runtime["durable_file_devices"]}
        if devices != {root_device} or (
            os.stat(self.capability.staging_root).st_dev != root_device
        ):
            raise PublicationRefused(
                "runtime_linkage_mismatch",
                "Production, its contents and staging are not one live filesystem.",
            )
        return sentinel

    # -- dispatcher ----------------------------------------------------------

    def _publish_locked(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        started_clock: datetime,
    ) -> PublicationOutcome:
        journals = self._journals()
        gate = read_publication_gate(self.data_root)
        if gate is not None:
            if gate["authorization_sha256"] != authorization.digest:
                raise PublicationRefused(
                    "authority_transition_pending",
                    f"Transaction {gate['transaction_id']} for {gate['date_utc']} "
                    "must be resumed with its own authorization first.",
                )
            owner = [j for j in journals if j.snapshot.descriptor.run_id == gate["transaction_id"]]
            if len(owner) != 1:
                raise PublicationRefused("authority_contradiction", "Gate has no journal.")
            return self._resume(candidate, authorization, owner[0], gate_present=True)
        own = [j for j in journals if _metadata(j).get("authorization_sha256") == authorization.digest]
        blocking = [
            j for j in journals
            if j not in own
            and _unfinished(j)
            and not (_is_publication_journal(j) and not j.snapshot.published)
        ]
        if blocking:
            raise PublicationRefused(
                "transaction_pending",
                f"Unfinished transaction {blocking[0].snapshot.descriptor.run_id} "
                f"({blocking[0].snapshot.outcome.value}) affects production authority.",
            )
        complete = [j for j in own if j.snapshot.outcome == JournalOutcome.COMPLETE]
        if complete:
            return self._replay(candidate, authorization, complete[-1])
        committed = [j for j in own if _unfinished(j) and j.snapshot.published]
        if committed:
            return self._resume(candidate, authorization, committed[-1], gate_present=False)
        for journal in journals:
            if _is_publication_journal(journal) and _unfinished(journal):
                self._abandon(journal)  # pre-gate only: provably NOT_COMMITTED
        self._sweep_terminal_residue(self._journals())
        return self._publish_new(candidate, authorization, self._journals(), started_clock)

    # -- NOT_COMMITTED phase -------------------------------------------------

    def _transaction_id(
        self, candidate: NightCandidate, authorization: PublicationAuthorization,
        journals: Sequence[TransactionJournal],
    ) -> str:
        prefix = f"v3pub-{candidate.date_utc}-{authorization.digest[:12]}-a"
        attempts = [
            int(j.snapshot.descriptor.run_id[len(prefix):])
            for j in journals if j.snapshot.descriptor.run_id.startswith(prefix)
        ]
        return f"{prefix}{max(attempts, default=0) + 1}"

    def _publish_new(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journals: Sequence[TransactionJournal],
        started_clock: datetime,
    ) -> PublicationOutcome:
        from .science import reopen_and_validate_artifacts

        transaction_started = _iso(started_clock)
        if _parse_utc(transaction_started, "now") >= _parse_utc(
            authorization.expires_at_utc, "expires_at_utc"
        ):
            raise PublicationRefused("authorization_expired", "Authorization has expired.")
        before = self._qualify_production(candidate, authorization)
        for journal in journals:
            other = _metadata(journal).get("authorization")
            if (
                isinstance(other, Mapping)
                and other.get("nonce") == authorization.nonce
                and _metadata(journal).get("authorization_sha256") != authorization.digest
            ):
                raise PublicationRefused(
                    "authorization_nonce_reused", "The authorization nonce was already used."
                )
        transaction_id = self._transaction_id(candidate, authorization, journals)
        paths = self._paths(candidate.date_utc, transaction_id)
        payloads = {name: _read_regular(candidate.candidate_dir / name) for name in EXPECTED_ARTIFACTS}
        if {name: _sha256_bytes(payloads[name]) for name in EXPECTED_ARTIFACTS} != dict(
            authorization.artifact_sha256
        ):
            raise PublicationRefused("candidate_hash_mismatch", "Candidate bytes changed.")
        manifest_bytes = build_authoritative_manifest(
            payloads["manifest.json"], candidate, authorization,
            transaction_id=transaction_id, publication_started_at_utc=transaction_started,
        )
        artifacts = {
            "loci.parquet": payloads["loci.parquet"],
            "alerts.parquet": payloads["alerts.parquet"],
            "manifest.json": manifest_bytes,
        }
        authoritative_manifest = dict(reopen_and_validate_artifacts(artifacts).manifest)
        plan = plan_cumulative_extension(
            self.data_root, authoritative_manifest, payloads["loci.parquet"]
        )
        if dict(plan.baseline_sha256) != authorization.production["cumulative_sha256"]:
            raise PublicationRefused("sentinel_drift", "Cumulative baseline changed.")
        if dict(plan.expected_sha256) != dict(authorization.expected_cumulative_sha256):
            raise PublicationRefused(
                "cumulative_plan_mismatch",
                "The cumulative extension differs from the authorized hash plan.",
            )
        if type(self.capability) is ProductionPublicationCapability:
            binding = ProductionPublicationBinding(**dict(self.capability.binding))
            mismatches = qualify_production_binding(
                binding,
                authorization,
                hostname=socket.getfqdn(),
                uid=os.geteuid(),
                sentinel=before,
                now=self.clock(),
                candidate=candidate,
                authority_lock=production_authority_lock_identity(),
            )
            if mismatches:
                raise PublicationRefused(
                    "authorization_binding_mismatch",
                    "Production capability qualification changed: "
                    + ",".join(mismatches),
                )
            if plan.schema_sha256 != dict(
                binding.expected_cumulative_schema_sha256
            ):
                raise PublicationRefused(
                    "cumulative_plan_mismatch",
                    "The cumulative schema plan differs from the Control binding.",
                )
        _ensure_private_tree(self.capability.journal_root, self.capability.root)
        _ensure_private_tree(self.capability.staging_root, self.capability.root)
        self._checkpoint("before_transaction_reservation", transaction_id=transaction_id)
        journal = TransactionJournal.create(
            paths.journal,
            TransactionDescriptor(
                run_id=transaction_id,
                operation=PUBLICATION_OPERATION,
                target_identity=paths.target_relative.as_posix(),
                target_path=str(paths.target),
                stage_path=str(paths.stage),
                lock_path=str(paths.writer_lock),
                profile=f"v3:{self.capability.environment}",
                plan_id=authorization.digest,
                release_sha=self.publisher_release_sha,
                metadata={
                    "schema_version": DESCRIPTOR_SCHEMA,
                    "target_utc_night": candidate.date_utc,
                    "authorization": authorization.as_dict(),
                    "authorization_sha256": authorization.digest,
                    "production_capability_binding_sha256": (
                        self.capability.binding_sha256
                        if type(self.capability) is ProductionPublicationCapability
                        else None
                    ),
                    "control_token_sha256": (
                        self.capability.control_token_sha256
                        if type(self.capability) is ProductionPublicationCapability
                        else None
                    ),
                    "candidate_kind": candidate.kind,
                    "candidate_dir": authorization.candidate_dir,
                    "candidate_record_sha256": candidate.record_sha256,
                    "authoritative_manifest_sha256": _sha256_bytes(manifest_bytes),
                    "baseline_production_fingerprint": before["durable_fingerprint_sha256"],
                    "baseline_manifest_count": before["durable_state"]["manifest_count"],
                    "cumulative": {**plan.journal_view(), **_relative_views(self.capability, paths)},
                    "chronology": {
                        "query_request_started_at_utc": authoritative_manifest["authority"][
                            "chronology"
                        ]["query_request_started_at_utc"],
                        "candidate_completed_at_utc": candidate.construction_completed_at_utc,
                        "publication_authorized_at_utc": authorization.authorized_at_utc,
                        "publication_transaction_started_at_utc": transaction_started,
                    },
                },
            ),
            at=started_clock,
        )
        writer_lock = WriterLock(
            self.capability, paths.target_relative.as_posix(), transaction_id,
            transaction_id=transaction_id, release_sha=self.publisher_release_sha,
        )
        transaction: Optional[PublicationTransaction] = None
        try:
            journal.transition(
                ExecutionState.PRECHECKED,
                validation={
                    "passed": True,
                    "baseline_fingerprint": before["durable_fingerprint_sha256"],
                    "predecessor": authorization.predecessor_date_utc,
                },
            )
            writer_lock.acquire(at=self.clock())
            journal.transition(ExecutionState.LOCKED, publication={"writer_lock_acquired": True})
            transaction = PublicationTransaction(
                self.capability, writer_lock, paths.target_relative, transaction_id,
                publication_event_hook=lambda event, details: self._checkpoint(event, **dict(details)),
            )
            transaction.prepare()
            sealed = {"sealed_acquisition_reused": True, "query_invoked": False, "fetch_invoked": False}
            journal.transition(ExecutionState.QUERYING, validation=sealed)
            transaction.begin_query()
            journal.transition(ExecutionState.FETCHING, validation=sealed)
            transaction.begin_fetch()
            self._checkpoint("after_transaction_reservation", transaction_id=transaction_id)
            evidence_block = authoritative_manifest["query_fetch_evidence"]
            evidence = QueryFetchEvidence(
                query_completed=evidence_block["query_completed"] is True,
                fetch_completed=evidence_block["fetch_completed"] is True,
                loci_rows=int(authoritative_manifest["actual_loci"]),
                alert_rows=int(authoritative_manifest["alert_rows"]),
                zero_row_proof=evidence_block.get("zero_row_proof"),
            )
            self._checkpoint("before_staging")
            transaction.stage_artifacts(artifacts, evidence)
            self._stage_cumulative(plan, paths)
            staged = {
                name: ArtifactIdentity.from_path(name, transaction.stage / name)
                for name in EXPECTED_ARTIFACTS
            }
            journal.transition(
                ExecutionState.STAGED,
                artifacts=staged,
                validation={"cumulative_staged_sha256": dict(plan.expected_sha256)},
            )
            self._checkpoint("after_staging")

            def staged_validator(stage_directory: Path) -> bool:
                reopened_stage = reopen_and_validate_artifacts(
                    {name: _read_regular(stage_directory / name) for name in EXPECTED_ARTIFACTS}
                )
                return bool(
                    reopened_stage.manifest.get("authority", {}).get("authorization_sha256")
                    == authorization.digest
                    and _file_sha256(stage_directory / "loci.parquet")
                    == authorization.artifact_sha256["loci.parquet"]
                    and _file_sha256(stage_directory / "alerts.parquet")
                    == authorization.artifact_sha256["alerts.parquet"]
                )

            transaction.validate(staged_validator)
            journal.transition(ExecutionState.VALIDATED, artifacts=staged, validation={"passed": True})
            self._checkpoint("before_final_qualification")
            final = self._qualify_production(
                candidate, authorization, own_transaction=transaction_id, baseline=before
            )
            for key in _CUMULATIVE_KEYS:
                if _file_sha256(paths.cumulative_staged[key]) != plan.expected_sha256[key]:
                    raise PublicationRefused("cumulative_contradiction", f"Staged {key} changed.")
            journal.update(
                publication={
                    "attempted": True,
                    "committed": False,
                    "boundary": "authority-gate",
                    "gate_intent_at_utc": _iso(self.clock()),
                },
                validation={
                    "final_qualification": {
                        "durable_fingerprint_sha256": final["durable_fingerprint_sha256"],
                        "mount_binding": dict(final["mount_binding"]),
                        "runtime_linkage_verified": True,
                    }
                },
            )
            self._checkpoint("before_authority_gate")
            self._create_gate(candidate, authorization, journal, paths)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if _gate_names(self.data_root, transaction_id):
                return self._interrupted(candidate, authorization, journal, paths, error, (writer_lock,))
            return self._not_committed(
                candidate, authorization, journal, paths, transaction, writer_lock, error
            )
        return self._advance(candidate, authorization, journal, paths, transaction, writer_lock)

    def _stage_cumulative(self, plan: CumulativePlan, paths: TransactionPaths) -> None:
        paths.cumulative_stage.mkdir(mode=0o700, parents=False, exist_ok=False)
        for key in _CUMULATIVE_KEYS:
            _write_new_file(paths.cumulative_staged[key], plan.payloads[key], plan.modes[key])
            if _file_sha256(paths.cumulative_staged[key]) != plan.expected_sha256[key]:
                raise PublicationRefused("cumulative_contradiction", f"Staged {key} differs.")
        _fsync_directory(paths.cumulative_stage)
        _fsync_directory(paths.cumulative_stage.parent)

    def _create_gate(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        paths: TransactionPaths,
    ) -> None:
        """Hard-link the gate into production: the first production mutation."""
        metadata = _metadata(journal)
        document = {
            "schema_version": GATE_SCHEMA,
            "state": "authority_transition_in_progress",
            "transaction_id": journal.snapshot.descriptor.run_id,
            "date_utc": candidate.date_utc,
            "predecessor_date_utc": authorization.predecessor_date_utc,
            "authorization_sha256": authorization.digest,
            "authoritative_manifest_sha256": metadata["authoritative_manifest_sha256"],
            "baseline_cumulative_sha256": dict(metadata["cumulative"]["baseline_sha256"]),
            "expected_cumulative_sha256": dict(metadata["cumulative"]["expected_sha256"]),
            "publisher_release_sha": self.publisher_release_sha,
            "created_at_utc": _iso(self.clock()),
        }
        parent = paths.gate.parent
        _assert_real_path(self.data_root, parent)
        if not parent.is_dir():
            raise PublicationRefused("path_contradiction", "Survey root is missing.")
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_new_file(paths.staged_gate, payload)
        _fsync_directory(paths.staged_gate.parent)
        try:
            os.link(str(paths.staged_gate), str(paths.gate), follow_symlinks=False)
        except FileExistsError as exc:
            raise PublicationRefused(
                "authority_transition_pending",
                "Another publication authority transition holds the production gate.",
            ) from exc
        _fsync_directory(parent)
        self._checkpoint("after_gate_link", transaction_id=document["transaction_id"])
        os.unlink(str(paths.staged_gate))
        _fsync_directory(paths.staged_gate.parent)

    def _not_committed(
        self, candidate, authorization, journal, paths, transaction, writer_lock, error,
    ) -> PublicationOutcome:
        """Failure before the gate: production is provably the authorized baseline."""
        category = classify_failure(error, authority_transition_begun=False)
        notes = []
        try:
            if journal.snapshot.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
                journal.transition(
                    ExecutionState.FAILED,
                    reason=(error.code if isinstance(error, PublicationRefused) else "interrupted_before_authority_transition"),
                    failure={
                        "category": category.value,
                        "error_type": type(error).__name__,
                        "message": str(error)[:500],
                        "authority_state": AuthorityState.NOT_COMMITTED.value,
                        "production_mutated": False,
                    },
                )
        except JournalError as journal_error:
            notes.append(f"journal: {journal_error}")
        if transaction is not None:
            try:
                transaction.abort("publication_failed_before_authority_transition")
            except Exception as cleanup_error:  # pragma: no cover - evidence retained
                notes.append(f"abort: {cleanup_error}")
        if writer_lock.held:
            try:
                writer_lock.release()
            except Exception as lock_error:  # pragma: no cover - adopted on resume
                notes.append(f"lock: {lock_error}")
        try:
            self._remove_owned_staging(paths, nightly=True)
        except PublicationRefused as cleanup_error:
            notes.append(f"staging: {cleanup_error}")
        if paths.target.exists() or paths.target.is_symlink():
            raise PublicationRefused(
                "authority_contradiction", "A target exists although no gate was created."
            )
        refused = isinstance(error, PublicationRefused)
        record = self._failure_record(
            candidate, authorization, journal, error, category,
            status="REFUSED" if refused else "NOT_COMMITTED",
            state=AuthorityState.NOT_COMMITTED,
            production_mutated=False,
            next_action=(
                "Production is unchanged; resolve the refusal before re-authorizing."
                if refused and category not in RETRYABLE_CATEGORIES
                else "Production is unchanged; re-run publish with the same authorization."
            ),
            notes=notes,
        )
        path = self._write_attempt_record(candidate.date_utc, journal, record)
        return PublicationOutcome("refused" if refused else "not_committed", False, record, path)

    # -- RECONCILIATION_REQUIRED phase ---------------------------------------

    def _advance(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        paths: TransactionPaths,
        transaction: Optional[PublicationTransaction],
        writer_lock: Optional[WriterLock],
    ) -> PublicationOutcome:
        """Roll one gated authority transition forward to COMPLETE."""
        transaction_id = journal.snapshot.descriptor.run_id
        metadata = _metadata(journal)
        reconciliation_lock: Optional[WriterLock] = None
        try:
            if journal.snapshot.state == ExecutionState.VALIDATED:
                journal.transition(
                    ExecutionState.PUBLISHED,
                    publication={
                        "authority_transition_begun": True,
                        "gate_created": True,
                        "committed": False,
                    },
                    durability={"status": "client_fsync_confirmed", "server_crash_survival_claimed": False},
                    reconciliation={"status": "required", "phase": "nightly_partition"},
                )
            self._checkpoint("after_authority_gate", transaction_id=transaction_id)
            if transaction is not None:
                transaction.publish()
                transaction.release_writer_lock()
            else:
                self._adopt_stranded_locks(paths, transaction_id)
                writer_lock = WriterLock(
                    self.capability, paths.target_relative.as_posix(), transaction_id,
                    transaction_id=transaction_id, release_sha=self.publisher_release_sha,
                )
                writer_lock.acquire(at=self.clock())
                self._complete_nightly_partition(authorization, journal, paths)
                writer_lock.release()
            if _file_sha256(paths.target / "manifest.json") != metadata["authoritative_manifest_sha256"]:
                raise PublicationRefused("authority_contradiction", "Linked manifest differs.")
            if journal.snapshot.state == ExecutionState.PUBLISHED:
                journal.transition(
                    ExecutionState.RECONCILING,
                    publication={"nightly_manifest_linked": True},
                    reconciliation={"status": "required", "phase": "cumulative"},
                )
            reconciliation_lock = self._acquire_reconciliation_lock(transaction_id)
            installed = self._install_cumulative(journal, paths)
            reconciliation_lock.release()
            reconciliation_lock = None
            verification = self._verify_products(
                candidate, authorization, journal, paths, require_tail=True, reopen=True
            )
            journal.update(
                reconciliation={
                    "status": "required",
                    "phase": "authority_commit",
                    "installed": installed,
                    "verified_at_utc": verification["verified_at_utc"],
                    "commit_intent_at_utc": _iso(self.clock()),
                }
            )
            self._checkpoint("before_authority_commit", transaction_id=transaction_id)
            self._remove_gate(transaction_id)
            committed_at = _iso(self.clock())
            self._checkpoint("after_authority_commit", transaction_id=transaction_id)
            journal.update(
                reconciliation={
                    "status": "required",
                    "phase": "authority_committed_finalization_pending",
                    "authority_committed_at_utc": committed_at,
                    "authority_commit_observed": True,
                }
            )
            return self._finalize(candidate, authorization, journal, paths, verification)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            held = tuple(lock for lock in (reconciliation_lock, writer_lock) if lock is not None)
            return self._interrupted(candidate, authorization, journal, paths, error, held)

    def _complete_nightly_partition(
        self,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        paths: TransactionPaths,
    ) -> None:
        """Idempotently finish a manifest-last partition from verified staging."""
        expected = {
            "loci.parquet": authorization.artifact_sha256["loci.parquet"],
            "alerts.parquet": authorization.artifact_sha256["alerts.parquet"],
            "manifest.json": _metadata(journal)["authoritative_manifest_sha256"],
        }
        target, stage = paths.target, paths.stage
        _assert_real_path(self.data_root, target)
        _assert_real_path(self.capability.staging_root, stage)
        if not (target / "manifest.json").exists():
            if stage.is_symlink() or not stage.is_dir():
                raise PublicationRefused(
                    "authority_contradiction",
                    "The nightly manifest is not linked and its staged material is missing.",
                )
            snapshots = {name: _artifact_snapshot(stage / name) for name in EXPECTED_ARTIFACTS}
            if {name: item.sha256 for name, item in snapshots.items()} != expected:
                raise PublicationRefused("authority_contradiction", "Staged nightly bytes differ.")
            if not target.exists():
                _ensure_directory_tree_fsynced(target.parent, self.capability.root)
                _publish_noreplace(
                    stage, target, snapshots,
                    lambda event, details: self._checkpoint(event, **dict(details)),
                )
            else:
                entries = {path.name for path in target.iterdir()}
                if not entries <= {"loci.parquet", "alerts.parquet", ".manifest.pending"}:
                    raise PublicationRefused("authority_contradiction", "Reserved target is foreign.")
                for name in ("loci.parquet", "alerts.parquet"):
                    if name not in entries:
                        os.link(str(stage / name), str(target / name), follow_symlinks=False)
                if ".manifest.pending" not in entries:
                    os.link(
                        str(stage / "manifest.json"), str(target / ".manifest.pending"),
                        follow_symlinks=False,
                    )
                _fsync_directory(target)
                for name, value in (
                    ("loci.parquet", expected["loci.parquet"]),
                    ("alerts.parquet", expected["alerts.parquet"]),
                    (".manifest.pending", expected["manifest.json"]),
                ):
                    if _file_sha256(target / name) != value:
                        raise PublicationRefused("authority_contradiction", f"{name} differs.")
                self._checkpoint("before_manifest_commit", resumed=True)
                os.link(
                    str(target / ".manifest.pending"), str(target / "manifest.json"),
                    follow_symlinks=False,
                )
                _fsync_directory(target)
                self._checkpoint("after_manifest_commit", resumed=True)
        if _file_sha256(target / "manifest.json") != expected["manifest.json"]:
            raise PublicationRefused("authority_contradiction", "Linked manifest differs.")
        pending = target / ".manifest.pending"
        if pending.exists():
            if _file_sha256(pending) != expected["manifest.json"]:
                raise PublicationRefused("authority_contradiction", "Pending manifest differs.")
            pending.unlink()
            _fsync_directory(target)
        entries = {path.name for path in target.iterdir()}
        if entries != set(EXPECTED_ARTIFACTS):
            raise PublicationRefused("authority_contradiction", "Nightly partition is not exact.")
        for name in EXPECTED_ARTIFACTS:
            if _file_sha256(target / name) != expected[name]:
                raise PublicationRefused("authority_contradiction", f"Published {name} differs.")
        self._remove_committed_stage(paths)

    def _remove_committed_stage(self, paths: TransactionPaths) -> None:
        """Drop staged links that are provably the published inodes (crash-tolerant)."""
        stage, target = paths.stage, paths.target
        _assert_real_path(self.capability.staging_root, stage)
        if not stage.exists():
            return
        if stage.is_symlink() or not stage.is_dir():
            raise PublicationRefused("transaction_residue", "Committed stage is unsafe.")
        for entry in list(stage.iterdir()):
            if (
                entry.name not in EXPECTED_ARTIFACTS
                or entry.is_symlink()
                or not entry.is_file()
                or not os.path.samefile(entry, target / entry.name)
            ):
                raise PublicationRefused(
                    "transaction_residue", f"Committed stage holds foreign {entry.name}."
                )
            entry.unlink()
        _fsync_directory(stage)
        stage.rmdir()
        _fsync_directory(stage.parent)

    def _install_cumulative(
        self, journal: TransactionJournal, paths: TransactionPaths
    ) -> Dict[str, str]:
        """current == expected: installed; == baseline: replace from stage; else refuse."""
        plan = _metadata(journal)["cumulative"]
        outcomes = {}
        for key in _CUMULATIVE_KEYS:
            target = paths.cumulative_targets[key]
            _assert_real_path(self.data_root, target)
            current = _optional_sha256(target)
            if current == plan["expected_sha256"][key]:
                outcomes[key] = "already_installed"
                continue
            if current != plan["baseline_sha256"][key]:
                raise PublicationRefused(
                    "cumulative_contradiction", f"Cumulative {key} matches neither plan nor baseline."
                )
            staged = paths.cumulative_staged[key]
            _assert_real_path(self.capability.staging_root, staged)
            if _optional_sha256(staged) != plan["expected_sha256"][key]:
                raise PublicationRefused(
                    "cumulative_contradiction", f"Staged cumulative {key} is missing or changed."
                )
            self._checkpoint(f"before_cumulative_install:{key}")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.replace(staged, target)
            _fsync_directory(target.parent)
            _fsync_directory(staged.parent)
            outcomes[key] = "installed"
        return outcomes

    def _remove_gate(self, transaction_id: str) -> None:
        """The authority commit: durable removal of this transaction's gate."""
        gate = read_publication_gate(self.data_root)
        if gate is None or gate["transaction_id"] != transaction_id:
            raise PublicationRefused("authority_contradiction", "Gate is not this transaction's.")
        path = history.publication_gate_path(self.data_root)
        os.unlink(str(path))
        _fsync_directory(path.parent)

    def _acquire_reconciliation_lock(self, transaction_id: str) -> WriterLock:
        lock = WriterLock(
            self.capability, SHARED_RECONCILIATION_LOCK_IDENTITY, transaction_id,
            transaction_id=f"{transaction_id}:reconciliation",
            release_sha=self.publisher_release_sha,
        )
        waited = time.monotonic()
        while True:
            try:
                lock.acquire(at=self.clock())
                return lock
            except LockUnavailable:
                if time.monotonic() - waited >= RECONCILIATION_LOCK_WAIT_SECONDS:
                    raise
                time.sleep(RECONCILIATION_LOCK_POLL_SECONDS)

    def _interrupted(
        self, candidate, authorization, journal, paths, error, held_locks,
    ) -> PublicationOutcome:
        """Failure after the gate: truthfully RECONCILIATION_REQUIRED (or COMPLETE)."""
        for lock in held_locks:
            if lock is not None and lock.held:
                try:
                    lock.release()
                except Exception:  # pragma: no cover - adopted on resume
                    pass
        category = classify_failure(error, authority_transition_begun=True)
        if (
            _gate_names(self.data_root, journal.snapshot.descriptor.run_id)
            and journal.snapshot.state == ExecutionState.VALIDATED
        ):
            try:  # the gate exists: make the journal say so too
                journal.transition(
                    ExecutionState.PUBLISHED,
                    publication={"authority_transition_begun": True, "gate_created": True},
                    durability={"status": "client_fsync_confirmed", "server_crash_survival_claimed": False},
                    reconciliation={"status": "required", "phase": "nightly_partition"},
                )
            except JournalError:
                pass
        try:
            state = AuthorityState(self.authority_state(candidate.date_utc)["state"])
        except PublicationRefused:
            state = AuthorityState.CONTRADICTION
        if _gate_names(self.data_root, journal.snapshot.descriptor.run_id):
            state = AuthorityState.RECONCILIATION_REQUIRED
        notes = []
        try:
            if journal.snapshot.state != ExecutionState.COMPLETE:
                interruptions = int(journal.snapshot.recovery.get("interruption_count", 0)) + 1
                journal.update(
                    reconciliation={"status": "required"},
                    recovery={
                        "interruption_count": interruptions,
                        "last_interruption": {
                            "category": category.value,
                            "error_type": type(error).__name__,
                            "message": str(error)[:500],
                            "authority_state": state.value,
                            "observed_at_utc": _iso(self.clock()),
                        },
                    },
                )
        except JournalError as journal_error:
            notes.append(f"journal: {journal_error}")
        status = {
            AuthorityState.RECONCILIATION_REQUIRED: "RECONCILIATION_REQUIRED",
            AuthorityState.COMPLETE: "COMMITTED_FINALIZATION_REQUIRED",
        }.get(state, "AUTHORITY_CONTRADICTION")
        record = self._failure_record(
            candidate, authorization, journal, error, category,
            status=status, state=state, production_mutated=True,
            next_action=(
                "Readers refuse until resolved; re-run publish with the same "
                "authorization to roll the transition forward (no reacquisition)."
                if state != AuthorityState.CONTRADICTION
                else "Operator recovery required; evidence is contradictory."
            ),
            notes=notes,
        )
        path = self._write_attempt_record(candidate.date_utc, journal, record)
        return PublicationOutcome(status.lower(), False, record, path)

    # -- COMPLETE phase -------------------------------------------------------

    def _finalize(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        paths: TransactionPaths,
        verification: Mapping[str, Any],
        *,
        commit_not_after: Optional[str] = None,
    ) -> PublicationOutcome:
        """Record COMPLETE after the gate is gone; never invent a commit time."""
        transaction_id = journal.snapshot.descriptor.run_id
        committed_at = journal.snapshot.reconciliation.get("authority_committed_at_utc")
        if journal.snapshot.state != ExecutionState.COMPLETE:
            post = self.sentinel(candidate.date_utc)
            if journal.snapshot.state == ExecutionState.PUBLISHED:
                journal.transition(ExecutionState.RECONCILING)
            intent = journal.snapshot.reconciliation.get("commit_intent_at_utc")
            chronology = _metadata(journal)["chronology"]
            if committed_at is not None:
                _require_order(
                    ("publication_transaction_started_at_utc", chronology["publication_transaction_started_at_utc"]),
                    ("authority_commit_intent_at_utc", intent),
                    ("authority_committed_at_utc", committed_at),
                )
            self._checkpoint("before_journal_complete", transaction_id=transaction_id)
            journal.transition(
                ExecutionState.COMPLETE,
                publication={"committed": True, "authority_committed": True},
                reconciliation={
                    "status": "complete",
                    "phase": "complete",
                    "authority_committed_at_utc": committed_at,
                    "authority_commit_observed": committed_at is not None,
                    "authority_commit_not_before_utc": intent,
                    "authority_commit_not_after_utc": committed_at or commit_not_after,
                    "resulting_production_fingerprint": post["durable_fingerprint_sha256"],
                    "resulting_manifest_count": post["durable_state"]["manifest_count"],
                    "resulting_cumulative_sha256": dict(
                        post["durable_state"]["cumulative_artifact_hashes"]
                    ),
                },
            )
            self._checkpoint("after_journal_complete", transaction_id=transaction_id)
        self._cleanup_after_complete(paths, transaction_id)
        record_path = self._record_path(candidate.date_utc, transaction_id)
        if record_path.exists():
            record = dict(_read_json(record_path))
            record["idempotent_replay"] = True
            record["replay_verification"] = dict(verification)
            return PublicationOutcome("already_published", True, record, record_path)
        record = self._terminal_record(candidate, authorization, journal, verification)
        path = self._write_record(candidate.date_utc, transaction_id, record)
        return PublicationOutcome("published", True, record, path)

    def _resume(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        *,
        gate_present: bool,
    ) -> PublicationOutcome:
        transaction_id = journal.snapshot.descriptor.run_id
        paths = self._paths(candidate.date_utc, transaction_id)
        self._verify_journal_identity(journal, authorization, candidate, paths)
        if gate_present:
            gate = read_publication_gate(self.data_root)
            metadata = _metadata(journal)
            if (
                gate is None
                or gate["transaction_id"] != transaction_id
                or gate["date_utc"] != candidate.date_utc
                or gate["authoritative_manifest_sha256"] != metadata["authoritative_manifest_sha256"]
                or gate["expected_cumulative_sha256"] != metadata["cumulative"]["expected_sha256"]
                or gate["baseline_cumulative_sha256"] != metadata["cumulative"]["baseline_sha256"]
            ):
                raise PublicationRefused("authority_contradiction", "Gate and journal disagree.")
            if journal.snapshot.state not in {
                ExecutionState.VALIDATED, ExecutionState.PUBLISHED, ExecutionState.RECONCILING,
            }:
                raise PublicationRefused(
                    "journal_contradiction",
                    f"Gate exists but journal is {journal.snapshot.state.value}.",
                )
            return self._advance(candidate, authorization, journal, paths, None, None)
        # Published journal without a gate: the commit happened; finalize.
        try:
            verification = self._verify_products(
                candidate, authorization, journal, paths, require_tail=True, reopen=True
            )
            return self._finalize(
                candidate, authorization, journal, paths, verification,
                commit_not_after=_iso(self.clock()),
            )
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return self._interrupted(candidate, authorization, journal, paths, error, ())

    def _replay(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
    ) -> PublicationOutcome:
        """Idempotent replay of a COMPLETE authorization: verifies, changes nothing."""
        transaction_id = journal.snapshot.descriptor.run_id
        paths = self._paths(candidate.date_utc, transaction_id)
        self._verify_journal_identity(journal, authorization, candidate, paths)
        nights = _physical_nights(self.data_root)
        verification = self._verify_products(
            candidate, authorization, journal, paths,
            require_tail=bool(nights and nights[-1] == candidate.date_utc),
            reopen=False,
        )
        return self._finalize(candidate, authorization, journal, paths, verification)

    # -- resume integrity ------------------------------------------------------

    def _verify_journal_identity(
        self,
        journal: TransactionJournal,
        authorization: PublicationAuthorization,
        candidate: NightCandidate,
        paths: TransactionPaths,
    ) -> None:
        """Re-derive every path and identity; the journal's copies are only checked."""
        descriptor = journal.snapshot.descriptor
        metadata = descriptor.metadata
        match = TRANSACTION_ID_PATTERN.fullmatch(descriptor.run_id)
        recorded_authorization = metadata.get("authorization")
        try:
            recorded_digest = PublicationAuthorization.from_dict(recorded_authorization).digest
        except (PublicationRefused, TypeError):
            recorded_digest = None
        checks = (
            match is not None and match.group(1) == candidate.date_utc,
            match is not None and match.group(2) == authorization.digest[:12],
            descriptor.operation == PUBLICATION_OPERATION,
            descriptor.plan_id == authorization.digest,
            descriptor.release_sha == self.publisher_release_sha,
            Path(journal.path) == paths.journal,
            descriptor.target_identity == paths.target_relative.as_posix(),
            descriptor.target_path == str(paths.target),
            descriptor.stage_path == str(paths.stage),
            descriptor.lock_path == str(paths.writer_lock),
            metadata.get("schema_version") == DESCRIPTOR_SCHEMA,
            metadata.get("target_utc_night") == candidate.date_utc,
            metadata.get("authorization_sha256") == authorization.digest,
            recorded_digest == authorization.digest,
            metadata.get("candidate_record_sha256") == candidate.record_sha256,
            metadata.get("candidate_dir") == authorization.candidate_dir,
            _is_hex64(metadata.get("authoritative_manifest_sha256")),
        )
        if not all(checks):
            raise PublicationRefused(
                "journal_contradiction", "Journal identity differs from the re-derived transaction."
            )
        relative = _relative_views(self.capability, paths)
        cumulative = metadata.get("cumulative", {})
        if any(cumulative.get(key) != value for key, value in relative.items()):
            raise PublicationRefused(
                "path_contradiction", "Journal cumulative paths differ from re-derived paths."
            )
        for root, path in (
            (self.data_root, paths.target),
            (self.data_root, paths.gate),
            (self.capability.staging_root, paths.stage),
            (self.capability.staging_root, paths.cumulative_stage),
            (self.capability.journal_root, paths.journal),
        ):
            _assert_real_path(root, path)
        for key in _CUMULATIVE_KEYS:
            _assert_real_path(self.data_root, paths.cumulative_targets[key])
        manifest = paths.target / "manifest.json"
        if manifest.exists() and _file_sha256(manifest) != metadata["authoritative_manifest_sha256"]:
            raise PublicationRefused(
                "authority_contradiction", "Authoritative manifest differs from the journal."
            )

    def _verify_products(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        paths: TransactionPaths,
        *,
        require_tail: bool,
        reopen: bool,
    ) -> Dict[str, Any]:
        """Verify one publication generation from physical products."""
        import pandas as pd

        metadata = _metadata(journal)
        plan = metadata["cumulative"]
        target = paths.target
        _assert_real_path(self.data_root, target)
        if reopen:
            identities = independent_reopen(target, None)
            published = {name: identities[name].sha256 for name in EXPECTED_ARTIFACTS}
        else:
            published = {name: _file_sha256(target / name) for name in EXPECTED_ARTIFACTS}
        manifest = _read_json(target / "manifest.json")
        authority = manifest.get("authority") if isinstance(manifest.get("authority"), dict) else {}
        index = pd.read_parquet(paths.cumulative_targets["loci_index"])
        summary = pd.read_parquet(paths.cumulative_targets["nightly_summary"])
        night_rows = int(index["night_date_utc"].eq(candidate.date_utc).sum())
        summary_rows = int(summary["date_utc"].eq(candidate.date_utc).sum())
        nights = _physical_nights(self.data_root)
        checks = {
            "artifact_set_exact": {p.name for p in target.iterdir()} == set(EXPECTED_ARTIFACTS),
            "manifest_sha_matches_journal": published["manifest.json"]
            == metadata["authoritative_manifest_sha256"],
            "science_bytes_equal_candidate": all(
                published[n] == authorization.artifact_sha256[n]
                for n in ("loci.parquet", "alerts.parquet")
            ),
            "authority_bound": authority.get("authorization_sha256") == authorization.digest
            and authority.get("transaction_id") == journal.snapshot.descriptor.run_id,
            "manifest_counts_match_candidate": manifest.get("actual_loci") == candidate.loci
            and manifest.get("alert_rows") == candidate.alerts,
            "night_published_once": nights.count(candidate.date_utc) == 1,
            "cumulative_index_holds_night_once": night_rows == candidate.loci,
            "cumulative_summary_holds_night_once": summary_rows == 1,
        }
        if require_tail:
            checks.update(
                {
                    "chronological_tail": bool(nights) and nights[-1] == candidate.date_utc,
                    "manifest_count_extended_exactly_once": len(nights)
                    == metadata["baseline_manifest_count"] + 1,
                    "cumulative_index_extended_exactly_once": len(index)
                    == plan["prior_rows"]["loci_index"] + candidate.loci,
                    "cumulative_summary_extended_exactly_once": len(summary)
                    == plan["prior_rows"]["nightly_summary"] + 1,
                    "cumulative_hashes_expected": all(
                        _file_sha256(paths.cumulative_targets[key]) == plan["expected_sha256"][key]
                        for key in _CUMULATIVE_KEYS
                    ),
                }
            )
        verification = {
            "strict": require_tail,
            "independent_reopen": reopen,
            "passed": all(checks.values()),
            "checks": checks,
            "published_artifacts": published,
            "loci_index_rows": len(index),
            "summary_rows": len(summary),
            "verified_at_utc": _iso(self.clock()),
        }
        if not verification["passed"]:
            failed = sorted(name for name, ok in checks.items() if not ok)
            raise PublicationRefused(
                "authority_contradiction", f"Publication verification failed: {failed}."
            )
        return verification

    # -- evidence-owned cleanup -------------------------------------------------

    def _abandon(self, journal: TransactionJournal) -> None:
        """Close a pre-gate unfinished publication journal as NOT_COMMITTED."""
        descriptor = journal.snapshot.descriptor
        date_utc = descriptor.metadata.get("target_utc_night")
        paths = self._paths(str(date_utc), descriptor.run_id)
        if (
            Path(journal.path) != paths.journal
            or descriptor.target_path != str(paths.target)
            or descriptor.stage_path != str(paths.stage)
        ):
            raise PublicationRefused("journal_contradiction", "Unfinished journal paths differ.")
        if journal.snapshot.published or paths.target.exists() or paths.target.is_symlink():
            raise PublicationRefused(
                "authority_contradiction", "An unfinished transaction without a gate reserved a target."
            )
        self._remove_owned_staging(paths, nightly=True)
        self._adopt_stranded_locks(paths, descriptor.run_id)
        journal.transition(
            ExecutionState.FAILED,
            reason="abandoned_before_authority_transition",
            failure={
                "category": FailureCategory.INTERRUPTED_BEFORE_COMMIT.value,
                "authority_state": AuthorityState.NOT_COMMITTED.value,
                "production_mutated": False,
            },
            recovery={"abandoned_at_utc": _iso(self.clock()), "abandoned_by": "publisher_resume"},
        )

    def _sweep_terminal_residue(self, journals: Sequence[TransactionJournal]) -> None:
        for journal in journals:
            if not _is_publication_journal(journal) or _unfinished(journal):
                continue
            descriptor = journal.snapshot.descriptor
            paths = self._paths(str(descriptor.metadata.get("target_utc_night")), descriptor.run_id)
            if journal.snapshot.outcome == JournalOutcome.COMPLETE:
                self._cleanup_after_complete(paths, descriptor.run_id)
            else:
                self._remove_owned_staging(paths, nightly=True)
                self._adopt_stranded_locks(paths, descriptor.run_id)

    def _remove_owned_staging(self, paths: TransactionPaths, *, nightly: bool) -> None:
        """Remove only this transaction's derived, never-authoritative staging."""
        staging = self.capability.staging_root
        directories = [(paths.cumulative_stage, {f"{key}.parquet" for key in _CUMULATIVE_KEYS})]
        if nightly:
            directories.insert(0, (paths.stage, set(EXPECTED_ARTIFACTS)))
        for directory, allowed in directories:
            _assert_real_path(staging, directory)
            if not directory.exists():
                continue
            entries = list(directory.iterdir())
            if any(
                entry.name not in allowed or entry.is_symlink() or not entry.is_file()
                for entry in entries
            ):
                raise PublicationRefused("transaction_residue", f"Unexpected staging in {directory.name}.")
            for entry in entries:
                entry.unlink()
            _fsync_directory(directory)
            directory.rmdir()
            _fsync_directory(directory.parent)
        if nightly and paths.stage_parent.exists() and not any(paths.stage_parent.iterdir()):
            paths.stage_parent.rmdir()
            _fsync_directory(staging)
        if paths.staged_gate.is_file() and not paths.staged_gate.is_symlink():
            paths.staged_gate.unlink()
            _fsync_directory(staging)

    def _cleanup_after_complete(self, paths: TransactionPaths, transaction_id: str) -> None:
        """Only after durable COMPLETE: drop redundant staging and stranded locks."""
        staging = self.capability.staging_root
        self._remove_committed_stage(paths)
        if paths.cumulative_stage.exists():
            _assert_real_path(staging, paths.cumulative_stage)
            if any(paths.cumulative_stage.iterdir()):
                raise PublicationRefused(
                    "transaction_residue", "Installed cumulative staging is not empty."
                )
            paths.cumulative_stage.rmdir()
            _fsync_directory(staging)
        if paths.stage_parent.exists() and not any(paths.stage_parent.iterdir()):
            paths.stage_parent.rmdir()
            _fsync_directory(staging)
        if paths.staged_gate.is_file() and not paths.staged_gate.is_symlink():
            paths.staged_gate.unlink()
            _fsync_directory(staging)
        self._adopt_stranded_locks(paths, transaction_id)

    def _adopt_stranded_locks(self, paths: TransactionPaths, transaction_id: str) -> None:
        """Release this transaction's own locks left by a dead process.

        Safe only because the caller holds the production-wide publication lock:
        no live publisher can own a lock naming this transaction.
        """
        for path, owner in (
            (paths.writer_lock, transaction_id),
            (paths.reconciliation_lock, f"{transaction_id}:reconciliation"),
        ):
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or not path.is_dir():
                raise PublicationRefused("transaction_residue", f"Lock {path.name} is unsafe.")
            entries = [entry.name for entry in path.iterdir()]
            observed = _lock_transaction(path)
            if entries == [] and path == paths.writer_lock:
                path.rmdir()  # crash between lock mkdir and metadata write
                _fsync_directory(path.parent)
                continue
            if observed != owner:
                if path == paths.reconciliation_lock and observed is not None:
                    continue  # another owner; acquisition waits or fails transiently
                raise PublicationRefused("transaction_residue", f"Lock {path.name} is foreign.")
            metadata = _read_json(path / "owner.json")
            if entries != ["owner.json"] or metadata.get("hostname") != socket.gethostname():
                raise PublicationRefused(
                    "transaction_residue", f"Lock {path.name} cannot be adopted safely."
                )
            (path / "owner.json").unlink()
            _fsync_directory(path)
            path.rmdir()
            _fsync_directory(path.parent)

    # -- records ----------------------------------------------------------------

    def _refused(
        self, candidate: NightCandidate, authorization: Any, refusal: PublicationRefused
    ) -> PublicationOutcome:
        try:
            state = self.authority_state(candidate.date_utc)["state"]
        except Exception:  # pragma: no cover - classification is best effort here
            state = AuthorityState.CONTRADICTION.value
        record = {
            "schema_version": PUBLICATION_RECORD_SCHEMA,
            "date_utc": candidate.date_utc,
            "status": "REFUSED",
            "authority_state": state,
            "refusal_code": refusal.code,
            "failure_category": refusal.category.value,
            "retryable": refusal.category in RETRYABLE_CATEGORIES,
            "message": str(refusal),
            "authorization_sha256": getattr(authorization, "digest", None),
            "production_mutated": False,
            "observed_at_utc": _iso(self.clock()),
        }
        return PublicationOutcome("refused", False, record, None)

    def _unexpected(
        self, candidate: NightCandidate, authorization: Any, error: Exception
    ) -> PublicationOutcome:
        """An error outside a journaled phase: report the evidence-derived state."""
        try:
            state = AuthorityState(self.authority_state(candidate.date_utc)["state"])
        except Exception:  # pragma: no cover - classification is best effort here
            state = AuthorityState.CONTRADICTION
        category = classify_failure(
            error, authority_transition_begun=state != AuthorityState.NOT_COMMITTED
        )
        record = {
            "schema_version": PUBLICATION_RECORD_SCHEMA,
            "date_utc": candidate.date_utc,
            "status": "NOT_COMMITTED" if state == AuthorityState.NOT_COMMITTED else state.value,
            "authority_state": state.value,
            "error_type": type(error).__name__,
            "failure_category": category.value,
            "retryable": category in RETRYABLE_CATEGORIES,
            "message": str(error)[:500],
            "authorization_sha256": getattr(authorization, "digest", None),
            "production_mutated": state != AuthorityState.NOT_COMMITTED,
            "observed_at_utc": _iso(self.clock()),
        }
        return PublicationOutcome(record["status"].lower(), False, record, None)

    def _failure_record(
        self, candidate, authorization, journal, error, category, *, status, state,
        production_mutated, next_action, notes,
    ) -> Dict[str, Any]:
        return {
            "schema_version": PUBLICATION_RECORD_SCHEMA,
            "date_utc": candidate.date_utc,
            "transaction_id": journal.snapshot.descriptor.run_id,
            "status": status,
            "authority_state": state.value,
            "production_mutated": production_mutated,
            "refusal_code": getattr(error, "code", None),
            "error_type": type(error).__name__,
            "failure_category": category.value,
            "retryable": category in RETRYABLE_CATEGORIES,
            "message": str(error)[:500],
            "authorization_sha256": authorization.digest,
            "journal": str(journal.path),
            "journal_state": journal.snapshot.state.value,
            "journal_outcome": journal.snapshot.outcome.value,
            "next_action": next_action,
            "notes": list(notes),
            "observed_at_utc": _iso(self.clock()),
        }

    def _terminal_record(
        self,
        candidate: NightCandidate,
        authorization: PublicationAuthorization,
        journal: TransactionJournal,
        verification: Mapping[str, Any],
    ) -> Dict[str, Any]:
        metadata = _metadata(journal)
        reconciliation = journal.snapshot.reconciliation
        manifest = _read_json(self._paths(candidate.date_utc, journal.snapshot.descriptor.run_id).target / "manifest.json")
        chronology = dict(metadata["chronology"])
        record_at = _iso(self.clock())
        committed = reconciliation.get("authority_committed_at_utc")
        _require_order(
            ("publication_transaction_started_at_utc", chronology["publication_transaction_started_at_utc"]),
            ("authority_commit_not_after_utc", reconciliation.get("authority_commit_not_after_utc")),
            ("terminal_record_at_utc", record_at),
        )
        chronology.update(
            {
                "authority_commit_intent_at_utc": reconciliation.get("authority_commit_not_before_utc"),
                "authority_committed_at_utc": committed,
                "authority_commit_observed": reconciliation.get("authority_commit_observed"),
                "authority_commit_not_before_utc": reconciliation.get("authority_commit_not_before_utc"),
                "authority_commit_not_after_utc": reconciliation.get("authority_commit_not_after_utc"),
                "verification_completed_at_utc": verification["verified_at_utc"],
                "terminal_record_at_utc": record_at,
                "candidate_manifest_original": manifest["authority"]["chronology"]["candidate_manifest_original"],
                "candidate_manifest_defects": manifest["authority"]["chronology"]["candidate_manifest_defects"],
                "ordering_verified": True,
            }
        )
        return {
            "schema_version": PUBLICATION_RECORD_SCHEMA,
            "date_utc": candidate.date_utc,
            "status": "PUBLISHED",
            "authority_state": AuthorityState.COMPLETE.value,
            "transaction_id": journal.snapshot.descriptor.run_id,
            "authoritative": True,
            "recovered_by_resume": int(journal.snapshot.recovery.get("interruption_count", 0)) > 0
            or reconciliation.get("authority_commit_observed") is False,
            "authorization": authorization.as_dict(),
            "authorization_sha256": authorization.digest,
            "production_capability": (
                {
                    "binding_sha256": metadata.get(
                        "production_capability_binding_sha256"
                    ),
                    "control_token_sha256": metadata.get(
                        "control_token_sha256"
                    ),
                    "maximum_successful_uses": 1,
                    "successful_use_count": 1,
                }
                if metadata.get("production_capability_binding_sha256")
                else None
            ),
            "publisher_release_sha": self.publisher_release_sha,
            "provenance_identities": manifest["authority"]["provenance_identities"],
            "chronology": chronology,
            "candidate": {
                "kind": candidate.kind,
                "record_path": str(candidate.record_path),
                "record_sha256": candidate.record_sha256,
                "release_sha": candidate.release_sha,
                "artifacts": {n: dict(candidate.artifacts[n]) for n in EXPECTED_ARTIFACTS},
                "provenance": dict(candidate.provenance),
            },
            "published_artifacts": dict(verification["published_artifacts"]),
            "counts": {"loci": candidate.loci, "alerts": candidate.alerts},
            "predecessor": {
                "date_utc": authorization.predecessor_date_utc,
                "production_fingerprint": metadata["baseline_production_fingerprint"],
            },
            "cumulative_plan": {
                key: metadata["cumulative"][key]
                for key in ("baseline_sha256", "expected_sha256", "dtype_changes", "schema_changes")
            },
            "resulting_production_fingerprint": reconciliation["resulting_production_fingerprint"],
            "resulting_manifest_count": reconciliation["resulting_manifest_count"],
            "verification": dict(verification),
            "journal": str(journal.path),
            "journal_outcome": journal.snapshot.outcome.value,
            "interruptions": int(journal.snapshot.recovery.get("interruption_count", 0)),
        }

    def _record_path(self, date_utc: str, name: str) -> Path:
        return contained_path(
            self.capability.evidence_root, Path("publications") / date_utc / f"{name}.json"
        )

    def _write_record(self, date_utc: str, name: str, record: Mapping[str, Any]) -> Path:
        path = self._record_path(date_utc, name)
        _ensure_private_tree(path.parent, self.capability.root)
        _write_json_new(path, record)
        return path

    def _write_attempt_record(
        self, date_utc: str, journal: TransactionJournal, record: Mapping[str, Any]
    ) -> Optional[Path]:
        """Per-attempt failure evidence; ``publications/<night>/*.json`` stays terminal-only."""
        path = contained_path(
            self.capability.evidence_root,
            Path("publications") / date_utc / "attempts"
            / f"{journal.snapshot.descriptor.run_id}-r{journal.snapshot.revision}.json",
        )
        try:
            _ensure_private_tree(path.parent, self.capability.root)
            if not path.exists():
                _write_json_new(path, record)
            return path
        except (OSError, PublicationError):  # pragma: no cover - journal remains truth
            return None


def _lock_transaction(path: Path) -> Optional[str]:
    try:
        metadata = json.loads((Path(path) / "owner.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    value = metadata.get("transaction_id") if isinstance(metadata, dict) else None
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Production capability contract: one Control-approved June 27 canary only
# ---------------------------------------------------------------------------

PRODUCTION_CAPABILITY_SCHEMA = "v3.production-publication-capability.v1"
CONTROL_APPROVED_NIGHT = "2026-06-27"
CONTROL_APPROVED_PREDECESSOR = "2026-06-26"
CONTROL_APPROVED_CANDIDATE_ROOT = Path(
    "/astro/store/shire/ANTARES/work/canary/"
    "phase6f-recovery-0.4.3-4378bce-20260627-20260928T185444Z"
)
CONTROL_APPROVED_CANDIDATE_RELEASE = "4378bce9a78e250dc897a9252b21dc69d41dcd0a"
CONTROL_APPROVED_LOCK = (
    PRODUCTION_CONTROL_ROOT / "locks" / AUTHORITY_LOCK_NAME
)
_AUTHORITY_LOCK_BINDING_FIELDS = frozenset(
    {"path", "device", "inode", "mode", "uid", "gid", "size"}
)
_PRODUCTION_BINDING_FIELDS = (
    "schema_version",
    "operation",
    "hostname",
    "service_uid",
    "production_root",
    "stage_root",
    "control_root",
    "evidence_root",
    "sentinel_cache_path",
    "segment_cache_root",
    "mount_binding",
    "sentinel_fingerprint_sha256",
    "manifest_count",
    "authority_lock",
    "night_utc",
    "predecessor_night_utc",
    "candidate_root",
    "candidate_record_sha256",
    "candidate_binding_sha256",
    "candidate_provenance_sha256",
    "artifact_sha256",
    "publisher_release_sha",
    "candidate_release_sha",
    "cumulative_baseline_sha256",
    "expected_cumulative_sha256",
    "expected_cumulative_schema_sha256",
    "authorization_sha256",
    "control_token_sha256",
    "nonce",
    "expires_at_utc",
    "max_successful_uses",
)


def production_authority_lock_identity(
    path: Path = CONTROL_APPROVED_LOCK,
) -> Dict[str, Any]:
    """Return the exact no-follow identity of the canonical authority lock."""
    path = Path(path)
    if not path.is_absolute() or path != CONTROL_APPROVED_LOCK:
        raise PublicationRefused(
            "path_contradiction", "Production authority lock path is not canonical."
        )
    if path.is_symlink():
        raise PublicationRefused(
            "path_contradiction", "Production authority lock is a symlink."
        )
    try:
        observed = path.lstat()
    except OSError as exc:
        raise PublicationRefused(
            "path_contradiction", "Production authority lock is absent."
        ) from exc
    if not stat.S_ISREG(observed.st_mode):
        raise PublicationRefused(
            "path_contradiction", "Production authority lock is not a regular file."
        )
    return {
        "path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
        "uid": int(observed.st_uid),
        "gid": int(observed.st_gid),
        "size": int(observed.st_size),
    }


@dataclass(frozen=True)
class ProductionPublicationBinding:
    """Everything a Control-authorized single-night production canary must bind.

    Constructing a binding validates its shape only. Issuance additionally
    requires the external Control token, exact live Sentinel and lock identity,
    approved candidate, host, UID, and matching authorization.
    """

    hostname: str
    service_uid: int
    production_root: str
    stage_root: str
    control_root: str
    evidence_root: str
    sentinel_cache_path: str
    segment_cache_root: str
    mount_binding: Mapping[str, str]
    sentinel_fingerprint_sha256: str
    manifest_count: int
    authority_lock: Mapping[str, Any]
    night_utc: str
    predecessor_night_utc: str
    candidate_root: str
    candidate_record_sha256: str
    candidate_binding_sha256: str
    candidate_provenance_sha256: str
    artifact_sha256: Mapping[str, str]
    publisher_release_sha: str
    candidate_release_sha: str
    cumulative_baseline_sha256: Mapping[str, str]
    expected_cumulative_sha256: Mapping[str, str]
    expected_cumulative_schema_sha256: Mapping[str, str]
    authorization_sha256: str
    control_token_sha256: str
    nonce: str
    expires_at_utc: str
    max_successful_uses: int = 1
    operation: str = AUTHORIZED_OPERATION
    schema_version: str = PRODUCTION_CAPABILITY_SCHEMA

    def __post_init__(self) -> None:
        def invalid(message: str) -> PublicationRefused:
            return PublicationRefused("authorization_invalid", f"Production binding: {message}")

        roots = {}
        for name in (
            "production_root", "stage_root", "control_root", "evidence_root",
            "sentinel_cache_path", "segment_cache_root", "candidate_root",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
                raise invalid(f"{name} must be an absolute canonical path.")
            roots[name] = Path(value)
        separated = ("production_root", "stage_root", "control_root", "evidence_root",
                     "segment_cache_root", "candidate_root")
        for left in separated:
            for right in separated:
                if left < right and (
                    roots[left] == roots[right]
                    or roots[left] in roots[right].parents
                    or roots[right] in roots[left].parents
                ):
                    raise invalid(f"{left} and {right} overlap.")
        _canonical_date(self.night_utc, "night_utc")
        lock = self.authority_lock
        if (
            self.schema_version != PRODUCTION_CAPABILITY_SCHEMA
            or self.operation != AUTHORIZED_OPERATION
            or not isinstance(self.hostname, str)
            or "." not in self.hostname
            or isinstance(self.service_uid, bool)
            or not isinstance(self.service_uid, int)
            or self.service_uid <= 0
            or self.predecessor_night_utc != _previous_night(self.night_utc)
            or not isinstance(self.mount_binding, Mapping)
            or set(self.mount_binding) != _MOUNT_FIELDS
            or isinstance(self.manifest_count, bool)
            or not isinstance(self.manifest_count, int)
            or self.manifest_count < 0
            or not isinstance(lock, Mapping)
            or set(lock) != _AUTHORITY_LOCK_BINDING_FIELDS
            or lock.get("path") != str(CONTROL_APPROVED_LOCK)
            or lock.get("mode") != "0600"
            or any(
                isinstance(lock.get(name), bool)
                or not isinstance(lock.get(name), int)
                or lock.get(name) < 0
                for name in ("device", "inode", "uid", "gid", "size")
            )
            or not all(_is_hex64(v) for v in (
                self.sentinel_fingerprint_sha256, self.candidate_record_sha256,
                self.candidate_binding_sha256, self.candidate_provenance_sha256,
                self.authorization_sha256, self.control_token_sha256,
            ))
            or set(self.artifact_sha256) != set(EXPECTED_ARTIFACTS)
            or not all(_is_hex64(v) for v in self.artifact_sha256.values())
            or not _is_sha40(self.publisher_release_sha)
            or not _is_sha40(self.candidate_release_sha)
            or self.publisher_release_sha == self.candidate_release_sha
            or not all(
                isinstance(value, Mapping)
                and set(value) == set(_CUMULATIVE_KEYS)
                and all(_is_hex64(item) for item in value.values())
                for value in (
                    self.cumulative_baseline_sha256, self.expected_cumulative_sha256,
                    self.expected_cumulative_schema_sha256,
                )
            )
            or not isinstance(self.nonce, str) or len(self.nonce) != 32
            or not set(self.nonce) <= _HEX64
            or self.max_successful_uses != 1
        ):
            raise invalid("fields are malformed.")
        _parse_utc(self.expires_at_utc, "expires_at_utc")
        for name in (
            "mount_binding", "authority_lock", "artifact_sha256",
            "cumulative_baseline_sha256", "expected_cumulative_sha256",
            "expected_cumulative_schema_sha256",
        ):
            object.__setattr__(
                self,
                name,
                json.loads(_canonical(dict(getattr(self, name)))),
            )

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(_canonical({name: getattr(self, name) for name in _PRODUCTION_BINDING_FIELDS}))

    @property
    def digest(self) -> str:
        return _sha256_bytes(_canonical(self.as_dict()))


def qualify_production_binding(
    binding: ProductionPublicationBinding,
    authorization: PublicationAuthorization,
    *,
    hostname: str,
    uid: int,
    sentinel: Mapping[str, Any],
    now: datetime,
    candidate: Optional[NightCandidate] = None,
    authority_lock: Optional[Mapping[str, Any]] = None,
    allow_expired_resume: bool = False,
) -> Tuple[str, ...]:
    """Return every mismatch between a binding and live/authorized evidence."""
    observed = production_binding_from_sentinel(sentinel)
    checks = {
        "hostname": hostname == binding.hostname,
        "service_uid": uid == binding.service_uid,
        "production_root": observed["canonical_root"] == binding.production_root
        == authorization.production["canonical_root"],
        "mount_binding": observed["mount_binding"] == dict(binding.mount_binding)
        == authorization.production["mount_binding"],
        "sentinel_fingerprint": observed["durable_fingerprint_sha256"]
        == binding.sentinel_fingerprint_sha256
        == authorization.baseline_production_fingerprint,
        "manifest_count": observed["manifest_count"] == binding.manifest_count
        == authorization.production["manifest_count"],
        "sentinel_predicates": observed["predicates"] == authorization.production["predicates"]
        and observed["predicates"]["cache_path"] == binding.sentinel_cache_path,
        "cumulative_baseline": observed["cumulative_sha256"]
        == dict(binding.cumulative_baseline_sha256)
        == authorization.production["cumulative_sha256"],
        "authorization_digest": authorization.digest == binding.authorization_sha256,
        "night": authorization.date_utc == binding.night_utc
        and authorization.predecessor_date_utc == binding.predecessor_night_utc,
        "candidate": authorization.candidate_record_sha256 == binding.candidate_record_sha256
        and authorization.candidate_provenance_sha256
        == binding.candidate_provenance_sha256
        and Path(authorization.candidate_dir).parent == Path(binding.candidate_root)
        and (
            candidate is None
            or (
                candidate.record_sha256 == binding.candidate_record_sha256
                and candidate.provenance_sha256 == binding.candidate_provenance_sha256
                and candidate.provenance.get("binding_sha256")
                == binding.candidate_binding_sha256
            )
        ),
        "artifacts": dict(authorization.artifact_sha256) == dict(binding.artifact_sha256),
        "releases": authorization.publisher_release_sha == binding.publisher_release_sha
        and authorization.candidate_release_sha == binding.candidate_release_sha,
        "cumulative_plan": dict(authorization.expected_cumulative_sha256)
        == dict(binding.expected_cumulative_sha256),
        "nonce": authorization.nonce == binding.nonce,
        "expiry": authorization.expires_at_utc == binding.expires_at_utc
        and (
            allow_expired_resume
            or now < _parse_utc(binding.expires_at_utc, "expires_at_utc")
        ),
        "authority_lock": authority_lock is None
        or dict(authority_lock) == dict(binding.authority_lock),
        "single_use": binding.max_successful_uses == 1,
    }
    return tuple(sorted(name for name, passed in checks.items() if not passed))


def _matching_resume_evidence(
    binding: ProductionPublicationBinding,
    authorization: PublicationAuthorization,
) -> bool:
    """Allow token-free issuance only after the exact transaction mutated authority.

    Ordinarily the durable gate proves that mutation began.  There is one
    crash-safe edge after authority commit: the gate has been removed but the
    journal is already marked published (or complete).  That exact journal is
    also sufficient; an active pre-gate journal is deliberately insufficient.
    """
    gate = read_publication_gate(Path(binding.production_root))
    journal_root = Path(binding.control_root) / "journals"
    owners = [
        journal
        for journal in load_transaction_journals(journal_root)
        if _metadata(journal).get("authorization_sha256") == authorization.digest
        and _metadata(journal).get("production_capability_binding_sha256")
        == binding.digest
    ]
    if len(owners) != 1:
        return False
    owner = owners[0]
    if gate is not None:
        return bool(
            gate.get("date_utc") == binding.night_utc
            and gate.get("authorization_sha256") == authorization.digest
            and gate.get("transaction_id") == owner.snapshot.descriptor.run_id
        )
    return bool(
        owner.snapshot.published
        or owner.snapshot.outcome == JournalOutcome.COMPLETE
    )


def issue_production_publication_capability(
    binding: ProductionPublicationBinding,
    authorization: PublicationAuthorization,
    candidate: NightCandidate,
    *,
    control_token: Optional[str],
    sentinel: Mapping[str, Any],
    now: Optional[datetime] = None,
    hostname: Optional[str] = None,
    uid: Optional[int] = None,
    authority_lock: Optional[Mapping[str, Any]] = None,
) -> ProductionPublicationCapability:
    """Issue the exact one-shot June 27 capability or its gated resume.

    Initial issuance requires a 256-bit hex Control token matching only the
    persisted digest. Token-free issuance is limited to deterministic resume
    of the matching durable gate and journal after mutation has begun.
    """
    if type(binding) is not ProductionPublicationBinding:
        raise ProductionAuthorizationUnavailable("Exact production binding is required.")
    if not isinstance(authorization, PublicationAuthorization):
        raise ProductionAuthorizationUnavailable("Exact publication authorization is required.")
    observed_hostname = hostname or socket.getfqdn()
    observed_uid = os.geteuid() if uid is None else int(uid)
    lock_identity = dict(
        authority_lock
        if authority_lock is not None
        else production_authority_lock_identity()
    )
    fixed = {
        "night": binding.night_utc == CONTROL_APPROVED_NIGHT,
        "predecessor": binding.predecessor_night_utc == CONTROL_APPROVED_PREDECESSOR,
        "production": Path(binding.production_root) == PRODUCTION_DATA_ROOT,
        "stage": Path(binding.stage_root) == PRODUCTION_STAGE_ROOT,
        "control": Path(binding.control_root) == PRODUCTION_CONTROL_ROOT,
        "evidence": Path(binding.evidence_root) == PRODUCTION_EVIDENCE_ROOT,
        "candidate_root": Path(binding.candidate_root)
        == CONTROL_APPROVED_CANDIDATE_ROOT,
        "candidate_release": binding.candidate_release_sha
        == CONTROL_APPROVED_CANDIDATE_RELEASE,
    }
    if not all(fixed.values()):
        failed = ",".join(sorted(name for name, passed in fixed.items() if not passed))
        raise ProductionAuthorizationUnavailable(
            f"Production capability is outside the approved June 27 canary: {failed}."
        )
    resume = control_token is None
    if resume and not _matching_resume_evidence(binding, authorization):
        raise ProductionAuthorizationUnavailable(
            "Token-free production capability is limited to the exact gated resume."
        )
    mismatches = qualify_production_binding(
        binding,
        authorization,
        hostname=observed_hostname,
        uid=observed_uid,
        sentinel=sentinel,
        now=now or _utc_now(),
        candidate=candidate,
        authority_lock=lock_identity,
        allow_expired_resume=resume,
    )
    if mismatches:
        raise ProductionAuthorizationUnavailable(
            "Production binding qualification failed: " + ",".join(mismatches)
        )
    if not resume:
        if (
            not isinstance(control_token, str)
            or len(control_token) != 64
            or any(character not in _HEX64 for character in control_token)
        ):
            raise ProductionAuthorizationUnavailable(
                "Control token must be a 256-bit lowercase hexadecimal secret."
            )
        observed_token_sha256 = _sha256_bytes(control_token.encode("ascii"))
        if not hmac.compare_digest(
            observed_token_sha256, binding.control_token_sha256
        ):
            raise ProductionAuthorizationUnavailable("Control token digest differs.")
    return _issue_production_publication_capability(
        run_id=f"production-{CONTROL_APPROVED_NIGHT}-{authorization.digest[:16]}",
        binding=binding.as_dict(),
        binding_sha256=binding.digest,
        authorization_sha256=authorization.digest,
        control_token_sha256=binding.control_token_sha256,
        publisher_release_sha=binding.publisher_release_sha,
    )


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORIZATION_SCHEMA",
    "AuthorityState",
    "CANDIDATE_RECORD_SCHEMA",
    "CumulativePlan",
    "FailureCategory",
    "GATE_SCHEMA",
    "NightCandidate",
    "NightPublisher",
    "PUBLICATION_RECORD_SCHEMA",
    "ProductionPublicationBinding",
    "PublicationAuthorityLock",
    "PublicationAuthorization",
    "PublicationError",
    "PublicationOutcome",
    "PublicationRefused",
    "RETRYABLE_CATEGORIES",
    "authoritative_nights",
    "authorize_publication",
    "build_authoritative_manifest",
    "candidate_chronology_defects",
    "classify_failure",
    "classify_night_authority",
    "issue_production_publication_capability",
    "load_authorization",
    "load_backfill_candidate",
    "load_offline_recovery_candidate",
    "load_transaction_journals",
    "plan_cumulative_extension",
    "production_binding_from_sentinel",
    "production_authority_lock_identity",
    "qualify_production_binding",
    "read_publication_gate",
    "summary_source_manifest",
    "transaction_paths",
    "write_authorization",
]
