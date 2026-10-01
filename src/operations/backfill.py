"""Resumable multi-night backfill: concurrent acquisition, ordered publication.

Each night is an independent scientific unit with its own run root::

    <capability.root>/backfill/nights/night-YYYY-MM-DD/
        request.json                       acquisition request (fixed, prior-free)
        checkpoints/query-result/          sealed query checkpoint (authority)
        checkpoints/live-fetch-v1/         segmented fetch checkpoint (authority)
        candidate/                         validated candidate + candidate-record.json
        events.jsonl                       append-only stage/metric/failure log

Night state is always *derived* from durable evidence, with this precedence:

1. unresolved publication evidence (a production gate or an unfinished
   publication journal) -> ``RECONCILIATION_REQUIRED``;
2. validated ``COMPLETE`` authority (gate absent, journal COMPLETE, manifest
   SHA equal to the journal's) -> ``PUBLISHED``;
3. candidate and acquisition evidence.

A visible manifest never overrides (1).  ``ranges/<start>_<end>.json`` is only
a small derived index rewritten after each change.

Acquisition (query + segmented fetch) runs concurrently across nights and is
prior-free; that is scientifically equivalent only for providers whose
selection ignores prior loci, so it is refused unless the exact provider
implementation carries an attestation in :data:`PRIOR_FREE_ACQUISITION_ATTESTATIONS`.
Construction is chained in date order because overlap validation depends on
every earlier night's loci.  Publication is performed by exactly one writer,
strictly chronologically, and never crosses an unresolved or missing
predecessor.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import pandas as pd

from .fetch_checkpoint import (
    FetchCheckpointBinding,
    FetchCheckpointCompletion,
    FetchObjectResult,
    SegmentedFetchCheckpoint,
)
from .publication import (
    CANDIDATE_RECORD_SCHEMA,
    RETRYABLE_CATEGORIES,
    AuthorityState,
    FailureCategory,
    NightPublisher,
    PublicationAuthorization,
    PublicationOutcome,
    PublicationRefused,
    authorize_publication,
    classify_failure,
    classify_night_authority,
    load_authorization,
    load_backfill_candidate,
    _is_hex64,
    _is_sha40,
    _parse_utc,
    _read_json,
    _write_json_new,
)
from .query_checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    QueryResultCheckpointBindings,
    load_query_result_checkpoint,
    seal_query_result_checkpoint,
)
from .storage import SyntheticWriteCapability, RangeWorkCapability, PublicationRoots


RANGE_SCHEMA = "v3.backfill-range.v1"
RANGE_AUTHORIZATION_SCHEMA = "v3.range-publication-authorization.v2"
MAX_ACQUISITION_CONCURRENCY = 8
MAX_CONSTRUCTION_CONCURRENCY = 2
PUBLICATION_QUIESCENCE_WAIT_SECONDS = 120.0

# The only reviewed prior-free acquisition: V3-G3R established that the
# unchanged LiveAntaresProvider never uses ``prior_locus_ids`` for selection or
# fetching (they feed construction-time overlap validation only).  Any edit to
# the provider module changes its SHA-256 and voids the attestation.
PRIOR_FREE_ACQUISITION_ATTESTATIONS: Tuple[Mapping[str, str], ...] = (
    {
        "provider_name": "live-antares",
        "scenario": "commissioning-v1",
        "provider_module": "src.operations.live_antares",
        "provider_implementation_sha256": (
            "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c"
        ),
        "adapter": "src.operations.live_antares.LiveAntaresProvider",
        "adapter_implementation_sha256": (
            "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c"
        ),
        "evidence": "V3-G6.2: V3-G5 qualification plus non-JSON response-body retry; same prior-free selection and fetch",
    },
)

# Separate from the historical provider attestation: the range adapter is
# reviewed and pinned independently, so an edited adapter fails closed too.
RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS: Tuple[Mapping[str, str], ...] = (
    {
        "provider_name": "live-antares", "scenario": "commissioning-v1",
        "provider_module": "src.operations.live_antares",
        "provider_implementation_sha256": "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c",
        "adapter": "src.operations.production_range.LiveRangeAdapter",
        "adapter_implementation_sha256": "81ce9209f182d61ca5f9c6bdb5c7722c90c09421b78156d1ac36be0a0da78f1c",
        "evidence": "V3-G6.2: V3-G5 per-night prior-free live range adapter; provider re-pinned for response-body retry",
    },
)


class BackfillError(RuntimeError):
    pass


class BackfillRefused(BackfillError):
    pass


class PublicationFailed(BackfillError):
    """One publication attempt ended without COMPLETE authority."""

    def __init__(self, outcome: PublicationOutcome) -> None:
        self.outcome = outcome
        self.category = FailureCategory(
            outcome.record.get("failure_category", FailureCategory.UNCLASSIFIED.value)
        )
        super().__init__(
            f"{outcome.record.get('status')}: {outcome.record.get('message', '')}"
        )


class NightStage(str, Enum):
    PLANNED = "PLANNED"
    QUERYING = "QUERYING"
    QUERY_COMPLETE = "QUERY_COMPLETE"
    FETCHING = "FETCHING"
    FETCH_COMPLETE = "FETCH_COMPLETE"
    CANDIDATE_BUILDING = "CANDIDATE_BUILDING"
    CANDIDATE_VALIDATED = "CANDIDATE_VALIDATED"
    WAITING_FOR_PUBLICATION = "WAITING_FOR_PUBLICATION"
    PUBLISHING = "PUBLISHING"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    PUBLISHED = "PUBLISHED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class BackfillSettings:
    """Operational throughput knobs; none of them changes scientific semantics."""

    acquisition_concurrency: int = 3
    construction_concurrency: int = 1
    publication_concurrency: int = 1
    segment_size: int = 256

    def __post_init__(self) -> None:
        if self.publication_concurrency != 1:
            raise BackfillRefused("Publication concurrency is fixed at exactly one writer.")
        if not 1 <= self.acquisition_concurrency <= MAX_ACQUISITION_CONCURRENCY:
            raise BackfillRefused(
                f"acquisition_concurrency must be 1..{MAX_ACQUISITION_CONCURRENCY}."
            )
        if not 1 <= self.construction_concurrency <= MAX_CONSTRUCTION_CONCURRENCY:
            raise BackfillRefused(
                f"construction_concurrency must be 1..{MAX_CONSTRUCTION_CONCURRENCY}."
            )
        if not 1 <= self.segment_size <= 4096:
            raise BackfillRefused("segment_size must be 1..4096.")

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


class NightAdapter(Protocol):
    """Provider boundary for one night; query/fetch are the only network edges."""

    provider_name: str
    scenario: str

    def execution_policy(self) -> Mapping[str, Any]: ...

    def scientific_contract(self, request: Any) -> Mapping[str, Any]: ...

    def acquisition_request(self, date_utc: str) -> Any: ...

    def query(self, request: Any) -> Any: ...

    def fetch_segment(self, request: Any, locus_ids: Tuple[str, ...]) -> Iterable[FetchObjectResult]: ...

    def construct(
        self,
        request: Any,
        query_result: Any,
        alerts: pd.DataFrame,
        completion: FetchCheckpointCompletion,
    ) -> Any: ...


ReadCapabilityFactory = Callable[[Path, str, str, str], Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identifier_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dates(start: str, end: str) -> Tuple[str, ...]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first.isoformat() != start or last.isoformat() != end or last < first:
        raise BackfillRefused("Range must be canonical YYYY-MM-DD with end >= start.")
    if (last - first).days >= 3660:
        raise BackfillRefused("Backfill ranges are limited to 3660 nights.")
    return tuple((first + timedelta(days=n)).isoformat() for n in range((last - first).days + 1))


def _previous(date_utc: str) -> str:
    return (date.fromisoformat(date_utc) - timedelta(days=1)).isoformat()


def failure_classification(error: BaseException) -> Tuple[FailureCategory, bool]:
    """Explicit category and retryability for a backfill stage failure."""
    category = (
        error.category if isinstance(error, PublicationFailed) else classify_failure(error)
    )
    return category, category in RETRYABLE_CATEGORIES


# ---------------------------------------------------------------------------
# Acquisition identity and the prior-free invariant
# ---------------------------------------------------------------------------


def _module_sha256(module_name: str) -> str:
    module = sys.modules.get(module_name)
    source = inspect.getsourcefile(module) if module is not None else None
    if source is None:
        raise BackfillRefused(f"Implementation of {module_name} cannot be identified.")
    return _sha256(Path(source).read_bytes())


def acquisition_identity(adapter: Any) -> Dict[str, str]:
    """Exact provider/adapter implementation identity used for acquisition."""
    adapter_type = type(adapter)
    provider_module = str(getattr(adapter, "provider_module", adapter_type.__module__))
    return {
        "provider_name": str(adapter.provider_name),
        "scenario": str(adapter.scenario),
        "provider_module": provider_module,
        "provider_implementation_sha256": _module_sha256(provider_module),
        "adapter": f"{adapter_type.__module__}.{adapter_type.__qualname__}",
        "adapter_implementation_sha256": _module_sha256(adapter_type.__module__),
    }


def require_prior_free_acquisition(
    identity: Mapping[str, str],
    attestations: Sequence[Mapping[str, str]] = PRIOR_FREE_ACQUISITION_ATTESTATIONS,
    *,
    science_contract_sha256: Optional[str] = None,
    configuration_sha256: Optional[str] = None,
) -> Mapping[str, str]:
    """Refuse unless the exact provider *and adapter* are attested.

    Range callers also supply the already-derived science/configuration
    identities.  They become part of the returned attestation record that is
    hashed into range authorization before any query or fetch may start.
    """
    keys = (
        "provider_name",
        "scenario",
        "provider_module",
        "provider_implementation_sha256",
        "adapter",
        "adapter_implementation_sha256",
    )
    if (science_contract_sha256 is None) != (configuration_sha256 is None):
        raise BackfillRefused(
            "Prior-free science and configuration identities must be bound together."
        )
    for label, value in (
        ("science contract", science_contract_sha256),
        ("configuration", configuration_sha256),
    ):
        if value is not None and not _is_hex64(value):
            raise BackfillRefused(f"Prior-free {label} identity is malformed.")
    for attestation in attestations:
        if all(attestation.get(key) == identity.get(key) for key in keys):
            bound = dict(attestation)
            if science_contract_sha256 is not None:
                bound["science_contract_sha256"] = science_contract_sha256
                bound["configuration_sha256"] = configuration_sha256
            return bound
    raise BackfillRefused(
        "Prior-free acquisition is not proven equivalent for exact provider/adapter "
        f"{identity.get('provider_name')}/{identity.get('scenario')} "
        f"({identity.get('provider_implementation_sha256')}/"
        f"{identity.get('adapter_implementation_sha256')}); refusing to assume it."
    )


# ---------------------------------------------------------------------------
# Range-level publication authority (chained into per-night authorizations)
# ---------------------------------------------------------------------------

_RANGE_FIELDS = (
    "schema_version",
    "start_date_utc",
    "end_date_utc",
    "initial_predecessor_date_utc",
    "initial_sentinel",
    "science_contract_sha256",
    "configuration_sha256",
    "acquisition_identity",
    "prior_free_attestation_sha256",
    "cache_root",
    "publication_concurrency",
    "candidate_release_sha",
    "publisher_release_sha",
    "authorized_by",
    "authorized_at_utc",
    "expires_at_utc",
)
_INITIAL_SENTINEL_FIELDS = frozenset(
    {"canonical_root", "mount_binding", "durable_fingerprint_sha256", "manifest_count", "cumulative_sha256"}
)
_IDENTITY_FIELDS = frozenset(
    {
        "provider_name", "scenario", "provider_module", "provider_implementation_sha256",
        "adapter", "adapter_implementation_sha256",
    }
)


@dataclass(frozen=True)
class RangePublicationAuthorization:
    """Control permission to publish validated candidates of one range in order.

    It binds the science contract, configuration, exact provider and adapter
    implementations, the prior-free attestation, the initial Sentinel state,
    the cache root and single-writer publication.  Each night receives a
    derived :class:`PublicationAuthorization` bound to its exact candidate and
    the production state its verified predecessor committed.  No production
    range capability is issued in this release.
    """

    start_date_utc: str
    end_date_utc: str
    initial_predecessor_date_utc: str
    initial_sentinel: Mapping[str, Any]
    science_contract_sha256: str
    configuration_sha256: str
    acquisition_identity: Mapping[str, str]
    prior_free_attestation_sha256: str
    cache_root: Optional[str]
    candidate_release_sha: str
    publisher_release_sha: str
    authorized_by: str
    authorized_at_utc: str
    expires_at_utc: str
    publication_concurrency: int = 1
    schema_version: str = RANGE_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        _dates(self.start_date_utc, self.end_date_utc)
        sentinel = self.initial_sentinel
        if (
            self.schema_version != RANGE_AUTHORIZATION_SCHEMA
            or self.initial_predecessor_date_utc != _previous(self.start_date_utc)
            or not isinstance(sentinel, Mapping)
            or set(sentinel) != _INITIAL_SENTINEL_FIELDS
            or not _is_hex64(sentinel.get("durable_fingerprint_sha256"))
            or not _is_hex64(self.science_contract_sha256)
            or not _is_hex64(self.configuration_sha256)
            or not isinstance(self.acquisition_identity, Mapping)
            or set(self.acquisition_identity) != _IDENTITY_FIELDS
            or not _is_hex64(self.prior_free_attestation_sha256)
            or (self.cache_root is not None and not Path(self.cache_root).is_absolute())
            or self.publication_concurrency != 1
            or not _is_sha40(self.candidate_release_sha)
            or not _is_sha40(self.publisher_release_sha)
            or not isinstance(self.authorized_by, str)
            or not self.authorized_by.strip()
        ):
            raise BackfillRefused("Range publication authorization is malformed.")
        if _parse_utc(self.authorized_at_utc, "authorized_at_utc") >= _parse_utc(
            self.expires_at_utc, "expires_at_utc"
        ):
            raise BackfillRefused("Range publication authorization has no validity.")

    def as_dict(self) -> Dict[str, Any]:
        return json.loads(_canonical({name: getattr(self, name) for name in _RANGE_FIELDS}))

    @property
    def digest(self) -> str:
        return _sha256(_canonical(self.as_dict()))


# ---------------------------------------------------------------------------
# Per-night workspace and derived state
# ---------------------------------------------------------------------------


class NightWorkspace:
    def __init__(self, nights_root: Path, date_utc: str) -> None:
        self.date_utc = date_utc
        self.run_id = f"night-{date_utc}"
        self.root = Path(nights_root) / self.run_id
        self.request_path = self.root / "request.json"
        self.query_checkpoint = self.root / "checkpoints" / DEFAULT_CHECKPOINT_NAME
        self.fetch_root = self.root / "checkpoints" / "live-fetch-v1"
        self.candidate = self.root / "candidate"
        self.events = self.root / "events.jsonl"

    def ensure(self) -> None:
        proposed = self.root.absolute()
        if proposed.resolve(strict=False) != proposed:
            raise BackfillRefused("Night work root rejects path aliases and symlinks.")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def append_event(self, event: str, **details: Any) -> None:
        self.ensure()
        line = _canonical({"event": event, "utc": _utc_now().isoformat(), **details}) + b"\n"
        descriptor = os.open(str(self.events), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read_events(self) -> List[Dict[str, Any]]:
        if not self.events.is_file():
            return []
        events = []
        for line in self.events.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"event": "unreadable_event_line"})
        return events

    def query_committed(self) -> bool:
        return (self.query_checkpoint / "COMMITTED.json").is_file()

    def fetch_complete(self) -> bool:
        return (self.fetch_root / "fetch-complete.json").is_file()

    def committed_segments(self) -> int:
        segments = self.fetch_root / "segments"
        return len(list(segments.glob("*.commit.json"))) if segments.is_dir() else 0

    def candidate_valid(self) -> bool:
        try:
            load_backfill_candidate(self.root)
        except PublicationRefused:
            return False
        return True


def derive_night_state(
    workspace: NightWorkspace,
    data_root: Path,
    journal_root: Path,
    evidence_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Derive one night's stage purely from durable evidence (see module docstring)."""
    events = workspace.read_events()
    last = events[-1] if events else {}
    try:
        authority = classify_night_authority(data_root, journal_root, workspace.date_utc)
    except PublicationRefused as refusal:
        authority = {"state": AuthorityState.CONTRADICTION.value, "reason": refusal.code}
    record_present = None
    if authority.get("transaction_id") and evidence_root is not None:
        record_present = (
            Path(evidence_root) / "publications" / workspace.date_utc
            / f"{authority['transaction_id']}.json"
        ).is_file()
    state_name = authority["state"]
    evidence = {
        "authority_state": state_name,
        "authority": authority,
        "published": False,
        "terminal_record_present": record_present,
        "candidate_valid": workspace.candidate_valid(),
        "fetch_complete": workspace.fetch_complete(),
        "query_committed": workspace.query_committed(),
        "committed_segments": workspace.committed_segments(),
    }
    blocked = None
    if state_name == AuthorityState.RECONCILIATION_REQUIRED.value or (
        state_name == AuthorityState.COMPLETE.value and not authority.get("finalized")
    ):
        stage = NightStage.RECONCILIATION_REQUIRED
    elif state_name == AuthorityState.CONTRADICTION.value:
        stage = NightStage.BLOCKED
        blocked = {
            "stage": NightStage.PUBLISHING.value,
            "error_type": "AuthorityContradiction",
            "message": str(authority.get("reason")),
            "failure_category": FailureCategory.JOURNAL_CONTRADICTION.value,
            "retryable": False,
            "segments_committed": evidence["committed_segments"],
        }
    elif state_name == AuthorityState.COMPLETE.value:
        stage = NightStage.PUBLISHED
        evidence["published"] = True
    elif last.get("event") == "failure":
        stage = NightStage.BLOCKED
    elif evidence["candidate_valid"]:
        stage = NightStage.WAITING_FOR_PUBLICATION
    elif evidence["fetch_complete"]:
        stage = NightStage.FETCH_COMPLETE
    elif evidence["query_committed"]:
        stage = NightStage.FETCHING if evidence["committed_segments"] else NightStage.QUERY_COMPLETE
    else:
        stage = NightStage.PLANNED
    state = {"date_utc": workspace.date_utc, "stage": stage.value, "evidence": evidence}
    if stage == NightStage.BLOCKED:
        state["blocked"] = blocked or {
            key: last.get(key)
            for key in (
                "stage", "error_type", "message", "failure_category", "retryable",
                "segments_committed",
            )
        }
    elif last.get("event") == "failure" and stage == NightStage.RECONCILIATION_REQUIRED:
        state["last_failure"] = {
            key: last.get(key) for key in ("stage", "error_type", "failure_category", "retryable")
        }
    state["metrics"] = _night_metrics(events)
    return state


def _night_metrics(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"retries": 0, "failures": 0}
    for event in events:
        name = event.get("event")
        if name == "failure":
            metrics["failures"] += 1
            metrics["failure_stage"] = event.get("stage")
        elif name == "retry":
            metrics["retries"] += 1
        elif name == "metrics":
            for key, value in event.items():
                if key in {"event", "utc", "stage"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool) and key in metrics:
                    metrics[key] = round(metrics[key] + value, 6)
                else:
                    metrics[key] = value
    return metrics


def inspect_backfill(
    run_root: Path,
    data_root: Path,
    start: str,
    end: str,
    *,
    journal_root: Optional[Path] = None,
    evidence_root: Optional[Path] = None,
    work_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inspect a complete range while publication is serialized."""
    from .. import history

    with history.authority_read_lock(data_root):
        return _inspect_backfill_locked(
            run_root,
            data_root,
            start,
            end,
            journal_root=journal_root,
            evidence_root=evidence_root,
            work_root=work_root,
        )


def _inspect_backfill_locked(
    run_root: Path,
    data_root: Path,
    start: str,
    end: str,
    *,
    journal_root: Optional[Path] = None,
    evidence_root: Optional[Path] = None,
    work_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read-only range state reconstructed from per-night evidence.

    ``run_root`` owns ``backfill/`` (and, for synthetic runs, ``control/`` and
    ``evidence/``); ``data_root`` is the authoritative data root.  Nothing is
    created or modified.
    """
    run_root = Path(run_root)
    journal_root = Path(journal_root) if journal_root else run_root / "control" / "journals"
    evidence_root = Path(evidence_root) if evidence_root else run_root / "evidence"
    nights_root = (Path(work_root) if work_root is not None else run_root / "backfill") / "nights"
    nights = [
        derive_night_state(
            NightWorkspace(nights_root, day), Path(data_root), journal_root, evidence_root
        )
        for day in _dates(start, end)
    ]
    return _range_document(start, end, nights, None, None)


def _range_document(
    start: str,
    end: str,
    nights: Sequence[Mapping[str, Any]],
    settings: Optional[BackfillSettings],
    wall_seconds: Optional[float],
) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for night in nights:
        counts[night["stage"]] = counts.get(night["stage"], 0) + 1
    published = counts.get(NightStage.PUBLISHED.value, 0)
    blocked = counts.get(NightStage.BLOCKED.value, 0)
    unresolved = counts.get(NightStage.RECONCILIATION_REQUIRED.value, 0)
    in_flight = len(nights) - published - blocked
    first_unpublished = next(
        (night["date_utc"] for night in nights if night["stage"] != NightStage.PUBLISHED.value),
        None,
    )
    summary: Dict[str, Any] = {
        "nights_total": len(nights),
        "nights_published": published,
        "nights_blocked": blocked,
        "nights_reconciliation_required": unresolved,
        "nights_in_flight": in_flight,
        "backlog": len(nights) - published,
        "stage_counts": dict(sorted(counts.items())),
        "publication_frontier": first_unpublished,
    }
    if wall_seconds is not None:
        summary["wall_seconds"] = round(wall_seconds, 6)
        summary["seconds_per_published_night"] = (
            round(wall_seconds / published, 6) if published else None
        )
    return {
        "schema_version": RANGE_SCHEMA,
        "start_date_utc": start,
        "end_date_utc": end,
        "derived_from": "per-night evidence; safe to delete and regenerate",
        "settings": settings.as_dict() if settings else None,
        "summary": summary,
        "nights": list(nights),
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class _ControllerLock:
    """Kernel-released exclusive lock: a crash or reboot never strands it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.descriptor: Optional[int] = None

    def __enter__(self) -> "_ControllerLock":
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise BackfillRefused("Another backfill controller is active.") from exc
        self.descriptor = descriptor
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


class BackfillController:
    def __init__(
        self,
        capability: SyntheticWriteCapability,
        adapter: NightAdapter,
        *,
        release_sha: str,
        read_capability_factory: Optional[ReadCapabilityFactory] = None,
        settings: BackfillSettings = BackfillSettings(),
        publisher: Optional[NightPublisher] = None,
        range_authorization: Optional[RangePublicationAuthorization] = None,
        cache: Any = None,
        event_hook: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
        prior_free_attestations: Sequence[Mapping[str, str]] = (PRIOR_FREE_ACQUISITION_ATTESTATIONS + RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS),
        work_capability: Optional[RangeWorkCapability] = None,
        publication_roots: Optional[PublicationRoots] = None,
    ) -> None:
        if work_capability is None:
            if type(capability) is not SyntheticWriteCapability:
                raise BackfillRefused("Backfill requires separate work authority or a synthetic capability.")
            work_root = capability.root / "backfill"
        else:
            if (capability is not None or type(work_capability) is not RangeWorkCapability
                    or type(publication_roots) is not PublicationRoots):
                raise BackfillRefused("Range work and publication observation roots must be explicit and separate.")
            work_root = work_capability.root
            capability = publication_roots
            for root in (capability.published_root, capability.staging_root,
                         capability.journal_root.parent, capability.evidence_root):
                if work_root == root or work_root in root.parents or root in work_root.parents:
                    raise BackfillRefused("Range work overlaps publication authority.")
        if not _is_sha40(release_sha):
            raise BackfillRefused("release_sha must be a full commit SHA.")
        if range_authorization is not None and publisher is None:
            raise BackfillRefused("Range authorization requires a publisher.")
        if publisher is not None and getattr(publisher, "roots", publisher.capability) != capability:
            raise BackfillRefused("Publisher and backfill capabilities differ.")
        if range_authorization is not None and (
            range_authorization.candidate_release_sha != release_sha
            or range_authorization.publisher_release_sha != publisher.publisher_release_sha
        ):
            raise BackfillRefused("Range authorization release identities differ.")
        if cache is not None:
            cache.require_disjoint(
                (
                    work_root, capability.published_root, capability.staging_root,
                    capability.journal_root, capability.lock_root, capability.evidence_root,
                )
            )
        self.capability = capability
        self.adapter = adapter
        self.release_sha = release_sha
        self.read_capability_factory = read_capability_factory
        self.settings = settings
        self.publisher = publisher
        if type(publisher) is NightPublisher:
            # Construction reads may briefly overlap the single writer. Wait
            # for their existing authority lock instead of failing the range.
            publisher.lock_wait_seconds = max(publisher.lock_wait_seconds, PUBLICATION_QUIESCENCE_WAIT_SECONDS)
        self.range_authorization = range_authorization
        self.cache = cache
        self.event_hook = event_hook
        self.prior_free_attestations = tuple(prior_free_attestations)
        self.work_root = work_root
        self.nights_root = self.work_root / "nights"
        self._range_lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def workspace(self, date_utc: str) -> NightWorkspace:
        return NightWorkspace(self.nights_root, date_utc)

    def night_state(self, date_utc: str) -> Dict[str, Any]:
        return derive_night_state(
            self.workspace(date_utc),
            self.capability.published_root,
            self.capability.journal_root,
            self.capability.evidence_root,
        )

    def _hook(self, point: str, **details: Any) -> None:
        if self.event_hook is not None:
            self.event_hook(point, details)

    def _configuration_hash(self) -> str:
        return _sha256(
            _canonical(
                {
                    "release_sha": self.release_sha,
                    "provider": self.adapter.provider_name,
                    "scenario": self.adapter.scenario,
                    "execution_policy": dict(self.adapter.execution_policy()),
                    "segment_size": self.settings.segment_size,
                }
            )
        )

    def science_contract_sha256(self, start: str, end: str) -> str:
        """Identity of the exact per-night scientific requests over the range."""
        return _sha256(
            _canonical(
                [
                    dict(self.adapter.scientific_contract(self.adapter.acquisition_request(day)))
                    for day in _dates(start, end)
                ]
            )
        )

    def range_binding(self, start: str, end: str) -> Dict[str, Any]:
        """The identities a range authorization must bind for this controller."""
        identity = acquisition_identity(self.adapter)
        science_contract_sha256 = self.science_contract_sha256(start, end)
        configuration_sha256 = self._configuration_hash()
        attestation = require_prior_free_acquisition(
            identity,
            self.prior_free_attestations,
            science_contract_sha256=science_contract_sha256,
            configuration_sha256=configuration_sha256,
        )
        return {
            "science_contract_sha256": science_contract_sha256,
            "configuration_sha256": configuration_sha256,
            "acquisition_identity": identity,
            "prior_free_attestation_sha256": _sha256(_canonical(dict(attestation))),
            "cache_root": str(self.cache.root) if self.cache is not None else None,
            "publication_concurrency": self.settings.publication_concurrency,
        }

    def _verify_range_authorization(self, start: str, end: str, *, recovery_only: bool = False) -> None:
        authority = self.range_authorization
        if authority is None:
            return
        if authority.start_date_utc != start or authority.end_date_utc != end:
            raise BackfillRefused("Range authorization covers a different range.")
        observed = self.range_binding(start, end)
        bound = {key: authority.as_dict()[key] for key in observed}
        if observed != bound:
            differing = sorted(key for key in observed if observed[key] != bound[key])
            raise BackfillRefused(f"Range authorization does not bind this run: {differing}.")
        if not recovery_only and _utc_now() >= _parse_utc(authority.expires_at_utc, "expires_at_utc"):
            raise BackfillRefused("Range publication authorization has expired.")

    def _request(self, workspace: NightWorkspace):
        from .science import NightScienceRequest

        request = self.adapter.acquisition_request(workspace.date_utc)
        if request.date_utc != workspace.date_utc or request.prior_locus_ids:
            raise BackfillError("Acquisition requests are per-night and prior-free.")
        document = {
            "date_utc": request.date_utc,
            "mjd_min": request.mjd_min,
            "mjd_max": request.mjd_max,
            "ingested_at_utc": request.ingested_at_utc,
            "query_tag": request.query_tag,
            "target_loci": request.target_loci,
            "range_label": request.range_label,
        }
        if workspace.request_path.is_file():
            persisted = _read_json(workspace.request_path)
            if persisted != document:
                raise BackfillError("Persisted acquisition request differs.")
        else:
            workspace.ensure()
            _write_json_new(workspace.request_path, document)
        return NightScienceRequest(**document)

    def _query_bindings(self, workspace: NightWorkspace, request) -> QueryResultCheckpointBindings:
        return QueryResultCheckpointBindings(
            run_id=workspace.run_id,
            release_sha=self.release_sha,
            configuration_hash=self._configuration_hash(),
            target_date_utc=workspace.date_utc,
            provider_name=self.adapter.provider_name,
            provider_scenario=self.adapter.scenario,
            query_policy={
                "scientific_contract": dict(self.adapter.scientific_contract(request)),
                "execution_policy": dict(self.adapter.execution_policy()),
            },
        )

    def _fetch_binding(self, workspace: NightWorkspace, request, loaded) -> FetchCheckpointBinding:
        details = loaded.query_result.evidence.details
        return FetchCheckpointBinding(
            run_id=workspace.run_id,
            release_sha=self.release_sha,
            configuration_sha256=self._configuration_hash(),
            target_date_utc=workspace.date_utc,
            mjd_min=request.mjd_min,
            mjd_max=request.mjd_max,
            provider_name=self.adapter.provider_name,
            provider_scenario=self.adapter.scenario,
            provider_policy_sha256=_sha256(_canonical(dict(self.adapter.execution_policy()))),
            query_contract_sha256=str(details.get("query_contract_sha256", "")),
            query_identity_sha256=loaded.integrity_sha256,
            query_locus_order_sha256=str(details.get("locus_order_sha256", "")),
            expected_objects=len(self._ordered_ids(loaded)),
            segment_size=self.settings.segment_size,
        )

    @staticmethod
    def _ordered_ids(loaded) -> List[str]:
        loci = loaded.query_result.loci
        if loci is None or loci.empty:
            return []
        return loci["locus_id"].astype(str).tolist()

    def _fail(self, workspace: NightWorkspace, stage: str, error: BaseException) -> Dict[str, Any]:
        category, retryable = failure_classification(error)
        failure = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error)[:500],
            "failure_category": category.value,
            "retryable": retryable,
            "segments_committed": workspace.committed_segments(),
        }
        if isinstance(error, PublicationFailed):
            failure["authority_state"] = error.outcome.record.get("authority_state")
        workspace.append_event("failure", **failure)
        return failure

    # -- stage 1: acquisition (concurrent) ----------------------------------

    def acquire(self, date_utc: str) -> Dict[str, Any]:
        """Query (or reuse the sealed query) and fetch only missing segments."""
        workspace = self.workspace(date_utc)
        stage = NightStage.QUERYING.value
        try:
            request = self._request(workspace)
            bindings = self._query_bindings(workspace, request)
            started = time.monotonic()
            if workspace.query_committed():
                loaded = load_query_result_checkpoint(workspace.root, request, bindings)
                workspace.append_event("query_reused", stage=NightStage.QUERY_COMPLETE.value)
            else:
                if workspace.query_checkpoint.exists():
                    # A crash before COMMITTED.json leaves uncommitted evidence the
                    # loader always refuses.  Preserve it aside; never delete.
                    aside = workspace.query_checkpoint.with_name(
                        f"{DEFAULT_CHECKPOINT_NAME}.uncommitted-{uuid.uuid4().hex[:8]}"
                    )
                    os.rename(workspace.query_checkpoint, aside)
                    workspace.append_event("query_uncommitted_preserved", path=str(aside))
                if self.read_capability_factory is None:
                    raise BackfillRefused("No live-read authority is issued for acquisition.")
                workspace.append_event("stage", stage=stage)
                self._hook("before_query", date_utc=date_utc)
                resumable_query = getattr(self.adapter, "query_resumable", None)
                query_result = (
                    resumable_query(request, bindings, event_hook=lambda event, details: self._hook(event, date_utc=date_utc, **details))
                    if resumable_query is not None else self.adapter.query(request)
                )
                query_result.require_completed()
                seal_query_result_checkpoint(workspace.root, query_result, bindings)
                loaded = load_query_result_checkpoint(workspace.root, request, bindings)
                workspace.append_event(
                    "metrics", stage=NightStage.QUERY_COMPLETE.value,
                    query_seconds=round(time.monotonic() - started, 6),
                    queried=1,
                )
            stage = NightStage.FETCHING.value
            ids = self._ordered_ids(loaded)
            binding = self._fetch_binding(workspace, request, loaded)
            if workspace.fetch_complete() and self.read_capability_factory is None:
                checkpoint = SegmentedFetchCheckpoint.open_read_only(workspace.root, binding)
                completion = checkpoint.inspect_complete(ids)
            else:
                if self.read_capability_factory is None:
                    raise BackfillRefused("No live-read authority is issued for acquisition.")
                capability = self.read_capability_factory(
                    workspace.root, workspace.run_id, date_utc, self.release_sha
                )
                checkpoint = SegmentedFetchCheckpoint.open(capability, binding)
                fetcher: Callable[[Tuple[str, ...]], Any] = (
                    lambda locus_ids: self.adapter.fetch_segment(request, locus_ids)
                )
                if self.cache is not None:
                    fetcher = self.cache.wrap(binding, fetcher)
                workspace.append_event("stage", stage=stage)
                fetch_started = time.monotonic()
                completion = checkpoint.fetch_missing(
                    ids,
                    fetcher,
                    event_hook=lambda event, details: self._hook(
                        f"fetch:{event}", date_utc=date_utc, **dict(details)
                    ),
                )
                workspace.append_event(
                    "metrics", stage=NightStage.FETCH_COMPLETE.value,
                    fetch_seconds=round(time.monotonic() - fetch_started, 6),
                    segments_total=completion.segment_count,
                    segments_reused=completion.reused_segments,
                    segments_fetched=completion.fetched_segments,
                    fetch_retry_count=completion.retry_count,
                    alert_rows=completion.alert_rows,
                    checkpoint_bytes=_tree_bytes(workspace.root / "checkpoints"),
                )
            return {"ok": True, "completion": completion.as_dict()}
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {"ok": False, "failure": self._fail(workspace, stage, error)}

    # -- stage 2: construction (chained) ------------------------------------

    def _read_prior_index(self, date_utc: str, mjd_min: float) -> pd.DataFrame:
        """Read committed history; wait out a short in-flight publication window."""
        from .. import history

        deadline = time.monotonic() + PUBLICATION_QUIESCENCE_WAIT_SECONDS
        while True:
            try:
                return history.load_cumulative_loci_index(
                    self.capability.published_root, before_mjd=mjd_min, before_date=date_utc
                )
            except history.PublicationInProgress:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

    def prior_locus_ids(self, date_utc: str, in_range_before: Sequence[str]) -> Tuple[str, ...]:
        """Published history before the night plus validated unpublished predecessors.

        Ordered exactly as the cumulative index will hold them after the
        predecessors publish, so the identity is re-provable at publication.
        """
        request = self.adapter.acquisition_request(date_utc)
        prior = self._read_prior_index(date_utc, request.mjd_min)
        ids = prior["locus_id"].dropna().astype(str).tolist() if "locus_id" in prior.columns else []
        published = set(prior["night_date_utc"].astype(str)) if "night_date_utc" in prior.columns else set()
        for day in in_range_before:
            if day in published:
                continue
            candidate = load_backfill_candidate(self.workspace(day).root)
            loci = pd.read_parquet(candidate.candidate_dir / "loci.parquet")
            ids.extend(sorted(loci["locus_id"].dropna().astype(str).tolist()))
        return tuple(ids)

    def construct(self, date_utc: str, predecessors: Sequence[str], *, supersede: bool = False) -> Dict[str, Any]:
        """Build and validate the candidate from sealed acquisition only."""
        from .science import build_night_artifacts, reopen_and_validate_artifacts

        workspace = self.workspace(date_utc)
        stage = NightStage.CANDIDATE_BUILDING.value
        try:
            if workspace.candidate.exists():
                if not supersede and workspace.candidate_valid():
                    return {"ok": True, "reused": True}
                aside = workspace.root / f"candidate.superseded-{uuid.uuid4().hex[:8]}"
                os.rename(workspace.candidate, aside)
                workspace.append_event("candidate_superseded", path=str(aside))
            started = time.monotonic()
            workspace.append_event("stage", stage=stage)
            request = self._request(workspace)
            bindings = self._query_bindings(workspace, request)
            loaded = load_query_result_checkpoint(workspace.root, request, bindings)
            ids = self._ordered_ids(loaded)
            binding = self._fetch_binding(workspace, request, loaded)
            checkpoint = SegmentedFetchCheckpoint.open_read_only(workspace.root, binding)
            completion = checkpoint.inspect_complete(ids)
            alerts = checkpoint.reconstruct_alerts(ids)
            prior = self.prior_locus_ids(date_utc, predecessors)
            construction_request = dataclasses.replace(request, prior_locus_ids=prior)
            query_result = dataclasses.replace(loaded.query_result, request=construction_request)
            self._hook("before_construct", date_utc=date_utc)
            checkpoint_constructor = getattr(self.adapter, "construct_checkpoint", None)
            result = (checkpoint_constructor(construction_request, query_result, checkpoint)
                      if checkpoint_constructor is not None else
                      self.adapter.construct(construction_request, query_result, alerts, completion))
            result.require_publishable()
            artifacts = build_night_artifacts(result)
            reopen_and_validate_artifacts(artifacts, expected=result)
            temporary = workspace.root / f"candidate.tmp-{uuid.uuid4().hex[:8]}"
            temporary.mkdir(mode=0o700)
            for name, payload in artifacts.items():
                (temporary / name).write_bytes(payload)
            manifest = json.loads(artifacts["manifest.json"])
            record = {
                "schema_version": CANDIDATE_RECORD_SCHEMA,
                "date_utc": date_utc,
                "release_sha": self.release_sha,
                "artifacts": {
                    name: {"bytes": len(payload), "sha256": _sha256(payload)}
                    for name, payload in artifacts.items()
                },
                "loci": manifest["actual_loci"],
                "alerts": manifest["alert_rows"],
                "validation_passed": True,
                "authoritative": False,
                "constructed_at_utc": _utc_now().isoformat(),
                "prior_locus_identity_sha256": _identifier_hash(prior),
                "provenance": {
                    "range_authorization_sha256": self.range_authorization.digest if self.range_authorization else None,
                    "configuration_sha256": self._configuration_hash(),
                    "night_query_contract_sha256": loaded.query_result.evidence.details["query_contract_sha256"],
                    "binding_sha256": _sha256(_canonical({
                        "request": _read_json(workspace.request_path),
                        "query_identity": loaded.integrity_sha256,
                        "fetch_identity": binding.identity_sha256,
                        "prior_identity": _identifier_hash(prior),
                    })),
                    "night_run_id": workspace.run_id,
                    "provider": self.adapter.provider_name,
                    "scenario": self.adapter.scenario,
                    "query_identity": loaded.integrity_sha256,
                    "fetch_identity": binding.identity_sha256,
                    "fetch_completion_sha256": completion.completion_sha256,
                    "segments": completion.segment_count,
                    "prior_locus_count": len(prior),
                    "prior_locus_identity_sha256": _identifier_hash(prior),
                },
            }
            _write_json_new(temporary / "candidate-record.json", record)
            os.rename(temporary, workspace.candidate)  # candidate appears atomically
            workspace.append_event(
                "metrics", stage=NightStage.CANDIDATE_VALIDATED.value,
                construction_seconds=round(time.monotonic() - started, 6),
                candidate_bytes=sum(len(p) for p in artifacts.values()),
                constructed=1,
            )
            return {"ok": True, "reused": False}
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {"ok": False, "failure": self._fail(workspace, stage, error)}

    # -- stage 3: ordered publication ---------------------------------------

    def _predecessor_fingerprint(self, date_utc: str) -> str:
        """Production state the verified predecessor committed (journal evidence)."""
        assert self.range_authorization is not None
        predecessor = _previous(date_utc)
        if predecessor == self.range_authorization.initial_predecessor_date_utc:
            return str(self.range_authorization.initial_sentinel["durable_fingerprint_sha256"])
        state = self.night_state(predecessor)
        fingerprint = state["evidence"]["authority"].get("resulting_production_fingerprint")
        if state["stage"] != NightStage.PUBLISHED.value or not _is_hex64(fingerprint):
            raise PublicationRefused(
                "predecessor_gap", "No verified COMPLETE predecessor publication."
            )
        return str(fingerprint)

    def _night_authorization(self, date_utc: str, candidate) -> PublicationAuthorization:
        """Derive (once) the per-night authorization chained from the range authority."""
        assert self.range_authorization is not None and self.publisher is not None
        authority = self.range_authorization
        authorized_by = f"range:{authority.digest}"
        directory = self.work_root / "authorizations"
        if directory.is_dir():
            for path in sorted(directory.glob(f"{date_utc}-*.json")):
                existing = load_authorization(path)
                if (
                    existing.candidate_record_sha256 == candidate.record_sha256
                    and existing.authorized_by == authorized_by
                ):
                    return existing
        baseline = self._predecessor_fingerprint(date_utc)
        inputs = self.publisher.authorization_inputs(candidate)
        production = inputs["production"]
        if production["durable_fingerprint_sha256"] != baseline:
            raise PublicationRefused(
                "sentinel_drift", "Production differs from the predecessor's committed state."
            )
        if _previous(date_utc) == authority.initial_predecessor_date_utc:
            initial = {key: production[key] for key in authority.initial_sentinel}
            if initial != dict(authority.initial_sentinel):
                raise PublicationRefused(
                    "sentinel_drift", "Production differs from the range's initial Sentinel state."
                )
        authorization = authorize_publication(
            candidate,
            production=production,
            predecessor_date_utc=_previous(date_utc),
            publisher_release_sha=self.publisher.publisher_release_sha,
            expected_cumulative_sha256=inputs["expected_cumulative_sha256"],
            authorized_by=authorized_by,
            authorized_at_utc=_utc_now().isoformat(),
            expires_at_utc=authority.expires_at_utc,
            nonce=_sha256(
                f"{authority.digest}:{date_utc}:{candidate.record_sha256}".encode("utf-8")
            )[:32],
        )
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        _write_json_new(directory / f"{date_utc}-{authorization.digest[:16]}.json", authorization.as_dict())
        return authorization

    def _current_prior_identity(self, date_utc: str) -> str:
        return _identifier_hash(self.prior_locus_ids(date_utc, ()))

    def _publish_with(self, workspace: NightWorkspace, candidate, authorization) -> Dict[str, Any]:
        workspace.append_event("stage", stage=NightStage.PUBLISHING.value)
        self._hook("before_publish", date_utc=workspace.date_utc)
        started = time.monotonic()
        outcome = self.publisher.publish(candidate, authorization)
        if not outcome.success:
            raise PublicationFailed(outcome)
        workspace.append_event(
            "metrics", stage=NightStage.PUBLISHED.value,
            publication_seconds=round(time.monotonic() - started, 6),
            publication_status=outcome.status,
            published=1 if outcome.status == "published" else 0,
            publication_record=str(outcome.record_path),
        )
        return {"ok": True, "status": outcome.status}

    def publish(self, date_utc: str, predecessors: Sequence[str]) -> Dict[str, Any]:
        workspace = self.workspace(date_utc)
        stage = NightStage.PUBLISHING.value
        try:
            candidate = load_backfill_candidate(workspace.root)
            record = _read_json(candidate.record_path)
            if record.get("prior_locus_identity_sha256") != self._current_prior_identity(date_utc):
                # Construction assumed a predecessor state that production does
                # not hold; rebuild from sealed acquisition (never re-acquire).
                rebuilt = self.construct(date_utc, (), supersede=True)
                if not rebuilt["ok"]:
                    return rebuilt
                candidate = load_backfill_candidate(workspace.root)
            authorization = self._night_authorization(date_utc, candidate)
            return self._publish_with(workspace, candidate, authorization)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {"ok": False, "failure": self._fail(workspace, stage, error)}

    def resume_publication(self, date_utc: str) -> Dict[str, Any]:
        """Deterministically finish an unresolved publication: no acquisition, no rebuild."""
        workspace = self.workspace(date_utc)
        try:
            candidate = load_backfill_candidate(workspace.root)
            authorization = self.publisher.pending_authorization(date_utc)
            if authorization is None:
                raise PublicationRefused(
                    "journal_contradiction", "Unresolved publication has no journaled authorization."
                )
            if authorization.authorized_by != f"range:{self.range_authorization.digest}":
                raise PublicationRefused(
                    "authorization_binding_mismatch",
                    "The unresolved publication belongs to another authority.",
                )
            return self._publish_with(workspace, candidate, authorization)
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {
                "ok": False,
                "failure": self._fail(workspace, NightStage.RECONCILIATION_REQUIRED.value, error),
            }

    def _emit_terminal_evidence(self, date_utc: str) -> None:
        """COMPLETE authority whose terminal record was lost: idempotent replay."""
        state = self.night_state(date_utc)
        authority = state["evidence"]["authority"]
        if state["evidence"].get("terminal_record_present") is not False:
            return
        for journal_path in sorted(self.capability.journal_root.glob("*.json")):
            if journal_path.stem != authority.get("transaction_id"):
                continue
            from .journal import TransactionJournal

            metadata = TransactionJournal.load(journal_path).snapshot.descriptor.metadata
            authorization = PublicationAuthorization.from_dict(metadata["authorization"])
            candidate = load_backfill_candidate(self.workspace(date_utc).root)
            outcome = self.publisher.publish(candidate, authorization)
            if not outcome.success:
                raise PublicationFailed(outcome)

    # -- orchestration -------------------------------------------------------

    def _range_path(self, start: str, end: str) -> Path:
        return self.work_root / "ranges" / f"{start}_{end}.json"

    def _write_range(self, start: str, end: str, wall_seconds: float) -> Dict[str, Any]:
        from .commissioning import _write_json_atomic
        from .. import history

        with history.authority_read_lock(self.capability.published_root):
            nights = [self.night_state(day) for day in _dates(start, end)]
        document = _range_document(start, end, nights, self.settings, wall_seconds)
        document["publication_authority"] = (
            self.range_authorization.digest if self.range_authorization else None
        )
        document["cache"] = self.cache.statistics() if self.cache is not None else None
        # Range provenance: the detached Control approval activating publication.
        document["control_approval_sha256"] = getattr(self.publisher, "control_approval_sha256", None)
        with self._range_lock:
            path = self._range_path(start, end)
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _write_json_atomic(path, document)
        return document

    def run(self, start: str, end: str, *, resume: bool = False, recovery_only: bool = False) -> Dict[str, Any]:
        dates = _dates(start, end)
        # Exact provider+adapter and science/configuration identities must be
        # proven before any query or fetch.  Range authorization then binds the
        # same composite record when publication is enabled.
        self.range_binding(start, end)
        self._verify_range_authorization(start, end, recovery_only=recovery_only)
        wall_started = time.monotonic()
        with _ControllerLock(self.work_root / "controller.lock"):
            if not resume and any(self.workspace(day).root.exists() for day in dates):
                raise BackfillRefused("Night evidence already exists; use resume.")
            from .. import history

            with history.authority_read_lock(self.capability.published_root):
                states = {day: self.night_state(day) for day in dates}
            if recovery_only:
                if not resume or self.publisher is None or self.range_authorization is None:
                    raise BackfillRefused("Recovery-only requires an existing authorized publication and resume.")
                for day, state in states.items():
                    if state["stage"] == NightStage.PUBLISHED.value:
                        self._emit_terminal_evidence(day)
                    elif state["stage"] == NightStage.RECONCILIATION_REQUIRED.value:
                        if not self.resume_publication(day)["ok"]:
                            break
                    else:
                        break  # Never start acquisition/construction/a new publication.
                return self._write_range(start, end, time.monotonic() - wall_started)
            for day, state in states.items():
                if state["stage"] == NightStage.BLOCKED.value:
                    if state["blocked"].get("retryable"):
                        self.workspace(day).append_event("retry", stage=state["blocked"].get("stage"))
                        states[day] = self.night_state(day)
            needs_acquisition = [
                day for day in dates
                if states[day]["stage"] in {
                    NightStage.PLANNED.value, NightStage.QUERY_COMPLETE.value,
                    NightStage.FETCHING.value, NightStage.FETCH_COMPLETE.value,
                }
                and not states[day]["evidence"]["fetch_complete"]
            ]
            acquisitions: Dict[str, Future] = {}
            acquisition_pool = ThreadPoolExecutor(
                max_workers=self.settings.acquisition_concurrency,
                thread_name_prefix="backfill-acquire",
            )
            construction_pool = ThreadPoolExecutor(
                max_workers=self.settings.construction_concurrency,
                thread_name_prefix="backfill-construct",
            )
            settled = {
                NightStage.PUBLISHED.value,
                NightStage.BLOCKED.value,
                NightStage.RECONCILIATION_REQUIRED.value,
            }
            try:
                for day in needs_acquisition:
                    acquisitions[day] = acquisition_pool.submit(self.acquire, day)
                chain_broken = False
                publication_broken = self.range_authorization is None
                constructions: Dict[str, Future] = {}

                def schedule_construction(index: int) -> None:
                    if index >= len(dates) or dates[index] in constructions:
                        return
                    day = dates[index]
                    predecessors = dates[:index]

                    def task() -> Dict[str, Any]:
                        if day in acquisitions:
                            acquired = acquisitions[day].result()
                            if not acquired["ok"]:
                                return acquired
                        if self.night_state(day)["stage"] in settled:
                            return {"ok": True, "skipped": True}
                        return self.construct(day, predecessors)

                    constructions[day] = construction_pool.submit(task)

                for index, day in enumerate(dates):
                    state = self.night_state(day)
                    if state["stage"] == NightStage.PUBLISHED.value:
                        if not publication_broken:
                            try:
                                self._emit_terminal_evidence(day)
                            except BaseException as error:
                                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                                    raise
                                self._fail(self.workspace(day), NightStage.PUBLISHED.value, error)
                        continue
                    if state["stage"] == NightStage.RECONCILIATION_REQUIRED.value:
                        # Unresolved authority precedes everything: resume it or stop.
                        if publication_broken:
                            chain_broken = True
                            continue
                        resumed = self.resume_publication(day)
                        self._write_range(start, end, time.monotonic() - wall_started)
                        if not resumed["ok"]:
                            chain_broken = publication_broken = True
                        continue
                    if state["stage"] == NightStage.BLOCKED.value:
                        chain_broken = publication_broken = True
                        continue
                    if chain_broken:
                        continue  # acquisition continues in the pool; construction waits
                    schedule_construction(index)
                    built = constructions[day].result()
                    if not built["ok"]:
                        chain_broken = publication_broken = True
                        continue
                    # Overlap the next construction with this night's publication.
                    schedule_construction(index + 1)
                    if publication_broken:
                        continue
                    published = self.publish(day, dates[:index])
                    if not published["ok"]:
                        publication_broken = True
                    self._write_range(start, end, time.monotonic() - wall_started)
            finally:
                acquisition_pool.shutdown(wait=True)
                construction_pool.shutdown(wait=True)
            return self._write_range(start, end, time.monotonic() - wall_started)


def _tree_bytes(root: Path) -> int:
    total = 0
    if root.is_dir():
        for directory, _, files in os.walk(root):
            for name in files:
                try:
                    total += os.lstat(os.path.join(directory, name)).st_size
                except FileNotFoundError:
                    pass
    return total


__all__ = [
    "BackfillController",
    "BackfillError",
    "BackfillRefused",
    "BackfillSettings",
    "NightAdapter",
    "NightStage",
    "NightWorkspace",
    "PRIOR_FREE_ACQUISITION_ATTESTATIONS",
    "PublicationFailed",
    "RangePublicationAuthorization",
    "acquisition_identity",
    "derive_night_state",
    "failure_classification",
    "inspect_backfill",
    "require_prior_free_acquisition",
]
