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

A Control-authorized range may instead *adopt* a night's saved acquisition:
the sealed query and fetch evidence of an earlier run, re-proved read-only by
:func:`describe_saved_acquisition` and bound into the range authorization.
Adoption never contacts ANTARES and never writes the source; construction
reads the source and rebuilds the candidate against the current prior, and
publication is unchanged.
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
from dataclasses import dataclass, field
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
    _read_regular,
    _write_json_new,
)
from .query_checkpoint import (
    DEFAULT_CHECKPOINT_NAME,
    QueryResultCheckpointBindings,
    load_query_result_checkpoint,
    seal_query_result_checkpoint,
)
from . import storage
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
        "adapter_implementation_sha256": "54c0fe236989972ab8f3469f3f98b8378f1491d90b4b3700e9d28bd9bc26b7d5",
        "evidence": "V3-G6.4B: V3-G6.2 range adapter plus Control-bound saved-acquisition adoption",
    },
)


# Saved acquisitions produced by these reviewed prior-free provider
# implementations may be adopted into a Control-authorized range: 0.4.5
# (7211b5c, f22578a5) and 0.4.6 (b808f28, afe11a1b).  Adoption re-proves the
# sealed evidence itself; this set only bounds which acquisition code made it.
ADOPTABLE_SOURCE_PROVIDERS = frozenset({
    "f22578a51ca65a3cf41d7fc8690fc026ccc0a16aefb300fc9c488cfb99c04199",
    "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c",
})
ADOPTION_SCHEMA = "v3.saved-acquisition-adoption.v1"
_ADOPTION_FIELDS = frozenset({
    "schema_version", "date_utc", "source_root", "source_run_id", "source_release_sha",
    "source_configuration_sha256", "source_provider_implementation_sha256",
    "source_request_sha256", "source_journal_head", "query_integrity_sha256",
    "query_contract_sha256", "query_tile_trace_sha256", "query_locus_order_sha256",
    "loci", "fetch_identity_sha256", "fetch_completion_sha256", "fetch_segments", "alert_rows",
})


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


def configuration_sha256(release_sha: str, adapter: Any, segment_size: int) -> str:
    """Acquisition configuration identity of one release (bound by every checkpoint)."""
    return _sha256(_canonical({
        "release_sha": release_sha,
        "provider": adapter.provider_name,
        "scenario": adapter.scenario,
        "execution_policy": dict(adapter.execution_policy()),
        "segment_size": segment_size,
    }))


def query_checkpoint_bindings(
    run_id: str, release_sha: str, configuration: str, adapter: Any, request: Any,
) -> QueryResultCheckpointBindings:
    return QueryResultCheckpointBindings(
        run_id=run_id,
        release_sha=release_sha,
        configuration_hash=configuration,
        target_date_utc=request.date_utc,
        provider_name=adapter.provider_name,
        provider_scenario=adapter.scenario,
        query_policy={
            "scientific_contract": dict(adapter.scientific_contract(request)),
            "execution_policy": dict(adapter.execution_policy()),
        },
    )


def fetch_checkpoint_binding(
    run_id: str, release_sha: str, configuration: str, adapter: Any, request: Any,
    loaded: Any, segment_size: int,
) -> FetchCheckpointBinding:
    details = loaded.query_result.evidence.details
    return FetchCheckpointBinding(
        run_id=run_id,
        release_sha=release_sha,
        configuration_sha256=configuration,
        target_date_utc=request.date_utc,
        mjd_min=request.mjd_min,
        mjd_max=request.mjd_max,
        provider_name=adapter.provider_name,
        provider_scenario=adapter.scenario,
        provider_policy_sha256=_sha256(_canonical(dict(adapter.execution_policy()))),
        query_contract_sha256=str(details.get("query_contract_sha256", "")),
        query_identity_sha256=loaded.integrity_sha256,
        query_locus_order_sha256=str(details.get("locus_order_sha256", "")),
        expected_objects=len(BackfillController._ordered_ids(loaded)),
        segment_size=segment_size,
    )


def _request_document(request: Any) -> Dict[str, Any]:
    return {
        "date_utc": request.date_utc,
        "mjd_min": request.mjd_min,
        "mjd_max": request.mjd_max,
        "ingested_at_utc": request.ingested_at_utc,
        "query_tag": request.query_tag,
        "target_loci": request.target_loci,
        "range_label": request.range_label,
    }


@dataclass(frozen=True)
class SavedAcquisition:
    """Re-proved, read-only materials of one adopted acquisition."""

    entry: Mapping[str, Any]
    request: Any
    loaded: Any
    binding: FetchCheckpointBinding
    checkpoint: SegmentedFetchCheckpoint
    completion: FetchCheckpointCompletion
    source_profile: Any = None
    selection_descriptor: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class SourceProfile:
    """Trusted verifier configuration, never constructed from source claims."""
    name: str
    provider_sha256s: frozenset[str]
    proof_profile: Any = None

    def __post_init__(self):
        from .live_antares import P2ProofProfile
        if (not isinstance(self.name, str) or not self.name
                or type(self.provider_sha256s) is not frozenset or not self.provider_sha256s
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in self.provider_sha256s)
                or (self.proof_profile is not None and type(self.proof_profile) is not P2ProofProfile)):
            raise BackfillRefused("Source profile requires exact finite provider identities and a trusted proof profile.")

    def execution_policy(self):
        from .live_antares import LiveAntaresProvider, default_query_policy
        provider = object.__new__(LiveAntaresProvider)
        provider.max_query_attempts, provider.retry_delay_seconds = default_query_policy(self.proof_profile)
        provider.max_fetch_attempts, provider.max_fetch_workers = 3, 4
        provider.proof_profile = self.proof_profile
        return provider.execution_policy()

    def scientific_contract(self, request):
        from .live_antares import scientific_contract_for_profile
        return scientific_contract_for_profile(request, self.proof_profile)


QUALIFIED_SOURCE_PROFILES = (SourceProfile("historical-P1", ADOPTABLE_SOURCE_PROVIDERS),)
SELECTION_DESCRIPTOR_SCHEMA = "v3.qualified-scientific-selection.v1"


def selection_descriptor_for_request(request):
    """Trusted intended interpretation; source callers invoke after full proof."""
    from .live_antares import night_mjd_interval
    from ..query import lsst_identifier_filter
    if ((request.mjd_min, request.mjd_max) != night_mjd_interval(request.date_utc)
            or request.query_tag is not None or request.lsst_only is not True
            or request.target_loci is not None or request.prior_locus_ids):
        raise BackfillRefused("Selection is not the exact prior-free exhaustive UTC night.")
    return {"schema_version": SELECTION_DESCRIPTOR_SCHEMA, "date_utc": request.date_utc,
        "time": {"field": "properties.newest_alert_observation_time", "mjd_min": request.mjd_min,
                 "mjd_max": request.mjd_max, "lower": "inclusive", "upper": "exclusive", "timezone": "UTC"},
        "spatial": {"ra_field": "ra", "dec_field": "dec", "units": "degrees",
            "ra_min": 0.0, "ra_max": 360.0, "ra_lower": "inclusive", "ra_upper": "exclusive",
            "dec_min": -90.0, "dec_max": 90.0, "dec_lower": "inclusive", "dec_upper": "inclusive_at_90_only"},
        "lsst_filter": lsst_identifier_filter(), "query_tag": None, "lsst_only": True,
        "target_loci": None, "prior_free": True,
        "normalization": "locus_to_record-properties-overlay;string-strip-nonblank-locus-id;tile-membership",
        "deduplication": {"key": "locus_id", "keep": "last", "scope": "accepted_tiles"},
        "input_order": "lower-child-first;within-leaf-qualified-service-order;keep-last;reset-index",
        "equivalence": "selection-semantics-only-not-service-snapshot"}


def selection_descriptor_identity(descriptor):
    return {"selection_descriptor_schema": descriptor["schema_version"],
            "selection_descriptor_sha256": _sha256(_canonical(dict(descriptor)))}


class _SourceProfileAdapter:
    def __init__(self, consumer, profile):
        self.consumer, self.profile = consumer, profile
        self.provider_name, self.scenario = consumer.provider_name, consumer.scenario

    def acquisition_request(self, night):
        return self.consumer.acquisition_request(night)

    def execution_policy(self):
        return self.profile.execution_policy()

    def scientific_contract(self, request):
        return self.profile.scientific_contract(request)

    def replay_query_journal(self, request, events):
        from .production_range import LiveRangeAdapter
        adapter = LiveRangeAdapter(self.consumer.work_root, self.consumer.release_sha, None,
                                   proof_profile=self.profile.proof_profile)
        return adapter.replay_query_journal(request, events)


def _identify_source_profile(root, manifest, request):
    from .query_checkpoint import CHECKPOINT_SCHEMA_VERSION, _parse_canonical_object
    from .query_progress import DIRECTORY, SCHEMA
    document = _parse_canonical_object(_read_regular(root / "checkpoints" / DIRECTORY / "identity.json"),
                                       "Saved source profile identity")
    identity = document.get("identity")
    if (document.get("schema_version") != SCHEMA or document.get("authoritative") is not False
            or not isinstance(identity, dict) or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION):
        raise BackfillRefused("Saved source profile schema is unsupported.")
    claimed = manifest.get("bindings", {}).get("query_policy")
    matches = [profile for profile in QUALIFIED_SOURCE_PROFILES
               if identity.get("provider_implementation_sha256") in profile.provider_sha256s
               and _sha256(_canonical(claimed)) == _sha256(_canonical(
                   {"scientific_contract": profile.scientific_contract(request),
                    "execution_policy": profile.execution_policy()}))]
    if len(matches) != 1:
        raise BackfillRefused("Saved source profile is unsupported or ambiguous; no verifier fallback.")
    return matches[0], identity["provider_implementation_sha256"]


def _adoptable_source_root(root: Path, date_utc: str) -> bool:
    """Only a canary run root or another range's night root may be adopted."""
    work_parent = Path(storage.RANGE_WORK_PARENT)
    return root.parent == Path(storage.ARNOR_CANARY_ROOT) or (
        root.name == f"night-{date_utc}"
        and root.parent.name == "nights"
        and root.parent.parent.parent == work_parent
    )


def validate_adoption_entry(date_utc: str, entry: Any) -> Dict[str, Any]:
    """Shape-check one authorized adoption; evidence is re-proved at use."""
    def hex64(*names: str) -> bool:
        return all(_is_hex64(entry.get(name)) for name in names)

    def count(*names: str) -> bool:
        return all(type(entry.get(name)) is int and entry[name] >= 0 for name in names)

    head = entry.get("source_journal_head") if isinstance(entry, Mapping) else None
    root = entry.get("source_root") if isinstance(entry, Mapping) else None
    if (
        not isinstance(entry, Mapping)
        or set(entry) != _ADOPTION_FIELDS
        or entry["schema_version"] != ADOPTION_SCHEMA
        or entry["date_utc"] != date_utc
        or not isinstance(root, str)
        or not Path(root).is_absolute()
        or ".." in Path(root).parts
        or entry["source_run_id"] != Path(root).name
        or not _is_sha40(entry["source_release_sha"])
        or entry["source_provider_implementation_sha256"] not in frozenset(
            value for profile in QUALIFIED_SOURCE_PROFILES for value in profile.provider_sha256s)
        or not hex64("source_configuration_sha256", "source_request_sha256", "query_integrity_sha256",
                     "query_contract_sha256", "query_tile_trace_sha256", "query_locus_order_sha256",
                     "fetch_identity_sha256", "fetch_completion_sha256")
        or not count("loci", "fetch_segments", "alert_rows")
        or not isinstance(head, Mapping)
        or set(head) != {"count", "last_sha256", "identity_sha256"}
        or type(head["count"]) is not int
        or not _is_hex64(head["last_sha256"])
        or not _is_hex64(head["identity_sha256"])
    ):
        raise BackfillRefused(f"Saved-acquisition adoption for {date_utc} is malformed.")
    return json.loads(_canonical(dict(entry)))


def exact_tile_partition(tiles: Sequence[Mapping[str, float]], mjd_min: float, mjd_max: float) -> bool:
    """Exact rational proof that half-open tiles partition the night's full sky."""
    from fractions import Fraction
    import bisect
    from .live_antares import _TILE_KEYS, _make_initial_tiles

    def volume(tile):
        edges = [Fraction(tile[key]) for key in _TILE_KEYS]
        return (edges[1] - edges[0]) * (edges[3] - edges[2]) * (edges[5] - edges[4])

    def disjoint(a, b):
        return any(a[high] <= b[low] or b[high] <= a[low]
                   for low, high in (("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max")))

    initial = _make_initial_tiles(mjd_min, mjd_max)
    domain = {"mjd_min": mjd_min, "mjd_max": mjd_max, "ra_min": 0.0, "ra_max": 360.0,
              "dec_min": -90.0, "dec_max": 90.0}
    if not tiles or sum(map(volume, initial), Fraction(0)) != volume(domain):
        return False
    axes = [sorted({tile[key] for tile in initial}) for key in ("mjd_min", "ra_min", "dec_min")]
    owners = {(tile["mjd_min"], tile["ra_min"], tile["dec_min"]): index for index, tile in enumerate(initial)}
    groups: Dict[int, List[Mapping[str, float]]] = {index: [] for index in range(len(initial))}
    for tile in tiles:
        corner = tuple(axis[max(0, bisect.bisect_right(axis, tile[key]) - 1)]
                       for axis, key in zip(axes, ("mjd_min", "ra_min", "dec_min")))
        owner = owners.get(corner)
        outer = initial[owner] if owner is not None else None
        if outer is None or not all(outer[low] <= tile[low] and tile[high] <= outer[high] for low, high in (
                ("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max"))):
            return False
        groups[owner].append(tile)
    return all(
        sum(map(volume, members), Fraction(0)) == volume(initial[index])
        and all(disjoint(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members)))
        for index, members in groups.items()
    )


def verify_saved_query_journal(
    root: Path, request: Any, bindings: QueryResultCheckpointBindings, loaded: Any, adapter: Any,
    *, expected_provider_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Strictly re-prove a saved query-progress journal, read-only.

    Never opens it through QueryProgress (which locks and rewrites HEAD).
    Proves the exact identity, the complete hash chain and ordering, HEAD,
    and that the provider's own replay of every committed decision yields
    the sealed query result and an exact 3-D partition. Returns (HEAD,
    source provider implementation).
    """
    from .live_antares import _TILE_KEYS, _make_initial_tiles
    from .query_checkpoint import (
        _binding_payload, _canonical_json_bytes, _parse_canonical_object, _request_binding,
    )
    from .query_progress import DIRECTORY, SCHEMA

    def normalized(value):
        return json.loads(_canonical_json_bytes(value))

    journal = root / "checkpoints" / DIRECTORY
    if journal.is_symlink() or not journal.is_dir() or journal.resolve(strict=True) != journal:
        raise BackfillRefused("Saved query journal directory is unsafe.")
    names = sorted(os.listdir(journal))
    event_names = [name for name in names if name.startswith("event-")]
    if set(names) - set(event_names) - {"identity.json", "HEAD.json", "LOCK"}:
        raise BackfillRefused("Saved query journal has unexpected or residual entries.")
    raw_identity = _read_regular(journal / "identity.json")
    document = _parse_canonical_object(raw_identity, "Saved query journal identity")
    identity = document.get("identity")
    if (
        set(document) != {"schema_version", "authoritative", "identity"}
        or document["schema_version"] != SCHEMA
        or document["authoritative"] is not False
        or not isinstance(identity, dict)
        or set(identity) != {"bindings", "request", "provider_implementation_sha256", "client", "initial_tiles"}
        or identity["bindings"] != normalized(_binding_payload(bindings))
        or identity["request"] != normalized(_request_binding(request))
        or identity["provider_implementation_sha256"] not in (
            adapter.profile.provider_sha256s if isinstance(adapter, _SourceProfileAdapter) else ADOPTABLE_SOURCE_PROVIDERS)
        or (expected_provider_sha256 is not None
            and identity["provider_implementation_sha256"] != expected_provider_sha256)
        or identity["initial_tiles"] != normalized(_make_initial_tiles(request.mjd_min, request.mjd_max))
    ):
        raise BackfillRefused("Saved query journal identity is not this acquisition's.")
    identity_sha = _sha256(raw_identity)
    previous, events = "0" * 64, []
    for index, name in enumerate(event_names):
        raw = _read_regular(journal / name)
        envelope = _parse_canonical_object(raw, "Saved query journal event")
        payload = envelope.get("payload")
        if (
            name != f"event-{index:08d}.json"
            or set(envelope) != {"payload", "sha256"}
            or not isinstance(payload, dict)
            or set(payload) != {"sequence", "previous_sha256", "identity_sha256", "event"}
            or payload["sequence"] != index
            or payload["previous_sha256"] != previous
            or payload["identity_sha256"] != identity_sha
            or envelope["sha256"] != _sha256(_canonical_json_bytes(payload))
        ):
            raise BackfillRefused("Saved query journal hash chain is broken.")
        previous = _sha256(raw)
        events.append(payload["event"])
    head = _parse_canonical_object(_read_regular(journal / "HEAD.json"), "Saved query journal head")
    if head != {"count": len(events), "last_sha256": previous, "identity_sha256": identity_sha}:
        raise BackfillRefused("Saved query journal HEAD does not commit to its event chain.")
    replayed = adapter.replay_query_journal(request, events)
    observed, sealed = replayed.evidence.details, loaded.query_result.evidence.details
    compared = (
        "tile_trace_sha256", "locus_order_sha256", "query_contract_sha256", "initial_tile_count",
        "accepted_tile_count", "split_count", "search_request_count", "returned_loci", "raw_returned_loci",
        "retry_count", "partial_rows_discarded", "coverage_complete", "terminal_pending_tile_count",
    )
    if isinstance(adapter, _SourceProfileAdapter) and adapter.profile.proof_profile is not None:
        compared += ("p2_events", "p2_budget", "deduplication", "retry_exception_types")
        if _sha256(_canonical({key: observed.get(key) for key in compared})) != _sha256(_canonical(
                {key: sealed.get(key) for key in compared})):
            raise BackfillRefused("P2 saved journal differs from its exact sealed operational proof.")
    replayed_loci, sealed_loci = replayed.loci, loaded.query_result.loci
    if (
        not replayed.clean
        or any(observed.get(key) != sealed.get(key) for key in compared)
        or (replayed_loci is None) != (sealed_loci is None)
        or (replayed_loci is not None and not (
            list(replayed_loci.columns) == list(sealed_loci.columns) and replayed_loci.equals(sealed_loci)))
    ):
        raise BackfillRefused("Saved query journal does not replay to its sealed query result.")
    accepted = [{key: row[key] for key in _TILE_KEYS} for row in observed["tile_trace"]
                if row["status"] == "accepted_exhausted"]
    if not exact_tile_partition(accepted, request.mjd_min, request.mjd_max):
        raise BackfillRefused("Saved query journal does not partition the night exactly.")
    return head, identity["provider_implementation_sha256"]


def describe_saved_acquisition(
    source_root: Path, date_utc: str, adapter: Any, segment_size: int,
    *, completion_sha256: Optional[str] = None,
) -> SavedAcquisition:
    """Re-prove one saved acquisition from its own sealed evidence, read-only.

    The source keeps its originating run id, release and configuration; the
    exact query policy, scientific request and segment size must equal this
    range's. The sealed query checkpoint, the complete query journal (see
    :func:`verify_saved_query_journal`) and every fetch segment are
    re-verified. With ``completion_sha256`` (an authorized adoption) the
    returned fetch checkpoint is pinned: anything that later reads it can
    only consume the authorized receipts and blobs. Nothing is created,
    repaired or contacted; any difference raises.
    """
    root = Path(os.path.abspath(os.fspath(source_root)))
    try:
        if (
            not _adoptable_source_root(root, date_utc)
            or root.is_symlink()
            or not root.is_dir()
            or root.resolve(strict=True) != root
        ):
            raise BackfillRefused("Saved acquisition root is not an adoptable source location.")
        request = adapter.acquisition_request(date_utc)
        if request.prior_locus_ids:
            raise BackfillRefused("Adopted acquisitions are prior-free.")
        raw_request = _read_regular(root / "request.json")
        if json.loads(raw_request) != _request_document(request):
            raise BackfillRefused("Saved acquisition request differs from this night's contract.")
        manifest = _read_json(root / "checkpoints" / DEFAULT_CHECKPOINT_NAME / "manifest.json")
        profile, source_provider_sha = _identify_source_profile(root, manifest, request)
        source_adapter = _SourceProfileAdapter(adapter, profile)
        claimed = manifest.get("bindings") if isinstance(manifest.get("bindings"), Mapping) else {}
        release, configuration = claimed.get("release_sha"), claimed.get("configuration_hash")
        if not _is_sha40(release) or configuration != configuration_sha256(release, source_adapter, segment_size):
            raise BackfillRefused("Saved acquisition configuration differs from this range's policy.")
        bindings = query_checkpoint_bindings(root.name, release, configuration, source_adapter, request)
        loaded = load_query_result_checkpoint(root, request, bindings)
        head, provider_sha = verify_saved_query_journal(root, request, bindings, loaded, source_adapter,
                                                      expected_provider_sha256=source_provider_sha)
        details = loaded.query_result.evidence.details
        if not (
            loaded.query_result.clean
            and details.get("coverage_complete") is True
            and details.get("terminal_pending_tile_count") == 0
            and details.get("query_contract_sha256")
            == _sha256(_canonical(dict(source_adapter.scientific_contract(request))))
        ):
            raise BackfillRefused("Saved acquisition query did not prove complete coverage.")
        ids = BackfillController._ordered_ids(loaded)
        if profile.proof_profile is not None:
            from .science import validate_p2_query_result
            validate_p2_query_result(request, loaded.query_result, profile.proof_profile)
        binding = fetch_checkpoint_binding(root.name, release, configuration, source_adapter, request,
                                           loaded, segment_size)
        checkpoint = SegmentedFetchCheckpoint.open_read_only(root, binding, completion_sha256=completion_sha256)
        completion = checkpoint.inspect_complete(ids)
    except BackfillRefused:
        raise
    except Exception as error:
        raise BackfillRefused(
            f"Saved acquisition for {date_utc} failed verification ({type(error).__name__})."
        ) from error
    entry = validate_adoption_entry(date_utc, {
        "schema_version": ADOPTION_SCHEMA,
        "date_utc": date_utc,
        "source_root": str(root),
        "source_run_id": root.name,
        "source_release_sha": release,
        "source_configuration_sha256": configuration,
        "source_provider_implementation_sha256": provider_sha,
        "source_request_sha256": _sha256(raw_request),
        "source_journal_head": head,
        "query_integrity_sha256": loaded.integrity_sha256,
        "query_contract_sha256": details["query_contract_sha256"],
        "query_tile_trace_sha256": details["tile_trace_sha256"],
        "query_locus_order_sha256": details["locus_order_sha256"],
        "loci": len(ids),
        "fetch_identity_sha256": binding.identity_sha256,
        "fetch_completion_sha256": completion.completion_sha256,
        "fetch_segments": completion.segment_count,
        "alert_rows": completion.alert_rows,
    })
    # Compatibility metadata follows all query/journal/partition/fetch proof.
    descriptor = selection_descriptor_for_request(request)
    if descriptor != selection_descriptor_for_request(adapter.acquisition_request(date_utc)):
        raise BackfillRefused("Verified source selection differs from current intended selection.")
    return SavedAcquisition(entry, request, loaded, binding, checkpoint, completion, profile, descriptor)


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
    # Night -> exact saved acquisition adopted instead of a live query.
    adopted_acquisitions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dates = _dates(self.start_date_utc, self.end_date_utc)
        adopted = self.adopted_acquisitions
        if not isinstance(adopted, Mapping) or not set(adopted) <= set(dates):
            raise BackfillRefused("Adopted acquisitions must be nights of this range.")
        normalized = {day: validate_adoption_entry(day, adopted[day]) for day in sorted(adopted)}
        roots = [entry["source_root"] for entry in normalized.values()]
        if len(set(roots)) != len(roots):
            raise BackfillRefused("Each adopted night needs its own saved acquisition.")
        object.__setattr__(self, "adopted_acquisitions", normalized)
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
        payload = {name: getattr(self, name) for name in _RANGE_FIELDS}
        # Ranges without adoption keep their accepted digest.
        if self.adopted_acquisitions:
            payload["adopted_acquisitions"] = self.adopted_acquisitions
        return json.loads(_canonical(payload))

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
        self.adoption = self.root / "adoption.json"

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

    def adopted(self) -> bool:
        """Acquisition was adopted from authorized saved evidence (see ``adoption.json``)."""
        return self.adoption.is_file()

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
    if workspace.adopted():
        evidence["adopted"] = True
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
    elif evidence["fetch_complete"] or evidence.get("adopted"):
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
        self._adoption_proofs: Dict[str, Mapping[str, Any]] = {}

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
        return configuration_sha256(self.release_sha, self.adapter, self.settings.segment_size)

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
            **self._adoption_binding(start, end),
        }

    def _adoption(self, date_utc: str) -> Optional[Mapping[str, Any]]:
        """The Control-authorized saved acquisition for this night, if any."""
        if self.range_authorization is None:
            return None
        return self.range_authorization.adopted_acquisitions.get(date_utc)

    def _saved_acquisition(self, date_utc: str) -> SavedAcquisition:
        """Re-prove the authorized source now, pinned to its authorized fetch completion."""
        authorized = self._adoption(date_utc)
        saved = describe_saved_acquisition(
            Path(authorized["source_root"]), date_utc, self.adapter, self.settings.segment_size,
            completion_sha256=authorized["fetch_completion_sha256"],
        )
        if saved.entry != authorized:
            raise BackfillRefused(f"Saved acquisition for {date_utc} differs from its authorized adoption.")
        return saved

    def _adoption_proven_locally(self, date_utc: str, authorized: Mapping[str, Any]) -> bool:
        """A validated candidate whose record binds exactly the consumed authorized evidence."""
        workspace = self.workspace(date_utc)
        try:
            if _read_json(workspace.adoption) != authorized:
                return False
            provenance = load_backfill_candidate(workspace.root).provenance
        except (OSError, ValueError, PublicationRefused):
            return False
        return (
            provenance.get("acquisition_source") == authorized
            and provenance.get("range_authorization_sha256") == self.range_authorization.digest
            and provenance.get("query_identity") == authorized["query_integrity_sha256"]
            and provenance.get("fetch_identity") == authorized["fetch_identity_sha256"]
            and provenance.get("fetch_completion_sha256") == authorized["fetch_completion_sha256"]
        )

    def _adoption_binding(self, start: str, end: str) -> Dict[str, Any]:
        if self.range_authorization is None or not self.range_authorization.adopted_acquisitions:
            return {}
        days = set(_dates(start, end))
        observed = {}
        for day, authorized in self.range_authorization.adopted_acquisitions.items():
            if day not in days:
                raise BackfillRefused("Adopted acquisition is outside the requested range.")
            if self._adoption_proven_locally(day, authorized):
                # Already consumed and bound by the night's own candidate record;
                # recovery never depends on the historical source again.
                observed[day] = authorized
                continue
            # Evidence re-proved once per controller run; construction re-proves again.
            if self._adoption_proofs.get(day) != authorized:
                self._adoption_proofs[day] = describe_saved_acquisition(
                    Path(authorized["source_root"]), day, self.adapter, self.settings.segment_size
                ).entry
            observed[day] = self._adoption_proofs[day]
        return {"adopted_acquisitions": observed}

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
        return query_checkpoint_bindings(
            workspace.run_id, self.release_sha, self._configuration_hash(), self.adapter, request
        )

    def _fetch_binding(self, workspace: NightWorkspace, request, loaded) -> FetchCheckpointBinding:
        return fetch_checkpoint_binding(
            workspace.run_id, self.release_sha, self._configuration_hash(), self.adapter, request,
            loaded, self.settings.segment_size,
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
        if self._adoption(date_utc) is not None:
            return self._adopt(workspace)
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

    def _adopt(self, workspace: NightWorkspace) -> Dict[str, Any]:
        """Record an authorized saved acquisition; no live read, no source write."""
        try:
            self._request(workspace)
            authorized = self._adoption(workspace.date_utc)
            entry = (self._adoption_proofs.get(workspace.date_utc)
                     if self._adoption_proofs.get(workspace.date_utc) == authorized
                     else self._saved_acquisition(workspace.date_utc).entry)
            if workspace.adoption.exists():
                if _read_json(workspace.adoption) != entry:
                    raise BackfillRefused("Recorded adoption differs from the authorized saved acquisition.")
            else:
                _write_json_new(workspace.adoption, entry)
                workspace.append_event(
                    "acquisition_adopted", stage=NightStage.FETCH_COMPLETE.value,
                    source_run_id=entry["source_run_id"], source_release_sha=entry["source_release_sha"],
                    loci=entry["loci"], alert_rows=entry["alert_rows"], segments_total=entry["fetch_segments"],
                )
            return {"ok": True, "adopted": True}
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return {"ok": False, "failure": self._fail(workspace, "ADOPTING", error)}

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
            adopted = self._adoption(date_utc)
            if adopted is None:
                bindings = self._query_bindings(workspace, request)
                loaded = load_query_result_checkpoint(workspace.root, request, bindings)
                ids = self._ordered_ids(loaded)
                binding = self._fetch_binding(workspace, request, loaded)
                checkpoint = SegmentedFetchCheckpoint.open_read_only(workspace.root, binding)
                completion = checkpoint.inspect_complete(ids)
            else:
                # Read the immutable source in place; never copy or adopt checkpoints.
                if not workspace.adopted() or _read_json(workspace.adoption) != adopted:
                    raise BackfillRefused("Adopted night lacks its exact recorded adoption.")
                saved = self._saved_acquisition(date_utc)
                loaded, binding = saved.loaded, saved.binding
                checkpoint, completion = saved.checkpoint, saved.completion
                ids = self._ordered_ids(loaded)
            prior = self.prior_locus_ids(date_utc, predecessors)
            construction_request = dataclasses.replace(request, prior_locus_ids=prior)
            query_result = dataclasses.replace(loaded.query_result, request=construction_request)
            self._hook("before_construct", date_utc=date_utc)
            checkpoint_constructor = getattr(self.adapter, "construct_checkpoint", None)
            constructor_options = ({"source_profile": saved.source_profile} if adopted is not None else {})
            result = (checkpoint_constructor(construction_request, query_result, checkpoint, **constructor_options)
                      if checkpoint_constructor is not None else
                      self.adapter.construct(construction_request, query_result,
                                             checkpoint.reconstruct_alerts(ids), completion))
            result.require_publishable()
            consumed_source = None
            if adopted is not None:
                # Provenance comes from what was consumed, never from the earlier proof.
                consumed = result.fetch_evidence.details["checkpoint"]
                consumed_source = dict(
                    adopted,
                    query_integrity_sha256=loaded.integrity_sha256,
                    loci=len(ids),
                    fetch_identity_sha256=consumed["identity_sha256"],
                    fetch_completion_sha256=consumed["completion_sha256"],
                    fetch_segments=consumed["segment_count"],
                    alert_rows=result.fetch_evidence.details["alert_rows"],
                )
                if consumed_source != adopted:
                    raise BackfillRefused("Construction consumed evidence other than its authorized adoption.")
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
                    "fetch_completion_sha256": (consumed_source["fetch_completion_sha256"]
                                                if consumed_source is not None else completion.completion_sha256),
                    "segments": completion.segment_count,
                    "prior_locus_count": len(prior),
                    "prior_locus_identity_sha256": _identifier_hash(prior),
                },
            }
            if consumed_source is not None:
                record["provenance"]["acquisition_source"] = consumed_source
                descriptor = saved.selection_descriptor
            elif self.adapter.provider_name == "live-antares":
                descriptor = selection_descriptor_for_request(self.adapter.acquisition_request(date_utc))
            else:
                descriptor = None
            if descriptor is not None:
                descriptor_identity = selection_descriptor_identity(descriptor)
                record["provenance"].update(descriptor_identity)
                record["provenance"]["binding_sha256"] = _sha256(_canonical({
                    "historical_binding_sha256": record["provenance"]["binding_sha256"],
                    **descriptor_identity}))
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
                and not states[day]["evidence"].get("adopted")
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
