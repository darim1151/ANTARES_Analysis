"""Fail-closed live ANTARES provider for Phase 6 commissioning.

The public ANTARES search API is anonymous.  Streaming credentials are not
used by this provider.  Live access still requires an explicitly issued
``LIVE_ANTARES_READ`` capability; network reachability or environment
variables alone never create that authority.

The pinned ``antares-client`` search iterator follows JSON:API pagination and
terminates normally only after a successful response has no ``links.next``.
This provider consumes that iterator to normal exhaustion.  Any exception,
malformed locus, duplicate locus, or abandoned retry is classified as
incomplete/failed and cannot be staged as science.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import pandas as pd

from .. import query
from ..history import prepare_alerts, prepare_loci, validation_summary
from .science import (
    FetchProviderError,
    FetchStageEvidence,
    NightQueryResult,
    NightScienceRequest,
    NightScienceResult,
    ProviderContractError,
    ProviderIssue,
    ProviderOutcome,
    ProviderStage,
    QueryStageEvidence,
    ZERO_ROW_PROOF,
)
from .storage import ARNOR_CANARY_ROOT, StorageContractError
from .transaction import QueryFetchEvidence


LIVE_ANTARES_READ = "LIVE_ANTARES_READ"
PINNED_CLIENT_VERSION = "1.14.0"
OFFICIAL_API_BASE_URL = "https://api.antares.noirlab.edu/v1/"
PHASE6_TARGET_DATE_UTC = "2026-06-27"
PHASE6_MJD_MIN = 61218.0
PHASE6_MJD_MAX = 61219.0
MAX_QUERY_ATTEMPTS = 2
MAX_FETCH_ATTEMPTS = 3
MAX_FETCH_WORKERS = 4
CLIENT_TIMEOUT_SECONDS = 60
PROBE_LIMIT = 50
PROBE_THRESHOLD = 50
TIME_BIN_MINUTES = 30
RA_BINS = 24
DEC_BINS = 6
MIN_TIME_SECONDS = 30.0
MIN_RA_DEGREES = 0.05
MIN_DEC_DEGREES = 0.05
EXTRACTION_METHOD = "probe_first_time_ra_dec"
CACHE_VERSION = "probe50_time_ra_dec_v1"
SECONDS_PER_DAY = 86400.0
_LIVE_READ_TOKEN = object()
_TILE_KEYS = (
    "mjd_min",
    "mjd_max",
    "ra_min",
    "ra_max",
    "dec_min",
    "dec_max",
)


class LiveCompletion(str, Enum):
    """Stable Phase 6 completion classification."""

    COMPLETE_ZERO = "COMPLETE_ZERO"
    COMPLETE_NONZERO = "COMPLETE_NONZERO"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class LiveCapabilityError(StorageContractError):
    """Explicit live-read authority could not be issued."""


def _safe_run_id(value: object) -> str:
    run_id = str(value).strip()
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or len(Path(run_id).parts) != 1
    ):
        raise LiveCapabilityError("Live commissioning requires one safe run id.")
    return run_id


def _canonical_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise LiveCapabilityError(
            "Live commissioning target must use canonical YYYY-MM-DD form."
        ) from exc
    if parsed.isoformat() != value:
        raise LiveCapabilityError(
            "Live commissioning target must use canonical YYYY-MM-DD form."
        )
    return value


def night_mjd_interval(value: str) -> Tuple[float, float]:
    """Exact UTC civil-midnight MJD bounds, independent of local timezone."""
    canonical = _canonical_date(value)
    first = float((date.fromisoformat(canonical) - date(1858, 11, 17)).days)
    return first, first + 1.0


def _real_directory(path: Path, label: str) -> Path:
    lexical = Path(path).expanduser()
    if lexical.is_symlink() or not lexical.is_dir():
        raise LiveCapabilityError(f"{label} must be an existing real directory.")
    return lexical.resolve(strict=True)


@dataclass(frozen=True)
class LiveAntaresReadCapability:
    """Sealed authority for one target and one exact commissioning run root.

    The capability intentionally has no production path and no write methods.
    """

    run_root: Path
    run_id: str
    target_date_utc: str
    release_sha: str
    environment: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _LIVE_READ_TOKEN:
            raise LiveCapabilityError(
                "Live ANTARES read capabilities must be issued by a sealed factory."
            )
        _safe_run_id(self.run_id)
        _canonical_date(self.target_date_utc)
        if (
            not isinstance(self.release_sha, str)
            or len(self.release_sha) != 40
            or any(character not in "0123456789abcdef" for character in self.release_sha)
        ):
            raise LiveCapabilityError("Live read capability requires a full release SHA.")
        if self.environment not in {"arnor-commissioning", "local-mock"}:
            raise LiveCapabilityError("Unknown live-read capability environment.")

    @classmethod
    def for_arnor_commissioning(
        cls,
        run_root: Path,
        *,
        run_id: str,
        target_date_utc: str,
        release_sha: str,
        authority: str,
        hostname: Optional[str] = None,
    ) -> "LiveAntaresReadCapability":
        if authority != LIVE_ANTARES_READ:
            raise LiveCapabilityError(
                f"Explicit authority {LIVE_ANTARES_READ!r} is required."
            )
        observed_host = hostname or socket.gethostname()
        if observed_host.strip().lower().split(".", 1)[0] != "arnor":
            raise LiveCapabilityError("Live commissioning authority is Arnor-only.")
        identity = _safe_run_id(run_id)
        _canonical_date(target_date_utc)
        expected = ARNOR_CANARY_ROOT / identity
        lexical = Path(os.path.abspath(os.fspath(Path(run_root).expanduser())))
        if lexical != expected or lexical.parent != ARNOR_CANARY_ROOT:
            raise LiveCapabilityError(
                f"Live commissioning requires the exact run root {expected}."
            )
        resolved = _real_directory(lexical, "Live commissioning run root")
        try:
            canonical_parent = ARNOR_CANARY_ROOT.resolve(strict=True)
        except OSError as exc:
            raise LiveCapabilityError("The Arnor canary parent is unavailable.") from exc
        if canonical_parent != ARNOR_CANARY_ROOT or resolved != expected:
            raise LiveCapabilityError("Live commissioning rejects path aliases.")
        return cls(
            resolved,
            identity,
            _canonical_date(target_date_utc),
            release_sha,
            "arnor-commissioning",
            _LIVE_READ_TOKEN,
        )

    @classmethod
    def for_local_mock(
        cls,
        run_root: Path,
        *,
        run_id: str,
        target_date_utc: str,
        release_sha: str,
        authority: str,
    ) -> "LiveAntaresReadCapability":
        """Issue test-only authority below the OS temporary directory."""
        if authority != LIVE_ANTARES_READ:
            raise LiveCapabilityError(
                f"Explicit authority {LIVE_ANTARES_READ!r} is required."
            )
        resolved = _real_directory(run_root, "Mock live-read run root")
        temporary = Path(tempfile.gettempdir()).resolve(strict=True)
        try:
            resolved.relative_to(temporary)
        except ValueError as exc:
            raise LiveCapabilityError(
                "Mock live-read authority is restricted to a temporary child."
            ) from exc
        identity = _safe_run_id(run_id)
        if resolved == temporary or resolved.name != identity:
            raise LiveCapabilityError("Mock run root must be named for its run id.")
        return cls(
            resolved,
            identity,
            _canonical_date(target_date_utc),
            release_sha,
            "local-mock",
            _LIVE_READ_TOKEN,
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def extraction_method_contract() -> Dict[str, Any]:
    """Return the accepted historical extractor identity."""
    return {
        "name": EXTRACTION_METHOD,
        "probe_limit": PROBE_LIMIT,
        "probe_threshold": PROBE_THRESHOLD,
        "time_bin_minutes": TIME_BIN_MINUTES,
        "ra_bins": RA_BINS,
        "dec_bins": DEC_BINS,
        "min_time_seconds": MIN_TIME_SECONDS,
        "min_ra_degrees": MIN_RA_DEGREES,
        "min_dec_degrees": MIN_DEC_DEGREES,
        "cache_version": CACHE_VERSION,
    }


def _scientific_query_contract(request: NightScienceRequest) -> Dict[str, Any]:
    """Bind the live request to the notebook-derived scientific semantics."""
    return {
        "target_date_utc": request.date_utc,
        "interval": {
            "mjd_min": float(request.mjd_min),
            "mjd_max": float(request.mjd_max),
            "lower_bound": "inclusive",
            "upper_bound": "exclusive",
            "timezone": "UTC",
        },
        "spatial_domain": {
            "ra_min": 0.0,
            "ra_max": 360.0,
            "ra_lower_bound": "inclusive",
            "ra_upper_bound": "exclusive",
            "dec_min": -90.0,
            "dec_max": 90.0,
            "dec_lower_bound": "inclusive",
            "dec_upper_bound": "inclusive_at_90_only",
        },
        "query_tag": None,
        "lsst_only": True,
        "lsst_filter": query.lsst_identifier_filter(),
        "sort_requested": None,
        "parallel_parent_shards": 1,
        "extraction_method": extraction_method_contract(),
        "deduplication": {
            "key": "locus_id",
            "keep": "last",
            "scope": "accepted_tiles",
        },
    }


def _build_tile_query(tile: Mapping[str, float]) -> Dict[str, Any]:
    """Reproduce the accepted notebook's half-open time/RA/Dec tile query."""
    dec_upper = "lte" if float(tile["dec_max"]) >= 90.0 else "lt"
    return {
        "query": {
            "bool": {
                "filter": [
                    {
                        "range": {
                            "properties.newest_alert_observation_time": {
                                "gte": float(tile["mjd_min"]),
                                "lt": float(tile["mjd_max"]),
                            }
                        }
                    },
                    {
                        "range": {
                            "ra": {
                                "gte": float(tile["ra_min"]),
                                "lt": float(tile["ra_max"]),
                            }
                        }
                    },
                    {
                        "range": {
                            "dec": {
                                "gte": float(tile["dec_min"]),
                                dec_upper: float(tile["dec_max"]),
                            }
                        }
                    },
                    query.lsst_identifier_filter(),
                ]
            }
        }
    }


def _make_initial_tiles(mjd_min: float, mjd_max: float) -> list[Dict[str, float]]:
    step = TIME_BIN_MINUTES / 1440.0
    time_count = int(math.ceil((float(mjd_max) - float(mjd_min)) / step))
    time_edges = [
        round(min(float(mjd_min) + index * step, float(mjd_max)), 12)
        for index in range(time_count + 1)
    ]
    time_edges[-1] = float(mjd_max)
    ra_edges = [360.0 * index / RA_BINS for index in range(RA_BINS + 1)]
    dec_edges = [-90.0 + 180.0 * index / DEC_BINS for index in range(DEC_BINS + 1)]
    tiles = []
    for time_start, time_end in zip(time_edges[:-1], time_edges[1:]):
        if time_end <= time_start:
            continue
        for ra_min, ra_max in zip(ra_edges[:-1], ra_edges[1:]):
            for dec_min, dec_max in zip(dec_edges[:-1], dec_edges[1:]):
                tiles.append(
                    {
                        "mjd_min": float(time_start),
                        "mjd_max": float(time_end),
                        "ra_min": float(ra_min),
                        "ra_max": float(ra_max),
                        "dec_min": float(dec_min),
                        "dec_max": float(dec_max),
                    }
                )
    return tiles


def _split_tile(tile: Mapping[str, float]) -> Tuple[Dict[str, float], ...]:
    ratios = {
        "time": (
            (float(tile["mjd_max"]) - float(tile["mjd_min"]))
            * SECONDS_PER_DAY
            / MIN_TIME_SECONDS
        ),
        "ra": (
            (float(tile["ra_max"]) - float(tile["ra_min"]))
            / MIN_RA_DEGREES
        ),
        "dec": (
            (float(tile["dec_max"]) - float(tile["dec_min"]))
            / MIN_DEC_DEGREES
        ),
    }
    dimension = max(ratios, key=ratios.get)
    if ratios[dimension] <= 1.0:
        return ()
    first = dict(tile)
    second = dict(tile)
    if dimension == "time":
        midpoint = (float(tile["mjd_min"]) + float(tile["mjd_max"])) / 2.0
        first["mjd_max"] = midpoint
        second["mjd_min"] = midpoint
    elif dimension == "ra":
        midpoint = (float(tile["ra_min"]) + float(tile["ra_max"])) / 2.0
        first["ra_max"] = midpoint
        second["ra_min"] = midpoint
    else:
        midpoint = (float(tile["dec_min"]) + float(tile["dec_max"])) / 2.0
        first["dec_max"] = midpoint
        second["dec_min"] = midpoint
    return first, second


def _canonical_tile(
    value: Mapping[str, Any],
    *,
    mjd_min: float,
    mjd_max: float,
) -> Dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_TILE_KEYS):
        raise ValueError("A query tile has an invalid field set.")
    try:
        tile = {key: float(value[key]) for key in _TILE_KEYS}
    except (TypeError, ValueError) as exc:
        raise ValueError("A query tile has a non-numeric boundary.") from exc
    if not all(math.isfinite(item) for item in tile.values()):
        raise ValueError("A query tile has a non-finite boundary.")
    if not (
        float(mjd_min) <= tile["mjd_min"] < tile["mjd_max"] <= float(mjd_max)
        and 0.0 <= tile["ra_min"] < tile["ra_max"] <= 360.0
        and -90.0 <= tile["dec_min"] < tile["dec_max"] <= 90.0
    ):
        raise ValueError("A query tile is outside the exact target domain.")
    return tile


def _record_matches_tile(record: Mapping[str, Any], tile: Mapping[str, float]) -> bool:
    try:
        observed_mjd = float(record["newest_alert_observation_time"])
        observed_ra = float(record["ra"])
        observed_dec = float(record["dec"])
    except (KeyError, TypeError, ValueError):
        return False
    dec_inside = (
        tile["dec_min"] <= observed_dec <= tile["dec_max"]
        if tile["dec_max"] >= 90.0
        else tile["dec_min"] <= observed_dec < tile["dec_max"]
    )
    return bool(
        tile["mjd_min"] <= observed_mjd < tile["mjd_max"]
        and tile["ra_min"] <= observed_ra < tile["ra_max"]
        and dec_inside
    )


def _identifier_hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _exception_type(error: BaseException) -> str:
    """Return non-secret exception identity; never serialize exception text."""
    return f"{type(error).__module__}.{type(error).__name__}"


def _retryable_query_error(error: BaseException) -> bool:
    """Whether one failed tile attempt may be repeated within its bounded budget.

    ``TypeError``/``ValueError`` mean a returned locus failed this provider's
    own validation, which repeating the request cannot repair.  The pinned
    client's ``requests.exceptions.JSONDecodeError`` is also a ``ValueError``,
    but it means an HTTP response body was not JSON (for example an HTML
    gateway error page that the client decodes before raising its own
    error): a service condition like any other transient failure.  A retry
    still restarts the whole tile and discards every partial row.
    """
    if not isinstance(error, (TypeError, ValueError)):
        return True
    try:
        from requests.exceptions import JSONDecodeError
    except ImportError:
        return False
    return isinstance(error, JSONDecodeError)


def _validated_base_url(value: str) -> str:
    parsed = urlsplit(str(value))
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(
            "Phase 6 requires a credential-free official ANTARES API URL."
        )
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = hostname + port
    path = parsed.path if parsed.path.endswith("/") else parsed.path + "/"
    normalized = urlunsplit((parsed.scheme, netloc, path, "", ""))
    if normalized != OFFICIAL_API_BASE_URL:
        raise RuntimeError(
            "Phase 6 requires the official ANTARES API base URL; override refused."
        )
    return normalized


def _combined_evidence(
    query_evidence: QueryStageEvidence,
    fetch_evidence: FetchStageEvidence,
) -> QueryFetchEvidence:
    zero_proof = None
    if (
        query_evidence.clean
        and fetch_evidence.clean
        and fetch_evidence.loci_rows == 0
        and fetch_evidence.alert_rows == 0
    ):
        zero_proof = ZERO_ROW_PROOF
    return QueryFetchEvidence(
        query_completed=query_evidence.completed,
        fetch_completed=fetch_evidence.completed,
        loci_rows=fetch_evidence.loci_rows,
        alert_rows=fetch_evidence.alert_rows,
        query_errors=tuple(issue.code for issue in query_evidence.errors),
        fetch_errors=tuple(issue.code for issue in fetch_evidence.errors),
        zero_row_proof=zero_proof,
    )


class LiveAntaresProvider:
    """Real ANTARES adapter implementing the Phase 5 provider abstraction."""

    provider_name = "live-antares"
    scenario = "commissioning-v1"

    def __init__(
        self,
        capability: LiveAntaresReadCapability,
        *,
        search_fn: Optional[Callable[[Dict[str, Any]], Iterable[Any]]] = None,
        get_by_id_fn: Optional[Callable[[str], Any]] = None,
        connectivity_fn: Optional[Callable[[], Any]] = None,
        initial_tiles_fn: Optional[
            Callable[[float, float], Iterable[Mapping[str, float]]]
        ] = None,
        max_query_attempts: int = 2,
        max_fetch_attempts: int = 3,
        max_fetch_workers: int = 4,
        retry_delay_seconds: float = 0.5,
        clock: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        proof_profile: Optional[P2ProofProfile] = None,
    ) -> None:
        if type(capability) is not LiveAntaresReadCapability:
            raise LiveCapabilityError("A sealed live-read capability is required.")
        for name, value in (
            ("max_query_attempts", max_query_attempts),
            ("max_fetch_attempts", max_fetch_attempts),
            ("max_fetch_workers", max_fetch_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if max_query_attempts > MAX_QUERY_ATTEMPTS:
            raise ValueError(f"max_query_attempts may not exceed {MAX_QUERY_ATTEMPTS}.")
        if max_fetch_attempts > MAX_FETCH_ATTEMPTS:
            raise ValueError(f"max_fetch_attempts may not exceed {MAX_FETCH_ATTEMPTS}.")
        if max_fetch_workers > MAX_FETCH_WORKERS:
            raise ValueError(f"max_fetch_workers may not exceed {MAX_FETCH_WORKERS}.")
        if retry_delay_seconds < 0 or retry_delay_seconds > 5:
            raise ValueError("retry_delay_seconds must be between 0 and 5 seconds.")
        if capability.environment != "local-mock" and any(
            value is not None
            for value in (search_fn, get_by_id_fn, connectivity_fn, initial_tiles_fn)
        ):
            raise LiveCapabilityError(
                "Callable injection is restricted to local mocked qualification."
            )
        supplied = tuple(
            value is not None for value in (search_fn, get_by_id_fn, connectivity_fn)
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "Mocked provider injection requires search, fetch, and connectivity callables."
            )
        if initial_tiles_fn is not None and not all(supplied):
            raise ValueError(
                "Mock initial-tile injection also requires all service callables."
            )
        self.capability = capability
        self._search_fn = search_fn
        self._get_by_id_fn = get_by_id_fn
        self._connectivity_fn = connectivity_fn
        self._initial_tiles_fn = initial_tiles_fn or _make_initial_tiles
        self._initial_tiles_overridden = initial_tiles_fn is not None
        self.max_query_attempts = max_query_attempts
        self.max_fetch_attempts = max_fetch_attempts
        self.max_fetch_workers = max_fetch_workers
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.clock = clock
        self.monotonic = monotonic
        self.sleeper = sleeper
        self._client_identity_cache: Optional[Mapping[str, Any]] = None
        if proof_profile is not None and type(proof_profile) is not P2ProofProfile:
            raise ValueError("P2 requires an explicit validated proof profile.")
        self.proof_profile = proof_profile

    def _load_client(self) -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        if getattr(self, "proof_profile", None) is not None:
            return _load_p2_client(self)
        if self._search_fn is not None:
            assert self._get_by_id_fn is not None
            assert self._connectivity_fn is not None
            return self._search_fn, self._get_by_id_fn, self._connectivity_fn
        try:
            from antares_client.config import config
            from antares_client.search import get_available_tags, get_by_id, search
        except ImportError as exc:
            raise RuntimeError("The pinned ANTARES client is unavailable.") from exc
        version = metadata.version("antares-client")
        if version != PINNED_CLIENT_VERSION:
            raise RuntimeError(
                f"Expected antares-client {PINNED_CLIENT_VERSION}; found {version}."
            )
        base_url = _validated_base_url(str(config.get("ANTARES_API_BASE_URL", "")))
        timeout = int(config.get("API_TIMEOUT", CLIENT_TIMEOUT_SECONDS))
        if timeout != CLIENT_TIMEOUT_SECONDS:
            raise RuntimeError(
                f"Phase 6 requires the pinned {CLIENT_TIMEOUT_SECONDS}-second API timeout."
            )
        self._client_identity_cache = {
            "distribution": "antares-client",
            "version": version,
            "api_base_url": base_url,
            "api_timeout_seconds": timeout,
            "authentication": "public-search-no-credentials",
            "pagination_contract": "jsonapi-links-next-until-null",
        }
        return search, get_by_id, get_available_tags

    def execution_policy(self) -> Mapping[str, Any]:
        """Return the bounded transport policy included in release provenance."""
        policy = {
            "max_query_attempts": self.max_query_attempts,
            "max_fetch_attempts_per_object": self.max_fetch_attempts,
            "max_fetch_workers": self.max_fetch_workers,
            "retry_delay_seconds": self.retry_delay_seconds,
            "api_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
            "probe_limit": PROBE_LIMIT,
            "probe_threshold": PROBE_THRESHOLD,
            "extraction_method": extraction_method_contract(),
            "tile_cache": False,
            "lightcurve_cache": False,
            "parallel_parent_shards": 1,
        }
        if getattr(self, "proof_profile", None) is not None:
            policy["extraction_method"] = self.proof_profile.extraction_method()
        return policy

    def scientific_contract(self, request: NightScienceRequest) -> Mapping[str, Any]:
        """Return the exact, side-effect-free scientific request contract."""
        self._validate_request(request)
        return scientific_contract_for_profile(request, getattr(self, "proof_profile", None))

    def client_identity(self) -> Mapping[str, Any]:
        self._load_client()
        if self._client_identity_cache is not None:
            return dict(self._client_identity_cache)
        return {
            "distribution": "mock-antares-client",
            "version": PINNED_CLIENT_VERSION,
            "api_base_url": OFFICIAL_API_BASE_URL,
            "api_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
            "authentication": "public-search-no-credentials",
            "pagination_contract": "mocked-jsonapi-links-next-until-null",
        }

    def check_connectivity(self) -> Mapping[str, Any]:
        """Perform the smallest supported anonymous API check (tag statistics)."""
        _search, _get_by_id, connectivity = self._load_client()
        started = self.clock()
        t0 = self.monotonic()
        try:
            tags = connectivity()
            if not isinstance(tags, (list, tuple, set)):
                raise TypeError("Connectivity response was not a tag collection.")
            normalized = sorted(str(item) for item in tags)
        except Exception as exc:
            raise RuntimeError(
                f"ANTARES connectivity check failed ({_exception_type(exc)})."
            ) from exc
        finished = self.clock()
        return {
            "passed": True,
            "started_at_utc": _iso(started),
            "completed_at_utc": _iso(finished),
            "runtime_seconds": round(max(0.0, self.monotonic() - t0), 6),
            "tag_count": len(normalized),
            "tag_identity_sha256": _identifier_hash(normalized),
            "authentication": "public-search-no-credentials",
            "credentials_consumed": False,
            "secret_material_recorded": False,
        }

    def _validate_request(self, request: NightScienceRequest) -> None:
        if not isinstance(request, NightScienceRequest):
            raise ProviderContractError(
                ProviderIssue(
                    "invalid_request_type",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.QUERY_FAILURE,
                    "Live query requires a NightScienceRequest.",
                )
            )
        if (
            request.date_utc != self.capability.target_date_utc
        ):
            raise ProviderContractError(
                ProviderIssue(
                    "live_capability_target_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.QUERY_FAILURE,
                    "The live-read capability belongs to a different UTC night.",
                )
            )
        if (
            (request.mjd_min, request.mjd_max) != night_mjd_interval(request.date_utc)
            or request.query_tag is not None
            or request.target_loci is not None
            or request.lsst_only is not True
        ):
            raise ProviderContractError(
                ProviderIssue(
                    "live_request_interval_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.QUERY_FAILURE,
                    "The request is not the exact untagged exhaustive Phase 6 LSST interval.",
                )
            )

    def query_resumable(self, request, bindings, *, event_hook=None):
        """Reprove committed tile decisions and acquire only the pending frontier."""
        from .query_progress import QueryProgress
        from .query_checkpoint import _binding_payload, _request_binding, QueryCheckpointError

        self._validate_request(request)
        if (bindings.run_id != self.capability.run_id
                or bindings.release_sha != self.capability.release_sha
                or bindings.provider_name != self.provider_name
                or bindings.provider_scenario != self.scenario
                or dict(bindings.query_policy) != {
                    "scientific_contract": self.scientific_contract(request),
                    "execution_policy": self.execution_policy()}):
            raise QueryCheckpointError("Query-progress caller binding differs from provider.")
        identity = {
            "bindings": _binding_payload(bindings), "request": _request_binding(request),
            "provider_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "client": self.client_identity(),
            "initial_tiles": list(self._initial_tiles_fn(request.mjd_min, request.mjd_max)),
        }
        with QueryProgress(self.capability.run_root, self.capability.run_id, identity) as progress:
            return self.query(request, _progress=progress, _event_hook=event_hook)

    def query(self, request: NightScienceRequest, *, _progress=None, _event_hook=None) -> NightQueryResult:
        """Run the accepted probe-first time/RA/Dec extractor to exhaustion.

        Every tile uses a half-open time interval.  A 50-row probe is
        provisional saturation: those rows are discarded and the tile is
        split along its largest normalized dimension.  Only tiles returning
        fewer than 50 rows *and* reaching normal iterator exhaustion are
        accepted.  Final accepted rows are deduplicated by ``locus_id`` with
        ``keep='last'``, exactly as in the accepted historical path.
        """
        if getattr(self, "proof_profile", None) is not None:
            return _run_p2_query(self, request, _progress, _event_hook)
        self._validate_request(request)
        search, _get_by_id, _connectivity = self._load_client()
        scientific_contract = _scientific_query_contract(request)
        contract_hash = _sha256_json(scientific_contract)
        started = self.clock()
        t0 = self.monotonic()

        initial_tiles = [
            _canonical_tile(
                value,
                mjd_min=request.mjd_min,
                mjd_max=request.mjd_max,
            )
            for value in self._initial_tiles_fn(request.mjd_min, request.mjd_max)
        ]
        if self._initial_tiles_overridden:
            full_domain = {
                "mjd_min": float(request.mjd_min),
                "mjd_max": float(request.mjd_max),
                "ra_min": 0.0,
                "ra_max": 360.0,
                "dec_min": -90.0,
                "dec_max": 90.0,
            }
            if initial_tiles != [full_domain]:
                raise ValueError(
                    "Mock initial-tile injection must be the one exact full domain."
                )
        elif initial_tiles != _make_initial_tiles(request.mjd_min, request.mjd_max):
            raise RuntimeError("The canonical initial tiling changed unexpectedly.")
        if not initial_tiles:
            raise RuntimeError("The exact query domain produced no initial tiles.")

        pending = deque(initial_tiles)
        accepted_records = []
        accepted_tiles = []
        trace = []
        split_count = 0
        search_request_count = 0
        retry_count = 0
        aggregate_partial_rows = 0
        retry_exception_types = set()
        resumed_attempt = 1

        def commit_decision(entry, records=(), retry_scheduled=False):
            if _progress is not None:
                from .query_progress import encode_records
                _progress.commit({"trace": entry, "records": encode_records(records),
                                  "retry_scheduled": retry_scheduled})
                if _event_hook is not None:
                    _event_hook("query_tile_committed", {"status": entry["status"], "tile": {k: entry[k] for k in _TILE_KEYS}})

        def boundary_for(tile):
            return {"tile": dict(tile), "query_sha256": _sha256_json(_build_tile_query(tile)),
                    "reason": "terminal_transient_retry_exhaustion"}

        if _progress is not None:
            from .query_progress import decode_records
            from .query_checkpoint import QueryCheckpointError
            # Attempt errors stay provisional until their tile resolves. A
            # terminal transient failure ends that invocation; only an explicit
            # boundary committed by a later invocation may renew the tile's
            # bounded budget, discarding the failed invocation's attempts.
            provisional = []
            awaiting_boundary = False
            for event in _progress.events:
                if not pending:
                    raise QueryCheckpointError("Query-progress traversal is inconsistent.")
                tile = pending[0]
                if set(event) == {"invocation_boundary"}:
                    if not awaiting_boundary or event["invocation_boundary"] != boundary_for(tile):
                        raise QueryCheckpointError("Query-progress invocation boundary is invalid.")
                    provisional.clear()
                    awaiting_boundary = False
                    resumed_attempt = 1
                    continue
                if awaiting_boundary or set(event) != {"trace", "records", "retry_scheduled"}:
                    raise QueryCheckpointError("Query-progress traversal is inconsistent.")
                entry = event["trace"]
                if (not isinstance(entry, dict) or {k: entry.get(k) for k in _TILE_KEYS} != tile
                        or entry.get("query_sha256") != _sha256_json(_build_tile_query(tile))
                        or type(entry.get("attempt")) is not int
                        or not 1 <= entry["attempt"] <= self.max_query_attempts
                        or entry["attempt"] != resumed_attempt):
                    raise QueryCheckpointError("Query-progress tile lineage/ref mismatch.")
                records = decode_records(event["records"])
                status = entry.get("status")
                if status in {"accepted_exhausted", "split_saturated"}:
                    for retried in provisional:
                        aggregate_partial_rows += retried["partial_rows_discarded"]
                        retry_exception_types.add(retried["exception_type"])
                        retry_count += 1
                        search_request_count += 1
                        trace.append(retried)
                    provisional.clear()
                if status == "accepted_exhausted":
                    if (entry.get("iterator_exhausted") is not True or len(records) >= PROBE_LIMIT
                            or entry.get("returned_loci") != len(records)
                            or event["retry_scheduled"] is not False
                            or any(not isinstance(r, dict) or not r.get("locus_id")
                                   or not _record_matches_tile(r, tile) for r in records)):
                        raise QueryCheckpointError("Query-progress terminal tile is incomplete.")
                    pending.popleft()
                    accepted_tiles.append(tile)
                    accepted_records.extend(records)
                elif status == "split_saturated":
                    children = _split_tile(tile)
                    if (not children or records or entry.get("returned_before_split") != PROBE_LIMIT
                            or entry.get("iterator_exhausted") is not False
                            or event["retry_scheduled"] is not False):
                        raise QueryCheckpointError("Query-progress split is invalid.")
                    pending.popleft()
                    pending.extendleft(reversed(children))
                    split_count += 1
                    aggregate_partial_rows += PROBE_LIMIT
                elif status == "attempt_error":
                    if (records or entry.get("iterator_exhausted") is not False
                            or type(entry.get("partial_rows_discarded")) is not int
                            or not 0 <= entry["partial_rows_discarded"] < PROBE_LIMIT
                            or not isinstance(entry.get("exception_type"), str)
                            or entry.get("retryable") is not True
                            or type(event["retry_scheduled"]) is not bool
                            or event["retry_scheduled"] != (entry["attempt"] < self.max_query_attempts)):
                        raise QueryCheckpointError("Query-progress retry evidence is invalid.")
                    provisional.append(entry)
                    awaiting_boundary = not event["retry_scheduled"]
                    resumed_attempt = entry["attempt"] + 1
                    continue
                else:
                    raise QueryCheckpointError("Query-progress decision is unsupported.")
                search_request_count += 1
                trace.append(entry)
                resumed_attempt = 1
            if awaiting_boundary:
                # New explicit invocation: the unfinished tile gets one fresh
                # bounded budget; its failed attempts contributed no science.
                _progress.commit({"invocation_boundary": boundary_for(pending[0])})
                provisional.clear()
                resumed_attempt = 1
            for retried in provisional:  # interrupted scheduled retry continues
                aggregate_partial_rows += retried["partial_rows_discarded"]
                retry_exception_types.add(retried["exception_type"])
                retry_count += 1
                search_request_count += 1
                trace.append(retried)

        def deduplicated_frame() -> Tuple[pd.DataFrame, int, list[str]]:
            if not accepted_records:
                return pd.DataFrame(), 0, []
            raw = pd.DataFrame(accepted_records)
            duplicate_mask = raw["locus_id"].duplicated(keep=False)
            duplicate_ids = sorted(
                set(raw.loc[duplicate_mask, "locus_id"].astype(str).tolist())
            )
            frame = raw.drop_duplicates(subset=["locus_id"], keep="last")
            frame = frame.reset_index(drop=True)
            return frame, len(raw) - len(frame), duplicate_ids

        def completion_details(
            classification: LiveCompletion,
            *,
            coverage_complete: bool,
            unresolved_saturated_tiles: int,
        ) -> Dict[str, Any]:
            finished = self.clock()
            frame, duplicate_rows_removed, duplicate_ids = deduplicated_frame()
            trace_payload = {"tiles": trace}
            accepted_count = len(accepted_tiles)
            processed_count = accepted_count + split_count + unresolved_saturated_tiles
            return {
                "completion_classification": classification.value,
                "target_date_utc": request.date_utc,
                "interval": dict(scientific_contract["interval"]),
                "spatial_domain": dict(scientific_contract["spatial_domain"]),
                "query_sha256": contract_hash,
                "query_contract_sha256": contract_hash,
                "query_tag": None,
                "lsst_only": True,
                "lsst_filter": scientific_contract["lsst_filter"],
                "lsst_filter_sha256": _sha256_json(
                    {"filter": scientific_contract["lsst_filter"]}
                ),
                "sort_requested": None,
                "service_ordering": "ANTARES client/API default",
                "pagination_mode": "antares-client-jsonapi-links-next",
                "terminal_evidence": (
                    "every accepted tile returned fewer than 50 rows and ended in normal iterator exhaustion"
                    if coverage_complete
                    else "positive exhaustion or exact 3-D coverage proof is incomplete"
                ),
                "extraction_method": extraction_method_contract(),
                "execution_policy": self.execution_policy(),
                "cache_used": False,
                "capability_environment": self.capability.environment,
                "initial_tile_override": self._initial_tiles_overridden,
                "initial_tile_count": len(initial_tiles),
                "search_request_count": search_request_count,
                "processed_tile_count": processed_count,
                "logical_chunk_count": accepted_count,
                "accepted_chunk_count": accepted_count,
                "accepted_tile_count": accepted_count,
                "split_count": split_count,
                "unresolved_saturated_chunk_count": unresolved_saturated_tiles,
                "unresolved_saturated_tile_count": unresolved_saturated_tiles,
                "iterator_exhausted_accepted_chunks": accepted_count,
                "iterator_exhausted_accepted_tiles": accepted_count,
                "all_accepted_iterators_exhausted": coverage_complete,
                "iterator_exhausted": coverage_complete,
                "coverage_complete": coverage_complete,
                "coverage_lineage_complete": coverage_complete,
                "terminal_pending_tile_count": len(pending),
                "raw_returned_loci": len(accepted_records),
                "returned_loci": len(frame),
                "deduplication": {
                    **scientific_contract["deduplication"],
                    "raw_rows": len(accepted_records),
                    "duplicate_rows_removed": duplicate_rows_removed,
                    "duplicate_identity_count": len(duplicate_ids),
                    "duplicate_identities": duplicate_ids,
                    "duplicate_identity_sha256": _identifier_hash(duplicate_ids),
                },
                "partial_rows_discarded": aggregate_partial_rows,
                "retry_count": retry_count,
                "retry_exception_types": sorted(retry_exception_types),
                "tile_trace_sha256": _sha256_json(trace_payload),
                "tile_trace": list(trace),
                "locus_order_sha256": _identifier_hash(
                    frame["locus_id"].astype(str).tolist()
                    if "locus_id" in frame.columns
                    else []
                ),
                "request_started_at_utc": _iso(started),
                "request_completed_at_utc": _iso(finished),
                "runtime_seconds": round(max(0.0, self.monotonic() - t0), 6),
                "client": self.client_identity(),
                "secret_material_recorded": False,
            }

        def failed_result(
            *,
            incomplete: bool,
            code: str,
            unresolved_saturated_tiles: int = 0,
        ) -> NightQueryResult:
            classification = LiveCompletion.INCOMPLETE if incomplete else LiveCompletion.FAILED
            outcome = (
                ProviderOutcome.QUERY_INTERRUPTION
                if incomplete
                else ProviderOutcome.QUERY_FAILURE
            )
            issue = ProviderIssue(
                code,
                ProviderStage.QUERY,
                outcome,
                "The probe-first ANTARES query did not prove complete 3-D coverage.",
                retryable=code != "live_query_malformed",
                partial=incomplete,
            )
            frame, _duplicates, _duplicate_ids = deduplicated_frame()
            details = completion_details(
                classification,
                coverage_complete=False,
                unresolved_saturated_tiles=unresolved_saturated_tiles,
            )
            return NightQueryResult(
                request,
                self.provider_name,
                self.scenario,
                outcome,
                frame if not frame.empty else None,
                QueryStageEvidence(
                    False,
                    incomplete,
                    len(frame),
                    (issue,),
                    details,
                ),
            )

        while pending:
            tile = pending.popleft()
            body = _build_tile_query(tile)
            tile_records = []
            saturated = False
            exhausted = False
            for attempt in range(resumed_attempt, self.max_query_attempts + 1):
                tile_records = []
                saturated = False
                exhausted = False
                search_request_count += 1
                try:
                    iterator = iter(search(body))
                    while True:
                        try:
                            locus = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break
                        try:
                            record = query.locus_to_record(locus)
                        except Exception as exc:
                            raise ValueError("ANTARES locus could not be normalized.") from exc
                        locus_id = str(record.get("locus_id") or "").strip()
                        if not locus_id:
                            raise ValueError("ANTARES locus lacks a locus_id.")
                        record["locus_id"] = locus_id
                        if not _record_matches_tile(record, tile):
                            raise ValueError("ANTARES locus is outside its query tile.")
                        tile_records.append(record)
                        if len(tile_records) >= PROBE_LIMIT:
                            saturated = True
                            close = getattr(iterator, "close", None)
                            if callable(close):
                                close()
                            break
                except Exception as exc:
                    aggregate_partial_rows += len(tile_records)
                    retryable = _retryable_query_error(exc)
                    exception_type = _exception_type(exc)
                    retry_exception_types.add(exception_type)
                    trace.append(
                        {
                            **tile,
                            "attempt": attempt,
                            "status": "attempt_error",
                            "iterator_exhausted": False,
                            "partial_rows_discarded": len(tile_records),
                            "exception_type": exception_type,
                            "retryable": retryable,
                            "query_sha256": _sha256_json(body),
                        }
                    )
                    commit_decision(trace[-1], retry_scheduled=retryable and attempt < self.max_query_attempts)
                    if retryable and attempt < self.max_query_attempts:
                        retry_count += 1
                        self.sleeper(self.retry_delay_seconds * attempt)
                        continue
                    incomplete = bool(accepted_records or aggregate_partial_rows)
                    return failed_result(
                        incomplete=incomplete,
                        code=(
                            "live_query_incomplete"
                            if retryable
                            else "live_query_malformed"
                        ),
                    )
                break

            resumed_attempt = 1

            if saturated:
                aggregate_partial_rows += len(tile_records)
                children = _split_tile(tile)
                if not children:
                    trace.append(
                        {
                            **tile,
                            "attempt": attempt,
                            "status": "unresolved_saturated_minimum",
                            "returned_before_split": len(tile_records),
                            "iterator_exhausted": False,
                            "query_sha256": _sha256_json(body),
                        }
                    )
                    return failed_result(
                        incomplete=True,
                        code="live_query_saturation_unresolved",
                        unresolved_saturated_tiles=1,
                    )
                pending.appendleft(children[1])
                pending.appendleft(children[0])
                split_count += 1
                trace.append(
                    {
                        **tile,
                        "attempt": attempt,
                        "status": "split_saturated",
                        "returned_before_split": len(tile_records),
                        "iterator_exhausted": False,
                        "query_sha256": _sha256_json(body),
                    }
                )
                commit_decision(trace[-1])
                continue
            if not exhausted:
                return failed_result(
                    incomplete=bool(accepted_records or tile_records),
                    code="live_query_incomplete",
                )
            accepted_records.extend(tile_records)
            accepted_tiles.append(tile)
            trace.append(
                {
                    **tile,
                    "attempt": attempt,
                    "status": "accepted_exhausted",
                    "returned_loci": len(tile_records),
                    "iterator_exhausted": True,
                    "query_sha256": _sha256_json(body),
                }
            )
            commit_decision(trace[-1], tile_records)

        coverage_complete = bool(
            len(accepted_tiles) == len(initial_tiles) + split_count
            and not pending
        )
        if not coverage_complete:
            return failed_result(
                incomplete=bool(accepted_records),
                code="live_query_coverage_gap",
            )

        frame, _duplicates, _duplicate_ids = deduplicated_frame()
        completion = (
            LiveCompletion.COMPLETE_ZERO
            if frame.empty
            else LiveCompletion.COMPLETE_NONZERO
        )
        details = completion_details(
            completion,
            coverage_complete=True,
            unresolved_saturated_tiles=0,
        )
        return NightQueryResult(
            request,
            self.provider_name,
            self.scenario,
            ProviderOutcome.SUCCESS_ZERO if frame.empty else ProviderOutcome.SUCCESS,
            frame,
            QueryStageEvidence(True, False, len(frame), (), details),
        )

    def _fetch_one(
        self,
        locus_id: str,
        get_by_id: Callable[[str], Any],
        label: str,
    ) -> Mapping[str, Any]:
        retries = 0
        errors = []
        for attempt in range(1, self.max_fetch_attempts + 1):
            try:
                locus = get_by_id(locus_id)
                if locus is None:
                    raise LookupError("ANTARES returned no locus for an enumerated id.")
                if str(getattr(locus, "locus_id", "")) != locus_id:
                    raise ValueError("Fetched ANTARES locus identity differs from request.")
                lightcurve = locus.lightcurve
                if lightcurve is not None and not isinstance(lightcurve, pd.DataFrame):
                    raise TypeError("ANTARES lightcurve is not a DataFrame or null.")
                if lightcurve is None or lightcurve.empty:
                    frame = None
                    rows = 0
                else:
                    frame = lightcurve.copy()
                    frame["locus_id"] = locus_id
                    frame["range_label"] = label
                    rows = len(frame)
                return {
                    "locus_id": locus_id,
                    "completed": True,
                    "frame": frame,
                    "alert_rows": rows,
                    "retry_count": retries,
                    "attempt_errors": errors,
                }
            except Exception as exc:
                errors.append(_exception_type(exc))
                if attempt < self.max_fetch_attempts:
                    retries += 1
                    self.sleeper(self.retry_delay_seconds * attempt)
                    continue
                return {
                    "locus_id": locus_id,
                    "completed": False,
                    "frame": None,
                    "alert_rows": 0,
                    "retry_count": retries,
                    "attempt_errors": errors,
                }
        raise AssertionError("Unreachable fetch attempt state.")

    def _failed_fetch(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
    ) -> NightScienceResult:
        issue = ProviderIssue(
            "fetch_not_attempted_after_query",
            ProviderStage.FETCH,
            query_result.outcome,
            "Fetch was refused because query completion was not clean.",
            retryable=True,
            partial=query_result.evidence.partial,
        )
        evidence = FetchStageEvidence(
            False,
            False,
            0,
            0,
            (issue,),
            {
                "completion_classification": LiveCompletion.FAILED.value,
                "requested_objects": 0,
                "completed_objects": 0,
                "failed_objects": 0,
                "retry_count": 0,
                "secret_material_recorded": False,
            },
        )
        return NightScienceResult(
            request,
            self.provider_name,
            self.scenario,
            query_result.outcome,
            query_result,
            None,
            None,
            evidence,
            {},
            (),
            _combined_evidence(query_result.evidence, evidence),
        )

    def _successful_fetch_result(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
        raw_alerts: pd.DataFrame,
        details: Mapping[str, Any],
    ) -> NightScienceResult:
        """Apply the unchanged Phase 6 preparation and science-validation path."""

        assert isinstance(query_result.loci, pd.DataFrame)
        loci = prepare_loci(
            query_result.loci,
            request.date_utc,
            request.mjd_min,
            request.mjd_max,
            query_result.evidence.details.get(
                "request_completed_at_utc", request.ingested_at_utc
            ),
            source_query_mode=(self.proof_profile.extraction_method()["name"]
                               if getattr(self, "proof_profile", None) is not None else EXTRACTION_METHOD),
        )
        alerts = prepare_alerts(raw_alerts, request.date_utc, request.range_label)
        fetch_evidence = FetchStageEvidence(
            True,
            False,
            len(loci),
            len(alerts),
            (),
            dict(details),
        )
        combined = _combined_evidence(query_result.evidence, fetch_evidence)
        validation = validation_summary(
            loci,
            alerts,
            mjd_min=request.mjd_min,
            mjd_max=request.mjd_max,
            prior_locus_ids=request.prior_locus_ids,
            lsst_only=True,
            query_completed=True,
            query_fetch_clean=combined.clean,
            mjd_upper_exclusive=True,
        )
        validation_errors: Tuple[ProviderIssue, ...] = ()
        outcome = (
            ProviderOutcome.SUCCESS_ZERO
            if loci.empty and alerts.empty
            else ProviderOutcome.SUCCESS
        )
        if validation.get("append_ready") is not True:
            outcome = ProviderOutcome.VALIDATION_FAILURE
            validation_errors = (
                ProviderIssue(
                    "live_science_validation_failed",
                    ProviderStage.VALIDATION,
                    outcome,
                    "Existing nightly validation did not grant append readiness.",
                    retryable=False,
                ),
            )
        return NightScienceResult(
            request,
            self.provider_name,
            self.scenario,
            outcome,
            query_result,
            loci,
            alerts,
            fetch_evidence,
            validation,
            validation_errors,
            combined,
        )

    def fetch_segment(self, request: NightScienceRequest, requested: Tuple[str, ...]):
        """The existing segmented per-object fetch path, shared by range acquisition."""
        from .fetch_checkpoint import FetchObjectResult, FetchCheckpointFetchError
        self._validate_request(request)
        _search, get_by_id, _connectivity = self._load_client()
        if not requested:
            return ()
        results: Dict[str, Mapping[str, Any]] = {}
        workers = min(self.max_fetch_workers, len(requested))
        batch_size = max(workers, workers * 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for offset in range(0, len(requested), batch_size):
                batch = requested[offset : offset + batch_size]
                futures = {
                    pool.submit(
                        self._fetch_one,
                        locus_id,
                        get_by_id,
                        request.range_label,
                    ): locus_id
                    for locus_id in batch
                }
                for future in as_completed(futures):
                    locus_id = futures[future]
                    try:
                        results[locus_id] = future.result()
                    except Exception as exc:
                        results[locus_id] = {
                            "locus_id": locus_id,
                            "completed": False,
                            "frame": None,
                            "retry_count": 0,
                            "attempt_errors": [_exception_type(exc)],
                        }
        failed = [value for value in requested if not results[value]["completed"]]
        if failed:
            raise FetchCheckpointFetchError(
                "A deterministic fetch segment did not complete every object."
            )
        return tuple(
            FetchObjectResult(
                locus_id,
                results[locus_id]["frame"],
                retry_count=int(results[locus_id]["retry_count"]),
                retry_exception_types=tuple(results[locus_id]["attempt_errors"]),
            )
            for locus_id in requested
        )


    def fetch_resumable(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
        checkpoint: object,
        *,
        event_hook: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    ) -> NightScienceResult:
        """Fetch or reopen exact query-ordered histories through sealed segments.

        The checkpoint owns only operational durability.  This method retains
        the existing per-object transport, preparation, validation, and result
        semantics; a valid completed checkpoint invokes no ANTARES callback.
        """

        from .fetch_checkpoint import (
            FetchCheckpointError,
            FetchCheckpointFetchError,
            FetchObjectResult,
            SegmentedFetchCheckpoint,
        )

        self._validate_request(request)
        if not isinstance(query_result, NightQueryResult):
            raise ProviderContractError(
                ProviderIssue(
                    "invalid_query_result_type",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "Live fetch requires a NightQueryResult.",
                )
            )
        if query_result.request != request:
            raise ProviderContractError(
                ProviderIssue(
                    "query_request_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "The query result belongs to another request.",
                ),
                result=query_result,
            )
        if (
            query_result.provider_name != self.provider_name
            or query_result.scenario != self.scenario
        ):
            raise ProviderContractError(
                ProviderIssue(
                    "query_provider_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "The query result belongs to another provider.",
                ),
                result=query_result,
            )
        if not query_result.clean:
            return self._failed_fetch(request, query_result)
        if not isinstance(query_result.loci, pd.DataFrame):
            raise ProviderContractError(
                ProviderIssue(
                    "query_loci_missing",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.MALFORMED_RESULT,
                    "A completed live query must carry a loci DataFrame.",
                ),
                result=query_result,
            )
        if type(checkpoint) is not SegmentedFetchCheckpoint:
            raise ProviderContractError(
                ProviderIssue(
                    "invalid_fetch_checkpoint_type",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "Resumable live fetch requires a sealed segmented checkpoint.",
                )
            )

        raw_loci = query_result.loci
        if "locus_id" not in raw_loci.columns and not raw_loci.empty:
            raise ProviderContractError(
                ProviderIssue(
                    "live_query_locus_id_missing",
                    ProviderStage.FETCH,
                    ProviderOutcome.MALFORMED_RESULT,
                    "Completed live query rows lack locus_id.",
                ),
                result=query_result,
            )
        locus_ids = (
            raw_loci["locus_id"].astype(str).tolist() if not raw_loci.empty else []
        )
        started = self.clock()
        t0 = self.monotonic()
        def fetch_segment(requested):
            return self.fetch_segment(request, requested)

        try:
            completion = (checkpoint.inspect_complete(locus_ids) if checkpoint.read_only else
                          checkpoint.fetch_missing(locus_ids, fetch_segment, event_hook=event_hook))
            segment_frames = []
            lightcurves_with_rows = 0
            lightcurves_empty = 0
            for segment in checkpoint.iter_segments(locus_ids):
                segment_frames.append(segment.alerts)
                lightcurves_with_rows += sum(
                    1 for value in segment.objects if int(value["alert_rows"]) > 0
                )
                lightcurves_empty += sum(
                    1 for value in segment.objects if int(value["alert_rows"]) == 0
                )
        except FetchCheckpointError as exc:
            raise FetchProviderError(
                ProviderIssue(
                    "live_fetch_checkpoint_failed",
                    ProviderStage.FETCH,
                    ProviderOutcome.FETCH_FAILURE,
                    "Run-local fetch checkpoint validation or completion failed.",
                    retryable=False,
                ),
                result=query_result,
            ) from exc

        raw_alerts = (
            pd.concat(segment_frames, ignore_index=True, sort=False)
            if segment_frames
            else pd.DataFrame()
        )
        finished = self.clock()
        effective_workers = min(self.max_fetch_workers, len(locus_ids)) if locus_ids else 0
        details = {
            "completion_classification": (
                LiveCompletion.COMPLETE_ZERO.value
                if not locus_ids
                else LiveCompletion.COMPLETE_NONZERO.value
            ),
            "requested_objects": len(locus_ids),
            "completed_objects": len(locus_ids),
            "failed_objects": 0,
            "failed_object_identity_sha256": _identifier_hash(()),
            "failure_exception_types": [],
            "retry_exception_types": list(completion.retry_exception_types),
            "retry_count": completion.retry_count,
            "lightcurves_with_rows": lightcurves_with_rows,
            "lightcurves_empty": lightcurves_empty,
            "full_locus_history_requests": len(locus_ids),
            "full_locus_history_completed": len(locus_ids),
            "alert_rows": len(raw_alerts),
            "request_started_at_utc": _iso(started),
            "request_completed_at_utc": _iso(finished),
            "runtime_seconds": round(max(0.0, self.monotonic() - t0), 6),
            "max_workers": self.max_fetch_workers,
            "effective_workers": effective_workers,
            "max_in_flight_futures": (
                min(len(locus_ids), effective_workers * 4) if locus_ids else 0
            ),
            "max_attempts_per_object": self.max_fetch_attempts,
            "cache_used": False,
            "secret_material_recorded": False,
            "checkpoint": {
                "schema_version": "phase6.segmented-fetch-checkpoint.v1",
                "identity_sha256": completion.checkpoint_identity_sha256,
                "completion_sha256": completion.completion_sha256,
                "segment_count": completion.segment_count,
                "reused_segments": completion.reused_segments,
                "fetched_segments": completion.fetched_segments,
                "authoritative": False,
                "production_eligible": False,
            },
        }
        return self._successful_fetch_result(
            request,
            query_result,
            raw_alerts,
            details,
        )

    def fetch(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
    ) -> NightScienceResult:
        self._validate_request(request)
        if not isinstance(query_result, NightQueryResult):
            raise ProviderContractError(
                ProviderIssue(
                    "invalid_query_result_type",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "Live fetch requires a NightQueryResult.",
                )
            )
        if query_result.request != request:
            raise ProviderContractError(
                ProviderIssue(
                    "query_request_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "The query result belongs to another request.",
                ),
                result=query_result,
            )
        if (
            query_result.provider_name != self.provider_name
            or query_result.scenario != self.scenario
        ):
            raise ProviderContractError(
                ProviderIssue(
                    "query_provider_mismatch",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.FETCH_FAILURE,
                    "The query result belongs to another provider.",
                ),
                result=query_result,
            )
        if not query_result.clean:
            return self._failed_fetch(request, query_result)
        if not isinstance(query_result.loci, pd.DataFrame):
            raise ProviderContractError(
                ProviderIssue(
                    "query_loci_missing",
                    ProviderStage.CONTRACT,
                    ProviderOutcome.MALFORMED_RESULT,
                    "A completed live query must carry a loci DataFrame.",
                ),
                result=query_result,
            )

        _search, get_by_id, _connectivity = self._load_client()
        raw_loci = query_result.loci
        if "locus_id" not in raw_loci.columns and not raw_loci.empty:
            issue = ProviderIssue(
                "live_query_locus_id_missing",
                ProviderStage.FETCH,
                ProviderOutcome.MALFORMED_RESULT,
                "Completed live query rows lack locus_id.",
            )
            evidence = FetchStageEvidence(False, False, 0, 0, (issue,), {})
            return NightScienceResult(
                request,
                self.provider_name,
                self.scenario,
                ProviderOutcome.MALFORMED_RESULT,
                query_result,
                None,
                None,
                evidence,
                {},
                (),
                _combined_evidence(query_result.evidence, evidence),
            )

        locus_ids = raw_loci["locus_id"].astype(str).tolist() if not raw_loci.empty else []
        started = self.clock()
        t0 = self.monotonic()
        results: Dict[str, Mapping[str, Any]] = {}
        if locus_ids:
            effective_workers = min(self.max_fetch_workers, len(locus_ids))
            batch_size = max(effective_workers, effective_workers * 4)
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                for offset in range(0, len(locus_ids), batch_size):
                    batch = locus_ids[offset : offset + batch_size]
                    futures = {
                        pool.submit(
                            self._fetch_one,
                            locus_id,
                            get_by_id,
                            request.range_label,
                        ): locus_id
                        for locus_id in batch
                    }
                    for future in as_completed(futures):
                        locus_id = futures[future]
                        try:
                            results[locus_id] = future.result()
                        except Exception as exc:
                            results[locus_id] = {
                                "locus_id": locus_id,
                                "completed": False,
                                "frame": None,
                                "alert_rows": 0,
                                "retry_count": 0,
                                "attempt_errors": [_exception_type(exc)],
                            }

        completed_ids = [item for item in locus_ids if results[item]["completed"]]
        failed_ids = [item for item in locus_ids if not results[item]["completed"]]
        retry_count = sum(int(results[item]["retry_count"]) for item in locus_ids)
        frames = [
            results[item]["frame"]
            for item in locus_ids
            if results[item]["completed"] and results[item]["frame"] is not None
        ]
        raw_alerts = (
            pd.concat(frames, ignore_index=True, sort=False)
            if frames
            else pd.DataFrame()
        )
        finished = self.clock()
        partial = bool(failed_ids and completed_ids)
        failed = bool(failed_ids and not completed_ids)
        completion = (
            LiveCompletion.INCOMPLETE
            if partial
            else LiveCompletion.FAILED
            if failed
            else LiveCompletion.COMPLETE_ZERO
            if not locus_ids
            else LiveCompletion.COMPLETE_NONZERO
        )
        details = {
            "completion_classification": completion.value,
            "requested_objects": len(locus_ids),
            "completed_objects": len(completed_ids),
            "failed_objects": len(failed_ids),
            "failed_object_identity_sha256": _identifier_hash(failed_ids),
            "failure_exception_types": sorted(
                {
                    error_type
                    for item in failed_ids
                    for error_type in results[item]["attempt_errors"]
                }
            ),
            "retry_exception_types": sorted(
                {
                    error_type
                    for item in locus_ids
                    for error_type in results[item]["attempt_errors"]
                }
            ),
            "retry_count": retry_count,
            "lightcurves_with_rows": len(frames),
            "lightcurves_empty": sum(
                1
                for item in completed_ids
                if results[item]["frame"] is None
            ),
            "full_locus_history_requests": len(locus_ids),
            "full_locus_history_completed": len(completed_ids),
            "alert_rows": len(raw_alerts),
            "request_started_at_utc": _iso(started),
            "request_completed_at_utc": _iso(finished),
            "runtime_seconds": round(max(0.0, self.monotonic() - t0), 6),
            "max_workers": self.max_fetch_workers,
            "effective_workers": min(self.max_fetch_workers, len(locus_ids)) if locus_ids else 0,
            "max_in_flight_futures": (
                min(
                    len(locus_ids),
                    min(self.max_fetch_workers, len(locus_ids)) * 4,
                )
                if locus_ids
                else 0
            ),
            "max_attempts_per_object": self.max_fetch_attempts,
            "cache_used": False,
            "secret_material_recorded": False,
        }
        if failed_ids:
            outcome = ProviderOutcome.PARTIAL_FETCH if partial else ProviderOutcome.FETCH_FAILURE
            issue = ProviderIssue(
                "live_partial_fetch" if partial else "live_fetch_failed",
                ProviderStage.FETCH,
                outcome,
                "One or more enumerated ANTARES loci were not completely fetched.",
                retryable=True,
                partial=partial,
            )
            fetch_evidence = FetchStageEvidence(
                False,
                partial,
                len(raw_loci),
                len(raw_alerts),
                (issue,),
                details,
            )
            loci = prepare_loci(
                raw_loci,
                request.date_utc,
                request.mjd_min,
                request.mjd_max,
                query_result.evidence.details.get(
                    "request_completed_at_utc", request.ingested_at_utc
                ),
                source_query_mode=EXTRACTION_METHOD,
            )
            alerts = prepare_alerts(raw_alerts, request.date_utc, request.range_label)
            return NightScienceResult(
                request,
                self.provider_name,
                self.scenario,
                outcome,
                query_result,
                loci,
                alerts,
                fetch_evidence,
                {},
                (),
                _combined_evidence(query_result.evidence, fetch_evidence),
            )

        return self._successful_fetch_result(
            request,
            query_result,
            raw_alerts,
            details,
        )

    def fetch_night(self, request: NightScienceRequest) -> NightScienceResult:
        return self.fetch(request, self.query(request))


__all__ = [
    "LIVE_ANTARES_READ",
    "LiveAntaresProvider",
    "LiveAntaresReadCapability",
    "LiveCapabilityError",
    "LiveCompletion",
    "OFFICIAL_API_BASE_URL",
    "PINNED_CLIENT_VERSION",
]


# Explicit offline opt-in. Historical functions and the default P1 grammar above
# retain their original meaning. No production limits or attestation live here.
P2_GRAMMAR = "v3.p2-query-decisions.v1"


@dataclass(frozen=True)
class P2ProofProfile:
    max_depth: int
    max_nodes_per_root: int
    max_nodes_per_night: int
    max_search_attempts: int
    crash_reserve: int
    max_event_bytes: int
    # None keeps the G6.6.2B offline-only profile (mocked callbacks only).
    # Exact P2TransportLimits is the only route to live searches (G6.6.3A).
    transport: Optional["P2TransportLimits"] = None

    def __post_init__(self):
        for name in ("max_depth", "max_nodes_per_root", "max_nodes_per_night",
                     "max_search_attempts", "crash_reserve", "max_event_bytes"):
            value = getattr(self, name)
            minimum = 0 if name in {"max_depth", "crash_reserve"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a finite integer >= {minimum}.")
        if self.transport is not None and type(self.transport) is not P2TransportLimits:
            raise ValueError("transport must be exact P2TransportLimits or None.")

    def as_dict(self):
        document = {"schema_version": "v3.p2-proof-profile.v1", "grammar": P2_GRAMMAR,
                "primary": extraction_method_contract(), "trigger": "saturated_primary_floor",
                "axis_ties": ["time", "ra", "dec"], "midpoint": "binary64-(lower+upper)/2",
                "traversal": "lower-child-first", "acceptance": "natural-exhaustion-below-50",
                **{name: getattr(self, name) for name in (
                    "max_depth", "max_nodes_per_root", "max_nodes_per_night",
                    "max_search_attempts", "crash_reserve", "max_event_bytes")}}
        if self.transport is not None:
            document.update(schema_version="v3.p2-proof-profile.v2",
                            transport=self.transport.as_dict())
        return document

    def extraction_method(self):
        return {**extraction_method_contract(), "name": "hierarchical_completeness_p2",
                "cache_version": "offline_p2_v1" if self.transport is None else "guarded_p2_v1",
                "proof_profile": self.as_dict()}


def scientific_contract_for_profile(request, profile=None):
    contract = _scientific_query_contract(request)
    if profile is not None:
        if type(profile) is not P2ProofProfile:
            raise ValueError("Unsupported proof profile.")
        contract = {**contract, "extraction_method": profile.extraction_method()}
    return contract


def _p2_split(tile):
    ratios = ((tile["mjd_max"] - tile["mjd_min"]) * 86400.0 / 30.0,
              (tile["ra_max"] - tile["ra_min"]) / 0.05,
              (tile["dec_max"] - tile["dec_min"]) / 0.05)
    index = max(range(3), key=lambda n: ratios[n])
    lower, upper = (("mjd_min", "mjd_max"), ("ra_min", "ra_max"),
                    ("dec_min", "dec_max"))[index]
    midpoint = (tile[lower] + tile[upper]) / 2.0
    if not math.isfinite(midpoint) or not tile[lower] < midpoint < tile[upper]:
        raise ValueError("p2_midpoint_collapse")
    children = ({**tile, upper: midpoint}, {**tile, lower: midpoint})
    return ("time", "ra", "dec")[index], midpoint, children


def _p2_node(tile, node_id, parent=None, root=None, depth=None):
    return {"id": node_id, "parent": parent, "root": root, "depth": depth,
            "tile": dict(tile)}


class _P2State:
    """Provider-side deterministic reducer. Science has a separate verifier."""

    def __init__(self, request, profile, initial, attempts):
        self.request, self.profile, self.attempt_limit = request, profile, attempts
        self.frontier = deque(_p2_node(tile, f"i{n:05d}") for n, tile in enumerate(initial))
        self.active = None
        self.starts = self.unknowns = self.splits = self.discarded = self.errors = 0
        self.root_nodes, self.outcomes = {}, {}
        self.records, self.trace, self.accepted, self.error_types = [], [], [], set()
        self.failed = None

    def budget(self):
        return {"search_attempts": self.starts, "unknown_attempts": self.unknowns,
                "secondary_nodes": sum(self.root_nodes.values()),
                "root_nodes": [[key, self.root_nodes[key]] for key in sorted(self.root_nodes)]}

    def base(self, kind, reservation=None):
        node = self.frontier[0]
        return {"grammar": P2_GRAMMAR, "profile_sha256": _sha256_json(self.profile.as_dict()),
                "kind": kind, "node": node,
                "query_sha256": _sha256_json(_build_tile_query(node["tile"])),
                "reservation": reservation, "before": self.budget()}

    def reservation(self):
        if self.active is not None or not self.frontier or self.failed:
            raise ValueError("P2 cannot reserve this frontier.")
        if self.starts >= self.profile.max_search_attempts:
            return None, "p2_search_budget_exhausted"
        attempt = self.outcomes.get(self.frontier[0]["id"], 0) + 1
        if attempt > self.attempt_limit:
            return None, "p2_retry_exhausted"
        reservation = {"id": self.starts + 1, "attempt": attempt}
        event = self.base("reserve", reservation)
        event["after"] = {**self.budget(), "search_attempts": self.starts + 1}
        return {"p2_event": event, "records": []}, None

    def unknown(self):
        if self.active is None:
            raise ValueError("P2 has no unknown reservation.")
        if self.unknowns >= self.profile.crash_reserve:
            return self.failure("p2_crash_reserve_exhausted")
        event = self.base("unknown", self.active)
        event["after"] = {**self.budget(), "unknown_attempts": self.unknowns + 1}
        return {"p2_event": event, "records": []}

    def failure(self, reason):
        event = self.base("failure", self.active)
        event["reason"] = reason
        event["after"] = self.budget()
        return {"p2_event": event, "records": []}

    def outcome(self, count, exhausted, *, exception_type=None, retryable=False, records=()):
        from .query_progress import encode_records
        if self.active is None:
            raise ValueError("P2 outcome lacks a reservation.")
        node = self.frontier[0]
        event = self.base("outcome", self.active)
        decision = {"validated_rows": count, "iterator_exhausted": exhausted,
                    "discarded_rows": count if exception_type or count == 50 else 0,
                    "exception_type": exception_type, "retryable": retryable,
                    "children": [], "dimension": None, "midpoint": None,
                    "entered_secondary": False, "reason": None}
        after = self.budget()
        if exception_type:
            decision["status"] = "attempt_error"
            if not retryable or self.active["attempt"] >= self.attempt_limit:
                decision["reason"] = ("p2_retry_exhausted" if retryable else
                                      "p2_transport_guard" if exception_type in _P2_TRANSPORT_GUARD_TYPES
                                      else "p2_malformed")
        elif exhausted and 0 <= count < 50:
            decision["status"] = "accepted_exhausted"
        elif count == 50 and not exhausted:
            decision["status"] = "split_saturated"
            children = _split_tile(node["tile"]) if node["root"] is None else ()
            if children:
                dimension, midpoint, independently = _p2_split(node["tile"])
                if children != independently:
                    raise ValueError("P1/P2 primary arithmetic differs.")
                root, depth = None, None
            else:
                root = node["root"] or node["id"]
                depth = node["depth"] if node["root"] is not None else 0
                decision["entered_secondary"] = node["root"] is None
                counts = dict(self.root_nodes)
                if decision["entered_secondary"]:
                    counts[root] = 1
                if depth >= self.profile.max_depth:
                    decision["reason"] = "p2_depth_exhausted"
                elif counts[root] + 2 > self.profile.max_nodes_per_root:
                    decision["reason"] = "p2_root_nodes_exhausted"
                elif sum(counts.values()) + 2 > self.profile.max_nodes_per_night:
                    decision["reason"] = "p2_night_nodes_exhausted"
                else:
                    try:
                        dimension, midpoint, children = _p2_split(node["tile"])
                    except ValueError:
                        decision["reason"] = "p2_midpoint_collapse"
                if decision["reason"] is None:
                    counts[root] += 2
                if sum(counts.values()) > self.profile.max_nodes_per_night:
                    # Entry itself cannot exceed the aggregate root capacity.
                    counts = dict(self.root_nodes)
                    decision["reason"] = "p2_night_nodes_exhausted"
                after = {**after, "secondary_nodes": sum(counts.values()),
                         "root_nodes": [[key, counts[key]] for key in sorted(counts)]}
            if decision["reason"] is None:
                decision.update(dimension=dimension, midpoint=midpoint,
                    children=[_p2_node(tile, node["id"] + str(n), node["id"], root,
                                       depth + 1 if root is not None else None)
                              for n, tile in enumerate(children)])
        else:
            raise ValueError("P2 outcome lacks positive exhaustion or saturation.")
        event.update(decision=decision, after=after)
        return {"p2_event": event, "records": encode_records(records)}

    def apply(self, event):
        from .query_progress import decode_records
        from .query_checkpoint import QueryCheckpointError
        try:
            if (not self.frontier or self.failed or not isinstance(event, dict)
                    or set(event) != {"p2_event", "records"}):
                raise ValueError("P2 event/frontier is invalid.")
            meta, records = event["p2_event"], decode_records(event["records"])
            if not isinstance(meta, dict):
                raise ValueError("P2 metadata is invalid.")
            kind = meta.get("kind")
            if kind == "reserve":
                expected, reason = self.reservation()
                if reason or _sha256_json(event) != _sha256_json(expected):
                    raise ValueError("P2 reservation differs.")
                self.starts += 1
                self.active = meta["reservation"]
                return
            if kind == "unknown":
                if _sha256_json(event) != _sha256_json(self.unknown()):
                    raise ValueError("P2 unknown-outcome evidence differs.")
                self.unknowns += 1
                self.active = None
                return
            if kind == "failure":
                reason = meta.get("reason")
                expected = self.failure(reason)
                permitted = (reason == "p2_crash_reserve_exhausted" and self.active is not None
                             and self.unknowns >= self.profile.crash_reserve)
                if self.active is None:
                    _, blocked = self.reservation()
                    permitted = reason == blocked and blocked is not None
                if not permitted or _sha256_json(event) != _sha256_json(expected):
                    raise ValueError("P2 failure is not justified.")
                self.failed = reason
                return
            if kind != "outcome" or self.active is None:
                raise ValueError("P2 outcome is unsupported/unreserved.")
            decision = meta.get("decision", {})
            count, exhausted = decision.get("validated_rows"), decision.get("iterator_exhausted")
            error, retryable = decision.get("exception_type"), decision.get("retryable")
            if (type(count) is not int or not 0 <= count <= 50 or type(exhausted) is not bool
                    or type(retryable) is not bool or (error is not None and
                    (not isinstance(error, str) or not error))):
                raise ValueError("P2 observed response is invalid.")
            if error is None and retryable:
                raise ValueError("P2 success cannot be retryable.")
            expected = self.outcome(count, exhausted, exception_type=error,
                                    retryable=retryable, records=records)
            if _sha256_json(event) != _sha256_json(expected):
                raise ValueError("P2 decision/budget transition differs.")
            accepted = decision["status"] == "accepted_exhausted"
            if ((accepted and (len(records) != count or any(
                    not isinstance(r, dict) or not isinstance(r.get("locus_id"), str)
                    or not r["locus_id"] or r["locus_id"] != r["locus_id"].strip()
                    or not _record_matches_tile(r, self.frontier[0]["tile"])
                    or query.lsst_identifier_counts(pd.DataFrame([r]))["lsst_identifier_count"] != 1
                    for r in records)))
                    or (not accepted and records)):
                raise ValueError("P2 accepted records differ.")
            node = self.frontier[0]
            self.outcomes[node["id"]] = self.outcomes.get(node["id"], 0) + 1
            self.active = None
            self.root_nodes = dict(meta["after"]["root_nodes"])
            self.discarded += decision["discarded_rows"]
            trace = {**node["tile"], "attempt": meta["reservation"]["attempt"],
                     "status": decision["status"], "iterator_exhausted": exhausted,
                     "query_sha256": meta["query_sha256"]}
            if error:
                self.errors += 1
                self.error_types.add(error)
                trace.update(partial_rows_discarded=count, exception_type=error, retryable=retryable)
            elif accepted:
                trace["returned_loci"] = count
                self.accepted.append(node["tile"])
                self.records.extend(records)
                self.frontier.popleft()
            else:
                trace["returned_before_split"] = count
                if decision["reason"] is None:
                    self.splits += 1
                    self.frontier.popleft()
                    self.frontier.extendleft(reversed(decision["children"]))
            self.trace.append(trace)
            if decision["reason"]:
                self.failed = decision["reason"]
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise QueryCheckpointError("P2 journal decision is contradictory.") from exc


def _run_p2_query(provider, request, progress, event_hook):
    from .query_checkpoint import QueryCheckpointError, _canonical_json_bytes
    provider._validate_request(request)
    profile = provider.proof_profile
    initial = [_canonical_tile(tile, mjd_min=request.mjd_min, mjd_max=request.mjd_max)
               for tile in provider._initial_tiles_fn(request.mjd_min, request.mjd_max)]
    if initial != _make_initial_tiles(request.mjd_min, request.mjd_max):
        raise QueryCheckpointError("P2 requires the frozen canonical P1 initial grid.")
    state = _P2State(request, profile, initial, provider.max_query_attempts)
    events = list(progress.events) if progress is not None else []
    for event in events:
        if len(_canonical_json_bytes(event)) > profile.max_event_bytes:
            raise QueryCheckpointError("P2 event exceeds its qualified byte ceiling.")
        state.apply(event)
    started, t0 = provider.clock(), provider.monotonic()

    def commit(event):
        if len(_canonical_json_bytes(event)) > profile.max_event_bytes:
            raise QueryCheckpointError("P2 event exceeds its qualified byte ceiling.")
        # Apply only after durable commit. A hook observes the complete boundary.
        if progress is not None:
            progress.commit(event)
        state.apply(event)
        events.append(event)
        if event_hook is not None:
            event_hook("p2_committed", {"kind": event["p2_event"]["kind"],
                       "event": json.loads(_canonical_json_bytes(event)), "budget": state.budget()})

    if state.active is not None and not state.failed:
        commit(state.unknown())
    search = None
    while state.frontier and not state.failed:
        reservation, reason = state.reservation()
        if reason:
            commit(state.failure(reason))
            break
        commit(reservation)
        tile = state.frontier[0]["tile"]
        records, exhausted, error, retryable = [], False, None, False
        iterator = None
        try:
            if search is None:
                try:
                    search, _, _ = provider._load_client()
                except Exception as exc:
                    raise ValueError("P2 offline client initialization failed.") from exc
            iterator = iter(search(_build_tile_query(tile)))
            while len(records) < 50:
                try:
                    locus = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                try:
                    record = query.locus_to_record(locus)
                    identity = str(record.get("locus_id") or "").strip()
                    if (not identity or not _record_matches_tile(record, tile)
                            or query.lsst_identifier_counts(pd.DataFrame([record]))["lsst_identifier_count"] != 1):
                        raise ValueError("P2 locus normalization/membership failed.")
                    record["locus_id"] = identity
                    from .query_progress import encode_records
                    if len(_canonical_json_bytes(encode_records([record]))) > profile.max_event_bytes // 50:
                        raise ValueError("P2 normalized record exceeds qualified byte ceiling.")
                except Exception as exc:
                    # All local record failures are malformed input, including
                    # numeric overflow and codec errors. Iterator transport
                    # failures remain outside this nonretryable boundary.
                    raise ValueError("P2 record validation failed.") from exc
                records.append(record)
        except Exception as exc:
            error, retryable = _exception_type(exc), _retryable_query_error(exc)
        finally:
            if iterator is not None:
                try:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
                except Exception as exc:
                    # Lookup and invocation failures are known NON-SCIENCE,
                    # never converted into an unknown crash on restart.
                    error = error or _exception_type(exc)
                    retryable = False
        outcome = state.outcome(len(records), exhausted, exception_type=error,
                                retryable=retryable, records=records if exhausted and error is None else ())
        commit(outcome)
        if error and not state.failed:
            provider.sleeper(provider.retry_delay_seconds * state.outcomes[state.frontier[0]["id"]])

    raw = pd.DataFrame(state.records)
    duplicate_ids = sorted(set(raw.loc[raw["locus_id"].duplicated(keep=False), "locus_id"])) if not raw.empty else []
    frame = raw.drop_duplicates("locus_id", keep="last").reset_index(drop=True) if not raw.empty else raw
    complete = not state.frontier and state.active is None and not state.failed
    if complete and len(state.accepted) != len(initial) + state.splits:
        raise QueryCheckpointError("P2 frontier does not cover the canonical night.")
    contract = provider.scientific_contract(request)
    finished = provider.clock()
    classification = (LiveCompletion.COMPLETE_ZERO if frame.empty else LiveCompletion.COMPLETE_NONZERO) if complete else LiveCompletion.INCOMPLETE
    details = {"completion_classification": classification.value, "target_date_utc": request.date_utc,
        "interval": contract["interval"], "spatial_domain": contract["spatial_domain"],
        "query_sha256": _sha256_json(contract), "query_contract_sha256": _sha256_json(contract),
        "query_tag": None, "lsst_only": True, "lsst_filter": contract["lsst_filter"],
        "lsst_filter_sha256": _sha256_json({"filter": contract["lsst_filter"]}),
        "sort_requested": None, "service_ordering": "ANTARES client/API default",
        "pagination_mode": "antares-client-jsonapi-links-next", "terminal_evidence": "natural-exhaustion-below-50" if complete else state.failed,
        "extraction_method": profile.extraction_method(), "execution_policy": provider.execution_policy(),
        "cache_used": False, "capability_environment": provider.capability.environment,
        "initial_tile_override": False, "initial_tile_count": len(initial),
        "search_request_count": state.starts, "processed_tile_count": len(state.accepted) + state.splits,
        "logical_chunk_count": len(state.accepted), "accepted_chunk_count": len(state.accepted),
        "accepted_tile_count": len(state.accepted), "split_count": state.splits,
        "unresolved_saturated_chunk_count": 0 if complete else 1,
        "unresolved_saturated_tile_count": 0 if complete else 1,
        "iterator_exhausted_accepted_chunks": len(state.accepted), "iterator_exhausted_accepted_tiles": len(state.accepted),
        "all_accepted_iterators_exhausted": complete, "iterator_exhausted": complete,
        "coverage_complete": complete, "coverage_lineage_complete": complete,
        "terminal_pending_tile_count": len(state.frontier), "raw_returned_loci": len(raw),
        "returned_loci": len(frame), "deduplication": {**contract["deduplication"],
            "raw_rows": len(raw), "duplicate_rows_removed": len(raw) - len(frame),
            "duplicate_identity_count": len(duplicate_ids), "duplicate_identities": duplicate_ids,
            "duplicate_identity_sha256": _identifier_hash(duplicate_ids)},
        "partial_rows_discarded": state.discarded, "retry_count": state.errors,
        "retry_exception_types": sorted(state.error_types), "tile_trace": state.trace,
        "tile_trace_sha256": _sha256_json({"tiles": state.trace}),
        "locus_order_sha256": _identifier_hash(frame["locus_id"].tolist() if not frame.empty else []),
        "p2_events": events, "p2_budget": state.budget(),
        "request_started_at_utc": _iso(started), "request_completed_at_utc": _iso(finished),
        "runtime_seconds": round(max(0.0, provider.monotonic() - t0), 6),
        "client": (provider.client_identity() if complete or search is not None
                   else {"distribution": "unavailable"}), "secret_material_recorded": False}
    errors = () if complete else (ProviderIssue(state.failed or "p2_incomplete", ProviderStage.QUERY,
        ProviderOutcome.QUERY_INTERRUPTION, "P2 did not prove complete coverage.", retryable=False, partial=True),)
    return NightQueryResult(request, provider.provider_name, provider.scenario,
        (ProviderOutcome.SUCCESS_ZERO if frame.empty else ProviderOutcome.SUCCESS) if complete else ProviderOutcome.QUERY_INTERRUPTION,
        frame, QueryStageEvidence(complete, not complete, len(frame), errors, details))


# ---------------------------------------------------------------------------
# G6.6.3A guarded P2 transport.
#
# antares-client 1.14.0 follows ``links.next`` inside one ``next()`` with no
# page, byte, time, cycle, scheme or origin bound, and lets requests follow
# redirects anywhere.  Only an explicit P2ProofProfile carrying exact
# P2TransportLimits may search live, and it then pages through this
# provider-owned paginator instead.  Scientific decoding is the client's: the
# same request URL and parameters, its own listing schema, page order, and
# termination on a null or missing ``links.next``.  Every bound is a refusal:
# reaching one leaves the logical iterator incomplete, never complete.  P1
# never reaches this code.
# ---------------------------------------------------------------------------

P2_TRANSPORT_SCHEMA = "v3.p2-transport-limits.v1"
P2_PAGINATION_CONTRACT = "p2-guarded-jsonapi-links-next-v1"
_P2_SORT = "-properties.newest_alert_observation_time"
_P2_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_P2_STREAM_CHUNK_BYTES = 8192
_P2_TRANSPORT_BOUNDS = {
    "max_pages": (1, 10_000),
    "max_consecutive_empty_pages": (0, 1_000),
    "max_page_bytes": (1, 1 << 30),
    "max_iterator_bytes": (1, 1 << 32),
    "iterator_deadline_seconds": (1, 86_400),
    "connect_timeout_seconds": (1, 600),
    "read_timeout_seconds": (1, 600),
    "max_redirects": (0, 10),
}


class P2TransportGuardError(ValueError):
    """A finite transport bound or continuation rule refused one iterator.

    The iterator is incomplete and nothing it yielded may become science.
    ``ValueError`` keeps the unchanged shared classifier terminal (no retry).
    Messages never carry response bodies, headers or URLs.
    """


class P2PageLimitError(P2TransportGuardError):
    """The iterator needed more HTTP pages than its profile allows."""


class P2EmptyPageLimitError(P2TransportGuardError):
    """Too many consecutive empty pages still carried a continuation."""


class P2PageBytesError(P2TransportGuardError):
    """One response body exceeded the per-page byte ceiling."""


class P2IteratorBytesError(P2TransportGuardError):
    """Cumulative response bodies exceeded the per-iterator byte ceiling."""


class P2DeadlineError(P2TransportGuardError):
    """The whole-iterator wall-clock deadline elapsed."""


class P2MalformedContinuationError(P2TransportGuardError):
    """``links``/``links.next`` is not one plain absolute URL string."""


class P2RelativeContinuationError(P2TransportGuardError):
    """``links.next`` is relative; the pinned client cannot follow it either."""


class P2InsecureContinuationError(P2TransportGuardError):
    """``links.next`` is not HTTPS."""


class P2CrossOriginContinuationError(P2TransportGuardError):
    """``links.next`` leaves the exact ANTARES API origin."""


class P2ContinuationPathError(P2TransportGuardError):
    """``links.next`` names a resource other than the listing being paged."""


class P2ContinuationCycleError(P2TransportGuardError):
    """``links.next`` repeats a page this iterator already requested."""


class P2UnsafeRedirectError(P2TransportGuardError):
    """A redirect lacks a same-origin HTTPS target inside the API prefix."""


class P2RedirectLimitError(P2TransportGuardError):
    """One page exceeded its redirect ceiling."""


class P2UnexpectedStatusError(P2TransportGuardError):
    """A non-200, non-redirect status below 400."""


class P2TransportHTTPError(RuntimeError):
    """HTTP status >= 400, transient like the client's ``AntaresException``.

    Retried only within the bounded per-node attempts; the body is not read.
    """


_P2_TRANSPORT_GUARD_TYPES = frozenset(
    f"{guard.__module__}.{guard.__name__}"
    for guard in (
        P2TransportGuardError, P2PageLimitError, P2EmptyPageLimitError, P2PageBytesError,
        P2IteratorBytesError, P2DeadlineError, P2MalformedContinuationError,
        P2RelativeContinuationError, P2InsecureContinuationError,
        P2CrossOriginContinuationError, P2ContinuationPathError, P2ContinuationCycleError,
        P2UnsafeRedirectError, P2RedirectLimitError, P2UnexpectedStatusError,
    )
)


@dataclass(frozen=True)
class P2TransportLimits:
    """Finite HTTP bounds for one logical P2 search iterator.

    Each iterator belongs to one durable search reservation and a night's
    reservations are finite, so these bounds also bound total live search
    traffic; restarting can never renew them.
    """

    max_pages: int
    max_consecutive_empty_pages: int
    max_page_bytes: int
    max_iterator_bytes: int
    iterator_deadline_seconds: int
    connect_timeout_seconds: int
    read_timeout_seconds: int
    max_redirects: int

    def __post_init__(self):
        for name, (minimum, maximum) in _P2_TRANSPORT_BOUNDS.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")

    def as_dict(self):
        return {"schema_version": P2_TRANSPORT_SCHEMA, "pagination": P2_PAGINATION_CONTRACT,
                "continuation": "absolute-https-same-origin-same-listing-path-never-repeated",
                "redirects": "same-origin-https-api-prefix-validated-before-request",
                "termination": "complete-page-then-links-next-null-or-missing",
                "limit_semantics": "refusal-never-completeness",
                **{name: getattr(self, name) for name in _P2_TRANSPORT_BOUNDS}}


def _p2_plain_url(value):
    """One printable-ASCII URL: no whitespace, control character or backslash.

    URL parsers disagree on these (``urlsplit`` drops tabs and keeps ``\\`` in
    the authority; urllib3 splits on ``\\``), so they are refused before any
    origin decision rather than interpreted.
    """
    return (isinstance(value, str) and bool(value) and value.isascii() and "\\" not in value
            and not any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value))


class _GuardedListing:
    """Provider-owned, finitely bounded replacement for the client's pagination.

    Each call returns an independent lazy iterator: like the client, the next
    page is requested only when the consumer asks past the previous page.
    Each HTTP page uses a fresh session, as ``requests.get`` does, so no
    cookie, connection or session state is shared between pages, iterators or
    concurrently acquired nights.  Nothing global is patched.
    """

    def __init__(self, limits, base_url, session_factory, schema_factory):
        if type(limits) is not P2TransportLimits:
            raise ValueError("Guarded transport requires exact P2TransportLimits.")
        if not callable(session_factory) or not callable(schema_factory):
            raise ValueError("Guarded transport requires session and schema factories.")
        base = _validated_base_url(base_url)
        self.limits = limits
        self.listing_url = urljoin(base, "loci")  # exactly the client's search URL
        listing = urlsplit(self.listing_url)
        self._host, self._listing_path = listing.hostname, listing.path
        self._service_prefix = urlsplit(base).path
        self._session_factory = session_factory
        self._schema_factory = schema_factory

    def search(self, query):
        return self._pages({"sort": _P2_SORT,
                            "elasticsearch_query[locus_listing]": json.dumps(query)})

    def _pages(self, params):
        import requests

        limits, url = self.limits, self.listing_url
        deadline = time.monotonic() + limits.iterator_deadline_seconds
        first = urlsplit(requests.Request("GET", url, params=params).prepare().url)
        requested = {(first.path, first.query)}
        pages = received = empty_run = 0
        while True:
            if pages >= limits.max_pages:
                raise P2PageLimitError("P2 iterator reached its HTTP page ceiling.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise P2DeadlineError("P2 iterator deadline elapsed.")
            response = self._request(url, params, limits.max_iterator_bytes - received, remaining)
            pages += 1
            received += len(response.content)
            payload = response.json()  # the client's own decoder and error type
            items = self._schema_factory().load(payload)
            yield from items
            target = self._continuation(payload)
            if target is None:
                return  # natural exhaustion: a complete page without links.next
            empty_run = 0 if items else empty_run + 1
            if empty_run > limits.max_consecutive_empty_pages:
                raise P2EmptyPageLimitError("P2 iterator received too many empty continuing pages.")
            key = self._validated_continuation(target)
            if key in requested:
                raise P2ContinuationCycleError("P2 links.next repeats an already requested page.")
            requested.add(key)
            url, params = target, None

    @staticmethod
    def _continuation(payload):
        # The client reads payload.get("links", {}).get("next"): absent or
        # null ends iteration.  A links member that is not an object refuses.
        links = payload.get("links", {}) if isinstance(payload, dict) else None
        if not isinstance(links, dict):
            raise P2MalformedContinuationError("P2 JSON:API links member is not an object.")
        return links.get("next")

    def _validated_continuation(self, target):
        if not _p2_plain_url(target):
            raise P2MalformedContinuationError("P2 links.next is not one plain URL string.")
        try:
            parts = urlsplit(target)
            port, host = parts.port, parts.hostname
        except ValueError as exc:
            raise P2MalformedContinuationError("P2 links.next is not a parseable URL.") from exc
        if not parts.scheme:
            raise P2RelativeContinuationError("P2 links.next is relative; it is never resolved.")
        if parts.scheme != "https":
            raise P2InsecureContinuationError("P2 links.next is not HTTPS.")
        if parts.username is not None or parts.password is not None or parts.fragment:
            raise P2MalformedContinuationError("P2 links.next carries credentials or a fragment.")
        if host != self._host or port not in (None, 443):
            raise P2CrossOriginContinuationError("P2 links.next leaves the ANTARES API origin.")
        if parts.path != self._listing_path:
            raise P2ContinuationPathError("P2 links.next is not the listing being paged.")
        return parts.path, parts.query

    def _validated_redirect(self, response, requested):
        location = response.headers.get("Location")
        if not _p2_plain_url(location):
            raise P2UnsafeRedirectError("P2 redirect lacks one plain Location.")
        base = response.url if isinstance(response.url, str) and response.url else requested
        target = urljoin(base, location)
        try:
            parts = urlsplit(target)
            port, host = parts.port, parts.hostname
        except ValueError as exc:
            raise P2UnsafeRedirectError("P2 redirect target is not parseable.") from exc
        if (parts.scheme != "https" or host != self._host or port not in (None, 443)
                or parts.username is not None or parts.password is not None or parts.fragment
                or not parts.path.startswith(self._service_prefix)):
            raise P2UnsafeRedirectError("P2 redirect leaves the ANTARES API boundary.")
        return target

    def _request(self, url, params, iterator_budget, remaining):
        outcome, cancelled = {}, threading.Event()

        def fetch():
            try:
                outcome["response"] = self._fetch_page(url, params, iterator_budget, cancelled)
            except BaseException as exc:  # re-raised below in the consuming thread
                outcome["error"] = exc

        worker = threading.Thread(target=fetch, name="antares-p2-page", daemon=True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            # The abandoned page stops at its next chunk or socket timeout and
            # can never deliver bytes to this (now failed) iterator.
            cancelled.set()
            raise P2DeadlineError("P2 iterator deadline elapsed during an HTTP page.")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["response"]

    def _send(self, session, url, params):
        """One HTTP exchange through the session's own transport adapter.

        ``Session.send`` pre-computes ``Response.next`` even with
        ``allow_redirects=False``, which reads a redirect response's whole body
        without any ceiling.  The request and environment settings here are
        exactly those ``Session.request`` derives for ``requests.get``; only
        that redirect bookkeeping is skipped, so every body this page reads
        passes through the bounded reader in :meth:`_fetch_page`.
        """
        import requests

        limits = self.limits
        prepared = session.prepare_request(requests.Request("GET", url, params=params or {}))
        settings = session.merge_environment_settings(prepared.url, {}, True, None, None)
        return session.get_adapter(prepared.url).send(
            prepared, stream=True, verify=settings["verify"], cert=settings["cert"],
            proxies=settings["proxies"],
            timeout=(limits.connect_timeout_seconds, limits.read_timeout_seconds))

    def _fetch_page(self, url, params, iterator_budget, cancelled):
        limits = self.limits
        budget = min(limits.max_page_bytes, iterator_budget)
        overflow = P2PageBytesError if limits.max_page_bytes <= iterator_budget else P2IteratorBytesError
        session = self._session_factory()
        try:
            for _hop in range(limits.max_redirects + 1):
                response = self._send(session, url, params)
                try:
                    status = int(response.status_code)
                    if status in _P2_REDIRECT_STATUSES:
                        url, params = self._validated_redirect(response, url), None
                        continue
                    if status >= 400:
                        raise P2TransportHTTPError(f"ANTARES returned HTTP {status}.")
                    if status != 200:
                        raise P2UnexpectedStatusError(f"ANTARES returned unexpected HTTP {status}.")
                    declared = response.headers.get("Content-Length")
                    if (isinstance(declared, str) and declared.isascii() and declared.isdigit()
                            and int(declared) > budget):
                        raise overflow("P2 response declares more bytes than its ceiling.")
                    body = bytearray()
                    for chunk in response.iter_content(_P2_STREAM_CHUNK_BYTES):
                        if cancelled.is_set():
                            raise P2DeadlineError("P2 page was abandoned at its deadline.")
                        body += chunk
                        if len(body) > budget:
                            raise overflow("P2 response exceeded its byte ceiling.")
                    # Exactly how requests caches a consumed body, so the
                    # client's Response.json() decoding then applies unchanged.
                    response._content = bytes(body)
                    return response
                finally:
                    response.close()
            raise P2RedirectLimitError("P2 page exceeded its redirect ceiling.")
        finally:
            session.close()


def _p2_client_identity(transport, base_url):
    return {"distribution": "antares-client", "version": PINNED_CLIENT_VERSION,
            "api_base_url": base_url, "api_timeout_seconds": CLIENT_TIMEOUT_SECONDS,
            "authentication": "public-search-no-credentials",
            "pagination_contract": P2_PAGINATION_CONTRACT,
            "transport_sha256": _sha256_json(transport.as_dict())}


def _load_p2_client(provider):
    """P2 service callables: mocked offline, or the guarded paginator on Arnor."""
    profile = provider.proof_profile
    callbacks = (provider._search_fn, provider._get_by_id_fn, provider._connectivity_fn)
    environment = getattr(provider.capability, "environment", None)
    if environment == "local-mock" and all(callable(callback) for callback in callbacks):
        return callbacks
    if (profile.transport is None or environment != "arnor-commissioning"
            or type(provider.capability) is not LiveAntaresReadCapability
            or any(callback is not None for callback in callbacks)):
        raise LiveCapabilityError(
            "P2 live search requires a guarded-transport profile and sealed Arnor LIVE_ANTARES_READ.")
    try:
        import requests
        from antares_client._api.schemas import _LocusListingSchema
        from antares_client.config import config
        from antares_client.search import get_available_tags, get_by_id
    except ImportError as exc:
        raise RuntimeError("The pinned ANTARES client is unavailable.") from exc
    version = metadata.version("antares-client")
    if version != PINNED_CLIENT_VERSION:
        raise RuntimeError(f"Expected antares-client {PINNED_CLIENT_VERSION}; found {version}.")
    base_url = _validated_base_url(str(config.get("ANTARES_API_BASE_URL", "")))
    timeout = int(config.get("API_TIMEOUT", CLIENT_TIMEOUT_SECONDS))
    if timeout != CLIENT_TIMEOUT_SECONDS:
        raise RuntimeError(
            f"Phase 6 requires the pinned {CLIENT_TIMEOUT_SECONDS}-second API timeout.")
    listing = _GuardedListing(profile.transport, base_url, requests.Session,
                              lambda: _LocusListingSchema(many=True, partial=True))
    provider._client_identity_cache = _p2_client_identity(profile.transport, base_url)
    return listing.search, get_by_id, get_available_tags


# SHA-256 of the canonical ``as_dict()`` JSON of the frozen canary profile and
# of its transport limits.  The transport digest is also the client identity's
# ``transport_sha256``; the profile digest is every P2 event's ``profile_sha256``.
G663_CANARY_P2_PROFILE_SHA256 = "d1dfee3b066e2a5f90f1b4842d1184c9e6bd32d4a1b772d3819f082b4b189684"
G663_CANARY_TRANSPORT_SHA256 = "41ad48a1c82a585498ce7838672962dfc5583bc91b4e49c497d9a62240dd0650"


def g663_canary_p2_profile() -> P2ProofProfile:
    """The one frozen G6.6.3A Jul07/Jul13 canary profile.

    Explicit opt-in only: no default path selects it and no source or
    artifact registry trusts it.  Any change is a new Control qualification
    identity, so the profile refuses to exist unless its canonical bytes still
    hash to ``G663_CANARY_P2_PROFILE_SHA256``.
    """
    profile = P2ProofProfile(
        max_depth=18, max_nodes_per_root=511, max_nodes_per_night=4095,
        max_search_attempts=250_000, crash_reserve=4, max_event_bytes=33_554_432,
        transport=P2TransportLimits(
            max_pages=64, max_consecutive_empty_pages=2, max_page_bytes=16_777_216,
            max_iterator_bytes=67_108_864, iterator_deadline_seconds=600,
            connect_timeout_seconds=60, read_timeout_seconds=60, max_redirects=2))
    if (_sha256_json(profile.as_dict()) != G663_CANARY_P2_PROFILE_SHA256
            or _sha256_json(profile.transport.as_dict()) != G663_CANARY_TRANSPORT_SHA256):
        raise RuntimeError("The frozen G6.6.3A canary profile identity changed.")
    return profile
