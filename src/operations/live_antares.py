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
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from .. import query
from ..history import prepare_alerts, prepare_loci, validation_summary
from .science import (
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
        if target_date_utc != PHASE6_TARGET_DATE_UTC:
            raise LiveCapabilityError(
                f"Arnor Phase 6 authority is restricted to {PHASE6_TARGET_DATE_UTC}."
            )
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

    def _load_client(self) -> Tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
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
        return {
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

    def scientific_contract(self, request: NightScienceRequest) -> Mapping[str, Any]:
        """Return the exact, side-effect-free scientific request contract."""
        self._validate_request(request)
        return _scientific_query_contract(request)

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
            self.capability.target_date_utc != PHASE6_TARGET_DATE_UTC
            or request.date_utc != PHASE6_TARGET_DATE_UTC
            or request.date_utc != self.capability.target_date_utc
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
            request.mjd_min != PHASE6_MJD_MIN
            or request.mjd_max != PHASE6_MJD_MAX
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

    def query(self, request: NightScienceRequest) -> NightQueryResult:
        """Run the accepted probe-first time/RA/Dec extractor to exhaustion.

        Every tile uses a half-open time interval.  A 50-row probe is
        provisional saturation: those rows are discarded and the tile is
        split along its largest normalized dimension.  Only tiles returning
        fewer than 50 rows *and* reaching normal iterator exhaustion are
        accepted.  Final accepted rows are deduplicated by ``locus_id`` with
        ``keep='last'``, exactly as in the accepted historical path.
        """
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
            for attempt in range(1, self.max_query_attempts + 1):
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
                    retryable = not isinstance(exc, (TypeError, ValueError))
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
        fetch_evidence = FetchStageEvidence(
            True,
            False,
            len(loci),
            len(alerts),
            (),
            details,
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
