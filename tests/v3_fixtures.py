"""Shared synthetic fixtures for V3 publication and backfill tests.

Everything here lives below the OS temporary directory.  No fixture touches
production data, contacts ANTARES, or populates a live cache.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from astropy.time import Time

from src import history
from src.operations.publication import (
    CANDIDATE_RECORD_SCHEMA,
    NightPublisher,
    authorize_publication,
    load_backfill_candidate,
)
from src.operations.science import (
    NightScienceRequest,
    SyntheticScienceProvider,
    build_night_artifacts,
)
from src.operations.storage import SyntheticWriteCapability
from src.operations.writer import nightly_target_relative


CANDIDATE_RELEASE = "a" * 40
PUBLISHER_RELEASE = "b" * 40
FIXED_NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=timezone.utc)


def fixed_clock() -> datetime:
    return FIXED_NOW


def mountinfo_for(path: Path):
    mount_point = Path(path).resolve()
    device = mount_point.stat().st_dev
    encoded = str(mount_point).replace(" ", r"\040")
    return [
        f"101 1 {os.major(device)}:{os.minor(device)} / {encoded} rw - "
        f"nfs4 fixture:/production rw"
    ]


def mjd_for(date_utc: str) -> float:
    return float(Time(f"{date_utc} 00:00:00", scale="utc").mjd)


def night_request(date_utc: str) -> NightScienceRequest:
    mjd_min = mjd_for(date_utc)
    return NightScienceRequest(date_utc, mjd_min, mjd_min + 1.0)


def synthetic_artifacts(date_utc: str):
    provider = SyntheticScienceProvider()
    return build_night_artifacts(provider.fetch_night(night_request(date_utc)))


def make_capability(parent: Path, run_id: str = "v3-production") -> SyntheticWriteCapability:
    root = Path(parent) / run_id
    root.mkdir(mode=0o700)
    for name in ("published", "staging", "control", "evidence"):
        (root / name).mkdir(mode=0o700)
    return SyntheticWriteCapability.for_local_run_root(root, run_id)


def write_night(data_root: Path, date_utc: str, artifacts) -> Path:
    target = Path(data_root) / nightly_target_relative(date_utc)
    target.mkdir(parents=True, mode=0o700)
    for name, payload in artifacts.items():
        (target / name).write_bytes(payload)
    return target


def seed_production(capability: SyntheticWriteCapability, dates) -> None:
    for date_utc in dates:
        write_night(capability.published_root, date_utc, synthetic_artifacts(date_utc))
    history.update_cumulative_indexes(data_root=capability.published_root)


def manifest_bytes(manifest) -> bytes:
    return (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def make_backfill_candidate(
    parent: Path,
    date_utc: str,
    *,
    constructed_at_utc: str = None,
    june27_style_defect: bool = False,
    release_sha: str = CANDIDATE_RELEASE,
):
    night_root = Path(parent) / "candidates" / f"night-{date_utc}"
    candidate_dir = night_root / "candidate"
    candidate_dir.mkdir(parents=True, mode=0o700)
    artifacts = dict(synthetic_artifacts(date_utc))
    if june27_style_defect:
        manifest = json.loads(artifacts["manifest.json"])
        # Mirror the accepted June 27 contradiction: acquisition started months
        # after the placeholder night-start "finished" time.
        manifest["started_at_utc"] = "2026-09-03T20:13:33+00:00"
        manifest["finished_at_utc"] = f"{date_utc}T00:00:00+00:00"
        artifacts["manifest.json"] = manifest_bytes(manifest)
    for name, payload in artifacts.items():
        (candidate_dir / name).write_bytes(payload)
    manifest = json.loads(artifacts["manifest.json"])
    record = {
        "schema_version": CANDIDATE_RECORD_SCHEMA,
        "date_utc": date_utc,
        "release_sha": release_sha,
        "artifacts": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in artifacts.items()
        },
        "loci": manifest["actual_loci"],
        "alerts": manifest["alert_rows"],
        "validation_passed": True,
        "authoritative": False,
        "constructed_at_utc": constructed_at_utc or "2026-09-20T10:00:00+00:00",
        "provenance": {"query_identity": "q" * 4, "fetch_identity": "f" * 4},
    }
    (candidate_dir / "candidate-record.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    return load_backfill_candidate(night_root)


def publisher_for(capability, *, fault_hook=None) -> NightPublisher:
    return NightPublisher(
        capability,
        publisher_release_sha=PUBLISHER_RELEASE,
        cache_root=capability.root / "absent-cache",
        mountinfo_lines=mountinfo_for(capability.published_root),
        clock=fixed_clock,
        fault_hook=fault_hook,
    )


AUTHORIZED_AT = "2026-09-29T00:00:00+00:00"
EXPIRES_AT = "2027-09-29T00:00:00+00:00"


def authorize(
    publisher: NightPublisher,
    candidate,
    *,
    predecessor=None,
    production=None,
    expected=None,
    nonce=None,
    authorized_at=AUTHORIZED_AT,
    expires_at=EXPIRES_AT,
):
    """Authorize exactly what Control would: the observed Sentinel binding and plan."""
    from datetime import date, timedelta

    inputs = publisher.authorization_inputs(candidate)
    predecessor = predecessor or (
        date.fromisoformat(candidate.date_utc) - timedelta(days=1)
    ).isoformat()
    return authorize_publication(
        candidate,
        production=production or inputs["production"],
        predecessor_date_utc=predecessor,
        publisher_release_sha=PUBLISHER_RELEASE,
        expected_cumulative_sha256=expected or inputs["expected_cumulative_sha256"],
        authorized_by="control-fixture",
        authorized_at_utc=authorized_at,
        expires_at_utc=expires_at,
        nonce=nonce or hashlib.sha256(
            f"{candidate.record_sha256}:{authorized_at}".encode()
        ).hexdigest()[:32],
    )


# ---------------------------------------------------------------------------
# Backfill adapter: drives the real query-seal and segmented-fetch primitives
# ---------------------------------------------------------------------------

import threading
import time
from collections import Counter

import pandas as pd

from src.operations.fetch_checkpoint import FetchObjectResult
from src.operations.live_antares import LIVE_ANTARES_READ, LiveAntaresReadCapability
from src.operations.query_checkpoint import _canonical_json_sha256, _identifier_hash
from src.operations.science import (
    FetchStageEvidence,
    NightQueryResult,
    NightScienceResult,
    ProviderOutcome,
    QueryStageEvidence,
    _result_evidence,
)


def mock_read_capability(run_root, run_id, date_utc, release_sha):
    return LiveAntaresReadCapability.for_local_mock(
        run_root,
        run_id=run_id,
        target_date_utc=date_utc,
        release_sha=release_sha,
        authority=LIVE_ANTARES_READ,
    )


class SyntheticBackfillAdapter:
    """Deterministic N-locus nights with fault injection and call accounting.

    Acquisition presents the only checkpointed live identity
    (``live-antares``/``commissioning-v1``) so the real query-seal and
    segmented-fetch primitives run unmodified; construction emits synthetic
    science because the Phase 6 manifest contract is specific to June 27.
    """

    provider_name = "live-antares"
    scenario = "commissioning-v1"

    def __init__(self, loci_per_night=10, *, delays=None):
        self.loci_per_night = loci_per_night
        self.delays = dict(delays or {})
        self.queries = Counter()
        self.fetched_segments = Counter()
        self.fetched_loci = Counter()
        self.constructions = Counter()
        self.fail_fetch_at = {}       # date -> segment ordinal (0-based) to fail once
        self.fail_construct = set()   # dates whose next construct raises OSError once
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self._segment_calls = Counter()

    def _enter(self, date_utc):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.delays.get(date_utc, 0.0))

    def _exit(self):
        with self._lock:
            self.in_flight -= 1

    def execution_policy(self):
        return {"adapter": "synthetic-backfill-v1", "segment_workers": 1}

    def scientific_contract(self, request):
        return {
            "target_date_utc": request.date_utc,
            "query_tag": None,
            "lsst_only": True,
            "interval": {"mjd_min": request.mjd_min, "mjd_max": request.mjd_max},
        }

    def acquisition_request(self, date_utc):
        mjd_min = mjd_for(date_utc)
        return NightScienceRequest(
            date_utc, mjd_min, mjd_min + 1.0, target_loci=None,
            range_label=f"Synthetic backfill {date_utc}",
        )

    def _loci(self, request):
        n = self.loci_per_night
        ids = ["ANT-SHARED-0001"] + [
            f"ANT-{request.date_utc}-{index:04d}" for index in range(1, n)
        ]
        return pd.DataFrame(
            {
                "locus_id": ids,
                "ra": [10.0 + index for index in range(n)],
                "dec": [-20.0 + index for index in range(n)],
                "newest_alert_observation_time": [
                    request.mjd_min + (index + 0.5) / (n + 1) for index in range(n)
                ],
                "tags": ["lsst"] * n,
                "dia_object_id": [f"DIA-{index}" for index in range(n)],
                "ss_object_id": [""] * n,
                "ztf_object_id": [""] * n,
                "brightest_alert_magnitude": [20.0 + index / 10 for index in range(n)],
                "num_mag_values": [2] * n,
            }
        )

    def query(self, request):
        self._enter(request.date_utc)
        try:
            with self._lock:
                self.queries[request.date_utc] += 1
            loci = self._loci(request)
            details = {
                "target_date_utc": request.date_utc,
                "returned_loci": len(loci),
                "locus_order_sha256": _identifier_hash(loci["locus_id"].tolist()),
                "query_contract_sha256": _canonical_json_sha256(self.scientific_contract(request)),
                "execution_policy": self.execution_policy(),
                "request_started_at_utc": f"{request.date_utc}T06:00:00+00:00",
                "request_completed_at_utc": f"{request.date_utc}T06:05:00+00:00",
            }
            return NightQueryResult(
                request, self.provider_name, self.scenario, ProviderOutcome.SUCCESS,
                loci, QueryStageEvidence(True, False, len(loci), (), details),
            )
        finally:
            self._exit()

    def fetch_segment(self, request, locus_ids):
        self._enter(request.date_utc)
        try:
            date_utc = request.date_utc
            with self._lock:
                ordinal = self._segment_calls[date_utc]
                self._segment_calls[date_utc] += 1
                if self.fail_fetch_at.get(date_utc) == ordinal:
                    del self.fail_fetch_at[date_utc]
                    raise ConnectionError("injected transient fetch failure")
                self.fetched_segments[date_utc] += 1
                self.fetched_loci[date_utc] += len(locus_ids)
            results = []
            for locus_id in locus_ids:
                frame = pd.DataFrame(
                    {
                        "locus_id": [locus_id, locus_id],
                        "alert_id": [f"{locus_id}-A1", f"{locus_id}-A2"],
                        "mjd": [request.mjd_min + 0.1, request.mjd_min + 0.2],
                        "ztf_magpsf": [20.0, 20.1],
                        "ztf_sigmapsf": [0.05, 0.06],
                        "ztf_fid": [1, 2],
                        "lsst_band": ["g", "r"],
                    }
                )
                results.append(FetchObjectResult(locus_id, frame))
            return results
        finally:
            self._exit()

    def construct(self, request, query_result, alerts, completion):
        from src.history import prepare_alerts, prepare_loci, validation_summary

        with self._lock:
            self.constructions[request.date_utc] += 1
            if request.date_utc in self.fail_construct:
                self.fail_construct.discard(request.date_utc)
                raise OSError(errno.EIO, "injected transient construction I/O failure")
        details = query_result.evidence.details
        loci = prepare_loci(
            query_result.loci, request.date_utc, request.mjd_min, request.mjd_max,
            details["request_completed_at_utc"],
        )
        prepared = prepare_alerts(alerts, request.date_utc, request.range_label)
        fetch_evidence = FetchStageEvidence(
            True, False, len(loci), len(prepared), (),
            {
                "request_completed_at_utc": f"{request.date_utc}T07:00:00+00:00",
                "segments": completion.segment_count,
            },
        )
        evidence = _result_evidence(query_result.evidence, fetch_evidence)
        validation = validation_summary(
            loci, prepared, mjd_min=request.mjd_min, mjd_max=request.mjd_max,
            prior_locus_ids=request.prior_locus_ids, lsst_only=True,
            query_completed=True, query_fetch_clean=evidence.clean,
        )
        return NightScienceResult(
            request, "synthetic", "success_nonzero", ProviderOutcome.SUCCESS,
            query_result, loci, prepared, fetch_evidence, validation, (), evidence,
        )


def fixture_prior_free_attestations():
    """Attest only this fixture adapter, whose query/fetch ignore prior loci."""
    from src.operations.backfill import acquisition_identity

    identity = acquisition_identity(SyntheticBackfillAdapter())
    return (
        {
            **{
                key: identity[key]
                for key in (
                    "provider_name", "scenario", "provider_module",
                    "provider_implementation_sha256", "adapter",
                    "adapter_implementation_sha256",
                )
            },
            "evidence": "synthetic fixture: query/fetch never read prior_locus_ids",
        },
    )


def make_segment_cache(root):
    """Build a cache for functional tests wherever TMPDIR lives.

    On Arnor TMPDIR sits inside the protected ANTARES tree, where SegmentCache
    correctly refuses every root except the approved one.  Placement rules are
    tested explicitly elsewhere; functional tests relax only placement here.
    """
    from src.operations import cache

    saved = cache.MIDDLE_EARTH_PROJECT_ROOT
    cache.MIDDLE_EARTH_PROJECT_ROOT = Path("/nonexistent-antares-project-root")
    try:
        return cache.SegmentCache(root, forbidden_roots=())
    finally:
        cache.MIDDLE_EARTH_PROJECT_ROOT = saved
