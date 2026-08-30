import contextlib
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

import pandas as pd

from src import cli, history, query as query_module
from src.operations.commissioning import (
    TARGET_DATE_UTC,
    capture_production_sentinel,
    compare_production_sentinels,
    establish_target_eligibility,
    qualify_live_night,
)
from src.operations.live_antares import (
    LIVE_ANTARES_READ,
    LiveAntaresProvider,
    LiveAntaresReadCapability,
    LiveCapabilityError,
    LiveCompletion,
    _build_tile_query,
    _make_initial_tiles,
    _split_tile,
    _validated_base_url,
    extraction_method_contract,
)
from src.operations.locking import LockUnavailable, WriterLock
from src.operations.science import (
    ArtifactValidationError,
    NightScienceRequest,
    ProviderIssue,
    ProviderOutcome,
    ProviderStage,
    SyntheticScienceProvider,
    build_night_artifacts,
    reopen_and_validate_artifacts,
)
from src.operations.storage import StorageLayout, SyntheticWriteCapability
from src.operations.writer import (
    SHARED_RECONCILIATION_LOCK_IDENTITY,
    NightExecutionSpec,
    execute_synthetic_night,
)


RELEASE_SHA = "6" * 40
FULL_TILE = {
    "mjd_min": 61218.0,
    "mjd_max": 61219.0,
    "ra_min": 0.0,
    "ra_max": 360.0,
    "dec_min": -90.0,
    "dec_max": 90.0,
}


class FakeLocus:
    def __init__(
        self,
        locus_id="ANT-LIVE-0001",
        mjd=61218.5,
        lightcurve=None,
        ra=12.5,
        dec=-2.5,
        survey=None,
    ):
        self.locus_id = locus_id
        self.ra = ra
        self.dec = dec
        self.tags = ["lsst"]
        self.properties = {
            "newest_alert_observation_time": mjd,
            "survey": (
                survey
                if survey is not None
                else {"lsst": {"dia_object_id": f"DIA-{locus_id}"}}
            ),
            "brightest_alert_magnitude": 20.0,
            "num_mag_values": 1,
        }
        self.lightcurve = lightcurve


def _lightcurve(value=20.0):
    return pd.DataFrame(
        {
            "mjd": [61218.25],
            "ztf_magpsf": [value],
            "ztf_sigmapsf": [0.1],
            "ztf_fid": [1],
        }
    )


def _mixed_identifier_provider(root):
    loci = [
        FakeLocus(
            "ANT-DIA",
            lightcurve=_lightcurve(20.1),
            survey={"lsst": {"dia_object_id": "DIA-1"}},
        ),
        FakeLocus(
            "ANT-SS",
            lightcurve=_lightcurve(20.2),
            survey={"lsst": {"ss_object_id": "SS-1"}},
        ),
        FakeLocus(
            "ANT-DIA-SS",
            lightcurve=_lightcurve(20.3),
            survey={
                "lsst": {
                    "dia_object_id": "DIA-2",
                    "ss_object_id": "SS-2",
                },
                "ztf": {"id": "ZTF-1"},
            },
        ),
    ]
    by_id = {locus.locus_id: locus for locus in loci}
    return _mock_provider(
        root,
        _canonical_search(loci),
        by_id.__getitem__,
        canonical_tiles=True,
    )


def _request():
    return NightScienceRequest(
        TARGET_DATE_UTC,
        61218.0,
        61219.0,
        target_loci=None,
        range_label="ANTARES commissioning 2026-06-27",
    )


def _body_matches_locus(body, locus):
    filters = body["query"]["bool"]["filter"]
    time_range = filters[0]["range"]["properties.newest_alert_observation_time"]
    ra_range = filters[1]["range"]["ra"]
    dec_range = filters[2]["range"]["dec"]
    mjd = float(locus.properties["newest_alert_observation_time"])
    dec_upper = dec_range.get("lt", dec_range.get("lte"))
    dec_ok = (
        float(dec_range["gte"]) <= float(locus.dec) <= float(dec_upper)
        if "lte" in dec_range
        else float(dec_range["gte"]) <= float(locus.dec) < float(dec_upper)
    )
    return bool(
        float(time_range["gte"]) <= mjd < float(time_range["lt"])
        and float(ra_range["gte"]) <= float(locus.ra) < float(ra_range["lt"])
        and dec_ok
    )


def _canonical_search(loci):
    return lambda body: [locus for locus in loci if _body_matches_locus(body, locus)]


def _mock_provider(
    root,
    search_fn,
    get_by_id_fn,
    connectivity_fn=lambda: ["tag"],
    *,
    canonical_tiles=False,
):
    capability = LiveAntaresReadCapability.for_local_mock(
        root,
        run_id=root.name,
        target_date_utc=TARGET_DATE_UTC,
        release_sha=RELEASE_SHA,
        authority=LIVE_ANTARES_READ,
    )
    return LiveAntaresProvider(
        capability,
        search_fn=search_fn,
        get_by_id_fn=get_by_id_fn,
        connectivity_fn=connectivity_fn,
        initial_tiles_fn=(None if canonical_tiles else lambda _lo, _hi: [FULL_TILE]),
        retry_delay_seconds=0,
        max_query_attempts=2,
        max_fetch_attempts=2,
        max_fetch_workers=4,
    )


def _accepted_fixture(root):
    data_root = root / "production"
    nightly_root = history.survey_data_root(data_root) / "nightly"
    start = date(2026, 2, 25)
    dates = [(start + timedelta(days=index)).isoformat() for index in range(89)]
    dates.append("2026-06-26")
    zero_dates = {"2026-03-05", "2026-03-11"}
    nonzero_dates = [value for value in dates if value not in zero_dates]
    for value in dates:
        paths = history.nightly_paths(data_root, value)
        paths["dir"].mkdir(parents=True)
        if value == "2026-06-26":
            mjd_min, mjd_max = 61217.0, 61218.0
        else:
            offset = (date.fromisoformat(value) - start).days
            mjd_min, mjd_max = 61096.0 + offset, 61097.0 + offset
        actual_loci = 0 if value in zero_dates else 1
        alert_rows = 0 if value in zero_dates else 1
        if value == nonzero_dates[0]:
            actual_loci = 993218 - (len(nonzero_dates) - 1)
            alert_rows = 13579707 - (len(nonzero_dates) - 1)
        manifest = {
            "date_utc": value,
            "mjd_min": mjd_min,
            "mjd_max": mjd_max,
            "query_tag": None,
            "target_loci": None,
            "lsst_filter_used": True,
            "lsst_filter": query_module.lsst_identifier_filter(),
            "parallel_shards": 1,
            "actual_loci": actual_loci,
            "alert_rows": alert_rows,
            "chunk_count": 1,
            "split_count": 0,
            "saturated_chunk_count": 0,
            "extraction_method": extraction_method_contract(),
            "status": "complete",
            "validation": {"append_ready": True},
        }
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        paths["loci"].write_bytes(b"fixture-loci")
        paths["alerts"].write_bytes(b"fixture-alerts")
    cumulative = history.cumulative_paths(data_root)
    cumulative["dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date_utc": dates}).to_parquet(
        cumulative["nightly_summary"], index=False
    )
    pd.DataFrame({"locus_id": []}).to_parquet(cumulative["loci_index"], index=False)
    return data_root, root / "cache"


def _commissioning_capabilities(base, run_id):
    run_root = Path(base) / run_id
    run_root.mkdir()
    return (
        run_root,
        SyntheticWriteCapability.for_local_run_root(run_root, run_id),
        LiveAntaresReadCapability.for_local_mock(
            run_root,
            run_id=run_id,
            target_date_utc=TARGET_DATE_UTC,
            release_sha=RELEASE_SHA,
            authority=LIVE_ANTARES_READ,
        ),
    )


class LiveCapabilityTests(unittest.TestCase):
    def test_live_authority_is_explicit_sealed_and_has_no_production_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            with self.assertRaisesRegex(LiveCapabilityError, "Explicit authority"):
                LiveAntaresReadCapability.for_local_mock(
                    root,
                    run_id="run",
                    target_date_utc=TARGET_DATE_UTC,
                    release_sha=RELEASE_SHA,
                    authority="",
                )
            with self.assertRaisesRegex(LiveCapabilityError, "sealed"):
                LiveAntaresReadCapability(
                    root, "run", TARGET_DATE_UTC, RELEASE_SHA, "local-mock", object()
                )
            capability = LiveAntaresReadCapability.for_local_mock(
                root,
                run_id="run",
                target_date_utc=TARGET_DATE_UTC,
                release_sha=RELEASE_SHA,
                authority=LIVE_ANTARES_READ,
            )
            self.assertFalse(hasattr(capability, "production_root"))
            self.assertFalse(hasattr(capability, "publish"))

    def test_provider_injection_is_all_or_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = LiveAntaresReadCapability.for_local_mock(
                root,
                run_id="run",
                target_date_utc=TARGET_DATE_UTC,
                release_sha=RELEASE_SHA,
                authority=LIVE_ANTARES_READ,
            )
            with self.assertRaisesRegex(ValueError, "requires search"):
                LiveAntaresProvider(capability, search_fn=lambda _query: [])

    def test_transport_limits_and_credential_bearing_url_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = LiveAntaresReadCapability.for_local_mock(
                root,
                run_id="run",
                target_date_utc=TARGET_DATE_UTC,
                release_sha=RELEASE_SHA,
                authority=LIVE_ANTARES_READ,
            )
            callables = {
                "search_fn": lambda _query: [],
                "get_by_id_fn": lambda _id: None,
                "connectivity_fn": lambda: [],
            }
            with self.assertRaisesRegex(ValueError, "may not exceed 2"):
                LiveAntaresProvider(capability, max_query_attempts=3, **callables)
            with self.assertRaisesRegex(ValueError, "may not exceed 3"):
                LiveAntaresProvider(capability, max_fetch_attempts=4, **callables)
            with self.assertRaisesRegex(ValueError, "may not exceed 4"):
                LiveAntaresProvider(capability, max_fetch_workers=5, **callables)
        with self.assertRaisesRegex(RuntimeError, "credential-free"):
            _validated_base_url(
                "https://user:secret@api.antares.noirlab.edu/v1/"
            )

    def test_cli_qualification_refuses_before_live_authority(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(
                [
                    "night",
                    "qualify",
                    TARGET_DATE_UTC,
                    "--run-id",
                    "phase6-test",
                    "--release-sha",
                    RELEASE_SHA,
                    "--json",
                ]
            )
        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("--authorize-live-read", stderr.getvalue())

    def test_cli_qualification_accepts_an_explicit_resume_attempt_identity(self):
        args = cli.build_parser().parse_args(
            [
                "night",
                "qualify",
                TARGET_DATE_UTC,
                "--run-id",
                "phase6-run",
                "--attempt-id",
                "phase6-resume-02",
                "--release-sha",
                RELEASE_SHA,
                "--authorize-live-read",
            ]
        )
        self.assertEqual(args.run_id, "phase6-run")
        self.assertEqual(args.attempt_id, "phase6-resume-02")

    def test_production_ingest_remains_refused(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = cli.main(["night", "ingest", TARGET_DATE_UTC, "--json"])
        self.assertEqual(status, 4)
        value = json.loads(stdout.getvalue())
        self.assertFalse(value["details"]["provider_constructed"])


class LiveProviderTests(unittest.TestCase):
    def test_exact_query_semantics_and_shifted_or_tagged_requests_are_refused(self):
        captured = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root,
                lambda body: captured.append(body) or [],
                lambda _id: None,
            )
            result = provider.query(_request())
            self.assertTrue(result.clean)
            self.assertEqual(
                captured,
                [_build_tile_query(FULL_TILE)],
            )
            time_range = captured[0]["query"]["bool"]["filter"][0]["range"][
                "properties.newest_alert_observation_time"
            ]
            ra_range = captured[0]["query"]["bool"]["filter"][1]["range"]["ra"]
            dec_range = captured[0]["query"]["bool"]["filter"][2]["range"]["dec"]
            self.assertEqual(time_range, {"gte": 61218.0, "lt": 61219.0})
            self.assertEqual(ra_range, {"gte": 0.0, "lt": 360.0})
            self.assertEqual(dec_range, {"gte": -90.0, "lte": 90.0})
            with self.assertRaisesRegex(Exception, "exact untagged exhaustive"):
                provider.query(
                    NightScienceRequest(
                        TARGET_DATE_UTC,
                        61218.25,
                        61219.25,
                        target_loci=None,
                    )
                )
            with self.assertRaisesRegex(Exception, "exact untagged exhaustive"):
                provider.query(
                    NightScienceRequest(
                        TARGET_DATE_UTC,
                        61218.0,
                        61219.0,
                        query_tag="changed-science",
                        target_loci=None,
                    )
                )

    def test_canonical_initial_tiling_and_boundary_operators(self):
        tiles = _make_initial_tiles(61218.0, 61219.0)
        self.assertEqual(len(tiles), 6912)
        self.assertEqual(len({tuple(tile.items()) for tile in tiles}), 6912)
        self.assertEqual(min(tile["mjd_min"] for tile in tiles), 61218.0)
        self.assertEqual(max(tile["mjd_max"] for tile in tiles), 61219.0)
        self.assertEqual({tile["ra_min"] for tile in tiles if tile["ra_min"] == 0.0}, {0.0})
        self.assertEqual(max(tile["ra_max"] for tile in tiles), 360.0)
        self.assertEqual(min(tile["dec_min"] for tile in tiles), -90.0)
        self.assertEqual(max(tile["dec_max"] for tile in tiles), 90.0)
        for tile in tiles:
            filters = _build_tile_query(tile)["query"]["bool"]["filter"]
            self.assertIn("lt", filters[0]["range"]["properties.newest_alert_observation_time"])
            self.assertIn("lt", filters[1]["range"]["ra"])
            dec_range = filters[2]["range"]["dec"]
            self.assertIn("lte" if tile["dec_max"] == 90.0 else "lt", dec_range)

    def test_split_policy_uses_largest_normalized_dimension_and_floor(self):
        children = _split_tile(FULL_TILE)
        self.assertEqual(children[0]["ra_max"], 180.0)
        self.assertEqual(children[1]["ra_min"], 180.0)
        production_tile = _make_initial_tiles(61218.0, 61219.0)[0]
        children = _split_tile(production_tile)
        self.assertEqual(children[0]["dec_max"], -75.0)
        self.assertEqual(children[1]["dec_min"], -75.0)
        time_dominant = {
            "mjd_min": 61218.0,
            "mjd_max": 61218.0 + 120.0 / 86400.0,
            "ra_min": 0.0,
            "ra_max": 0.05,
            "dec_min": 0.0,
            "dec_max": 0.05,
        }
        children = _split_tile(time_dominant)
        self.assertAlmostEqual(
            children[0]["mjd_max"], 61218.0 + 60.0 / 86400.0
        )
        self.assertEqual(children[0]["mjd_max"], children[1]["mjd_min"])
        floor = {
            "mjd_min": 61218.0,
            "mjd_max": 61218.0 + 29.0 / 86400.0,
            "ra_min": 0.0,
            "ra_max": 0.04,
            "dec_min": 0.0,
            "dec_max": 0.04,
        }
        self.assertEqual(_split_tile(floor), ())

    def test_probe_saturation_splits_with_half_open_boundaries(self):
        bodies = []

        def search(body):
            bodies.append(body)
            ra_range = body["query"]["bool"]["filter"][1]["range"]["ra"]
            lower = float(ra_range["gte"])
            upper = float(ra_range["lt"])
            if upper - lower == 360.0:
                return [FakeLocus(f"SAT-{index}") for index in range(50)]
            return [
                FakeLocus(
                    f"ANT-{lower:.0f}",
                    ra=(lower + upper) / 2.0,
                )
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, search, lambda _id: None)
            result = provider.query(_request())
            self.assertTrue(result.clean)
            details = result.evidence.details
            self.assertEqual(details["split_count"], 1)
            self.assertEqual(details["accepted_tile_count"], 2)
            self.assertEqual(details["search_request_count"], 3)
            self.assertTrue(details["coverage_complete"])
            ranges = [
                body["query"]["bool"]["filter"][1]["range"]["ra"]
                for body in bodies[1:]
            ]
            self.assertEqual(ranges, [{"gte": 0.0, "lt": 180.0}, {"gte": 180.0, "lt": 360.0}])
            self.assertEqual(
                [row["status"] for row in details["tile_trace"]],
                ["split_saturated", "accepted_exhausted", "accepted_exhausted"],
            )

    def test_saturation_at_minimum_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "src.operations.live_antares.MIN_TIME_SECONDS", 86400.0
        ), mock.patch(
            "src.operations.live_antares.MIN_RA_DEGREES", 360.0
        ), mock.patch(
            "src.operations.live_antares.MIN_DEC_DEGREES", 180.0
        ):
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root,
                lambda _body: [FakeLocus(f"SAT-{index}") for index in range(50)],
                lambda _id: None,
            )
            result = provider.query(_request())
            self.assertFalse(result.clean)
            self.assertTrue(result.evidence.partial)
            self.assertEqual(
                result.evidence.details["completion_classification"],
                LiveCompletion.INCOMPLETE.value,
            )
            self.assertEqual(
                result.evidence.details["unresolved_saturated_chunk_count"], 1
            )

    def test_cross_tile_duplicates_keep_last_with_structured_evidence(self):
        def search(body):
            ra_range = body["query"]["bool"]["filter"][1]["range"]["ra"]
            lower = float(ra_range["gte"])
            upper = float(ra_range["lt"])
            if upper - lower == 360.0:
                return [FakeLocus(f"SAT-{index}") for index in range(50)]
            return [FakeLocus("ANT-DUP", ra=(lower + upper) / 2.0)]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mock_provider(root, search, lambda _id: None).query(_request())
            self.assertTrue(result.clean)
            self.assertEqual(result.evidence.returned_loci, 1)
            self.assertEqual(float(result.loci.iloc[0]["ra"]), 270.0)
            dedup = result.evidence.details["deduplication"]
            self.assertEqual(dedup["raw_rows"], 2)
            self.assertEqual(dedup["duplicate_rows_removed"], 1)
            self.assertEqual(dedup["duplicate_identity_count"], 1)
            self.assertEqual(dedup["duplicate_identities"], ["ANT-DUP"])

    def test_whole_tile_retry_discards_partial_attempt_rows(self):
        calls = {"count": 0}

        def search(_body):
            calls["count"] += 1
            if calls["count"] == 1:
                def interrupted():
                    yield FakeLocus("ANT-DISCARDED")
                    raise ConnectionError("secret=not-evidence")

                return interrupted()
            return [FakeLocus("ANT-RETRY")]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mock_provider(root, search, lambda _id: None).query(_request())
            self.assertTrue(result.clean)
            self.assertEqual(result.loci["locus_id"].tolist(), ["ANT-RETRY"])
            details = result.evidence.details
            self.assertEqual(details["retry_count"], 1)
            self.assertEqual(details["search_request_count"], 2)
            self.assertEqual(details["partial_rows_discarded"], 1)
            self.assertEqual(
                [row["status"] for row in details["tile_trace"]],
                ["attempt_error", "accepted_exhausted"],
            )
            self.assertNotIn("secret=not-evidence", json.dumps(details))
            self.assertNotIn("not-evidence", json.dumps(details))

    def test_complete_nonzero_and_fetch_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            locus = FakeLocus(lightcurve=_lightcurve())
            provider = _mock_provider(
                root,
                _canonical_search([locus]),
                lambda _id: locus,
                canonical_tiles=True,
            )
            query_result = provider.query(_request())
            self.assertTrue(query_result.clean)
            self.assertEqual(
                query_result.evidence.details["completion_classification"],
                LiveCompletion.COMPLETE_NONZERO.value,
            )
            self.assertTrue(query_result.evidence.details["iterator_exhausted"])
            result = provider.fetch(_request(), query_result)
            self.assertTrue(result.publishable)
            self.assertEqual(result.fetch_evidence.details["requested_objects"], 1)
            self.assertEqual(result.fetch_evidence.details["completed_objects"], 1)
            self.assertEqual(result.fetch_evidence.details["failed_objects"], 0)
            self.assertEqual(result.fetch_evidence.details["max_workers"], 4)
            self.assertEqual(len(result.alerts), 1)
            artifacts = build_night_artifacts(result)
            reopened = reopen_and_validate_artifacts(artifacts, expected=result)
            self.assertEqual(reopened.manifest["schema_version"], "phase6.commissioning-candidate.v1")
            self.assertFalse(reopened.manifest["synthetic"])
            self.assertEqual(reopened.manifest["chunk_count"], 6912)
            self.assertEqual(
                reopened.manifest["extraction_method"], extraction_method_contract()
            )
            self.assertEqual(
                set(reopened.loci["source_query_mode"]),
                {"probe_first_time_ra_dec"},
            )

    def test_parquet_roundtrip_accepts_optional_nested_identifier_nulls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mixed_identifier_provider(root).fetch_night(_request())
            self.assertTrue(result.publishable)
            self.assertEqual(result.validation["lsst_dia_count"], 2)
            self.assertEqual(result.validation["lsst_ss_count"], 2)
            self.assertEqual(result.validation["lsst_identifier_count"], 3)
            self.assertEqual(result.validation["ztf_object_id_count"], 1)

            reopened = reopen_and_validate_artifacts(
                build_night_artifacts(result), expected=result
            )

            surveys = reopened.loci.set_index("locus_id")["survey"].to_dict()
            self.assertEqual(
                surveys["ANT-DIA"],
                {
                    "lsst": {
                        "dia_object_id": "DIA-1",
                        "ss_object_id": None,
                    },
                    "ztf": None,
                },
            )
            self.assertEqual(
                surveys["ANT-SS"],
                {
                    "lsst": {
                        "dia_object_id": None,
                        "ss_object_id": "SS-1",
                    },
                    "ztf": None,
                },
            )
            self.assertEqual(
                surveys["ANT-DIA-SS"],
                {
                    "lsst": {
                        "dia_object_id": "DIA-2",
                        "ss_object_id": "SS-2",
                    },
                    "ztf": {"id": "ZTF-1"},
                },
            )

    def test_parquet_roundtrip_rejects_changed_nested_non_null_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mixed_identifier_provider(root).fetch_night(_request())
            artifacts = build_night_artifacts(result)
            result.loci.at[0, "survey"]["lsst"]["dia_object_id"] = "DIA-CHANGED"

            with self.assertRaises(ArtifactValidationError) as raised:
                reopen_and_validate_artifacts(artifacts, expected=result)
            self.assertEqual(raised.exception.code, "artifact_frame_mismatch")

    def test_parquet_roundtrip_rejects_changed_scalar_and_numeric_values(self):
        for column, replacement in (("tags", "changed"), ("ra", 42.5)):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "run"
                root.mkdir()
                result = _mixed_identifier_provider(root).fetch_night(_request())
                artifacts = build_night_artifacts(result)
                result.loci.at[0, column] = replacement

                with self.assertRaises(ArtifactValidationError) as raised:
                    reopen_and_validate_artifacts(artifacts, expected=result)
                self.assertEqual(raised.exception.code, "artifact_frame_mismatch")

    def test_parquet_roundtrip_rejects_unrelated_mapping_null_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mixed_identifier_provider(root).fetch_night(_request())
            result.loci["unrelated_mapping"] = [
                {"optional_value": "present"},
                {},
                {},
            ]
            artifacts = build_night_artifacts(result)

            with self.assertRaises(ArtifactValidationError) as raised:
                reopen_and_validate_artifacts(artifacts, expected=result)
            self.assertEqual(raised.exception.code, "artifact_frame_mismatch")

    def test_parquet_roundtrip_rejects_undeclared_survey_null_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            result = _mixed_identifier_provider(root).fetch_night(_request())
            result.loci.at[0, "survey"]["lsst"]["processing_state"] = None
            artifacts = build_night_artifacts(result)

            with self.assertRaises(ArtifactValidationError) as raised:
                reopen_and_validate_artifacts(artifacts, expected=result)
            self.assertEqual(raised.exception.code, "artifact_frame_mismatch")

    def test_complete_zero_requires_normal_iterator_exhaustion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root,
                lambda _query: iter(()),
                lambda _id: None,
                canonical_tiles=True,
            )
            result = provider.fetch_night(_request())
            self.assertTrue(result.publishable)
            self.assertEqual(result.outcome, ProviderOutcome.SUCCESS_ZERO)
            self.assertEqual(
                result.query_evidence.details["completion_classification"],
                LiveCompletion.COMPLETE_ZERO.value,
            )
            self.assertIsNotNone(result.evidence.zero_row_proof)
            reopened = reopen_and_validate_artifacts(build_night_artifacts(result))
            self.assertTrue(reopened.loci.empty)
            self.assertTrue(reopened.alerts.empty)
            self.assertEqual(reopened.manifest["chunk_count"], 6912)

    def test_partial_pagination_is_incomplete_and_rows_are_not_publishable(self):
        def interrupted(_query):
            yield FakeLocus()
            raise ConnectionError("private-token=must-not-leak")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, interrupted, lambda _id: None)
            result = provider.query(_request())
            self.assertFalse(result.clean)
            self.assertTrue(result.evidence.partial)
            self.assertFalse(result.evidence.details["iterator_exhausted"])
            serialized = json.dumps(result.evidence.as_dict())
            self.assertNotIn("private-token", serialized)
            self.assertNotIn("must-not-leak", serialized)

    def test_query_retry_exhaustion_is_failed_not_zero(self):
        def failed(_query):
            raise TimeoutError("secret=never-record")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, failed, lambda _id: None)
            result = provider.query(_request())
            self.assertFalse(result.clean)
            self.assertFalse(result.evidence.partial)
            self.assertEqual(result.evidence.returned_loci, 0)
            self.assertEqual(
                result.evidence.details["completion_classification"],
                LiveCompletion.FAILED.value,
            )
            self.assertEqual(result.evidence.details["retry_count"], 1)
            self.assertNotIn("never-record", json.dumps(result.evidence.as_dict()))

    def test_malformed_response_fails_closed(self):
        class Malformed:
            locus_id = "ANT-BAD"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, lambda _query: [Malformed()], lambda _id: None)
            result = provider.query(_request())
            self.assertFalse(result.clean)
            self.assertEqual(result.outcome, ProviderOutcome.QUERY_FAILURE)

    def test_partial_fetch_and_retry_exhaustion_are_detected(self):
        loci = [FakeLocus("ANT-1"), FakeLocus("ANT-2")]

        def fetch(locus_id):
            if locus_id == "ANT-2":
                raise TimeoutError("password=redacted")
            return FakeLocus("ANT-1", lightcurve=_lightcurve())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, lambda _query: loci, fetch)
            query_result = provider.query(_request())
            result = provider.fetch(_request(), query_result)
            self.assertFalse(result.publishable)
            self.assertEqual(result.outcome, ProviderOutcome.PARTIAL_FETCH)
            details = result.fetch_evidence.details
            self.assertEqual(details["requested_objects"], 2)
            self.assertEqual(details["completed_objects"], 1)
            self.assertEqual(details["failed_objects"], 1)
            self.assertNotIn("password", json.dumps(result.fetch_evidence.as_dict()))

    def test_fetch_retry_can_complete_with_structured_count(self):
        calls = {"count": 0}
        locus = FakeLocus(lightcurve=_lightcurve())

        def fetch(_locus_id):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("transient")
            return locus

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(root, lambda _query: [locus], fetch)
            result = provider.fetch_night(_request())
            self.assertTrue(result.publishable)
            self.assertEqual(result.fetch_evidence.details["retry_count"], 1)

    def test_connectivity_uses_no_credentials_and_redacts_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root, lambda _query: [], lambda _id: None, lambda: ["alpha", "beta"]
            )
            evidence = provider.check_connectivity()
            self.assertFalse(evidence["credentials_consumed"])
            self.assertFalse(evidence["secret_material_recorded"])
            self.assertNotIn("alpha", json.dumps(evidence))


class CarriedInvariantTests(unittest.TestCase):
    def test_manifest_absent_means_not_published_across_readers(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data-root"
            cache_root = Path(temporary) / "cache-root"
            target = history.nightly_paths(data_root, TARGET_DATE_UTC)
            target["dir"].mkdir(parents=True)
            pd.DataFrame({"locus_id": ["ghost"]}).to_parquet(target["loci"], index=False)
            pd.DataFrame({"locus_id": ["ghost"]}).to_parquet(target["alerts"], index=False)
            cumulative = history.cumulative_paths(data_root)
            cumulative["dir"].mkdir(parents=True)
            pd.DataFrame().to_parquet(cumulative["loci_index"], index=False)
            pd.DataFrame().to_parquet(cumulative["nightly_summary"], index=False)
            self.assertIsNone(history.read_manifest(data_root, TARGET_DATE_UTC))
            self.assertEqual(history._manifest_paths(data_root), [])
            self.assertTrue(history.load_cumulative_alerts(data_root).empty)
            inspection = StorageLayout(data_root, cache_root).inspect_night(
                StorageLayout(data_root, cache_root).night(TARGET_DATE_UTC)
            )
            self.assertEqual(inspection.state, "incomplete")
            self.assertEqual(inspection.reason, "manifest_missing")

    def test_published_hardlinks_are_not_mutated_through_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = SyntheticWriteCapability.for_local_run_root(root, "run")
            request = NightScienceRequest(TARGET_DATE_UTC, 61218.0, 61219.0)
            spec = NightExecutionSpec(
                "hardlink-proof", "plan", RELEASE_SHA, "phase6-test", request
            )
            report = execute_synthetic_night(
                capability, SyntheticScienceProvider(), spec
            )
            self.assertTrue(report.success)
            published = capability.published_root / "data/lsst_only/nightly/2026/06/27"
            before = {
                path.name: (path.stat().st_ino, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in published.iterdir()
                if path.is_file()
            }
            self.assertFalse(any(capability.staging_root.rglob("*.parquet")))
            after = {
                path.name: (path.stat().st_ino, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in published.iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_reconciliation_lock_scope_is_shared_across_nights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = SyntheticWriteCapability.for_local_run_root(root, "run")
            first = WriterLock(
                capability, SHARED_RECONCILIATION_LOCK_IDENTITY, "night-a"
            )
            second = WriterLock(
                capability, SHARED_RECONCILIATION_LOCK_IDENTITY, "night-b"
            )
            first.acquire()
            try:
                with self.assertRaises(LockUnavailable):
                    second.acquire()
            finally:
                first.release()

    def test_distinct_night_writers_serialize_shared_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = SyntheticWriteCapability.for_local_run_root(root, "run")
            first_entered = threading.Event()
            release_first = threading.Event()
            reports = {}

            def first_hook(point, _details):
                if point == "during_reconciliation":
                    first_entered.set()
                    release_first.wait(10)

            def execute(name, day, hook=None):
                request = NightScienceRequest(
                    f"2026-06-{day:02d}",
                    61218.0 + day - 27,
                    61219.0 + day - 27,
                )
                reports[name] = execute_synthetic_night(
                    capability,
                    SyntheticScienceProvider(),
                    NightExecutionSpec(name, "plan", RELEASE_SHA, "config", request),
                    fault_hook=hook,
                )

            first_thread = threading.Thread(
                target=execute, args=("shared-first", 27, first_hook)
            )
            second_thread = threading.Thread(
                target=execute, args=("shared-second", 28)
            )
            first_thread.start()
            self.assertTrue(first_entered.wait(10))
            second_thread.start()
            second_journal = capability.journal_root / "shared-second.json"
            deadline = time.monotonic() + 10
            waiting = False
            while time.monotonic() < deadline:
                if second_journal.exists():
                    value = json.loads(second_journal.read_text(encoding="utf-8"))
                    waiting = (
                        value.get("reconciliation", {}).get("status")
                        == "waiting_for_shared_target_lock"
                    )
                    if waiting:
                        break
                time.sleep(0.02)
            self.assertTrue(waiting)
            self.assertTrue(second_thread.is_alive())
            release_first.set()
            first_thread.join(10)
            second_thread.join(10)
            self.assertTrue(reports["shared-first"].success)
            self.assertTrue(reports["shared-second"].success)
            lock_paths = {
                json.loads(
                    (capability.journal_root / f"{name}.json").read_text(
                        encoding="utf-8"
                    )
                )["reconciliation"]["lock_path"]
                for name in ("shared-first", "shared-second")
            }
            self.assertEqual(len(lock_paths), 1)


class CommissioningOrchestrationTests(unittest.TestCase):
    def test_target_eligibility_and_tripwire(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root, cache_root = _accepted_fixture(Path(temporary))
            eligibility = establish_target_eligibility(data_root)
            self.assertTrue(eligibility["passed"])
            self.assertEqual(eligibility["mjd_min"], 61218.0)
            self.assertEqual(eligibility["mjd_max"], 61219.0)
            before = capture_production_sentinel(data_root, cache_root)
            normalized_manifest = "".join(
                f"{item['sha256']}  ./{item['path']}\n"
                for item in before["durable_file_inventory"]
            ).encode("utf-8")
            self.assertEqual(
                before["checksum_manifest_sha256"],
                hashlib.sha256(normalized_manifest).hexdigest(),
            )
            after = capture_production_sentinel(data_root, cache_root)
            self.assertTrue(compare_production_sentinels(before, after)["passed"])
            paths = history.nightly_paths(data_root, "2026-06-26")
            paths["loci"].write_bytes(b"changed-in-place")
            changed = capture_production_sentinel(data_root, cache_root)
            comparison = compare_production_sentinels(before, changed)
            self.assertFalse(comparison["passed"])
            self.assertIsNone(comparison["scientific_bytes_changed"])

    def test_mocked_commissioning_retains_candidate_without_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root, cache_root = _accepted_fixture(base)
            run_root = base / "phase6-run"
            run_root.mkdir()
            write_capability = SyntheticWriteCapability.for_local_run_root(
                run_root, "phase6-run"
            )
            live_capability = LiveAntaresReadCapability.for_local_mock(
                run_root,
                run_id="phase6-run",
                target_date_utc=TARGET_DATE_UTC,
                release_sha=RELEASE_SHA,
                authority=LIVE_ANTARES_READ,
            )
            locus = FakeLocus(lightcurve=_lightcurve())
            provider = LiveAntaresProvider(
                live_capability,
                search_fn=_canonical_search([locus]),
                get_by_id_fn=lambda _id: locus,
                connectivity_fn=lambda: ["safe"],
                retry_delay_seconds=0,
                max_fetch_workers=4,
            )
            spec = NightExecutionSpec(
                "phase6-run",
                "phase6-2026-06-27",
                RELEASE_SHA,
                "phase6-test",
                _request(),
            )
            result = qualify_live_night(
                write_capability,
                live_capability,
                provider,
                spec,
                production_data_root=data_root,
                production_cache_root=cache_root,
            )
            self.assertTrue(result.report.success, result.report.to_json())
            self.assertIsNotNone(result.stage)
            self.assertTrue(result.stage.is_dir())
            self.assertFalse(history.nightly_paths(data_root, TARGET_DATE_UTC)["dir"].exists())
            self.assertFalse(cache_root.exists())
            self.assertFalse(result.report.details["publication_invoked"])
            self.assertFalse(result.report.details["publication_attempted"])
            self.assertTrue((result.evidence_root / "independent-reopen.json").is_file())
            self.assertTrue((result.evidence_root / "inventory.sha256").is_file())

    def test_artifact_failures_preserve_safe_structured_checkpoint_identity(self):
        cases = (
            ("build_night_artifacts", "artifact_build"),
            ("reopen_and_validate_artifacts", "artifact_reopen_validation"),
        )
        secret = "secret-provider-detail-must-not-survive"

        def fail_closed(*_args, **_kwargs):
            try:
                raise TypeError(secret)
            except TypeError as cause:
                raise ArtifactValidationError(
                    ProviderIssue(
                        "offline_artifact_validation_failed",
                        ProviderStage.ARTIFACT,
                        ProviderOutcome.VALIDATION_FAILURE,
                        secret,
                    )
                ) from cause

        for patched_name, expected_checkpoint in cases:
            with self.subTest(checkpoint=expected_checkpoint):
                with tempfile.TemporaryDirectory() as temporary:
                    base = Path(temporary)
                    data_root, cache_root = _accepted_fixture(base)
                    run_root = base / expected_checkpoint
                    run_root.mkdir()
                    write_capability = SyntheticWriteCapability.for_local_run_root(
                        run_root, expected_checkpoint
                    )
                    live_capability = LiveAntaresReadCapability.for_local_mock(
                        run_root,
                        run_id=expected_checkpoint,
                        target_date_utc=TARGET_DATE_UTC,
                        release_sha=RELEASE_SHA,
                        authority=LIVE_ANTARES_READ,
                    )
                    locus = FakeLocus(lightcurve=_lightcurve())
                    provider = LiveAntaresProvider(
                        live_capability,
                        search_fn=_canonical_search([locus]),
                        get_by_id_fn=lambda _id: locus,
                        connectivity_fn=lambda: ["safe"],
                        retry_delay_seconds=0,
                        max_fetch_workers=4,
                    )
                    spec = NightExecutionSpec(
                        expected_checkpoint,
                        "phase6-2026-06-27",
                        RELEASE_SHA,
                        "phase6-test",
                        _request(),
                    )
                    with mock.patch(
                        f"src.operations.commissioning.{patched_name}",
                        side_effect=fail_closed,
                    ):
                        result = qualify_live_night(
                            write_capability,
                            live_capability,
                            provider,
                            spec,
                            production_data_root=data_root,
                            production_cache_root=cache_root,
                        )

                    self.assertFalse(result.report.success)
                    failure = json.loads(
                        (result.evidence_root / "failure.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    journal = json.loads(
                        (
                            write_capability.journal_root
                            / f"{expected_checkpoint}.json"
                        ).read_text(encoding="utf-8")
                    )
                    for recorded in (
                        failure,
                        result.report.details,
                        journal["failure"],
                    ):
                        self.assertEqual(
                            recorded["failed_checkpoint"], expected_checkpoint
                        )
                        self.assertEqual(
                            recorded["error_code"],
                            "offline_artifact_validation_failed",
                        )
                        self.assertEqual(recorded["error_stage"], "artifact")
                        self.assertEqual(recorded["outcome"], "validation_failure")
                        self.assertEqual(recorded["cause_type"], "TypeError")
                    self.assertNotIn(secret, result.report.to_json())
                    self.assertNotIn(
                        secret,
                        json.dumps(failure, sort_keys=True),
                    )
                    self.assertNotIn(secret, json.dumps(journal, sort_keys=True))
                    secret_bytes = secret.encode("utf-8")
                    for path in run_root.rglob("*"):
                        if path.is_file() and not path.is_symlink():
                            self.assertNotIn(secret_bytes, path.read_bytes(), path)

    def test_restart_after_query_seal_skips_the_exhaustive_query(self):
        class StopAfterQueryCheckpoint(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root, cache_root = _accepted_fixture(base)
            run_root, write_capability, live_capability = (
                _commissioning_capabilities(base, "query-resume-run")
            )
            locus = FakeLocus(lightcurve=_lightcurve())
            first_provider = LiveAntaresProvider(
                live_capability,
                search_fn=_canonical_search([locus]),
                get_by_id_fn=lambda _id: locus,
                connectivity_fn=lambda: ["safe"],
                retry_delay_seconds=0,
                max_fetch_workers=4,
            )

            def stop(event, _details):
                if event == "after_query_checkpoint_commit":
                    raise StopAfterQueryCheckpoint("offline interruption")

            first = qualify_live_night(
                write_capability,
                live_capability,
                first_provider,
                NightExecutionSpec(
                    "query-attempt-one",
                    "phase6-2026-06-27",
                    RELEASE_SHA,
                    "phase6-test",
                    _request(),
                ),
                production_data_root=data_root,
                production_cache_root=cache_root,
                checkpoint_event_hook=stop,
            )
            self.assertFalse(first.report.success)
            self.assertEqual(
                first.report.details["failed_checkpoint"], "query_checkpoint_seal"
            )
            self.assertTrue(
                (run_root / "checkpoints" / "query-result" / "COMMITTED.json").is_file()
            )
            self.assertFalse((run_root / "checkpoints" / "live-fetch-v1").exists())

            def forbidden_query(_body):
                raise AssertionError("sealed query must not be executed again")

            resumed_provider = LiveAntaresProvider(
                live_capability,
                search_fn=forbidden_query,
                get_by_id_fn=lambda _id: locus,
                connectivity_fn=lambda: ["safe"],
                retry_delay_seconds=0,
                max_fetch_workers=4,
            )
            resumed = qualify_live_night(
                write_capability,
                live_capability,
                resumed_provider,
                NightExecutionSpec(
                    "query-attempt-two",
                    "phase6-2026-06-27",
                    RELEASE_SHA,
                    "phase6-test",
                    _request(),
                ),
                production_data_root=data_root,
                production_cache_root=cache_root,
            )
            self.assertTrue(resumed.report.success, resumed.report.to_json())
            self.assertTrue(resumed.report.details["query_checkpoint_reused"])
            self.assertEqual(resumed.report.details["fetch_checkpoint"]["fetched_segments"], 1)
            self.assertNotEqual(first.evidence_root, resumed.evidence_root)

    def test_downstream_artifact_failure_reuses_query_and_fetch_without_network(self):
        cases = (
            ("build_night_artifacts", "artifact_build"),
            ("reopen_and_validate_artifacts", "artifact_reopen_validation"),
        )
        for patched_name, expected_checkpoint in cases:
            with self.subTest(checkpoint=expected_checkpoint), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                data_root, cache_root = _accepted_fixture(base)
                run_id = f"resume-{expected_checkpoint}"
                run_root, write_capability, live_capability = (
                    _commissioning_capabilities(base, run_id)
                )
                locus = FakeLocus(lightcurve=_lightcurve())
                first_provider = LiveAntaresProvider(
                    live_capability,
                    search_fn=_canonical_search([locus]),
                    get_by_id_fn=lambda _id: locus,
                    connectivity_fn=lambda: ["safe"],
                    retry_delay_seconds=0,
                    max_fetch_workers=4,
                )
                injected = ArtifactValidationError(
                    ProviderIssue(
                        "offline_downstream_failure",
                        ProviderStage.ARTIFACT,
                        ProviderOutcome.VALIDATION_FAILURE,
                        "static offline failure",
                    )
                )
                with mock.patch(
                    f"src.operations.commissioning.{patched_name}",
                    side_effect=injected,
                ):
                    first = qualify_live_night(
                        write_capability,
                        live_capability,
                        first_provider,
                        NightExecutionSpec(
                            "artifact-attempt-one",
                            "phase6-2026-06-27",
                            RELEASE_SHA,
                            "phase6-test",
                            _request(),
                        ),
                        production_data_root=data_root,
                        production_cache_root=cache_root,
                    )
                self.assertFalse(first.report.success)
                self.assertEqual(
                    first.report.details["failed_checkpoint"], expected_checkpoint
                )
                self.assertTrue(
                    (run_root / "checkpoints" / "live-fetch-v1" / "fetch-complete.json").is_file()
                )

                def forbidden(*_args, **_kwargs):
                    raise AssertionError("complete checkpoints require no network callback")

                resumed_provider = LiveAntaresProvider(
                    live_capability,
                    search_fn=forbidden,
                    get_by_id_fn=forbidden,
                    connectivity_fn=forbidden,
                    retry_delay_seconds=0,
                    max_fetch_workers=4,
                )
                resumed = qualify_live_night(
                    write_capability,
                    live_capability,
                    resumed_provider,
                    NightExecutionSpec(
                        "artifact-attempt-two",
                        "phase6-2026-06-27",
                        RELEASE_SHA,
                        "phase6-test",
                        _request(),
                    ),
                    production_data_root=data_root,
                    production_cache_root=cache_root,
                )
                self.assertTrue(resumed.report.success, resumed.report.to_json())
                self.assertTrue(resumed.report.details["query_checkpoint_reused"])
                self.assertEqual(
                    resumed.report.details["fetch_checkpoint"]["fetched_segments"], 0
                )
                self.assertEqual(
                    resumed.report.details["fetch_checkpoint"]["reused_segments"], 1
                )
                connectivity = json.loads(
                    (resumed.evidence_root / "connectivity.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(connectivity["network_attempted"])
                self.assertFalse(
                    history.nightly_paths(data_root, TARGET_DATE_UTC)["dir"].exists()
                )
                self.assertFalse(cache_root.exists())

    def test_phase6_manifest_rejects_tampered_completion_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            locus = FakeLocus(lightcurve=_lightcurve())
            provider = _mock_provider(
                root,
                _canonical_search([locus]),
                lambda _id: locus,
                canonical_tiles=True,
            )
            artifacts = build_night_artifacts(provider.fetch_night(_request()))
            manifest = json.loads(artifacts["manifest.json"])
            manifest["query_evidence"]["details"]["coverage_complete"] = False
            artifacts["manifest.json"] = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self.assertRaises(ArtifactValidationError):
                reopen_and_validate_artifacts(artifacts)

    def test_phase6_manifest_rejects_authority_and_provenance_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            locus = FakeLocus(lightcurve=_lightcurve())
            provider = _mock_provider(
                root,
                _canonical_search([locus]),
                lambda _id: locus,
                canonical_tiles=True,
            )
            base = build_night_artifacts(provider.fetch_night(_request()))
            manifest_json = base["manifest.json"].decode("utf-8")
            mutations = {
                "status": lambda value: value.__setitem__("status", "incomplete"),
                "paths": lambda value: value.__setitem__(
                    "paths",
                    {
                        "loci": "/astro/store/shire/ANTARES/data/loci.parquet",
                        "alerts": "alerts.parquet",
                        "manifest": "manifest.json",
                    },
                ),
                "scenario": lambda value: value.__setitem__(
                    "provider_scenario", "unsealed-provider"
                ),
                "pagination": lambda value: value["query_evidence"]["details"].__setitem__(
                    "pagination_mode", "first-page-only"
                ),
                "capability": lambda value: value["query_evidence"]["details"].__setitem__(
                    "capability_environment", "production-write"
                ),
                "execution-policy": lambda value: value["query_evidence"]["details"][
                    "execution_policy"
                ].__setitem__("max_query_attempts", 99),
                "logical-count": lambda value: value.__setitem__(
                    "lsst_dia_count", int(value["lsst_dia_count"]) + 1
                ),
                "fetch-failure-types": lambda value: value["fetch_evidence"][
                    "details"
                ].__setitem__("failure_exception_types", ["builtins.RuntimeError"]),
                "fetch-attempt-limit": lambda value: value["fetch_evidence"][
                    "details"
                ].__setitem__("max_attempts_per_object", 99),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    artifacts = dict(base)
                    manifest = json.loads(manifest_json)
                    mutate(manifest)
                    artifacts["manifest.json"] = (
                        json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    with self.assertRaises(ArtifactValidationError):
                        reopen_and_validate_artifacts(artifacts)

    def test_phase6_manifest_rejects_impossible_zero_row_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root,
                lambda _query: [],
                lambda _id: None,
                canonical_tiles=True,
            )
            artifacts = build_night_artifacts(provider.fetch_night(_request()))
            manifest = json.loads(artifacts["manifest.json"])
            details = manifest["query_evidence"]["details"]
            details["tile_trace"][0]["returned_loci"] = 1
            details["raw_returned_loci"] = 1
            details["deduplication"].update(
                {
                    "raw_rows": 1,
                    "duplicate_rows_removed": 1,
                    "duplicate_identity_count": 1,
                    "duplicate_identities": ["ANT-IMPOSSIBLE"],
                    "duplicate_identity_sha256": hashlib.sha256(
                        b"ANT-IMPOSSIBLE\n"
                    ).hexdigest(),
                }
            )
            manifest["deduplication"] = dict(details["deduplication"])
            details["tile_trace_sha256"] = hashlib.sha256(
                json.dumps(
                    {"tiles": details["tile_trace"]},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            artifacts["manifest.json"] = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self.assertRaises(ArtifactValidationError):
                reopen_and_validate_artifacts(artifacts)

    def test_phase6_manifest_rejects_multiple_retries_for_one_tile(self):
        calls = {"count": 0}

        def retry_once(_query):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("secret=not-evidence")
            return []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            provider = _mock_provider(
                root,
                retry_once,
                lambda _id: None,
                canonical_tiles=True,
            )
            artifacts = build_night_artifacts(provider.fetch_night(_request()))
            manifest = json.loads(artifacts["manifest.json"])
            details = manifest["query_evidence"]["details"]
            self.assertEqual(details["tile_trace"][0]["status"], "attempt_error")
            details["tile_trace"].insert(1, dict(details["tile_trace"][0]))
            details["retry_count"] += 1
            details["search_request_count"] += 1
            details["tile_trace_sha256"] = hashlib.sha256(
                json.dumps(
                    {"tiles": details["tile_trace"]},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            artifacts["manifest.json"] = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self.assertRaises(ArtifactValidationError):
                reopen_and_validate_artifacts(artifacts)

    def test_phase6_manifest_rejects_tampered_fetch_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            locus = FakeLocus(lightcurve=_lightcurve())
            provider = _mock_provider(
                root,
                _canonical_search([locus]),
                lambda _id: locus,
                canonical_tiles=True,
            )
            artifacts = build_night_artifacts(provider.fetch_night(_request()))
            manifest = json.loads(artifacts["manifest.json"])
            manifest["fetch_evidence"]["details"]["lightcurves_empty"] = 1
            artifacts["manifest.json"] = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            with self.assertRaises(ArtifactValidationError):
                reopen_and_validate_artifacts(artifacts)

    def test_failed_fetch_still_runs_and_seals_post_failure_tripwire(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data_root, cache_root = _accepted_fixture(base)
            run_root = base / "failed-run"
            run_root.mkdir()
            write_capability = SyntheticWriteCapability.for_local_run_root(
                run_root, "failed-run"
            )
            live_capability = LiveAntaresReadCapability.for_local_mock(
                run_root,
                run_id="failed-run",
                target_date_utc=TARGET_DATE_UTC,
                release_sha=RELEASE_SHA,
                authority=LIVE_ANTARES_READ,
            )
            locus = FakeLocus()
            provider = LiveAntaresProvider(
                live_capability,
                search_fn=lambda _query: [locus],
                get_by_id_fn=lambda _id: (_ for _ in ()).throw(TimeoutError()),
                connectivity_fn=lambda: ["safe"],
                initial_tiles_fn=lambda _lo, _hi: [FULL_TILE],
                retry_delay_seconds=0,
                max_fetch_attempts=1,
                max_fetch_workers=4,
            )
            result = qualify_live_night(
                write_capability,
                live_capability,
                provider,
                NightExecutionSpec(
                    "failed-run", "phase6", RELEASE_SHA, "phase6-test", _request()
                ),
                production_data_root=data_root,
                production_cache_root=cache_root,
            )
            self.assertFalse(result.report.success)
            comparison = json.loads(
                (result.evidence_root / "production-sentinel-comparison.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(comparison["passed"])
            self.assertFalse(result.report.details["production_target_created"])
            self.assertFalse(result.report.details["scientific_bytes_changed"])
            self.assertTrue((result.evidence_root / "inventory.sha256").is_file())


if __name__ == "__main__":
    unittest.main()
