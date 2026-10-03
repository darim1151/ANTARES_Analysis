"""G6.2 regressions: a non-JSON ANTARES response page is transport, not malformed data.

The first forward production range (v0.4.5, 2026-06-30..07-06) blocked June 30
and July 6 with ``live_query_malformed``. Each preserved query journal ends in
one ``attempt_error`` on attempt 1: ``requests.exceptions.JSONDecodeError``
after 40 (June 30) or 30 (July 6) rows, ``retryable: false``. antares-client
1.14.0 calls ``response.json()`` on every page, including HTTP >= 400 bodies,
and ``requests.exceptions.JSONDecodeError`` subclasses ``ValueError``, which
the provider reserves for its own row validation. These tests drive the
canonical traversal to the exact recorded failing tiles and prove that
coverage either closes exactly or still fails closed.
"""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import requests
from requests.models import Response

import v3_fixtures as F
from test_operations_phase6 import FakeLocus
from src.operations import backfill as B
from src.operations.live_antares import (
    PROBE_LIMIT, LiveAntaresProvider, LiveCompletion, _TILE_KEYS, _make_initial_tiles,
    _retryable_query_error, night_mjd_interval,
)
from src.operations.publication import FailureCategory
from src.operations.query_checkpoint import QueryResultCheckpointBindings
from src.operations.science import NightScienceRequest, ProviderOutcome, QueryInterruptedError

PAGE_SIZE = 10
RETRY_ROWS = 45
BAD_GATEWAY = b"<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>"
JSON_DECODE_ERROR = "requests.exceptions.JSONDecodeError"
# Terminal events of the preserved Arnor journals (event 9898 / event 10665),
# both reached after exactly 12 saturated ancestor splits.
RECORDED_FAILURES = {
    "2026-06-30": {
        "tile": {"mjd_min": 61221.359375, "mjd_max": 61221.36458333333,
                 "ra_min": 349.6875, "ra_max": 350.15625, "dec_min": -15.0, "dec_max": -14.0625},
        "query_sha256": "d4055b25515a832e901d6a9e9407036aa1afd858fa27604045be2a3351501aaf",
        "partial_rows_discarded": 40,
    },
    "2026-07-06": {
        "tile": {"mjd_min": 61227.1875, "mjd_max": 61227.192708333336,
                 "ra_min": 255.46875, "ra_max": 255.9375, "dec_min": -21.5625, "dec_max": -20.625},
        "query_sha256": "63290278980f8157604e9e08067fe5e8412ab0932efa7d8ae6e2a6fc2c09e969",
        "partial_rows_discarded": 30,
    },
}
# tile_trace_sha256 of the error-free June-30 traversal, computed with the
# unmodified v0.4.5 provider (7211b5c): the fix must not alter clean traces.
V045_ERROR_FREE_TRACE_SHA256 = "a54b8e8565a5e8b0bca50f23c51e93f4beffd8b6f5ddef4794530cda3d745978"


def _client_pages(loci, *, fail_page=None):
    """antares-client 1.14.0 ``_list_all_resources`` control flow over fake pages.

    Page ``fail_page`` is an HTTP 502 with an HTML body; like the client, the
    status check calls ``response.json()`` first, so the real requests
    ``JSONDecodeError`` escapes mid-iteration after the earlier pages' rows.
    """
    page = 0
    while True:
        response = Response()
        response.encoding = "utf-8"
        if page == fail_page:
            response.status_code, response._content = 502, BAD_GATEWAY
        else:
            more = (page + 1) * PAGE_SIZE < len(loci)
            response.status_code = 200
            response._content = json.dumps({
                "data": list(range(page * PAGE_SIZE, min(len(loci), (page + 1) * PAGE_SIZE))),
                "links": {"next": f"https://api.invalid/v1/loci?page={page + 1}" if more else None},
            }).encode("utf-8")
        if response.status_code >= 400:
            raise RuntimeError(response.json())  # AntaresException(response.json())
        yield from (loci[index] for index in response.json()["data"])
        if response.json()["links"]["next"] is None:
            return
        page += 1


def _non_json_error():
    response = Response()
    response.status_code, response._content, response.encoding = 502, BAD_GATEWAY, "utf-8"
    try:
        response.json()
    except requests.exceptions.JSONDecodeError as error:
        return error
    raise AssertionError("A non-JSON body must not decode.")


def _tile(body):
    time, ra, dec = (item["range"] for item in body["query"]["bool"]["filter"][:3])
    time = time["properties.newest_alert_observation_time"]
    return {"mjd_min": time["gte"], "mjd_max": time["lt"], "ra_min": ra["ra"]["gte"],
            "ra_max": ra["ra"]["lt"], "dec_min": dec["dec"]["gte"],
            "dec_max": dec["dec"].get("lt", dec["dec"].get("lte"))}


def _contains(outer, inner):
    return all(outer[low] <= inner[low] and inner[high] <= outer[high]
               for low, high in (("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max")))


def _loci_inside(tile, count):
    return [FakeLocus(
        f"G62-{index:02d}",
        mjd=tile["mjd_min"] + (tile["mjd_max"] - tile["mjd_min"]) * (index + .5) / count,
        ra=tile["ra_min"] + (tile["ra_max"] - tile["ra_min"]) * (index + .5) / count,
        dec=tile["dec_min"] + (tile["dec_max"] - tile["dec_min"]) * (index + .5) / count,
    ) for index in range(count)]


def recorded_search(night, failures, calls):
    """Saturate exactly the recorded lineage; the failing tile's first
    ``failures`` attempts end on a non-JSON page after the recorded rows."""
    recorded = RECORDED_FAILURES[night]
    failing = recorded["tile"]

    def search(body):
        tile = _tile(body)
        calls.append(tile)
        if tile == failing:
            failed = calls.count(failing) <= failures
            return _client_pages(_loci_inside(tile, RETRY_ROWS),
                                 fail_page=recorded["partial_rows_discarded"] // PAGE_SIZE if failed else None)
        if _contains(tile, failing):
            return _client_pages(_loci_inside(tile, PROBE_LIMIT))
        return _client_pages([])
    return search


def provider_for(root, night, search, **options):
    capability = F.mock_read_capability(root, root.name, night, F.CANDIDATE_RELEASE)
    return LiveAntaresProvider(capability, search_fn=search, get_by_id_fn=lambda _: None,
                               connectivity_fn=lambda: [], retry_delay_seconds=0,
                               clock=F.fixed_clock, monotonic=lambda: 0.0, **options)


def request_for(night):
    return NightScienceRequest(night, *night_mjd_interval(night), target_loci=None)


def accepted_tiles(result):
    return [{key: row[key] for key in _TILE_KEYS} for row in result.evidence.details["tile_trace"]
            if row["status"] == "accepted_exhausted"]


class RecordedTransportFailureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name).resolve()

    def root(self, name):
        root = self.parent / name
        root.mkdir(mode=0o700, parents=True)
        return root

    def run_query(self, night, failures, calls=None, label="run"):
        calls = [] if calls is None else calls
        root = self.root(f"{label}-{night}") / f"night-{night}"
        root.mkdir(mode=0o700)
        provider = provider_for(root, night, recorded_search(night, failures, calls))
        return provider.query(request_for(night)), calls

    def assertExactPartition(self, tiles, night):
        """Exact rational proof against the canonical initial grid: every tile
        lies inside one initial tile, tiles inside one initial tile are
        pairwise disjoint half-open boxes, and their volumes sum exactly."""
        mjd_min, mjd_max = night_mjd_interval(night)
        initial = _make_initial_tiles(mjd_min, mjd_max)

        def volume(tile):
            edges = [Fraction(tile[key]) for key in _TILE_KEYS]
            return (edges[1] - edges[0]) * (edges[3] - edges[2]) * (edges[5] - edges[4])

        domain = {"mjd_min": mjd_min, "mjd_max": mjd_max, "ra_min": 0.0, "ra_max": 360.0,
                  "dec_min": -90.0, "dec_max": 90.0}
        self.assertEqual(sum(map(volume, initial), Fraction(0)), volume(domain))
        for key, extreme in (("mjd_min", min), ("ra_min", min), ("dec_min", min),
                             ("mjd_max", max), ("ra_max", max), ("dec_max", max)):
            self.assertEqual(extreme(tile[key] for tile in tiles), domain[key], key)
        owners = {(tile["mjd_min"], tile["ra_min"], tile["dec_min"]): index
                  for index, tile in enumerate(initial)}
        corners = [sorted({tile[key] for tile in initial}) for key in ("mjd_min", "ra_min", "dec_min")]
        groups = {index: [] for index in range(len(initial))}
        for tile in tiles:
            corner = tuple(max(edge for edge in axis if edge <= tile[key])
                           for axis, key in zip(corners, ("mjd_min", "ra_min", "dec_min")))
            owner = owners[corner]
            self.assertTrue(_contains(initial[owner], tile), tile)
            groups[owner].append(tile)
        for index, members in groups.items():
            self.assertEqual(sum(map(volume, members), Fraction(0)), volume(initial[index]), initial[index])
            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    a, b = members[first], members[second]
                    self.assertTrue(any(a[high] <= b[low] or b[high] <= a[low] for low, high in (
                        ("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max"))), (a, b))

    def test_recorded_failures_retry_once_and_close_exact_coverage(self):
        for night, recorded in RECORDED_FAILURES.items():
            with self.subTest(night=night):
                result, calls = self.run_query(night, failures=1)
                details = result.evidence.details
                trace = details["tile_trace"]
                index = next(i for i, row in enumerate(trace) if row["status"] == "attempt_error")
                error, retried = trace[index], trace[index + 1]
                # The live terminal event, now classified as a bounded transient retry.
                self.assertEqual({key: error[key] for key in _TILE_KEYS}, recorded["tile"])
                self.assertEqual(error["query_sha256"], recorded["query_sha256"])
                self.assertEqual(error["exception_type"], JSON_DECODE_ERROR)
                self.assertEqual(error["partial_rows_discarded"], recorded["partial_rows_discarded"])
                self.assertEqual((error["attempt"], error["retryable"]), (1, True))
                self.assertEqual({key: retried[key] for key in _TILE_KEYS}, recorded["tile"])
                self.assertEqual((retried["attempt"], retried["status"], retried["returned_loci"]),
                                 (2, "accepted_exhausted", RETRY_ROWS))
                self.assertEqual(calls.count(recorded["tile"]), 2)
                self.assertEqual(details["split_count"], 12)
                self.assertTrue(result.clean)
                self.assertTrue(details["coverage_complete"])
                self.assertEqual(details["terminal_pending_tile_count"], 0)
                self.assertEqual(details["completion_classification"], LiveCompletion.COMPLETE_NONZERO.value)
                self.assertEqual(details["retry_count"], 1)
                self.assertEqual(details["retry_exception_types"], [JSON_DECODE_ERROR])
                self.assertEqual(details["partial_rows_discarded"],
                                 12 * PROBE_LIMIT + recorded["partial_rows_discarded"])
                # Only the complete retry contributes rows; the failed page set is discarded.
                self.assertEqual(sorted(result.loci["locus_id"]), [f"G62-{i:02d}" for i in range(RETRY_ROWS)])
                self.assertEqual(details["accepted_tile_count"], len(_make_initial_tiles(
                    *night_mjd_interval(night))) + details["split_count"])
                self.assertExactPartition(accepted_tiles(result), night)

    def test_persistent_non_json_page_still_fails_closed(self):
        for night, recorded in RECORDED_FAILURES.items():
            with self.subTest(night=night):
                result, calls = self.run_query(night, failures=2)
                details = result.evidence.details
                self.assertFalse(result.clean)
                self.assertEqual(result.outcome, ProviderOutcome.QUERY_INTERRUPTION)
                self.assertEqual(details["completion_classification"], LiveCompletion.INCOMPLETE.value)
                self.assertFalse(details["coverage_complete"])
                self.assertGreater(details["terminal_pending_tile_count"], 0)
                self.assertEqual(calls.count(recorded["tile"]), 2)
                self.assertNotIn(recorded["tile"], accepted_tiles(result))
                self.assertIsNone(result.loci)
                (issue,) = result.evidence.errors
                self.assertEqual((issue.code, issue.retryable), ("live_query_incomplete", True))
                with self.assertRaises(QueryInterruptedError) as raised:
                    result.require_completed()
                self.assertEqual(B.failure_classification(raised.exception),
                                 (FailureCategory.TRANSIENT_NETWORK, True))

    def test_unresolved_saturation_still_fails_closed_after_transport_retry(self):
        night = "2026-06-30"
        mjd_min, mjd_max = night_mjd_interval(night)
        domain = {"mjd_min": mjd_min, "mjd_max": mjd_max, "ra_min": 0.0, "ra_max": 360.0,
                  "dec_min": -90.0, "dec_max": 90.0}
        attempts = []

        def search(body):
            attempts.append(_tile(body))
            loci = _loci_inside(domain, PROBE_LIMIT + 10)
            return _client_pages(loci, fail_page=1 if len(attempts) == 1 else None)

        root = self.root("minimum") / f"night-{night}"
        root.mkdir(mode=0o700)
        with mock.patch("src.operations.live_antares.MIN_TIME_SECONDS", 86400.0), \
                mock.patch("src.operations.live_antares.MIN_RA_DEGREES", 360.0), \
                mock.patch("src.operations.live_antares.MIN_DEC_DEGREES", 180.0):
            provider = provider_for(root, night, search, initial_tiles_fn=lambda _lo, _hi: [domain])
            result = provider.query(request_for(night))
        details = result.evidence.details
        self.assertEqual(len(attempts), 2)
        self.assertFalse(result.clean)
        self.assertEqual(details["completion_classification"], LiveCompletion.INCOMPLETE.value)
        self.assertEqual(details["unresolved_saturated_tile_count"], 1)
        self.assertFalse(details["coverage_complete"])
        self.assertEqual(result.evidence.errors[0].code, "live_query_saturation_unresolved")
        self.assertIsNone(result.loci)

    def test_provider_validation_errors_remain_non_retryable(self):
        self.assertTrue(_retryable_query_error(_non_json_error()))
        for error in (ConnectionError("reset"), TimeoutError("slow"),
                      requests.exceptions.ConnectionError("reset"), RuntimeError("AntaresException")):
            self.assertTrue(_retryable_query_error(error), error)
        for error in (ValueError("ANTARES locus is outside its query tile."), TypeError("bad locus"),
                      json.JSONDecodeError("not from a response", "", 0),
                      requests.exceptions.InvalidURL("malformed next link")):
            self.assertFalse(_retryable_query_error(error), error)

        night = "2026-06-30"
        outside = FakeLocus("G62-OUTSIDE", mjd=night_mjd_interval(night)[1] + 1.0)
        calls = []

        def search(body):
            calls.append(body)
            return _client_pages([outside])

        root = self.root("validation") / f"night-{night}"
        root.mkdir(mode=0o700)
        result = provider_for(root, night, search).query(request_for(night))
        self.assertEqual(len(calls), 1)
        (issue,) = result.evidence.errors
        self.assertEqual((issue.code, issue.retryable), ("live_query_malformed", False))
        self.assertEqual(result.evidence.details["tile_trace"][-1]["exception_type"], "builtins.ValueError")

    def test_error_free_traversal_is_unchanged_from_v045(self):
        result, _calls = self.run_query("2026-06-30", failures=0)
        details = result.evidence.details
        self.assertTrue(result.clean)
        self.assertEqual(details["retry_count"], 0)
        self.assertEqual(details["tile_trace_sha256"], V045_ERROR_FREE_TRACE_SHA256)
        self.assertExactPartition(accepted_tiles(result), "2026-06-30")

    def test_deterministic_replay_yields_identical_partition_and_proof(self):
        first, _ = self.run_query("2026-07-06", failures=1, label="first")
        second, _ = self.run_query("2026-07-06", failures=1, label="second")
        self.assertEqual(first.evidence.details, second.evidence.details)
        self.assertEqual(accepted_tiles(first), accepted_tiles(second))
        self.assertTrue(first.loci.equals(second.loci))


class RecordedTransportResumeTests(unittest.TestCase):
    """Durable query-progress journal: death after the retry decision, and a
    persistent failure resumed by a new invocation, both reprove the same
    frontier and finish with the uninterrupted run's partition and rows."""

    night = "2026-06-30"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name).resolve()
        self.failing = RECORDED_FAILURES[self.night]["tile"]

    def resumable(self, label, failures, calls, **options):
        root = self.parent / label / f"night-{self.night}"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        provider = provider_for(root, self.night, recorded_search(self.night, failures, calls), **options)
        request = request_for(self.night)
        bindings = QueryResultCheckpointBindings(
            root.name, F.CANDIDATE_RELEASE, "c" * 64, self.night, provider.provider_name, provider.scenario,
            {"scientific_contract": provider.scientific_contract(request),
             "execution_policy": provider.execution_policy()})
        return provider, request, bindings, root

    def journal(self, root):
        directory = root / "checkpoints/query-progress-v2"
        return [json.loads(path.read_text())["payload"]["event"]
                for path in sorted(directory.glob("event-*.json"))]

    def reference(self):
        provider, request, bindings, _root = self.resumable("reference", 1, [])
        return provider.query_resumable(request, bindings)

    def test_death_after_durable_retry_decision_resumes_same_frontier(self):
        reference = self.reference()
        provider, request, bindings, root = self.resumable("killed", 1, [])
        provider.sleeper = lambda _: (_ for _ in ()).throw(SystemExit("death after durable retry"))
        with self.assertRaises(SystemExit):
            provider.query_resumable(request, bindings)
        last = self.journal(root)[-1]
        self.assertEqual({key: last["trace"][key] for key in _TILE_KEYS}, self.failing)
        self.assertEqual((last["trace"]["status"], last["trace"]["retryable"], last["retry_scheduled"]),
                         ("attempt_error", True, True))
        calls = []
        # The resumed service is healthy: the failure was one transient page.
        provider, request, bindings, _ = self.resumable("killed", 0, calls)
        resumed = provider.query_resumable(request, bindings)
        self.assertEqual(calls[0], self.failing)
        self.assertEqual(calls.count(self.failing), 1)
        self.assertEqual(resumed.evidence.details["tile_trace"],
                         reference.evidence.details["tile_trace"])
        self.assertEqual(resumed.evidence.details["tile_trace_sha256"],
                         reference.evidence.details["tile_trace_sha256"])
        self.assertTrue(resumed.loci.equals(reference.loci))

    def test_persistent_failure_resumes_with_fresh_bounded_invocation(self):
        reference = self.reference()
        calls = []
        provider, request, bindings, root = self.resumable("persistent", 2, calls)
        failed = provider.query_resumable(request, bindings)
        self.assertFalse(failed.clean)
        self.assertTrue(all(issue.retryable for issue in failed.evidence.errors))
        self.assertEqual(calls.count(self.failing), 2)
        calls = []
        provider, request, bindings, _ = self.resumable("persistent", 0, calls)
        resumed = provider.query_resumable(request, bindings)
        events = self.journal(root)
        boundaries = [event for event in events if "invocation_boundary" in event]
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["invocation_boundary"]["tile"], self.failing)
        self.assertEqual(calls[0], self.failing)
        self.assertTrue(resumed.clean)
        self.assertTrue(resumed.evidence.details["coverage_complete"])
        self.assertEqual(accepted_tiles(resumed), accepted_tiles(reference))
        self.assertTrue(resumed.loci.equals(reference.loci))
        self.assertEqual(resumed.evidence.details["locus_order_sha256"],
                         reference.evidence.details["locus_order_sha256"])


if __name__ == "__main__":
    unittest.main()
