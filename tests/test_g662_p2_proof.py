"""Explicit offline P2 qualification; these limits are synthetic test values."""
import copy
import hashlib
import json
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pandas as pd

import v3_fixtures as fixtures
from test_operations_phase6 import FakeLocus, _body_matches_locus
from src.operations import live_antares as L, science as S
from src.operations import backfill as B, production_range as R
from src.operations import query_checkpoint as Q
from src.operations.query_checkpoint import QueryResultCheckpointBindings, QueryCheckpointError

NIGHT = "2026-06-27"
RELEASE = "a" * 40


def test_profile(**changes):
    return L.P2ProofProfile(**{
        **dict(max_depth=4, max_nodes_per_root=31, max_nodes_per_night=128,
               max_search_attempts=16000, crash_reserve=3, max_event_bytes=1000000), **changes})


def floor_tile():
    point = (61218.1, 258.02, -23.36)
    def contains(tile):
        return all(tile[low] <= value < tile[high] for value, (low, high) in zip(point,
            (("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max"))))
    tile = next(tile for tile in L._make_initial_tiles(61218., 61219.) if contains(tile))
    while L._split_tile(tile):
        tile = next(child for child in L._split_tile(tile) if contains(child))
    return tile


def dense_loci(tile=None, *, colocated=False, axis="time", count=50):
    tile = tile or floor_tile()
    coordinates = [(tile[low] + tile[high]) / 2 for low, high in (
        ("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max"))]
    index = {"time": 0, "ra": 1, "dec": 2}[axis]
    low, high = [("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max")][index]
    result = []
    for n in range(count):
        point = list(coordinates)
        if not colocated:
            point[index] = tile[low] + (tile[high] - tile[low]) * (.25 if n < count // 2 else .75)
        result.append(FakeLocus(f"ID-{count-n:03d}", mjd=point[0], ra=point[1], dec=point[2]))
    return result


def provider_for(root, loci, profile, *, search=None):
    capability = fixtures.mock_read_capability(root, root.name, NIGHT, RELEASE)
    by_id = {locus.locus_id: locus for locus in loci}
    return L.LiveAntaresProvider(capability, proof_profile=profile,
        search_fn=search or (lambda body: [locus for locus in loci if _body_matches_locus(body, locus)]),
        get_by_id_fn=by_id.get, connectivity_fn=lambda: [], sleeper=lambda _: None,
        clock=fixtures.fixed_clock, monotonic=lambda: 0.0)


def bindings(provider, root, request):
    return QueryResultCheckpointBindings(root.name, RELEASE, "b" * 64, NIGHT,
        provider.provider_name, provider.scenario, {"scientific_contract": provider.scientific_contract(request),
                                               "execution_policy": provider.execution_policy()})


class P2ProofTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "night-2026-06-27"
        self.root.mkdir(mode=0o700)
        self.request = R.LiveRangeAdapter(self.root, RELEASE, None).acquisition_request(NIGHT)

    def run_query(self, loci=(), profile=None, search=None):
        profile = profile or test_profile()
        provider = provider_for(self.root, loci, profile, search=search)
        result = provider.query(self.request)
        if result.clean:
            S.validate_p2_query_result(self.request, result, profile)
        return provider, result

    def test_p1_is_default_and_p2_identity_is_distinct(self):
        capability = fixtures.mock_read_capability(self.root, self.root.name, NIGHT, RELEASE)
        p1 = L.LiveAntaresProvider(capability, search_fn=lambda _: [], get_by_id_fn=lambda _: None,
                                  connectivity_fn=lambda: [])
        p2 = provider_for(self.root, [], test_profile())
        self.assertEqual(p1.scientific_contract(self.request), L._scientific_query_contract(self.request))
        self.assertNotEqual(p1.scientific_contract(self.request), p2.scientific_contract(self.request))
        self.assertNotIn("extraction_method", B.selection_descriptor_for_request(self.request))
        with self.assertRaises(L.LiveCapabilityError):
            p2.capability = types.SimpleNamespace(environment="arnor-commissioning")
            p2._load_client()
        offline = L.LiveAntaresProvider(capability, proof_profile=test_profile())
        with mock.patch("requests.sessions.Session.request", side_effect=AssertionError("live access")):
            with self.assertRaises(L.LiveCapabilityError):
                offline._load_client()
            refused = offline.query(self.request)
        self.assertFalse(refused.clean)
        self.assertEqual(refused.evidence.errors[0].code, "p2_malformed")
        self.assertEqual(refused.evidence.details["p2_budget"]["unknown_attempts"], 0)

    def test_empty_and_49_natural_exhaustion(self):
        for count in (0, 49):
            with self.subTest(count=count):
                _, result = self.run_query(dense_loci(count=count))
                self.assertTrue(result.clean)
                self.assertEqual(len(result.loci), count)
                self.assertEqual(result.evidence.details["p2_budget"]["secondary_nodes"], 0)

    def test_floor_entry_one_extra_level_and_no_repeated_root(self):
        _, result = self.run_query(dense_loci())
        self.assertTrue(result.clean)
        events = result.evidence.details["p2_events"]
        entries = [e["p2_event"] for e in events if e["p2_event"].get("decision", {}).get("entered_secondary")]
        self.assertEqual(len(entries), 1)
        root = entries[0]["node"]
        self.assertFalse(L._split_tile(root["tile"]))
        self.assertEqual(sum(e["p2_event"]["kind"] == "reserve" and e["p2_event"]["node"]["id"] == root["id"] for e in events), 1)
        self.assertEqual(result.evidence.details["p2_budget"]["secondary_nodes"], 3)
        # Service order is deliberate descending IDs, not lexical normalization.
        self.assertEqual(result.loci.locus_id.tolist(), [l.locus_id for l in dense_loci()])

    def test_multiple_levels_time_identical_and_spatial_identical(self):
        for axis in ("time", "ra", "dec"):
            with self.subTest(axis=axis):
                _, result = self.run_query(dense_loci(axis=axis))
                self.assertTrue(result.clean)
                self.assertEqual(len(result.loci), 50)
                if axis != "time":
                    depths = [e["p2_event"]["node"]["depth"] for e in result.evidence.details["p2_events"]
                              if e["p2_event"]["node"]["depth"] is not None]
                    self.assertGreater(max(depths), 1)

    def test_depth_parent_night_and_attempt_budgets_fail_closed(self):
        for changes, reason in (({"max_depth": 0}, "p2_depth_exhausted"),
                ({"max_nodes_per_root": 1}, "p2_root_nodes_exhausted"),
                ({"max_nodes_per_night": 1}, "p2_night_nodes_exhausted"),
                ({"max_search_attempts": 1}, "p2_search_budget_exhausted")):
            with self.subTest(changes=changes):
                _, result = self.run_query(dense_loci(), test_profile(**changes))
                self.assertFalse(result.clean)
                self.assertEqual(result.evidence.errors[0].code, reason)
                with self.assertRaises(Exception):
                    result.require_completed()

    def test_inseparable_50_and_51_never_accept_saturation(self):
        for count in (50, 51):
            _, result = self.run_query(dense_loci(colocated=True, count=count))
            self.assertFalse(result.clean)
            self.assertEqual(result.evidence.errors[0].code, "p2_depth_exhausted")
            self.assertEqual(result.evidence.details["returned_loci"], 0)

    def test_partial_malformed_and_retry_exhaustion(self):
        def broken(_body):
            yield FakeLocus("out-of-domain", mjd=0.)
        _, malformed = self.run_query(search=broken)
        self.assertEqual(malformed.evidence.errors[0].code, "p2_malformed")
        def outage(_body):
            raise ConnectionError("never retain this secret exception text")
        _, exhausted = self.run_query(search=outage)
        self.assertEqual(exhausted.evidence.errors[0].code, "p2_retry_exhausted")
        self.assertNotIn("secret", json.dumps(exhausted.evidence.details["p2_events"]))

    def test_retry_discards_partial_rows_and_succeeds(self):
        calls = 0
        def transient(body):
            nonlocal calls
            calls += 1
            if calls == 1:
                first = L._make_initial_tiles(self.request.mjd_min, self.request.mjd_max)[0]
                yield FakeLocus("partial", mjd=(first["mjd_min"]+first["mjd_max"])/2,
                                ra=(first["ra_min"]+first["ra_max"])/2,
                                dec=(first["dec_min"]+first["dec_max"])/2)
                raise ConnectionError("transient")
            return
        _, result = self.run_query(search=transient)
        self.assertTrue(result.clean)
        self.assertEqual(result.evidence.details["retry_count"], 1)
        self.assertEqual(result.evidence.details["partial_rows_discarded"], 1)
        self.assertTrue(result.loci.empty)

    def test_exact_split_boundaries_and_midpoint_collapse(self):
        import math
        tile = floor_tile()
        axis, midpoint, children = L._p2_split(tile)
        self.assertEqual(axis, "time")
        self.assertEqual(children[0]["mjd_max"], children[1]["mjd_min"])
        for low, high in (("mjd_min", "mjd_max"), ("ra_min", "ra_max"), ("dec_min", "dec_max")):
            self.assertEqual(children[0][low], tile[low])
            self.assertEqual(children[1][high], tile[high])
        tiny = {"mjd_min": 1., "mjd_max": math.nextafter(1., 2.), "ra_min": 0.,
                "ra_max": math.nextafter(0., 1.), "dec_min": 0., "dec_max": math.nextafter(0., 1.)}
        with self.assertRaises(ValueError):
            L._p2_split(tiny)

    def test_independent_verifier_rejects_coherent_decision_corruption(self):
        _, result = self.run_query(dense_loci())
        for mutation in ("query", "budget", "child", "exhaustion", "profile", "duplicate", "bool"):
            with self.subTest(mutation=mutation):
                details = copy.deepcopy(result.evidence.details)
                events = details["p2_events"]
                if mutation == "query": events[0]["p2_event"]["query_sha256"] = "0" * 64
                if mutation == "budget": events[0]["p2_event"]["after"]["search_attempts"] = 100
                if mutation == "profile": events[0]["p2_event"]["profile_sha256"] = "0" * 64
                if mutation == "bool": events[0]["p2_event"]["after"]["search_attempts"] = True
                if mutation == "duplicate": events.insert(1, copy.deepcopy(events[0]))
                if mutation == "child":
                    decision = next(e["p2_event"]["decision"] for e in events if e["p2_event"].get("decision", {}).get("children"))
                    decision["children"].pop()
                if mutation == "exhaustion":
                    decision = next(e["p2_event"]["decision"] for e in events if e["p2_event"].get("decision", {}).get("status") == "accepted_exhausted")
                    decision["iterator_exhausted"] = False
                corrupted = replace(result, evidence=replace(result.evidence, details=details))
                with self.assertRaises(S.ArtifactValidationError):
                    S.validate_p2_query_result(self.request, corrupted, test_profile())

    def test_supported_crash_boundaries_resume_exact_science(self):
        loci, profile = dense_loci(), test_profile()
        reference = provider_for(self.root, loci, profile).query(self.request)
        class Crash(BaseException): pass
        for target in ("reserve", "entry", "split", "accepted"):
            with self.subTest(target=target):
                root = self.root.parent / target / self.root.name
                root.mkdir(mode=0o700, parents=True)
                provider = provider_for(root, loci, profile)
                binding = bindings(provider, root, self.request)
                fired = False
                def crash(_name, details):
                    nonlocal fired
                    meta = details["event"]["p2_event"]
                    decision = meta.get("decision", {})
                    match = (meta["kind"] == "reserve" if target == "reserve" else
                        decision.get("entered_secondary") if target == "entry" else
                        decision.get("status") == ("split_saturated" if target == "split" else "accepted_exhausted"))
                    if match and not fired:
                        fired = True
                        raise Crash()
                with self.assertRaises(Crash):
                    provider.query_resumable(self.request, binding, event_hook=crash)
                resumed = provider_for(root, loci, profile).query_resumable(self.request, binding)
                self.assertTrue(resumed.clean)
                pd.testing.assert_frame_equal(resumed.loci, reference.loci)
                self.assertEqual(resumed.evidence.details["tile_trace_sha256"], reference.evidence.details["tile_trace_sha256"])
                S.validate_p2_query_result(self.request, resumed, profile)

    def test_crash_reserve_never_resets(self):
        profile = test_profile(crash_reserve=0)
        provider = provider_for(self.root, [], profile)
        binding = bindings(provider, self.root, self.request)
        class Crash(BaseException): pass
        def crash(_name, _details): raise Crash()
        with self.assertRaises(Crash):
            provider.query_resumable(self.request, binding, event_hook=crash)
        result = provider_for(self.root, [], profile).query_resumable(self.request, binding)
        self.assertEqual(result.evidence.errors[0].code, "p2_crash_reserve_exhausted")
        again = provider_for(self.root, [], profile).query_resumable(self.request, binding)
        self.assertEqual(result.evidence.details["p2_events"], again.evidence.details["p2_events"])



    def test_malformed_object_lsst_and_close_errors_are_durable_non_science(self):
        for value in (object(), FakeLocus("non-lsst", mjd=61218.01, ra=1., dec=-89., survey={"ztf": {"id": "Z"}})):
            calls = []
            def search(_body):
                calls.append(True)
                return [value] if len(calls) == 1 else []
            _, result = self.run_query(search=search)
            self.assertFalse(result.clean)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.evidence.errors[0].code, "p2_malformed")
        class ClosingIterator:
            def __init__(self, items): self.items = iter(items)
            def __iter__(self): return self
            def __next__(self): return next(self.items)
            def close(self): raise RuntimeError("private cleanup body")
        for count in (0, 1, 50):
            root = self.root.parent / ("close-"+str(count)) / self.root.name
            root.mkdir(parents=True)
            first = L._make_initial_tiles(61218., 61219.)[0]
            rows = [FakeLocus(str(n), mjd=61218.001, ra=1., dec=-89.) for n in range(count)]
            profile = test_profile()
            provider = provider_for(root, rows, profile, search=lambda _: ClosingIterator(rows))
            binding = bindings(provider, root, self.request)
            result = provider.query_resumable(self.request, binding)
            self.assertFalse(result.clean)
            decision = result.evidence.details["p2_events"][-1]["p2_event"]["decision"]
            self.assertEqual(decision["validated_rows"], count)
            self.assertEqual(decision["iterator_exhausted"], count < 50)
            self.assertFalse(decision["retryable"])
            resumed = provider_for(root, [], profile).query_resumable(self.request, binding)
            self.assertEqual(resumed.evidence.details["p2_events"], result.evidence.details["p2_events"])
            self.assertEqual(resumed.evidence.details["p2_budget"]["unknown_attempts"], 0)
            self.assertNotIn("private", json.dumps(decision))

    def test_all_record_validation_failures_are_terminal_across_restart(self):
        class BadString:
            def __str__(self): raise OverflowError("private identifier body")
        cyclic = {}
        cyclic["self"] = cyclic
        mutations = (("newest_alert_observation_time", 10**500), ("ra", 10**500),
                     ("dec", 10**500), ("locus_id", BadString()),
                     ("survey", {"lsst": {"dia_object_id": BadString()}}),
                     ("payload", cyclic))
        for ordinal, (field, value) in enumerate(mutations):
            with self.subTest(field=field):
                root = self.root.parent / ("malformed-"+str(ordinal)) / self.root.name
                root.mkdir(parents=True)
                row = FakeLocus("valid", mjd=61218.001, ra=1., dec=-89.)
                row.properties[field] = value
                calls = []
                def search(_body):
                    calls.append(True)
                    return [row] if len(calls) == 1 else []
                profile = test_profile()
                provider = provider_for(root, [], profile, search=search)
                binding = bindings(provider, root, self.request)
                result = provider.query_resumable(self.request, binding)
                self.assertFalse(result.clean)
                self.assertEqual(result.evidence.errors[0].code, "p2_malformed")
                self.assertEqual(len(calls), 1)
                decision = result.evidence.details["p2_events"][-1]["p2_event"]["decision"]
                self.assertFalse(decision["retryable"])
                self.assertEqual(decision["validated_rows"], 0)
                self.assertEqual(decision["exception_type"], "builtins.ValueError")
                resumed = provider_for(root, [], profile, search=search).query_resumable(self.request, binding)
                self.assertFalse(resumed.clean)
                self.assertEqual(resumed.evidence.details["p2_events"], result.evidence.details["p2_events"])
                self.assertEqual(resumed.evidence.details["p2_budget"]["unknown_attempts"], 0)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("private", json.dumps(result.evidence.details["p2_events"]))

    def test_raising_close_property_is_known_terminal_not_unknown(self):
        class ClosingIterator:
            def __init__(self, rows): self.rows = iter(rows)
            def __iter__(self): return self
            def __next__(self): return next(self.rows)
            @property
            def close(self): raise OverflowError("private close-property body")
        for count in (0, 1, 50):
            with self.subTest(count=count):
                root = self.root.parent / ("close-property-"+str(count)) / self.root.name
                root.mkdir(parents=True)
                rows = [FakeLocus(str(n), mjd=61218.001, ra=1., dec=-89.) for n in range(count)]
                calls = []
                def search(_body):
                    calls.append(True)
                    return ClosingIterator(rows)
                profile = test_profile()
                provider = provider_for(root, rows, profile, search=search)
                binding = bindings(provider, root, self.request)
                result = provider.query_resumable(self.request, binding)
                self.assertFalse(result.clean)
                self.assertEqual(result.evidence.errors[0].code, "p2_malformed")
                self.assertEqual(len(calls), 1)
                decision = result.evidence.details["p2_events"][-1]["p2_event"]["decision"]
                self.assertEqual(decision["validated_rows"], count)
                self.assertEqual(decision["iterator_exhausted"], count < 50)
                self.assertFalse(decision["retryable"])
                resumed = provider_for(root, [], profile, search=search).query_resumable(self.request, binding)
                self.assertEqual(resumed.evidence.details["p2_events"], result.evidence.details["p2_events"])
                self.assertEqual(resumed.evidence.details["p2_budget"]["unknown_attempts"], 0)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("private", json.dumps(decision))

    def test_exact_partition_rational_volumes_boundaries_poles_and_keep_last(self):
        from fractions import Fraction
        tile = floor_tile()
        rows = dense_loci()
        axis, middle, _ = L._p2_split(tile)
        rows[25].properties["newest_alert_observation_time"] = middle
        rows[0].locus_id = rows[-1].locus_id = "DUP"
        rows[-1].properties["brightest_alert_magnitude"] = 17.
        rows += [FakeLocus("south", mjd=61218., ra=0., dec=-90.),
                 FakeLocus("north", mjd=61218.999, ra=359.999, dec=90.)]
        _, result = self.run_query(rows)
        self.assertTrue(result.clean)
        self.assertEqual(result.loci.loc[result.loci.locus_id == "DUP", "brightest_alert_magnitude"].tolist(), [17.])
        self.assertIn("north", result.loci.locus_id.tolist())
        self.assertIn("south", result.loci.locus_id.tolist())
        def volume(box):
            value = Fraction(1)
            for lo, hi in (("mjd_min","mjd_max"),("ra_min","ra_max"),("dec_min","dec_max")):
                value *= Fraction(box[hi]) - Fraction(box[lo])
            return value
        leaves = []
        for event in result.evidence.details["p2_events"]:
            meta = event["p2_event"]
            decision = meta.get("decision", {})
            if decision.get("children"):
                boxes = [child["tile"] for child in decision["children"]]
                self.assertEqual(sum(map(volume, boxes)), volume(meta["node"]["tile"]))
                self.assertTrue(all(volume(box) > 0 for box in boxes))
            if decision.get("status") == "accepted_exhausted":
                leaves.append(meta["node"]["tile"])
        self.assertEqual(sum(map(volume, leaves)), Fraction(1)*360*180)

    def test_complete_evidence_policy_reservations_and_records_refuse_corruption(self):
        _, result = self.run_query([FakeLocus("valid", mjd=61218.001, ra=1., dec=-89.)])
        changes = {"completion_classification": "INCOMPLETE", "coverage_lineage_complete": False,
                   "all_accepted_iterators_exhausted": False, "iterator_exhausted": False,
                   "unresolved_saturated_tile_count": 1, "processed_tile_count": 1,
                   "logical_chunk_count": True, "query_sha256": "0"*64, "capability_environment": "arnor-commissioning"}
        for key, value in changes.items():
            with self.subTest(key=key):
                details = copy.deepcopy(result.evidence.details)
                details[key] = value
                with self.assertRaises(S.ArtifactValidationError):
                    S.validate_p2_query_result(self.request, replace(result, evidence=replace(result.evidence, details=details)), test_profile())
        for mutation in ("reservation-bool", "policy-bool", "record"):
            details = copy.deepcopy(result.evidence.details)
            if mutation == "reservation-bool": details["p2_events"][1]["p2_event"]["reservation"]["id"] = True
            if mutation == "policy-bool": details["execution_policy"]["parallel_parent_shards"] = True
            if mutation == "record":
                from src.operations.query_progress import decode_records, encode_records
                accepted = details["p2_events"][1]
                records = decode_records(accepted["records"])
                records[0]["survey"] = {"ztf": {"id": "Z"}}
                accepted["records"] = encode_records(records)
            with self.assertRaises(S.ArtifactValidationError):
                S.validate_p2_query_result(self.request, replace(result, evidence=replace(result.evidence, details=details)), test_profile())
        for field, value in (("provider_name", "wrong"), ("scenario", "wrong"), ("outcome", S.ProviderOutcome.SUCCESS_ZERO)):
            with self.assertRaises(S.ArtifactValidationError):
                S.validate_p2_query_result(self.request, replace(result, **{field:value}), test_profile())

    def test_persisted_bounds_record_byte_limits_and_p1_grammar_refusal(self):
        for changes in ({"max_search_attempts": 1}, {"max_depth": 0},
                        {"max_nodes_per_root": 1}, {"max_nodes_per_night": 1}):
            root = self.root.parent / next(iter(changes)) / self.root.name
            root.mkdir(parents=True)
            profile = test_profile(**changes)
            provider = provider_for(root, dense_loci(), profile)
            binding = bindings(provider, root, self.request)
            failed = provider.query_resumable(self.request, binding)
            repeated = provider_for(root, [], profile).query_resumable(self.request, binding)
            self.assertFalse(repeated.clean)
            self.assertEqual(failed.evidence.details["p2_events"], repeated.evidence.details["p2_events"])
            self.assertEqual(failed.evidence.details["p2_budget"], repeated.evidence.details["p2_budget"])
        huge = FakeLocus("huge", mjd=61218.001, ra=1., dec=-89.)
        huge.properties["oversize"] = "X"*5000
        _, failed = self.run_query([huge], test_profile(max_event_bytes=4000))
        self.assertFalse(failed.clean)
        with self.assertRaises(Q.QueryCheckpointError):
            provider_for(self.root, [], test_profile(max_event_bytes=1)).query(self.request)
        with self.assertRaises(Q.QueryCheckpointError):
            provider_for(self.root, [], test_profile()).query(self.request,
                _progress=types.SimpleNamespace(events=[{"trace":{},"records":[]}], commit=lambda _:None))


    def test_stale_head_commit_boundary_replays_unknown_within_reserve(self):
        from src.operations.query_progress import QueryProgress
        profile = test_profile()
        provider = provider_for(self.root, [], profile)
        binding = bindings(provider, self.root, self.request)
        original = QueryProgress._commit_head
        class Crash(BaseException): pass
        def stop_after_event(progress):
            if progress.events:
                raise Crash()
            original(progress)
        with mock.patch.object(QueryProgress, "_commit_head", stop_after_event):
            with self.assertRaises(Crash):
                provider.query_resumable(self.request, binding)
        resumed = provider_for(self.root, [], profile).query_resumable(self.request, binding)
        self.assertTrue(resumed.clean)
        self.assertEqual(resumed.evidence.details["p2_budget"]["unknown_attempts"], 1)
        self.assertEqual(resumed.evidence.details["search_request_count"], 6913)
        S.validate_p2_query_result(self.request, resumed, profile)

    def test_pinned_client_empty_and_cyclic_pages_are_not_bounded_by_search_calls(self):
        import importlib
        api = importlib.import_module("antares_client._api.api")
        search_module = importlib.import_module("antares_client.search")
        class SafetyStop(RuntimeError): pass
        class Response:
            status_code = 200
            def __init__(self, next_url): self.next_url = next_url
            def json(self): return {"data": [], "links": {"next": self.next_url}}
        page = lambda key: L.OFFICIAL_API_BASE_URL + "loci?page=" + key
        for pattern in ([page("same")]*6, [page("a"), page("b")]*3,
                        [page(str(n)) for n in range(6)]):
            pages = []
            def get(url, **_kwargs):
                pages.append(url)
                if len(pages) == 6:
                    raise SafetyStop("test-only finite stop")
                return Response(pattern[len(pages)-1])
            # No actual HTTP call. Five empty pages occur inside one next().
            with mock.patch.object(api.requests, "get", side_effect=get):
                iterator = search_module.search({"query": {"match_all": {}}})
                with self.assertRaises(SafetyStop):
                    next(iterator)
            self.assertEqual(len(pages), 6)


    def test_p2_artifact_metadata_and_dedup_survivor_are_independently_strict(self):
        profile = test_profile()
        provider, query_result = self.run_query(dense_loci(), profile)
        fetched = provider.fetch(self.request, query_result)
        with mock.patch.object(S, "QUALIFIED_P2_PROFILES", (profile,)):
            artifacts = S.build_night_artifacts(fetched)
            S.reopen_and_validate_artifacts(artifacts)
            for mutation in ("policy", "method", "client", "interval", "coverage"):
                modified = dict(artifacts)
                manifest = json.loads(modified["manifest.json"])
                details = manifest["query_evidence"]["details"]
                if mutation == "policy": details["execution_policy"]["parallel_parent_shards"] = True
                if mutation == "method": details["extraction_method"]["proof_profile"]["max_depth"] = True
                if mutation == "client": details["client"]["api_timeout_seconds"] = 60.0
                if mutation == "interval": details["spatial_domain"]["ra_min"] = False
                if mutation == "coverage": details["coverage_lineage_complete"] = False
                modified["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
                with self.assertRaises(S.ArtifactValidationError):
                    S.reopen_and_validate_artifacts(modified)
            # A coherent parquet change cannot replace the independently proved winner.
            import io
            changed = fetched.loci.copy()
            changed.loc[0, "brightest_alert_magnitude"] = 99.
            stream = io.BytesIO()
            changed.to_parquet(stream, index=False)
            modified = dict(artifacts)
            modified["loci.parquet"] = stream.getvalue()
            with self.assertRaises(S.ArtifactValidationError):
                S.reopen_and_validate_artifacts(modified)

if __name__ == "__main__":
    unittest.main()
