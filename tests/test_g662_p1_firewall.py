"""Immutable P1 firewall and synthetic mixed-profile qualification only."""
import ast
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pandas as pd
import v3_fixtures as F
import test_g64b_saved_acquisition_adoption as G64
from test_operations_phase6 import FakeLocus, _body_matches_locus
from test_g662_p2_proof import test_profile, dense_loci
from src.operations import live_antares as L, science as S, backfill as B, production_range as R
from src.operations import query_checkpoint as Q
from src.operations.fetch_checkpoint import SegmentedFetchCheckpoint

BASELINE = "812c545e14693cdce7ff7458f1d2b50b0804dcd8"
BASELINE_PROVIDER_SHA = "afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c"
ROOT = Path(__file__).resolve().parents[1]


def baseline_bytes(path):
    return subprocess.check_output(["git", "show", BASELINE + ":" + path], cwd=ROOT)


def baseline_provider():
    payload = baseline_bytes("src/operations/live_antares.py")
    if hashlib.sha256(payload).hexdigest() != BASELINE_PROVIDER_SHA:
        raise AssertionError("Immutable baseline provider differs.")
    name = "src.operations._g662_frozen_provider"
    module = types.ModuleType(name)
    module.__package__ = "src.operations"
    module.__file__ = str(ROOT / "src/operations/live_antares.py")
    sys.modules[name] = module
    exec(compile(payload, "<immutable-812c545-provider>", "exec"), module.__dict__)
    return module


def make_provider(module, root, request, loci, **kwargs):
    cap = module.LiveAntaresReadCapability.for_local_mock(
        root, run_id=root.name, target_date_utc=request.date_utc,
        release_sha="a" * 40, authority=module.LIVE_ANTARES_READ)
    lookup = {locus.locus_id: locus for locus in loci}
    return module.LiveAntaresProvider(cap,
        search_fn=lambda body: [locus for locus in loci if _body_matches_locus(body, locus)],
        get_by_id_fn=lookup.__getitem__, connectivity_fn=lambda: [],
        clock=F.fixed_clock, monotonic=lambda: 0., sleeper=lambda _: None, **kwargs)


class P1FirewallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = baseline_provider()

    def test_frozen_helpers_are_exact_immutable_source(self):
        for path, names in (
            ("src/operations/live_antares.py", (
                "extraction_method_contract", "_scientific_query_contract", "_make_initial_tiles",
                "_split_tile", "_build_tile_query", "_record_matches_tile")),
            ("src/operations/science.py", (
                "_phase6_initial_tiles", "_phase6_split_tile", "_phase6_tile_query", "_phase6_replay_trace"))):
            old = baseline_bytes(path).decode()
            new = (ROOT / path).read_text()
            old_nodes = {n.name: n for n in ast.parse(old).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            new_nodes = {n.name: n for n in ast.parse(new).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name in names:
                self.assertEqual(ast.get_source_segment(old, old_nodes[name]),
                                 ast.get_source_segment(new, new_nodes[name]), name)

    def test_full_contract_policy_grid_and_split_goldens(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "night-2026-06-27"
            root.mkdir()
            for day in ("2026-06-27", "2026-07-08", "2000-02-29", "2016-12-31"):
                request = R.LiveRangeAdapter(root, "a"*40, None).acquisition_request(day)
                old = make_provider(self.old, root, request, [])
                new = make_provider(L, root, request, [])
                self.assertEqual(B._canonical(old.execution_policy()), B._canonical(new.execution_policy()))
                self.assertEqual(B._canonical(old.scientific_contract(request)), B._canonical(new.scientific_contract(request)))
                tiles = self.old._make_initial_tiles(request.mjd_min, request.mjd_max)
                self.assertEqual(len(tiles), 6912)
                self.assertEqual(B._canonical(tiles), B._canonical(L._make_initial_tiles(request.mjd_min, request.mjd_max)))
                tile = tiles[0]
                while True:
                    children = self.old._split_tile(tile)
                    self.assertEqual(children, L._split_tile(tile))
                    if not children:
                        break
                    tile = children[0]
            ties = {"mjd_min": 0., "mjd_max": 30./86400., "ra_min": 0., "ra_max": .05,
                    "dec_min": 0., "dec_max": .05}
            self.assertEqual(self.old._split_tile(ties), L._split_tile(ties))

    def test_query_events_replay_order_keep_last_and_fetch_are_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "night-2026-06-27"
            root.mkdir()
            request = R.LiveRangeAdapter(root, "a"*40, None).acquisition_request("2026-06-27")
            loci = [FakeLocus("Z", ra=10., lightcurve=pd.DataFrame()),
                    FakeLocus("A", ra=10., lightcurve=pd.DataFrame()),
                    FakeLocus("Z", ra=250., lightcurve=pd.DataFrame())]
            traces = []
            results = []
            for module in (self.old, L):
                provider = make_provider(module, root, request, loci)
                events = []
                result = provider.query(request, _progress=types.SimpleNamespace(
                    events=[], commit=lambda event: events.append(copy.deepcopy(event))))
                replay = provider.query(request, _progress=types.SimpleNamespace(events=events,
                    commit=lambda _: self.fail("Complete replay attempted commit.")))
                pd.testing.assert_frame_equal(result.loci, replay.loci)
                self.assertEqual(result.loci.locus_id.tolist(), ["A", "Z"])
                self.assertEqual(result.loci.ra.tolist(), [10., 250.])
                fetched = provider.fetch(request, result)
                self.assertTrue(fetched.fetch_evidence.clean)
                results.append((result, fetched))
                traces.append(events)
            self.assertEqual(traces[0], traces[1])
            pd.testing.assert_frame_equal(results[0][0].loci, results[1][0].loci)
            self.assertEqual(results[0][0].evidence, results[1][0].evidence)
            pd.testing.assert_frame_equal(results[0][1].loci, results[1][1].loci)
            self.assertEqual(results[0][1].fetch_evidence, results[1][1].fetch_evidence)
            for module in (self.old, L):
                with self.assertRaises(Q.QueryCheckpointError):
                    make_provider(module, root, request, []).query(request, _progress=types.SimpleNamespace(
                        events=[{"p2_event": {}, "records": []}], commit=lambda _: None))

    def test_candidate_has_no_production_attestation_or_default_p2_registry(self):
        adapter = R.LiveRangeAdapter(ROOT, "a"*40, None)
        with self.assertRaises(B.BackfillRefused):
            B.require_prior_free_acquisition(B.acquisition_identity(adapter), B.RANGE_PRIOR_FREE_ACQUISITION_ATTESTATIONS)
        self.assertEqual(S.QUALIFIED_P2_PROFILES, ())
        self.assertEqual(len(B.QUALIFIED_SOURCE_PROFILES), 1)
        self.assertIsNone(B.QUALIFIED_SOURCE_PROFILES[0].proof_profile)
        self.assertEqual(B.QUALIFIED_SOURCE_PROFILES[0].provider_sha256s, B.ADOPTABLE_SOURCE_PROVIDERS)

    def test_selection_descriptor_canonical_golden_and_incompatible_intent(self):
        request = R.LiveRangeAdapter(ROOT, "a"*40, None).acquisition_request("2026-06-27")
        descriptor = B.selection_descriptor_for_request(request)
        # Literal independent expected canonical object; no runtime extractor fields.
        expected = {
            "schema_version": "v3.qualified-scientific-selection.v1", "date_utc": "2026-06-27",
            "time": {"field": "properties.newest_alert_observation_time", "mjd_min": 61218., "mjd_max": 61219.,
                     "lower": "inclusive", "upper": "exclusive", "timezone": "UTC"},
            "spatial": {"ra_field": "ra", "dec_field": "dec", "units": "degrees", "ra_min": 0., "ra_max": 360.,
                        "ra_lower": "inclusive", "ra_upper": "exclusive", "dec_min": -90., "dec_max": 90.,
                        "dec_lower": "inclusive", "dec_upper": "inclusive_at_90_only"},
            "lsst_filter": {"bool": {"should": [{"exists": {"field": "properties.survey.lsst.dia_object_id"}},
                                               {"exists": {"field": "properties.survey.lsst.ss_object_id"}}],
                                    "minimum_should_match": 1}},
            "query_tag": None, "lsst_only": True, "target_loci": None, "prior_free": True,
            "normalization": "locus_to_record-properties-overlay;string-strip-nonblank-locus-id;tile-membership",
            "deduplication": {"key": "locus_id", "keep": "last", "scope": "accepted_tiles"},
            "input_order": "lower-child-first;within-leaf-qualified-service-order;keep-last;reset-index",
            "equivalence": "selection-semantics-only-not-service-snapshot"}
        canonical = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(B._canonical(descriptor), canonical)
        self.assertEqual(B.selection_descriptor_identity(descriptor)["selection_descriptor_sha256"], hashlib.sha256(canonical).hexdigest())
        for changes in ({"query_tag": "x"}, {"target_loci": 1}, {"mjd_max": request.mjd_max+.5},
                        {"prior_locus_ids": ("prior",)}, {"lsst_only": False}):
            with self.assertRaises((B.BackfillRefused, ValueError)):
                B.selection_descriptor_for_request(replace(request, **changes))


class MixedProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = G64.SavedAcquisitionAdoptionTests("test_saved_canary_and_range_acquisitions_publish_in_order_without_antares")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.profile = test_profile()
        identity = B.acquisition_identity(self.fixture.adapter)
        self.p2_source_profile = B.SourceProfile("explicit-P2-test", frozenset({identity["provider_implementation_sha256"]}), self.profile)
        self.fixture.stack.enter_context(mock.patch.object(B, "QUALIFIED_SOURCE_PROFILES",
            (*B.QUALIFIED_SOURCE_PROFILES, self.p2_source_profile)))
        self.fixture.stack.enter_context(mock.patch.object(S, "QUALIFIED_P2_PROFILES", (self.profile,)))

    def p2_source(self, day):
        f = self.fixture
        root = f.canary / ("g662-p2-" + day)
        root.mkdir(mode=0o700)
        adapter = R.LiveRangeAdapter(root, G64.CANARY_RELEASE, None, proof_profile=self.profile)
        request = adapter.acquisition_request(day)
        loci = dense_loci()
        shift = request.mjd_min - 61218.
        for locus in loci:
            locus.locus_id = day + "-" + locus.locus_id
            locus.properties["newest_alert_observation_time"] += shift
            locus.lightcurve = pd.DataFrame({"mjd": [locus.properties["newest_alert_observation_time"]],
                "ztf_magpsf": [20.], "ztf_sigmapsf": [.1], "ztf_fid": [1]})
        cap = F.mock_read_capability(root, root.name, day, G64.CANARY_RELEASE)
        provider = make_provider(L, root, request, loci, proof_profile=self.profile)
        # Capability release must equal the actual source binding.
        provider.capability = cap
        B._write_json_new(root/"request.json", B._request_document(request))
        configuration = B.configuration_sha256(G64.CANARY_RELEASE, adapter, f.settings.segment_size)
        bindings = B.query_checkpoint_bindings(root.name, G64.CANARY_RELEASE, configuration, adapter, request)
        result = provider.query_resumable(request, bindings).require_completed()
        Q.seal_query_result_checkpoint(root, result, bindings)
        loaded = Q.load_query_result_checkpoint(root, request, bindings)
        binding = B.fetch_checkpoint_binding(root.name, G64.CANARY_RELEASE, configuration, adapter, request, loaded, f.settings.segment_size)
        SegmentedFetchCheckpoint.open(cap, binding).fetch_missing(
            B.BackfillController._ordered_ids(loaded), lambda ids: provider.fetch_segment(request, ids))
        return root

    def test_seven_night_mixed_adoption_constructs_qualifies_and_publishes_only_synthetic_roots(self):
        days = tuple(["2026-06-28", "2026-06-29", "2026-06-30"] + ["2026-07-0"+str(n) for n in range(1,5)])
        f = self.fixture
        sources = {day: self.p2_source(day) if day == days[1] else f.range_source(day) for day in days}
        before = {day: G64.tree_digest(root) for day, root in sources.items()}
        with f.no_network():
            saved = {day: B.describe_saved_acquisition(root, day, f.adapter, f.settings.segment_size) for day, root in sources.items()}
            for day, verified in saved.items():
                self.assertEqual(verified.selection_descriptor, B.selection_descriptor_for_request(f.adapter.acquisition_request(day)))
                self.assertEqual(set(verified.entry), B._ADOPTION_FIELDS)
                self.assertEqual(verified.source_profile.proof_profile is not None, day == days[1])
            authority, publisher = f.adoption_authority(days[0], days[-1], sources)
            f.queried.clear(); f.fetched.clear()
            controller = f.controller(publisher, authority.scope)
            result = controller.run(days[0], days[-1])
        self.assertEqual([night["stage"] for night in result["nights"]], ["PUBLISHED"]*7, result)
        self.assertEqual((f.queried, f.fetched), ([], []))
        self.assertEqual({day: G64.tree_digest(root) for day, root in sources.items()}, before)
        for day in days:
            record = json.loads((controller.workspace(day).candidate/"candidate-record.json").read_text())
            provenance = record["provenance"]
            self.assertEqual(provenance["acquisition_source"], saved[day].entry)
            self.assertEqual(provenance["night_query_contract_sha256"], saved[day].entry["query_contract_sha256"])
            self.assertEqual({key: provenance[key] for key in B.selection_descriptor_identity(saved[day].selection_descriptor)},
                             B.selection_descriptor_identity(saved[day].selection_descriptor))
            if day == days[1]:
                self.assertNotEqual(provenance["night_query_contract_sha256"],
                    B._sha256(B._canonical(f.adapter.scientific_contract(f.adapter.acquisition_request(day)))))
        self.assertEqual(f.published()[-7:], days)

    def test_unknown_ambiguous_profiles_and_descriptor_cannot_rescue_source(self):
        f = self.fixture
        day = "2026-06-28"
        root = f.range_source(day)
        with mock.patch.object(B, "QUALIFIED_SOURCE_PROFILES", ()):
            with self.assertRaises(B.BackfillRefused):
                B.describe_saved_acquisition(root, day, f.adapter, f.settings.segment_size)
        current = next(p for p in B.QUALIFIED_SOURCE_PROFILES if p.name == "candidate-P1-test")
        with mock.patch.object(B, "QUALIFIED_SOURCE_PROFILES", (*B.QUALIFIED_SOURCE_PROFILES, current)):
            with self.assertRaises(B.BackfillRefused):
                B.describe_saved_acquisition(root, day, f.adapter, f.settings.segment_size)
        manifest = root/"checkpoints/query-result/manifest.json"
        # Existing strict corruption matrix tests the actual checkpoint paths.
        journal = root/"checkpoints/query-progress-v2/event-00000001.json"
        G64.flip_last_byte(journal)
        with self.assertRaises(Exception):
            B.describe_saved_acquisition(root, day, f.adapter, f.settings.segment_size)

    def test_narrow_unadopted_full_contract_and_mixed_descriptor_guards(self):
        f = self.fixture
        request = f.adapter.acquisition_request("2026-06-28")
        contract = B._sha256(B._canonical(f.adapter.scientific_contract(request)))
        descriptor = B.selection_descriptor_identity(B.selection_descriptor_for_request(request))
        candidate = types.SimpleNamespace(date_utc=request.date_utc,
            provenance={"night_query_contract_sha256": contract, **descriptor})
        R.qualify_candidate_acquisition(f.adapter, candidate)
        candidate.provenance["night_query_contract_sha256"] = "0"*64
        with self.assertRaises(B.BackfillRefused):
            R.qualify_candidate_acquisition(f.adapter, candidate)
        root = self.p2_source(request.date_utc)
        saved = B.describe_saved_acquisition(root, request.date_utc, f.adapter, f.settings.segment_size)
        candidate.provenance = {"night_query_contract_sha256": saved.entry["query_contract_sha256"], **descriptor}
        R.qualify_candidate_acquisition(f.adapter, candidate, saved.entry, saved=saved)
        for key in descriptor:
            bad = types.SimpleNamespace(date_utc=request.date_utc, provenance={**candidate.provenance})
            bad.provenance.pop(key)
            with self.assertRaises(B.BackfillRefused):
                R.qualify_candidate_acquisition(f.adapter, bad, saved.entry, saved=saved)
        for values in (frozenset(), frozenset({"*"}), {"a"*64}):
            with self.assertRaises(B.BackfillRefused):
                B.SourceProfile("bad", values)

if __name__ == "__main__":
    unittest.main()
