"""G5 local qualification: no service calls or configured production mutation."""
import contextlib
import dataclasses
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import pandas as pd
from astropy.time import Time

import v3_fixtures as F
from test_operations_phase6 import FakeLocus, _body_matches_locus
from src.operations import backfill as B, publication as P, production_range as R
from src.operations.live_antares import (
    LIVE_ANTARES_READ, LiveAntaresProvider, LiveAntaresReadCapability, LiveCapabilityError,
    night_mjd_interval,
)
from src.operations.query_checkpoint import QueryCheckpointError, QueryResultCheckpointBindings
from src.operations.science import NightScienceRequest, ProviderContractError, build_night_artifacts
from src import history
from src.operations.storage import PublicationRoots, RangeWorkCapability, StorageContractError
from src.operations import storage as S
from src.operations.writer import InjectedWriterFailure

NIGHT = "2026-06-28"
RELEASE = F.CANDIDATE_RELEASE


def request_for(night=NIGHT):
    return NightScienceRequest(night, *night_mjd_interval(night), target_loci=None)


def provider_fixture(root, mode=None, calls=None):
    request = request_for()
    loci = [FakeLocus(f"ANT-{index % 30:03d}", mjd=request.mjd_min + .5,
                     ra=10.0 if index < 30 else 190.0,
                     survey={"lsst": {"dia_object_id": f"DIA-{index}"},
                             "nested": {"null": None} if index % 2 else {}})
            for index in range(60)]
    capability = F.mock_read_capability(root, root.name, NIGHT, RELEASE)

    def search(body):
        tile = body["query"]["bool"]["filter"][1]["range"]["ra"]
        key = (tile["gte"], tile["lt"])
        if calls is not None:
            calls.append(key)
        if mode:
            with (root / "service-calls.jsonl").open("a") as handle:
                handle.write(json.dumps(key) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        try:
            for index, locus in enumerate(value for value in loci if _body_matches_locus(body, value)):
                if mode == "in_flight" and key == (180.0, 360.0) and index == 5:
                    os.kill(os.getpid(), signal.SIGKILL)
                yield locus
        finally:
            if mode == "before_split" and key == (0.0, 360.0):
                os.kill(os.getpid(), signal.SIGKILL)

    provider = LiveAntaresProvider(capability, search_fn=search, get_by_id_fn=lambda _: None,
        connectivity_fn=lambda: [], initial_tiles_fn=lambda minimum, maximum: [{
            "mjd_min": minimum, "mjd_max": maximum, "ra_min": 0., "ra_max": 360.,
            "dec_min": -90., "dec_max": 90.}], retry_delay_seconds=0,
        clock=F.fixed_clock, monotonic=lambda: 0.0)
    binding = QueryResultCheckpointBindings(root.name, RELEASE, "c" * 64, NIGHT,
        provider.provider_name, provider.scenario, {
            "scientific_contract": provider.scientific_contract(request),
            "execution_policy": provider.execution_policy()})
    return provider, request, binding


def crash_child(root, mode):
    root = Path(root)
    provider, request, bindings = provider_fixture(root, mode=mode)
    def hook(_, details):
        if ((mode == "after_split" and details["status"] == "split_saturated")
                or (mode == "after_terminal" and details["status"] == "accepted_exhausted")):
            os.kill(os.getpid(), signal.SIGKILL)
    provider.query_resumable(request, bindings, event_hook=hook)


class QueryProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self.tmp.name).resolve()
        self.root = self.parent / f"night-{NIGHT}"
        self.root.mkdir(mode=0o700)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sigkill_boundaries_replay_exact_science_and_skip_terminal_tiles(self):
        for mode, expected_calls in (
            ("before_split", [(0., 360.), (0., 180.), (180., 360.)]),
            ("after_split", [(0., 180.), (180., 360.)]),
            ("after_terminal", [(180., 360.)]),
            ("in_flight", [(180., 360.)]),
        ):
            with self.subTest(mode=mode):
                root = self.parent / mode / f"night-{NIGHT}"
                root.mkdir(mode=0o700, parents=True)
                env = dict(os.environ)
                env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(Path(__file__).parents[1]), env.get("PYTHONPATH", "")])
                killed = subprocess.run([sys.executable, "-c",
                    "import sys; from test_production_range_g5 import crash_child; crash_child(sys.argv[1], sys.argv[2])",
                    str(root), mode], env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(killed.returncode, -signal.SIGKILL, killed.stderr)
                calls = []
                provider, request, bindings = provider_fixture(root, calls=calls)
                resumed = provider.query_resumable(request, bindings)
                reference = provider.query(request)
                self.assertEqual(calls[:len(expected_calls)], expected_calls)
                self.assertTrue(resumed.clean)
                pd.testing.assert_frame_equal(resumed.loci, reference.loci)
                self.assertEqual(dict(resumed.evidence.details), dict(reference.evidence.details))
                calls.clear()
                repeated = provider.query_resumable(request, bindings)
                self.assertEqual(calls, [])
                pd.testing.assert_frame_equal(repeated.loci, resumed.loci)
                self.assertEqual(repeated.evidence.details, resumed.evidence.details)

    def test_corruption_and_binding_drift_fail_before_search(self):
        calls = []
        provider, request, binding = provider_fixture(self.root, calls=calls)
        provider.query_resumable(request, binding)
        calls.clear()
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, dataclasses.replace(binding, configuration_hash="d" * 64))
        self.assertEqual(calls, [])
        event = self.root / "checkpoints/query-progress-v1/event-00000001.json"
        document = json.loads(event.read_text())
        document["payload"]["event"]["records"] = []
        event.write_text(json.dumps(document))
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)
        self.assertEqual(calls, [])

    def test_missing_event_symlink_and_unknown_schema_fail_closed(self):
        provider, request, binding = provider_fixture(self.root)
        provider.query_resumable(request, binding)
        journal = self.root / "checkpoints/query-progress-v1"
        path = journal / "event-00000001.json"
        content = path.read_bytes()
        path.unlink()
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)
        target = self.parent / "event"
        target.write_bytes(content)
        path.symlink_to(target)
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)

    def test_query_progress_cannot_seal_partial_coverage(self):
        provider, request, binding = provider_fixture(self.root)
        def interrupt(_, details):
            if details["status"] == "accepted_exhausted":
                raise SystemExit("process death")
        with self.assertRaises(SystemExit):
            provider.query_resumable(request, binding, event_hook=interrupt)
        self.assertFalse((self.root / "checkpoints/query-result/COMMITTED.json").exists())

    def test_tail_truncation_and_initial_tiling_drift_fail_closed(self):
        calls = []
        provider, request, binding = provider_fixture(self.root, calls=calls)
        provider.query_resumable(request, binding)
        calls.clear()
        provider._initial_tiles_fn = __import__("src.operations.live_antares", fromlist=["_make_initial_tiles"])._make_initial_tiles
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)
        self.assertEqual(calls, [])
        provider, request, binding = provider_fixture(self.root, calls=calls)
        (self.root / "checkpoints/query-progress-v1/event-00000002.json").unlink()
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)
        self.assertEqual(calls, [])

    def test_durable_retry_evidence_resumes_at_next_attempt(self):
        provider, request, binding = provider_fixture(self.root)
        original = provider._search_fn
        failed = [False]
        def transient(body):
            ra = body["query"]["bool"]["filter"][1]["range"]["ra"]
            if ra["gte"] == 180.0 and not failed[0]:
                failed[0] = True
                yield next(iter(original(body)))
                raise ConnectionError("interrupted transport")
            yield from original(body)
        provider._search_fn = transient
        provider.sleeper = lambda _: (_ for _ in ()).throw(SystemExit("death after durable retry"))
        with self.assertRaises(SystemExit):
            provider.query_resumable(request, binding)
        calls = []
        resumed_provider, _, _ = provider_fixture(self.root, calls=calls)
        result = resumed_provider.query_resumable(request, binding)
        self.assertEqual(calls, [(180., 360.)])
        self.assertEqual(result.evidence.details["retry_count"], 1)
        self.assertEqual(result.evidence.details["tile_trace"][-1]["attempt"], 2)
        pd.testing.assert_frame_equal(result.loci, resumed_provider.query(request).loci)


class DateAndRequestTests(unittest.TestCase):
    def test_canonical_nights_match_utc_astropy(self):
        for night in ("1858-11-17", "2000-02-29", "2016-12-31", "2026-06-27", NIGHT, "2026-06-29", "2100-03-01"):
            minimum, maximum = night_mjd_interval(night)
            self.assertEqual(minimum, float(Time(night + "T00:00:00", scale="utc").mjd))
            self.assertEqual(maximum, minimum + 1.0)
        for night in ("20260628", "2026-6-28", "2026-06-28Z", "2026-02-29", "2026-06-28T00:00:00", " 2026-06-28", None):
            with self.subTest(night=night), self.assertRaises(LiveCapabilityError):
                night_mjd_interval(night)

    def test_live_capability_rejects_other_night_interval_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / f"night-{NIGHT}"
            root.mkdir()
            provider, request, _ = provider_fixture(root)
            provider.scientific_contract(request)
            for changed in (dataclasses.replace(request, date_utc="2026-06-29"),
                            dataclasses.replace(request, mjd_max=request.mjd_max + .1),
                            dataclasses.replace(request, mjd_min=request.mjd_min - 1),
                            dataclasses.replace(request, query_tag="tag"),
                            dataclasses.replace(request, target_loci=10)):
                with self.assertRaises(ProviderContractError):
                    provider.query(changed)


class SeparatedWorkTests(unittest.TestCase):
    def test_two_nights_acquire_concurrently_then_publish_in_order_with_separate_work(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            capability = F.make_capability(parent)
            F.seed_production(capability, ["2026-06-26", "2026-06-27"])
            publisher = P.NightPublisher(capability, publisher_release_sha=F.PUBLISHER_RELEASE,
                         cache_root=capability.root / "absent-cache", mountinfo_lines=F.mountinfo_for(capability.published_root))
            roots = PublicationRoots(capability.published_root, capability.staging_root,
                     capability.journal_root, capability.lock_root, capability.evidence_root)
            publisher.roots = roots
            work_root = parent / "range-work"
            work_root.mkdir(mode=0o700)
            work = RangeWorkCapability.for_local(work_root, work_root.name)
            adapter = F.SyntheticBackfillAdapter(8)
            barrier = threading.Barrier(2)
            original = adapter.query
            def concurrent(request):
                barrier.wait(timeout=10)
                return original(request)
            adapter.query = concurrent
            kwargs = dict(work_capability=work, publication_roots=roots, release_sha=RELEASE,
                read_capability_factory=F.mock_read_capability, settings=B.BackfillSettings(segment_size=4),
                prior_free_attestations=F.fixture_prior_free_attestations())
            controller = B.BackfillController(None, adapter, **kwargs)
            production = P.production_binding_from_sentinel(publisher.sentinel(NIGHT))
            authority = B.RangePublicationAuthorization(NIGHT, "2026-06-29", "2026-06-27",
                {key: production[key] for key in B._INITIAL_SENTINEL_FIELDS},
                candidate_release_sha=RELEASE, publisher_release_sha=F.PUBLISHER_RELEASE,
                authorized_by="Control", authorized_at_utc=F.AUTHORIZED_AT,
                expires_at_utc=F.EXPIRES_AT, **controller.range_binding(NIGHT, "2026-06-29"))
            events = []
            controller = B.BackfillController(None, adapter, publisher=publisher,
                range_authorization=authority, event_hook=lambda point, details: events.append((point, details["date_utc"])), **kwargs)
            result = controller.run(NIGHT, "2026-06-29")
            self.assertEqual([night["stage"] for night in result["nights"]], ["PUBLISHED", "PUBLISHED"])
            self.assertEqual([day for point, day in events if point == "before_publish"], [NIGHT, "2026-06-29"])
            self.assertEqual([day for point, day in events if point == "before_construct"], [NIGHT, "2026-06-29"])
            self.assertTrue((work_root / "nights" / f"night-{NIGHT}" / "candidate").is_dir())
            self.assertFalse((capability.root / "backfill").exists())
            before = dict(adapter.queries)
            controller.run(NIGHT, "2026-06-29", resume=True)
            self.assertEqual(dict(adapter.queries), before)

    def test_production_work_cannot_be_derived_from_publication_root(self):
        with self.assertRaises((StorageContractError, FileNotFoundError)):
            RangeWorkCapability.for_arnor(Path("/astro/store/shire/ANTARES/backfill"), "backfill", hostname="arnor")
        with self.assertRaises(B.BackfillRefused):
            B.BackfillController(object(), F.SyntheticBackfillAdapter(), release_sha=RELEASE)

    def test_live_execution_requires_explicit_gate_before_any_access(self):
        args = ["execute", "--start", NIGHT, "--end", "2026-06-29", "--work-root",
                "/astro/store/shire/ANTARES/work/backfill/g5-canary", "--candidate-release", RELEASE]
        with mock.patch.object(R, "range_read_capability", side_effect=AssertionError("live call")), self.assertRaises(B.BackfillRefused):
            R.main(args)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            R.main(["plan", *args[1:]])
        self.assertEqual(json.loads(output.getvalue())["execution"], "NOT EXECUTED")


class ProductionAuthorityTests(unittest.TestCase):
    """Actual sealed production capability flow with all roots replaced by temp fixtures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self.tmp.name).resolve()
        self.capability = F.make_capability(self.parent)
        cap = self.capability
        def live_artifacts(night):
            root = self.parent / f"seed-{night}"
            root.mkdir(mode=0o700)
            capability = F.mock_read_capability(root, root.name, night, RELEASE)
            minimum, maximum = night_mjd_interval(night)
            loci = [FakeLocus(f"ANT-{night}-{index}", mjd=minimum + .5,
                     lightcurve=pd.DataFrame({"mjd": [minimum + .5], "ztf_magpsf": [20.],
                     "ztf_sigmapsf": [.1], "ztf_fid": [1]})) for index in range(3)]
            by_id = {locus.locus_id: locus for locus in loci}
            provider = LiveAntaresProvider(capability,
                search_fn=lambda body: [locus for locus in loci if _body_matches_locus(body, locus)],
                get_by_id_fn=by_id.__getitem__, connectivity_fn=lambda: [])
            return build_night_artifacts(provider.fetch_night(NightScienceRequest(
                night, minimum, maximum, target_loci=None)))
        F.write_night(cap.published_root, "2026-06-26", live_artifacts("2026-06-26"))
        history.update_cumulative_indexes(data_root=cap.published_root)
        self.observer = P.NightPublisher(cap, publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=cap.root / "absent-cache", mountinfo_lines=F.mountinfo_for(cap.published_root))
        predecessor = F.make_backfill_candidate(self.parent, "2026-06-27")
        artifacts = live_artifacts("2026-06-27")
        for name, payload in artifacts.items():
            (predecessor.candidate_dir / name).write_bytes(payload)
        record = P._read_json(predecessor.record_path)
        record.update(artifacts={name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                                 for name, payload in artifacts.items()}, loci=3, alerts=3,
                      constructed_at_utc=B._utc_now().isoformat())
        predecessor.record_path.write_text(json.dumps(record))
        predecessor = P.load_backfill_candidate(predecessor.candidate_dir.parent)
        outcome = self.observer.publish(predecessor, F.authorize(self.observer, predecessor,
                        authorized_at=B._utc_now().isoformat()))
        self.assertTrue(outcome.success, outcome.record)
        self.work_root = cap.root / "work/backfill/g5-range"
        self.work_root.mkdir(mode=0o700, parents=True)
        self.work = RangeWorkCapability.for_local(self.work_root, self.work_root.name)
        self.roots = PublicationRoots(cap.published_root, cap.staging_root,
            cap.journal_root, cap.lock_root, cap.evidence_root)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(self.tmp.cleanup)
        mapped = dict(PRODUCTION_DATA_ROOT=cap.published_root, PRODUCTION_STAGE_ROOT=cap.staging_root,
            PRODUCTION_CONTROL_ROOT=cap.journal_root.parent, PRODUCTION_EVIDENCE_ROOT=cap.evidence_root)
        self.stack.enter_context(mock.patch.multiple(P, **mapped,
            CONTROL_APPROVED_LOCK=cap.lock_root / P.AUTHORITY_LOCK_NAME))
        self.stack.enter_context(mock.patch.multiple(S, **mapped, PRODUCTION_AUTHORITY_ROOT=cap.root))
        self.stack.enter_context(mock.patch.multiple(R, **mapped, RANGE_WORK_PARENT=self.work_root.parent,
            SENTINEL_CACHE=self.observer.cache_root, SEGMENT_CACHE=cap.root / "work/segment-cache"))
        self.stack.enter_context(mock.patch.object(R.ProductionRangePublisher, "roots", self.roots))
        self.stack.enter_context(mock.patch.object(R.ProductionRangePublisher, "capability", self.roots))
        self.stack.enter_context(mock.patch.object(R.socket, "getfqdn", return_value="arnor.fixture"))
        self.stack.enter_context(mock.patch.object(R, "_verify_release", return_value={}))
        self.stack.enter_context(mock.patch.object(R.sys, "prefix", str(Path(__file__).parents[1])))
        self.fetched = []
        self.queried = []

        def provider_factory(capability):
            minimum, _ = night_mjd_interval(capability.target_date_utc)
            loci = [FakeLocus(f"ANT-{capability.target_date_utc}-{index}", mjd=minimum + .5,
                    lightcurve=pd.DataFrame({"mjd": [minimum + .5], "ztf_magpsf": [20.],
                    "ztf_sigmapsf": [.1], "ztf_fid": [1]})) for index in range(3)]
            by_id = {locus.locus_id: locus for locus in loci}
            def search(body):
                self.queried.append(capability.target_date_utc)
                return [locus for locus in loci if _body_matches_locus(body, locus)]
            def fetch(locus_id):
                self.fetched.append(locus_id)
                return by_id[locus_id]
            return LiveAntaresProvider(capability, search_fn=search, get_by_id_fn=fetch,
                connectivity_fn=lambda: [], sleeper=lambda _: None)
        self.adapter = R.LiveRangeAdapter(self.work_root, RELEASE, F.mock_read_capability,
                                          provider_factory=provider_factory)
        self.settings = B.BackfillSettings(acquisition_concurrency=2, segment_size=2)
        probe = self.controller()
        initial = P.production_binding_from_sentinel(self.observer.sentinel(NIGHT))
        scope = B.RangePublicationAuthorization(NIGHT, "2026-06-29", "2026-06-27",
            {key: initial[key] for key in B._INITIAL_SENTINEL_FIELDS},
            candidate_release_sha=RELEASE, publisher_release_sha=F.PUBLISHER_RELEASE,
            authorized_by="Control", authorized_at_utc=F.AUTHORIZED_AT,
            expires_at_utc=F.EXPIRES_AT, **probe.range_binding(NIGHT, "2026-06-29"))
        self.token = "d" * 64
        self.authority = R.ProductionRangeAuthorization(scope, str(self.work_root), "arnor.fixture",
            os.geteuid(), hashlib.sha256(self.token.encode()).hexdigest(), R.implementation_identity(),
            publisher_wheel_sha256="e" * 64, segment_size=2)
        self.publisher = R.ProductionRangePublisher(self.authority, control_token=self.token)
        self.publisher.mountinfo_lines = F.mountinfo_for(self.capability.published_root)

    def controller(self, publisher=None, scope=None):
        return B.BackfillController(None, self.adapter, release_sha=RELEASE,
            work_capability=self.work, publication_roots=self.roots,
            read_capability_factory=F.mock_read_capability, settings=self.settings,
            publisher=publisher, range_authorization=scope)

    def build_first(self):
        controller = self.controller(self.publisher, self.authority.scope)
        self.assertTrue(controller.acquire(NIGHT)["ok"])
        constructed = controller.construct(NIGHT, ())
        self.assertTrue(constructed["ok"], constructed)
        candidate = P.load_backfill_candidate(controller.workspace(NIGHT).root)
        authorization = controller._night_authorization(NIGHT, candidate)
        return controller, candidate, authorization

    def test_two_nights_use_distinct_one_shot_capabilities_and_complete_predecessors(self):
        controller = self.controller(self.publisher, self.authority.scope)
        result = controller.run(NIGHT, "2026-06-29")
        self.assertEqual([night["stage"] for night in result["nights"]], ["PUBLISHED", "PUBLISHED"], result)
        bindings = sorted((self.work_root / "production-bindings").glob("*.json"))
        bindings = [path for path in bindings if not path.name.endswith(".sentinel.json")]
        self.assertEqual(len(bindings), 2)
        parsed = [P.ProductionPublicationBinding(**P._read_json(path)) for path in bindings]
        self.assertEqual([binding.night_utc for binding in parsed], [NIGHT, "2026-06-29"])
        self.assertNotEqual(parsed[0].digest, parsed[1].digest)
        self.assertEqual({binding.range_authorization_sha256 for binding in parsed}, {self.authority.digest})
        self.assertEqual({binding.max_successful_uses for binding in parsed}, {1})
        counts = len(self.queried), len(self.fetched)
        result = controller.run(NIGHT, "2026-06-29", resume=True)
        self.assertEqual(len(self.queried), counts[0])
        self.assertEqual(len(self.fetched), counts[1])
        self.assertEqual(result["summary"]["nights_published"], 2)
        self.assertFalse(R.SENTINEL_CACHE.exists())
        self.assertFalse(R.SEGMENT_CACHE.exists())

    def test_capability_cannot_authorize_another_night_range_candidate_or_token(self):
        controller, candidate, authorization = self.build_first()
        issued = []
        original = R.issue_production_publication_capability
        def capture(*args, **kwargs):
            capability = original(*args, **kwargs)
            issued.append(capability)
            return capability
        with mock.patch.object(R, "issue_production_publication_capability", side_effect=capture):
            outcome = self.publisher.publish(candidate, authorization)
        self.assertTrue(outcome.success, outcome.record)
        capability = issued[0]
        binding = P.ProductionPublicationBinding(**dict(capability.binding))
        saved = P._read_json(next((self.work_root / "production-bindings").glob("*.sentinel.json")))
        for changed in (dataclasses.replace(binding, range_authorization_sha256="0" * 64),
                        dataclasses.replace(binding, night_utc="2026-06-29", predecessor_night_utc=NIGHT),
                        dataclasses.replace(binding, candidate_record_sha256="0" * 64)):
            with self.subTest(changed=changed.digest), self.assertRaises((B.BackfillRefused, P.PublicationRefused, P.ProductionAuthorizationUnavailable)):
                P.issue_production_publication_capability(changed, authorization, candidate,
                    control_token=self.token, sentinel=saved, range_authorization=self.authority)
        with self.assertRaises(P.ProductionAuthorizationUnavailable):
            P.issue_production_publication_capability(binding, authorization, candidate,
                control_token="f" * 64, sentinel=saved, range_authorization=self.authority)
        other = F.make_backfill_candidate(self.parent, "2026-06-29")
        publisher = P.NightPublisher(capability, publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=R.SENTINEL_CACHE, mountinfo_lines=self.publisher.mountinfo_lines)
        refused = publisher.publish(other, authorization)
        self.assertFalse(refused.success)
        replay = publisher.publish(candidate, authorization)
        self.assertTrue(replay.success, replay.record)
        self.assertEqual(replay.status, "already_published")

    def test_incomplete_predecessor_blocks_later_publication_after_acquisition(self):
        controller = self.controller(self.publisher, self.authority.scope)
        for night in (NIGHT, "2026-06-29"):
            self.assertTrue(controller.acquire(night)["ok"])
        self.assertTrue(controller.construct(NIGHT, ())["ok"])
        self.assertTrue(controller.construct("2026-06-29", (NIGHT,))["ok"])
        outcome = controller.publish("2026-06-29", (NIGHT,))
        self.assertFalse(outcome["ok"])
        self.assertTrue(controller.workspace("2026-06-29").fetch_complete())
        self.assertFalse((self.work_root / "production-bindings").exists())

    def test_range_authorization_drift_and_unsupported_semantics_fail_closed(self):
        for changes in ({"work_root": str(self.capability.root / "backfill")},
                        {"segment_size": 3}, {"implementation_sha256": {}},
                        {"scope": dataclasses.replace(self.authority.scope, science_contract_sha256="0" * 64)},
                        {"scope": dataclasses.replace(self.authority.scope, configuration_sha256="0" * 64)},
                        {"scope": dataclasses.replace(self.authority.scope, cache_root=str(self.parent / "cache"))}):
            with self.subTest(changes=changes), self.assertRaises(B.BackfillRefused):
                dataclasses.replace(self.authority, **changes)
        with self.assertRaises(B.BackfillRefused):
            self.authority.verify(control_token="f" * 64)
        with self.assertRaises(B.BackfillRefused):
            self.authority.verify(control_token=self.token, hostname="other.fixture")
        with self.assertRaises(B.BackfillRefused):
            R.range_read_capability(self.authority, self.work_root, "wrong", "2026-06-30",
                                    RELEASE, control_token=self.token)

    def test_token_free_recovery_finishes_only_the_exact_gated_transaction(self):
        controller, candidate, authorization = self.build_first()
        original = P.NightPublisher
        fired = []
        def fault(point, details):
            if point == "after_manifest_commit" and not fired:
                fired.append(point)
                raise InjectedWriterFailure(point)
        def failing_publisher(*args, **kwargs):
            return original(*args, **kwargs, fault_hook=fault)
        with mock.patch.object(R, "NightPublisher", side_effect=failing_publisher):
            first = self.publisher.publish(candidate, authorization)
        self.assertFalse(first.success, first.record)
        self.assertEqual(controller.night_state(NIGHT)["stage"], "RECONCILIATION_REQUIRED")
        counts = len(self.queried), len(self.fetched)
        recovery = R.ProductionRangePublisher(self.authority, control_token=None)
        recovery.mountinfo_lines = self.publisher.mountinfo_lines
        recovered_controller = self.controller(recovery, self.authority.scope)
        future = datetime(2028, 1, 1, tzinfo=timezone.utc)
        with mock.patch.object(B, "_utc_now", return_value=future), mock.patch.object(R, "_utc_now", return_value=future), \
                mock.patch.object(self.adapter, "query_resumable", side_effect=AssertionError("query during recovery")), \
                mock.patch.object(self.adapter, "fetch_segment", side_effect=AssertionError("fetch during recovery")), \
                mock.patch.object(self.adapter, "construct_checkpoint", side_effect=AssertionError("construct during recovery")):
            recovered = recovered_controller.run(NIGHT, "2026-06-29", resume=True, recovery_only=True)
            self.assertEqual(recovered["nights"][0]["stage"], "PUBLISHED", recovered)
            self.assertEqual(recovered["nights"][1]["stage"], "PLANNED")
            recovered_controller.run(NIGHT, "2026-06-29", resume=True, recovery_only=True)
        self.assertEqual((len(self.queried), len(self.fetched)), counts)

    def test_token_free_issuance_before_gate_is_refused(self):
        controller, candidate, authorization = self.build_first()
        original = P.NightPublisher
        def fault(point, details):
            if point == "before_authority_gate":
                raise InjectedWriterFailure(point)
        with mock.patch.object(R, "NightPublisher", side_effect=lambda *args, **kwargs: original(*args, **kwargs, fault_hook=fault)):
            refused = self.publisher.publish(candidate, authorization)
        self.assertFalse(refused.success)
        recovery = R.ProductionRangePublisher(self.authority, control_token=None)
        recovery.mountinfo_lines = self.publisher.mountinfo_lines
        with self.assertRaises(P.ProductionAuthorizationUnavailable):
            recovery.publish(candidate, authorization)

    def test_g4_legacy_binding_digest_is_preserved(self):
        from test_publication_authority_v3 import JUNE27
        with mock.patch.object(P, "CONTROL_APPROVED_LOCK", Path(JUNE27["authority_lock"]["path"])):
            binding = P.ProductionPublicationBinding(**JUNE27)
        self.assertNotIn("range_authorization_sha256", binding.as_dict())
        self.assertEqual(binding.digest, hashlib.sha256(P._canonical(JUNE27 | {
            "schema_version": P.PRODUCTION_CAPABILITY_SCHEMA, "operation": P.AUTHORIZED_OPERATION,
            "max_successful_uses": 1})).hexdigest())


if __name__ == "__main__":
    unittest.main()
