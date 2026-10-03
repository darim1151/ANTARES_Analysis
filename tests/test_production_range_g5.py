"""G5 local qualification: no service calls or configured production mutation."""
import contextlib
import dataclasses
from datetime import datetime, timezone
import hashlib
import hmac
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
from requests.models import Response

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


def outage_provider(root, failing=(180.0, 360.0)):
    """Fixture provider whose service keeps failing one tile transiently."""
    provider, request, bindings = provider_fixture(root)
    original, seen = provider._search_fn, []
    def outage(body):
        tile = body["query"]["bool"]["filter"][1]["range"]["ra"]
        seen.append((tile["gte"], tile["lt"]))
        if seen[-1] == failing:
            raise ConnectionError("transient outage")
        yield from original(body)
    provider._search_fn = outage
    return provider, request, bindings, seen


def journal_events(root):
    journal = root / "checkpoints/query-progress-v2"
    return [json.loads(path.read_text())["payload"]["event"] for path in sorted(journal.glob("event-*.json"))]


def control_approval(authority, token, path, *, range_digest=None, approved_by="ANTARES-Control-G5"):
    """Test-only stand-in for Control: the package itself never creates approvals."""
    body = {"schema_version": R.APPROVAL_SCHEMA, "range_authorization_sha256": range_digest or authority.digest,
            "approved_by": approved_by, "approved_at_utc": F.AUTHORIZED_AT}
    mac = hmac.new(token.encode("ascii"), B._canonical(body), hashlib.sha256).hexdigest()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(body, approval_hmac_sha256=mac)))
    path.chmod(0o600)
    return path


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
        event = self.root / "checkpoints/query-progress-v2/event-00000001.json"
        document = json.loads(event.read_text())
        document["payload"]["event"]["records"] = []
        event.write_text(json.dumps(document))
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)
        self.assertEqual(calls, [])

    def test_missing_event_symlink_and_unknown_schema_fail_closed(self):
        provider, request, binding = provider_fixture(self.root)
        provider.query_resumable(request, binding)
        journal = self.root / "checkpoints/query-progress-v2"
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
        (self.root / "checkpoints/query-progress-v2/event-00000002.json").unlink()
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

    def test_terminal_transient_exhaustion_resumes_with_fresh_bounded_invocation(self):
        split, first, second = (0., 360.), (0., 180.), (180., 360.)
        for invocation, expected in ((1, [split, first, second, second]), (2, [second, second])):
            provider, request, binding, seen = outage_provider(self.root)
            failed = provider.query_resumable(request, binding)
            self.assertFalse(failed.clean)
            self.assertTrue(failed.evidence.errors and all(issue.retryable for issue in failed.evidence.errors))
            # Bounded: exactly max_query_attempts searches per invocation, never a loop.
            self.assertEqual(seen, expected, invocation)
            self.assertFalse((self.root / "checkpoints/query-result/COMMITTED.json").exists())
        events = journal_events(self.root)
        boundary = [index for index, event in enumerate(events) if "invocation_boundary" in event]
        self.assertEqual(len(boundary), 1)
        self.assertEqual(events[boundary[0] + 1]["trace"]["attempt"], 1)
        calls = []
        provider, request, binding = provider_fixture(self.root, calls=calls)
        resumed = provider.query_resumable(request, binding)
        # Completed split and terminal tile are skipped; only the unfinished tile is repeated.
        self.assertEqual(calls, [second])
        self.assertTrue(resumed.clean)
        self.assertEqual(sum("invocation_boundary" in event for event in journal_events(self.root)), 2)
        reference = provider.query(request)
        pd.testing.assert_frame_equal(resumed.loci, reference.loci)
        self.assertEqual(dict(resumed.evidence.details), dict(reference.evidence.details))
        calls.clear()
        repeated = provider.query_resumable(request, binding)
        self.assertEqual(calls, [])
        pd.testing.assert_frame_equal(repeated.loci, resumed.loci)
        self.assertEqual(sum("invocation_boundary" in event for event in journal_events(self.root)), 2)

    def test_invocation_boundary_contradictions_fail_closed(self):
        from src.operations.query_progress import QueryProgress
        second = {"mjd_min": request_for().mjd_min, "mjd_max": request_for().mjd_max,
                  "ra_min": 180., "ra_max": 360., "dec_min": -90., "dec_max": 90.}
        from src.operations.live_antares import _build_tile_query, _sha256_json
        def boundary(tile):
            return {"invocation_boundary": {"tile": tile, "query_sha256": _sha256_json(_build_tile_query(tile)),
                                            "reason": "terminal_transient_retry_exhaustion"}}
        def forged_retry(events):
            event = json.loads(json.dumps(events[-1]))
            event["trace"]["attempt"], event["retry_scheduled"] = 1, True
            return event
        cases = {
            "retry_without_boundary": lambda events: [forged_retry(events)],
            "duplicate_boundary": lambda events: [boundary(second), boundary(second)],
            "boundary_for_wrong_tile": lambda events: [boundary(dict(second, ra_min=0., ra_max=180.))],
        }
        for name, forged in cases.items():
            with self.subTest(name=name):
                root = self.parent / name / f"night-{NIGHT}"
                root.mkdir(mode=0o700, parents=True)
                provider, request, binding, _ = outage_provider(root)
                self.assertFalse(provider.query_resumable(request, binding).clean)
                identity = json.loads((root / "checkpoints/query-progress-v2/identity.json").read_text())["identity"]
                with QueryProgress(root, root.name, identity) as progress:
                    for event in forged(progress.events):
                        progress.commit(event)
                calls = []
                provider, request, binding = provider_fixture(root, calls=calls)
                with self.assertRaises(QueryCheckpointError):
                    provider.query_resumable(request, binding)
                self.assertEqual(calls, [])
        # A boundary is never needed or accepted after a completed traversal.
        root = self.parent / "complete" / f"night-{NIGHT}"
        root.mkdir(mode=0o700, parents=True)
        provider, request, binding = provider_fixture(root)
        self.assertTrue(provider.query_resumable(request, binding).clean)
        identity = json.loads((root / "checkpoints/query-progress-v2/identity.json").read_text())["identity"]
        with QueryProgress(root, root.name, identity) as progress:
            progress.commit(boundary(second))
        with self.assertRaises(QueryCheckpointError):
            provider.query_resumable(request, binding)


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
        self.approval_path = control_approval(self.authority, self.token, self.parent / "control" / "approval.json")
        self.approval = R.ControlRangeApproval.load(self.approval_path, work_root=self.work_root)
        self.publisher = R.ProductionRangePublisher(self.authority, control_token=self.token, approval=self.approval)
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
        self.assertEqual({binding.control_approval_sha256 for binding in parsed}, {self.approval.sha256})
        self.assertEqual(result["control_approval_sha256"], self.approval.sha256)
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
                        dataclasses.replace(binding, control_approval_sha256="0" * 64),
                        dataclasses.replace(binding, night_utc="2026-06-29", predecessor_night_utc=NIGHT),
                        dataclasses.replace(binding, candidate_record_sha256="0" * 64)):
            with self.subTest(changed=changed.digest), self.assertRaises((B.BackfillRefused, P.PublicationRefused, P.ProductionAuthorizationUnavailable)):
                P.issue_production_publication_capability(changed, authorization, candidate,
                    control_token=self.token, sentinel=saved, range_authorization=self.authority)
        with self.assertRaises(P.ProductionAuthorizationUnavailable):
            P.issue_production_publication_capability(binding, authorization, candidate,
                control_token="f" * 64, sentinel=saved, range_authorization=self.authority,
                range_approval=self.approval)
        with self.assertRaises(B.BackfillRefused):  # the token alone never activates range authority
            P.issue_production_publication_capability(binding, authorization, candidate,
                control_token=self.token, sentinel=saved, range_authorization=self.authority)
        with self.assertRaises(P.PublicationRefused):  # range bindings always record their approval
            dataclasses.replace(binding, control_approval_sha256=None)
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
                                    RELEASE, control_token=self.token, approval=self.approval)

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
            # A detached approval without the token grants no new acquisition or publication.
            approved = R.ProductionRangePublisher(self.authority, control_token=None, approval=self.approval)
            approved.mountinfo_lines = self.publisher.mountinfo_lines
            again = self.controller(approved, self.authority.scope).run(
                NIGHT, "2026-06-29", resume=True, recovery_only=True)
            self.assertEqual([night["stage"] for night in again["nights"]], ["PUBLISHED", "PLANNED"])
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
        approved = R.ProductionRangePublisher(self.authority, control_token=None, approval=self.approval)
        approved.mountinfo_lines = self.publisher.mountinfo_lines
        with self.assertRaises(P.ProductionAuthorizationUnavailable):
            approved.publish(candidate, authorization)

    def test_detached_control_approval_is_required_and_exact(self):
        night_root = self.work_root / "nights" / f"night-{NIGHT}"
        night_root.mkdir(mode=0o700, parents=True)
        def read(approval, authority=None):
            return R.range_read_capability(authority or self.authority, night_root, night_root.name, NIGHT,
                                           RELEASE, control_token=self.token, approval=approval)
        self.assertEqual(read(self.approval).target_date_utc, NIGHT)
        changed = dataclasses.replace(self.authority, publisher_wheel_sha256="f" * 64)
        control = self.parent / "control"
        refused = {
            "missing": None,
            "wrong_token": R.ControlRangeApproval.load(control_approval(
                self.authority, "f" * 64, control / "wrong-token.json"), work_root=self.work_root),
            "other_authorization": R.ControlRangeApproval.load(control_approval(
                changed, self.token, control / "other.json"), work_root=self.work_root),
            "tampered_approval": dataclasses.replace(self.approval, approved_by="someone-else"),
        }
        for name, approval in refused.items():
            with self.subTest(name=name), self.assertRaises(B.BackfillRefused):
                read(approval)
        with self.assertRaises(B.BackfillRefused):  # changed authorization invalidates its approval
            read(self.approval, changed)
        journals = sorted(path.name for path in self.capability.journal_root.iterdir())
        controller, candidate, authorization = self.build_first()
        for approval in (None, refused["other_authorization"]):
            with self.subTest(approval=approval), self.assertRaises(B.BackfillRefused):
                R.ProductionRangePublisher(self.authority, control_token=self.token,
                                           approval=approval).publish(candidate, authorization)
        self.assertFalse((self.work_root / "production-bindings").exists())
        self.assertEqual(sorted(path.name for path in self.capability.journal_root.iterdir()), journals)

    def test_control_approval_file_is_strict_and_detached(self):
        good = json.loads(self.approval_path.read_text())
        control = self.parent / "control"
        link = control / "link.json"
        link.symlink_to(self.approval_path)
        writable = control_approval(self.authority, self.token, control / "writable.json")
        writable.chmod(0o620)
        extra = control / "extra.json"
        extra.write_text(json.dumps(dict(good, token="d" * 64)))
        extra.chmod(0o600)
        inside = control_approval(self.authority, self.token, self.work_root / "approval.json")
        for path in (link, writable, extra, inside, Path("approval.json")):
            with self.subTest(path=str(path)), self.assertRaises((B.BackfillRefused, OSError)):
                R.ControlRangeApproval.load(path, work_root=self.work_root)

    def test_operator_cli_requires_detached_approval_before_live_access(self):
        base = ["--start", NIGHT, "--end", "2026-06-29", "--work-root", str(self.work_root),
                "--candidate-release", RELEASE, "--authorization", str(self.parent / "authorization.json"),
                "--publisher-wheel-sha256", "e" * 64, "--execute-authorized-range"]
        token_file = self.parent / "token"
        token_file.write_text(self.token)
        with mock.patch.object(R, "range_read_capability", side_effect=AssertionError("live call")), \
                mock.patch.object(R, "_read_json", side_effect=AssertionError("authorization read")):
            for argv in (["execute", *base, "--control-token-file", str(token_file)],
                         ["recover", *base, "--resume", "--control-approval", str(self.approval_path)]):
                with self.subTest(argv=argv[0]), self.assertRaises(B.BackfillRefused):
                    R.main(argv)
        self.assertFalse((self.work_root / "nights").exists())

    def test_controller_resume_after_transient_retry_exhaustion_skips_completed_tiles(self):
        original = self.adapter.provider_factory
        searches, armed = [], [True]
        def flaky(capability):
            provider = original(capability)
            search = provider._search_fn
            def outage(body):
                searches.append(body)
                if armed[0] and len(searches) in (11, 12):
                    raise ConnectionError("transient outage")
                return search(body)
            provider._search_fn = outage
            return provider
        self.adapter.provider_factory = flaky
        controller = self.controller()
        first = controller.run(NIGHT, NIGHT)["nights"][0]
        self.assertEqual(first["stage"], "BLOCKED", first)
        self.assertTrue(first["blocked"]["retryable"], first)
        self.assertEqual(len(searches), 12)
        armed[0] = False
        searches.clear()
        resumed = controller.run(NIGHT, NIGHT, resume=True)["nights"][0]
        # Candidate construction independently re-validated the canonical trace.
        self.assertEqual(resumed["stage"], "WAITING_FOR_PUBLICATION", resumed)
        self.assertEqual(len(searches), 6912 - 10)

    def test_controller_non_json_response_page_is_transient_not_malformed(self):
        # G6.2: an HTML error body makes the client's response.json() raise
        # requests' JSONDecodeError, a ValueError; v0.4.5 blocked such nights
        # as non-retryable live_query_malformed with an unresumable journal.
        original = self.adapter.provider_factory
        searches, armed = [], [True]
        def flaky(capability):
            provider = original(capability)
            search = provider._search_fn
            def outage(body):
                searches.append(body)
                if armed[0] and len(searches) in (11, 12):
                    response = Response()
                    response.status_code, response._content = 502, b"<html>502 Bad Gateway</html>"
                    response.json()
                return search(body)
            provider._search_fn = outage
            return provider
        self.adapter.provider_factory = flaky
        controller = self.controller()
        first = controller.run(NIGHT, NIGHT)["nights"][0]
        self.assertEqual(first["stage"], "BLOCKED", first)
        self.assertEqual(first["blocked"]["failure_category"], P.FailureCategory.TRANSIENT_NETWORK.value, first)
        self.assertTrue(first["blocked"]["retryable"], first)
        self.assertTrue(first["blocked"]["message"].startswith("live_query_incomplete:"), first)
        self.assertEqual(len(searches), 12)
        armed[0] = False
        searches.clear()
        resumed = controller.run(NIGHT, NIGHT, resume=True)["nights"][0]
        self.assertEqual(resumed["stage"], "WAITING_FOR_PUBLICATION", resumed)
        self.assertEqual(len(searches), 6912 - 10)

    def test_g4_legacy_binding_digest_is_preserved(self):
        from test_publication_authority_v3 import JUNE27
        with mock.patch.object(P, "CONTROL_APPROVED_LOCK", Path(JUNE27["authority_lock"]["path"])):
            binding = P.ProductionPublicationBinding(**JUNE27)
        self.assertNotIn("range_authorization_sha256", binding.as_dict())
        self.assertNotIn("control_approval_sha256", binding.as_dict())
        self.assertEqual(binding.digest, hashlib.sha256(P._canonical(JUNE27 | {
            "schema_version": P.PRODUCTION_CAPABILITY_SCHEMA, "operation": P.AUTHORIZED_OPERATION,
            "max_successful_uses": 1})).hexdigest())


if __name__ == "__main__":
    unittest.main()
