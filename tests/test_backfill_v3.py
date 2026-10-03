"""V3 resumable backfill controller, cache, and concurrency contracts."""

import json
import tempfile
import threading
import unittest
from pathlib import Path

import pandas as pd

import v3_fixtures as F
from src import history
from src.operations.backfill import (
    BackfillController,
    BackfillRefused,
    BackfillSettings,
    NightStage,
    RangePublicationAuthorization,
    inspect_backfill,
)
from src.operations.cache import SegmentCache, SegmentCacheRefused
from src.operations.publication import (
    NightPublisher,
    authoritative_nights,
    production_binding_from_sentinel,
)


START, END = "2026-07-01", "2026-07-04"
NIGHTS = ("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04")


class BackfillFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def production(self, name="v3-production"):
        capability = F.make_capability(self.tmp, name)
        F.seed_production(capability, ["2026-06-29", "2026-06-30"])
        publisher = NightPublisher(
            capability,
            publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=capability.root / "absent-cache",
            mountinfo_lines=F.mountinfo_for(capability.published_root),
        )
        return capability, publisher

    def range_authority(self, publisher, adapter, start=START, end=END, *, cache=None):
        """What Control binds: range, initial Sentinel state and exact identities."""
        from datetime import date, timedelta

        binding = self.controller(publisher.capability, adapter, cache=cache).range_binding(
            start, end
        )
        production = production_binding_from_sentinel(publisher.sentinel(start))
        return RangePublicationAuthorization(
            start_date_utc=start,
            end_date_utc=end,
            initial_predecessor_date_utc=(
                date.fromisoformat(start) - timedelta(days=1)
            ).isoformat(),
            initial_sentinel={
                key: production[key]
                for key in (
                    "canonical_root", "mount_binding", "durable_fingerprint_sha256",
                    "manifest_count", "cumulative_sha256",
                )
            },
            candidate_release_sha=F.CANDIDATE_RELEASE,
            publisher_release_sha=F.PUBLISHER_RELEASE,
            authorized_by="control-fixture",
            authorized_at_utc="2026-09-28T00:00:00+00:00",
            expires_at_utc=F.EXPIRES_AT,
            **binding,
        )

    def controller(self, capability, adapter, publisher=None, authority=None, **kwargs):
        settings = kwargs.pop("settings", BackfillSettings(segment_size=4))
        kwargs.setdefault("prior_free_attestations", F.fixture_prior_free_attestations())
        return BackfillController(
            capability,
            adapter,
            release_sha=F.CANDIDATE_RELEASE,
            read_capability_factory=F.mock_read_capability,
            settings=settings,
            publisher=publisher,
            range_authorization=authority,
            **kwargs,
        )

    @staticmethod
    def stages(document):
        return {night["date_utc"]: night["stage"] for night in document["nights"]}

    @staticmethod
    def night(document, day):
        return next(night for night in document["nights"] if night["date_utc"] == day)

    @staticmethod
    def publication_records(capability, day):
        directory = capability.evidence_root / "publications" / day
        return [
            json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
        ] if directory.is_dir() else []


class MultiNightProgressionTests(BackfillFixture):
    def test_range_publishes_every_night_in_order_with_chained_authority(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(10)
        document = self.controller(
            capability, adapter, publisher, self.range_authority(publisher, adapter)
        ).run(START, END)
        self.assertEqual(set(self.stages(document).values()), {"PUBLISHED"})
        self.assertEqual(
            authoritative_nights(capability.published_root),
            ("2026-06-29", "2026-06-30") + NIGHTS,
        )
        previous = None
        for day in NIGHTS:
            records = self.publication_records(capability, day)
            self.assertEqual([record["status"] for record in records], ["PUBLISHED"])
            if previous is not None:
                self.assertEqual(
                    records[0]["predecessor"]["production_fingerprint"],
                    previous["resulting_production_fingerprint"],
                )
            previous = records[0]
        index = pd.read_parquet(history.cumulative_paths(capability.published_root)["loci_index"])
        for day in NIGHTS:
            self.assertEqual(int(index["night_date_utc"].eq(day).sum()), 10)
        # Overlap validation saw every earlier night (shared locus recurs nightly).
        target = capability.published_root / "data/lsst_only/nightly/2026/07/04/manifest.json"
        self.assertEqual(json.loads(target.read_text())["validation"]["overlap_count"], 1)
        summary = document["summary"]
        self.assertEqual(summary["nights_published"], 4)
        self.assertEqual(summary["backlog"], 0)
        self.assertIsNotNone(summary["seconds_per_published_night"])
        self.assertTrue((capability.root / "backfill/ranges/2026-07-01_2026-07-04.json").is_file())
        metrics = self.night(document, "2026-07-02")["metrics"]
        for key in ("query_seconds", "fetch_seconds", "construction_seconds",
                    "publication_seconds", "segments_fetched", "checkpoint_bytes"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["segments_fetched"], 3)

    def test_without_publication_authority_nights_wait(self):
        capability, _ = self.production()
        document = self.controller(capability, F.SyntheticBackfillAdapter(6)).run(START, END)
        self.assertEqual(set(self.stages(document).values()), {"WAITING_FOR_PUBLICATION"})
        self.assertEqual(authoritative_nights(capability.published_root), ("2026-06-29", "2026-06-30"))

    def test_fresh_run_refuses_existing_evidence_and_settings_are_bounded(self):
        capability, _ = self.production()
        controller = self.controller(capability, F.SyntheticBackfillAdapter(4))
        controller.run(START, START)
        with self.assertRaises(BackfillRefused):
            controller.run(START, START)
        with self.assertRaises(BackfillRefused):
            BackfillSettings(publication_concurrency=2)
        with self.assertRaises(BackfillRefused):
            BackfillSettings(acquisition_concurrency=0)


class ResumeTests(BackfillFixture):
    def test_interrupted_fetch_resumes_from_missing_segments_only(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(12)  # 3 segments of 4
        adapter.fail_fetch_at["2026-07-01"] = 2  # third segment fails once
        authority = self.range_authority(publisher, adapter, START, START)
        first = self.controller(capability, adapter, publisher, authority).run(START, START)
        night = self.night(first, START)
        self.assertEqual(night["stage"], "BLOCKED")
        self.assertEqual(night["blocked"]["stage"], "FETCHING")
        self.assertTrue(night["blocked"]["retryable"])
        self.assertEqual(night["blocked"]["segments_committed"], 2)
        self.assertEqual(adapter.fetched_segments[START], 2)
        resumed = self.controller(capability, adapter, publisher, authority).run(
            START, START, resume=True
        )
        self.assertEqual(self.stages(resumed)[START], "PUBLISHED")
        self.assertEqual(adapter.queries[START], 1)  # sealed query reused
        self.assertEqual(adapter.fetched_segments[START], 3)  # only the missing one
        metrics = self.night(resumed, START)["metrics"]
        self.assertEqual(metrics["segments_reused"], 2)
        self.assertEqual(metrics["retries"], 1)

    def test_candidate_failure_rebuilds_from_sealed_acquisition(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(8)
        adapter.fail_construct.add("2026-07-02")
        authority = self.range_authority(publisher, adapter)
        first = self.controller(capability, adapter, publisher, authority).run(START, END)
        self.assertEqual(self.stages(first)["2026-07-01"], "PUBLISHED")
        self.assertEqual(self.night(first, "2026-07-02")["blocked"]["stage"], "CANDIDATE_BUILDING")
        queries, fetches = dict(adapter.queries), dict(adapter.fetched_segments)
        resumed = self.controller(capability, adapter, publisher, authority).run(
            START, END, resume=True
        )
        self.assertEqual(set(self.stages(resumed).values()), {"PUBLISHED"})
        self.assertEqual(dict(adapter.queries), queries)  # never re-queried
        self.assertEqual(dict(adapter.fetched_segments), fetches)  # never re-fetched

    def test_publication_failure_never_reacquires(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(8)
        failing = NightPublisher(
            capability, publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=capability.root / "absent-cache",
            mountinfo_lines=F.mountinfo_for(capability.published_root),
            fault_hook=_fail_nth("before_authority_gate", 2),  # 2026-07-02, pre-gate
        )
        authority = self.range_authority(publisher, adapter)
        first = self.controller(capability, adapter, failing, authority).run(START, END)
        self.assertEqual(self.stages(first)["2026-07-01"], "PUBLISHED")
        self.assertEqual(self.night(first, "2026-07-02")["blocked"]["stage"], "PUBLISHING")
        queries, fetches = dict(adapter.queries), dict(adapter.fetched_segments)
        constructions = dict(adapter.constructions)
        resumed = self.controller(capability, adapter, publisher, authority).run(
            START, END, resume=True
        )
        self.assertEqual(set(self.stages(resumed).values()), {"PUBLISHED"})
        self.assertEqual(dict(adapter.queries), queries)
        self.assertEqual(dict(adapter.fetched_segments), fetches)
        self.assertEqual(adapter.constructions["2026-07-02"], constructions["2026-07-02"])

    def test_restart_after_controller_interruption_and_derived_index(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(6)
        authority = self.range_authority(publisher, adapter)

        def crash(point, details):
            if point == "before_publish" and details.get("date_utc") == "2026-07-03":
                raise SystemExit("simulated process death")

        with self.assertRaises(SystemExit):
            self.controller(capability, adapter, publisher, authority, event_hook=crash).run(
                START, END
            )
        range_path = capability.root / "backfill/ranges/2026-07-01_2026-07-04.json"
        range_path.unlink()  # the global index is derived and disposable
        inspected = inspect_backfill(capability.root, capability.published_root, START, END)
        self.assertEqual(self.stages(inspected)["2026-07-02"], "PUBLISHED")
        self.assertEqual(self.stages(inspected)["2026-07-03"], "WAITING_FOR_PUBLICATION")
        queries = dict(adapter.queries)
        resumed = self.controller(capability, adapter, publisher, authority).run(
            START, END, resume=True
        )
        self.assertEqual(set(self.stages(resumed).values()), {"PUBLISHED"})
        self.assertEqual(dict(adapter.queries), queries)
        for day in NIGHTS:
            self.assertEqual(len(self.publication_records(capability, day)), 1)

    def test_repeated_resume_is_idempotent(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(6)
        authority = self.range_authority(publisher, adapter)
        self.controller(capability, adapter, publisher, authority).run(START, END)
        fingerprint = publisher.sentinel(END)["durable_fingerprint_sha256"]
        counters = (dict(adapter.queries), dict(adapter.fetched_segments), dict(adapter.constructions))
        for _ in range(2):
            document = self.controller(capability, adapter, publisher, authority).run(
                START, END, resume=True
            )
            self.assertEqual(set(self.stages(document).values()), {"PUBLISHED"})
        self.assertEqual(
            (dict(adapter.queries), dict(adapter.fetched_segments), dict(adapter.constructions)),
            counters,
        )
        self.assertEqual(publisher.sentinel(END)["durable_fingerprint_sha256"], fingerprint)
        for day in NIGHTS:
            self.assertEqual(len(self.publication_records(capability, day)), 1)
        self.assertEqual(authoritative_nights(capability.published_root).count(END), 1)


class PredecessorOrderingTests(BackfillFixture):
    def test_later_nights_acquire_but_cannot_publish_across_blocked_predecessor(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(8)
        adapter.fail_fetch_at["2026-07-02"] = 1
        authority = self.range_authority(publisher, adapter)
        document = self.controller(capability, adapter, publisher, authority).run(START, END)
        stages = self.stages(document)
        self.assertEqual(stages["2026-07-01"], "PUBLISHED")
        self.assertEqual(stages["2026-07-02"], "BLOCKED")
        self.assertEqual(stages["2026-07-03"], "FETCH_COMPLETE")
        self.assertEqual(stages["2026-07-04"], "FETCH_COMPLETE")
        self.assertEqual(
            authoritative_nights(capability.published_root)[-1], "2026-07-01"
        )
        self.assertEqual(document["summary"]["publication_frontier"], "2026-07-02")
        resumed = self.controller(capability, adapter, publisher, authority).run(
            START, END, resume=True
        )
        self.assertEqual(set(self.stages(resumed).values()), {"PUBLISHED"})
        self.assertEqual(adapter.queries["2026-07-04"], 1)


class CacheTests(BackfillFixture):
    def test_corrupted_cache_entry_is_rejected_and_reacquired(self):
        cache_root = self.tmp / "segment-cache"
        cache_root.mkdir()
        first_cap, _ = self.production("workspace-a")
        first_adapter = F.SyntheticBackfillAdapter(12)
        cache = F.make_segment_cache(cache_root)
        self.controller(first_cap, first_adapter, cache=cache).run(START, START)
        self.assertEqual(cache.statistics()["stored"], 3)
        objects = sorted(path for path in (cache_root / "objects").rglob("*") if path.is_file())
        objects[0].write_bytes(objects[0].read_bytes()[:-5] + b"xxxxx")

        second_cap, _ = self.production("workspace-b")
        second_adapter = F.SyntheticBackfillAdapter(12)
        second_cache = F.make_segment_cache(cache_root)
        self.controller(second_cap, second_adapter, cache=second_cache).run(START, START)
        stats = second_cache.statistics()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["rejected"], 1)
        self.assertEqual(second_adapter.fetched_segments[START], 1)
        # Cache reuse never changes science: candidates are byte-identical.
        for name in ("loci.parquet", "alerts.parquet", "manifest.json"):
            a = first_cap.root / "backfill/nights/night-2026-07-01/candidate" / name
            b = second_cap.root / "backfill/nights/night-2026-07-01/candidate" / name
            self.assertEqual(a.read_bytes(), b.read_bytes(), name)

    def test_cache_refuses_protected_roots(self):
        protected = self.tmp / "protected"
        (protected / "inner").mkdir(parents=True)
        with self.assertRaises(SegmentCacheRefused):
            SegmentCache(protected / "inner", forbidden_roots=(protected,))
        with self.assertRaises(SegmentCacheRefused):
            SegmentCache(self.tmp / "missing")


class StatusCliTests(BackfillFixture):
    def test_backfill_status_is_read_only_and_signals_blocked_nights(self):
        import contextlib
        import hashlib
        import io

        from src import cli

        capability, _ = self.production()
        adapter = F.SyntheticBackfillAdapter(8)
        adapter.fail_fetch_at["2026-07-02"] = 0
        self.controller(capability, adapter).run(START, "2026-07-02")

        def digest():
            h = hashlib.sha256()
            for path in sorted(capability.root.rglob("*")):
                h.update(str(path).encode())
                if path.is_file():
                    h.update(path.read_bytes())
            return h.hexdigest()

        before = digest()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = cli.main([
                "backfill", "status", START, "2026-07-02",
                "--run-root", str(capability.root), "--json",
            ])
        self.assertEqual(code, 1)
        document = json.loads(output.getvalue())
        self.assertEqual(self.stages(document), {
            "2026-07-01": "WAITING_FOR_PUBLICATION", "2026-07-02": "BLOCKED",
        })
        self.assertEqual(digest(), before)


class ConcurrencyTests(BackfillFixture):
    def test_single_ordered_writer_and_bounded_acquisition(self):
        capability, publisher = self.production()
        adapter = F.SyntheticBackfillAdapter(8, delays={day: 0.05 for day in NIGHTS})
        active = {"now": 0, "max": 0, "order": []}
        lock = threading.Lock()
        original = publisher.publish

        def tracked(candidate, authorization):
            with lock:
                active["now"] += 1
                active["max"] = max(active["max"], active["now"])
                active["order"].append(candidate.date_utc)
            try:
                return original(candidate, authorization)
            finally:
                with lock:
                    active["now"] -= 1

        publisher.publish = tracked
        settings = BackfillSettings(acquisition_concurrency=2, construction_concurrency=2,
                                    segment_size=4)
        self.controller(
            capability, adapter, publisher, self.range_authority(publisher, adapter),
            settings=settings,
        ).run(START, END)
        self.assertEqual(active["max"], 1)
        self.assertEqual(active["order"], list(NIGHTS))
        self.assertLessEqual(adapter.max_in_flight, 2)
        self.assertEqual(adapter.max_in_flight, 2)  # concurrency was actually exercised

    def test_state_is_deterministic_under_completion_order_differences(self):
        results = {}
        for label, delays, acquisition in (
            ("reverse", {"2026-07-01": 0.2, "2026-07-02": 0.1}, 3),
            ("serial", {}, 1),
        ):
            capability, publisher = self.production(f"production-{label}")
            adapter = F.SyntheticBackfillAdapter(8, delays=delays)
            settings = BackfillSettings(acquisition_concurrency=acquisition, segment_size=4)
            document = self.controller(
                capability, adapter, publisher, self.range_authority(publisher, adapter), settings=settings
            ).run(START, END)
            paths = history.cumulative_paths(capability.published_root)
            candidates = {
                day: {
                    name: (capability.root / f"backfill/nights/night-{day}/candidate" / name).read_bytes()
                    for name in ("loci.parquet", "alerts.parquet", "manifest.json")
                }
                for day in NIGHTS
            }
            results[label] = (
                self.stages(document),
                pd.read_parquet(paths["loci_index"]),
                candidates,
            )
        self.assertEqual(results["reverse"][0], results["serial"][0])
        pd.testing.assert_frame_equal(results["reverse"][1], results["serial"][1])
        self.assertEqual(results["reverse"][2], results["serial"][2])


def _fail_nth(point, occurrence):
    """Raise once at the n-th occurrence of a publication boundary."""
    seen = {"count": 0}

    def hook(name, details):
        if name == point:
            seen["count"] += 1
            if seen["count"] == occurrence:
                from src.operations.writer import InjectedWriterFailure

                raise InjectedWriterFailure(point)

    return hook


if __name__ == "__main__":
    unittest.main()
