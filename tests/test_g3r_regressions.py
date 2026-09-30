"""Regressions for the two G3R BLOCKER defects.

Both tests use only APIs that also existed in the pre-remediation G3 candidate
so they can be (and were) shown to fail against it:

* BLOCKER 1: a fault immediately after the nightly manifest link was reported
  ``UNPUBLISHED_FAILURE`` with ``production_mutated=False`` and the staged
  cumulative material was deleted, so retry failed.
* BLOCKER 2: a fault before ``nightly_summary`` installation left a visible
  manifest with a stale summary, yet the controller derived ``PUBLISHED``.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import v3_fixtures as F
from src import history
from src.operations.backfill import inspect_backfill
from src.operations.publication import load_backfill_candidate
from src.operations.writer import InjectedWriterFailure


NIGHT = "2026-07-01"


class _FailOnce:
    def __init__(self, point):
        self.point = point
        self.fired = False

    def __call__(self, point, details):
        if point == self.point and not self.fired:
            self.fired = True
            raise InjectedWriterFailure(point)


class G3RBlockerRegressionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.capability = F.make_capability(self.tmp)
        F.seed_production(self.capability, ["2026-06-29", "2026-06-30"])
        made = F.make_backfill_candidate(self.tmp, NIGHT)
        night_root = self.capability.root / "backfill" / "nights" / f"night-{NIGHT}"
        night_root.parent.mkdir(parents=True)
        shutil.copytree(made.candidate_dir.parent, night_root)
        self.candidate = load_backfill_candidate(night_root)
        self.publisher = F.publisher_for(self.capability)
        self.authorization = F.authorize(self.publisher, self.candidate)

    def tearDown(self):
        self._tmp.cleanup()

    def cumulative_rows(self):
        paths = history.cumulative_paths(self.capability.published_root)
        index = pd.read_parquet(paths["loci_index"])
        summary = pd.read_parquet(paths["nightly_summary"])
        return (
            int(index["night_date_utc"].eq(NIGHT).sum()),
            int(summary["date_utc"].eq(NIGHT).sum()),
        )

    def controller_stage(self):
        document = inspect_backfill(
            self.capability.root, self.capability.published_root, NIGHT, NIGHT
        )
        return document["nights"][0]["stage"]

    def test_blocker1_post_manifest_fault_is_truthful_and_resumable(self):
        failing = F.publisher_for(
            self.capability, fault_hook=_FailOnce("after_manifest_commit")
        )
        outcome = failing.publish(self.candidate, self.authorization)
        self.assertFalse(outcome.success)
        self.assertTrue(outcome.record["production_mutated"], outcome.record)
        self.assertNotEqual(outcome.record["status"], "UNPUBLISHED_FAILURE")
        staged = [
            path for path in self.capability.staging_root.iterdir()
            if path.name.endswith("-cumulative")
        ]
        self.assertEqual(len(staged), 1)
        self.assertEqual(
            sorted(p.name for p in staged[0].iterdir()),
            ["loci_index.parquet", "nightly_summary.parquet"],
        )
        self.assertNotEqual(self.controller_stage(), "PUBLISHED")
        retried = self.publisher.publish(self.candidate, self.authorization)
        self.assertTrue(retried.success, retried.record)
        self.assertEqual(self.cumulative_rows(), (self.candidate.loci, 1))
        self.assertEqual(self.controller_stage(), "PUBLISHED")

    def test_blocker2_stale_summary_is_never_derived_published(self):
        failing = F.publisher_for(
            self.capability,
            fault_hook=_FailOnce("before_cumulative_install:nightly_summary"),
        )
        outcome = failing.publish(self.candidate, self.authorization)
        self.assertFalse(outcome.success)
        self.assertEqual(self.cumulative_rows(), (self.candidate.loci, 0))
        self.assertNotEqual(self.controller_stage(), "PUBLISHED")
        self.assertEqual(self.controller_stage(), "RECONCILIATION_REQUIRED")
        # Repeated resume repairs deterministically and then changes nothing.
        first = self.publisher.publish(self.candidate, self.authorization)
        second = self.publisher.publish(self.candidate, self.authorization)
        self.assertTrue(first.success, first.record)
        self.assertEqual(second.status, "already_published")
        self.assertEqual(self.cumulative_rows(), (self.candidate.loci, 1))
        self.assertEqual(self.controller_stage(), "PUBLISHED")


if __name__ == "__main__":
    unittest.main()
