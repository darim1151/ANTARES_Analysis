import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.cli_profiles import StorageProfile
from src.operations import (
    DevelopmentWriteCapability,
    ExecutionState,
    ExecutionStateMachine,
    ExitCode,
    IllegalTransition,
    Issue,
    LockOwnershipError,
    LockUnavailable,
    OperationReport,
    PublicationTransaction,
    QueryFetchEvidence,
    StorageContractError,
    StorageLayout,
    TransactionError,
    WriterLock,
    contained_path,
    context_from_environment,
    context_from_profile,
    plan_backfill,
    plan_night,
    validate_root_separation,
    valid_zero_row_evidence,
)


FIXED_TIME = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def fixed_clock():
    return FIXED_TIME


class OperationContextTests(unittest.TestCase):
    def test_construction_is_explicit_deterministic_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / "missing-data"
            cache = base / "missing-cache"
            profile = StorageProfile(
                "fixture", "fixture", data, cache, "private", source="test"
            )
            context = context_from_profile(
                profile,
                run_id="run-fixed",
                execution_metadata={"z": 2, "a": 1},
                clock=fixed_clock,
            )

            self.assertFalse(data.exists())
            self.assertFalse(cache.exists())
            self.assertEqual(context.run_id, "run-fixed")
            self.assertEqual(context.execution_metadata, (("a", "1"), ("z", "2")))
            self.assertEqual(context.now(), FIXED_TIME)

    def test_environment_resolution_supports_private_and_shared_group(self):
        private = context_from_environment(
            "environment",
            environ={
                "ANTARES_ANALYSIS_DATA_ROOT": "/tmp/private-data",
                "ANTARES_ANALYSIS_CACHE_ROOT": "/tmp/private-cache",
                "ANTARES_STORAGE_POLICY": "private",
            },
            run_id="private",
            clock=fixed_clock,
        )
        shared = context_from_environment(
            "environment",
            environ={
                "ANTARES_ANALYSIS_DATA_ROOT": "/tmp/shared-data",
                "ANTARES_ANALYSIS_CACHE_ROOT": "/tmp/shared-cache",
                "ANTARES_STORAGE_POLICY": "shared-group",
                "ANTARES_SHARED_GROUP": "g_antares_analysis",
            },
            run_id="shared",
            clock=fixed_clock,
        )

        self.assertEqual(private.storage_policy, "private")
        self.assertIsNone(private.shared_group)
        self.assertEqual(shared.storage_policy, "shared-group")
        self.assertEqual(shared.shared_group, "g_antares_analysis")


class OperationReportTests(unittest.TestCase):
    def _report(self):
        return OperationReport(
            operation="fixture",
            success=False,
            status="refused",
            started_at_utc="2026-08-23T12:00:00+00:00",
            finished_at_utc="2026-08-23T12:00:01+00:00",
            elapsed_seconds=1.0,
            exit_code=ExitCode.REFUSED,
            warnings=(Issue("warning", "Warning text"),),
            errors=(Issue("error", "Error text"),),
            refusal_reasons=(Issue("disabled", "Writer disabled"),),
            counts={"b": 2, "a": 1},
            details={"nested": {"b": 2, "a": 1}},
        )

    def test_json_and_human_rendering_are_stable(self):
        report = self._report()
        first = report.to_json()
        second = report.to_json()

        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["exit_code"], 4)
        self.assertNotIn("object at", first)
        human = report.render_human()
        self.assertIn("disabled", human)
        self.assertIn("a=1, b=2", human)

    def test_serialization_rejects_unknown_objects(self):
        report = OperationReport(
            operation="fixture",
            success=True,
            status="ok",
            started_at_utc="2026-08-23T12:00:00+00:00",
            finished_at_utc="2026-08-23T12:00:00+00:00",
            elapsed_seconds=0,
            details={"bad": object()},
        )
        with self.assertRaisesRegex(TypeError, "cannot serialize"):
            report.to_json()


class ExecutionStateTests(unittest.TestCase):
    def test_complete_legal_path(self):
        machine = ExecutionStateMachine()
        for state in (
            ExecutionState.PRECHECKED,
            ExecutionState.LOCKED,
            ExecutionState.QUERYING,
            ExecutionState.FETCHING,
            ExecutionState.STAGED,
            ExecutionState.VALIDATED,
            ExecutionState.PUBLISHED,
            ExecutionState.RECONCILING,
            ExecutionState.COMPLETE,
        ):
            machine.transition(state, at=FIXED_TIME)

        snapshot = machine.snapshot()
        self.assertEqual(snapshot.state, ExecutionState.COMPLETE)
        self.assertTrue(snapshot.published)
        self.assertFalse(snapshot.reconciliation_required)
        self.assertEqual(len(snapshot.transitions), 9)

    def test_illegal_transition_fails_loudly(self):
        machine = ExecutionStateMachine()
        with self.assertRaisesRegex(IllegalTransition, "planned -> published"):
            machine.transition(ExecutionState.PUBLISHED)

    def test_failure_before_publication_is_not_published(self):
        machine = ExecutionStateMachine()
        machine.transition(ExecutionState.PRECHECKED)
        snapshot = machine.fail("preflight_failed")

        self.assertEqual(snapshot.state, ExecutionState.FAILED)
        self.assertFalse(snapshot.published)
        self.assertFalse(snapshot.reconciliation_required)

    def test_reconciliation_failure_preserves_publication(self):
        machine = ExecutionStateMachine()
        for state in (
            ExecutionState.PRECHECKED,
            ExecutionState.LOCKED,
            ExecutionState.QUERYING,
            ExecutionState.FETCHING,
            ExecutionState.STAGED,
            ExecutionState.VALIDATED,
            ExecutionState.PUBLISHED,
            ExecutionState.RECONCILING,
        ):
            machine.transition(state)
        snapshot = machine.fail("cumulative_rebuild_failed")

        self.assertEqual(snapshot.state, ExecutionState.FAILED)
        self.assertTrue(snapshot.published)
        self.assertTrue(snapshot.reconciliation_required)


class StorageContractTests(unittest.TestCase):
    def test_traversal_and_absolute_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for target in (Path("../outside"), Path("/tmp/outside")):
                with self.subTest(target=target):
                    with self.assertRaises(StorageContractError):
                        contained_path(root, target)

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            outside = Path(temporary) / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "link").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(StorageContractError, "symlink"):
                contained_path(root, Path("link") / "partition")

    def test_data_cache_overlap_fails_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(StorageContractError):
                validate_root_separation(root / "data", root / "data" / "cache")
            with self.assertRaises(StorageContractError):
                validate_root_separation(root / "cache" / "data", root / "cache")
            with self.assertRaises(StorageContractError):
                validate_root_separation(root / "same", root / "same")

            real = root / "real"
            alias = root / "alias"
            real.mkdir()
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(StorageContractError):
                validate_root_separation(real, alias)

    def test_manifest_science_paths_use_siblings_not_embedded_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            data = base / "data-root"
            cache = base / "cache-root"
            layout = StorageLayout(data, cache)
            target = layout.night("2026-06-27")
            target.directory.mkdir(parents=True)
            target.manifest.write_text(
                json.dumps(
                    {
                        "paths": {
                            "loci": "/stale/outside/loci.parquet",
                            "alerts": "/stale/outside/alerts.parquet",
                        }
                    }
                ),
                encoding="utf-8",
            )

            loci, alerts = layout.manifest_science_paths(target.manifest)
            self.assertEqual(loci, target.loci)
            self.assertEqual(alerts, target.alerts)

    def test_zero_row_requires_accepted_date_and_clean_completed_evidence(self):
        manifest = {
            "date_utc": "2026-03-05",
            "status": "complete",
            "actual_loci": 0,
            "alert_rows": 0,
            "chunk_count": 1,
            "finished_at_utc": "2026-03-06T00:00:00+00:00",
            "validation": {
                "append_ready": True,
                "query_completed_pass": True,
                "query_fetch_clean": True,
                "zero_row_schema_pass": True,
            },
        }
        self.assertTrue(valid_zero_row_evidence(manifest))
        failed = dict(manifest, query_error="timeout")
        self.assertFalse(valid_zero_row_evidence(failed))
        unknown = dict(manifest, date_utc="2026-03-12")
        self.assertFalse(valid_zero_row_evidence(unknown))


class PlannerTests(unittest.TestCase):
    def _context(self, root, cache):
        profile = StorageProfile(
            "fixture", "fixture", Path(root), Path(cache), "private", source="test"
        )
        return context_from_profile(
            profile, run_id="run-plan", clock=fixed_clock
        )

    @staticmethod
    def _snapshot(root):
        root = Path(root)
        return [
            (
                path.relative_to(root).as_posix(),
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in sorted(root.rglob("*"))
        ]

    @staticmethod
    def _complete_night(layout, night, *, zero=False, clean=True):
        target = layout.night(night)
        target.directory.mkdir(parents=True)
        target.loci.write_bytes(b"PAR1-loci")
        target.alerts.write_bytes(b"PAR1-alerts")
        loci = alerts = 0 if zero else 3
        manifest = {
            "date_utc": night,
            "status": "complete",
            "actual_loci": loci,
            "alert_rows": alerts,
            "chunk_count": 1,
            "finished_at_utc": "2026-03-06T00:00:00+00:00",
            "validation": {
                "append_ready": True,
                "query_completed_pass": clean,
                "query_fetch_clean": clean,
                "zero_row_schema_pass": True if zero else None,
            },
        }
        if not clean:
            manifest["query_error"] = "timeout"
        target.manifest.write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def test_missing_night_plan_is_deterministic_and_non_mutating(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            context = self._context(root, cache)
            before = self._snapshot(root)
            first = plan_night(context, "2026-06-27")
            second = plan_night(context, "2026-06-27")
            after = self._snapshot(root)

            self.assertEqual(first.to_json(), second.to_json())
            self.assertEqual(before, after)
            self.assertTrue(first.success)
            self.assertEqual(first.details["current_partition"]["state"], "missing")
            self.assertFalse(first.details["future_execution"]["authorized"])
            self.assertIsNone(first.details["storage"]["estimated_bytes"])

    def test_existing_complete_and_incomplete_nights_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            layout = StorageLayout(root, cache)
            complete = self._complete_night(layout, "2026-06-27")
            incomplete = layout.night("2026-06-28")
            incomplete.directory.mkdir(parents=True)
            incomplete.loci.write_bytes(b"partial")
            context = self._context(root, cache)

            complete_report = plan_night(context, "2026-06-27")
            incomplete_report = plan_night(context, "2026-06-28")

            self.assertEqual(
                complete_report.details["current_partition"]["state"], "complete"
            )
            self.assertIn(
                "target_already_complete",
                [item.code for item in complete_report.refusal_reasons],
            )
            self.assertEqual(
                incomplete_report.details["current_partition"]["state"], "incomplete"
            )
            self.assertIn("target-partition", incomplete_report.details["blockers"])
            self.assertTrue(complete.manifest.is_file())

    def test_conflicting_partition_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            layout = StorageLayout(root, cache)
            conflicting = layout.night("2026-06-29")
            conflicting.directory.parent.mkdir(parents=True)
            conflicting.directory.write_text("not-a-directory", encoding="utf-8")

            report = plan_night(self._context(root, cache), "2026-06-29")

            self.assertEqual(
                report.details["current_partition"]["state"], "conflicting"
            )
            self.assertIn("target-partition", report.details["blockers"])

    def test_accepted_zero_row_and_failed_zero_row_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            layout = StorageLayout(root, cache)
            self._complete_night(layout, "2026-03-05", zero=True, clean=True)
            self._complete_night(layout, "2026-03-11", zero=True, clean=False)
            context = self._context(root, cache)

            accepted = plan_night(context, "2026-03-05")
            failed = plan_night(context, "2026-03-11")

            self.assertEqual(accepted.details["current_partition"]["state"], "complete")
            self.assertTrue(
                accepted.details["current_partition"]["zero_row_evidence_valid"]
            )
            self.assertEqual(failed.details["current_partition"]["state"], "incomplete")
            self.assertFalse(failed.details["current_partition"]["zero_row_evidence_valid"])

    def test_invalid_date_and_backfill_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            context = self._context(root, cache)

            invalid = plan_night(context, "2026-6-7")
            backwards = plan_backfill(context, "2026-06-29", "2026-06-27")
            planned = plan_backfill(context, "2026-06-27", "2026-06-29")

            self.assertEqual(invalid.exit_code, ExitCode.INVALID_REQUEST)
            self.assertEqual(backwards.exit_code, ExitCode.INVALID_REQUEST)
            self.assertEqual(planned.counts["requested_nights"], 3)
            self.assertEqual(
                [item["night"] for item in planned.details["nights"]],
                ["2026-06-27", "2026-06-28", "2026-06-29"],
            )
            self.assertTrue(planned.details["stop_on_first_anomaly"])


class LockAndTransactionTests(unittest.TestCase):
    TARGET = Path("data/lsst_only/nightly/2026/06/27")

    def _capability(self, temporary):
        root = Path(temporary) / "transaction-root"
        root.mkdir()
        return DevelopmentWriteCapability.for_temporary_root(root)

    def _lock(self, capability, run_id="run-1"):
        lock = WriterLock(
            capability,
            self.TARGET.as_posix(),
            run_id,
            owner="test-owner",
            hostname="test-host",
            pid=123,
        )
        lock.acquire(at=FIXED_TIME)
        return lock

    @staticmethod
    def _artifacts(loci=1, alerts=1, zero_validation=False):
        validation = {"append_ready": True}
        if zero_validation:
            validation.update(
                {
                    "query_completed_pass": True,
                    "query_fetch_clean": True,
                    "zero_row_schema_pass": True,
                }
            )
        manifest = {
            "date_utc": "2026-06-27",
            "actual_loci": loci,
            "alert_rows": alerts,
            "validation": validation,
        }
        return {
            "loci.parquet": b"PAR1-loci",
            "alerts.parquet": b"PAR1-alerts",
            "manifest.json": json.dumps(manifest).encode("utf-8"),
        }

    def _staged(self, capability, lock, evidence=None, artifacts=None):
        transaction = PublicationTransaction(
            capability, lock, self.TARGET, lock.run_id
        )
        transaction.prepare()
        transaction.begin_query()
        transaction.begin_fetch()
        transaction.stage_artifacts(
            artifacts or self._artifacts(),
            evidence
            or QueryFetchEvidence(True, True, loci_rows=1, alert_rows=1),
        )
        return transaction

    def test_lock_refuses_second_writer_and_releases_only_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            first = self._lock(capability)
            second = WriterLock(
                capability, self.TARGET.as_posix(), "run-2", owner="other"
            )
            with self.assertRaisesRegex(LockUnavailable, "not be stolen"):
                second.acquire()
            with self.assertRaisesRegex(LockOwnershipError, "does not own"):
                second.release()

            inspection = first.inspect(
                now=FIXED_TIME + timedelta(days=2), stale_after_seconds=60
            )
            self.assertTrue(inspection.stale_candidate)
            self.assertTrue(first.path.exists())
            first.release()
            self.assertFalse(first.path.exists())

    def test_ambiguous_lock_is_never_stolen(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = WriterLock(
                capability, self.TARGET.as_posix(), "run-ambiguous"
            )
            lock.path.mkdir(parents=True)

            self.assertTrue(lock.inspect().ambiguous)
            with self.assertRaisesRegex(LockUnavailable, "not be stolen"):
                lock.acquire()
            self.assertTrue(lock.path.exists())

    def test_transaction_cannot_publish_before_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = self._staged(capability, lock)

            with self.assertRaisesRegex(TransactionError, "requires a validated"):
                transaction.publish()
            transaction.abort()
            lock.release()

    def test_validation_failure_is_terminal_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = self._staged(capability, lock)

            with self.assertRaisesRegex(TransactionError, "validation failed"):
                transaction.validate(lambda stage: False)
            self.assertEqual(transaction.state, ExecutionState.FAILED)
            self.assertFalse(transaction.target.exists())
            transaction.abort()
            lock.release()

    def test_staging_write_failure_is_terminal_and_abort_cleans_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = PublicationTransaction(
                capability, lock, self.TARGET, lock.run_id
            )
            transaction.prepare()
            transaction.begin_query()
            transaction.begin_fetch()
            artifacts = self._artifacts()
            artifacts["alerts.parquet"] = None

            with self.assertRaises(TypeError):
                transaction.stage_artifacts(
                    artifacts,
                    QueryFetchEvidence(True, True, loci_rows=1, alert_rows=1),
                )
            self.assertEqual(transaction.state, ExecutionState.FAILED)
            stage = transaction.stage
            transaction.abort()
            self.assertFalse(stage.exists())
            self.assertFalse(transaction.target.exists())
            lock.release()

    def test_successful_publication_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = self._staged(capability, lock)
            transaction.validate(
                lambda stage: all(
                    (stage / name).is_file()
                    for name in transaction.REQUIRED_ARTIFACTS
                )
            )
            published = transaction.publish()
            with self.assertRaisesRegex(TransactionError, "Release the nightly"):
                transaction.begin_reconciliation()
            transaction.release_writer_lock()
            transaction.begin_reconciliation()
            complete = transaction.complete_reconciliation()

            self.assertTrue(published.published)
            self.assertEqual(complete.state, ExecutionState.COMPLETE)
            self.assertTrue(transaction.target.is_dir())
            self.assertTrue((transaction.target / "manifest.json").is_file())

    def test_existing_partition_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            target = capability.root / self.TARGET
            target.mkdir(parents=True)
            marker = target / "existing"
            marker.write_text("preserve", encoding="utf-8")
            lock = self._lock(capability)
            transaction = PublicationTransaction(
                capability, lock, self.TARGET, lock.run_id
            )

            with self.assertRaisesRegex(TransactionError, "overwrite refused"):
                transaction.prepare()
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            lock.release()

    def test_abort_before_publication_removes_only_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = self._staged(capability, lock)
            stage = transaction.stage
            snapshot = transaction.abort("validation_failed")

            self.assertEqual(snapshot.state, ExecutionState.FAILED)
            self.assertFalse(snapshot.published)
            self.assertFalse(stage.exists())
            self.assertFalse(transaction.target.exists())
            lock.release()

    def test_failure_after_publication_never_deletes_science(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = self._staged(capability, lock)
            transaction.validate(lambda stage: True)
            transaction.publish()
            transaction.release_writer_lock()
            transaction.begin_reconciliation()
            snapshot = transaction.fail_reconciliation("summary_failed")
            transaction.abort("cleanup_requested")

            self.assertTrue(snapshot.published)
            self.assertTrue(snapshot.reconciliation_required)
            self.assertTrue(transaction.target.is_dir())

    def test_query_failure_cannot_become_zero_row_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            transaction = PublicationTransaction(
                capability, lock, self.TARGET, lock.run_id
            )
            transaction.prepare()
            transaction.begin_query()
            transaction.begin_fetch()
            evidence = QueryFetchEvidence(
                False,
                False,
                loci_rows=0,
                alert_rows=0,
                query_errors=("timeout",),
                zero_row_proof=None,
            )

            with self.assertRaisesRegex(TransactionError, "cannot be staged"):
                transaction.stage_artifacts(
                    self._artifacts(0, 0, zero_validation=True), evidence
                )
            self.assertEqual(transaction.state, ExecutionState.FAILED)
            transaction.abort()
            lock.release()

    def test_clean_zero_row_requires_manifest_success_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            lock = self._lock(capability)
            evidence = QueryFetchEvidence(
                True,
                True,
                loci_rows=0,
                alert_rows=0,
                zero_row_proof="completed_successful_query",
            )
            transaction = self._staged(
                capability,
                lock,
                evidence=evidence,
                artifacts=self._artifacts(0, 0, zero_validation=False),
            )

            with self.assertRaisesRegex(TransactionError, "Zero-row manifest"):
                transaction.validate(lambda stage: True)
            transaction.abort()
            lock.release()

    def test_production_path_cannot_receive_development_capability(self):
        with self.assertRaisesRegex(StorageContractError, "temporary"):
            DevelopmentWriteCapability.for_temporary_root(Path.home())
        with self.assertRaisesRegex(StorageContractError, "factory"):
            DevelopmentWriteCapability(Path.home(), object())


if __name__ == "__main__":
    unittest.main()
