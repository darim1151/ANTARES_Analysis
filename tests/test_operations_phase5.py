import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from src import cli, history
from src.operations import (
    DevelopmentWriteCapability,
    ExecutionState,
    JournalCorrupt,
    JournalError,
    JournalOutcome,
    NightExecutionSpec,
    RecoveryDisposition,
    StorageContractError,
    SyntheticWriteCapability,
    TransactionDescriptor,
    TransactionJournal,
    TransactionalNightWriter,
    WriterLock,
    WriterError,
    execute_synthetic_night,
    inspect_recovery,
)
from src.operations.journal import ArtifactIdentity
from src.operations.canary import (
    CANARY_IDENTITY_NAME,
    _inventory,
    _inventory_contract,
    _remove_inventoried_tree,
)
from src.operations.science import (
    ArtifactValidationError,
    NightScienceRequest,
    SyntheticScenario,
    SyntheticScienceProvider,
    build_night_artifacts,
    reopen_and_validate_artifacts,
)
from src.operations.writer import FailureInjector
from src.operations.writer import SyntheticReconciler


RELEASE_SHA = "5" * 40


def _request(day=27):
    return NightScienceRequest(
        f"2026-06-{day:02d}",
        61218.0 + day - 27,
        61219.0 + day - 27,
        target_loci=2,
    )


def _spec(transaction_id, request=None):
    return NightExecutionSpec(
        transaction_id,
        "plan-phase5-test",
        RELEASE_SHA,
        "phase5-test-configuration",
        request or _request(),
    )


def _death_worker(root_text, run_id, transaction_id, point):
    root = Path(root_text)
    capability = SyntheticWriteCapability.for_local_run_root(root, run_id)

    def terminate(observed, details):
        del details
        if observed == point:
            os._exit(91)

    execute_synthetic_night(
        capability,
        SyntheticScienceProvider(),
        _spec(transaction_id),
        fault_hook=terminate,
    )
    os._exit(0)


def _concurrent_worker(
    root_text,
    run_id,
    transaction_id,
    day,
    result_queue,
    acquired_event=None,
    release_event=None,
):
    capability = SyntheticWriteCapability.for_local_run_root(
        Path(root_text), run_id
    )

    def hold(point, details):
        del details
        if point == "after_lock" and acquired_event is not None:
            acquired_event.set()
            if release_event is not None:
                release_event.wait(20)

    report = execute_synthetic_night(
        capability,
        SyntheticScienceProvider(),
        _spec(transaction_id, _request(day)),
        fault_hook=hold,
    )
    result_queue.put(
        {
            "transaction_id": transaction_id,
            "success": report.success,
            "status": report.status,
            "errors": [item.code for item in report.errors],
        }
    )


class CapabilityAndAuthorizationTests(unittest.TestCase):
    def test_synthetic_layout_is_separated_and_run_scoped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run-a"
            root.mkdir()
            capability = SyntheticWriteCapability.for_local_run_root(
                root, "run-a"
            )
            canonical = root.resolve()
            self.assertEqual(capability.published_root, canonical / "published")
            self.assertEqual(capability.staging_root, canonical / "staging")
            self.assertEqual(capability.lock_root, canonical / "control" / "locks")
            self.assertEqual(
                capability.journal_root, canonical / "control" / "journals"
            )
            self.assertEqual(capability.evidence_root, canonical / "evidence")
            self.assertEqual(list(root.iterdir()), [])

    def test_synthetic_capability_rejects_non_temp_and_forgery(self):
        with self.assertRaisesRegex(StorageContractError, "temporary"):
            SyntheticWriteCapability.for_local_run_root(Path.home(), "home")
        with self.assertRaisesRegex(StorageContractError, "sealed"):
            SyntheticWriteCapability(Path("/tmp/x"), "x", "local-temporary", object())

    def test_arnor_capability_rejects_wrong_host_and_wrong_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            with self.assertRaisesRegex(StorageContractError, "host arnor"):
                SyntheticWriteCapability.for_arnor_canary_root(
                    root, "run", hostname="not-arnor"
                )
            with self.assertRaisesRegex(StorageContractError, "exact direct child"):
                SyntheticWriteCapability.for_arnor_canary_root(
                    root, "run", hostname="arnor"
                )

    def test_no_production_capability_factory_exists(self):
        self.assertFalse(hasattr(SyntheticWriteCapability, "for_production_root"))
        self.assertFalse(hasattr(SyntheticWriteCapability, "for_data_root"))

    def test_writer_rejects_development_capability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "legacy"
            root.mkdir()
            capability = DevelopmentWriteCapability.for_temporary_root(root)
            with self.assertRaisesRegex(WriterError, "synthetic"):
                TransactionalNightWriter(capability, SyntheticScienceProvider())

    def test_writer_rejects_synthetic_provider_and_capability_subclasses(self):
        class ProviderSubclass(SyntheticScienceProvider):
            pass

        class CapabilitySubclass(SyntheticWriteCapability):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            capability = SyntheticWriteCapability.for_local_run_root(root, "run")
            with self.assertRaisesRegex(WriterError, "live science-provider"):
                TransactionalNightWriter(capability, ProviderSubclass())
            subclass_capability = CapabilitySubclass.for_local_run_root(root, "run")
            with self.assertRaisesRegex(WriterError, "sealed synthetic"):
                TransactionalNightWriter(
                    subclass_capability, SyntheticScienceProvider()
                )

    def test_legacy_ingest_fails_before_query_or_filesystem_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "must-remain-absent"
            with mock.patch(
                "src.chunked_query.query_range_adaptive",
                side_effect=AssertionError("live query must not run"),
            ):
                with self.assertRaises(history.ProductionWriterUnavailable):
                    history.ingest_night(
                        data_root=target,
                        mjd_min=61218,
                        mjd_max=61219,
                    )
            self.assertFalse(target.exists())

    def test_cli_ingest_is_a_deterministic_pre_provider_refusal(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(["night", "ingest", "2026-06-27", "--json"])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(status, 4)
        self.assertEqual(stderr.getvalue(), "")
        self.assertFalse(payload["details"]["provider_constructed"])
        self.assertFalse(payload["details"]["writer_capability_issued"])
        self.assertEqual(
            payload["refusal_reasons"][0]["code"],
            "production_authorization_unavailable",
        )

    def test_cli_ingest_rejects_invalid_date_without_traceback(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(["night", "ingest", "not-a-date", "--json"])
        self.assertEqual(status, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("canonical YYYY-MM-DD", stderr.getvalue())


class SyntheticScienceTests(unittest.TestCase):
    def test_nonzero_and_zero_round_trip_with_existing_science_validation(self):
        for scenario, counts in (
            (SyntheticScenario.SUCCESS_NONZERO.value, (2, 4)),
            (SyntheticScenario.SUCCESS_ZERO.value, (0, 0)),
        ):
            with self.subTest(scenario=scenario):
                result = SyntheticScienceProvider(scenario).fetch_night(_request())
                artifacts = build_night_artifacts(result)
                reopened = reopen_and_validate_artifacts(artifacts, expected=result)
                self.assertEqual((len(reopened.loci), len(reopened.alerts)), counts)

    def test_successful_zero_has_positive_completion_proof(self):
        result = SyntheticScienceProvider("success_zero").fetch_night(_request())
        self.assertTrue(result.evidence.clean)
        self.assertTrue(result.evidence.valid_zero_row)
        self.assertEqual(
            result.evidence.zero_row_proof, "completed_successful_query"
        )

    def test_all_failure_scenarios_are_not_publishable(self):
        for scenario in (
            "query_failure",
            "query_interruption",
            "fetch_failure",
            "partial_fetch",
            "malformed_result",
            "validation_failure",
        ):
            with self.subTest(scenario=scenario):
                result = SyntheticScienceProvider(scenario).fetch_night(_request())
                self.assertFalse(result.publishable)
                with self.assertRaises(Exception):
                    build_night_artifacts(result)

    def test_artifacts_are_deterministic(self):
        provider = SyntheticScienceProvider()
        first = build_night_artifacts(provider.fetch_night(_request()))
        second = build_night_artifacts(provider.fetch_night(_request()))
        self.assertEqual(first, second)

    def test_corrupt_artifact_fails_independent_reopen(self):
        result = SyntheticScienceProvider().fetch_night(_request())
        artifacts = build_night_artifacts(result)
        artifacts["loci.parquet"] += b"corrupt"
        with self.assertRaises(ArtifactValidationError):
            reopen_and_validate_artifacts(artifacts, expected=result)

    def test_non_lsst_request_is_refused(self):
        with self.assertRaisesRegex(ValueError, "LSST-only"):
            NightScienceRequest("2026-06-27", 61218, 61219, lsst_only=False)


class JournalTests(unittest.TestCase):
    def _journal(self, root, run_id="txn"):
        parent = Path(root) / "journals"
        parent.mkdir(exist_ok=True)
        path = parent / f"{run_id}.json"
        descriptor = TransactionDescriptor(
            run_id=run_id,
            operation="test",
            target_identity="data/night",
            target_path=str(Path(root) / "target"),
            stage_path=str(Path(root) / "stage"),
            lock_path=str(Path(root) / "lock"),
            profile="test",
            plan_id="plan",
            release_sha=RELEASE_SHA,
            metadata={"provider": "synthetic"},
        )
        return path, TransactionJournal.create(path, descriptor)

    @staticmethod
    def _advance(journal, through):
        order = (
            ExecutionState.PRECHECKED,
            ExecutionState.LOCKED,
            ExecutionState.QUERYING,
            ExecutionState.FETCHING,
            ExecutionState.STAGED,
            ExecutionState.VALIDATED,
            ExecutionState.PUBLISHED,
            ExecutionState.RECONCILING,
            ExecutionState.COMPLETE,
        )
        for state in order:
            journal.transition(state)
            if state == through:
                break

    def test_atomic_journal_round_trip_and_monotonic_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, journal = self._journal(temporary)
            journal.transition(ExecutionState.PRECHECKED)
            journal.update(validation={"passed": True})
            loaded = TransactionJournal.load(path).snapshot
            self.assertEqual(loaded.revision, 2)
            self.assertEqual(loaded.state, ExecutionState.PRECHECKED)
            self.assertEqual(loaded.outcome, JournalOutcome.ACTIVE)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_unpublished_failure_has_explicit_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, journal = self._journal(temporary)
            journal.transition(ExecutionState.FAILED, reason="preflight_failed")
            self.assertEqual(
                journal.snapshot.outcome, JournalOutcome.UNPUBLISHED_FAILURE
            )

    def test_postcommit_outcomes_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, uncertain = self._journal(temporary, "uncertain")
            self._advance(uncertain, ExecutionState.PUBLISHED)
            uncertain.update(durability={"status": "uncertain"})
            self.assertEqual(
                uncertain.snapshot.outcome,
                JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            )

            _, reconcile = self._journal(temporary, "reconcile")
            self._advance(reconcile, ExecutionState.PUBLISHED)
            reconcile.transition(
                ExecutionState.FAILED,
                reason="reconciliation_failed",
                durability={"status": "client_fsync_confirmed"},
                reconciliation={"status": "failed"},
            )
            self.assertEqual(
                reconcile.snapshot.outcome,
                JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED,
            )

    def test_complete_outcome_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, journal = self._journal(temporary)
            self._advance(journal, ExecutionState.COMPLETE)
            self.assertEqual(
                TransactionJournal.load(path).snapshot.outcome,
                JournalOutcome.COMPLETE,
            )

    def test_illegal_transition_and_revision_conflict_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, journal = self._journal(temporary)
            second = TransactionJournal.load(path)
            journal.transition(ExecutionState.PRECHECKED)
            with self.assertRaisesRegex(JournalError, "Illegal"):
                journal.transition(ExecutionState.PUBLISHED)
            with self.assertRaisesRegex(JournalError, "revision conflict"):
                second.update(validation={"stale": True})

    def test_torn_duplicate_key_and_unknown_schema_are_corrupt(self):
        variants = (
            b'{"schema_version":',
            b'{"schema_version":"1.0","schema_version":"1.0"}',
            json.dumps({"schema_version": "999"}).encode("utf-8"),
        )
        for index, payload in enumerate(variants):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary) / "journals"
                parent.mkdir()
                path = parent / "bad.json"
                path.write_bytes(payload)
                with self.assertRaises(JournalCorrupt):
                    TransactionJournal.load(path)

    def test_failed_atomic_update_preserves_loadable_previous_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, journal = self._journal(temporary)
            before = path.read_bytes()
            with mock.patch(
                "src.operations.journal.os.replace",
                side_effect=OSError("synthetic replace failure"),
            ):
                with self.assertRaises(JournalError):
                    journal.update(validation={"new": True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(TransactionJournal.load(path).snapshot.revision, 0)


class WriterAndRecoveryTests(unittest.TestCase):
    def _capability(self, temporary, run_id="phase5-run"):
        root = Path(temporary) / run_id
        root.mkdir()
        return SyntheticWriteCapability.for_local_run_root(root, run_id)

    def _assessment(self, capability, transaction_id):
        path = capability.journal_root / f"{transaction_id}.json"
        snapshot = TransactionJournal.load(path).snapshot
        return inspect_recovery(
            path,
            target_path=Path(snapshot.descriptor.target_path),
            stage_path=Path(snapshot.descriptor.stage_path),
            lock_path=Path(snapshot.descriptor.lock_path),
        )

    def test_golden_nonzero_and_zero_writer_lifecycle(self):
        for scenario in ("success_nonzero", "success_zero"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                capability = self._capability(temporary)
                transaction_id = f"txn-{scenario}"
                report = execute_synthetic_night(
                    capability,
                    SyntheticScienceProvider(scenario),
                    _spec(transaction_id),
                )
                snapshot = TransactionJournal.load(
                    capability.journal_root / f"{transaction_id}.json"
                ).snapshot
                self.assertTrue(report.success)
                self.assertEqual(snapshot.state, ExecutionState.COMPLETE)
                self.assertEqual(snapshot.outcome, JournalOutcome.COMPLETE)
                self.assertEqual(
                    [item.current for item in snapshot.transitions],
                    [
                        ExecutionState.PRECHECKED,
                        ExecutionState.LOCKED,
                        ExecutionState.QUERYING,
                        ExecutionState.FETCHING,
                        ExecutionState.STAGED,
                        ExecutionState.VALIDATED,
                        ExecutionState.PUBLISHED,
                        ExecutionState.RECONCILING,
                        ExecutionState.COMPLETE,
                    ],
                )
                assessment = self._assessment(capability, transaction_id)
                self.assertEqual(
                    assessment.dispositions,
                    (RecoveryDisposition.MUST_NOT_AUTO_DELETE,),
                )

    def test_provider_failures_never_publish(self):
        for scenario in (
            "query_failure",
            "query_interruption",
            "fetch_failure",
            "partial_fetch",
            "malformed_result",
            "validation_failure",
        ):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                capability = self._capability(temporary)
                transaction_id = f"txn-{scenario}"
                report = execute_synthetic_night(
                    capability,
                    SyntheticScienceProvider(scenario),
                    _spec(transaction_id),
                )
                snapshot = TransactionJournal.load(
                    capability.journal_root / f"{transaction_id}.json"
                ).snapshot
                self.assertFalse(report.success)
                self.assertFalse(snapshot.published)
                self.assertEqual(
                    snapshot.outcome, JournalOutcome.UNPUBLISHED_FAILURE
                )
                self.assertFalse(Path(snapshot.descriptor.target_path).exists())

    def test_failure_injection_distinguishes_publication_outcomes(self):
        cases = {
            "before_preflight": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_preflight": JournalOutcome.UNPUBLISHED_FAILURE,
            "before_lock": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_lock": JournalOutcome.UNPUBLISHED_FAILURE,
            "during_query": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_successful_query": JournalOutcome.UNPUBLISHED_FAILURE,
            "during_fetch": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_successful_fetch": JournalOutcome.UNPUBLISHED_FAILURE,
            "during_staging": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_staging": JournalOutcome.UNPUBLISHED_FAILURE,
            "during_validation": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_validation": JournalOutcome.UNPUBLISHED_FAILURE,
            "during_precommit_reproof": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_precommit_reproof": JournalOutcome.UNPUBLISHED_FAILURE,
            "immediately_before_publication": JournalOutcome.UNPUBLISHED_FAILURE,
            "before_target_reservation": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_target_reservation": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_data_links": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_pending_manifest": JournalOutcome.UNPUBLISHED_FAILURE,
            "before_manifest_commit": JournalOutcome.UNPUBLISHED_FAILURE,
            "after_manifest_commit": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "after_publication_fsync": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "after_pending_unlink": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "before_stage_cleanup": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "after_stage_cleanup": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "during_durability_confirmation": JournalOutcome.PUBLISHED_DURABILITY_UNCERTAIN,
            "before_unlock": JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED,
            "during_reconciliation": JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED,
            "after_reconciliation_before_completion": JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED,
        }
        for point, outcome in cases.items():
            with self.subTest(point=point), tempfile.TemporaryDirectory() as temporary:
                capability = self._capability(temporary)
                transaction_id = "txn-" + point.replace("_", "-")
                report = execute_synthetic_night(
                    capability,
                    SyntheticScienceProvider(),
                    _spec(transaction_id),
                    fault_hook=FailureInjector(point),
                )
                snapshot = TransactionJournal.load(
                    capability.journal_root / f"{transaction_id}.json"
                ).snapshot
                self.assertFalse(report.success)
                self.assertEqual(snapshot.outcome, outcome)
                manifest = Path(snapshot.descriptor.target_path) / "manifest.json"
                self.assertEqual(manifest.exists(), snapshot.published)

    def test_preflight_capacity_failure_occurs_before_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            provider = SyntheticScienceProvider()
            fake = type(
                "StatVfs",
                (),
                {"f_bavail": 0, "f_frsize": 4096, "f_favail": 0, "f_files": 1},
            )()
            with mock.patch(
                "src.operations.writer.os.statvfs", return_value=fake
            ), mock.patch.object(
                provider, "query", side_effect=AssertionError("provider invoked")
            ):
                report = execute_synthetic_night(
                    capability, provider, _spec("txn-no-space")
                )
            self.assertFalse(report.success)
            snapshot = TransactionJournal.load(
                capability.journal_root / "txn-no-space.json"
            ).snapshot
            self.assertEqual(snapshot.state, ExecutionState.FAILED)
            self.assertFalse(snapshot.published)

    def test_reconciliation_failure_preserves_nightly_science(self):
        class FailingReconciler:
            def reconcile(self, *args, **kwargs):
                del args, kwargs
                raise OSError("synthetic derived failure")

        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            report = execute_synthetic_night(
                capability,
                SyntheticScienceProvider(),
                _spec("txn-reconcile-fail"),
                reconciler=FailingReconciler(),
            )
            snapshot = TransactionJournal.load(
                capability.journal_root / "txn-reconcile-fail.json"
            ).snapshot
            self.assertFalse(report.success)
            self.assertEqual(
                snapshot.outcome,
                JournalOutcome.PUBLISHED_RECONCILIATION_REQUIRED,
            )
            target = Path(snapshot.descriptor.target_path)
            self.assertTrue((target / "manifest.json").is_file())
            self.assertEqual(
                {item.name for item in target.iterdir()},
                {"loci.parquet", "alerts.parquet", "manifest.json"},
            )

    def test_reconciliation_replay_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            execute_synthetic_night(
                capability, SyntheticScienceProvider(), _spec("txn-reconcile")
            )
            snapshot = TransactionJournal.load(
                capability.journal_root / "txn-reconcile.json"
            ).snapshot
            target = Path(snapshot.descriptor.target_path)
            identities = {
                name: ArtifactIdentity.from_path(name, target / name)
                for name in ("loci.parquet", "alerts.parquet", "manifest.json")
            }
            repeated = SyntheticReconciler().reconcile(
                capability,
                date_utc="2026-06-27",
                published_artifacts=identities,
                transaction_id="txn-reconcile-replay",
            )
            self.assertTrue(repeated.idempotent_replay)

    def test_duplicate_valid_target_is_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            first = execute_synthetic_night(
                capability, SyntheticScienceProvider(), _spec("txn-first")
            )
            manifest = Path(first.artifacts[2].path)
            before = hashlib.sha256(manifest.read_bytes()).hexdigest()
            second = execute_synthetic_night(
                capability, SyntheticScienceProvider(), _spec("txn-second")
            )
            self.assertFalse(second.success)
            self.assertEqual(second.status, "existing_valid_target")
            self.assertEqual(
                hashlib.sha256(manifest.read_bytes()).hexdigest(), before
            )

    def test_duplicate_transaction_identity_never_replays(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            execute_synthetic_night(
                capability, SyntheticScienceProvider(), _spec("txn-same")
            )
            repeated = execute_synthetic_night(
                capability, SyntheticScienceProvider(), _spec("txn-same")
            )
            self.assertEqual(repeated.status, "duplicate_transaction")
            self.assertTrue(repeated.read_only)
            self.assertFalse(repeated.details["provider_invoked"])

    def test_late_target_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            marker = []

            def create_target(point, details):
                if point == "before_target_reservation" and not marker:
                    target = Path(details["target"])
                    target.mkdir()
                    (target / "foreign.marker").write_text("preserve")
                    marker.append(target)

            report = execute_synthetic_night(
                capability,
                SyntheticScienceProvider(),
                _spec("txn-late-target"),
                fault_hook=create_target,
            )
            self.assertFalse(report.success)
            self.assertEqual((marker[0] / "foreign.marker").read_text(), "preserve")
            self.assertFalse((marker[0] / "manifest.json").exists())
            assessment = self._assessment(capability, "txn-late-target")
            self.assertIn(
                RecoveryDisposition.REQUIRES_OPERATOR_DECISION,
                assessment.dispositions,
            )

    def test_precommit_tamper_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)

            def tamper(point, details):
                del details
                if point == "after_validation":
                    journals = list(capability.journal_root.glob("*.json"))
                    snapshot = TransactionJournal.load(journals[0]).snapshot
                    Path(snapshot.descriptor.stage_path, "loci.parquet").write_bytes(
                        b"tampered"
                    )

            report = execute_synthetic_night(
                capability,
                SyntheticScienceProvider(),
                _spec("txn-tamper"),
                fault_hook=tamper,
            )
            snapshot = TransactionJournal.load(
                capability.journal_root / "txn-tamper.json"
            ).snapshot
            self.assertFalse(report.success)
            self.assertFalse(snapshot.published)
            self.assertFalse(Path(snapshot.descriptor.target_path).exists())

    def test_corrupt_journal_and_reserved_target_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            report = execute_synthetic_night(
                capability,
                SyntheticScienceProvider(),
                _spec("txn-corrupt"),
                fault_hook=FailureInjector("after_lock"),
            )
            self.assertFalse(report.success)
            journal_path = capability.journal_root / "txn-corrupt.json"
            snapshot = TransactionJournal.load(journal_path).snapshot
            target = Path(snapshot.descriptor.target_path)
            target.mkdir(parents=True)
            journal_path.write_text("{torn", encoding="utf-8")
            assessment = inspect_recovery(
                journal_path,
                target_path=target,
                stage_path=Path(snapshot.descriptor.stage_path),
                lock_path=Path(snapshot.descriptor.lock_path),
            )
            self.assertIn(
                RecoveryDisposition.REQUIRES_OPERATOR_DECISION,
                assessment.dispositions,
            )
            self.assertIn(
                RecoveryDisposition.MUST_NOT_AUTO_DELETE,
                assessment.dispositions,
            )
            self.assertIn(
                RecoveryDisposition.REQUIRES_REVALIDATION,
                assessment.dispositions,
            )

    def test_inspector_distinguishes_active_pid_reuse_and_ambiguous_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            capability = self._capability(temporary)
            target_relative = Path("data/lsst_only/nightly/2026/06/27")
            target = capability.published_root / target_relative
            lock = WriterLock(
                capability,
                target_relative.as_posix(),
                "lock-inspection",
                transaction_id="lock-inspection",
                release_sha=RELEASE_SHA,
                process_start_identity="test-process-start",
            )
            journal_parent = capability.journal_root
            journal_parent.mkdir(parents=True)
            journal_path = journal_parent / "lock-inspection.json"
            stage = capability.staging_root / "lock-inspection"
            journal = TransactionJournal.create(
                journal_path,
                TransactionDescriptor(
                    run_id="lock-inspection",
                    operation="night.synthetic_ingest",
                    target_identity=target_relative.as_posix(),
                    target_path=str(target),
                    stage_path=str(stage),
                    lock_path=str(lock.path),
                    profile="synthetic:local-temporary",
                    plan_id="lock-plan",
                    release_sha=RELEASE_SHA,
                ),
            )
            journal.transition(ExecutionState.PRECHECKED)
            lock.acquire()
            journal.transition(ExecutionState.LOCKED)

            with mock.patch(
                "src.operations.locking._process_start_identity",
                return_value="test-process-start",
            ):
                active = inspect_recovery(
                    journal_path,
                    target_path=target,
                    stage_path=stage,
                    lock_path=lock.path,
                )
            self.assertEqual(active.lock_owner_status, "ACTIVE")
            self.assertIn(
                RecoveryDisposition.MUST_NOT_AUTO_DELETE, active.dispositions
            )

            metadata = json.loads(lock.metadata_path.read_text(encoding="utf-8"))
            metadata["process_start_identity"] = "reused-pid-different-start"
            lock.metadata_path.write_text(
                json.dumps(metadata, sort_keys=True), encoding="utf-8"
            )
            with mock.patch(
                "src.operations.locking._process_start_identity",
                return_value="test-process-start",
            ):
                reused = inspect_recovery(
                    journal_path,
                    target_path=target,
                    stage_path=stage,
                    lock_path=lock.path,
                )
            self.assertEqual(reused.lock_owner_status, "STALE")

            lock.metadata_path.write_text("{malformed", encoding="utf-8")
            ambiguous = inspect_recovery(
                journal_path,
                target_path=target,
                stage_path=stage,
                lock_path=lock.path,
            )
            self.assertEqual(ambiguous.lock_owner_status, "AMBIGUOUS")
            self.assertIn(
                RecoveryDisposition.REQUIRES_OPERATOR_DECISION,
                ambiguous.dispositions,
            )


class ProcessDeathAndConcurrencyTests(unittest.TestCase):
    def _context(self):
        return multiprocessing.get_context("spawn")

    def test_process_death_is_classified_from_durable_evidence(self):
        for point, expected, lock_status in (
            ("after_lock", RecoveryDisposition.REQUIRES_OPERATOR_DECISION, "STALE"),
            ("during_staging", RecoveryDisposition.REQUIRES_OPERATOR_DECISION, "STALE"),
            ("after_staging", RecoveryDisposition.REQUIRES_REVALIDATION, "STALE"),
            ("after_validation", RecoveryDisposition.REQUIRES_REVALIDATION, "STALE"),
            ("immediately_before_publication", RecoveryDisposition.REQUIRES_REVALIDATION, "STALE"),
            ("after_manifest_commit", RecoveryDisposition.MUST_NOT_AUTO_DELETE, "STALE"),
            ("after_reconciliation_before_completion", RecoveryDisposition.REQUIRES_RECONCILIATION, "ABSENT"),
        ):
            with self.subTest(point=point), tempfile.TemporaryDirectory() as temporary:
                run_id = "death-run"
                transaction_id = "death-" + point.replace("_", "-")
                root = Path(temporary) / run_id
                root.mkdir()
                process = self._context().Process(
                    target=_death_worker,
                    args=(str(root), run_id, transaction_id, point),
                )
                process.start()
                process.join(30)
                self.assertEqual(process.exitcode, 91)
                capability = SyntheticWriteCapability.for_local_run_root(
                    root, run_id
                )
                journal_path = capability.journal_root / f"{transaction_id}.json"
                snapshot = TransactionJournal.load(journal_path).snapshot
                assessment = inspect_recovery(
                    journal_path,
                    target_path=Path(snapshot.descriptor.target_path),
                    stage_path=Path(snapshot.descriptor.stage_path),
                    lock_path=Path(snapshot.descriptor.lock_path),
                )
                self.assertIn(expected, assessment.dispositions)
                self.assertEqual(assessment.lock_owner_status, lock_status)

    def test_eight_same_target_processes_produce_one_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_id = "contention-run"
            root = Path(temporary) / run_id
            root.mkdir()
            context = self._context()
            queue = context.Queue()
            acquired, release = context.Event(), context.Event()
            winner = context.Process(
                target=_concurrent_worker,
                args=(
                    str(root), run_id, "contender-0", 27, queue, acquired, release
                ),
            )
            winner.start()
            self.assertTrue(acquired.wait(20))
            capability = SyntheticWriteCapability.for_local_run_root(root, run_id)
            active_journal = TransactionJournal.load(
                capability.journal_root / "contender-0.json"
            ).snapshot
            lock_metadata = json.loads(
                Path(active_journal.descriptor.lock_path, "owner.json").read_text(
                    encoding="utf-8"
                )
            )
            expected_lock_status = (
                "AMBIGUOUS"
                if str(lock_metadata["process_start_identity"]).startswith(
                    "unavailable:"
                )
                else "ACTIVE"
            )
            active_assessment = inspect_recovery(
                capability.journal_root / "contender-0.json",
                target_path=Path(active_journal.descriptor.target_path),
                stage_path=Path(active_journal.descriptor.stage_path),
                lock_path=Path(active_journal.descriptor.lock_path),
            )
            self.assertEqual(
                active_assessment.lock_owner_status, expected_lock_status
            )
            self.assertIn(
                RecoveryDisposition.MUST_NOT_AUTO_DELETE,
                active_assessment.dispositions,
            )
            losers = [
                context.Process(
                    target=_concurrent_worker,
                    args=(str(root), run_id, f"contender-{index}", 27, queue),
                )
                for index in range(1, 8)
            ]
            for process in losers:
                process.start()
            for process in losers:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            release.set()
            winner.join(30)
            self.assertEqual(winner.exitcode, 0)
            results = [queue.get(timeout=5) for _ in range(8)]
            self.assertEqual(sum(item["success"] for item in results), 1)
            capability = SyntheticWriteCapability.for_local_run_root(root, run_id)
            target = capability.published_root / "data/lsst_only/nightly/2026/06/27"
            self.assertEqual(
                {item.name for item in target.iterdir()},
                {"loci.parquet", "alerts.parquet", "manifest.json"},
            )

    def test_four_different_target_processes_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_id = "parallel-run"
            root = Path(temporary) / run_id
            root.mkdir()
            context = self._context()
            queue = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_worker,
                    args=(str(root), run_id, f"parallel-{day}", day, queue),
                )
                for day in range(27, 31)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            results = [queue.get(timeout=5) for _ in processes]
            self.assertTrue(all(item["success"] for item in results), results)


class CanarySafetyTests(unittest.TestCase):
    def test_inventory_contract_and_descriptor_pinned_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "synthetic-run"
            target = root / "published/data/lsst_only/nightly/2098/01/01"
            target.mkdir(parents=True, mode=0o700)
            for directory in (root, *target.parents):
                if directory == Path(temporary).parent:
                    break
                if directory == Path(temporary):
                    continue
                if directory.exists() and (
                    directory == root or root in directory.parents
                ):
                    directory.chmod(0o700)
            marker = root / CANARY_IDENTITY_NAME
            marker.write_text("{}\n", encoding="utf-8")
            marker.chmod(0o600)
            for name, payload in (
                ("loci.parquet", b"loci"),
                ("alerts.parquet", b"alerts"),
                ("manifest.json", b"{}\n"),
            ):
                artifact = target / name
                artifact.write_bytes(payload)
                artifact.chmod(0o600)
            external = Path(temporary) / "external-evidence.json"
            external.write_text("preserve\n", encoding="utf-8")

            inventory = _inventory(root)
            contract = _inventory_contract(inventory, root.stat().st_dev)
            self.assertTrue(all(contract.values()), contract)
            _remove_inventoried_tree(root, inventory)
            self.assertEqual(list(root.iterdir()), [])
            self.assertEqual(external.read_text(encoding="utf-8"), "preserve\n")


if __name__ == "__main__":
    unittest.main()
