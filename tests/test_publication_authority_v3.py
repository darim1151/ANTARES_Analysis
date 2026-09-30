"""V3-G3.1 crash-safe publication authority.

Covers the explicit fault matrix (caught exceptions and real SIGKILL process
death), canonical-reader consistency in every split state, production-wide
writer exclusion, full Sentinel V2 binding, journal/path integrity, publication
chronology, retry classification, the production/range capability contracts,
the prior-free acquisition invariant, and segment-cache confinement.
"""

import errno
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import v3_fixtures as F
from src import authority, cli_diagnostics, history
from src.cli_profiles import StorageProfile
from src.operations import publication as P
from src.operations.commissioning import CommissioningError
from src.operations.backfill import (
    BackfillController,
    BackfillRefused,
    BackfillSettings,
    PRIOR_FREE_ACQUISITION_ATTESTATIONS,
    RangePublicationAuthorization,
    acquisition_identity,
    inspect_backfill,
    require_prior_free_acquisition,
)
from src.operations.cache import SegmentCache, SegmentCacheRefused
from src.operations.journal import TransactionDescriptor, TransactionJournal
from src.operations.locking import LockUnavailable, WriterLock
from src.operations.writer import (
    SHARED_RECONCILIATION_LOCK_IDENTITY,
    InjectedWriterFailure,
    ProductionAuthorizationUnavailable,
)


NIGHT = "2026-07-01"
BASELINE_NIGHTS = ("2026-06-29", "2026-06-30")
HERE = Path(__file__).resolve().parent
CHILD = HERE / "v3_crash_child.py"


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_child(spec: dict, directory: Path) -> subprocess.CompletedProcess:
    spec_path = Path(directory) / f"child-{time.monotonic_ns()}.json"
    spec_path.write_text(json.dumps(spec))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE.parent), str(HERE)])
    return subprocess.run(
        [sys.executable, str(CHILD), str(spec_path)],
        env=env, capture_output=True, text=True, timeout=600,
    )


class _FailAt:
    def __init__(self, point, action=None):
        self.point = point
        self.action = action
        self.fired = False

    def __call__(self, point, details):
        if point == self.point and not self.fired:
            self.fired = True
            if self.action is not None:
                self.action()
                return
            raise InjectedWriterFailure(point)


class AuthorityFixture(unittest.TestCase):
    """Production fixture plus one backfill-layout candidate for NIGHT."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.capability = F.make_capability(self.tmp)
        self.root = self.capability.published_root
        F.seed_production(self.capability, BASELINE_NIGHTS)
        made = F.make_backfill_candidate(self.tmp, NIGHT, june27_style_defect=True)
        self.night_root = self.capability.root / "backfill" / "nights" / f"night-{NIGHT}"
        self.night_root.parent.mkdir(parents=True)
        shutil.copytree(made.candidate_dir.parent, self.night_root)
        self.candidate = P.load_backfill_candidate(self.night_root)
        self.publisher = F.publisher_for(self.capability)
        self.authorization = F.authorize(self.publisher, self.candidate)
        self.authorization_path = self.tmp / "authorization.json"
        P.write_authorization(self.authorization_path, self.authorization)
        self.baseline_fingerprint = self.authorization.baseline_production_fingerprint
        self.baseline_cumulative = dict(self.authorization.production["cumulative_sha256"])
        self.expected_cumulative = dict(self.authorization.expected_cumulative_sha256)
        self.candidate_digest = _tree_digest(self.candidate.candidate_dir)

    def tearDown(self):
        for path in self.tmp.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
        self._tmp.cleanup()

    # -- observation helpers ------------------------------------------------

    def cumulative_hashes(self):
        paths = history.cumulative_paths(self.root)
        return {key: _sha(paths[key]) for key in ("loci_index", "nightly_summary")}

    def gate_present(self):
        return history.publication_gate_path(self.root).exists()

    def publication_journals(self):
        return [
            journal for journal in P.load_transaction_journals(self.capability.journal_root)
            if journal.snapshot.descriptor.operation == P.PUBLICATION_OPERATION
        ]

    def classify(self):
        return P.classify_night_authority(self.root, self.capability.journal_root, NIGHT)

    def controller_stage(self):
        document = inspect_backfill(self.capability.root, self.root, NIGHT, NIGHT)
        return document["nights"][0]["stage"]

    def fingerprint(self):
        return self.publisher.sentinel(NIGHT)["durable_fingerprint_sha256"]

    def assert_readers_refuse(self):
        readers = {
            "nightly discovery": lambda: P.authoritative_nights(self.root),
            "cumulative loci": lambda: history.load_cumulative_loci_index(self.root),
            "nightly summary": lambda: history.load_nightly_summary(self.root),
            "manifest": lambda: history.read_manifest(self.root, "2026-06-30"),
            "alerts": lambda: history.load_cumulative_alerts(self.root),
            "canonical rebuild": lambda: history.update_cumulative_indexes(
                self.root, output_dir=self.tmp / "refused-rebuild"
            ),
        }
        for label, reader in readers.items():
            with self.subTest(reader=label):
                with self.assertRaises(history.PublicationInProgress):
                    reader()
        status = cli_diagnostics.collect_data_status(
            StorageProfile("fixture", "fixture", self.root, self.tmp / "absent-cache", "private")
        )
        self.assertTrue(status["publication_in_progress"])
        self.assertFalse(status["ok"])

    def assert_readers_consistent(self, published: bool):
        nights = P.authoritative_nights(self.root)
        index = history.load_cumulative_loci_index(self.root)
        summary = history.load_nightly_summary(self.root)
        self.assertEqual(nights, BASELINE_NIGHTS + ((NIGHT,) if published else ()))
        self.assertEqual(
            int(index["night_date_utc"].eq(NIGHT).sum()), self.candidate.loci if published else 0
        )
        self.assertEqual(int(summary["date_utc"].eq(NIGHT).sum()), 1 if published else 0)

    def assert_complete_and_idempotent(self, outcome):
        self.assertTrue(outcome.success, outcome.record)
        self.assertFalse(self.gate_present())
        self.assert_readers_consistent(published=True)
        self.assertEqual(self.cumulative_hashes(), self.expected_cumulative)
        rebuilt_index, rebuilt_summary = history.update_cumulative_indexes(
            self.root, output_dir=self.tmp / f"rebuild-{time.monotonic_ns()}"
        )
        pd.testing.assert_frame_equal(
            history.load_cumulative_loci_index(self.root), rebuilt_index.reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            history.load_nightly_summary(self.root), rebuilt_summary.reset_index(drop=True),
            check_dtype=False,
        )
        self.assertEqual(list(self.capability.staging_root.iterdir()), [])
        self.assertEqual(
            [p.name for p in self.capability.lock_root.iterdir()], [P.PUBLICATION_LOCK_NAME]
        )
        self.assertEqual(_tree_digest(self.candidate.candidate_dir), self.candidate_digest)
        state = self.classify()
        self.assertEqual((state["state"], state["finalized"]), ("COMPLETE", True))
        self.assertEqual(self.controller_stage(), "PUBLISHED")
        after = self.fingerprint()
        self.assertEqual(outcome.record["resulting_production_fingerprint"], after)
        replay = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(replay.status, "already_published")
        self.assertEqual(self.fingerprint(), after)
        records = list((self.capability.evidence_root / "publications" / NIGHT).glob("*.json"))
        self.assertEqual(len(records), 1)
        return outcome


# ---------------------------------------------------------------------------
# Explicit fault matrix
# ---------------------------------------------------------------------------

# boundary, authority state, gate present, journal state after SIGKILL,
# controller stage, nightly manifest physically linked, cumulative state,
# transaction resumed (True) or re-attempted after NOT_COMMITTED (False)
MATRIX = (
    ("before_transaction_reservation", "NOT_COMMITTED", False, None, "WAITING_FOR_PUBLICATION", False, "baseline", False),
    ("after_transaction_reservation", "NOT_COMMITTED", False, "fetching", "WAITING_FOR_PUBLICATION", False, "baseline", False),
    ("after_staging", "NOT_COMMITTED", False, "staged", "WAITING_FOR_PUBLICATION", False, "baseline", False),
    ("before_authority_gate", "NOT_COMMITTED", False, "validated", "WAITING_FOR_PUBLICATION", False, "baseline", False),
    ("after_gate_link", "RECONCILIATION_REQUIRED", True, "validated", "RECONCILIATION_REQUIRED", False, "baseline", True),
    ("after_authority_gate", "RECONCILIATION_REQUIRED", True, "published", "RECONCILIATION_REQUIRED", False, "baseline", True),
    ("after_target_reservation", "RECONCILIATION_REQUIRED", True, "published", "RECONCILIATION_REQUIRED", False, "baseline", True),
    ("after_data_links", "RECONCILIATION_REQUIRED", True, "published", "RECONCILIATION_REQUIRED", False, "baseline", True),
    ("before_manifest_commit", "RECONCILIATION_REQUIRED", True, "published", "RECONCILIATION_REQUIRED", False, "baseline", True),
    ("after_manifest_commit", "RECONCILIATION_REQUIRED", True, "published", "RECONCILIATION_REQUIRED", True, "baseline", True),
    ("before_cumulative_install:loci_index", "RECONCILIATION_REQUIRED", True, "reconciling", "RECONCILIATION_REQUIRED", True, "baseline", True),
    ("before_cumulative_install:nightly_summary", "RECONCILIATION_REQUIRED", True, "reconciling", "RECONCILIATION_REQUIRED", True, "index_only", True),
    ("before_authority_commit", "RECONCILIATION_REQUIRED", True, "reconciling", "RECONCILIATION_REQUIRED", True, "expected", True),
    ("after_authority_commit", "COMPLETE", False, "reconciling", "RECONCILIATION_REQUIRED", True, "expected", True),
    ("before_journal_complete", "COMPLETE", False, "reconciling", "RECONCILIATION_REQUIRED", True, "expected", True),
    ("after_journal_complete", "COMPLETE", False, "complete", "PUBLISHED", True, "expected", True),
)


class _MatrixCase(AuthorityFixture):
    hard_kill = False

    def inject(self, boundary):
        if self.hard_kill:
            completed = run_child(
                {
                    "mode": "publish",
                    "run_root": str(self.capability.root),
                    "run_id": self.capability.run_id,
                    "night_root": str(self.night_root),
                    "authorization": str(self.authorization_path),
                    "boundary": boundary,
                },
                self.tmp,
            )
            self.assertEqual(completed.returncode, -signal.SIGKILL, completed.stderr[-2000:])
            return None
        failing = F.publisher_for(self.capability, fault_hook=_FailAt(boundary))
        return failing.publish(self.candidate, self.authorization)

    def check_boundary(self, row):
        boundary, state, gate, journal_state, stage, linked, cumulative, resumed = row
        outcome = self.inject(boundary)
        observed = self.classify()
        # 1. authority classification and production-visible state
        self.assertEqual(observed["state"], state, observed)
        self.assertEqual(self.gate_present(), gate)
        target = self.root / "data/lsst_only/nightly/2026/07/01"
        self.assertEqual((target / "manifest.json").is_file(), linked)
        expected_cumulative = {
            "baseline": self.baseline_cumulative,
            "index_only": {
                "loci_index": self.expected_cumulative["loci_index"],
                "nightly_summary": self.baseline_cumulative["nightly_summary"],
            },
            "expected": self.expected_cumulative,
        }[cumulative]
        self.assertEqual(self.cumulative_hashes(), expected_cumulative)
        if gate:
            self.assert_readers_refuse()
        else:
            self.assert_readers_consistent(published=state == "COMPLETE")
        if state == "NOT_COMMITTED":
            self.assertEqual(self.fingerprint(), self.baseline_fingerprint)
        # 2. journal state and staged-material preservation
        journals = self.publication_journals()
        if self.hard_kill:
            recorded = journals[0].snapshot.state.value if journals else None
            self.assertEqual(recorded, journal_state)
        elif state == "NOT_COMMITTED" and journals:
            self.assertEqual(journals[0].snapshot.state.value, "failed")
            self.assertEqual(journals[0].snapshot.outcome.value, "UNPUBLISHED_FAILURE")
        elif journals:
            self.assertIn(
                journals[0].snapshot.outcome.value,
                {"PUBLISHED_RECONCILIATION_REQUIRED", "COMPLETE"},
            )
        if gate and cumulative != "expected":
            staged = self.capability.staging_root / f"{journals[0].snapshot.descriptor.run_id}-cumulative"
            for key in ("loci_index", "nightly_summary"):
                installed = expected_cumulative[key] == self.expected_cumulative[key]
                self.assertEqual((staged / f"{key}.parquet").is_file(), not installed, key)
        # 3. truthful caller-visible outcome (in-process only)
        if outcome is not None:
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.record["authority_state"], state)
            self.assertEqual(outcome.record["production_mutated"], state != "NOT_COMMITTED")
            self.assertTrue(outcome.record["retryable"])
        # 4. controller classification
        self.assertEqual(self.controller_stage(), stage)
        # 5. deterministic resume with the same authorization
        before_ids = {j.snapshot.descriptor.run_id for j in journals}
        resumed_outcome = self.publisher.publish(self.candidate, self.authorization)
        self.assert_complete_and_idempotent(resumed_outcome)
        transaction = resumed_outcome.record["transaction_id"]
        if resumed:
            self.assertIn(transaction, before_ids)
        else:
            self.assertNotIn(transaction, before_ids)
            for journal in self.publication_journals():
                if journal.snapshot.descriptor.run_id in before_ids:
                    self.assertEqual(journal.snapshot.outcome.value, "UNPUBLISHED_FAILURE")
        chronology = resumed_outcome.record["chronology"]
        if boundary == "after_authority_commit" and self.hard_kill:
            self.assertIsNone(chronology["authority_committed_at_utc"])
            self.assertFalse(chronology["authority_commit_observed"])
            self.assertIsNotNone(chronology["authority_commit_not_before_utc"])
            self.assertIsNotNone(chronology["authority_commit_not_after_utc"])
        elif state in {"RECONCILIATION_REQUIRED", "NOT_COMMITTED"}:
            self.assertTrue(chronology["authority_commit_observed"])


def _add_matrix_tests(cls, hard_kill):
    for row in MATRIX:
        name = "test_" + row[0].replace(":", "_")

        def test(self, row=row):
            self.check_boundary(row)

        setattr(cls, name, test)
    cls.hard_kill = hard_kill


class InProcessFaultMatrixTests(_MatrixCase):
    pass


class HardProcessDeathMatrixTests(_MatrixCase):
    pass


_add_matrix_tests(InProcessFaultMatrixTests, hard_kill=False)
_add_matrix_tests(HardProcessDeathMatrixTests, hard_kill=True)
del _MatrixCase


# ---------------------------------------------------------------------------
# Controller-level hard process death: truthful state, repeated --resume
# ---------------------------------------------------------------------------


class ControllerHardDeathTests(unittest.TestCase):
    START, END = "2026-07-01", "2026-07-03"
    NIGHTS = ("2026-07-01", "2026-07-02", "2026-07-03")

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.capability = F.make_capability(self.tmp)
        F.seed_production(self.capability, BASELINE_NIGHTS)
        self.publisher = P.NightPublisher(
            self.capability, publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=self.capability.root / "absent-cache",
            mountinfo_lines=F.mountinfo_for(self.capability.published_root),
        )

    def tearDown(self):
        self._tmp.cleanup()

    def controller(self, adapter, authority):
        return BackfillController(
            self.capability, adapter, release_sha=F.CANDIDATE_RELEASE,
            read_capability_factory=F.mock_read_capability,
            settings=BackfillSettings(segment_size=4), publisher=self.publisher,
            range_authorization=authority,
            prior_free_attestations=F.fixture_prior_free_attestations(),
        )

    def authority(self, adapter):
        binding = self.controller(adapter, None).range_binding(self.START, self.END)
        production = P.production_binding_from_sentinel(self.publisher.sentinel(self.START))
        return RangePublicationAuthorization(
            start_date_utc=self.START, end_date_utc=self.END,
            initial_predecessor_date_utc="2026-06-30",
            initial_sentinel={
                key: production[key]
                for key in ("canonical_root", "mount_binding", "durable_fingerprint_sha256",
                            "manifest_count", "cumulative_sha256")
            },
            candidate_release_sha=F.CANDIDATE_RELEASE, publisher_release_sha=F.PUBLISHER_RELEASE,
            authorized_by="control-fixture", authorized_at_utc="2026-09-28T00:00:00+00:00",
            expires_at_utc=F.EXPIRES_AT, **binding,
        )

    def stages(self):
        document = inspect_backfill(
            self.capability.root, self.capability.published_root, self.START, self.END
        )
        return {night["date_utc"]: night["stage"] for night in document["nights"]}

    def crash_then_resume(self, boundary, occurrence):
        authority = self.authority(F.SyntheticBackfillAdapter(6))
        path = self.tmp / "range-authority.json"
        path.write_text(json.dumps(authority.as_dict()))
        completed = run_child(
            {
                "mode": "backfill", "run_root": str(self.capability.root),
                "run_id": self.capability.run_id, "range_authorization": str(path),
                "boundary": boundary, "occurrence": occurrence, "loci": 6,
                "start": self.START, "end": self.END,
            },
            self.tmp,
        )
        self.assertEqual(completed.returncode, -signal.SIGKILL, completed.stderr[-3000:])
        return authority

    def assert_resumes_without_reacquisition(self, authority, crashed):
        crashed_stage = self.stages()
        self.assertEqual(crashed_stage["2026-07-01"], "PUBLISHED")
        self.assertEqual(crashed_stage[crashed], "RECONCILIATION_REQUIRED")
        self.assertNotEqual(crashed_stage["2026-07-03"], "PUBLISHED")
        adapter = F.SyntheticBackfillAdapter(6)  # fresh counters prove no reacquisition
        for attempt in range(2):
            document = self.controller(adapter, authority).run(self.START, self.END, resume=True)
            stages = {night["date_utc"]: night["stage"] for night in document["nights"]}
            self.assertEqual(set(stages.values()), {"PUBLISHED"}, (attempt, stages))
            if attempt == 0:
                fingerprint = self.publisher.sentinel(self.END)["durable_fingerprint_sha256"]
        self.assertEqual(adapter.queries[crashed], 0)
        self.assertEqual(adapter.fetched_segments[crashed], 0)
        self.assertEqual(adapter.constructions[crashed], 0)  # candidate never rebuilt
        self.assertEqual(adapter.queries["2026-07-01"], 0)
        self.assertEqual(
            self.publisher.sentinel(self.END)["durable_fingerprint_sha256"], fingerprint
        )
        self.assertEqual(
            P.authoritative_nights(self.capability.published_root),
            BASELINE_NIGHTS + self.NIGHTS,
        )
        index = history.load_cumulative_loci_index(self.capability.published_root)
        summary = history.load_nightly_summary(self.capability.published_root)
        for day in self.NIGHTS:
            self.assertEqual(int(index["night_date_utc"].eq(day).sum()), 6)
            self.assertEqual(int(summary["date_utc"].eq(day).sum()), 1)
            records = list((self.capability.evidence_root / "publications" / day).glob("*.json"))
            self.assertEqual(len(records), 1, day)

    def test_hard_death_after_second_manifest_link(self):
        authority = self.crash_then_resume("after_manifest_commit", 2)
        self.assert_resumes_without_reacquisition(authority, "2026-07-02")

    def test_hard_death_before_second_summary_install(self):
        authority = self.crash_then_resume("before_cumulative_install:nightly_summary", 2)
        self.assert_resumes_without_reacquisition(authority, "2026-07-02")

    def test_hard_death_after_commit_before_terminal_evidence(self):
        authority = self.crash_then_resume("after_journal_complete", 2)
        stages = self.stages()
        self.assertEqual(stages["2026-07-02"], "PUBLISHED")  # COMPLETE authority
        document = inspect_backfill(
            self.capability.root, self.capability.published_root, self.START, self.END
        )
        night = next(n for n in document["nights"] if n["date_utc"] == "2026-07-02")
        self.assertFalse(night["evidence"]["terminal_record_present"])
        adapter = F.SyntheticBackfillAdapter(6)
        document = self.controller(adapter, authority).run(self.START, self.END, resume=True)
        self.assertEqual({n["stage"] for n in document["nights"]}, {"PUBLISHED"})
        for day in self.NIGHTS:
            records = list((self.capability.evidence_root / "publications" / day).glob("*.json"))
            self.assertEqual(len(records), 1, day)
        self.assertEqual(adapter.queries["2026-07-02"], 0)


# ---------------------------------------------------------------------------
# Readers never consume a mixed generation
# ---------------------------------------------------------------------------


class ReaderConsistencyTests(AuthorityFixture):
    def test_two_shared_readers_coexist(self):
        entered = threading.Event()
        release = threading.Event()

        def second_reader():
            with history.authority_read_lock(self.root):
                entered.set()
                release.wait(10)

        with history.authority_read_lock(self.root):
            worker = threading.Thread(target=second_reader)
            worker.start()
            self.assertTrue(entered.wait(2))
            release.set()
        worker.join(10)
        self.assertFalse(worker.is_alive())

    def test_shared_reader_excludes_publication_for_its_full_operation(self):
        result = {}
        with history.authoritative_read(self.root):
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "outcome", self.publisher.publish(self.candidate, self.authorization)
                )
            )
            worker.start()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            outcome = result["outcome"]
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.record["refusal_code"], "publication_lock_busy")
            self.assertFalse(self.gate_present())
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)

    @staticmethod
    def _skypulse_module():
        spec = importlib.util.spec_from_file_location(
            "export_skypulse_public_data_g32",
            HERE.parent / "scripts/export_skypulse_public_data.py",
        )
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        return export

    def test_skypulse_reader_first_blocks_writer_and_prevents_split_generation(self):
        export = self._skypulse_module()
        entered = threading.Event()
        release = threading.Event()
        result = {}

        def logical_export(_args):
            first_index = history.load_cumulative_loci_index(self.root)
            entered.set()
            release.wait(10)
            second_summary = history.load_nightly_summary(self.root)
            result["target_index"] = int(first_index["night_date_utc"].eq(NIGHT).sum())
            result["target_summary"] = int(second_summary["date_utc"].eq(NIGHT).sum())
            return {"ok": {}}

        with mock.patch.object(export, "_build_rsp_payloads_locked", side_effect=logical_export):
            worker = threading.Thread(
                target=lambda: result.setdefault(
                    "payload", export.build_rsp_payloads(SimpleNamespace(data_root=self.root))
                )
            )
            worker.start()
            self.assertTrue(entered.wait(10))
            refused = self.publisher.publish(self.candidate, self.authorization)
            self.assertEqual(refused.record["refusal_code"], "publication_lock_busy")
            self.assertFalse(self.gate_present())
            release.set()
            worker.join(10)
            self.assertFalse(worker.is_alive())
        self.assertEqual((result["target_index"], result["target_summary"]), (0, 0))
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)

    def test_publication_first_blocks_skypulse_until_exclusive_release(self):
        export = self._skypulse_module()
        entered = threading.Event()
        completed = threading.Event()
        errors = []

        def logical_export(_args):
            entered.set()
            return {"ok": {}}

        def run_export():
            try:
                export.build_rsp_payloads(SimpleNamespace(data_root=self.root))
            except BaseException as exc:  # captured for assertion in the parent
                errors.append(exc)
            finally:
                completed.set()

        with mock.patch.object(export, "_build_rsp_payloads_locked", side_effect=logical_export):
            with P.PublicationAuthorityLock(self.capability):
                worker = threading.Thread(target=run_export)
                worker.start()
                time.sleep(0.15)
                self.assertFalse(entered.is_set())
                self.assertFalse(completed.is_set())
            self.assertTrue(completed.wait(10))
            worker.join(10)
        self.assertEqual(errors, [])
        self.assertTrue(entered.is_set())

    def test_audit_and_diagnostics_wait_for_the_same_generation_lock(self):
        spec = importlib.util.spec_from_file_location(
            "audit_antares_data_root_g32",
            HERE.parent / "scripts/audit_antares_data_root.py",
        )
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        started = threading.Event()
        finished = threading.Event()

        def locked_audit(*_args, **_kwargs):
            started.set()
            return {"ok": True}

        with mock.patch.object(audit, "_audit_data_root_locked", side_effect=locked_audit):
            with P.PublicationAuthorityLock(self.capability):
                worker = threading.Thread(
                    target=lambda: (audit.audit_data_root(self.root, self.tmp / "audit-out"), finished.set())
                )
                worker.start()
                time.sleep(0.15)
                self.assertFalse(started.is_set())
                self.assertFalse(finished.is_set())
            worker.join(10)
        self.assertTrue(started.is_set())
        self.assertTrue(finished.is_set())

        profile = StorageProfile(
            "fixture", "fixture", self.root, self.tmp / "absent-cache", "private"
        )
        started.clear()
        finished.clear()
        with mock.patch.object(
            cli_diagnostics,
            "_collect_data_status_locked",
            side_effect=lambda *_: (started.set() or {"ok": True}),
        ):
            with P.PublicationAuthorityLock(self.capability):
                worker = threading.Thread(
                    target=lambda: (cli_diagnostics.collect_data_status(profile), finished.set())
                )
                worker.start()
                time.sleep(0.15)
                self.assertFalse(started.is_set())
                self.assertFalse(finished.is_set())
            worker.join(10)
        self.assertTrue(started.is_set())
        self.assertTrue(finished.is_set())

    def test_every_canonical_reader_refuses_while_gated(self):
        outcome = F.publisher_for(
            self.capability, fault_hook=_FailAt("before_cumulative_install:nightly_summary")
        ).publish(self.candidate, self.authorization)
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        self.assert_readers_refuse()
        from src import feature_analysis

        with self.assertRaises(history.PublicationInProgress):
            feature_analysis._source_inventory(self.root)
        spec = importlib.util.spec_from_file_location(
            "export_skypulse_public_data", HERE.parent / "scripts/export_skypulse_public_data.py"
        )
        export = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(export)
        with self.assertRaises(export.ExportError):
            export.discover_nights(self.root)
        from src import cli
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([
                "backfill", "status", NIGHT, NIGHT, "--run-root", str(self.capability.root),
                "--json",
            ])
        self.assertEqual(code, 1)
        self.assertEqual(
            json.loads(buffer.getvalue())["nights"][0]["stage"], "RECONCILIATION_REQUIRED"
        )

    def test_gate_name_mirrors_are_exact(self):
        self.assertEqual(cli_diagnostics.PUBLICATION_GATE_NAME, history.PUBLICATION_GATE_NAME)
        for script in ("export_skypulse_public_data.py", "audit_antares_data_root.py"):
            text = (HERE.parent / "scripts" / script).read_text()
            self.assertIn(f'PUBLICATION_GATE_NAME = "{history.PUBLICATION_GATE_NAME}"', text)
        self.assertIn("transaction", history.PUBLICATION_GATE_NAME.lower())

    def test_sentinel_flags_the_gate_as_a_transaction_artifact(self):
        F.publisher_for(
            self.capability, fault_hook=_FailAt("after_authority_gate")
        ).publish(self.candidate, self.authorization)
        with self.assertRaises(CommissioningError):
            self.publisher.sentinel(NIGHT)


# ---------------------------------------------------------------------------
# Production-wide writer exclusion
# ---------------------------------------------------------------------------


class GlobalExclusionTests(AuthorityFixture):
    def test_lock_symlink_and_unsupported_protocol_fail_closed(self):
        lock_path = self.capability.lock_root / P.PUBLICATION_LOCK_NAME
        target = self.tmp / "foreign-lock"
        target.write_text("not authority\n")
        lock_path.unlink()
        lock_path.symlink_to(target)
        with self.assertRaises(history.PublicationInProgress):
            history.load_nightly_summary(self.root)
        refused = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(refused.record["refusal_code"], "path_contradiction")

        lock_path.unlink()
        lock_path.write_text("")
        unsupported = OSError(errno.EOPNOTSUPP, "unsupported")
        with mock.patch.object(authority.fcntl, "flock", side_effect=unsupported):
            with self.assertRaises(history.PublicationInProgress):
                history.load_nightly_summary(self.root)
            refused = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(refused.record["refusal_code"], "path_contradiction")

    def test_shared_to_exclusive_upgrade_fails_without_deadlock(self):
        with history.authority_read_lock(self.root):
            with self.assertRaises(authority.AuthorityLockOrderError):
                with authority.exclusive_authority_lock(self.root):
                    self.fail("exclusive upgrade must not enter")

    def test_second_writer_in_process_refuses_transiently(self):
        with P.PublicationAuthorityLock(self.capability):
            outcome = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(outcome.record["refusal_code"], "publication_lock_busy")
        self.assertEqual(outcome.record["failure_category"], "transient_lock_contention")
        self.assertTrue(outcome.record["retryable"])
        self.assertEqual(self.fingerprint(), self.baseline_fingerprint)
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)

    def test_lock_held_by_another_process_and_released_by_its_death(self):
        ready = self.tmp / "ready"
        spec = self.tmp / "hold.json"
        spec.write_text(json.dumps({
            "mode": "hold-lock", "run_root": str(self.capability.root),
            "run_id": self.capability.run_id, "ready": str(ready), "boundary": "none",
        }))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE.parent), str(HERE)])
        holder = subprocess.Popen([sys.executable, str(CHILD), str(spec)], env=env)
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
            outcome = self.publisher.publish(self.candidate, self.authorization)
            self.assertEqual(outcome.record["refusal_code"], "publication_lock_busy")
        finally:
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=60)
        self.assertEqual(self.fingerprint(), self.baseline_fingerprint)
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)

    def test_shared_lock_held_by_process_death_releases_for_writer(self):
        ready = self.tmp / "shared-ready"
        spec = self.tmp / "hold-shared.json"
        spec.write_text(json.dumps({
            "mode": "hold-shared-lock", "run_root": str(self.capability.root),
            "run_id": self.capability.run_id, "ready": str(ready), "boundary": "none",
        }))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(HERE.parent), str(HERE)])
        holder = subprocess.Popen([sys.executable, str(CHILD), str(spec)], env=env)
        try:
            deadline = time.monotonic() + 120
            while not ready.exists():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
            refused = self.publisher.publish(self.candidate, self.authorization)
            self.assertEqual(refused.record["refusal_code"], "publication_lock_busy")
        finally:
            holder.send_signal(signal.SIGKILL)
            holder.wait(timeout=60)
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)

    def test_unresolved_transition_excludes_other_authorizations(self):
        failing = F.publisher_for(self.capability, fault_hook=_FailAt("after_manifest_commit"))
        failing.publish(self.candidate, self.authorization)
        other = P.PublicationAuthorization.from_dict(
            {**self.authorization.as_dict(), "nonce": "c" * 32}
        )
        refused = self.publisher.publish(self.candidate, other)
        self.assertEqual(refused.record["refusal_code"], "authority_transition_pending")
        self.assertTrue(self.gate_present())
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)
        self.assertEqual(
            self.publisher.publish(self.candidate, other).record["refusal_code"],
            "already_authoritative",
        )

    def test_unfinished_foreign_transaction_class_blocks(self):
        from src.operations.state import ExecutionState

        journal_root = self.capability.journal_root
        journal_root.mkdir(parents=True, exist_ok=True)
        journal = TransactionJournal.create(
            journal_root / "synthetic-writer.json",
            TransactionDescriptor(
                run_id="synthetic-writer", operation="night.synthetic_ingest",
                target_identity="data/lsst_only/nightly/2026/07/01",
                target_path=str(self.root / "data/lsst_only/nightly/2026/07/01"),
                stage_path=str(self.capability.staging_root / "x"),
                lock_path=str(self.capability.lock_root / "x"), profile="synthetic",
            ),
        )
        journal.transition(ExecutionState.PRECHECKED)
        outcome = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(outcome.record["refusal_code"], "transaction_pending")
        self.assertEqual(self.fingerprint(), self.baseline_fingerprint)


# ---------------------------------------------------------------------------
# Full Sentinel V2 binding at authorization and at the mutation boundary
# ---------------------------------------------------------------------------


class SentinelBindingTests(AuthorityFixture):
    def assert_not_committed(self, outcome, code):
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.record["refusal_code"], code, outcome.record)
        self.assertEqual(outcome.record["authority_state"], "NOT_COMMITTED")
        self.assertFalse(outcome.record["production_mutated"])
        self.assertFalse(self.gate_present())

    def test_authorization_binds_the_full_production_context(self):
        production = self.authorization.production
        self.assertEqual(production["canonical_root"], str(self.root))
        self.assertEqual(set(production["mount_binding"]), {"mount_point", "filesystem_type", "source"})
        self.assertEqual(production["predicates"]["transaction_artifacts"], [])
        self.assertTrue(production["predicates"]["target_absent"])
        self.assertTrue(production["predicates"]["cache_absent"])
        self.assertNotIn("device", json.dumps(self.authorization.as_dict()))
        for field in ("candidate_dir", "candidate_record_sha256", "candidate_provenance_sha256",
                      "candidate_release_sha", "publisher_release_sha", "artifact_sha256",
                      "expected_cumulative_sha256", "nonce", "expires_at_utc"):
            self.assertIn(field, self.authorization.as_dict())

    def test_mount_binding_drift(self):
        drifted = [line.replace("fixture:/production", "other:/production")
                   for line in F.mountinfo_for(self.root)]
        publisher = P.NightPublisher(
            self.capability, publisher_release_sha=F.PUBLISHER_RELEASE,
            cache_root=self.capability.root / "absent-cache", mountinfo_lines=drifted,
            clock=F.fixed_clock,
        )
        self.assert_not_committed(
            publisher.publish(self.candidate, self.authorization), "mount_binding_drift"
        )

    def test_drift_between_preflight_and_mutation_boundary(self):
        def drift():
            note = self.root / "analysis" / "drift.txt"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text("drift")

        outcome = F.publisher_for(
            self.capability, fault_hook=_FailAt("before_final_qualification", drift)
        ).publish(self.candidate, self.authorization)
        self.assert_not_committed(outcome, "sentinel_drift")
        self.assertEqual(list(self.capability.staging_root.iterdir()), [])

    def test_runtime_linkage_is_rechecked_live(self):
        original = self.publisher.sentinel

        def foreign_device(date_utc):
            sentinel = json.loads(json.dumps(original(date_utc)))
            sentinel["runtime_observation"]["durable_file_devices"][0]["device"] += 1
            return sentinel

        self.publisher.sentinel = foreign_device
        self.assert_not_committed(
            self.publisher.publish(self.candidate, self.authorization), "runtime_linkage_mismatch"
        )

    def test_cache_predicate_expiry_nonce_and_plan(self):
        (self.capability.root / "absent-cache").mkdir()
        self.assert_not_committed(
            self.publisher.publish(self.candidate, self.authorization), "cache_present"
        )
        (self.capability.root / "absent-cache").rmdir()
        expired = F.authorize(
            self.publisher, self.candidate,
            authorized_at="2026-09-21T00:00:00+00:00", expires_at="2026-09-22T00:00:00+00:00",
        )
        self.assert_not_committed(
            self.publisher.publish(self.candidate, expired), "authorization_expired"
        )
        wrong_plan = F.authorize(
            self.publisher, self.candidate,
            expected={"loci_index": "0" * 64, "nightly_summary": "1" * 64},
        )
        outcome = self.publisher.publish(self.candidate, wrong_plan)
        self.assertEqual(outcome.record["refusal_code"], "cumulative_plan_mismatch")
        self.assertFalse(self.gate_present())
        self.assertTrue(self.publisher.publish(self.candidate, self.authorization).success)
        later = F.make_backfill_candidate(self.tmp / "later", "2026-07-02")
        reused = F.authorize(self.publisher, later, nonce=self.authorization.nonce)
        self.assertEqual(
            self.publisher.publish(later, reused).record["refusal_code"],
            "authorization_nonce_reused",
        )

    def test_authorization_bound_to_another_production_root(self):
        production = dict(self.authorization.production)
        production["canonical_root"] = str(self.tmp / "elsewhere")
        production["predicates"] = {
            **production["predicates"],
            "target_path": str(history.nightly_paths(self.tmp / "elsewhere", NIGHT)["dir"]),
        }
        wrong = F.authorize(self.publisher, self.candidate, production=production)
        self.assertEqual(
            self.publisher.publish(self.candidate, wrong).record["refusal_code"],
            "production_root_mismatch",
        )


class CumulativeExactnessTests(AuthorityFixture):
    def test_persisted_prior_row_drift_is_refused_before_any_mutation(self):
        original = P._parquet_bytes

        def drifting(frame):
            frame = frame.copy()
            if "ra" in frame.columns and len(frame):
                frame.loc[0, "ra"] = float(frame.loc[0, "ra"]) + 1e-9  # a prior row
            return original(frame)

        P._parquet_bytes = drifting
        try:
            with self.assertRaises(P.PublicationRefused) as refused:
                self.publisher.authorization_inputs(self.candidate)
            outcome = self.publisher.publish(self.candidate, self.authorization)
        finally:
            P._parquet_bytes = original
        self.assertEqual(refused.exception.code, "cumulative_extension_not_exact")
        self.assertEqual(outcome.record["refusal_code"], "cumulative_extension_not_exact")
        self.assertFalse(self.gate_present())
        self.assertEqual(self.fingerprint(), self.baseline_fingerprint)


# ---------------------------------------------------------------------------
# Journal/path integrity on resume
# ---------------------------------------------------------------------------


class JournalIntegrityTests(AuthorityFixture):
    def interrupt(self, boundary="after_manifest_commit"):
        outcome = F.publisher_for(self.capability, fault_hook=_FailAt(boundary)).publish(
            self.candidate, self.authorization
        )
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        return Path(outcome.record["journal"])

    def edit_journal(self, path, edit):
        document = json.loads(path.read_text())
        edit(document)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    def assert_resume_refused(self, code):
        before = self.cumulative_hashes()
        outcome = self.publisher.publish(self.candidate, self.authorization)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.record["refusal_code"], code, outcome.record)
        self.assertEqual(self.cumulative_hashes(), before)
        self.assertTrue(self.gate_present())  # readers stay protected

    def test_redirected_journal_path_is_rejected(self):
        journal = self.interrupt()
        decoy = self.tmp / "decoy"
        decoy.mkdir()

        def redirect(document):
            document["descriptor"]["target_path"] = str(decoy)

        self.edit_journal(journal, redirect)
        self.assert_resume_refused("journal_contradiction")
        self.assertEqual(list(decoy.iterdir()), [])

    def test_redirected_cumulative_staging_is_rejected(self):
        journal = self.interrupt()

        def redirect(document):
            document["descriptor"]["metadata"]["cumulative"]["staged_relative"]["loci_index"] = (
                "../../elsewhere/loci_index.parquet"
            )

        self.edit_journal(journal, redirect)
        self.assert_resume_refused("path_contradiction")

    def test_symlink_substituted_staging_is_rejected(self):
        journal = self.interrupt()
        run_id = json.loads(journal.read_text())["descriptor"]["run_id"]
        staged = self.capability.staging_root / f"{run_id}-cumulative"
        attacker = self.tmp / "attacker"
        shutil.copytree(staged, attacker)
        shutil.rmtree(staged)
        staged.symlink_to(attacker, target_is_directory=True)
        self.assert_resume_refused("path_contradiction")

    def test_tampered_authoritative_manifest_sha_is_rejected(self):
        self.interrupt()
        manifest = self.root / "data/lsst_only/nightly/2026/07/01/manifest.json"
        payload = manifest.read_bytes()
        manifest.unlink()
        manifest.write_bytes(payload.replace(b'"authoritative":true', b'"authoritative":true '))
        self.assert_resume_refused("authority_contradiction")

    def test_gate_disagreeing_with_journal_is_rejected(self):
        self.interrupt()
        gate = history.publication_gate_path(self.root)
        document = json.loads(gate.read_text())
        document["expected_cumulative_sha256"]["loci_index"] = "0" * 64
        gate.write_text(json.dumps(document))
        self.assert_resume_refused("authority_contradiction")

    def test_corrupt_journal_fails_closed(self):
        journal = self.interrupt()
        journal.write_text("{not json")
        outcome = self.publisher.publish(self.candidate, self.authorization)
        self.assertEqual(outcome.record["refusal_code"], "journal_contradiction")
        self.assertEqual(outcome.record["failure_category"], "unrecoverable_journal_contradiction")
        self.assertFalse(outcome.record["retryable"])


# ---------------------------------------------------------------------------
# Chronology
# ---------------------------------------------------------------------------


class ChronologyTests(AuthorityFixture):
    def test_publication_chronology_is_machine_readable_and_ordered(self):
        outcome = self.publisher.publish(self.candidate, self.authorization)
        manifest = json.loads(
            (self.root / "data/lsst_only/nightly/2026/07/01/manifest.json").read_text()
        )
        chronology = manifest["authority"]["chronology"]
        self.assertEqual(
            chronology["query_request_started_at_utc"], "2026-09-03T20:13:33+00:00"
        )
        self.assertEqual(chronology["candidate_completed_at_utc"], "2026-09-20T10:00:00+00:00")
        self.assertEqual(chronology["publication_authorized_at_utc"], F.AUTHORIZED_AT)
        self.assertEqual(
            chronology["publication_transaction_started_at_utc"], F.FIXED_NOW.isoformat()
        )
        # The contradictory candidate placeholder is provenance, never chronology.
        self.assertEqual(manifest["finished_at_utc"], "2026-09-20T10:00:00+00:00")
        self.assertEqual(
            chronology["candidate_manifest_original"]["finished_at_utc"], f"{NIGHT}T00:00:00+00:00"
        )
        identities = manifest["authority"]["provenance_identities"]
        self.assertEqual(identities["candidate_execution_release_sha"], F.CANDIDATE_RELEASE)
        self.assertEqual(identities["publisher_release_sha"], F.PUBLISHER_RELEASE)
        record = outcome.record["chronology"]
        ordered = [
            record["query_request_started_at_utc"], record["candidate_completed_at_utc"],
            record["publication_authorized_at_utc"],
            record["publication_transaction_started_at_utc"],
            record["authority_commit_intent_at_utc"], record["authority_committed_at_utc"],
            record["terminal_record_at_utc"],
        ]
        parsed = [P._parse_utc(value, "t") for value in ordered]
        self.assertEqual(parsed, sorted(parsed))
        self.assertTrue(record["authority_commit_observed"])

    def test_authorization_cannot_precede_candidate_completion(self):
        early = F.authorize(self.publisher, self.candidate, authorized_at="2026-09-19T00:00:00+00:00")
        outcome = self.publisher.publish(self.candidate, early)
        self.assertEqual(outcome.record["refusal_code"], "chronology_invalid")
        self.assertFalse(self.gate_present())


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------


class RetryClassificationTests(AuthorityFixture):
    def test_categories_are_explicit(self):
        C = P.FailureCategory
        cases = (
            (OSError(errno.EACCES, "denied"), False, C.FILESYSTEM_PERMANENT),
            (OSError(errno.EROFS, "ro"), True, C.FILESYSTEM_PERMANENT),
            (OSError("no errno"), False, C.FILESYSTEM_PERMANENT),
            (OSError(errno.EIO, "io"), False, C.TRANSIENT_NETWORK),
            (OSError(errno.EIO, "io"), True, C.RECONCILIATION_INTERRUPTED),
            (ConnectionError("reset"), False, C.TRANSIENT_NETWORK),
            (LockUnavailable("busy"), True, C.LOCK_CONTENTION),
            (history.PublicationInProgress("gated"), False, C.LOCK_CONTENTION),
            (P.PublicationRefused("sentinel_drift", "x"), False, C.SENTINEL_DRIFT),
            (P.PublicationRefused("predecessor_gap", "x"), False, C.PREDECESSOR_GAP),
            (P.PublicationRefused("candidate_hash_mismatch", "x"), False, C.CANDIDATE_CORRUPTION),
            (P.PublicationRefused("authorization_expired", "x"), False, C.AUTHORIZATION_REFUSED),
            (P.PublicationRefused("journal_contradiction", "x"), False, C.JOURNAL_CONTRADICTION),
            (ValueError("bug"), False, C.UNCLASSIFIED),
        )
        for error, begun, expected in cases:
            with self.subTest(error=repr(error), begun=begun):
                self.assertEqual(P.classify_failure(error, authority_transition_begun=begun), expected)
        self.assertNotIn(C.FILESYSTEM_PERMANENT, P.RETRYABLE_CATEGORIES)
        self.assertIn(C.RECONCILIATION_INTERRUPTED, P.RETRYABLE_CATEGORIES)

    def test_reconciliation_lock_conflict_is_retryable_not_terminal(self):
        # A pre-existing foreign lock is refused as residue before any mutation;
        # here the conflict appears only after the authority gate exists.
        foreign = WriterLock(
            self.capability, SHARED_RECONCILIATION_LOCK_IDENTITY, "foreign-writer",
            transaction_id="foreign-writer:reconciliation",
        )
        original = P.RECONCILIATION_LOCK_WAIT_SECONDS
        P.RECONCILIATION_LOCK_WAIT_SECONDS = 0.05
        try:
            outcome = F.publisher_for(
                self.capability, fault_hook=_FailAt("after_authority_gate", foreign.acquire)
            ).publish(self.candidate, self.authorization)
        finally:
            P.RECONCILIATION_LOCK_WAIT_SECONDS = original
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        self.assertEqual(outcome.record["failure_category"], "transient_lock_contention")
        self.assertTrue(outcome.record["retryable"])
        foreign.release()
        self.assert_complete_and_idempotent(
            self.publisher.publish(self.candidate, self.authorization)
        )

    def test_permission_failure_after_the_gate_is_permanent_until_repaired(self):
        cumulative = history.cumulative_paths(self.root)["dir"]
        outcome = F.publisher_for(
            self.capability,
            fault_hook=_FailAt(
                "before_cumulative_install:loci_index", lambda: cumulative.chmod(0o500)
            ),
        ).publish(self.candidate, self.authorization)
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        self.assertEqual(outcome.record["failure_category"], "permanent_filesystem_or_permission")
        self.assertFalse(outcome.record["retryable"])
        self.assertTrue(outcome.record["production_mutated"])
        cumulative.chmod(0o700)
        self.assert_complete_and_idempotent(
            self.publisher.publish(self.candidate, self.authorization)
        )


# ---------------------------------------------------------------------------
# Capability contracts and the prior-free invariant
# ---------------------------------------------------------------------------

JUNE27 = {
    "hostname": "arnor.astro.washington.edu",
    "service_uid": 1533564,
    "production_root": "/astro/store/shire/ANTARES/data",
    "stage_root": "/astro/store/shire/ANTARES/work/publication/staging",
    "control_root": "/astro/store/shire/ANTARES/work/publication/control",
    "evidence_root": "/astro/store/shire/ANTARES/work/publication/evidence",
    "sentinel_cache_path": "/astro/store/shire/ANTARES/cache",
    "segment_cache_root": "/astro/store/shire/ANTARES/work/cache/fetch-segments-v1",
    "mount_binding": {
        "mount_point": "/astro/store/shire", "filesystem_type": "nfs4",
        "source": "shire.infiniband:/data/shire",
    },
    "sentinel_fingerprint_sha256": "52d9d30f0e004622485ba819af3bb56c81b704c22e7618b086a4ac398a76bf63",
    "manifest_count": 90,
    "authority_lock": {
        "path": "/astro/store/shire/ANTARES/work/publication/control/locks/publication-authority.flock",
        "device": 60,
        "inode": 1825692724162,
        "mode": "0600",
        "uid": 1533564,
        "gid": 2121533564,
        "size": 0,
    },
    "night_utc": "2026-06-27",
    "predecessor_night_utc": "2026-06-26",
    "candidate_root": "/astro/store/shire/ANTARES/work/canary/phase6f-recovery-0.4.3-4378bce-20260627-20260928T185444Z",
    "candidate_record_sha256": "1" * 64,
    "candidate_binding_sha256": "2" * 64,
    "candidate_provenance_sha256": "9" * 64,
    "artifact_sha256": {
        "loci.parquet": "86974614dc66349b5f0ad575905e81b8affdf6761205e4dbf7b59efa583f6f0c",
        "alerts.parquet": "2942e07c190d8bed1931c49c4f1f46149b4e39bcc2a440d36860c972d9cd5198",
        "manifest.json": "e5a9b2d3803df2bfec8f4bfd2caf230b12eda51f713eda2280bf3f22526647ad",
    },
    "publisher_release_sha": "e" * 40,
    "candidate_release_sha": "4378bce9a78e250dc897a9252b21dc69d41dcd0a",
    "cumulative_baseline_sha256": {
        "loci_index": "f75196d18690e610ab6e79231b244c3fddca396a68eea08dd2d0408e91d8b587",
        "nightly_summary": "85c5fac9c242fa2e7993155036ada649336b0affe8ffc8843d2c5733ea765114",
    },
    "expected_cumulative_sha256": {"loci_index": "3" * 64, "nightly_summary": "4" * 64},
    "expected_cumulative_schema_sha256": {"loci_index": "5" * 64, "nightly_summary": "6" * 64},
    "authorization_sha256": "7" * 64,
    "control_token_sha256": "a" * 64,
    "nonce": "8" * 32,
    "expires_at_utc": "2026-10-31T00:00:00+00:00",
}


class CapabilityContractTests(AuthorityFixture):
    def test_unsealed_production_binding_is_refused(self):
        binding = P.ProductionPublicationBinding(**JUNE27)
        with self.assertRaises(ProductionAuthorizationUnavailable):
            P.NightPublisher(binding, publisher_release_sha=F.PUBLISHER_RELEASE,
                             cache_root=self.tmp / "cache")

    def test_exact_control_token_issues_one_shot_capability(self):
        candidate_binding = "c" * 64
        candidate = replace(
            self.candidate,
            provenance={**self.candidate.provenance, "binding_sha256": candidate_binding},
        )
        authorization = F.authorize(self.publisher, candidate)
        sentinel = self.publisher.sentinel(NIGHT)
        production = P.production_binding_from_sentinel(sentinel)
        token = "d" * 64
        token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
        lock_path = self.tmp / "publication-authority.flock"
        lock_identity = {
            "path": str(lock_path), "device": 1, "inode": 2, "mode": "0600",
            "uid": 3, "gid": 4, "size": 0,
        }
        stage_root = self.capability.staging_root
        control_root = self.capability.journal_root.parent
        evidence_root = self.capability.evidence_root
        candidate_root = candidate.candidate_dir.parent
        with mock.patch.multiple(
            P,
            CONTROL_APPROVED_NIGHT=NIGHT,
            CONTROL_APPROVED_PREDECESSOR="2026-06-30",
            CONTROL_APPROVED_LOCK=lock_path,
            CONTROL_APPROVED_CANDIDATE_ROOT=candidate_root,
            CONTROL_APPROVED_CANDIDATE_RELEASE=F.CANDIDATE_RELEASE,
            PRODUCTION_DATA_ROOT=self.root,
            PRODUCTION_STAGE_ROOT=stage_root,
            PRODUCTION_CONTROL_ROOT=control_root,
            PRODUCTION_EVIDENCE_ROOT=evidence_root,
        ):
            binding = P.ProductionPublicationBinding(**{
                **JUNE27,
                "hostname": "arnor.example",
                "service_uid": 1234,
                "production_root": production["canonical_root"],
                "stage_root": str(stage_root),
                "control_root": str(control_root),
                "evidence_root": str(evidence_root),
                "sentinel_cache_path": production["predicates"]["cache_path"],
                "segment_cache_root": str(self.tmp / "segments"),
                "mount_binding": production["mount_binding"],
                "sentinel_fingerprint_sha256": production["durable_fingerprint_sha256"],
                "manifest_count": production["manifest_count"],
                "authority_lock": lock_identity,
                "night_utc": NIGHT,
                "predecessor_night_utc": "2026-06-30",
                "candidate_root": str(candidate_root),
                "candidate_record_sha256": candidate.record_sha256,
                "candidate_binding_sha256": candidate_binding,
                "candidate_provenance_sha256": candidate.provenance_sha256,
                "artifact_sha256": candidate.artifact_sha256(),
                "publisher_release_sha": F.PUBLISHER_RELEASE,
                "candidate_release_sha": F.CANDIDATE_RELEASE,
                "cumulative_baseline_sha256": production["cumulative_sha256"],
                "expected_cumulative_sha256": authorization.expected_cumulative_sha256,
                "authorization_sha256": authorization.digest,
                "control_token_sha256": token_sha256,
                "nonce": authorization.nonce,
                "expires_at_utc": authorization.expires_at_utc,
            })
            with self.assertRaises(ProductionAuthorizationUnavailable):
                P.issue_production_publication_capability(
                    binding, authorization, candidate, control_token="e" * 64,
                    sentinel=sentinel, now=F.FIXED_NOW, hostname="arnor.example",
                    uid=1234, authority_lock=lock_identity,
                )
            capability = P.issue_production_publication_capability(
                binding, authorization, candidate, control_token=token,
                sentinel=sentinel, now=F.FIXED_NOW, hostname="arnor.example",
                uid=1234, authority_lock=lock_identity,
            )
        self.assertEqual(capability.binding_sha256, binding.digest)
        self.assertEqual(capability.authorization_sha256, authorization.digest)
        self.assertEqual(capability.control_token_sha256, token_sha256)

    def test_june27_binding_contract_is_exact(self):
        binding = P.ProductionPublicationBinding(**JUNE27)
        self.assertEqual(binding.max_successful_uses, 1)
        self.assertEqual(binding.operation, "publish-night")
        self.assertEqual(len(binding.digest), 64)
        for field, value in (
            ("stage_root", "/astro/store/shire/ANTARES/data/staging"),  # inside production
            ("segment_cache_root", JUNE27["candidate_root"] + "/cache"),  # inside evidence
            ("predecessor_night_utc", "2026-06-25"),
            ("max_successful_uses", 2),
            ("publisher_release_sha", JUNE27["candidate_release_sha"]),
            ("nonce", "short"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(P.PublicationRefused):
                    P.ProductionPublicationBinding(**{**JUNE27, field: value})

    def test_binding_qualification_reports_every_mismatch(self):
        sentinel = self.publisher.sentinel(NIGHT)
        production = P.production_binding_from_sentinel(sentinel)
        binding = P.ProductionPublicationBinding(**{
            **JUNE27,
            "production_root": production["canonical_root"],
            "sentinel_cache_path": production["predicates"]["cache_path"],
            "stage_root": str(self.capability.staging_root),
            "control_root": str(self.capability.journal_root.parent),
            "evidence_root": str(self.capability.evidence_root),
            "segment_cache_root": str(self.tmp / "segments"),
            "candidate_root": str(self.night_root),
            "mount_binding": production["mount_binding"],
            "sentinel_fingerprint_sha256": production["durable_fingerprint_sha256"],
            "manifest_count": production["manifest_count"],
            "night_utc": NIGHT, "predecessor_night_utc": "2026-06-30",
            "candidate_record_sha256": self.candidate.record_sha256,
            "candidate_provenance_sha256": self.candidate.provenance_sha256,
            "artifact_sha256": self.candidate.artifact_sha256(),
            "publisher_release_sha": F.PUBLISHER_RELEASE,
            "candidate_release_sha": F.CANDIDATE_RELEASE,
            "cumulative_baseline_sha256": production["cumulative_sha256"],
            "expected_cumulative_sha256": self.authorization.expected_cumulative_sha256,
            "authorization_sha256": self.authorization.digest,
            "nonce": self.authorization.nonce,
            "expires_at_utc": self.authorization.expires_at_utc,
        })
        clean = P.qualify_production_binding(
            binding, self.authorization, hostname=JUNE27["hostname"], uid=JUNE27["service_uid"],
            sentinel=sentinel, now=F.FIXED_NOW,
        )
        self.assertEqual(clean, ())
        drifted = P.qualify_production_binding(
            binding, self.authorization, hostname="elsewhere.example", uid=1,
            sentinel=sentinel, now=F.FIXED_NOW,
        )
        self.assertEqual(drifted, ("hostname", "service_uid"))


class PriorFreeAndRangeContractTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.capability = F.make_capability(self.tmp)
        F.seed_production(self.capability, BASELINE_NIGHTS)

    def tearDown(self):
        self._tmp.cleanup()

    def test_live_provider_attestation_pins_the_reviewed_implementation(self):
        import src.operations.live_antares as live

        class LiveShapedAdapter:
            provider_name = "live-antares"
            scenario = "commissioning-v1"
            provider_module = "src.operations.live_antares"

        shaped_identity = acquisition_identity(LiveShapedAdapter())
        self.assertEqual(
            shaped_identity["provider_implementation_sha256"],
            hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest(),
        )
        with self.assertRaises(BackfillRefused):
            require_prior_free_acquisition(shaped_identity)

        approved = acquisition_identity(object.__new__(live.LiveAntaresProvider))
        require_prior_free_acquisition(approved)
        altered = {**approved, "adapter_implementation_sha256": "0" * 64}
        with self.assertRaises(BackfillRefused):
            require_prior_free_acquisition(altered)
        self.assertEqual(len(PRIOR_FREE_ACQUISITION_ATTESTATIONS), 1)

    def test_modified_adapter_refused_before_any_query(self):
        class ModifiedAdapter(F.SyntheticBackfillAdapter):
            provider_module = F.SyntheticBackfillAdapter.__module__

        adapter = ModifiedAdapter(4)
        controller = BackfillController(
            self.capability,
            adapter,
            release_sha=F.CANDIDATE_RELEASE,
            read_capability_factory=F.mock_read_capability,
            settings=BackfillSettings(segment_size=4),
            prior_free_attestations=F.fixture_prior_free_attestations(),
        )
        with self.assertRaises(BackfillRefused):
            controller.run("2026-07-01", "2026-07-01")
        self.assertEqual(adapter.queries["2026-07-01"], 0)

    def test_unattested_provider_cannot_acquire_prior_free(self):
        adapter = F.SyntheticBackfillAdapter(4)
        controller = BackfillController(
            self.capability, adapter, release_sha=F.CANDIDATE_RELEASE,
            read_capability_factory=F.mock_read_capability,
            settings=BackfillSettings(segment_size=4),
        )
        with self.assertRaises(BackfillRefused):
            controller.run("2026-07-01", "2026-07-01")
        self.assertEqual(adapter.queries["2026-07-01"], 0)

    def test_range_authorization_binds_science_provider_adapter_and_cache(self):
        publisher = F.publisher_for(self.capability)
        adapter = F.SyntheticBackfillAdapter(4)
        controller = BackfillController(
            self.capability, adapter, release_sha=F.CANDIDATE_RELEASE,
            read_capability_factory=F.mock_read_capability,
            settings=BackfillSettings(segment_size=4), publisher=publisher,
            prior_free_attestations=F.fixture_prior_free_attestations(),
        )
        binding = controller.range_binding("2026-07-01", "2026-07-02")
        self.assertEqual(binding["publication_concurrency"], 1)
        self.assertIsNone(binding["cache_root"])
        production = P.production_binding_from_sentinel(publisher.sentinel("2026-07-01"))

        def authority(**changes):
            fields = dict(
                start_date_utc="2026-07-01", end_date_utc="2026-07-02",
                initial_predecessor_date_utc="2026-06-30",
                initial_sentinel={k: production[k] for k in (
                    "canonical_root", "mount_binding", "durable_fingerprint_sha256",
                    "manifest_count", "cumulative_sha256")},
                candidate_release_sha=F.CANDIDATE_RELEASE,
                publisher_release_sha=F.PUBLISHER_RELEASE, authorized_by="control-fixture",
                authorized_at_utc="2026-09-28T00:00:00+00:00", expires_at_utc=F.EXPIRES_AT,
                **binding,
            )
            fields.update(changes)
            return RangePublicationAuthorization(**fields)

        for label, changed in (
            ("science", authority(science_contract_sha256="0" * 64)),
            ("configuration", authority(configuration_sha256="0" * 64)),
            ("adapter", authority(acquisition_identity={
                **binding["acquisition_identity"], "adapter_implementation_sha256": "0" * 64})),
            ("cache", authority(cache_root=str(self.tmp / "segments"))),
        ):
            with self.subTest(label=label):
                refused = BackfillController(
                    self.capability, adapter, release_sha=F.CANDIDATE_RELEASE,
                    read_capability_factory=F.mock_read_capability,
                    settings=BackfillSettings(segment_size=4), publisher=publisher,
                    range_authorization=changed,
                    prior_free_attestations=F.fixture_prior_free_attestations(),
                )
                with self.assertRaises(BackfillRefused):
                    refused.run("2026-07-01", "2026-07-02")
        with self.assertRaises(BackfillRefused):
            authority(publication_concurrency=2)
        self.assertEqual(adapter.queries, {})


# ---------------------------------------------------------------------------
# Segment cache confinement
# ---------------------------------------------------------------------------


class CacheConfinementTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def results(self, ids):
        from src.operations.fetch_checkpoint import FetchObjectResult

        return tuple(
            FetchObjectResult(
                locus_id,
                pd.DataFrame({"locus_id": [locus_id, locus_id], "mjd": [1.0, 2.0]}),
            )
            for locus_id in ids
        )

    def test_internal_symlink_substitution_is_refused(self):
        root = self.tmp / "cache"
        root.mkdir()
        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        (root / "objects").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(SegmentCacheRefused):
            F.make_segment_cache(root)
        (root / "objects").unlink()
        cache = F.make_segment_cache(root)
        key = "a" * 64
        (cache.index / "aa").symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(SegmentCacheRefused):
            cache.put(key, self.results(("A",)))
        self.assertIsNone(cache.get(key, ("A",)))
        self.assertEqual(list(elsewhere.iterdir()), [])

    def test_hit_requires_per_object_locus_ownership(self):
        cache = F.make_segment_cache(self.tmp)
        key = "b" * 64
        cache.put(key, self.results(("A", "B")))
        self.assertIsNotNone(cache.get(key, ("A", "B")))
        self.assertIsNone(cache.get(key, ("B", "A")))  # order is identity
        # Forge an entry whose rows for object A belong to B.
        from src.operations.cache import SEGMENT_CACHE_SCHEMA, _canonical, _sha256
        from src.operations.fetch_checkpoint import _parquet_payload

        forged_key = "c" * 64
        alerts = pd.DataFrame({"locus_id": ["B", "B", "A", "A"], "mjd": [1.0, 2.0, 1.0, 2.0]})
        parquet, schema = _parquet_payload(alerts)
        header = {
            "schema_version": SEGMENT_CACHE_SCHEMA, "key": forged_key,
            "objects": [
                {"locus_id": "A", "alert_rows": 2, "retry_count": 0, "retry_exception_types": []},
                {"locus_id": "B", "alert_rows": 2, "retry_count": 0, "retry_exception_types": []},
            ],
            "alerts_sha256": _sha256(parquet), "schema_sha256": schema,
        }
        payload = _canonical(header) + b"\n" + parquet
        digest = _sha256(payload)
        cache._atomic_write(cache._shard(cache.objects, digest), payload)
        cache._atomic_write(cache._shard(cache.index, forged_key), (digest + "\n").encode())
        rejected = cache.statistics()["rejected"]
        self.assertIsNone(cache.get(forged_key, ("A", "B")))
        self.assertEqual(cache.statistics()["rejected"], rejected + 1)

    def test_cache_cannot_overlap_any_authority_root(self):
        capability = F.make_capability(self.tmp, "workspace")
        inside = capability.root / "backfill" / "cache"
        inside.mkdir(parents=True)
        cache = F.make_segment_cache(inside)
        with self.assertRaises(SegmentCacheRefused):
            BackfillController(
                capability, F.SyntheticBackfillAdapter(4), release_sha=F.CANDIDATE_RELEASE,
                cache=cache,
            )
        separate = self.tmp / "segments"
        separate.mkdir()
        BackfillController(
            capability, F.SyntheticBackfillAdapter(4), release_sha=F.CANDIDATE_RELEASE,
            cache=F.make_segment_cache(separate),
        )

    def test_only_the_approved_root_is_acceptable_on_shire(self):
        from src.operations import cache as cache_module

        project = self.tmp / "ANTARES"
        approved = project / "work" / "cache" / "fetch-segments-v1"
        approved.mkdir(parents=True)
        other = project / "work" / "cache" / "other"
        other.mkdir()
        saved = (cache_module.MIDDLE_EARTH_PROJECT_ROOT, cache_module.PROPOSED_ARNOR_CACHE_ROOT)
        cache_module.MIDDLE_EARTH_PROJECT_ROOT = project.resolve()
        cache_module.PROPOSED_ARNOR_CACHE_ROOT = approved.resolve()
        try:
            with self.assertRaises(SegmentCacheRefused):
                SegmentCache(other.resolve(), forbidden_roots=())
            self.assertEqual(
                SegmentCache(approved.resolve(), forbidden_roots=()).root, approved.resolve()
            )
        finally:
            cache_module.MIDDLE_EARTH_PROJECT_ROOT, cache_module.PROPOSED_ARNOR_CACHE_ROOT = saved
        self.assertEqual(
            str(cache_module.PROPOSED_ARNOR_CACHE_ROOT),
            "/astro/store/shire/ANTARES/work/cache/fetch-segments-v1",
        )


if __name__ == "__main__":
    unittest.main()
