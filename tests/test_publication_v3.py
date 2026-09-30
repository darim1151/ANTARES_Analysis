"""V3 authority/publication contract against synthetic production fixtures."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import v3_fixtures as F
from src import history
from src.operations import offline_recovery
from src.operations.publication import (
    PUBLICATION_LOCK_NAME,
    NightPublisher,
    PublicationAuthorization,
    PublicationRefused,
    _physical_nights,
    authoritative_nights,
    load_authorization,
    load_backfill_candidate,
    load_offline_recovery_candidate,
    write_authorization,
)
from src.operations.writer import (
    InjectedWriterFailure,
    ProductionAuthorizationUnavailable,
    nightly_target_relative,
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        digest.update(str(path.relative_to(root)).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


class _FailOnce:
    def __init__(self, point):
        self.point = point
        self.fired = False

    def __call__(self, point, details):
        if point == self.point and not self.fired:
            self.fired = True
            raise InjectedWriterFailure(point)


class PublicationFixture(unittest.TestCase):
    NIGHT = "2026-07-01"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.capability = F.make_capability(self.tmp)
        F.seed_production(self.capability, ["2026-06-29", "2026-06-30"])
        self.candidate = F.make_backfill_candidate(
            self.tmp, self.NIGHT, june27_style_defect=True
        )
        self.publisher = F.publisher_for(self.capability)
        self.baseline = self.publisher.sentinel(self.NIGHT)["durable_fingerprint_sha256"]

    def tearDown(self):
        self._tmp.cleanup()

    def fingerprint(self):
        return self.publisher.sentinel(self.NIGHT)["durable_fingerprint_sha256"]

    def assert_production_unchanged(self):
        self.assertEqual(self.fingerprint(), self.baseline)
        self.assertEqual(
            authoritative_nights(self.capability.published_root), ("2026-06-29", "2026-06-30")
        )


class PublicationHappyPathTests(PublicationFixture):
    def test_atomic_publication_extends_production_exactly_once(self):
        candidate_before = _tree_digest(self.candidate.candidate_dir)
        outcome = self.publisher.publish(
            self.candidate, F.authorize(self.publisher, self.candidate)
        )
        self.assertTrue(outcome.success, outcome.record)
        self.assertEqual(outcome.status, "published")
        record = outcome.record
        self.assertEqual(record["status"], "PUBLISHED")
        self.assertTrue(record["verification"]["passed"])
        self.assertTrue(all(record["verification"]["checks"].values()))
        self.assertNotEqual(record["resulting_production_fingerprint"], self.baseline)
        self.assertEqual(record["predecessor"]["production_fingerprint"], self.baseline)
        self.assertEqual(
            authoritative_nights(self.capability.published_root),
            ("2026-06-29", "2026-06-30", self.NIGHT),
        )
        target = self.capability.published_root / nightly_target_relative(self.NIGHT)
        self.assertEqual(sorted(p.name for p in target.iterdir()),
                         ["alerts.parquet", "loci.parquet", "manifest.json"])
        for name in ("loci.parquet", "alerts.parquet"):
            self.assertEqual(
                hashlib.sha256((target / name).read_bytes()).hexdigest(),
                self.candidate.artifacts[name]["sha256"],
            )
        # Candidate bytes and record are immutable.
        self.assertEqual(_tree_digest(self.candidate.candidate_dir), candidate_before)
        # No transaction residue anywhere.
        self.assertEqual(list(self.capability.staging_root.iterdir()), [])
        # Only the persistent kernel-released lifecycle lock file remains.
        self.assertEqual(
            [p.name for p in self.capability.lock_root.iterdir()], [PUBLICATION_LOCK_NAME]
        )
        self.assertTrue(Path(outcome.record_path).is_file())

    def test_cumulative_extension_matches_canonical_full_rebuild(self):
        self.publisher.publish(self.candidate, F.authorize(self.publisher, self.candidate))
        rebuilt_root = self.tmp / "rebuild"
        loci, summary = history.update_cumulative_indexes(
            data_root=self.capability.published_root, output_dir=rebuilt_root
        )
        paths = history.cumulative_paths(self.capability.published_root)
        pd.testing.assert_frame_equal(
            pd.read_parquet(paths["loci_index"]), loci.reset_index(drop=True),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(paths["nightly_summary"]), summary.reset_index(drop=True),
            check_dtype=False,
        )

    def test_authoritative_chronology_and_preserved_candidate_provenance(self):
        outcome = self.publisher.publish(
            self.candidate, F.authorize(self.publisher, self.candidate)
        )
        target = self.capability.published_root / nightly_target_relative(self.NIGHT)
        manifest = json.loads((target / "manifest.json").read_text())
        candidate_manifest = json.loads((self.candidate.candidate_dir / "manifest.json").read_text())
        self.assertIs(manifest["authoritative"], True)
        self.assertIs(candidate_manifest.get("authoritative", False), False)
        chronology = manifest["authority"]["chronology"]
        self.assertEqual(manifest["finished_at_utc"], self.candidate.construction_completed_at_utc)
        self.assertEqual(chronology["candidate_manifest_original"]["finished_at_utc"],
                         f"{self.NIGHT}T00:00:00+00:00")
        self.assertIn("finished_at_utc_precedes_started_at_utc",
                      chronology["candidate_manifest_defects"])
        self.assertLessEqual(manifest["started_at_utc"], manifest["finished_at_utc"])
        self.assertLessEqual(
            manifest["finished_at_utc"],
            chronology["publication_transaction_started_at_utc"],
        )
        self.assertEqual(chronology["publication_authorized_at_utc"], F.AUTHORIZED_AT)
        self.assertEqual(
            manifest["authority"]["candidate"]["manifest_sha256"],
            self.candidate.artifacts["manifest.json"]["sha256"],
        )
        record_chronology = outcome.record["chronology"]
        self.assertIsNotNone(record_chronology["authority_committed_at_utc"])
        self.assertTrue(record_chronology["authority_commit_observed"])
        self.assertTrue(record_chronology["ordering_verified"])
        self.assertNotIn("publication_committed_at_utc", manifest["authority"])
        # The contradictory placeholder never reaches authoritative cumulative state.
        summary = pd.read_parquet(
            history.cumulative_paths(self.capability.published_root)["nightly_summary"]
        )
        row = summary[summary["date_utc"] == self.NIGHT].iloc[0]
        self.assertEqual(row["finished_at_utc"], self.candidate.construction_completed_at_utc)

    def test_chronology_contradiction_without_trustworthy_completion_is_refused(self):
        bad = F.make_backfill_candidate(
            self.tmp / "bad", self.NIGHT, june27_style_defect=True,
            constructed_at_utc="2026-08-01T00:00:00+00:00",  # precedes acquisition start
        )
        # The contradiction is refused while planning, before any authorization.
        with self.assertRaises(PublicationRefused) as refused:
            F.authorize(self.publisher, bad)
        self.assertEqual(refused.exception.code, "chronology_invalid")
        self.assert_production_unchanged()


class PublicationRefusalTests(PublicationFixture):
    def assert_refused(self, outcome, code):
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.status, "refused")
        self.assertEqual(outcome.record["refusal_code"], code, outcome.record)
        self.assert_production_unchanged()

    def test_authorization_absent(self):
        self.assert_refused(self.publisher.publish(self.candidate, None), "authorization_absent")

    def test_wrong_predecessor_in_authorization(self):
        with self.assertRaises(PublicationRefused) as refused:
            F.authorize(self.publisher, self.candidate, predecessor="2026-06-29")
        self.assertEqual(refused.exception.code, "authorization_invalid")
        self.assert_production_unchanged()

    def test_production_predecessor_gap(self):
        later = F.make_backfill_candidate(self.tmp / "later", "2026-07-02")
        authorization = F.authorize(self.publisher, later)  # predecessor 07-01 absent
        self.assert_refused(self.publisher.publish(later, authorization), "predecessor_gap")

    def test_sentinel_mismatch(self):
        production = dict(self.publisher.authorization_inputs(self.candidate)["production"])
        production["durable_fingerprint_sha256"] = "0" * 64
        authorization = F.authorize(self.publisher, self.candidate, production=production)
        self.assert_refused(self.publisher.publish(self.candidate, authorization), "sentinel_drift")

    def test_production_drift_after_authorization(self):
        authorization = F.authorize(self.publisher, self.candidate)
        drift = self.capability.published_root / "analysis" / "note.txt"
        drift.parent.mkdir(parents=True)
        drift.write_text("drift")
        outcome = self.publisher.publish(self.candidate, authorization)
        self.assertEqual(outcome.record["refusal_code"], "sentinel_drift")
        self.assertEqual(
            authoritative_nights(self.capability.published_root), ("2026-06-29", "2026-06-30")
        )

    def test_candidate_hash_mismatch(self):
        authorization = F.authorize(self.publisher, self.candidate)
        loci = self.candidate.candidate_dir / "loci.parquet"
        original = loci.read_bytes()
        loci.write_bytes(original + b"\0")
        try:
            self.assert_refused(
                self.publisher.publish(self.candidate, authorization), "candidate_hash_mismatch"
            )
        finally:
            loci.write_bytes(original)

    def test_wrong_release_identities(self):
        authorization = F.authorize(self.publisher, self.candidate)
        other = NightPublisher(
            self.capability, publisher_release_sha="c" * 40,
            cache_root=self.capability.root / "absent-cache",
            mountinfo_lines=F.mountinfo_for(self.capability.published_root),
            clock=F.fixed_clock,
        )
        self.assert_refused(other.publish(self.candidate, authorization),
                            "authorization_release_mismatch")
        foreign = F.make_backfill_candidate(self.tmp / "foreign", self.NIGHT,
                                            release_sha="d" * 40)
        self.assert_refused(self.publisher.publish(foreign, authorization),
                            "authorization_candidate_mismatch")

    def test_wrong_night_candidate(self):
        authorization = F.authorize(self.publisher, self.candidate)
        other = F.make_backfill_candidate(self.tmp / "other", "2026-07-02")
        self.assert_refused(self.publisher.publish(other, authorization),
                            "authorization_night_mismatch")

    def test_unvalidated_candidate_cannot_load(self):
        record_path = self.candidate.record_path
        record = json.loads(record_path.read_text())
        record["validation_passed"] = False
        record_path.write_text(json.dumps(record))
        with self.assertRaises(PublicationRefused):
            load_backfill_candidate(record_path.parent.parent)

    def test_transaction_residue_blocks(self):
        (self.capability.staging_root / "leftover").mkdir()
        self.assert_refused(
            self.publisher.publish(self.candidate, F.authorize(self.publisher, self.candidate)),
            "transaction_residue",
        )

    def test_production_capability_is_not_issued(self):
        with self.assertRaises(ProductionAuthorizationUnavailable):
            NightPublisher(object(), publisher_release_sha=F.PUBLISHER_RELEASE,
                           cache_root=self.tmp / "cache")

    def test_authorization_round_trip_and_tamper_detection(self):
        authorization = F.authorize(self.publisher, self.candidate)
        path = self.tmp / "auth.json"
        write_authorization(path, authorization)
        self.assertEqual(load_authorization(path).digest, authorization.digest)
        document = json.loads(path.read_text())
        document["unexpected"] = True
        path.write_text(json.dumps(document))
        with self.assertRaises(PublicationRefused):
            load_authorization(path)
        with self.assertRaises(FileExistsError if False else Exception):
            write_authorization(path, authorization)  # never overwrites evidence


class PublicationIdempotencyAndRecoveryTests(PublicationFixture):
    def test_retry_is_idempotent_and_duplicate_authority_refused(self):
        authorization = F.authorize(self.publisher, self.candidate)
        first = self.publisher.publish(self.candidate, authorization)
        after = self.fingerprint()
        again = self.publisher.publish(self.candidate, authorization)
        self.assertTrue(again.success)
        self.assertEqual(again.status, "already_published")
        self.assertTrue(again.record["idempotent_replay"])
        self.assertEqual(self.fingerprint(), after)
        duplicate = PublicationAuthorization(**{
            **authorization.as_dict(), "authorized_at_utc": "2026-09-28T01:00:00+00:00",
        })
        refused = self.publisher.publish(self.candidate, duplicate)
        self.assertEqual(refused.record["refusal_code"], "already_authoritative")
        self.assertEqual(self.fingerprint(), after)
        self.assertEqual(
            authoritative_nights(self.capability.published_root).count(self.NIGHT), 1
        )
        self.assertEqual(first.record["transaction_id"], again.record["transaction_id"])

    def test_failure_before_commit_leaves_production_exactly_unchanged(self):
        authorization = F.authorize(self.publisher, self.candidate)
        failing = F.publisher_for(
            self.capability, fault_hook=_FailOnce("before_authority_gate")
        )
        outcome = failing.publish(self.candidate, authorization)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.record["status"], "NOT_COMMITTED")
        self.assertEqual(outcome.record["authority_state"], "NOT_COMMITTED")
        self.assertTrue(outcome.record["retryable"])
        self.assertFalse(outcome.record["production_mutated"])
        self.assert_production_unchanged()
        self.assertEqual(list(self.capability.staging_root.iterdir()), [])
        # Retry with the same authorization: no re-acquisition, new attempt id.
        retried = self.publisher.publish(self.candidate, authorization)
        self.assertTrue(retried.success, retried.record)
        self.assertTrue(retried.record["transaction_id"].endswith("-a2"))

    def test_failure_inside_reserved_target_is_never_partially_authoritative(self):
        authorization = F.authorize(self.publisher, self.candidate)
        failing = F.publisher_for(self.capability, fault_hook=_FailOnce("after_data_links"))
        outcome = failing.publish(self.candidate, authorization)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        self.assertTrue(outcome.record["production_mutated"])
        # Canonical readers refuse; the physical partition has no manifest yet.
        with self.assertRaises(history.PublicationInProgress):
            authoritative_nights(self.capability.published_root)
        self.assertEqual(
            _physical_nights(self.capability.published_root), ("2026-06-29", "2026-06-30")
        )
        paths = history.cumulative_paths(self.capability.published_root)
        self.assertFalse(
            pd.read_parquet(paths["loci_index"])["night_date_utc"].eq(self.NIGHT).any()
        )
        # The same authorization rolls the gated transition forward.
        again = self.publisher.publish(self.candidate, authorization)
        self.assertTrue(again.success, again.record)
        self.assertTrue(again.record["recovered_by_resume"])
        self.assertEqual(
            authoritative_nights(self.capability.published_root),
            ("2026-06-29", "2026-06-30", self.NIGHT),
        )

    def test_failure_after_commit_resumes_reconciliation_without_reacquisition(self):
        authorization = F.authorize(self.publisher, self.candidate)
        later = F.make_backfill_candidate(self.tmp / "later", "2026-07-02")
        later_authorization = F.authorize(self.publisher, later)
        failing = F.publisher_for(
            self.capability,
            fault_hook=_FailOnce("before_cumulative_install:nightly_summary"),
        )
        outcome = failing.publish(self.candidate, authorization)
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.record["status"], "RECONCILIATION_REQUIRED")
        self.assertTrue(outcome.record["retryable"])
        # A later night cannot publish while reconciliation is pending.
        blocked = self.publisher.publish(later, later_authorization)
        self.assertEqual(blocked.record["refusal_code"], "authority_transition_pending")
        resumed = self.publisher.publish(self.candidate, authorization)
        self.assertTrue(resumed.success, resumed.record)
        self.assertTrue(resumed.record["recovered_by_resume"])
        self.assertTrue(resumed.record["verification"]["strict"])
        self.assertTrue(resumed.record["verification"]["passed"])
        replay = self.publisher.publish(self.candidate, authorization)
        self.assertEqual(replay.status, "already_published")

    def test_older_night_replay_after_later_publication(self):
        first = F.authorize(self.publisher, self.candidate)
        self.publisher.publish(self.candidate, first)
        later = F.make_backfill_candidate(self.tmp / "later", "2026-07-02")
        self.assertTrue(
            self.publisher.publish(later, F.authorize(self.publisher, later)).success
        )
        replay = self.publisher.publish(self.candidate, first)
        self.assertEqual(replay.status, "already_published")
        self.assertFalse(replay.record["replay_verification"]["strict"])
        self.assertTrue(replay.record["verification"]["strict"])  # commit-time evidence


class Phase6SchemaAuthorityTests(unittest.TestCase):
    """The authoritative manifest must still satisfy the strict Phase 6 validator."""

    def test_authoritative_phase6_manifest_revalidates_with_unchanged_science(self):
        from datetime import datetime, timedelta, timezone

        import test_operations_phase6 as phase6
        from src.operations.publication import (
            NightCandidate,
            build_authoritative_manifest,
        )
        from src.operations.science import build_night_artifacts, reopen_and_validate_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            locus = phase6.FakeLocus(lightcurve=phase6._lightcurve())
            provider = phase6._mock_provider(
                root, phase6._canonical_search([locus]), lambda _id: locus,
                canonical_tiles=True,
            )
            artifacts = build_night_artifacts(provider.fetch_night(phase6._request()))
        manifest = json.loads(artifacts["manifest.json"])
        self.assertEqual(manifest["schema_version"], "phase6.commissioning-candidate.v1")
        now = datetime.now(timezone.utc)
        hashes = {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in artifacts.items()
        }
        candidate = NightCandidate(
            kind="offline-recovery", date_utc="2026-06-27",
            candidate_dir=Path(temporary), record_path=Path(temporary) / "record",
            record_sha256="1" * 64, release_sha=F.CANDIDATE_RELEASE, artifacts=hashes,
            loci=manifest["actual_loci"], alerts=manifest["alert_rows"],
            validation_passed=True, authoritative=False,
            construction_completed_at_utc=now.isoformat(),
        )
        root = "/fixture/production"
        authorization = PublicationAuthorization(
            date_utc="2026-06-27", predecessor_date_utc="2026-06-26",
            candidate_kind="offline-recovery", candidate_dir=str(Path(temporary)),
            candidate_record_sha256="1" * 64,
            candidate_provenance_sha256=candidate.provenance_sha256,
            candidate_release_sha=F.CANDIDATE_RELEASE,
            publisher_release_sha=F.PUBLISHER_RELEASE,
            artifact_sha256={name: value["sha256"] for name, value in hashes.items()},
            production={
                "canonical_root": root,
                "mount_binding": {"mount_point": "/fixture", "filesystem_type": "nfs4",
                                  "source": "fixture:/production"},
                "durable_fingerprint_sha256": "2" * 64,
                "manifest_count": 90,
                "cumulative_sha256": {"loci_index": "3" * 64, "nightly_summary": "4" * 64},
                "predicates": {
                    "target_path": str(history.nightly_paths(Path(root), "2026-06-27")["dir"]),
                    "target_absent": True, "cache_path": "/fixture/cache",
                    "cache_absent": True, "transaction_artifacts": [],
                },
            },
            expected_cumulative_sha256={"loci_index": "5" * 64, "nightly_summary": "6" * 64},
            nonce="7" * 32, authorized_by="control-fixture",
            authorized_at_utc=(now + timedelta(milliseconds=1)).isoformat(),
            expires_at_utc=(now + timedelta(days=1)).isoformat(),
        )
        authoritative = build_authoritative_manifest(
            artifacts["manifest.json"], candidate, authorization,
            transaction_id="v3pub-fixture",
            publication_started_at_utc=(now + timedelta(seconds=1)).isoformat(),
        )
        reopened = reopen_and_validate_artifacts({
            "loci.parquet": artifacts["loci.parquet"],
            "alerts.parquet": artifacts["alerts.parquet"],
            "manifest.json": authoritative,
        })
        self.assertIs(reopened.manifest["authoritative"], True)
        self.assertEqual(
            reopened.manifest["authority"]["candidate"]["manifest_sha256"],
            hashes["manifest.json"]["sha256"],
        )


def _offline_recovery_root(parent: Path, night: str) -> Path:
    root = parent / "phase6f-recovery-0.4.3-fixture"
    for name in ("candidate", "evidence", "status"):
        (root / name).mkdir(parents=True)
    artifacts = dict(F.synthetic_artifacts(night))
    manifest = json.loads(artifacts["manifest.json"])
    manifest.update({
        "authoritative": False, "publishable": False, "publication_authorized": False,
        "offline_recovery_contract": offline_recovery.SCIENTIFIC_ARTIFACT_CONTRACT,
        "started_at_utc": "2026-09-03T20:13:33+00:00",
        "finished_at_utc": f"{night}T00:00:00+00:00",
    })
    artifacts["manifest.json"] = F.manifest_bytes(manifest)
    hashes = {}
    for name, payload in artifacts.items():
        (root / "candidate" / name).write_bytes(payload)
        hashes[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (root / "evidence" / "artifacts.json").write_text(json.dumps(hashes))
    source = "7" * 64
    binding = {
        "schema_version": offline_recovery.CONTRACT, "run_id": root.name,
        "night": night, "consumer_sha": F.CANDIDATE_RELEASE,
        "source_sha": offline_recovery.SOURCE_SHA, "source_root": "/fixture/source",
        "source_durable_identity": source, "query_identity": "e" * 64,
        "fetch_identity": "f" * 64, "authoritative": False, "publishable": False,
        "publication_authorized": False,
    }
    (root / "binding.json").write_text(json.dumps(binding))
    seal = hashlib.sha256((root / "binding.json").read_bytes()).hexdigest()
    (root / "binding.sha256").write_text(seal + "\n")
    final = {
        "schema_version": offline_recovery.CONTRACT, "run_id": root.name,
        "success": True, "status": "RECOVERY_COMPLETE_UNPUBLISHED",
        "authoritative": False, "publishable": False, "publication_attempted": False,
        "binding_sha256": seal, "artifacts": hashes,
        "callback_and_network_counts": {"network_attempts": 0, "fetch_callbacks": 0},
        "source_before_sha256": source, "source_after_sha256": source,
        "production_before_sha256": "5" * 64, "production_after_sha256": "5" * 64,
        "validation": {"append_ready": True},
        "fetch_checkpoint": {"reused_segments": 3, "fetched_segments": 0},
        "finished_at_utc": "2026-09-28T20:42:01.050479+00:00",
    }
    (root / "status" / "RECOVERY_FINAL.json").write_text(json.dumps(final))
    return root


class OfflineRecoveryCandidateTests(PublicationFixture):
    def test_offline_recovery_candidate_publishes_with_separate_authority(self):
        root = _offline_recovery_root(self.tmp, self.NIGHT)
        before = _tree_digest(root)
        candidate = load_offline_recovery_candidate(root)
        self.assertEqual(candidate.kind, "offline-recovery")
        self.assertEqual(candidate.provenance["fetched_segments"], 0)
        outcome = self.publisher.publish(candidate, F.authorize(self.publisher, candidate))
        self.assertTrue(outcome.success, outcome.record)
        chronology = outcome.record["chronology"]
        self.assertEqual(chronology["candidate_completed_at_utc"],
                         "2026-09-28T20:42:01.050479+00:00")
        self.assertIn("finished_at_utc_is_night_start_request_placeholder",
                      chronology["candidate_manifest_defects"])
        self.assertEqual(_tree_digest(root), before)  # recovery root untouched

    def test_non_terminal_or_dirty_recovery_is_refused(self):
        root = _offline_recovery_root(self.tmp, self.NIGHT)
        final_path = root / "status" / "RECOVERY_FINAL.json"
        clean = final_path.read_text()
        for mutation in (
            {"status": "BLOCKED"},
            {"success": False},
            {"publication_attempted": True},
            {"callback_and_network_counts": {"network_attempts": 1}},
            {"source_after_sha256": "8" * 64},
        ):
            with self.subTest(mutation=mutation):
                final_path.write_text(json.dumps({**json.loads(clean), **mutation}))
                with self.assertRaises(PublicationRefused):
                    load_offline_recovery_candidate(root)
        final_path.write_text(clean)
        (root / "binding.sha256").write_text("0" * 64 + "\n")
        with self.assertRaises(PublicationRefused):
            load_offline_recovery_candidate(root)


if __name__ == "__main__":
    unittest.main()
