"""Focused offline tests for the Phase 6 query-result checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.operations.query_checkpoint import (
    COMMIT_MARKER_NAME,
    NON_AUTHORITATIVE_CLASSIFICATION,
    QueryCheckpointError,
    QueryResultCheckpointBindings,
    load_query_result_checkpoint,
    seal_query_result_checkpoint,
)
from src.operations.science import (
    NightQueryResult,
    NightScienceRequest,
    ProviderOutcome,
    QueryStageEvidence,
)


TARGET_DATE = "2026-06-27"
RELEASE_SHA = "2" * 40
CONFIGURATION_HASH = "3" * 64


def _identifier_hash(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _scientific_contract():
    return {
        "target_date_utc": TARGET_DATE,
        "interval": {
            "mjd_min": 61218.0,
            "mjd_max": 61219.0,
            "lower_bound": "inclusive",
            "upper_bound": "exclusive",
            "timezone": "UTC",
        },
        "query_tag": None,
        "lsst_only": True,
    }


def _execution_policy():
    return {
        "max_query_attempts": 2,
        "max_fetch_attempts_per_object": 3,
        "max_fetch_workers": 4,
    }


def _request(*, ingested_at="2026-08-27T00:00:00+00:00", range_label=None):
    return NightScienceRequest(
        TARGET_DATE,
        61218.0,
        61219.0,
        ingested_at_utc=ingested_at,
        query_tag=None,
        target_loci=None,
        range_label=range_label or f"ANTARES commissioning {TARGET_DATE}",
        lsst_only=True,
        prior_locus_ids=("ANT-PRIOR-2", "ANT-PRIOR-1"),
    )


def _frame():
    """Include nested missing-vs-null values that Arrow structs can conflate."""
    return pd.DataFrame(
        {
            "locus_id": pd.Series(
                ["ANT-LIVE-0002", "ANT-LIVE-0001", "ANT-LIVE-0003"],
                dtype="object",
            ),
            "ra": pd.Series([20.0, 10.0, 30.0], dtype="float64"),
            "dec": pd.Series([-2.0, -1.0, -3.0], dtype="float64"),
            "tags": pd.Series(["lsst", "lsst", "lsst"], dtype="object"),
            "survey": pd.Series(
                [
                    {"lsst": {"dia_object_id": None}},
                    {"lsst": {}},
                    None,
                ],
                dtype="object",
            ),
            "nested": pd.Series(
                [
                    {"present_null": None, "items": [None, {"x": None}]},
                    {"items": [None, {}]},
                    {},
                ],
                dtype="object",
            ),
            "newest_alert_observation_time": pd.Series(
                [61218.75, 61218.25, 61218.9], dtype="float64"
            ),
        }
    )


def _query_result(frame=None, request=None):
    frame = _frame() if frame is None else frame
    request = _request() if request is None else request
    locus_ids = frame["locus_id"].tolist() if "locus_id" in frame.columns else []
    details = {
        "completion_classification": (
            "COMPLETE_ZERO" if frame.empty else "COMPLETE_NONZERO"
        ),
        "target_date_utc": TARGET_DATE,
        "returned_loci": len(frame),
        "locus_order_sha256": _identifier_hash(locus_ids),
        "query_contract_sha256": _json_hash(_scientific_contract()),
        "execution_policy": _execution_policy(),
        "coverage_complete": True,
    }
    evidence = QueryStageEvidence(True, False, len(frame), (), details)
    return NightQueryResult(
        request,
        "live-antares",
        "commissioning-v1",
        ProviderOutcome.SUCCESS_ZERO if frame.empty else ProviderOutcome.SUCCESS,
        frame,
        evidence,
    )


def _bindings():
    return QueryResultCheckpointBindings(
        run_id="phase6-query-checkpoint-test",
        release_sha=RELEASE_SHA,
        configuration_hash=CONFIGURATION_HASH,
        target_date_utc=TARGET_DATE,
        provider_name="live-antares",
        provider_scenario="commissioning-v1",
        query_policy={
            "scientific_contract": _scientific_contract(),
            "execution_policy": _execution_policy(),
        },
    )


class Phase6QueryCheckpointTests(unittest.TestCase):
    def _run_root(self, temporary):
        root = Path(temporary) / _bindings().run_id
        root.mkdir(mode=0o700)
        return root

    def test_seal_and_load_preserve_order_dtypes_and_nested_null_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(
                root, source, _bindings(), rows_per_shard=2
            )
            loaded = load_query_result_checkpoint(root, source.request, _bindings())

            self.assertEqual(loaded.path, checkpoint)
            self.assertTrue(loaded.query_result.clean)
            self.assertEqual(
                loaded.query_result.query_completed, source.query_completed
            )
            pd.testing.assert_frame_equal(
                loaded.query_result.loci,
                source.loci,
                check_dtype=True,
                check_like=False,
            )
            # Present-null and missing are distinct after a complete process round trip.
            first_lsst = loaded.query_result.loci.iloc[0]["survey"]["lsst"]
            second_lsst = loaded.query_result.loci.iloc[1]["survey"]["lsst"]
            self.assertIn("dia_object_id", first_lsst)
            self.assertIsNone(first_lsst["dia_object_id"])
            self.assertNotIn("dia_object_id", second_lsst)
            self.assertEqual(
                loaded.manifest["classification"], NON_AUTHORITATIVE_CLASSIFICATION
            )
            self.assertFalse(loaded.manifest["authoritative"])
            self.assertFalse(loaded.manifest["publication_eligible"])
            self.assertFalse(loaded.manifest["frame"]["arrow_used"])
            self.assertTrue(
                loaded.manifest["semantic_round_trip"][
                    "nested_missing_vs_null_preserved"
                ]
            )
            self.assertEqual(loaded.query_evidence, source.evidence.as_dict())
            self.assertEqual(
                loaded.query_evidence_document,
                {
                    "provider": source.provider_name,
                    "outcome": source.outcome.value,
                    "evidence": source.evidence.as_dict(),
                },
            )
            self.assertRegex(loaded.query_evidence_sha256, r"^[0-9a-f]{64}$")
            self.assertRegex(loaded.integrity_sha256, r"^[0-9a-f]{64}$")

    def test_load_accepts_new_operational_timestamp_but_binds_scientific_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            seal_query_result_checkpoint(root, source, _bindings())

            later = _request(ingested_at="2026-08-29T12:34:56+00:00")
            loaded = load_query_result_checkpoint(root, later, _bindings())
            self.assertEqual(loaded.query_result.request, later)
            self.assertNotEqual(
                loaded.query_result.request.ingested_at_utc,
                source.request.ingested_at_utc,
            )
            changed_label = _request(
                ingested_at="2026-08-29T12:34:56+00:00",
                range_label="scientifically different fetch label",
            )
            with self.assertRaisesRegex(
                QueryCheckpointError, "scientific request"
            ):
                load_query_result_checkpoint(root, changed_label, _bindings())

    def test_corrupt_shard_is_rejected_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(root, source, _bindings())
            shard = checkpoint / "records-000000.jsonl"
            payload = bytearray(shard.read_bytes())
            payload[len(payload) // 2] ^= 1
            shard.write_bytes(payload)
            shard.chmod(0o600)
            with self.assertRaises(QueryCheckpointError):
                load_query_result_checkpoint(root, source.request, _bindings())

    def test_corrupt_query_evidence_and_contract_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(root, source, _bindings())
            evidence_path = checkpoint / "query-evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["evidence"]["returned_loci"] += 1
            evidence_path.write_text(
                json.dumps(
                    evidence,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            evidence_path.chmod(0o600)
            with self.assertRaises(QueryCheckpointError):
                load_query_result_checkpoint(root, source.request, _bindings())

        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            changed_details = dict(source.evidence.details)
            changed_details["query_contract_sha256"] = "f" * 64
            mismatched = NightQueryResult(
                source.request,
                source.provider_name,
                source.scenario,
                source.outcome,
                source.loci,
                QueryStageEvidence(True, False, len(source.loci), (), changed_details),
            )
            with self.assertRaisesRegex(QueryCheckpointError, "completion evidence"):
                seal_query_result_checkpoint(root, mismatched, _bindings())

    def test_stable_binding_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            seal_query_result_checkpoint(root, source, _bindings())
            mismatches = {
                "release": replace(_bindings(), release_sha="a" * 40),
                "configuration": replace(
                    _bindings(), configuration_hash="b" * 64
                ),
                "provider": replace(_bindings(), provider_name="other-provider"),
                "scenario": replace(_bindings(), provider_scenario="other-scenario"),
                "policy": replace(
                    _bindings(),
                    query_policy={
                        "scientific_contract": _scientific_contract(),
                        "execution_policy": {
                            **_execution_policy(),
                            "max_fetch_workers": 99,
                        },
                    },
                ),
            }
            for label, expected in mismatches.items():
                with self.subTest(label=label), self.assertRaisesRegex(
                    QueryCheckpointError, "stable bindings"
                ):
                    load_query_result_checkpoint(root, source.request, expected)

    def test_absent_commit_marker_and_unexpected_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(root, source, _bindings())
            marker = checkpoint / COMMIT_MARKER_NAME
            marker_payload = marker.read_bytes()
            marker.unlink()
            with self.assertRaisesRegex(QueryCheckpointError, "commit marker"):
                load_query_result_checkpoint(root, source.request, _bindings())
            marker.write_bytes(marker_payload)
            marker.chmod(0o600)
            unexpected = checkpoint / "unsealed-extra"
            unexpected.write_text("not part of checkpoint", encoding="utf-8")
            unexpected.chmod(0o600)
            with self.assertRaisesRegex(QueryCheckpointError, "unexpected"):
                load_query_result_checkpoint(root, source.request, _bindings())

    def test_existing_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(root, source, _bindings())
            marker_before = (checkpoint / COMMIT_MARKER_NAME).read_bytes()
            with self.assertRaisesRegex(QueryCheckpointError, "already exists"):
                seal_query_result_checkpoint(root, source, _bindings())
            self.assertEqual(
                (checkpoint / COMMIT_MARKER_NAME).read_bytes(), marker_before
            )

    def test_zero_result_outcome_is_reconstructed_internally(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            empty = pd.DataFrame()
            source = _query_result(empty)
            seal_query_result_checkpoint(root, source, _bindings())
            loaded = load_query_result_checkpoint(root, source.request, _bindings())
            self.assertEqual(loaded.query_result.outcome, ProviderOutcome.SUCCESS_ZERO)
            self.assertTrue(loaded.query_result.loci.empty)
            self.assertEqual(loaded.query_result.evidence.returned_loci, 0)

    def test_symlinked_parent_or_checkpoint_is_refused(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlinks unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            escape = Path(temporary) / "escape"
            escape.mkdir()
            (root / "checkpoints").symlink_to(escape, target_is_directory=True)
            with self.assertRaises(QueryCheckpointError):
                seal_query_result_checkpoint(root, _query_result(), _bindings())

        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            checkpoint = seal_query_result_checkpoint(root, source, _bindings())
            moved = checkpoint.with_name("real-query-result")
            checkpoint.rename(moved)
            checkpoint.symlink_to(moved, target_is_directory=True)
            with self.assertRaises(QueryCheckpointError):
                load_query_result_checkpoint(root, source.request, _bindings())

    def test_incomplete_or_duplicate_query_result_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            issue_evidence = QueryStageEvidence(
                False,
                True,
                len(source.loci),
                (),
                source.evidence.details,
            )
            incomplete = NightQueryResult(
                source.request,
                source.provider_name,
                source.scenario,
                ProviderOutcome.QUERY_INTERRUPTION,
                source.loci,
                issue_evidence,
            )
            with self.assertRaisesRegex(QueryCheckpointError, "clean completed"):
                seal_query_result_checkpoint(root, incomplete, _bindings())

        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            duplicate = _frame()
            duplicate.loc[2, "locus_id"] = duplicate.loc[0, "locus_id"]
            result = _query_result(duplicate)
            with self.assertRaisesRegex(QueryCheckpointError, "unique"):
                seal_query_result_checkpoint(root, result, _bindings())

    def test_fresh_process_reopens_without_search_or_full_external_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._run_root(temporary)
            source = _query_result()
            seal_query_result_checkpoint(root, source, _bindings(), rows_per_shard=1)
            script = r'''
import json
import sys
from pathlib import Path
from src.operations.query_checkpoint import QueryResultCheckpointBindings, load_query_result_checkpoint
from src.operations.science import NightScienceRequest

root = Path(sys.argv[1])
bindings = QueryResultCheckpointBindings(
    run_id=root.name,
    release_sha="2" * 40,
    configuration_hash="3" * 64,
    target_date_utc="2026-06-27",
    provider_name="live-antares",
    provider_scenario="commissioning-v1",
    query_policy={
        "scientific_contract": {
            "target_date_utc": "2026-06-27",
            "interval": {
                "mjd_min": 61218.0, "mjd_max": 61219.0,
                "lower_bound": "inclusive", "upper_bound": "exclusive",
                "timezone": "UTC",
            },
            "lsst_only": True, "query_tag": None,
        },
        "execution_policy": {
            "max_query_attempts": 2,
            "max_fetch_attempts_per_object": 3,
            "max_fetch_workers": 4,
        },
    },
)
request = NightScienceRequest(
    "2026-06-27", 61218.0, 61219.0,
    ingested_at_utc="2026-08-29T00:00:00+00:00",
    query_tag=None, target_loci=None,
    range_label="ANTARES commissioning 2026-06-27",
    lsst_only=True,
    prior_locus_ids=("ANT-PRIOR-2", "ANT-PRIOR-1"),
)
loaded = load_query_result_checkpoint(root, request, bindings)
print(json.dumps({
    "ids": loaded.query_result.loci["locus_id"].tolist(),
    "evidence_sha256": loaded.query_evidence_sha256,
    "clean": loaded.query_result.clean,
    "request_equal": loaded.query_result.request == request,
}, sort_keys=True))
'''
            completed = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=Path(__file__).resolve().parents[1],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            )
            value = json.loads(completed.stdout)
            self.assertTrue(value["clean"])
            self.assertTrue(value["request_equal"])
            self.assertEqual(value["ids"], source.loci["locus_id"].tolist())
            self.assertRegex(value["evidence_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
