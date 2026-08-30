import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.operations.fetch_checkpoint import (
    FetchCheckpointAmbiguous,
    FetchCheckpointBinding,
    FetchCheckpointBindingError,
    FetchCheckpointCorrupt,
    FetchCheckpointFetchError,
    FetchObjectResult,
    SegmentedFetchCheckpoint,
)
from src.operations.live_antares import LIVE_ANTARES_READ, LiveAntaresReadCapability


RELEASE_SHA = "1" * 40
TARGET_DATE = "2026-06-27"


def _digest(value):
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identifier_digest(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _binding(ids, *, config="configuration", segment_size=2):
    return FetchCheckpointBinding(
        run_id="run",
        release_sha=RELEASE_SHA,
        configuration_sha256=_digest(config.encode("utf-8")),
        target_date_utc=TARGET_DATE,
        mjd_min=61218.0,
        mjd_max=61219.0,
        provider_name="live-antares",
        provider_scenario="commissioning-v1",
        provider_policy_sha256=_digest({"workers": 4, "attempts": 3}),
        query_contract_sha256=_digest({"mjd": [61218.0, 61219.0]}),
        query_identity_sha256=_digest({"result": list(ids)}),
        query_locus_order_sha256=_identifier_digest(ids),
        expected_objects=len(ids),
        segment_size=segment_size,
    )


def _capability(run_root):
    return LiveAntaresReadCapability.for_local_mock(
        run_root,
        run_id="run",
        target_date_utc=TARGET_DATE,
        release_sha=RELEASE_SHA,
        authority=LIVE_ANTARES_READ,
    )


def _frame(locus_id):
    number = int(locus_id.rsplit("-", 1)[1])
    if number == 2:
        return None
    return pd.DataFrame(
        {
            "mjd": [61218.1 + number / 1000.0, 61218.2 + number / 1000.0],
            "flux": [float(number), float(number) + 0.5],
            "range_label": ["commissioning", "commissioning"],
        }
    )


def _results_for(requested):
    # Deliberately reverse completion order. The checkpoint must restore sealed
    # query order, not preserve concurrent-future completion order.
    results = []
    for locus_id in reversed(requested):
        retry = 1 if locus_id.endswith("-3") else 0
        results.append(
            FetchObjectResult(
                locus_id,
                _frame(locus_id),
                retry_count=retry,
                retry_exception_types=("requests.exceptions.JSONDecodeError",)
                if retry
                else (),
            )
        )
    return results


class SegmentedFetchCheckpointTests(unittest.TestCase):
    def _new(self, base, ids, *, binding=None):
        run_root = Path(base) / "run"
        run_root.mkdir()
        capability = _capability(run_root)
        return (
            SegmentedFetchCheckpoint.open(
                capability, binding if binding is not None else _binding(ids)
            ),
            capability,
        )

    def test_complete_checkpoint_reuses_every_segment_and_reconstructs_order(self):
        ids = tuple(f"ANT-{index}" for index in range(5))
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)
            calls = []

            def fetch(requested):
                calls.append(requested)
                return _results_for(requested)

            completed = checkpoint.fetch_missing(ids, fetch)
            self.assertEqual(
                calls,
                [("ANT-0", "ANT-1"), ("ANT-2", "ANT-3"), ("ANT-4",)],
            )
            self.assertEqual(completed.requested_objects, 5)
            self.assertEqual(completed.completed_objects, 5)
            self.assertEqual(completed.segment_count, 3)
            self.assertEqual(completed.alert_rows, 8)
            self.assertEqual(completed.retry_count, 1)
            self.assertEqual(completed.fetched_segments, 3)
            self.assertTrue(checkpoint.complete_path.is_file())

            reopened = SegmentedFetchCheckpoint.open(capability, _binding(ids))

            def forbidden(_requested):
                raise AssertionError("network callback must not be invoked")

            reused = reopened.fetch_missing(ids, forbidden)
            self.assertEqual(reused.reused_segments, 3)
            self.assertEqual(reused.fetched_segments, 0)
            objects = list(reopened.iter_objects(ids))
            self.assertEqual([item.locus_id for item in objects], list(ids))
            self.assertIsNone(objects[2].alerts)
            self.assertEqual(objects[3].retry_count, 1)
            self.assertEqual(
                objects[3].retry_exception_types,
                ("requests.exceptions.JSONDecodeError",),
            )
            alerts = reopened.reconstruct_alerts(ids)
            self.assertEqual(
                alerts["locus_id"].astype(str).drop_duplicates().tolist(),
                ["ANT-0", "ANT-1", "ANT-3", "ANT-4"],
            )

    def test_checkpoint_content_is_deterministic_across_fresh_roots(self):
        ids = ("ANT-0", "ANT-1", "ANT-2")
        snapshots = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                checkpoint, _cap = self._new(temporary, ids)
                checkpoint.fetch_missing(ids, _results_for)
                segment_payloads = {
                    path.name: path.read_bytes()
                    for path in sorted(checkpoint.segments.iterdir())
                }
                snapshots.append(
                    (
                        checkpoint.header_path.read_bytes(),
                        checkpoint.complete_path.read_bytes(),
                        segment_payloads,
                        sorted(path.name for path in checkpoint.blobs.iterdir()),
                    )
                )
        self.assertEqual(snapshots[0], snapshots[1])

    def test_process_death_after_blob_leaves_no_committed_segment(self):
        ids = ("ANT-0", "ANT-1", "ANT-2")

        class ProcessDeath(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)

            def die(event, details):
                if event == "after_segment_blob_committed" and details["segment_index"] == 0:
                    raise ProcessDeath("simulated abrupt process death")

            with self.assertRaises(ProcessDeath):
                checkpoint.fetch_missing(ids, _results_for, event_hook=die)
            self.assertEqual(list(checkpoint.segments.iterdir()), [])
            self.assertFalse(checkpoint.complete_path.exists())
            orphan_blobs = list(checkpoint.blobs.iterdir())
            self.assertEqual(len(orphan_blobs), 1)

            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []

            def fetch(requested):
                calls.append(requested)
                return _results_for(requested)

            completed = restarted.fetch_missing(ids, fetch)
            self.assertEqual(calls, [("ANT-0", "ANT-1"), ("ANT-2",)])
            self.assertEqual(completed.completed_objects, 3)
            self.assertTrue(orphan_blobs[0].exists())

    def test_restart_reuses_segment_committed_immediately_before_death(self):
        ids = ("ANT-0", "ANT-1", "ANT-2", "ANT-3", "ANT-4")

        class ProcessDeath(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)

            def die(event, details):
                if event == "after_segment_commit" and details["segment_index"] == 0:
                    raise ProcessDeath("simulated death after durable commit")

            with self.assertRaises(ProcessDeath):
                checkpoint.fetch_missing(ids, _results_for, event_hook=die)
            self.assertEqual(len(list(checkpoint.segments.iterdir())), 1)
            self.assertFalse(checkpoint.complete_path.exists())

            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []

            def fetch(requested):
                calls.append(requested)
                return _results_for(requested)

            completed = restarted.fetch_missing(ids, fetch)
            self.assertEqual(calls, [("ANT-2", "ANT-3"), ("ANT-4",)])
            self.assertEqual(completed.reused_segments, 1)
            self.assertEqual(completed.fetched_segments, 2)

    def test_all_segments_before_fetch_complete_restart_without_refetch(self):
        ids = ("ANT-0", "ANT-1", "ANT-2", "ANT-3")

        class StopBeforeCompletion(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)

            def stop(event, _details):
                if event == "before_fetch_complete_commit":
                    raise StopBeforeCompletion("simulated completion-marker interruption")

            with self.assertRaises(StopBeforeCompletion):
                checkpoint.fetch_missing(ids, _results_for, event_hook=stop)
            self.assertEqual(len(list(checkpoint.segments.iterdir())), 2)
            self.assertFalse(checkpoint.complete_path.exists())

            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []
            completed = restarted.fetch_missing(
                ids,
                lambda requested: calls.append(requested) or _results_for(requested),
            )
            self.assertEqual(calls, [])
            self.assertEqual(completed.reused_segments, 2)
            self.assertEqual(completed.fetched_segments, 0)
            self.assertTrue(restarted.complete_path.is_file())

    def test_actual_process_death_preserves_committed_segment_for_restart(self):
        ids = ("ANT-0", "ANT-1", "ANT-2", "ANT-3")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)
            script = r'''
import hashlib
import json
import os
import sys
from pathlib import Path
import pandas as pd
from src.operations.fetch_checkpoint import FetchCheckpointBinding, FetchObjectResult, SegmentedFetchCheckpoint
from src.operations.live_antares import LIVE_ANTARES_READ, LiveAntaresReadCapability

def digest(value):
    if isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def ids_digest(values):
    result = hashlib.sha256()
    for value in values:
        result.update(value.encode("utf-8"))
        result.update(b"\n")
    return result.hexdigest()

root = Path(sys.argv[1])
ids = ("ANT-0", "ANT-1", "ANT-2", "ANT-3")
binding = FetchCheckpointBinding(
    run_id="run", release_sha="1" * 40,
    configuration_sha256=digest(b"configuration"),
    target_date_utc="2026-06-27", mjd_min=61218.0, mjd_max=61219.0,
    provider_name="live-antares", provider_scenario="commissioning-v1",
    provider_policy_sha256=digest({"workers": 4, "attempts": 3}),
    query_contract_sha256=digest({"mjd": [61218.0, 61219.0]}),
    query_identity_sha256=digest({"result": list(ids)}),
    query_locus_order_sha256=ids_digest(ids), expected_objects=4, segment_size=2,
)
capability = LiveAntaresReadCapability.for_local_mock(
    root, run_id="run", target_date_utc="2026-06-27",
    release_sha="1" * 40, authority=LIVE_ANTARES_READ,
)
checkpoint = SegmentedFetchCheckpoint.open(capability, binding)

def fetch(requested):
    return [
        FetchObjectResult(
            locus_id,
            pd.DataFrame({"mjd": [61218.1], "flux": [1.0], "range_label": ["commissioning"]}),
        )
        for locus_id in requested
    ]

def die(event, details):
    if event == "after_segment_commit" and details["segment_index"] == 0:
        os._exit(73)

checkpoint.fetch_missing(ids, fetch, event_hook=die)
'''
            completed = subprocess.run(
                [sys.executable, "-c", script, str(checkpoint.run_root)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                },
            )
            self.assertEqual(completed.returncode, 73, completed.stderr)
            self.assertEqual(len(list(checkpoint.segments.iterdir())), 1)
            self.assertFalse(checkpoint.complete_path.exists())

            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []
            recovered = restarted.fetch_missing(
                ids,
                lambda requested: calls.append(requested) or _results_for(requested),
            )
            self.assertEqual(calls, [("ANT-2", "ANT-3")])
            self.assertEqual(recovered.reused_segments, 1)
            self.assertEqual(recovered.fetched_segments, 1)

    def test_partial_callback_result_is_not_a_committed_segment(self):
        ids = ("ANT-0", "ANT-1", "ANT-2")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)

            def partial(requested):
                return _results_for(requested[:-1])

            with self.assertRaises(FetchCheckpointFetchError):
                checkpoint.fetch_missing(ids, partial)
            self.assertEqual(list(checkpoint.segments.iterdir()), [])
            self.assertFalse(checkpoint.complete_path.exists())

            # A process may leave private temporary bytes. They are deliberately
            # ignored rather than interpreted as completed science or deleted.
            dead_temp = checkpoint.tmp / ".tmp-interrupted-process"
            dead_temp.write_bytes(b"partial")
            dead_temp.chmod(0o400)
            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            completed = restarted.fetch_missing(ids, _results_for)
            self.assertEqual(completed.completed_objects, len(ids))
            self.assertTrue(dead_temp.exists())

    def test_callback_exception_payload_is_not_persisted(self):
        ids = ("ANT-0", "ANT-1")
        secret = b"credential-sentinel-must-never-enter-checkpoint"
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _capability_value = self._new(temporary, ids)

            def fail(_requested):
                raise RuntimeError(secret.decode("ascii"))

            with self.assertRaisesRegex(
                FetchCheckpointFetchError, "RuntimeError"
            ) as raised:
                checkpoint.fetch_missing(ids, fail)
            self.assertNotIn(secret.decode("ascii"), str(raised.exception))
            for path in checkpoint.root.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    self.assertNotIn(secret, path.read_bytes(), path)

    def test_corrupt_blob_fails_before_fetch_and_is_not_deleted(self):
        ids = ("ANT-0", "ANT-1")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)
            checkpoint.fetch_missing(ids, _results_for)
            blob = next(checkpoint.blobs.iterdir())
            blob.chmod(0o600)
            with blob.open("ab") as stream:
                stream.write(b"tamper")
            blob.chmod(0o400)
            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []

            def fetch(requested):
                calls.append(requested)
                return _results_for(requested)

            with self.assertRaises(FetchCheckpointCorrupt):
                restarted.fetch_missing(ids, fetch)
            self.assertEqual(calls, [])
            self.assertTrue(blob.exists())

    def test_corrupt_receipt_fails_before_fetch_and_is_not_replaced(self):
        ids = ("ANT-0", "ANT-1")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)
            checkpoint.fetch_missing(ids, _results_for)
            receipt = next(checkpoint.segments.iterdir())
            document = json.loads(receipt.read_text(encoding="utf-8"))
            document["completed_objects"] = 1
            receipt.chmod(0o600)
            receipt.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            receipt.chmod(0o400)
            before = receipt.read_bytes()
            restarted = SegmentedFetchCheckpoint.open(capability, _binding(ids))
            calls = []
            with self.assertRaises(FetchCheckpointCorrupt):
                restarted.fetch_missing(
                    ids,
                    lambda requested: calls.append(requested) or _results_for(requested),
                )
            self.assertEqual(calls, [])
            self.assertEqual(receipt.read_bytes(), before)

    def test_symlink_and_unexpected_segment_are_ambiguous_before_fetch(self):
        ids = ("ANT-0",)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _capability_value = self._new(temporary, ids)
            target = checkpoint.tmp / ".tmp-target"
            target.write_bytes(b"x")
            link = checkpoint.segments / "segment-00000000-000000000000-000000000001.commit.json"
            link.symlink_to(target)
            calls = []
            with self.assertRaises(FetchCheckpointAmbiguous):
                checkpoint.fetch_missing(
                    ids,
                    lambda requested: calls.append(requested) or _results_for(requested),
                )
            self.assertEqual(calls, [])
            self.assertTrue(link.is_symlink())

    def test_lexical_checkpoint_symlink_is_refused(self):
        ids = ("ANT-0",)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run_root = base / "run"
            run_root.mkdir()
            elsewhere = run_root / "elsewhere"
            elsewhere.mkdir()
            (run_root / "checkpoints").symlink_to(elsewhere, target_is_directory=True)
            with self.assertRaises(FetchCheckpointCorrupt):
                SegmentedFetchCheckpoint.open(_capability(run_root), _binding(ids))

    def test_header_binding_mismatch_is_refused_without_mutation(self):
        ids = ("ANT-0", "ANT-1")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, capability = self._new(temporary, ids)
            header = checkpoint.header_path.read_bytes()
            with self.assertRaises(FetchCheckpointBindingError):
                SegmentedFetchCheckpoint.open(
                    capability, _binding(ids, config="different-configuration")
                )
            self.assertEqual(checkpoint.header_path.read_bytes(), header)
            self.assertEqual(list(checkpoint.segments.iterdir()), [])

    def test_header_removed_after_open_fails_before_fetch(self):
        ids = ("ANT-0",)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _capability_value = self._new(temporary, ids)
            checkpoint.header_path.unlink()
            calls = []
            with self.assertRaises(FetchCheckpointCorrupt):
                checkpoint.fetch_missing(
                    ids,
                    lambda requested: calls.append(requested) or _results_for(requested),
                )
            self.assertEqual(calls, [])
            self.assertFalse(checkpoint.header_path.exists())

    def test_zero_object_query_commits_exact_empty_completion_without_fetch(self):
        ids = ()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, _capability_value = self._new(temporary, ids)
            calls = []
            completed = checkpoint.fetch_missing(
                ids,
                lambda requested: calls.append(requested) or (),
            )
            self.assertEqual(calls, [])
            self.assertEqual(completed.completed_objects, 0)
            self.assertEqual(completed.segment_count, 0)
            self.assertTrue(checkpoint.complete_path.is_file())
            self.assertTrue(checkpoint.reconstruct_alerts(ids).empty)


if __name__ == "__main__":
    unittest.main()
