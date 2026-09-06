import copy
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.operations import offline_recovery as recovery
from src.operations.fetch_checkpoint import (
    FetchCheckpointBindingError, FetchCheckpointError, SegmentedFetchCheckpoint,
)
from test_fetch_checkpoint import _binding, _capability, _results_for


class ReadOnlyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "run"
        self.root.mkdir(mode=0o700)
        self.ids = ("ANT-1", "ANT-2", "ANT-3")
        self.binding = _binding(self.ids)
        self.writer = SegmentedFetchCheckpoint.open(_capability(self.root), self.binding)
        self.writer.fetch_missing(self.ids, _results_for)

    def reader(self):
        return SegmentedFetchCheckpoint.open_read_only(self.root, self.binding)

    def mutate(self, path, payload):
        path.chmod(0o600)
        path.write_bytes(payload)

    def test_complete_read_only_reopen_preserves_every_source_byte_and_metadata(self):
        before = recovery.source_identity(self.root)
        with mock.patch("src.operations.fetch_checkpoint._ensure_private_directory", side_effect=AssertionError("write")), mock.patch("src.operations.fetch_checkpoint._commit_json_noreplace", side_effect=AssertionError("commit")):
            reader = self.reader()
            completion = reader.inspect_complete(self.ids)
            self.assertEqual(completion.reused_segments, 2)
            self.assertEqual(completion.fetched_segments, 0)
            self.assertEqual(len(list(reader.iter_objects(self.ids))), 3)
            self.assertEqual(len(reader.reconstruct_alerts(self.ids)), 4)
        self.assertEqual(before, recovery.source_identity(self.root))

    def test_callback_attempt_fails_before_callback_or_source_mutation(self):
        before = recovery.source_identity(self.root)
        callback = mock.Mock(side_effect=AssertionError("callback"))
        with self.assertRaises(FetchCheckpointBindingError):
            self.reader().fetch_missing(self.ids, callback)
        callback.assert_not_called()
        self.assertEqual(before, recovery.source_identity(self.root))

    def test_release_mismatch(self):
        binding = copy.copy(self.binding)
        object.__setattr__(binding, "release_sha", "2" * 40)
        with self.assertRaises(FetchCheckpointBindingError):
            SegmentedFetchCheckpoint.open_read_only(self.root, binding)

    def test_missing_segment(self):
        next(self.writer.segments.iterdir()).unlink()
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_extra_segment(self):
        (self.writer.segments / "segment-99999999-000000000000-000000000001.commit.json").write_text("{}")
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_corrupt_segment(self):
        self.mutate(next(self.writer.segments.iterdir()), b"corrupt")
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_corrupt_blob(self):
        self.mutate(next(self.writer.blobs.iterdir()), b"corrupt")
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_extra_valid_blob(self):
        payload = b"unreferenced"
        (self.writer.blobs / (hashlib.sha256(payload).hexdigest() + ".parquet")).write_bytes(payload)
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_missing_completion_never_recreated(self):
        self.writer.complete_path.unlink()
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)
        self.assertFalse(self.writer.complete_path.exists())

    def test_schema_mismatch(self):
        header = json.loads(self.writer.header_path.read_text())
        header["schema_version"] = "future.v2"
        self.mutate(self.writer.header_path, recovery._json(header))
        with self.assertRaises(FetchCheckpointError):
            self.reader()

    def test_parquet_schema_mismatch(self):
        path = next(self.writer.segments.iterdir())
        receipt = json.loads(path.read_text())
        receipt["artifact"]["parquet_schema_sha256"] = "0" * 64
        self.mutate(path, recovery._json(receipt))
        with self.assertRaises(FetchCheckpointError):
            self.reader().inspect_complete(self.ids)

    def test_temporary_residue(self):
        (self.writer.tmp / ".tmp-abandoned").write_bytes(b"preserve")
        with self.assertRaises(FetchCheckpointError):
            self.reader()

    def test_symlink_escape(self):
        original = self.writer.segments
        moved = original.with_name("elsewhere")
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(FetchCheckpointError):
            self.reader()

    def test_source_mutation_detected(self):
        before = recovery.source_identity(self.root)
        self.mutate(next(self.writer.blobs.iterdir()), b"changed")
        self.assertNotEqual(before["sha256"], recovery.source_identity(self.root)["sha256"])


class RecoveryContractTests(unittest.TestCase):
    def source_metadata(self):
        from src.operations.live_antares import _scientific_query_contract, extraction_method_contract
        from src.operations.science import NightScienceRequest
        request = NightScienceRequest(recovery.NIGHT, *recovery.MJD, target_loci=None)
        policy = {
            "scientific_contract": _scientific_query_contract(request),
            "execution_policy": {
                "api_timeout_seconds": 60,
                "extraction_method": extraction_method_contract(),
                "lightcurve_cache": False,
                "max_fetch_attempts_per_object": 3,
                "max_fetch_workers": 4,
                "max_query_attempts": 2,
                "parallel_parent_shards": 1,
                "probe_limit": 50,
                "probe_threshold": 50,
                "retry_delay_seconds": 0.5,
                "tile_cache": False,
            },
        }
        query = {
            "schema_version": "phase6.query-result-checkpoint.v1",
            "content_integrity_sha256": recovery.QUERY_ID,
            "bindings": {"run_id": recovery.SOURCE_RUN_ID, "release_sha": recovery.SOURCE_SHA, "configuration_hash": recovery.CONFIGURATION, "target_date_utc": recovery.NIGHT, "provider_name": "live-antares", "provider_scenario": "commissioning-v1", "query_policy_sha256": recovery.QUERY_POLICY, "query_policy": policy},
            "scientific_request": {"date_utc": recovery.NIGHT, "mjd_min": recovery.MJD[0], "mjd_max": recovery.MJD[1], "lsst_only": True, "query_tag": None, "target_loci": None},
        }
        fetch = {"schema_version": "phase6.segmented-fetch-checkpoint.v1", "checkpoint_identity_sha256": recovery.FETCH_ID, "binding": {"run_id": recovery.SOURCE_RUN_ID, "release_sha": recovery.SOURCE_SHA, "configuration_sha256": recovery.CONFIGURATION, "target_date_utc": recovery.NIGHT, "mjd_min": recovery.MJD[0], "mjd_max": recovery.MJD[1], "provider_name": "live-antares", "provider_scenario": "commissioning-v1", "provider_policy_sha256": recovery.FETCH_POLICY, "query_contract_sha256": recovery.QUERY_CONTRACT, "query_identity_sha256": recovery.QUERY_ID, "query_locus_order_sha256": recovery.QUERY_ORDER, "expected_objects": recovery.OBJECTS, "segment_size": 256}}
        return query, fetch

    def test_exact_source_metadata_and_failure_matrix(self):
        query, fetch = self.source_metadata()
        recovery.validate_source_metadata(query, fetch)
        changes = (
            ("query", ("schema_version",), "future.v2"),
            ("query", ("content_integrity_sha256",), "0" * 64),
            ("query", ("bindings", "run_id"), "another-run"),
            ("query", ("bindings", "release_sha"), "2" * 40),
            ("query", ("bindings", "configuration_hash"), "0" * 64),
            ("query", ("bindings", "query_policy", "execution_policy", "max_fetch_workers"), 2),
            ("query", ("scientific_request", "date_utc"), "2026-06-28"),
            ("query", ("scientific_request", "mjd_min"), 61217.0),
            ("query", ("scientific_request", "mjd_max"), 61220.0),
            ("query", ("scientific_request", "lsst_only"), False),
            ("fetch", ("schema_version",), "future.v2"),
            ("fetch", ("binding", "provider_policy_sha256"), "0" * 64),
            ("fetch", ("binding", "query_identity_sha256"), "0" * 64),
            ("fetch", ("binding", "expected_objects"), 331785),
            ("fetch", ("binding", "segment_size"), 512),
        )
        for kind, path, value in changes:
            pair = {"query": copy.deepcopy(query), "fetch": copy.deepcopy(fetch)}
            target = pair[kind]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(kind=kind, path=path), self.assertRaises(recovery.OfflineRecoveryError):
                recovery.validate_source_metadata(pair["query"], pair["fetch"])

    def test_only_approved_release_direction(self):
        recovery.validate_release_pair("0.4.1", recovery.SOURCE_SHA, "0.4.2", "2" * 40)
        combinations = (
            ("0.4.0", recovery.SOURCE_SHA, "0.4.2", "2" * 40),
            ("0.4.1", "1" * 40, "0.4.2", "2" * 40),
            ("0.4.2", "2" * 40, "0.4.1", recovery.SOURCE_SHA),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.3", "2" * 40),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.2", recovery.SOURCE_SHA),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.2", "../other"),
        )
        for pair in combinations:
            with self.subTest(pair=pair), self.assertRaises(recovery.OfflineRecoveryError):
                recovery.validate_release_pair(*pair)

    def test_existing_destination_is_refused_before_production_or_source_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            run_id = "phase6f-recovery-0.4.2-test"
            root = parent / run_id
            root.mkdir(mode=0o700)
            with mock.patch.object(recovery, "CANARY_ROOT", parent), mock.patch.object(recovery, "SOURCE_ROOT", root), mock.patch.object(recovery, "release_environment", return_value={}), mock.patch.object(recovery.os, "uname") as uname, mock.patch.object(recovery, "production_snapshot") as production:
                uname.return_value.nodename = "arnor"
                with self.assertRaisesRegex(recovery.OfflineRecoveryError, "already exists"):
                    recovery.prepare(run_id, "2" * 40)
                production.assert_not_called()

    def test_guard_rejects_network_client_callback_path_and_outside_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guard = recovery.OfflineGuard(root)
            for event, args in (
                ("socket.__new__", ()),
                ("socket.connect", ()),
                ("import", ("antares_client",)),
                ("open", (str(root.parent / "outside"), "w", os.O_WRONLY)),
                ("os.mkdir", (str(root.parent / "outside"), 0o700, -1)),
                ("subprocess.Popen", ("python", ["python"], None, None)),
            ):
                with self.subTest(event=event), self.assertRaises(recovery.OfflineRecoveryError):
                    guard.audit(event, args)
            guard.audit("open", (str(root / "allowed"), "w", os.O_WRONLY))
            (root / "link").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(recovery.OfflineRecoveryError):
                guard.audit("open", (str(root / "link/escape"), "w", os.O_WRONLY))

    def test_read_only_audit_guard_forbids_even_destination_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            guard = recovery.OfflineGuard(root, writable=False)
            with self.assertRaises(recovery.OfflineRecoveryError):
                guard.audit("open", (str(root / "file"), "w", os.O_WRONLY))

    def test_installed_guard_blocks_real_operations_and_allows_only_one_audit_child(self):
        script = r'''
import json, socket, subprocess, sys
from pathlib import Path
from src.operations.offline_recovery import OfflineGuard, OfflineRecoveryError
from src.operations.live_antares import LiveAntaresProvider
root = Path(sys.argv[1])
guard = OfflineGuard(root)
guard.install()
for operation in (
    lambda: socket.socket(),
    lambda: LiveAntaresProvider(None),
    lambda: LiveAntaresProvider.query(None, None),
    lambda: LiveAntaresProvider.fetch(None, None, None),
    lambda: (root.parent / "outside-guard-test").write_bytes(b"refuse"),
):
    try:
        operation()
    except OfflineRecoveryError:
        pass
    else:
        raise AssertionError("forbidden operation succeeded")
(root / "allowed").write_bytes(b"allowed")
command = [sys.executable, "-c", "print('child-ok')"]
guard.allowed_subprocess = command
with open('/dev/null', 'rb') as null_input:
    child = subprocess.run(command, stdin=null_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
assert child.stdout == b'child-ok\n'
try:
    subprocess.run(command, stdout=subprocess.PIPE)
except OfflineRecoveryError:
    pass
else:
    raise AssertionError("second child permitted")
print(json.dumps(guard.counts))
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            result = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True, check=True)
            counts = json.loads(result.stdout)
            self.assertEqual(counts["network_attempts"], 1)
            self.assertEqual(counts["provider_initializations"], 1)
            self.assertEqual(counts["query_callbacks"], 1)
            self.assertEqual(counts["fetch_callbacks"], 1)
            self.assertEqual(counts["outside_write_attempts"], 1)
            self.assertEqual((root / "allowed").read_bytes(), b"allowed")


class RecoveryIntegrationTests(unittest.TestCase):
    def test_reconstruction_and_terminal_evidence_without_any_live_entry(self):
        import pandas as pd
        from src import history
        from src.operations import science
        from src.operations.fetch_checkpoint import FetchCheckpointBinding
        from src.operations.query_checkpoint import (
            QueryResultCheckpointBindings, load_query_result_checkpoint,
            seal_query_result_checkpoint,
        )
        from src.operations.live_antares import LiveAntaresProvider
        from test_operations_phase6 import _mixed_identifier_provider
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            parent = Path(temporary).resolve()
            source = parent / recovery.SOURCE_RUN_ID
            source.mkdir(mode=0o700)
            # Construct the sealed input using the existing mock acquisition path.
            # Recovery starts only after these source checkpoints are complete.
            stack.enter_context(mock.patch("test_operations_phase6.RELEASE_SHA", recovery.SOURCE_SHA))
            provider = _mixed_identifier_provider(source)
            provider.max_fetch_attempts = 3
            provider.retry_delay_seconds = 0.5
            request = science.NightScienceRequest(recovery.NIGHT, *recovery.MJD, target_loci=None, range_label="ANTARES commissioning 2026-06-27")
            queried = provider.query(request)
            policy = {"scientific_contract": provider.scientific_contract(request), "execution_policy": provider.execution_policy()}
            bindings = QueryResultCheckpointBindings(source.name, recovery.SOURCE_SHA, recovery.CONFIGURATION, recovery.NIGHT, provider.provider_name, provider.scenario, policy)
            seal_query_result_checkpoint(source, queried, bindings)
            loaded = load_query_result_checkpoint(source, request, bindings)
            fetch_binding = FetchCheckpointBinding(source.name, recovery.SOURCE_SHA, recovery.CONFIGURATION, recovery.NIGHT, *recovery.MJD, provider.provider_name, provider.scenario, recovery.FETCH_POLICY, recovery.QUERY_CONTRACT, loaded.integrity_sha256, queried.evidence.details["locus_order_sha256"], 3)
            writer = SegmentedFetchCheckpoint.open(provider.capability, fetch_binding)
            original = provider.fetch_resumable(request, queried, writer)
            self.assertTrue(original.publishable)
            source_before = recovery.source_identity(source)
            root = parent / "phase6f-recovery-0.4.2-integration"
            root.mkdir(mode=0o700)
            for name in ("logs", "status", "evidence", "candidate", "tmp"):
                (root / name).mkdir(mode=0o700)
            values = {"CANARY_ROOT": parent, "SOURCE_ROOT": source, "OBJECTS": 3, "ALERTS": 3, "SEGMENTS": 1, "QUERY_ID": loaded.integrity_sha256, "QUERY_ORDER": fetch_binding.query_locus_order_sha256, "FETCH_ID": fetch_binding.identity_sha256}
            for name, value in values.items():
                stack.enter_context(mock.patch.object(recovery, name, value))
            binding = {"schema_version": recovery.CONTRACT, "run_id": root.name, "run_root": str(root), "source_root": str(source), "source_version": recovery.SOURCE_VERSION, "source_sha": recovery.SOURCE_SHA, "consumer_version": recovery.CONSUMER_VERSION, "consumer_sha": "2" * 40, "night": recovery.NIGHT, "mjd": list(recovery.MJD), "query_identity": recovery.QUERY_ID, "fetch_identity": recovery.FETCH_ID, "authoritative": False, "publishable": False, "publication_authorized": False, "timeout_seconds": recovery.TIMEOUT_SECONDS, "source_identity": source_before["sha256"], "production_sentinel": recovery.PRODUCTION_SENTINEL}
            recovery._write_new(root / "binding.json", recovery._json(binding))
            recovery._write_new(root / "binding.sha256", (recovery._hash(root / "binding.json") + "\n").encode())
            stack.enter_context(mock.patch.object(recovery, "release_environment", return_value={"test_environment": True}))
            stack.enter_context(mock.patch.object(recovery, "process_identity", return_value={"pid": os.getpid()}))
            stack.enter_context(mock.patch.object(recovery, "production_snapshot", return_value={"sentinel": {"fingerprint_sha256": recovery.PRODUCTION_SENTINEL}}))
            stack.enter_context(mock.patch.object(history, "load_cumulative_loci_index", return_value=pd.DataFrame({"locus_id": []})))
            # The real process-lifetime guard is exercised in a separate-process
            # test above; do not install an irreversible hook in unittest itself.
            stack.enter_context(mock.patch.object(recovery.OfflineGuard, "install"))
            for name in ("__init__", "query", "fetch", "fetch_resumable", "_load_client"):
                stack.enter_context(mock.patch.object(LiveAntaresProvider, name, side_effect=AssertionError("live entry")))
            def independent_reopen(command, **kwargs):
                self.assertEqual(command[1:3], ["-m", "src.operations.offline_recovery"])
                # Re-read and validate only persisted bytes, without provider frames.
                result = recovery.audit(root, "2" * 40)
                return subprocess.CompletedProcess(command, 0, recovery._json(result), b"")
            stack.enter_context(mock.patch.object(recovery.subprocess, "run", side_effect=independent_reopen))
            final = recovery.reconstruct(root, "2" * 40)
            self.assertTrue(final["success"], final)
            self.assertEqual(final["status"], "RECOVERY_COMPLETE_UNPUBLISHED")
            self.assertEqual(final["fetch_checkpoint"]["reused_segments"], 1)
            self.assertEqual(final["fetch_checkpoint"]["fetched_segments"], 0)
            self.assertFalse(any(final["callback_and_network_counts"].values()))
            self.assertEqual(final["source_before_sha256"], final["source_after_sha256"])
            self.assertEqual(source_before, recovery.source_identity(source))
            self.assertFalse(final["publication_attempted"])
            self.assertFalse(final["publishable"])
            reopened = science.reopen_and_validate_artifacts({name: (root / "candidate" / name).read_bytes() for name in final["artifacts"]}, expected=original)
            self.assertEqual(len(reopened.loci), 3)
            self.assertEqual(recovery._read(root / "status/RECOVERY_FINAL.json"), final)
            with self.assertRaises((recovery.OfflineRecoveryError, FileExistsError)):
                recovery.reconstruct(root, "2" * 40)


if __name__ == "__main__":
    unittest.main()
