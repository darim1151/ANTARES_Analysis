import copy
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from src.operations import offline_recovery as recovery
from src.operations.fetch_checkpoint import (
    FetchCheckpointBindingError, FetchCheckpointError, SegmentedFetchCheckpoint,
)
from test_fetch_checkpoint import _binding, _capability, _results_for


def _production_sentinel(device=42, *, source="fixture:/production"):
    durable_state = {
        "canonical_data_root": "/fixture",
        "directory_metadata": {
            "data_root": {
                "path": "/fixture",
                "inode": 1,
                "mode": "0700",
                "mtime_ns": 1,
            }
        },
        "manifest_inventory": [],
        "durable_file_inventory": [],
        "durable_file_count": 0,
        "durable_bytes": 0,
        "manifest_count": 0,
        "checksum_manifest_sha256": hashlib.sha256(b"").hexdigest(),
        "cumulative_artifact_hashes": {
            "loci_index": hashlib.sha256(b"loci").hexdigest(),
            "nightly_summary": hashlib.sha256(b"summary").hexdigest(),
        },
    }
    return {
        "schema_version": recovery.PRODUCTION_SENTINEL_SCHEMA,
        "captured_at_utc": "2026-09-13T00:00:00+00:00",
        "durable_state": durable_state,
        "durable_fingerprint_sha256": recovery._mapping_digest(durable_state),
        "mount_binding": {
            "mount_point": "/fixture",
            "filesystem_type": "nfs4",
            "source": source,
        },
        "qualification_predicates": {
            "target_path": "/fixture/target",
            "target_absent": True,
            "cache_path": "/fixture/cache",
            "cache_absent": True,
            "transaction_artifacts": [],
        },
        "runtime_observation": {
            "directory_devices": {
                "data_root": {"path": "/fixture", "device": device}
            },
            "manifest_devices": [],
            "durable_file_devices": [],
        },
    }


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
        self.assertNotEqual(
            before["durable_sha256"],
            recovery.source_identity(self.root)["durable_sha256"],
        )

    def test_source_identity_separates_persistent_and_runtime_device_identity(self):
        before = recovery.source_identity(self.root)
        later_session = copy.deepcopy(before)
        for item in later_session["runtime_observation"]["devices"]:
            item["device"] += 1
        self.assertEqual(before["durable_state"], later_session["durable_state"])
        self.assertEqual(
            before["durable_sha256"], later_session["durable_sha256"]
        )
        self.assertTrue(
            recovery.qualify_persisted_source_identity(
                later_session,
                expected_schema=before["schema_version"],
                expected_durable_sha256=before["durable_sha256"],
            )["passed"]
        )
        comparison = recovery.compare_source_identities(before, later_session)
        self.assertFalse(comparison["passed"])
        self.assertTrue(comparison["durable_state_equal"])
        self.assertFalse(comparison["runtime_device_projection_equal"])
        missing_runtime = copy.deepcopy(before)
        missing_runtime.pop("runtime_observation")
        self.assertFalse(
            recovery.compare_source_identities(before, missing_runtime)["passed"]
        )


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
        recovery.validate_release_pair("0.4.1", recovery.SOURCE_SHA, "0.4.3", "2" * 40)
        combinations = (
            ("0.4.0", recovery.SOURCE_SHA, "0.4.3", "2" * 40),
            ("0.4.1", "1" * 40, "0.4.3", "2" * 40),
            ("0.4.3", "2" * 40, "0.4.1", recovery.SOURCE_SHA),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.2", "2" * 40),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.3", recovery.SOURCE_SHA),
            ("0.4.1", recovery.SOURCE_SHA, "0.4.3", "../other"),
        )
        for pair in combinations:
            with self.subTest(pair=pair), self.assertRaises(recovery.OfflineRecoveryError):
                recovery.validate_release_pair(*pair)

    def test_production_v2_unconfigured_override_fails_before_capture(self):
        with mock.patch.object(
            recovery, "PRODUCTION_DURABLE_FINGERPRINT", None
        ), mock.patch.object(
            recovery, "PRODUCTION_MOUNT_BINDING", None
        ), mock.patch(
            "src.operations.commissioning.capture_production_sentinel"
        ) as capture, self.assertRaisesRegex(
            recovery.OfflineRecoveryError, "pins are unconfigured"
        ):
            recovery.production_snapshot()
        capture.assert_not_called()

    def test_production_snapshot_uses_configured_v2_pins_end_to_end(self):
        self.assertEqual(
            recovery.PRODUCTION_DURABLE_FINGERPRINT,
            "52d9d30f0e004622485ba819af3bb56c81b704c22e7618b086a4ac398a76bf63",
        )
        self.assertEqual(
            recovery.PRODUCTION_MOUNT_BINDING,
            {
                "mount_point": "/astro/store/shire",
                "filesystem_type": "nfs4",
                "source": "shire.infiniband:/data/shire",
            },
        )
        matching = _production_sentinel(device=4242)
        matching["durable_state"]["durable_file_count"] = 324
        matching["durable_state"]["durable_bytes"] = 1141241743
        matching["durable_fingerprint_sha256"] = (
            recovery.PRODUCTION_DURABLE_FINGERPRINT
        )
        matching["mount_binding"] = dict(recovery.PRODUCTION_MOUNT_BINDING)
        fingerprint_mismatch = copy.deepcopy(matching)
        fingerprint_mismatch["durable_fingerprint_sha256"] = "0" * 64
        mount_mismatch = copy.deepcopy(matching)
        mount_mismatch["mount_binding"]["source"] = "other:/data/shire"
        eligibility = {
            "passed": True,
            "authoritative_manifest_count": 90,
            "total_loci": 993218,
            "total_alerts": 13579707,
        }
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            parent = Path(temporary).resolve()
            data_root = parent / "synthetic-production"
            cache_root = parent / "synthetic-cache"
            cumulative = {
                "loci_index": data_root / "loci-index.parquet",
                "nightly_summary": data_root / "nightly-summary.parquet",
            }
            mountinfo = ["synthetic mountinfo observation"]
            observations = iter(
                (matching, fingerprint_mismatch, mount_mismatch)
            )
            stack.enter_context(mock.patch.object(recovery, "DATA_ROOT", data_root))
            stack.enter_context(mock.patch.object(recovery, "CACHE_ROOT", cache_root))
            capture = stack.enter_context(
                mock.patch(
                    "src.operations.commissioning.capture_production_sentinel",
                    side_effect=lambda *args, **kwargs: copy.deepcopy(
                        next(observations)
                    ),
                )
            )
            stack.enter_context(
                mock.patch(
                    "src.operations.commissioning.establish_target_eligibility",
                    return_value=eligibility,
                )
            )
            stack.enter_context(
                mock.patch("src.history.cumulative_paths", return_value=cumulative)
            )
            # The 324-file production inventory is not copied into the test;
            # this seam supplies its separately authorized digest.
            mapping_digest = stack.enter_context(
                mock.patch.object(
                    recovery,
                    "_mapping_digest",
                    return_value=recovery.PRODUCTION_DURABLE_FINGERPRINT,
                )
            )
            stack.enter_context(
                mock.patch.object(
                    recovery,
                    "_hash",
                    side_effect=[recovery.PRIOR_HASH, recovery.SUMMARY_HASH],
                )
            )
            write_new = stack.enter_context(mock.patch.object(recovery, "_write_new"))

            snapshot = recovery.production_snapshot(mountinfo_lines=mountinfo)
            self.assertTrue(snapshot["qualification"]["passed"])
            self.assertEqual(
                snapshot["qualification"]["durable_fingerprint_sha256"],
                recovery.PRODUCTION_DURABLE_FINGERPRINT,
            )
            self.assertEqual(
                snapshot["qualification"]["mount_binding"],
                recovery.PRODUCTION_MOUNT_BINDING,
            )
            persistent = {
                key: snapshot["sentinel"][key]
                for key in (
                    "durable_state",
                    "durable_fingerprint_sha256",
                    "mount_binding",
                )
            }
            self.assertNotIn('"device"', json.dumps(persistent))
            self.assertEqual(
                snapshot["sentinel"]["runtime_observation"][
                    "directory_devices"
                ]["data_root"]["device"],
                4242,
            )
            for _ in range(2):
                with self.assertRaises(recovery.OfflineRecoveryError):
                    recovery.production_snapshot(mountinfo_lines=mountinfo)

            expected_call = mock.call(
                data_root, cache_root, mountinfo_lines=mountinfo
            )
            self.assertEqual(capture.call_args_list, [expected_call] * 3)
            self.assertEqual(mapping_digest.call_count, 3)
            write_new.assert_not_called()
            self.assertFalse(data_root.exists())
            self.assertFalse(cache_root.exists())

    def test_prepare_with_unconfigured_pins_creates_nothing_and_writes_no_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            source = parent / "source"
            source.mkdir()
            evidence = parent / "existing-evidence.json"
            evidence.write_bytes(b'{"preserve":true}\n')
            before_stat = evidence.stat()
            before = (
                evidence.read_bytes(),
                before_stat.st_ino,
                before_stat.st_mode,
                before_stat.st_size,
                before_stat.st_mtime_ns,
            )
            run_id = "phase6f-recovery-0.4.3-unconfigured-pins"
            destination = parent / run_id
            host = mock.Mock(nodename="arnor")
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(recovery, "CANARY_ROOT", parent)
                )
                stack.enter_context(
                    mock.patch.object(recovery, "SOURCE_ROOT", source)
                )
                stack.enter_context(
                    mock.patch.object(
                        recovery, "PRODUCTION_DURABLE_FINGERPRINT", None
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        recovery, "PRODUCTION_MOUNT_BINDING", None
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        recovery, "release_environment", return_value={}
                    )
                )
                stack.enter_context(
                    mock.patch.object(recovery.os, "uname", return_value=host)
                )
                write_new = stack.enter_context(
                    mock.patch.object(recovery, "_write_new")
                )
                with self.assertRaisesRegex(
                    recovery.OfflineRecoveryError, "pins are unconfigured"
                ):
                    recovery.prepare(run_id, "2" * 40)
            after_stat = evidence.stat()
            after = (
                evidence.read_bytes(),
                after_stat.st_ino,
                after_stat.st_mode,
                after_stat.st_size,
                after_stat.st_mtime_ns,
            )
            self.assertFalse(destination.exists())
            self.assertEqual(before, after)
            write_new.assert_not_called()

    def test_persistent_v2_qualification_ignores_cross_session_device(self):
        first = _production_sentinel(42)
        later = _production_sentinel(59)
        expected = {
            "expected_schema": recovery.PRODUCTION_SENTINEL_SCHEMA,
            "expected_durable_fingerprint": first[
                "durable_fingerprint_sha256"
            ],
            "expected_mount_binding": first["mount_binding"],
        }
        self.assertTrue(
            recovery.qualify_production_sentinel(first, **expected)["passed"]
        )
        self.assertTrue(
            recovery.qualify_production_sentinel(later, **expected)["passed"]
        )

    def test_persistent_v2_qualification_rejects_mutation_and_substitution(self):
        accepted = _production_sentinel()
        expected = {
            "expected_schema": recovery.PRODUCTION_SENTINEL_SCHEMA,
            "expected_durable_fingerprint": accepted[
                "durable_fingerprint_sha256"
            ],
            "expected_mount_binding": accepted["mount_binding"],
        }
        cases = {}
        content = copy.deepcopy(accepted)
        content["durable_state"]["fixture"] = "changed"
        content["durable_fingerprint_sha256"] = recovery._mapping_digest(
            content["durable_state"]
        )
        cases["content"] = content
        cases["mount-source"] = _production_sentinel(
            source="substitute:/production"
        )
        for predicate, value in (
            ("target_absent", False),
            ("cache_absent", False),
            ("transaction_artifacts", ["transaction-residue"]),
        ):
            changed = copy.deepcopy(accepted)
            changed["qualification_predicates"][predicate] = value
            cases[predicate] = changed
        old = copy.deepcopy(accepted)
        old["schema_version"] = "phase6.production-sentinel.v1"
        cases["v1-schema"] = old
        for name, value in cases.items():
            with self.subTest(case=name), self.assertRaises(
                recovery.OfflineRecoveryError
            ):
                recovery.qualify_production_sentinel(value, **expected)

    def test_v0_4_2_prepared_root_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            recovery, "CANARY_ROOT", Path(temporary).resolve()
        ):
            with self.assertRaises(recovery.OfflineRecoveryError):
                recovery._root("phase6f-recovery-0.4.2-existing")

    def test_v1_preparation_binding_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            parent = Path(temporary).resolve()
            root = parent / "phase6f-recovery-0.4.3-old-binding"
            root.mkdir(mode=0o700)
            sentinel = _production_sentinel()
            stack.enter_context(mock.patch.object(recovery, "CANARY_ROOT", parent))
            stack.enter_context(
                mock.patch.object(
                    recovery,
                    "PRODUCTION_DURABLE_FINGERPRINT",
                    sentinel["durable_fingerprint_sha256"],
                )
            )
            stack.enter_context(
                mock.patch.object(
                    recovery, "PRODUCTION_MOUNT_BINDING", sentinel["mount_binding"]
                )
            )
            old = {
                "schema_version": "phase6f.offline-recovery.0.4.1-to-0.4.2.v1",
                "run_id": root.name,
            }
            recovery._write_new(root / "binding.json", recovery._json(old))
            recovery._write_new(
                root / "binding.sha256",
                (recovery._hash(root / "binding.json") + "\n").encode(),
            )
            with self.assertRaisesRegex(
                recovery.OfflineRecoveryError, "binding differs"
            ):
                recovery._binding(root, "2" * 40)

    def test_existing_destination_is_refused_before_production_or_source_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve()
            run_id = "phase6f-recovery-0.4.3-test"
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
            provider.clock = lambda: datetime(2026, 9, 3, tzinfo=timezone.utc)
            provider.monotonic = lambda: 0.0
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
            root = parent / "phase6f-recovery-0.4.3-integration"
            root.mkdir(mode=0o700)
            for name in ("logs", "status", "evidence", "candidate", "tmp"):
                (root / name).mkdir(mode=0o700)
            production_sentinel = _production_sentinel()
            values = {"CANARY_ROOT": parent, "SOURCE_ROOT": source, "OBJECTS": 3, "ALERTS": 3, "SEGMENTS": 1, "QUERY_ID": loaded.integrity_sha256, "QUERY_ORDER": fetch_binding.query_locus_order_sha256, "FETCH_ID": fetch_binding.identity_sha256, "PRODUCTION_DURABLE_FINGERPRINT": production_sentinel["durable_fingerprint_sha256"], "PRODUCTION_MOUNT_BINDING": production_sentinel["mount_binding"]}
            for name, value in values.items():
                stack.enter_context(mock.patch.object(recovery, name, value))
            binding = {"schema_version": recovery.CONTRACT, "run_id": root.name, "run_root": str(root), "source_root": str(source), "source_version": recovery.SOURCE_VERSION, "source_sha": recovery.SOURCE_SHA, "consumer_version": recovery.CONSUMER_VERSION, "consumer_sha": "2" * 40, "night": recovery.NIGHT, "mjd": list(recovery.MJD), "query_identity": recovery.QUERY_ID, "fetch_identity": recovery.FETCH_ID, "authoritative": False, "publishable": False, "publication_authorized": False, "timeout_seconds": recovery.TIMEOUT_SECONDS, "source_identity_schema": recovery.SOURCE_IDENTITY_SCHEMA, "source_durable_identity": source_before["durable_sha256"], "production_qualification": recovery._configured_production_pins()}
            recovery._write_new(root / "binding.json", recovery._json(binding))
            recovery._write_new(root / "binding.sha256", (recovery._hash(root / "binding.json") + "\n").encode())
            stack.enter_context(mock.patch.object(recovery, "release_environment", return_value={"test_environment": True}))
            stack.enter_context(mock.patch.object(recovery, "process_identity", return_value={"pid": os.getpid()}))
            stack.enter_context(mock.patch.object(recovery, "production_snapshot", return_value={"sentinel": production_sentinel}))
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
            # The temporary source path is deliberately embedded in provenance.
            # Normalize only that fixture input, then pin the exact 0.4.2
            # candidate bytes so qualification-schema changes cannot leak into
            # scientific serialization.
            normalized_artifact_hashes = {
                name: hashlib.sha256(
                    (root / "candidate" / name)
                    .read_bytes()
                    .replace(str(source).encode(), b"<SOURCE_ROOT>")
                ).hexdigest()
                for name in ("alerts.parquet", "loci.parquet", "manifest.json")
            }
            self.assertEqual(
                normalized_artifact_hashes,
                {
                    "alerts.parquet": "39e99169af25b18eb40a979bee989ba0368d38439229307a69fb84e99da0993b",
                    "loci.parquet": "1fc5a6225a575c7d811b2b343700cf972b0b65f2259f1d3d3c73045f89c2ec13",
                    "manifest.json": "19df0fc3e978bc7128ae483a21ea1203c33b710d155aafcf05d86294584296c6",
                },
            )
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
            self.assertEqual(
                reopened.manifest["offline_recovery_contract"],
                "phase6f.offline-recovery.0.4.1-to-0.4.2.v1",
            )
            self.assertEqual(recovery._read(root / "status/RECOVERY_FINAL.json"), final)
            with self.assertRaises((recovery.OfflineRecoveryError, FileExistsError)):
                recovery.reconstruct(root, "2" * 40)


if __name__ == "__main__":
    unittest.main()
