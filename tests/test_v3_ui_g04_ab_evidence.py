"""Local-only evidence-harness qualification; no Arnor paths are opened.

Fixtures use release TransactionJournal and AuthorityLock APIs, with minimal
non-secret metadata. No publication, authorization, candidate, or provider is
constructed by this suite.
"""
import ast
import builtins
import contextlib
import fcntl
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


LOCAL_GUARD = '''
import errno, os, sys
def protected(path):
    if isinstance(path, int): return False
    path = os.path.abspath(os.fsdecode(path))
    return any(path == root or path.startswith(root + '/') for root in
               ('/astro/store/shire/ANTARES', '/astro/users/mdarim/antares-control'))
def audit(event, args):
    if event.startswith('socket.') and event != 'socket.gethostname':
        raise AssertionError('local qualification denies network')
    if event in ('open', 'os.listdir', 'os.scandir') and args and protected(args[0]):
        raise AssertionError('local qualification denies protected reads')
class GuardedStat:
    # A callable object, not a Python function: Python 3.9 pathlib stores these
    # on its accessor class and must not bind an extra accessor argument.
    def __init__(self, original): self.original = original
    def __call__(self, path, *args, **kwargs):
        # Existing release fixtures resolve protected path CONSTANTS. Model
        # absence before any syscall, even if production is accidentally mounted.
        if protected(path): raise FileNotFoundError(errno.ENOENT, 'disabled local root')
        return self.original(path, *args, **kwargs)
os.stat = GuardedStat(os.stat)
os.lstat = GuardedStat(os.lstat)
from pathlib import Path
accessor = getattr(Path('.'), '_accessor', None)
if accessor is not None:
    type(accessor).stat = os.stat
    type(accessor).lstat = os.lstat
sys.addaudithook(audit)
'''
exec(compile(LOCAL_GUARD, "<g04-local-tripwire>", "exec"), {})
_guard_temporary = tempfile.TemporaryDirectory(prefix="g04-local-tripwire-")
_guard_directory = Path(_guard_temporary.name).resolve()
(_guard_directory / "sitecustomize.py").write_text(LOCAL_GUARD)
_original_popen = subprocess.Popen


def local_popen(*args, **kwargs):
    # Existing crash utilities replace PYTHONPATH. Reattach the tripwire to
    # their Python children; isolated harness children already have their own.
    environment = dict(kwargs.get("env") or os.environ)
    environment["PYTHONPATH"] = str(_guard_directory) + os.pathsep + environment.get("PYTHONPATH", "")
    kwargs["env"] = environment
    return _original_popen(*args, **kwargs)


subprocess.Popen = local_popen

import pandas as pd

from src import authority, history
from src.operations import publication as publication
from src.operations.journal import TransactionDescriptor, TransactionJournal
from src.operations.state import ExecutionState

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v3_ui_g04_ab_evidence.py"
SPEC = importlib.util.spec_from_file_location("g04_evidence", SCRIPT)
H = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = H
SPEC.loader.exec_module(H)
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
DATES = ["2026-06-27", "2026-07-01", "2026-02-25"]


def quiet():
    return {"active_controller_present": False, "controllers": []}


class HarnessFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name).resolve()
        root = self.tmp / "fixture"
        data = root / "published"
        survey = history.survey_data_root(data)
        control = root / "control"
        self.layout = H.Layout(root, data, survey, control / "journals",
                              control / "locks" / authority.AUTHORITY_LOCK_NAME,
                              root / "evidence", root / "backfill", history.publication_gate_path(data),
                              history.cumulative_paths(data)["loci_index"],
                              history.cumulative_paths(data)["nightly_summary"])
        for path in (data, self.layout.journals, self.layout.lock.parent,
                     self.layout.index.parent, self.layout.ranges):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.layout.lock.touch(mode=0o600)
        for day in DATES:
            partition = self.layout.partition(day)
            partition.mkdir(parents=True)
            payload = {"date_utc": day}
            if day != DATES[-1]:
                payload["authority"] = {}
            (partition / "manifest.json").write_text(json.dumps(payload))
            (partition / "loci.parquet").write_bytes(b"metadata fixture")
            (partition / "alerts.parquet").write_bytes(b"metadata fixture")
        summary = [{"date_utc": day, "status": "complete", "actual_loci": 2,
                    "started_at_utc": "2026-09-30T01:00:00Z", "finished_at_utc": "2026-09-30T02:00:00Z",
                    "lsst_dia_count": 2, "lsst_ss_count": 0, "ztf_object_id_count": 1} for day in DATES]
        pd.DataFrame(summary).to_parquet(self.layout.summary, index=False)
        pd.DataFrame([{"night_date_utc": day, "ingested_at_utc": stamp} for day in DATES
                      for stamp in ("2026-09-30T01:00:00Z", "2026-09-30T02:00:00Z")]).to_parquet(self.layout.index, index=False)
        for day in DATES[:2]:
            self.journal(day)

    def tearDown(self):
        self.temporary.cleanup()

    def journal(self, day, *, finalized=True):
        run_id = f"v3pub-{day}-abcdef012345-a1"
        path = self.layout.journals / (run_id + ".json")
        if path.exists():
            path.unlink()
        descriptor = TransactionDescriptor(
            run_id, publication.PUBLICATION_OPERATION, day, str(self.layout.partition(day)),
            str(self.tmp / "unused-stage"), str(self.layout.lock), "synthetic",
            metadata={"target_utc_night": day,
                      "authoritative_manifest_sha256": H.sha256_file(self.layout.partition(day) / "manifest.json"),
                      "cumulative": {"expected_sha256": {"loci_index": H.sha256_file(self.layout.index),
                                                           "nightly_summary": H.sha256_file(self.layout.summary)}}})
        journal = TransactionJournal.create(path, descriptor, at=NOW)
        stages = list(ExecutionState)[:10 if finalized else 8]
        for stage in stages[1:]:
            journal.transition(stage, at=NOW, reconciliation={"resulting_production_fingerprint": "a" * 64})
        H.scan_forbidden(json.loads(path.read_text()))
        return journal

    def safety(self, sleeper=lambda seconds: None, controllers=quiet):
        return H.ap0(self.layout, sleep=sleeper, clock=lambda: NOW, controllers=controllers)

    def event(self, stage, *, stamp=NOW, event="stage", extras=None):
        path = self.layout.ranges / "run-1/nights/night-2026-07-02/events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps({"stage": stage, "event": event, "utc": stamp.isoformat(), **(extras or {})}) + "\n")
        return path

    def discovery(self):
        return {"AP-1": {"tier_b_uid_eligible": True}, "AP-3": H.ap3(self.layout)}

    def evidence(self):
        return H.ap6_locked(self.layout, DATES, sorted(DATES), [DATES[-1]], history=history,
                            classify=publication.classify_night_authority)


class IdentityAndCLI(HarnessFixture):
    def test_release_source_identity_and_static_paths(self):
        self.assertEqual(H.verify_repository(ROOT)["source_sha"], H.BASELINE)
        layout = H.static_layout(ROOT)
        self.assertEqual(layout.index, history.cumulative_paths(layout.data)["loci_index"])
        self.assertEqual(layout.gate, history.publication_gate_path(layout.data))
        self.assertEqual(layout.partition(DATES[0]), history.nightly_paths(layout.data, DATES[0])["dir"])
        self.assertEqual(str(layout.survey), "/astro/store/shire/ANTARES/data/data/lsst_only")

    def test_wrong_repository_rejected(self):
        with self.assertRaises(H.Refuse):
            H.verify_repository(self.tmp)

    def test_wrong_release_source_rejected(self):
        with mock.patch.object(H, "source_digest", return_value="different"):
            with self.assertRaisesRegex(H.Refuse, "RELEASE_SOURCE_IDENTITY"):
                H.verify_repository(ROOT)

    def test_wrong_installed_release_rejected_before_import(self):
        with self.assertRaisesRegex(H.Refuse, "IMMUTABLE_RELEASE_IDENTITY"):
            H.verify_installed_release()

    def test_default_no_tier_has_no_reads(self):
        with mock.patch.object(H, "verify_repository", side_effect=AssertionError("unexpected read")), \
                mock.patch.object(H, "metadata", side_effect=AssertionError("unexpected read")), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                H.main([])
        self.assertEqual(stopped.exception.code, 2)

    def test_tier_b_requires_acknowledgement_before_any_read(self):
        with mock.patch.object(H, "output_root", side_effect=AssertionError("unexpected read")):
            with self.assertRaisesRegex(H.Refuse, "CONTROL_ACKNOWLEDGEMENT_REQUIRED"):
                H.main(["--tier-b", "--output-dir", str(self.tmp / "out")])

    def test_plan_no_package_import_no_production_read(self):
        output = self.tmp / "plan"
        # Child instrumentation rejects even a production lstat, not only open.
        code = """
import pathlib,runpy,sys
original = pathlib.Path.lstat
def no_production(self, *a, **k):
    if str(self).startswith('/astro/store/shire/ANTARES'):
        raise AssertionError('production metadata read')
    return original(self,*a,**k)
pathlib.Path.lstat = no_production
sys.argv = [sys.argv[1], '--plan', '--output-dir', sys.argv[2]]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as e:
    if e.code: raise
assert 'src.history' not in sys.modules
assert 'antares_client' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SCRIPT), str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        packet = json.loads(next(output.glob("*.json")).read_text())
        H.scan_forbidden(packet)
        self.assertEqual(packet["status"], "NOT_EXECUTED")

    def test_tier_c_has_no_options_or_executable_paths(self):
        help_text = H.parser().format_help()
        tree = ast.parse(SCRIPT.read_text())
        literals = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        for label in ("--tier-c", "AP-7", "AP-8", "AP-9", "AP-10"):
            self.assertNotIn(label, help_text)
            self.assertFalse(any(label in value for value in literals))
        self.assertNotIn("src.operations.production_range", [ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))])


class ConfinementTests(HarnessFixture):
    def test_protected_output_rejected_without_protected_metadata_reads(self):
        with mock.patch.object(Path, "lstat", side_effect=AssertionError("protected stat")):
            for root in (H.AUTHORITY, H.CONTROL_SECRETS):
                for path in (root, root / "out", root.parent, Path("/")):
                    with self.subTest(path=path), self.assertRaisesRegex(H.Refuse, "PROTECTED_OUTPUT_ROOT"):
                        H.output_root(path)

    def test_relative_and_traversal_rejected(self):
        for path in ("relative", str(self.tmp / "x/../out")):
            with self.assertRaises(H.Refuse):
                H.output_root(path)

    def test_symlink_output_components_and_dangling_output_rejected(self):
        target = self.tmp / "target"
        target.mkdir(mode=0o700)
        link = self.tmp / "alias"
        link.symlink_to(target, target_is_directory=True)
        dangling = self.tmp / "dangling"
        dangling.symlink_to(self.tmp / "missing")
        for path in (link, link / "out", dangling):
            with self.assertRaisesRegex(H.Refuse, "SYMLINK_PATH"):
                H.output_root(path)

    def test_nonprivate_or_non_directory_output_rejected(self):
        output = self.tmp / "public"
        output.mkdir(mode=0o755)
        with self.assertRaises(H.Refuse):
            H.output_root(output)
        with self.assertRaises(H.Refuse):
            H.output_root(self.layout.summary)

    def test_outputs_are_new_private_and_recursive_scan_passes(self):
        output = self.tmp / "out"
        packet = self.evidence()
        first = H.write_evidence(output, "b", packet)
        second = H.write_evidence(output, "b", packet)
        self.assertNotEqual(first, second)
        self.assertEqual(first.stat().st_mode & 0o777, 0o600)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual(set(path.parent for path in output.iterdir()), {output})
        for path in output.iterdir():
            H.scan_forbidden(json.loads(path.read_text()))

    def test_forbidden_key_scan_recursive_and_no_value_disclosure(self):
        for key in H.FORBIDDEN_KEYS:
            with self.subTest(key=key), self.assertRaisesRegex(H.Refuse, "^FORBIDDEN_EVIDENCE_KEY$"):
                H.scan_forbidden({"allowed": [{key: None}]})
        output = self.tmp / "refused"
        with self.assertRaises(H.Refuse):
            H.write_evidence(output, "a", {"nonce": None})
        self.assertFalse(output.exists())

    def test_audit_denies_control_writes_network_and_client(self):
        guard = H.ReadBoundary(H.static_layout(ROOT), self.tmp / "out", worker=True)
        for path in (H.CONTROL_SECRETS / "token", Path("/private/tmp/production-bindings/record"),
                     self.layout.ranges / "run/authorizations/authorization.json"):
            with self.assertRaises(H.Refuse):
                guard("open", (str(path), "r", 0))
        for event, args in (("socket.__new__", ()), ("socket.getaddrinfo", ()),
                            ("import", ("antares_client.search",)),
                            ("open", (str(H.AUTHORITY / "data/file"), "w", os.O_WRONLY)),
                            ("os.mkdir", (str(H.AUTHORITY / "new"),)),
                            ("subprocess.Popen", ("python", []))):
            with self.subTest(event=event), self.assertRaises(H.Refuse):
                guard(event, args)

    def test_actual_audit_hook_stops_socket_construction(self):
        code = """
import runpy,sys,socket
h=runpy.run_path(sys.argv[1]); layout=h['static_layout'](__import__('pathlib').Path(sys.argv[1]).parents[1])
sys.addaudithook(h['ReadBoundary'](layout,worker=True))
for action in (lambda:socket.socket(),lambda:socket.getaddrinfo('invalid',80)):
    try: action()
    except h['Refuse']: pass
    else: raise AssertionError('network permitted')
"""
        child = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)


class SafetyGateTests(HarnessFixture):
    def test_process_descriptor_owner_rechecked_before_contents(self):
        path = self.tmp / "cmdline"
        path.write_bytes(b"bounded")
        with self.assertRaisesRegex(H.Refuse, "PROCESS_OWNER_CHANGED"):
            H.read_small(path, expected_uid=os.geteuid() + 1)
        self.assertEqual(H.read_small(path, expected_uid=os.geteuid()), b"bounded")

    def test_ap0_pass_observes_exact_60_seconds(self):
        sleeper = mock.Mock()
        result = self.safety(sleeper)
        sleeper.assert_called_once_with(60)
        self.assertTrue(result["tier_b_safe_to_attempt"])

    def test_gate_present_and_dangling_gate_refuse_b(self):
        for linked in (False, True):
            if linked:
                self.layout.gate.symlink_to(self.tmp / "absent")
            else:
                self.layout.gate.touch()
            safety = self.safety()
            self.assertFalse(safety["gate_clear"])
            supervisor = mock.Mock(side_effect=AssertionError("worker started"))
            self.assertEqual(H.run_tier_b(self.layout, self.discovery(), safety, True, supervisor=supervisor)["status"], "REFUSE")
            self.layout.gate.unlink()

    def test_generation_movement_refuses(self):
        def move(seconds):
            path = self.layout.summary
            path.write_bytes(path.read_bytes() + b"x")
        result = self.safety(move)
        self.assertFalse(result["generation_stable"])
        self.assertFalse(result["tier_b_safe_to_attempt"])

    def test_missing_generation_cannot_pass(self):
        self.layout.index.unlink()
        self.assertFalse(self.safety()["generation_stable"])

    def test_recent_activity_remains_unsafe_after_terminal_event(self):
        self.event("QUERYING", stamp=NOW - timedelta(minutes=29), extras={"unused": None})
        self.event("PUBLISHED")
        result = self.safety()
        self.assertFalse(result["no_recent_nonterminal_activity"])
        record = result["range_activity"][0]
        self.assertEqual(set(record), {"run_id", "most_recent_event", "most_recent_stage", "timestamp"})
        self.assertEqual(record["most_recent_stage"], "PUBLISHED")

    def test_old_activity_and_quiet_live_controller_refuse(self):
        self.event("PUBLISHING", stamp=NOW - timedelta(hours=2))
        controller = lambda: {"active_controller_present": True,
                              "controllers": [{"controller_pid": 123, "classification": "production_range.execute_or_recover"}]}
        result = self.safety(controllers=controller)
        self.assertTrue(result["no_recent_nonterminal_activity"])
        self.assertFalse(result["no_live_controller"])
        self.assertFalse(result["tier_b_safe_to_attempt"])

    def test_actual_import_module_and_recover_controller_shapes(self):
        own, other = os.geteuid(), os.geteuid() + 1
        rows = [f"{own} 123 python -I -B -c 'from src.operations.production_range import main; main()' execute --control-token-file /not-read.token",
                f"{own} 124 python -m src.operations.production_range recover --resume",
                f"{own} 125 python -c 'from src.operations.production_range import main; main()' inspect --start x",
                f"{other} 126 python -m src.operations.production_range execute"]
        result = H.live_controllers(rows)
        self.assertEqual([item["controller_pid"] for item in result["controllers"]], [123, 124])
        self.assertNotIn("--control-token-file", json.dumps(result))
        self.assertEqual(set(result), {"active_controller_present", "controllers"})

    def test_discovery_uncertainty_refuses_without_preventing_tier_a(self):
        with mock.patch.object(H, "range_activity", side_effect=H.Refuse("EVENT_WINDOW_INCOMPLETE")):
            result = self.safety(controllers=mock.Mock(side_effect=OSError()))
        self.assertFalse(result["tier_b_safe_to_attempt"])
        self.assertFalse(result["no_live_controller"])
        self.assertFalse(result["no_recent_nonterminal_activity"])
        self.assertIsNotNone(H.ap3(self.layout))

    def test_partial_and_overfull_recent_tail_fail_closed(self):
        path = self.event("QUERYING")
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        self.assertFalse(self.safety()["no_recent_nonterminal_activity"])
        line = json.dumps({"event": "stage", "stage": "PUBLISHED", "utc": NOW.isoformat(), "unused": "x" * 1000}).encode() + b"\n"
        path.write_bytes(line * 1100)
        self.assertFalse(self.safety()["no_recent_nonterminal_activity"])

    def test_uid_or_any_individual_gate_refuses_without_worker(self):
        discovery = self.discovery()
        for flag in ("gate_clear", "generation_stable", "no_live_controller", "no_recent_nonterminal_activity"):
            safe = self.safety()
            safe[flag] = False
            self.assertEqual(H.run_tier_b(self.layout, discovery, safe, True, supervisor=mock.Mock(side_effect=AssertionError()))["status"], "REFUSE")
        discovery["AP-1"]["tier_b_uid_eligible"] = False
        self.assertEqual(H.run_tier_b(self.layout, discovery, self.safety(), True, supervisor=mock.Mock(side_effect=AssertionError()))["status"], "INELIGIBLE")


class DiscoveryTests(HarnessFixture):
    def test_host_and_capability_discovery_without_dns_or_mutation(self):
        facts = {"/etc/hosts": "127.0.0.1 arnor.fixture arnor", "/etc/passwd": f"reader:x:{os.geteuid()}:1::/home/reader:/bin/sh",
                 "/proc/self/cgroup": "0::/user.slice/reader", "/sys/kernel/security/lsm": "capability,selinux,landlock",
                 "/proc/sys/user/max_user_namespaces": "1000", "/proc/sys/kernel/unprivileged_userns_clone": "1"}
        with mock.patch.object(H, "host_text", side_effect=lambda path, *args: facts.get(path, "UNKNOWN")), \
                mock.patch.object(H.socket, "gethostname", return_value="arnor"), \
                mock.patch.object(H.os, "getuid", return_value=H.EXPECTED_UID), \
                mock.patch.object(H.os, "geteuid", return_value=H.EXPECTED_UID), \
                mock.patch.object(H, "landlock_abi", return_value=3), \
                mock.patch.object(H.shutil, "which", return_value="/usr/bin/discovery-only"), \
                mock.patch.object(H.subprocess, "run", side_effect=AssertionError("executable probe")):
            identity, capabilities = H.ap1(), H.ap2()
        self.assertTrue(identity["tier_b_uid_eligible"])
        self.assertEqual(identity["hostname_fqdn_local"], "arnor.fixture")
        self.assertEqual(capabilities["landlock_abi"], 3)
        self.assertEqual(capabilities["unprivileged_userns_clone"], "1")

    def test_host_budget_has_fixed_system_python_probe_and_outside_disk(self):
        with mock.patch.object(H.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Python 3.9.25\n")) as probe, \
                mock.patch.object(H, "host_text", return_value="MemTotal: 1024 kB"):
            result = H.ap4(self.tmp / "output")
        self.assertEqual(result["memory_bytes"], 1024 * 1024)
        self.assertEqual(result["system_python_version"], "Python 3.9.25")
        self.assertEqual(result["console_store_decision"], "UNKNOWN")
        self.assertTrue(all(not H.inside(Path(item["path"]), H.AUTHORITY) for item in result["disk_candidates"]))
        self.assertEqual(probe.call_args.args[0], ["/usr/bin/python3", "-I", "-B", "--version"])

    def test_ap3_metadata_only_actual_tail_and_first_measurement(self):
        with mock.patch.object(H, "read_small", side_effect=AssertionError("content read")), \
                mock.patch.object(Path, "open", side_effect=AssertionError("content read")):
            result = H.ap3(self.layout)
        self.assertEqual(result["directory_tail"], DATES[1])
        self.assertEqual(result["directory_night_count"], 3)
        self.assertEqual(result["newest_v3_candidate"], DATES[1])
        self.assertEqual(result["legacy_candidate"], DATES[-1])
        self.assertEqual(result["journal_count"], 2)
        self.assertGreater(result["journal_total_bytes"], 0)
        self.assertEqual(len(result["p2_x1_first_measured_manifest_sizes"]), 3)

    def test_ap3_unexpected_symlink_and_type_stop(self):
        path = self.layout.partition(DATES[1]) / "unexpected"
        path.symlink_to(self.tmp / "missing")
        with self.assertRaisesRegex(H.Refuse, "SYMLINK_PATH"):
            H.ap3(self.layout)
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(H.Refuse, "UNEXPECTED_FILE_TYPE"):
            H.ap3(self.layout)

    def test_nfs_protocol_attribute_cache_evidence(self):
        result = H.mount_facts(["20 1 0:60 / /astro/store/shire rw - nfs4 shire.infiniband:/data/shire rw,vers=4.2,proto=rdma,local_lock=none,acregmin=3,acregmax=60,acdirmin=30,acdirmax=60"])
        self.assertEqual(result["filesystem_type"], "nfs4")
        self.assertIn("vers=4.2", result["nfs_options"])
        self.assertIn("proto=rdma", result["nfs_options"])
        self.assertIn("acregmin=3", result["nfs_options"])

    def test_policy_questions_exact_and_unknown(self):
        with mock.patch.object(H, "ap1", return_value={}), mock.patch.object(H, "ap2", return_value={}), \
                mock.patch.object(H, "ap4", return_value={}):
            result = H.tier_a(self.layout, self.tmp)
        self.assertEqual(len(result["AP-5"]), 6)
        self.assertTrue(all(item["answer"] == "UNKNOWN" for item in result["AP-5"]))
        self.assertEqual([item["question"] for item in result["AP-5"]], list(H.POLICY_QUESTIONS))


class ReleaseReadTests(HarnessFixture):
    def test_nullable_count_storage_preserves_available_counts(self):
        pd.DataFrame({"date_utc": DATES, "lsst_dia_count": [2.0, 3.0, None],
                      "lsst_ss_count": [0.0, None, 4.0]}).to_parquet(self.layout.summary, index=False)
        result = self.evidence()["nightly_summary"]
        self.assertEqual(result["rows"][0]["lsst_dia_count"], 2)
        self.assertEqual(result["rows"][0]["lsst_ss_count"], 0)
        self.assertIsNone(result["rows"][2]["lsst_dia_count"])

    def test_nested_shared_classification_uses_one_flock_and_no_terminal(self):
        with mock.patch.object(authority, "_flock", wraps=authority._flock) as flock:
            result = self.evidence()
        self.assertEqual(flock.call_count, 1)
        self.assertTrue(result["classifications"][DATES[0]]["finalized"])
        self.assertFalse((self.layout.evidence / "publications").exists())
        self.assertTrue(result["classifications"][DATES[-1]]["legacy"])
        self.assertEqual(result["loci_index"]["nights"][DATES[-1]]["row_count"], 2)
        self.assertEqual(result["legacy_metadata_candidate_time_evidence_classes"], {"ORDERED": 1})

    def test_committed_unfinalized_not_science_input(self):
        self.journal(DATES[1], finalized=False)
        actual = publication.classify_night_authority(self.layout.data, self.layout.journals, DATES[1])
        self.assertEqual(actual["state"], "COMPLETE")
        self.assertFalse(actual["finalized"])
        with self.assertRaisesRegex(H.Refuse, "UNFINALIZED_OR_UNPUBLISHED_NIGHT"):
            self.evidence()

    def test_existing_gate_refuses_authoritative_read(self):
        self.layout.gate.touch()
        classify = mock.Mock(side_effect=AssertionError("classification under gate"))
        with self.assertRaises(history.PublicationInProgress):
            H.ap6_locked(self.layout, DATES, DATES, [DATES[-1]], history=history, classify=classify)
        self.assertFalse(classify.called)

    def test_generation_drift_discards_evidence(self):
        original = H.parquet_rows
        def drift(path, columns):
            result = original(path, columns)
            if columns == H.INDEX_COLUMNS:
                with self.layout.summary.open("ab") as handle:
                    handle.write(b"drift")
            return result
        with mock.patch.object(H, "parquet_rows", side_effect=drift):
            with self.assertRaises(history.PublicationInProgress):
                self.evidence()

    def test_shared_lock_wait_has_one_bounded_attempt(self):
        code = "import fcntl,sys,time; f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_EX); print('held',flush=True); time.sleep(20)"
        child = subprocess.Popen([sys.executable, "-I", "-B", "-c", code, str(self.layout.lock)], stdout=subprocess.PIPE)
        try:
            self.assertEqual(child.stdout.readline().strip(), b"held")
            with mock.patch.object(history, "authoritative_read", wraps=history.authoritative_read) as entered:
                started = time.monotonic()
                with self.assertRaises(history.PublicationInProgress) as refused:
                    self.evidence()
                duration = time.monotonic() - started
                entered.assert_called_once_with(self.layout.data, wait_seconds=5)
            self.assertIsInstance(refused.exception.__cause__, authority.AuthorityLockUnavailable)
            self.assertGreaterEqual(duration, 4.9)
            self.assertLess(duration, 5.5)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()

    def test_missing_columns_are_absent_and_extra_values_never_emitted(self):
        pd.DataFrame({"date_utc": DATES, "status": ["complete"] * 3, "unused": [None] * 3}).to_parquet(self.layout.summary, index=False)
        pd.DataFrame({"night_date_utc": DATES}).to_parquet(self.layout.index, index=False)
        result = self.evidence()
        self.assertEqual(result["nightly_summary"]["columns"]["lsst_ss_count"], "ABSENT")
        self.assertNotIn("lsst_ss_count", result["nightly_summary"]["rows"][0])
        self.assertNotIn("unused", json.dumps(result))
        self.assertEqual(result["loci_index"]["columns"]["ingested_at_utc"], "ABSENT")
        self.assertEqual(result["loci_index"]["aggregation_status"], "INCOMPLETE")
        self.assertIsNone(result["partition_candidates_without_index_rows"])

    def test_published_partition_exclusion_and_zero_row_caveat(self):
        pd.DataFrame({"night_date_utc": [DATES[0]], "ingested_at_utc": [NOW.isoformat()]}).to_parquet(self.layout.index, index=False)
        result = self.evidence()
        self.assertEqual(result["confirmed_published_representatives_without_index_rows"], DATES[1:])
        self.assertIn("ALONE", result["zero_row_interpretation"])

    def test_descriptor_handles_closed_before_context_exit(self):
        handles = []
        original_io, original_fd = io.open, os.fdopen
        def opened(*args, **kwargs):
            handle = original_io(*args, **kwargs)
            handles.append(handle)
            return handle
        def fdopened(*args, **kwargs):
            handle = original_fd(*args, **kwargs)
            handles.append(handle)
            return handle
        def checked(*args):
            self.assertTrue(handles)
            self.assertTrue(all(handle.closed for handle in handles))
            return "PASS_INSTRUMENTED"
        with mock.patch("io.open", side_effect=opened), mock.patch("os.fdopen", side_effect=fdopened), \
                mock.patch.object(H, "assert_descriptors_closed", side_effect=checked):
            result = self.evidence()
        self.assertEqual(result["descriptor_check_before_release"], "PASS_INSTRUMENTED")

    def test_descriptor_leak_detection_refuses(self):
        with mock.patch.object(H, "descriptor_targets", return_value={8: str(self.layout.summary)}):
            with self.assertRaisesRegex(H.Refuse, "DESCRIPTOR_LEAK"):
                H.assert_descriptors_closed(self.layout, {8: str(self.layout.summary)})

    def test_classification_allowlist_strips_unneeded_fields(self):
        source = publication.classify_night_authority(self.layout.data, self.layout.journals, DATES[0])
        result = H.classification_fields(source)
        self.assertLessEqual(set(result), set(H.CLASSIFICATION_FIELDS))
        H.scan_forbidden(result)
        self.assertNotIn("authorization_sha256", result)


class SupervisorTests(HarnessFixture):
    def test_local_kernel_rule_construction_and_bpf_decisions(self):
        import ctypes
        # Exercise the actual rule builder without applying restrictions to the
        # macOS test runner. A small BPF interpreter checks its syscall decisions.
        allowed = self.tmp / "read-only"
        allowed.touch()
        directory = self.tmp / "directory-only"
        directory.mkdir()
        base = os.open(allowed, os.O_RDONLY)
        captured = {}
        class Syscall:
            restype = None
            def __call__(self, number, *args):
                if number == 444 and args[-1] == 1:
                    return 3
                if number == 444:
                    captured["handled"] = args[0]._obj.handled_access_fs
                    return os.dup(base)
                if number == 445:
                    captured.setdefault("allowed", []).append(args[2]._obj.allowed_access)
                return 0
        def prctl(option, *args):
            if option == 22:
                program = args[1]._obj
                captured["bpf"] = [(row.code, row.jt, row.jf, row.k)
                                   for row in program.filters[:program.length]]
            return 0
        library = SimpleNamespace(syscall=Syscall(), prctl=prctl)
        try:
            with mock.patch.object(H.sys, "platform", "linux"), \
                    mock.patch.object(H.platform, "machine", return_value="x86_64"), \
                    mock.patch("ctypes.CDLL", return_value=library), \
                    mock.patch.object(os, "O_PATH", 0, create=True), \
                    mock.patch.object(os, "open", side_effect=lambda *args: os.dup(base)):
                result = H.kernel_read_boundary([allowed], directory_only=(directory,))
            self.assertEqual(result["native_child_processes"], "DENIED")
            self.assertEqual(captured["handled"], (1 << 15) - 1)
            self.assertEqual(sorted(captured["allowed"]), [1 << 2, 1 << 3])
            def decision(number, flags=0, arch=0xC000003E):
                pc, accumulator = 0, 0
                while True:
                    code, yes, no, value = captured["bpf"][pc]
                    if code == 0x20:
                        accumulator = {0: number, 4: arch, 16: flags}[value]
                    elif code in {0x15, 0x35, 0x45}:
                        matched = accumulator == value if code == 0x15 else accumulator >= value if code == 0x35 else bool(accumulator & value)
                        pc += yes if matched else no
                    elif code == 0x06:
                        return value
                    else:
                        self.fail("unexpected BPF operation")
                    pc += 1
            for syscall in (41, 42, 53, 57, 58, 59, 288, 299, 307, 322, 425, 426, 427, 0x40000000):
                self.assertEqual(decision(syscall), 0x00050001)
            self.assertEqual(decision(56), 0x00050001)
            self.assertEqual(decision(56, 0x10000), 0x7FFF0000)
            self.assertEqual(decision(435), 0x00050026)
            self.assertEqual(decision(1), 0x7FFF0000)  # pipes/stdout
            self.assertEqual(decision(1, arch=0), 0x00050001)
        finally:
            os.close(base)

    def test_kernel_unavailable_has_no_fallback(self):
        with mock.patch.object(H.sys, "platform", "darwin"):
            with self.assertRaisesRegex(H.Refuse, "KERNEL_BOUNDARY_UNAVAILABLE"):
                H.kernel_read_boundary([])

    def test_hard_timeout_kills_noncooperative_lock_holder_and_releases_lock(self):
        code = """
import fcntl,json,os,signal,sys,time
sys.stdin.buffer.read()
signal.signal(signal.SIGTERM,signal.SIG_IGN)
f=open(sys.argv[1],'rb'); fcntl.flock(f,fcntl.LOCK_SH)
print(json.dumps({'phase':'lock_acquired','at_ns':time.monotonic_ns()}),flush=True)
while True: pass
"""
        result = H.supervise([sys.executable, "-I", "-B", "-c", code, str(self.layout.lock)], {}, hard_seconds=0.6)
        self.assertEqual(result["status"], "HOLD_TIMEOUT")
        self.assertEqual(result["worker_returncode"], -signal.SIGKILL)
        self.assertGreater(result["hold_upper_bound_seconds"], 0)
        self.assertLess(result["worker_lifetime_seconds"], 1.2)
        with self.layout.lock.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_deadline_also_covers_blocked_input(self):
        code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"
        result = H.supervise([sys.executable, "-I", "-B", "-c", code], {"bounded": "x" * 150000}, hard_seconds=0.3)
        self.assertEqual(result["status"], "HOLD_TIMEOUT")
        self.assertEqual(result["worker_returncode"], -signal.SIGKILL)

    def test_deadline_cannot_be_extended(self):
        with self.assertRaises(H.Refuse):
            H.supervise([], {}, hard_seconds=10.001)

    def test_worker_receives_only_bounded_request_and_no_local_src_injection(self):
        supervisor = mock.Mock(return_value={"status": "PASS"})
        H.run_tier_b(self.layout, self.discovery(), self.safety(), True, supervisor=supervisor)
        command, request = supervisor.call_args.args
        self.assertEqual(command[1:4], ["-I", "-B", "-c"])
        self.assertEqual(request["representatives"], DATES)
        self.assertNotIn("PYTHONPATH", json.dumps(request))
        H.scan_forbidden(request)

    def test_supervisor_worker_refusal_is_clean_plain_data(self):
        code = "import json,sys; sys.stdin.buffer.read(); print(json.dumps({'phase':'result','status':'REFUSE','refusal':'LOCK_UNAVAILABLE'}))"
        result = H.supervise([sys.executable, "-I", "-B", "-c", code], {})
        self.assertEqual(result["status"], "REFUSE")
        H.scan_forbidden(result)

    @unittest.skipUnless(sys.platform == "linux" and __import__("platform").machine() == "x86_64", "native Linux boundary requires Linux/x86_64; local host is macOS")
    def test_native_kernel_write_read_and_network_boundary(self):
        # Separate process because restrictions are irreversible. No production paths.
        allowed = self.tmp / "allowed"
        allowed.write_text("bounded")
        excluded = self.tmp / "excluded"
        excluded.write_text("bounded")
        code = """
import ctypes,pathlib,runpy,socket,sys
h=runpy.run_path(sys.argv[1]); allowed=pathlib.Path(sys.argv[2]); excluded=pathlib.Path(sys.argv[3])
if h['landlock_abi']() == 'UNAVAILABLE' or h['landlock_abi']() < 3: sys.exit(77)
paths=[allowed,pathlib.Path('/usr')]
for name in ('/lib','/lib64'):
    if pathlib.Path(name).exists(): paths.append(pathlib.Path(name).resolve())
h['kernel_read_boundary'](paths)
assert allowed.read_text() == 'bounded'
for action in (lambda:allowed.write_text('attempt'), lambda:excluded.read_text(), lambda:socket.socket()):
    try: action()
    except PermissionError: pass
    else: raise AssertionError('kernel boundary permitted access')
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(SCRIPT), str(allowed), str(excluded)], capture_output=True, text=True)
        if result.returncode == 77:
            self.skipTest("Landlock ABI >=3 unavailable")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
