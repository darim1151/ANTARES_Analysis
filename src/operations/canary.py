"""One-shot installed-package Phase 5 Arnor/Shire canary harness.

Every Shire write is confined to one exact existing/created direct child of
``/astro/store/shire/ANTARES/work/canary``.  The harness never imports or
constructs a live ANTARES provider and has no production capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from src.cli_profiles import (
    MIDDLE_EARTH_CACHE_ROOT,
    MIDDLE_EARTH_CANARY_ROOT,
    MIDDLE_EARTH_PROJECT_ROOT,
)

from .journal import TransactionJournal
from .locking import _process_start_identity
from .recovery import RecoveryDisposition, inspect_recovery
from .science import NightScienceRequest, SyntheticScienceProvider
from .storage import SyntheticWriteCapability
from .writer import (
    FailureInjector,
    NightExecutionSpec,
    SyntheticReconciler,
    _ensure_private_tree,
    execute_synthetic_night,
    nightly_target_relative,
)


CANARY_SCHEMA_VERSION = "phase5.arnor-canary.v1"
CANARY_ROOT_SCHEMA_VERSION = "phase5.arnor-canary-root.v1"
CANARY_IDENTITY_NAME = ".canary-root.json"


class CanaryError(RuntimeError):
    pass


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _host_is_arnor() -> bool:
    return socket.gethostname().split(".", 1)[0].lower() == "arnor"


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Short canary evidence write.")
        offset += written


def _write_exclusive_private(path: Path, payload: bytes) -> None:
    parent_fd = os.open(
        str(path.parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _unlink_private_marker(path: Path) -> None:
    parent_fd = os.open(
        str(path.parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid():
            raise CanaryError("Canary marker identity is unsafe.")
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _stop_processes(processes: Iterable[subprocess.Popen]) -> None:
    values = list(processes)
    for process in values:
        if process.poll() is None:
            process.terminate()
    for process in values:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    for process in values:
        if process.poll() is None:
            process.wait(timeout=5)


def _read_root_identity(run_root: Path, run_id: str) -> Mapping[str, Any]:
    root_fd = os.open(
        str(run_root),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(root_fd)
        marker_fd = os.open(
            CANARY_IDENTITY_NAME,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        try:
            marker_stat = os.fstat(marker_fd)
            if (
                not stat.S_ISREG(marker_stat.st_mode)
                or marker_stat.st_uid != os.geteuid()
                or marker_stat.st_size > 64 * 1024
            ):
                raise CanaryError("Canary root identity marker is unsafe.")
            chunks = []
            remaining = 64 * 1024 + 1
            while remaining > 0:
                block = os.read(marker_fd, min(4096, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
        finally:
            os.close(marker_fd)
    finally:
        os.close(root_fd)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CanaryError("Canary root identity marker is invalid.") from exc
    expected = {
        "schema_version": CANARY_ROOT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_root": str(run_root),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
    }
    if value != expected or observed.st_uid != os.geteuid():
        raise CanaryError("Canary root identity does not match its bootstrap seal.")
    return value


def _assert_pristine_run_root(run_root: Path) -> None:
    descriptor = os.open(
        str(run_root),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        if set(os.listdir(descriptor)) != {CANARY_IDENTITY_NAME}:
            raise CanaryError(
                "Canary qualification requires a pristine one-shot bootstrap root."
            )
    finally:
        os.close(descriptor)


def _decode_mount_path(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def _assert_no_nested_mounts(run_root: Path) -> None:
    mountinfo = Path("/proc/self/mountinfo")
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CanaryError("Linux mount topology is unavailable for canary safety.") from exc
    canonical_root = run_root.resolve(strict=True)
    for line in lines:
        fields = line.split()
        if len(fields) < 6:
            raise CanaryError("Linux mount topology is malformed.")
        mount_path = Path(_decode_mount_path(fields[4]))
        try:
            mount_path.relative_to(canonical_root)
        except ValueError:
            continue
        if mount_path != canonical_root:
            raise CanaryError(
                f"Nested mount beneath canary root is refused: {mount_path}."
            )


def _safe_component(value: str, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or len(Path(text).parts) != 1
    ):
        raise CanaryError(f"{label} must be one safe path component.")
    return text


def _expected_root(run_id: str) -> Path:
    return MIDDLE_EARTH_CANARY_ROOT / _safe_component(run_id, "run_id")


def bootstrap_run_root(run_id: str) -> Mapping[str, Any]:
    """Create exactly one absent Arnor canary run root through a pinned parent."""
    if not _host_is_arnor():
        raise CanaryError("Canary bootstrap is restricted to host arnor.")
    run_id = _safe_component(run_id, "run_id")
    parent = MIDDLE_EARTH_CANARY_ROOT
    if parent.is_symlink() or not parent.is_dir():
        raise CanaryError("Canonical canary parent is missing or unsafe.")
    if parent.resolve(strict=True) != parent:
        raise CanaryError("Canonical canary parent uses a path alias.")
    target = parent / run_id
    try:
        target.lstat()
    except FileNotFoundError:
        pass
    else:
        raise CanaryError("Canary run root already exists; run ids are one-shot.")
    parent_fd = os.open(
        str(parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_stat = os.fstat(parent_fd)
        path_parent_stat = os.stat(parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or (parent_stat.st_dev, parent_stat.st_ino)
            != (path_parent_stat.st_dev, path_parent_stat.st_ino)
        ):
            raise CanaryError("Canonical canary parent identity changed.")
        os.mkdir(run_id, mode=0o700, dir_fd=parent_fd)
        root_fd = os.open(
            run_id,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(root_fd, 0o700)
            observed = os.fstat(root_fd)
            identity = {
                "schema_version": CANARY_ROOT_SCHEMA_VERSION,
                "run_id": run_id,
                "run_root": str(target),
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "uid": observed.st_uid,
                "gid": observed.st_gid,
            }
            marker_fd = os.open(
                CANARY_IDENTITY_NAME,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                os.fchmod(marker_fd, 0o600)
                _write_all(
                    marker_fd,
                    json.dumps(
                        identity, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                    + b"\n",
                )
                os.fsync(marker_fd)
            finally:
                os.close(marker_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return {
        "run_id": run_id,
        "run_root": str(target),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "identity_marker": CANARY_IDENTITY_NAME,
    }


def _mjd(date_utc: str) -> float:
    from astropy.time import Time

    return float(Time(date_utc, format="iso", scale="utc").mjd)


def _spec(
    transaction_id: str,
    date_utc: str,
    release_sha: str,
) -> NightExecutionSpec:
    lower = _mjd(date_utc)
    return NightExecutionSpec(
        transaction_id=transaction_id,
        plan_id=f"canary-plan-{transaction_id}",
        release_sha=release_sha,
        configuration_identity="phase5-arnor-canary-v1",
        science_request=NightScienceRequest(
            date_utc,
            lower,
            lower + 1.0,
            target_loci=2,
        ),
    )


def _assessment(capability: SyntheticWriteCapability, transaction_id: str):
    journal_path = capability.journal_root / f"{transaction_id}.json"
    snapshot = TransactionJournal.load(journal_path).snapshot
    return inspect_recovery(
        journal_path,
        target_path=Path(snapshot.descriptor.target_path),
        stage_path=Path(snapshot.descriptor.stage_path),
        lock_path=Path(snapshot.descriptor.lock_path),
    )


def _worker_command(
    *,
    run_root: Path,
    run_id: str,
    transaction_id: str,
    date_utc: str,
    release_sha: str,
    scenario: str = "success_nonzero",
    death_point: Optional[str] = None,
    hold_ready: Optional[Path] = None,
    hold_release: Optional[Path] = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.operations.canary",
        "_worker",
        "--run-root",
        str(run_root),
        "--run-id",
        run_id,
        "--transaction-id",
        transaction_id,
        "--date",
        date_utc,
        "--release-sha",
        release_sha,
        "--scenario",
        scenario,
    ]
    if death_point:
        command += ["--death-point", death_point]
    if hold_ready:
        command += ["--hold-ready", str(hold_ready)]
    if hold_release:
        command += ["--hold-release", str(hold_release)]
    return command


def _worker(args: argparse.Namespace) -> int:
    capability = SyntheticWriteCapability.for_arnor_canary_root(
        args.run_root, args.run_id
    )
    _read_root_identity(capability.root, capability.run_id)
    _assert_no_nested_mounts(capability.root)
    ready = Path(args.hold_ready) if args.hold_ready else None
    release = Path(args.hold_release) if args.hold_release else None
    if ready is not None:
        _ensure_private_tree(capability.evidence_root, capability.root)
        ready = ready.resolve(strict=False)
        release = release.resolve(strict=False) if release is not None else None
        for path in (ready, release):
            if path is None:
                continue
            try:
                path.relative_to(capability.evidence_root)
            except ValueError as exc:
                raise CanaryError("Hold marker escaped the evidence root.") from exc

    def hook(point: str, details: Mapping[str, Any]) -> None:
        del details
        if args.death_point and point == args.death_point:
            os._exit(91)
        if ready is not None and point == "after_lock":
            _write_exclusive_private(ready, b"ready\n")
            deadline = time.monotonic() + 30.0
            while release is not None and not release.exists():
                if time.monotonic() >= deadline:
                    raise CanaryError("Timed out waiting for contention release.")
                time.sleep(0.02)

    report = execute_synthetic_night(
        capability,
        SyntheticScienceProvider(args.scenario),
        _spec(args.transaction_id, args.date, args.release_sha),
        fault_hook=hook,
    )
    print(report.to_json(indent=None))
    return 0


def _run_worker(command: Sequence[str], timeout: float = 60.0) -> Mapping[str, Any]:
    completed = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise CanaryError(
            f"Worker exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise CanaryError("Worker did not emit one JSON report.") from exc
    if not isinstance(value, dict):
        raise CanaryError("Worker JSON report is not an object.")
    return value


def _wait_for_path(path: Path, processes: Iterable[subprocess.Popen], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if any(process.poll() is not None for process in processes):
            raise CanaryError("Worker exited before its readiness marker.")
        if time.monotonic() >= deadline:
            raise CanaryError("Timed out waiting for worker readiness.")
        time.sleep(0.02)


def _inventory(root: Path) -> list[Mapping[str, Any]]:
    root_identity = root.lstat()
    values = [
        {
            "path": ".",
            "kind": "directory",
            "device": root_identity.st_dev,
            "inode": root_identity.st_ino,
            "mode": stat.S_IMODE(root_identity.st_mode),
            "uid": root_identity.st_uid,
            "gid": root_identity.st_gid,
            "size": None,
            "links": root_identity.st_nlink,
            "sha256": None,
        }
    ]
    for path in sorted(root.rglob("*")):
        observed = path.lstat()
        if stat.S_ISLNK(observed.st_mode):
            raise CanaryError(f"Canary inventory found a symlink: {path}.")
        if stat.S_ISDIR(observed.st_mode):
            kind, digest, size = "directory", None, None
        elif stat.S_ISREG(observed.st_mode):
            kind = "file"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            size = observed.st_size
        else:
            raise CanaryError(f"Canary inventory found an unsupported object: {path}.")
        values.append(
            {
                "path": path.relative_to(root).as_posix(),
                "kind": kind,
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "mode": stat.S_IMODE(observed.st_mode),
                "uid": observed.st_uid,
                "gid": observed.st_gid,
                "size": size,
                "links": observed.st_nlink,
                "sha256": digest,
            }
        )
    return values


_SAFE_COMPONENT_PATTERN = r"[A-Za-z0-9._-]+"
_STAGE_ID_PATTERN = r"writer-[0-9a-f]{24}"
_LOCK_PATTERN = _STAGE_ID_PATTERN + r"\.lock"
_DATE_TREE_PATTERN = r"[0-9]{4}/[0-9]{2}/[0-9]{2}"
_DATE_PATTERN = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"


def _synthetic_inventory_path_allowed(item: Mapping[str, Any]) -> bool:
    path = str(item.get("path"))
    kind = item.get("kind")
    if path == ".":
        return kind == "directory"
    directory_patterns = (
        r"published",
        r"published/data",
        r"published/data/lsst_only",
        r"published/data/lsst_only/nightly",
        r"published/data/lsst_only/nightly/[0-9]{4}",
        r"published/data/lsst_only/nightly/[0-9]{4}/[0-9]{2}",
        rf"published/data/lsst_only/nightly/{_DATE_TREE_PATTERN}",
        r"published/derived",
        r"published/derived/nightly",
        r"staging",
        rf"staging/{_SAFE_COMPONENT_PATTERN}",
        rf"staging/{_SAFE_COMPONENT_PATTERN}/{_STAGE_ID_PATTERN}",
        rf"staging/{_SAFE_COMPONENT_PATTERN}/reconciliation",
        r"control",
        r"control/journals",
        r"control/locks",
        rf"control/locks/{_LOCK_PATTERN}",
        r"evidence",
    )
    file_patterns = (
        re.escape(CANARY_IDENTITY_NAME),
        rf"control/journals/{_SAFE_COMPONENT_PATTERN}\.json",
        rf"control/locks/{_LOCK_PATTERN}/owner\.json",
        rf"staging/{_SAFE_COMPONENT_PATTERN}/{_STAGE_ID_PATTERN}/(loci\.parquet|alerts\.parquet|manifest\.json)",
        rf"staging/{_SAFE_COMPONENT_PATTERN}/reconciliation/nightly-summary\.json",
        rf"published/data/lsst_only/nightly/{_DATE_TREE_PATTERN}/(loci\.parquet|alerts\.parquet|manifest\.json|\.manifest\.pending|foreign\.marker)",
        rf"published/derived/nightly/{_DATE_PATTERN}\.json",
    )
    patterns = directory_patterns if kind == "directory" else file_patterns
    return any(re.fullmatch(pattern, path) is not None for pattern in patterns)


def _inventory_contract(
    inventory: Sequence[Mapping[str, Any]], root_device: int
) -> Mapping[str, bool]:
    inode_counts: Dict[tuple[int, int], int] = {}
    link_counts: Dict[tuple[int, int], int] = {}
    private = True
    same_device = True
    allowlisted = True
    for item in inventory:
        same_device = same_device and item.get("device") == root_device
        private = private and item.get("uid") == os.geteuid()
        private = private and item.get("mode") == (
            0o700 if item.get("kind") == "directory" else 0o600
        )
        allowlisted = allowlisted and _synthetic_inventory_path_allowed(item)
        if item.get("kind") == "file":
            identity = (int(item["device"]), int(item["inode"]))
            inode_counts[identity] = inode_counts.get(identity, 0) + 1
            link_counts[identity] = int(item["links"])
    closed_hardlinks = all(
        inode_counts[identity] == links
        for identity, links in link_counts.items()
    )
    return {
        "private_user_owned_modes": private,
        "single_device_tree": same_device,
        "synthetic_path_allowlist": allowlisted,
        "hardlink_set_closed_within_run": closed_hardlinks,
    }


def qualify(run_root: Path, run_id: str, release_sha: str) -> Mapping[str, Any]:
    """Run the fixed one-shot installed-package synthetic qualification."""
    capability = SyntheticWriteCapability.for_arnor_canary_root(run_root, run_id)
    bootstrap_identity = _read_root_identity(capability.root, capability.run_id)
    _assert_pristine_run_root(capability.root)
    _assert_no_nested_mounts(capability.root)
    if (
        len(release_sha) != 40
        or release_sha != release_sha.lower()
        or any(character not in "0123456789abcdef" for character in release_sha)
    ):
        raise CanaryError("release_sha must be one lowercase 40-character Git SHA.")
    results: Dict[str, Any] = {}

    def execute_case(case: str, date_utc: str, scenario: str, fault: Optional[str] = None):
        transaction_id = f"{run_id}-{case}"
        report = execute_synthetic_night(
            capability,
            SyntheticScienceProvider(scenario),
            _spec(transaction_id, date_utc, release_sha),
            fault_hook=FailureInjector(fault) if fault else None,
        )
        assessment = _assessment(capability, transaction_id)
        results[case] = {
            "report": report.as_dict(),
            "recovery": assessment.as_dict(),
        }
        return report, assessment

    golden, golden_recovery = execute_case("C01-golden", "2098-01-01", "success_nonzero")
    zero, _ = execute_case("C02-zero", "2098-01-02", "success_zero")
    provider_failures = {}
    for scenario in (
        "query_failure",
        "query_interruption",
        "fetch_failure",
        "partial_fetch",
        "malformed_result",
        "validation_failure",
    ):
        case = f"C03-provider-{scenario.replace('_', '-')}"
        report, assessment = execute_case(case, "2098-01-03", scenario)
        provider_failures[scenario] = {
            "report": report,
            "assessment": assessment,
        }
    uncertain, uncertain_recovery = execute_case(
        "C05-postcommit", "2098-01-05", "success_nonzero", "after_manifest_commit"
    )
    reconciliation, reconciliation_recovery = execute_case(
        "C06-reconcile", "2098-01-06", "success_nonzero", "during_reconciliation"
    )
    golden_journal = TransactionJournal.load(
        capability.journal_root / f"{run_id}-C01-golden.json"
    ).snapshot
    reconciliation_replay = SyntheticReconciler().reconcile(
        capability,
        date_utc="2098-01-01",
        published_artifacts=golden_journal.artifacts,
        transaction_id=f"{run_id}-C01-replay",
    )
    results["C01-reconciliation-replay"] = {
        "state": reconciliation_replay.state,
        "path": reconciliation_replay.path,
        "sha256": reconciliation_replay.sha256,
        "idempotent_replay": reconciliation_replay.idempotent_replay,
    }

    late_target_path: Optional[Path] = None

    def late_target(point: str, details: Mapping[str, Any]) -> None:
        nonlocal late_target_path
        if point == "before_target_reservation":
            target = Path(str(details["target"]))
            target.mkdir(mode=0o700)
            target_fd = os.open(
                str(target),
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fchmod(target_fd, 0o700)
                os.fsync(target_fd)
            finally:
                os.close(target_fd)
            parent_fd = os.open(
                str(target.parent),
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            _write_exclusive_private(
                target / "foreign.marker", b"synthetic-late-target\n"
            )
            late_target_path = target

    late_id = f"{run_id}-C07-late"
    late = execute_synthetic_night(
        capability,
        SyntheticScienceProvider(),
        _spec(late_id, "2098-01-07", release_sha),
        fault_hook=late_target,
    )
    late_recovery = _assessment(capability, late_id)
    results["C07-late"] = {
        "report": late.as_dict(),
        "recovery": late_recovery.as_dict(),
        "foreign_marker_preserved": bool(
            late_target_path is not None
            and (late_target_path / "foreign.marker").is_file()
        ),
    }

    death_cases = (
        ("C08-death-lock", "2098-01-08", "after_lock"),
        ("C09-death-validated", "2098-01-09", "after_validation"),
        ("C10-death-commit", "2098-01-10", "after_manifest_commit"),
    )
    for case, date_utc, point in death_cases:
        transaction_id = f"{run_id}-{case}"
        completed = subprocess.run(
            _worker_command(
                run_root=run_root,
                run_id=run_id,
                transaction_id=transaction_id,
                date_utc=date_utc,
                release_sha=release_sha,
                death_point=point,
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        if completed.returncode != 91:
            raise CanaryError(f"Death worker {case} exited {completed.returncode}.")
        assessment = _assessment(capability, transaction_id)
        results[case] = {
            "exit_code": completed.returncode,
            "recovery": assessment.as_dict(),
        }

    _ensure_private_tree(capability.evidence_root, capability.root)
    ready = capability.evidence_root / "C11-lock-ready"
    release = capability.evidence_root / "C11-lock-release"
    winner_id = f"{run_id}-C11-winner"
    winner = subprocess.Popen(
        _worker_command(
            run_root=run_root,
            run_id=run_id,
            transaction_id=winner_id,
            date_utc="2098-01-11",
            release_sha=release_sha,
            hold_ready=ready,
            hold_release=release,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    losers = []
    contention_processes = [winner]
    try:
        _wait_for_path(ready, (winner,), 30)
        active_recovery = _assessment(capability, winner_id)
        for index in range(1, 8):
            process = subprocess.Popen(
                _worker_command(
                    run_root=run_root,
                    run_id=run_id,
                    transaction_id=f"{run_id}-C11-loser-{index}",
                    date_utc="2098-01-11",
                    release_sha=release_sha,
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            losers.append(process)
            contention_processes.append(process)
        loser_reports = []
        for process in losers:
            stdout, stderr = process.communicate(timeout=60)
            if process.returncode != 0:
                raise CanaryError(f"Contention loser failed: {stderr[-1000:]}")
            loser_reports.append(json.loads(stdout))
        _write_exclusive_private(release, b"release\n")
        winner_stdout, winner_stderr = winner.communicate(timeout=60)
        if winner.returncode != 0:
            raise CanaryError(f"Contention winner failed: {winner_stderr[-1000:]}")
        winner_report = json.loads(winner_stdout)
    finally:
        _stop_processes(contention_processes)
    _unlink_private_marker(ready)
    _unlink_private_marker(release)
    results["C11-contention"] = {
        "winner": winner_report,
        "losers": loser_reports,
        "active_recovery": active_recovery.as_dict(),
        "success_count": int(winner_report.get("success", False))
        + sum(int(item.get("success", False)) for item in loser_reports),
    }

    parallel = []
    try:
        for offset in range(4):
            parallel.append(
                subprocess.Popen(
                    _worker_command(
                        run_root=run_root,
                        run_id=run_id,
                        transaction_id=f"{run_id}-C12-parallel-{offset}",
                        date_utc=f"2098-01-{12 + offset:02d}",
                        release_sha=release_sha,
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        parallel_reports = []
        for process in parallel:
            stdout, stderr = process.communicate(timeout=60)
            if process.returncode != 0:
                raise CanaryError(f"Parallel worker failed: {stderr[-1000:]}")
            parallel_reports.append(json.loads(stdout))
    finally:
        _stop_processes(parallel)
    results["C12-parallel"] = {"reports": parallel_reports}

    all_recovery = {}
    for journal_path in sorted(capability.journal_root.glob("*.json")):
        transaction_id = journal_path.stem
        try:
            all_recovery[transaction_id] = _assessment(
                capability, transaction_id
            ).as_dict()
        except Exception as exc:
            all_recovery[transaction_id] = {"error": f"{type(exc).__name__}: {exc}"}
    results["C13-recovery-scan"] = all_recovery

    death_expectations = {
        "C08-death-lock": (
            "STALE",
            RecoveryDisposition.REQUIRES_OPERATOR_DECISION.value,
        ),
        "C09-death-validated": (
            "STALE",
            RecoveryDisposition.REQUIRES_REVALIDATION.value,
        ),
        "C10-death-commit": (
            "STALE",
            RecoveryDisposition.MUST_NOT_AUTO_DELETE.value,
        ),
    }
    death_classifications_pass = all(
        results[case]["exit_code"] == 91
        and results[case]["recovery"]["lock_owner_status"] == expected_lock
        and expected_disposition
        in results[case]["recovery"]["dispositions"]
        for case, (expected_lock, expected_disposition) in death_expectations.items()
    )
    unexpected_top_level = sorted(
        path.name
        for path in MIDDLE_EARTH_PROJECT_ROOT.parent.glob("ANTARES_*")
    )
    _assert_no_nested_mounts(run_root)
    ending_bootstrap_identity = _read_root_identity(run_root, run_id)
    inventory = _inventory(run_root)
    inventory_checks = _inventory_contract(
        inventory, int(bootstrap_identity["device"])
    )
    checks = {
        "bootstrap_identity_stable": (
            ending_bootstrap_identity == bootstrap_identity
        ),
        "golden_complete": golden.success,
        "golden_precommit_reproved": (
            golden_journal.validation.get("precommit_reproved") is True
        ),
        "golden_independent_reopen": (
            golden.details.get("independent_reopen") is True
        ),
        "golden_recovery_preserves_science": (
            RecoveryDisposition.MUST_NOT_AUTO_DELETE.value
            in golden_recovery.as_dict()["dispositions"]
        ),
        "zero_complete_with_positive_proof": (
            zero.success
            and zero.counts.get("loci") == 0
            and zero.counts.get("alerts") == 0
        ),
        "all_provider_failures_unpublished": all(
            not item["report"].success
            and item["report"].details.get("published") is False
            and item["report"].details.get("journal_outcome")
            == "UNPUBLISHED_FAILURE"
            for item in provider_failures.values()
        ),
        "postcommit_preserved": (
            not uncertain.success
            and uncertain.details.get("published") is True
            and uncertain.details.get("journal_outcome")
            == "PUBLISHED_DURABILITY_UNCERTAIN"
            and "MUST_NOT_AUTO_DELETE"
            in uncertain_recovery.as_dict()["dispositions"]
        ),
        "reconciliation_independent": (
            not reconciliation.success
            and reconciliation.details.get("published") is True
            and reconciliation.details.get("journal_outcome")
            == "PUBLISHED_RECONCILIATION_REQUIRED"
            and "REQUIRES_RECONCILIATION"
            in reconciliation_recovery.as_dict()["dispositions"]
        ),
        "reconciliation_replay_idempotent": (
            reconciliation_replay.idempotent_replay
        ),
        "late_target_preserved": (
            not late.success
            and results["C07-late"]["foreign_marker_preserved"]
            and late_target_path is not None
            and not (late_target_path / "manifest.json").exists()
        ),
        "process_death_classified_from_disk": death_classifications_pass,
        "active_lock_inspection_safe": (
            active_recovery.lock_owner_status == "ACTIVE"
            and RecoveryDisposition.MUST_NOT_AUTO_DELETE
            in active_recovery.dispositions
        ),
        "same_target_one_winner": (
            results["C11-contention"]["success_count"] == 1
            and all(not item.get("success") for item in loser_reports)
        ),
        "different_targets_all_succeeded": all(
            item.get("success") for item in parallel_reports
        ),
        "recovery_scan_cleanly_executed": all(
            "error" not in item for item in all_recovery.values()
        ),
        "no_live_provider": True,
        "canonical_cache_absent": (
            not MIDDLE_EARTH_CACHE_ROOT.exists()
            and not MIDDLE_EARTH_CACHE_ROOT.is_symlink()
        ),
        "no_extra_antares_top_level": not unexpected_top_level,
        **inventory_checks,
    }
    result = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "run_id": run_id,
        "run_root": str(run_root),
        "release_sha": release_sha,
        "host": socket.gethostname(),
        "python": sys.version,
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "bootstrap_identity": bootstrap_identity,
        "unexpected_antares_top_level": unexpected_top_level,
        "checks": checks,
        "passed": all(checks.values()),
        "cases": results,
    }
    result["inventory"] = inventory
    result["inventory_sha256"] = hashlib.sha256(
        json.dumps(result["inventory"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def _load_external_evidence(evidence_path: Path, run_root: Path) -> Mapping[str, Any]:
    if evidence_path.is_symlink():
        raise CanaryError("External qualification evidence must not be a symlink.")
    try:
        resolved = evidence_path.resolve(strict=True)
        canonical_root = run_root.resolve(strict=False)
    except OSError as exc:
        raise CanaryError("External qualification evidence is inaccessible.") from exc
    try:
        resolved.relative_to(canonical_root)
    except ValueError:
        pass
    else:
        raise CanaryError("Qualification evidence must be outside the cleanup target.")
    descriptor = os.open(
        str(resolved),
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.geteuid():
            raise CanaryError("External evidence is not a user-owned regular file.")
        if observed.st_size > 32 * 1024 * 1024:
            raise CanaryError("External evidence exceeds the cleanup safety limit.")
        chunks = []
        remaining = 32 * 1024 * 1024 + 1
        while remaining > 0:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if len(payload) > 32 * 1024 * 1024:
        raise CanaryError("External evidence exceeds the cleanup safety limit.")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CanaryError("External evidence is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise CanaryError("External evidence must be one JSON object.")
    return value


def _assert_no_active_or_ambiguous_lock(run_root: Path) -> None:
    lock_root = run_root / "control" / "locks"
    if not lock_root.exists():
        return
    local_host = socket.gethostname().split(".", 1)[0].lower()
    for lock_path in sorted(lock_root.iterdir()):
        if (
            lock_path.is_symlink()
            or not lock_path.is_dir()
            or re.fullmatch(_LOCK_PATTERN, lock_path.name) is None
            or {item.name for item in lock_path.iterdir()} != {"owner.json"}
        ):
            raise CanaryError("Canary cleanup found an ambiguous lock directory.")
        metadata_path = lock_path / "owner.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise CanaryError("Canary cleanup found ambiguous lock metadata.")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            hostname = metadata["hostname"]
            pid = metadata["pid"]
            recorded_start = metadata["process_start_identity"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CanaryError("Canary cleanup found ambiguous lock metadata.") from exc
        if (
            not isinstance(hostname, str)
            or hostname.split(".", 1)[0].lower() != local_host
            or type(pid) is not int
            or pid <= 0
            or not isinstance(recorded_start, str)
            or not recorded_start
        ):
            raise CanaryError("Canary cleanup found an ambiguous lock owner.")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except (PermissionError, OSError) as exc:
            raise CanaryError("Canary lock liveness is ambiguous.") from exc
        observed_start = _process_start_identity(pid, hostname)
        if observed_start.startswith("unavailable:"):
            raise CanaryError("Canary lock process identity is ambiguous.")
        if observed_start == recorded_start:
            raise CanaryError("Canary cleanup refused an active writer lock.")


def _entry_matches(observed: os.stat_result, expected: Mapping[str, Any]) -> bool:
    return bool(
        observed.st_dev == expected.get("device")
        and observed.st_ino == expected.get("inode")
        and stat.S_IMODE(observed.st_mode) == expected.get("mode")
        and observed.st_uid == expected.get("uid")
        and observed.st_gid == expected.get("gid")
        and (
            expected.get("kind") == "directory"
            or observed.st_size == expected.get("size")
        )
    )


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _remove_inventoried_tree(run_root: Path, inventory: Sequence[Mapping[str, Any]]) -> None:
    entries = {str(item.get("path")): item for item in inventory}
    if len(entries) != len(inventory):
        raise CanaryError("Cleanup inventory contains duplicate paths.")
    children: Dict[str, list[Mapping[str, Any]]] = {}
    for path_text, item in entries.items():
        if path_text == ".":
            continue
        path = Path(path_text)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise CanaryError("Cleanup inventory contains an unsafe path.")
        parent = path.parent.as_posix() if path.parent != Path(".") else "."
        children.setdefault(parent, []).append(item)

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def remove_children(directory_fd: int, relative: str) -> None:
        expected_children = children.get(relative, [])
        expected_names = {Path(str(item["path"])).name for item in expected_children}
        if set(os.listdir(directory_fd)) != expected_names:
            raise CanaryError("Canary tree changed during descriptor-pinned cleanup.")
        files = sorted(
            (item for item in expected_children if item.get("kind") == "file"),
            key=lambda item: str(item["path"]),
        )
        directories = sorted(
            (item for item in expected_children if item.get("kind") == "directory"),
            key=lambda item: str(item["path"]),
        )
        if len(files) + len(directories) != len(expected_children):
            raise CanaryError("Cleanup inventory contains an unsupported object.")
        for item in files:
            name = Path(str(item["path"])).name
            descriptor = os.open(name, file_flags, dir_fd=directory_fd)
            try:
                observed = os.fstat(descriptor)
                if not stat.S_ISREG(observed.st_mode) or not _entry_matches(observed, item):
                    raise CanaryError("Canary file identity changed before cleanup.")
                if _hash_descriptor(descriptor) != item.get("sha256"):
                    raise CanaryError("Canary file content changed before cleanup.")
            finally:
                os.close(descriptor)
            os.unlink(name, dir_fd=directory_fd)
        for item in directories:
            name = Path(str(item["path"])).name
            descriptor = os.open(name, directory_flags, dir_fd=directory_fd)
            try:
                observed = os.fstat(descriptor)
                if not stat.S_ISDIR(observed.st_mode) or not _entry_matches(observed, item):
                    raise CanaryError("Canary directory identity changed before cleanup.")
                remove_children(descriptor, str(item["path"]))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.rmdir(name, dir_fd=directory_fd)
        os.fsync(directory_fd)

    root_descriptor = os.open(str(run_root), directory_flags)
    try:
        root_entry = entries.get(".")
        observed_root = os.fstat(root_descriptor)
        if root_entry is None or not _entry_matches(observed_root, root_entry):
            raise CanaryError("Canary root identity changed before cleanup.")
        remove_children(root_descriptor, ".")
    finally:
        os.close(root_descriptor)


def cleanup_exact(run_root: Path, run_id: str, evidence_path: Path) -> Mapping[str, Any]:
    """Delete only an exact inventoried canary run after external evidence seal."""
    capability = SyntheticWriteCapability.for_arnor_canary_root(run_root, run_id)
    bootstrap_identity = _read_root_identity(capability.root, capability.run_id)
    _assert_no_nested_mounts(capability.root)
    evidence = _load_external_evidence(evidence_path, capability.root)
    checks = evidence.get("checks")
    if (
        evidence.get("schema_version") != CANARY_SCHEMA_VERSION
        or evidence.get("run_id") != run_id
        or evidence.get("run_root") != str(run_root)
        or evidence.get("passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
        or evidence.get("bootstrap_identity") != bootstrap_identity
    ):
        raise CanaryError("External evidence does not authorize this exact cleanup.")
    expected = evidence.get("inventory")
    if not isinstance(expected, list) or not expected:
        raise CanaryError("External evidence lacks a complete cleanup inventory.")
    inventory_digest = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if inventory_digest != evidence.get("inventory_sha256"):
        raise CanaryError("External cleanup inventory digest is invalid.")
    if any(item.get("uid") != os.geteuid() for item in expected):
        raise CanaryError("Canary cleanup inventory has unexpected ownership.")
    if expected != _inventory(run_root):
        raise CanaryError("Current canary tree differs from its sealed inventory.")
    if not all(
        _inventory_contract(expected, int(bootstrap_identity["device"])).values()
    ):
        raise CanaryError("Canary cleanup inventory violates its synthetic contract.")
    root_entry = expected[0]
    current_root = run_root.lstat()
    if (
        root_entry.get("path") != "."
        or current_root.st_dev != root_entry.get("device")
        or current_root.st_ino != root_entry.get("inode")
    ):
        raise CanaryError("Canary root identity changed before cleanup.")
    _assert_no_active_or_ambiguous_lock(run_root)
    _remove_inventoried_tree(run_root, expected)
    parent_fd = os.open(
        str(MIDDLE_EARTH_CANARY_ROOT),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_dev != root_entry.get("device")
            or observed.st_ino != root_entry.get("inode")
        ):
            raise CanaryError("Canary root changed before final removal.")
        os.rmdir(run_id, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            os.stat(run_id, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            removed = True
        else:
            removed = False
    finally:
        os.close(parent_fd)
    if not removed:
        raise CanaryError("Canary root remains present after exact cleanup.")
    if _load_external_evidence(evidence_path, run_root) != evidence:
        raise CanaryError("External cleanup evidence changed during cleanup.")
    return {
        "run_id": run_id,
        "run_root": str(run_root),
        "removed": removed,
        "objects_removed": len(expected),
        "evidence_preserved": str(evidence_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.operations.canary")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--run-id", required=True)
    qualify_parser = commands.add_parser("qualify")
    qualify_parser.add_argument("--run-root", type=Path, required=True)
    qualify_parser.add_argument("--run-id", required=True)
    qualify_parser.add_argument("--release-sha", required=True)
    cleanup = commands.add_parser("cleanup")
    cleanup.add_argument("--run-root", type=Path, required=True)
    cleanup.add_argument("--run-id", required=True)
    cleanup.add_argument("--evidence", type=Path, required=True)
    worker = commands.add_parser("_worker")
    worker.add_argument("--run-root", type=Path, required=True)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--transaction-id", required=True)
    worker.add_argument("--date", required=True)
    worker.add_argument("--release-sha", required=True)
    worker.add_argument("--scenario", default="success_nonzero")
    worker.add_argument("--death-point")
    worker.add_argument("--hold-ready")
    worker.add_argument("--hold-release")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            result = bootstrap_run_root(args.run_id)
        elif args.command == "qualify":
            result = qualify(args.run_root, args.run_id, args.release_sha)
        elif args.command == "cleanup":
            result = cleanup_exact(
                args.run_root, args.run_id, args.evidence
            )
        elif args.command == "_worker":
            return _worker(args)
        else:
            raise CanaryError("Unknown canary command.")
    except Exception as exc:
        print(
            _json(
                {
                    "schema_version": CANARY_SCHEMA_VERSION,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        )
        return 1
    print(_json(result))
    return 0 if result.get("passed", result.get("removed", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
