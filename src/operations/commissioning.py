"""Phase 6 live commissioning orchestration with publication unavailable.

This module reuses the Phase 5 lock, journal, state machine, transaction,
artifact builder, validator, and pre-commit reproof.  It deliberately stops at
``VALIDATED`` and releases the run-local writer lock without calling
``PublicationTransaction.publish``.  The only write capability is the existing
sealed canary capability rooted beneath one exact Shire run directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from importlib import metadata as distribution_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from astropy.time import Time

from .. import history, query as historical_query
from ..cli_profiles import MIDDLE_EARTH_CACHE_ROOT, MIDDLE_EARTH_DATA_ROOT
from .journal import ArtifactIdentity, TransactionDescriptor, TransactionJournal
from .live_antares import (
    LiveAntaresProvider,
    LiveAntaresReadCapability,
    extraction_method_contract,
)
from .locking import WriterLock
from .report import Artifact, Evidence, ExitCode, Issue, OperationReport
from .science import (
    NightScienceRequest,
    ProviderError,
    build_night_artifacts,
    reopen_and_validate_artifacts,
)
from .state import ExecutionState
from .storage import SyntheticWriteCapability, contained_path
from .transaction import PublicationTransaction
from .writer import (
    EXPECTED_ARTIFACTS,
    NightExecutionSpec,
    WriterError,
    _ensure_private_tree,
    _read_nofollow,
    nightly_target_relative,
)


COMMISSIONING_SCHEMA_VERSION = "phase6.live-commissioning.v1"
TARGET_DATE_UTC = "2026-06-27"
ACCEPTED_PREDECESSOR_DATE = "2026-06-26"
ACCEPTED_BASELINE = {
    "night_count": 90,
    "first_date": "2026-02-25",
    "last_date": "2026-06-26",
    "total_loci": 993218,
    "total_alerts": 13579707,
    "zero_row_nights": ("2026-03-05", "2026-03-11"),
    "durable_file_count": 324,
    "durable_bytes": 1141241743,
    "checksum_manifest_sha256": "594ced3dfc7ed2d36f1e5562defabb6e9475f2a72f6ebeab54c09020f6db5908",
    "loci_index_sha256": "f75196d18690e610ab6e79231b244c3fddca396a68eea08dd2d0408e91d8b587",
    "nightly_summary_sha256": "85c5fac9c242fa2e7993155036ada649336b0affe8ffc8843d2c5733ea765114",
}


class CommissioningError(WriterError):
    """A Phase 6 commissioning gate failed closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _json_bytes(value: Mapping[str, Any], *, pretty: bool = True) -> bytes:
    text = json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=True,
        separators=None if pretty else (",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json_bytes(value, pretty=False)).hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sequence_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_bytes(value)
    parent_fd = os.open(
        str(path.parent),
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    temporary = f".{path.name}.tmp-{os.getpid()}"
    created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        created = True
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise OSError("Short commissioning evidence write.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        created = False
        os.fsync(parent_fd)
    finally:
        if created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def _file_sha256(path: Path) -> str:
    descriptor = os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CommissioningError(f"Evidence object is not regular: {path}.")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CommissioningError(f"Evidence object changed while hashing: {path}.")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _regular_identity(name: str, path: Path) -> ArtifactIdentity:
    return ArtifactIdentity.from_path(name, path)


def _artifact_identities(stage: Path) -> Mapping[str, ArtifactIdentity]:
    return {
        name: _regular_identity(name, stage / name)
        for name in EXPECTED_ARTIFACTS
    }


def _artifact_hashes(values: Mapping[str, ArtifactIdentity]) -> Mapping[str, str]:
    return {name: values[name].sha256 for name in sorted(values)}


def _production_target(data_root: Path, target_date_utc: str) -> Path:
    return history.nightly_paths(data_root, target_date_utc)["dir"]


def _manifest_paths(data_root: Path) -> Sequence[Path]:
    root = history.survey_data_root(data_root) / "nightly"
    if root.is_symlink() or not root.is_dir():
        raise CommissioningError("Authoritative nightly root is missing or unsafe.")
    return sorted(root.glob("*/*/*/manifest.json"))


def establish_target_eligibility(
    data_root: Path,
    target_date_utc: str = TARGET_DATE_UTC,
) -> Mapping[str, Any]:
    """Prove the intended target is the next chronological accepted night."""
    if target_date_utc != TARGET_DATE_UTC:
        raise CommissioningError(
            f"Phase 6 is restricted to {TARGET_DATE_UTC}; alternate target refused."
        )
    parsed = date.fromisoformat(target_date_utc)
    predecessor = (parsed - timedelta(days=1)).isoformat()
    if predecessor != ACCEPTED_PREDECESSOR_DATE:
        raise CommissioningError("The target predecessor is not the accepted endpoint.")
    mjd_min = float(Time(f"{target_date_utc}T00:00:00", format="isot", scale="utc").mjd)
    end_date = (parsed + timedelta(days=1)).isoformat()
    mjd_max = float(Time(f"{end_date}T00:00:00", format="isot", scale="utc").mjd)
    if mjd_max - mjd_min != 1.0:
        raise CommissioningError("Target UTC interval is not exactly one MJD day.")

    manifests = _manifest_paths(data_root)
    rows = []
    for path in manifests:
        if path.is_symlink() or not path.is_file():
            raise CommissioningError(f"Authoritative manifest is unsafe: {path}.")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CommissioningError(f"Authoritative manifest is unreadable: {path}.") from exc
        if not isinstance(value, dict):
            raise CommissioningError(f"Authoritative manifest is not an object: {path}.")
        path_date = "-".join(path.parent.parts[-3:])
        if value.get("date_utc") != path_date:
            raise CommissioningError(
                f"Authoritative manifest date disagrees with its partition: {path}."
            )
        rows.append(value)
    try:
        dates = sorted(
            date.fromisoformat(str(item.get("date_utc"))).isoformat() for item in rows
        )
        total_loci = sum(int(item["actual_loci"]) for item in rows)
        total_alerts = sum(int(item["alert_rows"]) for item in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise CommissioningError("Authoritative manifest inventory is malformed.") from exc
    if len(dates) != ACCEPTED_BASELINE["night_count"]:
        raise CommissioningError("Authoritative night count differs from the accepted baseline.")
    if len(set(dates)) != len(dates):
        raise CommissioningError("Authoritative manifest dates are duplicated.")
    if not dates or dates[0] != ACCEPTED_BASELINE["first_date"] or dates[-1] != predecessor:
        raise CommissioningError("Authoritative date range differs from the accepted baseline.")
    if (
        total_loci != ACCEPTED_BASELINE["total_loci"]
        or total_alerts != ACCEPTED_BASELINE["total_alerts"]
    ):
        raise CommissioningError("Authoritative scientific totals differ from the baseline.")
    zero_row_nights = sorted(
        str(item["date_utc"])
        for item in rows
        if int(item["actual_loci"]) == 0 and int(item["alert_rows"]) == 0
    )
    if zero_row_nights != list(ACCEPTED_BASELINE["zero_row_nights"]):
        raise CommissioningError("Accepted zero-row-night evidence differs from the baseline.")
    if target_date_utc in dates:
        raise CommissioningError("The Phase 6 target already has an authoritative manifest.")
    predecessor_rows = [item for item in rows if item.get("date_utc") == predecessor]
    if len(predecessor_rows) != 1:
        raise CommissioningError("The accepted predecessor manifest is missing or duplicated.")
    previous = predecessor_rows[0]
    if (
        float(previous.get("mjd_max")) != mjd_min
        or float(previous.get("mjd_min")) != mjd_min - 1.0
        or previous.get("lsst_filter_used") is not True
        or previous.get("lsst_filter") != historical_query.lsst_identifier_filter()
        or previous.get("query_tag") is not None
        or previous.get("target_loci") is not None
        or previous.get("parallel_shards") != 1
        or previous.get("status") != "complete"
        or previous.get("saturated_chunk_count") != 0
        or previous.get("extraction_method") != extraction_method_contract()
    ):
        raise CommissioningError("Target semantics disagree with the accepted predecessor.")
    target = _production_target(data_root, target_date_utc)
    if target.exists() or target.is_symlink():
        raise CommissioningError("The authoritative production target is not absent.")

    summary_path = history.cumulative_paths(data_root)["nightly_summary"]
    summary = pd.read_parquet(summary_path)
    if "date_utc" not in summary.columns:
        raise CommissioningError("Cumulative nightly summary lacks date_utc.")
    summary_dates = summary["date_utc"].astype(str)
    if len(summary) != ACCEPTED_BASELINE["night_count"] or (summary_dates == target_date_utc).any():
        raise CommissioningError("Cumulative nightly state makes the target ineligible.")
    if summary_dates.max() != predecessor:
        raise CommissioningError("Cumulative nightly state does not end at the predecessor.")
    if sorted(summary_dates.tolist()) != dates:
        raise CommissioningError("Cumulative and nightly accepted date sets disagree.")
    return {
        "passed": True,
        "target_date_utc": target_date_utc,
        "utc_start": f"{target_date_utc}T00:00:00+00:00",
        "utc_end": f"{end_date}T00:00:00+00:00",
        "mjd_min": mjd_min,
        "mjd_max": mjd_max,
        "lower_bound": "inclusive",
        "upper_bound": "exclusive",
        "timezone": "UTC",
        "historical_filter": "properties.newest_alert_observation_time",
        "lsst_only": True,
        "lsst_filter": previous.get("lsst_filter"),
        "query_tag": None,
        "target_loci": None,
        "parallel_parent_shards": 1,
        "accepted_extraction_method": previous.get("extraction_method"),
        "accepted_predecessor": predecessor,
        "authoritative_manifest_count": len(manifests),
        "total_loci": total_loci,
        "total_alerts": total_alerts,
        "zero_row_nights": zero_row_nights,
        "target_directory_absent": True,
        "cumulative_target_absent": True,
    }


def resource_preflight(
    run_root: Path,
    data_root: Path,
    predecessor_date: str = ACCEPTED_PREDECESSOR_DATE,
    provider_policy: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Bound capacity risk using the newest accepted real partition."""
    filesystem = os.statvfs(run_root)
    free_bytes = int(filesystem.f_bavail * filesystem.f_frsize)
    free_inodes = int(filesystem.f_favail)
    predecessor = history.nightly_paths(data_root, predecessor_date)["dir"]
    durable_sizes = {}
    for name in ("loci.parquet", "alerts.parquet", "manifest.json"):
        path = predecessor / name
        if path.is_symlink() or not path.is_file():
            raise CommissioningError(f"Predecessor artifact is missing or unsafe: {path}.")
        durable_sizes[name] = path.stat().st_size
    predecessor_bytes = sum(durable_sizes.values())
    manifest = json.loads(
        (predecessor / "manifest.json").read_text(encoding="utf-8")
    )
    if not isinstance(manifest, dict):
        raise CommissioningError("Predecessor manifest is not a JSON object.")
    predecessor_loci = int(manifest.get("actual_loci", 0))
    predecessor_alerts = int(manifest.get("alert_rows", 0))
    predecessor_runtime = manifest.get("runtime_seconds")
    # Query frames, per-locus fetch frames, in-memory serialization, and staged
    # output coexist transiently.  Disk pressure is bounded at four times the
    # newest durable partition plus one GiB of evidence/headroom.
    required_free_bytes = predecessor_bytes * 4 + 1024 ** 3
    required_free_inodes = 10000
    available_memory = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                available_memory = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError, IndexError):
        try:
            available_memory = int(os.sysconf("SC_AVPHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (OSError, ValueError):
            available_memory = 0
    estimated_peak_memory = max(predecessor_bytes * 8, 512 * 1024 ** 2)
    if (
        free_bytes < required_free_bytes
        or free_inodes < required_free_inodes
        or (available_memory and available_memory < estimated_peak_memory)
    ):
        raise CommissioningError("Shire capacity preflight did not pass.")
    quota_binary = shutil.which("quota")
    quota_status = "command_unavailable"
    if quota_binary:
        try:
            quota_result = subprocess.run(
                [quota_binary, "-s"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            quota_status = (
                "visible_command_passed"
                if quota_result.returncode == 0
                else f"command_exit_{quota_result.returncode}"
            )
        except (OSError, subprocess.SubprocessError):
            quota_status = "command_failed"
    return {
        "passed": True,
        "filesystem_device": run_root.stat().st_dev,
        "free_bytes": free_bytes,
        "free_inodes": free_inodes,
        "quota_visibility": quota_status,
        "predecessor_date": predecessor_date,
        "predecessor_durable_bytes": predecessor_bytes,
        "predecessor_artifact_bytes": durable_sizes,
        "required_free_bytes": required_free_bytes,
        "required_free_inodes": required_free_inodes,
        "available_memory_bytes": available_memory or None,
        "estimated_peak_memory_bytes": estimated_peak_memory,
        "predecessor_loci": predecessor_loci,
        "predecessor_alerts": predecessor_alerts,
        "predecessor_runtime_seconds": predecessor_runtime,
        "estimated_target_loci": predecessor_loci,
        "estimated_target_alerts": predecessor_alerts,
        "estimated_intermediate_bytes": predecessor_bytes * 3,
        "estimated_staged_bytes": predecessor_bytes,
        "runtime_expectation": "same order as the accepted predecessor; bounded by per-request client timeouts",
        "provider_policy": dict(provider_policy or {}),
        "network_query_state": "memory_plus-client-pagination",
        "temporary_state": "exact-run-root-only",
        "validated_stage": "exact-run-root-only",
        "future_cache": "absent-and-out-of-scope",
    }


def release_environment_preflight(
    write_capability: SyntheticWriteCapability,
    provider: LiveAntaresProvider,
    spec: NightExecutionSpec,
) -> Mapping[str, Any]:
    """Bind the commissioning request to one installed candidate environment."""
    import antares_client

    from . import live_antares as live_module
    from . import science as science_module

    module_origins = {
        "commissioning": str(Path(__file__).resolve(strict=True)),
        "live_antares": str(Path(live_module.__file__).resolve(strict=True)),
        "science": str(Path(science_module.__file__).resolve(strict=True)),
        "antares_client": str(Path(antares_client.__file__).resolve(strict=True)),
    }
    try:
        package_version = distribution_metadata.version("antares-analysis")
    except distribution_metadata.PackageNotFoundError:
        package_version = None
    value: Dict[str, Any] = {
        "passed": True,
        "release_sha": spec.release_sha,
        "package": "antares-analysis",
        "package_version": package_version,
        "python_version": platform.python_version(),
        "python_executable": os.path.abspath(sys.executable),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "module_origins": module_origins,
        "client": provider.client_identity(),
        "execution_policy": provider.execution_policy(),
        "configuration_identity": spec.configuration_identity,
        "wheel_path": None,
        "wheel_sha256": None,
        "imports_from_source_checkout": write_capability.environment != "arnor-canary",
    }
    if write_capability.environment == "arnor-canary":
        if package_version != "0.4.0":
            raise CommissioningError("Phase 6 requires installed antares-analysis 0.4.0.")
        release_root = Path(
            f"/astro/users/mdarim/opt/antares-analysis/releases/{spec.release_sha}"
        )
        if release_root.is_symlink() or not release_root.is_dir():
            raise CommissioningError("The exact immutable Phase 6 release is missing.")
        lexical_executable = Path(os.path.abspath(sys.executable))
        try:
            lexical_executable.relative_to(release_root)
        except ValueError as exc:
            raise CommissioningError("CLI Python is outside the exact candidate release.") from exc
        for label, origin in module_origins.items():
            try:
                Path(origin).relative_to(release_root)
            except ValueError as exc:
                raise CommissioningError(
                    f"Installed module {label} is outside the exact candidate release."
                ) from exc
        wheels = sorted((release_root / "artifacts").glob("*.whl"))
        if len(wheels) != 1 or wheels[0].is_symlink() or not wheels[0].is_file():
            raise CommissioningError("The immutable release must retain exactly one wheel.")
        value.update(
            {
                "release_root": str(release_root),
                "wheel_path": str(wheels[0]),
                "wheel_sha256": _file_sha256(wheels[0]),
                "imports_from_source_checkout": False,
            }
        )
    return value


def capture_production_sentinel(
    data_root: Path,
    cache_root: Path,
    target_date_utc: str = TARGET_DATE_UTC,
) -> Mapping[str, Any]:
    """Capture an application-independent metadata mutation tripwire."""
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    if data_root.is_symlink() or not data_root.is_dir():
        raise CommissioningError("Production data root is missing or unsafe.")
    target = _production_target(data_root, target_date_utc)
    manifests = []
    for path in _manifest_paths(data_root):
        observed = path.stat()
        manifests.append(
            {
                "path": path.relative_to(data_root).as_posix(),
                "device": observed.st_dev,
                "inode": observed.st_ino,
                "size": observed.st_size,
                "mtime_ns": observed.st_mtime_ns,
            }
        )
    directory_metadata = {}
    for name, path in (
        ("data_root", data_root),
        ("nightly_root", history.survey_data_root(data_root) / "nightly"),
        ("cumulative_root", history.cumulative_paths(data_root)["dir"]),
    ):
        observed = path.stat()
        directory_metadata[name] = {
            "path": str(path),
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
            "mtime_ns": observed.st_mtime_ns,
        }
    transaction_artifacts = []
    file_inventory = []
    for directory, names, files in os.walk(data_root, followlinks=False):
        safe_names = []
        for name in names:
            child = Path(directory) / name
            if child.is_symlink():
                raise CommissioningError(f"Production sentinel found a symlink: {child}.")
            safe_names.append(name)
        names[:] = safe_names
        for name in sorted(files):
            path = Path(directory) / name
            observed = path.lstat()
            if not stat.S_ISREG(observed.st_mode):
                raise CommissioningError(
                    f"Production sentinel found a non-regular file: {path}."
                )
            file_inventory.append(
                {
                    "path": path.relative_to(data_root).as_posix(),
                    "device": observed.st_dev,
                    "inode": observed.st_ino,
                    "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                    "bytes": observed.st_size,
                    "mtime_ns": observed.st_mtime_ns,
                    "sha256": _file_sha256(path),
                }
            )
        for name in names + files:
            lowered = name.lower()
            if (
                name == ".antares-operations"
                or "transaction" in lowered
                or lowered.endswith(".pending")
                or lowered.startswith(".manifest.")
            ):
                transaction_artifacts.append(
                    (Path(directory) / name).relative_to(data_root).as_posix()
                )
    ordered_inventory = sorted(file_inventory, key=lambda item: item["path"])
    checksum_manifest = "".join(
        f"{item['sha256']}  ./{item['path']}\n" for item in ordered_inventory
    ).encode("utf-8")
    stable = {
        "data_root_identity": directory_metadata["data_root"],
        "directory_metadata": directory_metadata,
        "manifest_inventory": manifests,
        "durable_file_inventory": ordered_inventory,
        "durable_file_count": len(file_inventory),
        "durable_bytes": sum(int(item["bytes"]) for item in file_inventory),
        "checksum_manifest_sha256": hashlib.sha256(checksum_manifest).hexdigest(),
        "target_path": str(target),
        "target_absent": not target.exists() and not target.is_symlink(),
        "transaction_artifacts": sorted(transaction_artifacts),
        "cache_path": str(cache_root),
        "cache_absent": not cache_root.exists() and not cache_root.is_symlink(),
    }
    return {
        "schema_version": "phase6.production-sentinel.v1",
        "captured_at_utc": _iso(_utc_now()),
        **stable,
        "fingerprint_sha256": _digest(stable),
    }


def compare_production_sentinels(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    passed = bool(
        before.get("fingerprint_sha256") == after.get("fingerprint_sha256")
        and after.get("target_absent") is True
        and after.get("cache_absent") is True
        and after.get("transaction_artifacts") == []
    )
    return {
        "passed": passed,
        "before_sha256": before.get("fingerprint_sha256"),
        "after_sha256": after.get("fingerprint_sha256"),
        "publication_attempted": False,
        "production_target_created": False if after.get("target_absent") is True else True,
        "scientific_bytes_changed": False if passed else None,
        "cache_absent": after.get("cache_absent"),
        "transaction_artifacts": after.get("transaction_artifacts"),
    }


def _evidence_inventory(run_root: Path, evidence_root: Path) -> Mapping[str, Any]:
    inventory_path = evidence_root / "inventory.json"
    seal_path = evidence_root / "inventory.sha256"
    rows = []
    for directory, names, files in os.walk(run_root, followlinks=False):
        safe_names = []
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                raise CommissioningError(f"Run-root inventory found a symlink: {path}.")
            safe_names.append(name)
        names[:] = safe_names
        for name in sorted(files):
            path = Path(directory) / name
            if path in {inventory_path, seal_path}:
                continue
            observed = path.lstat()
            if not stat.S_ISREG(observed.st_mode):
                raise CommissioningError(f"Run-root inventory found a non-file: {path}.")
            rows.append(
                {
                    "path": path.relative_to(run_root).as_posix(),
                    "bytes": observed.st_size,
                    "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                    "sha256": _file_sha256(path),
                }
            )
    payload = {
        "schema_version": "phase6.evidence-inventory.v1",
        "run_root": str(run_root),
        "objects": sorted(rows, key=lambda item: item["path"]),
    }
    encoded = _json_bytes(payload)
    _write_json_atomic(inventory_path, payload)
    seal = hashlib.sha256(encoded).hexdigest()
    descriptor = os.open(
        str(seal_path),
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        os.write(descriptor, (seal + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "inventory_path": str(inventory_path),
        "inventory_sha256": seal,
        "object_count": len(rows),
    }


@dataclass(frozen=True)
class CommissioningResult:
    report: OperationReport
    stage: Optional[Path]
    evidence_root: Path


def qualify_live_night(
    write_capability: SyntheticWriteCapability,
    live_read_capability: LiveAntaresReadCapability,
    provider: LiveAntaresProvider,
    spec: NightExecutionSpec,
    *,
    production_data_root: Path,
    production_cache_root: Path,
    clock: Any = _utc_now,
) -> CommissioningResult:
    """Run query through pre-commit reproof and stop without publication."""
    started = clock()
    t0 = time.monotonic()
    if type(write_capability) is not SyntheticWriteCapability:
        raise CommissioningError("Commissioning requires the sealed canary capability.")
    if type(live_read_capability) is not LiveAntaresReadCapability:
        raise CommissioningError("Commissioning requires sealed LIVE_ANTARES_READ.")
    if type(provider) is not LiveAntaresProvider:
        raise CommissioningError("Commissioning requires the exact live provider.")
    if (
        write_capability.root != live_read_capability.run_root
        or write_capability.run_id != live_read_capability.run_id
        or spec.release_sha != live_read_capability.release_sha
        or spec.science_request.date_utc != live_read_capability.target_date_utc
    ):
        raise CommissioningError("Commissioning capability identities disagree.")
    request = spec.science_request
    if (
        request.date_utc != TARGET_DATE_UTC
        or request.mjd_min != 61218.0
        or request.mjd_max != 61219.0
        or request.query_tag is not None
        or request.target_loci is not None
        or request.lsst_only is not True
    ):
        raise CommissioningError(
            "Commissioning requires the exact untagged exhaustive 2026-06-27 request."
        )
    if write_capability.environment == "arnor-canary" and (
        Path(production_data_root) != MIDDLE_EARTH_DATA_ROOT
        or Path(production_cache_root) != MIDDLE_EARTH_CACHE_ROOT
    ):
        raise CommissioningError("Arnor live commissioning requires canonical roots.")

    evidence_root = write_capability.evidence_root
    _ensure_private_tree(evidence_root, write_capability.root)
    target_relative = nightly_target_relative(spec.science_request.date_utc)
    candidate_target = contained_path(write_capability.published_root, target_relative)
    plan = {
        "schema_version": COMMISSIONING_SCHEMA_VERSION,
        "transaction_id": spec.transaction_id,
        "plan_id": spec.plan_id,
        "release_sha": spec.release_sha,
        "configuration_identity": spec.configuration_identity,
        "provider": provider.provider_name,
        "provider_scenario": provider.scenario,
        "target_date_utc": spec.science_request.date_utc,
        "candidate_target": str(candidate_target),
        "production_target": str(
            _production_target(production_data_root, spec.science_request.date_utc)
        ),
        "publication_available": False,
        "reconciliation_available": False,
        "cache_mutation_available": False,
    }
    plan_hash = _digest(plan)
    config_hash = _digest(
        {
            "configuration_identity": spec.configuration_identity,
            "write_environment": write_capability.environment,
            "live_environment": live_read_capability.environment,
            "provider": provider.provider_name,
            "expected_client": {
                "distribution": "antares-client",
                "version": "1.14.0",
                "api_base_url": "https://api.antares.noirlab.edu/v1/",
            },
            "execution_policy": provider.execution_policy(),
        }
    )
    writer_lock = WriterLock(
        write_capability,
        target_relative.as_posix(),
        spec.transaction_id,
        transaction_id=spec.transaction_id,
        release_sha=spec.release_sha,
        plan_hash=plan_hash,
        config_hash=config_hash,
    )
    stage = contained_path(
        write_capability.staging_root,
        Path(spec.transaction_id) / writer_lock.path.name.removesuffix(".lock"),
    )
    journal_path = contained_path(
        write_capability.journal_root, Path(f"{spec.transaction_id}.json")
    )
    _ensure_private_tree(write_capability.journal_root, write_capability.root)
    if journal_path.exists() or journal_path.is_symlink():
        raise CommissioningError("Commissioning transaction identity already exists.")
    descriptor = TransactionDescriptor(
        run_id=spec.transaction_id,
        operation="night.live_qualify",
        target_identity=target_relative.as_posix(),
        target_path=str(candidate_target),
        stage_path=str(stage),
        lock_path=str(writer_lock.path),
        profile=f"phase6:{write_capability.environment}",
        plan_id=spec.plan_id,
        release_sha=spec.release_sha,
        metadata={
            **plan,
            "plan_hash": plan_hash,
            "configuration_hash": config_hash,
            "run_root": str(write_capability.root),
            "evidence_root": str(evidence_root),
            "live_read_authority": True,
            "production_publication_authority": False,
            "production_reconciliation_authority": False,
            "production_cache_mutation_authority": False,
        },
    )
    journal = TransactionJournal.create(journal_path, descriptor, at=started)
    transaction: Optional[PublicationTransaction] = None
    stage_validated = False
    query_result = None
    science_result = None
    before_sentinel = None
    checkpoint = "preflight"
    try:
        _write_json_atomic(evidence_root / "plan.json", plan)
        environment = release_environment_preflight(
            write_capability, provider, spec
        )
        _write_json_atomic(
            evidence_root / "release-environment.json", environment
        )
        eligibility = establish_target_eligibility(
            production_data_root, spec.science_request.date_utc
        )
        _write_json_atomic(
            evidence_root / "target-eligibility.json", eligibility
        )
        scientific_contract = dict(provider.scientific_contract(spec.science_request))
        historical_semantics = {
            "passed": True,
            "target_date_utc": TARGET_DATE_UTC,
            "mjd_min": 61218.0,
            "mjd_max": 61219.0,
            "extractor": "probe_first_time_ra_dec",
            "scientific_contract": scientific_contract,
            "query_contract_sha256": _canonical_digest(scientific_contract),
            "lsst_filter": historical_query.lsst_identifier_filter(),
            "query_tag": None,
            "seed": None,
            "time_lower_bound": "inclusive",
            "time_upper_bound": "exclusive",
            "spatial_partition": "24 RA bins x 6 Dec bins per 30-minute time bin",
            "saturation_rule": "50 provisional rows trigger largest-normalized-dimension split",
            "deduplication": "locus_id keep-last after accepted-tile concatenation",
            "locus_transform": "src.history.prepare_loci",
            "alert_transform": "src.history.prepare_alerts",
            "validation": "src.history.validation_summary",
            "fetch": "antares_client.search.get_by_id full locus/lightcurve",
            "fetch_workers": 4,
            "source_query_mode": "probe_first_time_ra_dec",
            "accepted_predecessor_extraction_method": eligibility[
                "accepted_extraction_method"
            ],
            "predecessor_extraction_match": (
                eligibility["accepted_extraction_method"]
                == extraction_method_contract()
            ),
            "cache_version_identity": "probe50_time_ra_dec_v1",
            "cache_used": False,
            "operational_retry_policy_changed": True,
            "operational_cache_policy_changed": True,
            "scientific_semantics_changed": False,
        }
        _write_json_atomic(
            evidence_root / "historical-semantic-equivalence.json",
            historical_semantics,
        )
        prior = history.load_cumulative_loci_index(
            production_data_root,
            before_mjd=spec.science_request.mjd_min,
            before_date=spec.science_request.date_utc,
        )
        prior_ids = (
            prior[history.LOCUS_ID_COL].dropna().astype(str).tolist()
            if history.LOCUS_ID_COL in prior.columns
            else []
        )
        prior_state = {
            "source": str(history.cumulative_paths(production_data_root)["loci_index"]),
            "before_mjd": spec.science_request.mjd_min,
            "before_date": spec.science_request.date_utc,
            "prior_locus_count": len(prior_ids),
            "prior_locus_identity_sha256": _sequence_digest(prior_ids),
            "used_for_overlap_validation": True,
        }
        _write_json_atomic(
            evidence_root / "prior-cumulative-state.json", prior_state
        )
        spec = replace(
            spec,
            science_request=replace(
                spec.science_request,
                ingested_at_utc=_iso(started),
                prior_locus_ids=tuple(prior_ids),
            ),
        )
        capacity = resource_preflight(
            write_capability.root,
            production_data_root,
            provider_policy=provider.execution_policy(),
        )
        _write_json_atomic(
            evidence_root / "resource-preflight.json", capacity
        )
        connectivity = provider.check_connectivity()
        _write_json_atomic(evidence_root / "connectivity.json", connectivity)
        before_sentinel = capture_production_sentinel(
            production_data_root,
            production_cache_root,
            spec.science_request.date_utc,
        )
        _write_json_atomic(
            evidence_root / "production-sentinel-before.json", before_sentinel
        )
        if write_capability.environment == "arnor-canary":
            inventory_by_path = {
                item["path"]: item
                for item in before_sentinel["durable_file_inventory"]
            }
            loci_relative = history.cumulative_paths(production_data_root)[
                "loci_index"
            ].relative_to(production_data_root).as_posix()
            summary_relative = history.cumulative_paths(production_data_root)[
                "nightly_summary"
            ].relative_to(production_data_root).as_posix()
            if (
                before_sentinel.get("durable_file_count")
                != ACCEPTED_BASELINE["durable_file_count"]
                or before_sentinel.get("durable_bytes")
                != ACCEPTED_BASELINE["durable_bytes"]
                or before_sentinel.get("checksum_manifest_sha256")
                != ACCEPTED_BASELINE["checksum_manifest_sha256"]
                or inventory_by_path.get(loci_relative, {}).get("sha256")
                != ACCEPTED_BASELINE["loci_index_sha256"]
                or inventory_by_path.get(summary_relative, {}).get("sha256")
                != ACCEPTED_BASELINE["nightly_summary_sha256"]
            ):
                raise CommissioningError(
                    "The pre-live production sentinel differs from the accepted baseline."
                )
        journal.transition(
            ExecutionState.PRECHECKED,
            validation={
                "eligibility": eligibility,
                "release_environment": environment,
                "prior_cumulative_state": prior_state,
                "resource_preflight": capacity,
                "connectivity": connectivity,
                "production_sentinel": before_sentinel,
                "passed": True,
            },
            publication={"available": False, "attempted": False},
            reconciliation={"available": False, "attempted": False},
        )
        writer_lock.acquire(at=clock())
        journal.transition(
            ExecutionState.LOCKED,
            publication={"writer_lock_acquired": True, "attempted": False},
        )
        transaction = PublicationTransaction(
            write_capability, writer_lock, target_relative, spec.transaction_id
        )
        transaction.prepare()

        checkpoint = "live_query"
        journal.transition(ExecutionState.QUERYING)
        transaction.begin_query()
        query_result = provider.query(spec.science_request)
        _write_json_atomic(
            evidence_root / "query-evidence.json",
            {
                "provider": query_result.provider_name,
                "outcome": query_result.outcome.value,
                "evidence": query_result.evidence.as_dict(),
            },
        )
        query_result.require_completed()

        checkpoint = "live_fetch"
        journal.transition(ExecutionState.FETCHING)
        transaction.begin_fetch()
        science_result = provider.fetch(spec.science_request, query_result)
        _write_json_atomic(
            evidence_root / "fetch-evidence.json",
            {
                "provider": science_result.provider_name,
                "outcome": science_result.outcome.value,
                "evidence": science_result.fetch_evidence.as_dict(),
                "validation": dict(science_result.validation),
            },
        )
        science_result.require_publishable()
        checkpoint = "artifact_construction"
        artifacts = build_night_artifacts(science_result)
        reopen_and_validate_artifacts(artifacts, expected=science_result)

        transaction.stage_artifacts(artifacts, science_result.evidence)
        checkpoint = "staged_validation"
        staged = _artifact_identities(transaction.stage)
        journal.transition(
            ExecutionState.STAGED,
            artifacts=staged,
            validation={
                "query_evidence": science_result.query_evidence.as_dict(),
                "fetch_evidence": science_result.fetch_evidence.as_dict(),
                "science": dict(science_result.validation),
            },
        )

        writer_validation: Dict[str, Any] = {}

        def staged_validator(stage_directory: Path) -> bool:
            payloads = {
                name: _read_nofollow(stage_directory / name)
                for name in EXPECTED_ARTIFACTS
            }
            reopened = reopen_and_validate_artifacts(payloads, expected=science_result)
            writer_validation.update(
                {
                    "passed": True,
                    "date_utc": reopened.manifest.get("date_utc"),
                    "actual_loci": len(reopened.loci),
                    "alert_rows": len(reopened.alerts),
                    "validation": reopened.manifest.get("validation"),
                }
            )
            return True

        transaction.validate(staged_validator)
        validated = _artifact_identities(transaction.stage)
        if validated != staged:
            raise CommissioningError("Staged identities changed across validation.")
        writer_lock.update_artifact_hashes(_artifact_hashes(validated))
        journal.transition(
            ExecutionState.VALIDATED,
            artifacts=validated,
            validation={"passed": True, "result": writer_validation},
        )
        stage_validated = True
        checkpoint = "precommit_reproof"
        reproof = transaction.precommit_reprove()
        if {
            name: str(reproof[name]["sha256"]) for name in sorted(reproof)
        } != _artifact_hashes(validated):
            raise CommissioningError("Pre-commit reproof differs from validation.")
        production_target = _production_target(
            production_data_root, spec.science_request.date_utc
        )
        if production_target.exists() or production_target.is_symlink():
            raise CommissioningError("Production target appeared before qualification stop.")
        precommit_report = {
            "passed": True,
            "artifact_set": list(EXPECTED_ARTIFACTS),
            "artifacts": reproof,
            "candidate_target_absent": not candidate_target.exists(),
            "production_target": str(production_target),
            "production_target_absent": True,
            "publication_invoked": False,
            "production_publication_capability": False,
            "production_reconciliation_capability": False,
            "cache_mutation_capability": False,
        }
        _write_json_atomic(evidence_root / "precommit-reproof.json", precommit_report)
        journal.update(
            validation={
                "precommit_reproved": True,
                "precommit_artifact_sha256": _artifact_hashes(validated),
                "commissioning_complete": False,
                "forensic_reopen_pending": True,
            },
            publication={
                "available": False,
                "attempted": False,
                "invoked": False,
                "committed": False,
                "writer_lock_released": False,
            },
            reconciliation={"available": False, "attempted": False},
            durability={"stage_retained": True, "status": "validated-non-authoritative"},
        )

        checkpoint = "independent_reopen"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.operations.commissioning_reopen",
                "--stage",
                str(transaction.stage),
                "--run-root",
                str(write_capability.root),
                "--journal",
                str(journal_path),
                "--query-evidence",
                str(evidence_root / "query-evidence.json"),
                "--fetch-evidence",
                str(evidence_root / "fetch-evidence.json"),
                "--production-target",
                str(production_target),
                "--expected-release-sha",
                spec.release_sha,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,
        )
        try:
            independent = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CommissioningError("Fresh-process reopen returned invalid JSON.") from exc
        if completed.returncode != 0 or independent.get("passed") is not True:
            checkpoint = (
                "independent_reopen:"
                + str(
                    independent.get(
                        "error", independent.get("error_type", "unknown")
                    )
                )
            )
            _write_json_atomic(
                evidence_root / "independent-reopen-failure.json",
                {
                    "return_code": completed.returncode,
                    "report": independent,
                    "stderr_recorded": False,
                },
            )
            raise CommissioningError("Fresh-process forensic reopen failed.")
        for name in EXPECTED_ARTIFACTS:
            observed = independent["artifacts"][name]
            expected = validated[name]
            if (
                observed.get("device") != expected.device
                or observed.get("inode") != expected.inode
                or observed.get("bytes") != expected.size
                or observed.get("mode") != f"{expected.mode:04o}"
                or observed.get("sha256") != expected.sha256
            ):
                raise CommissioningError(
                    "Fresh-process reopen artifact identity disagrees."
                )
        if (
            independent.get("loci_rows") != science_result.evidence.loci_rows
            or independent.get("alert_rows") != science_result.evidence.alert_rows
            or independent.get("production_target_absent") is not True
            or independent.get("journal", {}).get("writer_lock_observed") is not True
        ):
            raise CommissioningError("Fresh-process forensic report disagrees.")
        _write_json_atomic(evidence_root / "independent-reopen.json", independent)

        checkpoint = "post_run_production_tripwire"
        after_sentinel = capture_production_sentinel(
            production_data_root,
            production_cache_root,
            spec.science_request.date_utc,
        )
        comparison = compare_production_sentinels(before_sentinel, after_sentinel)
        _write_json_atomic(evidence_root / "production-sentinel-after.json", after_sentinel)
        _write_json_atomic(evidence_root / "production-sentinel-comparison.json", comparison)
        if comparison["passed"] is not True:
            raise CommissioningError("Production mutation tripwire changed.")

        checkpoint = "candidate_finalization"
        writer_lock.release()
        post_release_identities = _artifact_identities(transaction.stage)
        if post_release_identities != validated:
            raise CommissioningError("Candidate identity changed after lock release.")
        journal.update(
            validation={
                "commissioning_complete": True,
                "forensic_reopen_pending": False,
                "independent_reopen_passed": True,
                "post_release_reproof_passed": True,
            },
            publication={"writer_lock_released": True},
        )

        candidate = {
            "schema_version": COMMISSIONING_SCHEMA_VERSION,
            "passed": True,
            "release_sha": spec.release_sha,
            "run_id": write_capability.run_id,
            "transaction_id": spec.transaction_id,
            "target_date_utc": spec.science_request.date_utc,
            "stage": str(transaction.stage),
            "artifacts": {
                name: validated[name].as_dict() for name in sorted(validated)
            },
            "loci_rows": science_result.evidence.loci_rows,
            "alert_rows": science_result.evidence.alert_rows,
            "science_validation": dict(science_result.validation),
            "query_completion": science_result.query_evidence.as_dict(),
            "fetch_completion": science_result.fetch_evidence.as_dict(),
            "publication_invoked": False,
            "publication_attempted": False,
            "production_target_created": False,
            "scientific_bytes_changed": False,
            "candidate_retained": True,
            "authoritative": False,
        }
        _write_json_atomic(evidence_root / "qualification.json", candidate)
        inventory = _evidence_inventory(write_capability.root, evidence_root)
        finished = clock()
        report = OperationReport(
            operation="night.live_qualify",
            success=True,
            status="validated_candidate_retained",
            started_at_utc=_iso(started),
            finished_at_utc=_iso(finished),
            elapsed_seconds=max(0.0, time.monotonic() - t0),
            exit_code=ExitCode.SUCCESS,
            run_id=spec.transaction_id,
            counts={
                "loci": science_result.evidence.loci_rows,
                "alerts": science_result.evidence.alert_rows,
                "staged_artifacts": len(validated),
            },
            artifacts=(
                Artifact("candidate-stage", "non-authoritative-science", str(transaction.stage), "retained"),
                Artifact("transaction-journal", "operational-state", str(journal_path), "validated"),
                Artifact("evidence", "qualification", str(evidence_root), "sealed"),
            ),
            evidence=(
                Evidence("target-eligibility", "pass", "Target is the next accepted UTC night.", eligibility),
                Evidence("query-completion", "pass", "ANTARES iterator reached positive exhaustion.", science_result.query_evidence.as_dict()),
                Evidence("fetch-completion", "pass", "Every enumerated locus was accounted for.", science_result.fetch_evidence.as_dict()),
                Evidence("production-tripwire", "pass", "Production sentinel is unchanged.", comparison),
            ),
            next_actions=(
                "Review the retained commissioning evidence; do not publish in Phase 6.",
            ),
            details={
                "schema_version": COMMISSIONING_SCHEMA_VERSION,
                "plan_hash": plan_hash,
                "configuration_hash": config_hash,
                "stage": str(transaction.stage),
                "journal": str(journal_path),
                "evidence_root": str(evidence_root),
                "inventory": inventory,
                "independent_reopen": True,
                "publication_invoked": False,
                "publication_attempted": False,
                "production_publication_authority": False,
                "production_reconciliation_authority": False,
                "production_cache_mutation_authority": False,
                "production_target_created": False,
                "scientific_bytes_changed": False,
                "candidate_retained": True,
            },
            read_only=False,
        )
        return CommissioningResult(report, transaction.stage, evidence_root)
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        if transaction is not None and not stage_validated:
            try:
                transaction.abort("commissioning_failed_before_validated_stage")
            except Exception:
                pass
        if writer_lock.held:
            try:
                writer_lock.release()
            except Exception:
                pass
        failure_comparison: Optional[Mapping[str, Any]] = None
        if before_sentinel is not None:
            try:
                failure_after = capture_production_sentinel(
                    production_data_root,
                    production_cache_root,
                    spec.science_request.date_utc,
                )
                failure_comparison = compare_production_sentinels(
                    before_sentinel, failure_after
                )
                _write_json_atomic(
                    evidence_root / "production-sentinel-after.json",
                    failure_after,
                )
                _write_json_atomic(
                    evidence_root / "production-sentinel-comparison.json",
                    failure_comparison,
                )
            except Exception:
                failure_comparison = {
                    "passed": False,
                    "production_target_created": None,
                    "scientific_bytes_changed": None,
                    "cache_absent": None,
                    "post_failure_tripwire_error": True,
                }
        try:
            if journal.snapshot.state not in {ExecutionState.FAILED, ExecutionState.COMPLETE}:
                journal.transition(
                    ExecutionState.FAILED,
                    reason="commissioning_failed",
                    failure={
                        "error_type": type(error).__name__,
                        "message": "Phase 6 failed closed; provider exception text was redacted.",
                    },
                    at=clock(),
                    publication={"available": False, "attempted": False, "invoked": False},
                    reconciliation={"available": False, "attempted": False},
                )
        except Exception:
            pass
        failure = {
            "schema_version": COMMISSIONING_SCHEMA_VERSION,
            "passed": False,
            "error_type": type(error).__name__,
            "failed_checkpoint": checkpoint,
            "error_text_recorded": False,
            "stage_retained": bool(stage_validated and stage.exists()),
            "publication_invoked": False,
            "publication_attempted": False,
            "production_target_created": (
                failure_comparison.get("production_target_created")
                if failure_comparison is not None
                else None
            ),
            "scientific_bytes_changed": (
                failure_comparison.get("scientific_bytes_changed")
                if failure_comparison is not None
                else None
            ),
            "post_failure_production_tripwire": failure_comparison,
            "secret_material_recorded": False,
        }
        try:
            _write_json_atomic(evidence_root / "failure.json", failure)
        except Exception:
            pass
        failure_inventory = None
        try:
            if not (evidence_root / "inventory.sha256").exists():
                failure_inventory = _evidence_inventory(
                    write_capability.root, evidence_root
                )
        except Exception:
            failure_inventory = None
        finished = clock()
        report = OperationReport(
            operation="night.live_qualify",
            success=False,
            status="failed_closed",
            started_at_utc=_iso(started),
            finished_at_utc=_iso(finished),
            elapsed_seconds=max(0.0, time.monotonic() - t0),
            exit_code=(
                ExitCode.VALIDATION_FAILURE
                if isinstance(error, ProviderError)
                else ExitCode.OPERATIONAL_FAILURE
            ),
            run_id=spec.transaction_id,
            errors=(Issue(type(error).__name__, "Phase 6 failed closed; inspect non-secret evidence."),),
            artifacts=(
                Artifact("transaction-journal", "operational-state", str(journal_path), journal.snapshot.state.value),
                Artifact("evidence", "qualification", str(evidence_root), "failure-preserved"),
            ),
            next_actions=("Correct the demonstrated gate failure before another live query.",),
            details={
                **failure,
                "journal": str(journal_path),
                "evidence_root": str(evidence_root),
                "stage": str(stage),
                "inventory": failure_inventory,
            },
            read_only=False,
        )
        return CommissioningResult(report, stage if stage.exists() else None, evidence_root)


__all__ = [
    "ACCEPTED_BASELINE",
    "CommissioningError",
    "CommissioningResult",
    "TARGET_DATE_UTC",
    "capture_production_sentinel",
    "compare_production_sentinels",
    "establish_target_eligibility",
    "qualify_live_night",
    "resource_preflight",
]
