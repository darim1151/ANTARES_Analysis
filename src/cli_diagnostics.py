"""Read-only diagnostics and fast dataset inventory for the CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from src.cli_notebooks import NOTEBOOKS, discover_repo_root
from src.cli_profiles import StorageProfile


REQUIRED_IMPORTS = (
    "antares_client",
    "astropy",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
)
VALID_NIGHT_STATUSES = {"complete", "under_target", "saturated_unresolved"}


@dataclass(frozen=True)
class DiagnosticCheck:
    code: str
    status: str
    summary: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _path_state(path: Path) -> dict[str, object]:
    """Describe a path through metadata only; never create or open for writing."""
    state: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
        "is_symlink": path.is_symlink(),
        "is_directory": False,
        "is_file": False,
        "readable": False,
        "writable": False,
        "executable": False,
        "mode": None,
        "mode_bits": None,
        "uid": None,
        "gid": None,
        "size_bytes": None,
    }
    if not state["exists"]:
        return state
    try:
        metadata = path.stat()
    except OSError as exc:
        state["error"] = str(exc)
        return state
    mode_bits = stat.S_IMODE(metadata.st_mode)
    state.update(
        {
            "is_directory": stat.S_ISDIR(metadata.st_mode),
            "is_file": stat.S_ISREG(metadata.st_mode),
            "readable": os.access(path, os.R_OK),
            "writable": os.access(path, os.W_OK),
            "executable": os.access(path, os.X_OK),
            "mode": stat.filemode(metadata.st_mode),
            "mode_bits": f"{mode_bits:04o}",
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "size_bytes": metadata.st_size if stat.S_ISREG(metadata.st_mode) else None,
        }
    )
    return state


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.expanduser().resolve(strict=False).relative_to(
            parent.expanduser().resolve(strict=False)
        )
    except ValueError:
        return False
    return True


def _manifest_inventory(nightly_root: Path) -> tuple[list[Path], list[str]]:
    if not nightly_root.is_dir():
        return [], []
    if nightly_root.is_symlink():
        return [], [f"{nightly_root}: nightly root must not be a symlink"]
    paths: list[Path] = []
    errors: list[str] = []
    for directory, names, files in os.walk(nightly_root, followlinks=False):
        safe_names = []
        for name in names:
            candidate = Path(directory) / name
            if candidate.is_symlink():
                errors.append(f"{candidate}: nightly directory must not be a symlink")
            else:
                safe_names.append(name)
        names[:] = safe_names
        if "manifest.json" in files:
            manifest_path = Path(directory) / "manifest.json"
            if manifest_path.is_symlink():
                errors.append(f"{manifest_path}: manifest must not be a symlink")
            else:
                paths.append(manifest_path)
    return sorted(paths), errors


def collect_data_status(profile: StorageProfile) -> dict[str, object]:
    """Collect a bounded manifest/layout summary without decoding Parquet data."""
    data_root = profile.data_root
    lsst_root = data_root / "data" / "lsst_only"
    nightly_root = lsst_root / "nightly"
    cumulative_root = lsst_root / "cumulative"
    loci_index = cumulative_root / "loci_index.parquet"
    nightly_summary = cumulative_root / "nightly_summary.parquet"
    manifest_paths, inventory_errors = _manifest_inventory(nightly_root)

    dates: list[str] = []
    total_loci = 0
    total_alerts = 0
    status_counts = {status: 0 for status in sorted(VALID_NIGHT_STATUSES)}
    append_ready = 0
    zero_row_nights: list[str] = []
    seen_dates: set[str] = set()
    errors: list[str] = list(inventory_errors)
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest_path}: {exc}")
            continue
        if not isinstance(manifest, dict):
            errors.append(f"{manifest_path}: manifest root must be a JSON object")
            continue
        date_value = str(manifest.get("date_utc") or "")
        status = str(manifest.get("status") or "")
        validation = manifest.get("validation")
        validation = validation if isinstance(validation, dict) else {}
        loci = manifest.get("actual_loci")
        alerts = manifest.get("alert_rows")
        parsed_date = None
        if not date_value:
            errors.append(f"{manifest_path}: missing date_utc")
        else:
            try:
                parsed_date = date.fromisoformat(date_value)
            except ValueError:
                errors.append(f"{manifest_path}: invalid date_utc {date_value!r}")
                parsed_date = None
        if parsed_date is not None:
            canonical_date = parsed_date.isoformat()
            if canonical_date != date_value:
                errors.append(f"{manifest_path}: invalid date_utc {date_value!r}")
            expected_parts = (
                f"{parsed_date.year:04d}",
                f"{parsed_date.month:02d}",
                f"{parsed_date.day:02d}",
            )
            if tuple(manifest_path.parts[-4:-1]) != expected_parts:
                errors.append(
                    f"{manifest_path}: date_utc {date_value!r} does not match its directory"
                )
            if date_value in seen_dates:
                errors.append(f"{manifest_path}: duplicate date_utc {date_value!r}")
            seen_dates.add(date_value)
            dates.append(date_value)
        if status in VALID_NIGHT_STATUSES:
            status_counts[status] += 1
        else:
            errors.append(f"{manifest_path}: invalid status {status!r}")
        append_ready_value = validation.get("append_ready")
        if append_ready_value is True:
            append_ready += 1
        elif append_ready_value is not False:
            errors.append(f"{manifest_path}: validation.append_ready must be boolean")
        if type(loci) is int and loci >= 0:
            total_loci += loci
        else:
            errors.append(f"{manifest_path}: invalid actual_loci {loci!r}")
        if type(alerts) is int and alerts >= 0:
            total_alerts += alerts
        else:
            errors.append(f"{manifest_path}: invalid alert_rows {alerts!r}")
        if loci == 0 and alerts == 0 and date_value:
            zero_row_nights.append(date_value)

    path_states = {
        "data_root": _path_state(data_root),
        "data_directory": _path_state(data_root / "data"),
        "lsst_root": _path_state(lsst_root),
        "nightly_root": _path_state(nightly_root),
        "cumulative_root": _path_state(cumulative_root),
        "loci_index": _path_state(loci_index),
        "nightly_summary": _path_state(nightly_summary),
        "cache_root": _path_state(profile.cache_root),
        "forbidden_in_tree_cache": _path_state(data_root / "cache"),
    }
    required_directories = (
        "data_root",
        "data_directory",
        "lsst_root",
        "nightly_root",
        "cumulative_root",
    )
    required_files = ("loci_index", "nightly_summary")
    directory_results = {
        name: bool(
            path_states[name]["is_directory"]
            and not path_states[name]["is_symlink"]
            and path_states[name]["readable"]
            and path_states[name]["executable"]
        )
        for name in required_directories
    }
    file_results = {
        name: bool(
            path_states[name]["is_file"]
            and not path_states[name]["is_symlink"]
            and path_states[name]["readable"]
            and path_states[name]["size_bytes"]
        )
        for name in required_files
    }
    for name, valid in directory_results.items():
        if not valid:
            state = path_states[name]
            errors.append(
                f"{state['path']}: required directory is missing, unsafe, or unreadable"
            )
    for name, valid in file_results.items():
        if not valid:
            state = path_states[name]
            errors.append(
                f"{state['path']}: required cumulative file is missing, unsafe, "
                "unreadable, or empty"
            )
    required_paths_ok = all(directory_results.values()) and all(file_results.values())
    forbidden_cache_ok = not bool(
        path_states["forbidden_in_tree_cache"]["exists"]
        or path_states["forbidden_in_tree_cache"]["is_symlink"]
    )
    if not forbidden_cache_ok:
        errors.append(
            f"{path_states['forbidden_in_tree_cache']['path']}: "
            "cache must remain outside durable data"
        )
    return {
        "profile": profile.as_dict(),
        "read_only": True,
        "ok": bool(required_paths_ok and manifest_paths and not errors and forbidden_cache_ok),
        "paths": path_states,
        "summary": {
            "manifest_count": len(manifest_paths),
            "complete_nights": status_counts["complete"],
            "under_target_nights": status_counts["under_target"],
            "saturated_unresolved_nights": status_counts["saturated_unresolved"],
            "append_ready_nights": append_ready,
            "first_date": min(dates) if dates else None,
            "last_date": max(dates) if dates else None,
            "total_loci": total_loci,
            "total_alerts": total_alerts,
            "zero_row_nights": sorted(zero_row_nights),
        },
        "errors": errors,
    }


def collect_doctor_checks(
    profile: StorageProfile,
    *,
    repo_root: Optional[Path] = None,
    check_dependencies: bool = True,
    check_jupyter: bool = True,
) -> list[DiagnosticCheck]:
    """Return fail-closed, read-only environment checks."""
    checks: list[DiagnosticCheck] = []
    supported = (3, 9) <= sys.version_info[:2] < (3, 12)
    checks.append(
        DiagnosticCheck(
            "python-version",
            "pass" if supported else "fail",
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "Supported production range is >=3.9,<3.12.",
        )
    )

    data_state = _path_state(profile.data_root)
    if data_state["is_symlink"]:
        checks.append(
            DiagnosticCheck(
                "data-root",
                "fail",
                "Data root must not be a symlink.",
                str(profile.data_root),
            )
        )
    elif not data_state["exists"]:
        checks.append(
            DiagnosticCheck(
                "data-root",
                "fail",
                "Data root is missing.",
                str(profile.data_root),
            )
        )
    elif not data_state["is_directory"]:
        checks.append(
            DiagnosticCheck(
                "data-root", "fail", "Data root is not a directory.", str(profile.data_root)
            )
        )
    elif not data_state["readable"] or not data_state["executable"]:
        checks.append(
            DiagnosticCheck(
                "data-root",
                "fail",
                "Data root cannot be traversed and read.",
                str(profile.data_root),
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                "data-root",
                "pass",
                "Data root is readable.",
                f"{profile.data_root} ({data_state['mode']})",
            )
        )

    if profile.storage_policy == "private" and data_state["mode_bits"] is not None:
        mode_bits = int(str(data_state["mode_bits"]), 8)
        private = not bool(mode_bits & 0o077)
        checks.append(
            DiagnosticCheck(
                "private-mode",
                "pass" if private else "fail",
                "Private data-root permissions are owner-only."
                if private
                else "Private policy conflicts with group/world permission bits.",
                f"mode={data_state['mode_bits']}",
            )
        )

        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None:
            current_uid = get_effective_uid()
            owned = data_state["uid"] == current_uid
            checks.append(
                DiagnosticCheck(
                    "private-owner",
                    "pass" if owned else "fail",
                    "Private data root is owned by the current user."
                    if owned
                    else "Private data root is owned by a different user.",
                    f"root_uid={data_state['uid']}; current_uid={current_uid}",
                )
            )

    separated = profile.cache_root != profile.data_root and not _is_inside(
        profile.cache_root, profile.data_root
    )
    checks.append(
        DiagnosticCheck(
            "cache-separation",
            "pass" if separated else "fail",
            "Cache root is separate from durable data."
            if separated
            else "Cache root must not be the durable root or a child of it.",
            str(profile.cache_root),
        )
    )
    in_tree_cache = profile.data_root / "cache"
    in_tree_cache_state = _path_state(in_tree_cache)
    in_tree_cache_present = bool(
        in_tree_cache_state["exists"] or in_tree_cache_state["is_symlink"]
    )
    checks.append(
        DiagnosticCheck(
            "in-tree-cache",
            "fail" if in_tree_cache_present else "pass",
            "Durable data root has no cache directory."
            if not in_tree_cache_present
            else "Durable data root contains a forbidden cache path.",
            str(in_tree_cache),
        )
    )

    cache_state = _path_state(profile.cache_root)
    if cache_state["is_symlink"]:
        cache_status = "fail"
        cache_summary = "External cache root must not be a symlink."
    elif cache_state["exists"] and not cache_state["is_directory"]:
        cache_status = "fail"
        cache_summary = "External cache root is not a directory."
    elif cache_state["is_directory"] and not (
        cache_state["readable"]
        and cache_state["writable"]
        and cache_state["executable"]
    ):
        cache_status = "fail"
        cache_summary = "External cache root is not readable, writable, and traversable."
    elif cache_state["is_directory"]:
        cache_status = "pass"
        cache_summary = "External cache root is available."
    else:
        cache_status = "info"
        cache_summary = (
            "External cache root is not present; no cache will be created by doctor."
        )
    checks.append(
        DiagnosticCheck(
            "cache-root",
            cache_status,
            cache_summary,
            str(profile.cache_root),
        )
    )

    status = collect_data_status(profile)
    checks.append(
        DiagnosticCheck(
            "dataset-layout",
            "pass" if status["ok"] else "fail",
            "Nightly manifests and cumulative products are present."
            if status["ok"]
            else "Dataset layout or manifest inventory is incomplete.",
            "; ".join(status["errors"][:3])
            if status["errors"]
            else f"manifests={status['summary']['manifest_count']}",
        )
    )

    if check_dependencies:
        missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
        checks.append(
            DiagnosticCheck(
                "python-dependencies",
                "pass" if not missing else "fail",
                "Scientific Python dependency modules are discoverable."
                if not missing
                else "Required Python dependencies are missing.",
                ", ".join(missing) if missing else ", ".join(REQUIRED_IMPORTS),
            )
        )

    try:
        root = discover_repo_root(repo_root)
    except ValueError as exc:
        if repo_root is not None:
            checks.append(
                DiagnosticCheck(
                    "repository",
                    "fail",
                    "Requested source checkout was not found.",
                    str(exc),
                )
            )
        else:
            checks.append(
                DiagnosticCheck(
                    "repository",
                    "info",
                    "Source checkout is unavailable; notebook checks were skipped.",
                    "Installed runtime and storage checks remain valid. Pass "
                    "--repo-root to require and validate a source checkout.",
                )
            )
    else:
        missing_notebooks = [
            spec.filename
            for spec in NOTEBOOKS
            if not (root / "notebooks" / spec.filename).is_file()
        ]
        checks.append(
            DiagnosticCheck(
                "notebooks",
                "pass" if not missing_notebooks else "fail",
                "All navigable notebooks are present."
                if not missing_notebooks
                else "One or more notebooks are missing.",
                ", ".join(missing_notebooks) if missing_notebooks else str(root / "notebooks"),
            )
        )

    if check_jupyter:
        launcher = shutil.which("jupyter") or shutil.which("jupyter-lab")
        checks.append(
            DiagnosticCheck(
                "jupyter-launcher",
                "pass" if launcher else "warn",
                "Jupyter launcher is on PATH."
                if launcher
                else "Jupyter launcher is not on PATH in this shell.",
                launcher or "Use the Middle Earth/Jupyter environment or install notebook tooling.",
            )
        )
    return checks


def doctor_result(profile: StorageProfile, checks: list[DiagnosticCheck]) -> dict[str, object]:
    counts = {
        status: sum(check.status == status for check in checks)
        for status in ("pass", "info", "warn", "fail")
    }
    return {
        "profile": profile.as_dict(),
        "read_only": True,
        "ok": counts["fail"] == 0,
        "counts": counts,
        "checks": [check.as_dict() for check in checks],
    }
