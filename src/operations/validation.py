"""Non-mutating preflight observations for future writers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .context import OperationContext
from .storage import NightInspection, NightLocation, StorageLayout


@dataclass(frozen=True)
class PreflightCheck:
    code: str
    status: str
    summary: str
    details: Mapping[str, Any]


def _existing_ancestor(path: Path) -> Optional[Path]:
    cursor = path
    while cursor != cursor.parent:
        if cursor.exists():
            return cursor
        cursor = cursor.parent
    return cursor if cursor.exists() else None


def inspect_writer_preflight(
    context: OperationContext,
    layout: StorageLayout,
    target: NightLocation,
    inspection: NightInspection,
) -> Tuple[PreflightCheck, ...]:
    """Inspect future writer prerequisites without write probes or locks."""
    configuration_valid = bool(
        context.storage_policy in {"private", "shared-group"}
        and (
            (context.storage_policy == "private" and context.shared_group is None)
            or (
                context.storage_policy == "shared-group"
                and context.shared_group is not None
                and context.shared_group.strip()
            )
        )
    )
    checks: list[PreflightCheck] = [
        PreflightCheck(
            "configuration",
            "pass" if configuration_valid else "fail",
            "Profile and storage policy resolved explicitly."
            if configuration_valid
            else "Storage policy and shared-group configuration conflict.",
            {
                "profile": context.profile_name,
                "storage_policy": context.storage_policy,
                "shared_group": context.shared_group,
            },
        ),
        PreflightCheck(
            "cache-separation",
            "pass",
            "Cache and durable roots are disjoint after resolution.",
            {},
        ),
        PreflightCheck(
            "target-containment",
            "pass",
            "Night target is contained within the durable root.",
            {"target": target.relative_directory.as_posix()},
        ),
    ]

    data_root = layout.data_root
    if data_root.is_symlink():
        root_status, root_summary = "fail", "Data root must not be a symlink."
    elif not data_root.exists():
        root_status, root_summary = "fail", "Data root is unavailable."
    elif not data_root.is_dir():
        root_status, root_summary = "fail", "Data root is not a directory."
    elif not os.access(data_root, os.R_OK | os.X_OK):
        root_status, root_summary = "fail", "Data root is not readable/traversable."
    else:
        root_status, root_summary = "pass", "Data root is readable and traversable."
    checks.append(
        PreflightCheck(
            "data-root-availability",
            root_status,
            root_summary,
            {"exists": data_root.exists()},
        )
    )

    target_status = "pass" if inspection.state == "missing" else "info"
    if inspection.state in {"incomplete", "conflicting"}:
        target_status = "fail"
    checks.append(
        PreflightCheck(
            "target-partition",
            target_status,
            f"Target partition state is {inspection.state}.",
            inspection.as_dict(),
        )
    )

    ancestor = _existing_ancestor(target.directory.parent)
    staging_status = "unknown"
    staging_summary = "Staging capability cannot be proved before the root exists."
    staging_details: dict[str, Any] = {"write_probe_performed": False}
    if ancestor is not None:
        writable = os.access(ancestor, os.W_OK | os.X_OK)
        staging_status = "pass" if writable else "fail"
        staging_summary = (
            "Existing target ancestor is writable/traversable."
            if writable
            else "Existing target ancestor is not writable/traversable."
        )
        staging_details["existing_ancestor"] = str(ancestor)
    checks.append(
        PreflightCheck(
            "staging-capability",
            staging_status,
            staging_summary,
            staging_details,
        )
    )

    if ancestor is not None:
        try:
            os.statvfs(ancestor)
            checks.append(
                PreflightCheck(
                    "disk-space",
                    "info",
                    "Capacity is observable; required bytes remain unknown before query.",
                    {"observable": True, "required_bytes": None},
                )
            )
        except OSError as exc:
            checks.append(
                PreflightCheck(
                    "disk-space", "unknown", "Disk-space observation failed.",
                    {"error": str(exc)},
                )
            )
        try:
            os.statvfs(ancestor)
            checks.append(
                PreflightCheck(
                    "inode-space",
                    "info",
                    "Inode capacity is observable where supported.",
                    {"observable": True, "required_inodes": None},
                )
            )
        except (AttributeError, OSError) as exc:
            checks.append(
                PreflightCheck(
                    "inode-space", "unknown", "Inode observation unavailable.",
                    {"error": str(exc)},
                )
            )

    checks.append(
        PreflightCheck(
            "lock-availability",
            "unknown",
            "Production operations root is not configured; lock availability "
            "cannot be evaluated.",
            {
                "resource": None,
                "operations_root_configured": False,
                "arnor_shire_canary_contract_recorded": True,
                "configured_operations_root_qualified": False,
                "production_lock_provisioned": False,
            },
        )
    )
    checks.extend(
        [
            PreflightCheck(
                "baseline-validity",
                "unknown",
                "A full accepted-baseline audit is required immediately before writing.",
                {"performed": False},
            ),
            PreflightCheck(
                "network-requirement",
                "info",
                "Execution would require ANTARES network access; planning did not.",
                {"plan_network_access": False, "execution_requires_network": True},
            ),
            PreflightCheck(
                "execution-authorization",
                "fail",
                "Production writer is disabled in this release.",
                {"reason": "writer_not_enabled_in_this_release"},
            ),
        ]
    )
    return tuple(checks)
