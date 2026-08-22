"""Storage profiles for the ANTARES Analysis command-line interface.

This module intentionally uses only the Python standard library.  CLI help,
profile inspection, and environment rendering must work before the scientific
Python stack is imported and must never create a storage path.
"""

from __future__ import annotations

import os
import shlex
import socket
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional


MIDDLE_EARTH_DATA_ROOT = Path("/astro/store/shire/ANTARES_Analysis_Data")
MIDDLE_EARTH_CACHE_ROOT = Path("/astro/store/shire/ANTARES_Analysis_cache")
RSP_DATA_ROOT = Path("/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data")
RSP_SHARED_GROUP = "g_antares_analysis"
VALID_STORAGE_POLICIES = ("private", "shared-group")


@dataclass(frozen=True)
class StorageProfile:
    """Resolved, non-mutating storage and execution configuration."""

    name: str
    description: str
    data_root: Path
    cache_root: Path
    storage_policy: str
    shared_group: Optional[str] = None
    source: str = "built-in"

    def environment(self) -> dict[str, str]:
        """Return canonical environment values for notebooks and commands."""
        values = {
            "ANTARES_ANALYSIS_PROFILE": self.name,
            "ANTARES_ANALYSIS_DATA_ROOT": str(self.data_root),
            "ANTARES_ANALYSIS_CACHE_ROOT": str(self.cache_root),
            "ANTARES_STORAGE_POLICY": self.storage_policy,
        }
        if self.shared_group:
            values["ANTARES_SHARED_GROUP"] = self.shared_group
        return values

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["data_root"] = str(self.data_root)
        values["cache_root"] = str(self.cache_root)
        values["environment"] = self.environment()
        return values


BUILTIN_PROFILES: dict[str, StorageProfile] = {
    "middle-earth": StorageProfile(
        name="middle-earth",
        description="Private migrated dataset on Arnor/Gondor Shire storage.",
        data_root=MIDDLE_EARTH_DATA_ROOT,
        cache_root=MIDDLE_EARTH_CACHE_ROOT,
        storage_policy="private",
    ),
    "rsp": StorageProfile(
        name="rsp",
        description="Legacy Rubin Science Platform shared-group compatibility.",
        data_root=RSP_DATA_ROOT,
        cache_root=RSP_DATA_ROOT / "cache",
        storage_policy="shared-group",
        shared_group=RSP_SHARED_GROUP,
    ),
}


def _environment_profile(environ: Mapping[str, str]) -> StorageProfile:
    data_value = environ.get("ANTARES_ANALYSIS_DATA_ROOT") or environ.get(
        "ANTARES_DATA_ROOT"
    )
    data_root = Path(data_value).expanduser() if data_value else MIDDLE_EARTH_DATA_ROOT
    cache_value = environ.get("ANTARES_ANALYSIS_CACHE_ROOT")
    if cache_value:
        cache_root = Path(cache_value).expanduser()
    elif data_root == MIDDLE_EARTH_DATA_ROOT:
        cache_root = MIDDLE_EARTH_CACHE_ROOT
    else:
        cache_root = data_root / "cache"

    policy = environ.get("ANTARES_STORAGE_POLICY", "private").strip().lower()
    if policy not in VALID_STORAGE_POLICIES:
        choices = ", ".join(VALID_STORAGE_POLICIES)
        raise ValueError(
            f"Invalid ANTARES_STORAGE_POLICY {policy!r}; expected one of: {choices}."
        )
    shared_group = None
    if policy == "shared-group":
        shared_group = environ.get("ANTARES_SHARED_GROUP", RSP_SHARED_GROUP).strip()
        if not shared_group:
            raise ValueError(
                "ANTARES_SHARED_GROUP must be non-empty for shared-group storage."
            )
    return StorageProfile(
        name="environment",
        description="Configuration resolved from the current process environment.",
        data_root=data_root,
        cache_root=cache_root,
        storage_policy=policy,
        shared_group=shared_group,
        source="environment",
    )


def detect_profile_name(
    environ: Optional[Mapping[str, str]] = None,
    hostname: Optional[str] = None,
) -> str:
    """Choose a safe default profile without touching any configured path."""
    values = os.environ if environ is None else environ
    explicit = values.get("ANTARES_ANALYSIS_PROFILE", "").strip().lower()
    if explicit:
        if explicit not in {*BUILTIN_PROFILES, "environment", "auto"}:
            choices = ", ".join([*sorted(BUILTIN_PROFILES), "environment"])
            raise ValueError(
                f"Invalid ANTARES_ANALYSIS_PROFILE {explicit!r}; expected: {choices}."
            )
        if explicit != "auto":
            return explicit

    if values.get("ANTARES_ANALYSIS_DATA_ROOT") or values.get("ANTARES_DATA_ROOT"):
        return "environment"

    host = (hostname or socket.gethostname()).lower()
    if "arnor" in host or "gondor" in host:
        return "middle-earth"
    if "rsp" in host or "rubin" in host:
        return "rsp"
    return "middle-earth"


def resolve_profile(
    name: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    hostname: Optional[str] = None,
    data_root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
    storage_policy: Optional[str] = None,
    shared_group: Optional[str] = None,
) -> StorageProfile:
    """Resolve one profile plus explicit CLI overrides, without filesystem I/O."""
    values = os.environ if environ is None else environ
    selected = (name or "auto").strip().lower()
    if selected == "auto":
        selected = detect_profile_name(values, hostname)

    if selected == "environment":
        profile = _environment_profile(values)
    elif selected in BUILTIN_PROFILES:
        profile = BUILTIN_PROFILES[selected]
    else:
        choices = ", ".join(["auto", *sorted(BUILTIN_PROFILES), "environment"])
        raise ValueError(f"Unknown profile {selected!r}; expected one of: {choices}.")

    policy = (storage_policy or profile.storage_policy).strip().lower()
    if policy not in VALID_STORAGE_POLICIES:
        choices = ", ".join(VALID_STORAGE_POLICIES)
        raise ValueError(f"Unknown storage policy {policy!r}; expected: {choices}.")

    resolved_data = Path(data_root).expanduser() if data_root else profile.data_root
    resolved_cache = Path(cache_root).expanduser() if cache_root else profile.cache_root
    resolved_group = shared_group if shared_group is not None else profile.shared_group
    if policy == "private":
        if shared_group is not None:
            raise ValueError(
                "--shared-group cannot be used with the private storage policy."
            )
        resolved_group = None
    else:
        resolved_group = (resolved_group or "").strip()
        if not resolved_group:
            raise ValueError(
                "A shared Unix group is required for shared-group storage; "
                "pass --shared-group explicitly."
            )

    changed = any(
        value is not None
        for value in (data_root, cache_root, storage_policy, shared_group)
    )
    return replace(
        profile,
        data_root=resolved_data,
        cache_root=resolved_cache,
        storage_policy=policy,
        shared_group=resolved_group,
        source="CLI override" if changed else profile.source,
    )


def render_shell_environment(profile: StorageProfile) -> str:
    """Render a copy/paste-safe POSIX shell environment block."""
    lines = [
        f"export {name}={shlex.quote(value)}"
        for name, value in profile.environment().items()
    ]
    if not profile.shared_group:
        lines.append("unset ANTARES_SHARED_GROUP")
    return "\n".join(lines)


def shell_command(profile: StorageProfile, arguments: list[str]) -> str:
    """Render an environment-scoped command without changing the caller's shell."""
    assignments = [
        f"{name}={shlex.quote(value)}" for name, value in profile.environment().items()
    ]
    return " ".join([*assignments, shlex.join(arguments)])
