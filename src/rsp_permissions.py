"""Portable storage-policy checks and permission helpers.

The historical module name is retained for RSP compatibility.  New writers
should use :func:`ensure_storage_path` and :func:`mark_file_for_storage` so an
explicit ``private`` deployment never enters shared-group permission logic.
"""

from __future__ import annotations

import grp
import os
import pwd
import stat
import tempfile
from pathlib import Path

from . import config


DEFAULT_EXPECTED_GROUP = config.DEFAULT_SHARED_GROUP


def get_current_user():
    """Return the current Unix username."""
    return pwd.getpwuid(os.getuid()).pw_name


def get_current_groups():
    """Return active group names for the current process."""
    group_ids = set(os.getgroups())
    group_ids.add(os.getgid())
    groups = []
    for gid in sorted(group_ids):
        try:
            groups.append(grp.getgrgid(gid).gr_name)
        except KeyError:
            groups.append(str(gid))
    return groups


def _mode_string(mode):
    return stat.filemode(mode)


def _group_name(gid):
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def _owner_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def _storage_policy(policy=None):
    value = config.STORAGE_POLICY if policy is None else policy
    return config.normalize_storage_policy(value)


def _shared_group(expected_group=None):
    group = expected_group or config.SHARED_GROUP or config.DEFAULT_SHARED_GROUP
    group = str(group).strip()
    if not group:
        raise ValueError("A non-empty shared group is required in shared-group mode.")
    return group


def path_permission_summary(path):
    """Return an ls -ld style permission summary for one path."""
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "is_directory": False,
            "mode": None,
            "mode_bits": None,
            "owner": None,
            "group": None,
            "group_writable": False,
            "setgid": False,
            "private": False,
            "readable": False,
            "writable": False,
            "executable": False,
        }
    st = path.stat()
    mode_bits = stat.S_IMODE(st.st_mode)
    return {
        "path": str(path),
        "exists": True,
        "is_directory": path.is_dir(),
        "mode": _mode_string(st.st_mode),
        "mode_bits": mode_bits,
        "owner": _owner_name(st.st_uid),
        "group": _group_name(st.st_gid),
        "group_writable": bool(st.st_mode & stat.S_IWGRP),
        "setgid": bool(st.st_mode & stat.S_ISGID),
        "private": not bool(mode_bits & (0o077 | stat.S_ISGID)),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
        "executable": os.access(path, os.X_OK),
    }


def check_umask_recommendation(policy=None):
    """Return the current umask and its policy-specific recommendation."""
    policy = _storage_policy(policy)
    current = os.umask(0)
    os.umask(current)
    recommended = 0o077 if policy == "private" else 0o002
    return {
        "umask": f"{current:03o}",
        "recommended": f"{recommended:03o}",
        "policy_compatible": (
            (current & 0o077) == 0o077
            if policy == "private"
            else not bool(current & 0o020)
        ),
        # Retained for callers of the historical shared-only API.
        "group_write_preserved": not bool(current & 0o020),
    }


def configure_process_umask(policy=None):
    """Set and return the policy-appropriate process umask explicitly."""
    policy = _storage_policy(policy)
    value = 0o077 if policy == "private" else 0o002
    previous = os.umask(value)
    return {
        "policy": policy,
        "previous": f"{previous:03o}",
        "current": f"{value:03o}",
    }


# ---------------------------------------------------------------------------
# Explicit RSP shared-group compatibility helpers
# ---------------------------------------------------------------------------
def mark_file_group_writable(path):
    """Add group read/write bits without changing world permissions."""
    path = Path(path)
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IRGRP | stat.S_IWGRP)


def mark_directory_group_shared(path):
    """Create a directory if needed, then add group rwx and setgid."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID)


def ensure_group_shared_path(path, expected_group=None):
    """Create and mark a path for the explicit RSP shared-group workflow."""
    expected_group = _shared_group(expected_group)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    mark_directory_group_shared(path)
    try:
        group = path_permission_summary(path)["group"]
        if group != expected_group and get_current_user() == path.owner():
            try:
                gid = grp.getgrnam(expected_group).gr_gid
                os.chown(path, -1, gid)
            except (KeyError, PermissionError, OSError) as exc:
                print(f"[WARN] Could not chgrp {path} to {expected_group}: {exc}")
    except Exception as exc:
        print(f"[WARN] Could not verify group for {path}: {exc}")
    return path


# ---------------------------------------------------------------------------
# Policy-neutral writer helpers
# ---------------------------------------------------------------------------
def _ensure_private_path(path):
    """Create only missing directories with owner-only permissions."""
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            # A concurrent creator won the race; never chmod an existing path.
            continue
        directory.chmod(0o700)
    return path


def ensure_storage_path(path, policy=None, expected_group=None):
    """Create a path according to the explicit storage policy.

    Private mode never changes an existing directory's mode, group, or ACL.
    Shared-group mode delegates to the historical group/setgid helper.
    """
    policy = _storage_policy(policy)
    if policy == "private":
        return _ensure_private_path(path)
    return ensure_group_shared_path(path, expected_group=_shared_group(expected_group))


def mark_file_for_storage(path, policy=None, expected_group=None):
    """Apply the explicit policy to one newly written regular file."""
    policy = _storage_policy(policy)
    path = Path(path)
    if not path.exists():
        return path
    if policy == "private":
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    else:
        # Resolve the group only to validate shared-mode configuration.  File
        # group inheritance/chgrp remains the directory workflow's job.
        _shared_group(expected_group)
        mark_file_group_writable(path)
    return path


def ensure_project_directories(
    data_root,
    expected_group=None,
    cache_root=None,
    policy=None,
):
    """Explicitly create the standard durable tree and optional cache root.

    ``cache_root`` is opt-in.  Omitting it can never create ``data_root/cache``.
    """
    data_root = Path(data_root)
    paths = [
        data_root,
        data_root / "data",
        data_root / "data" / "lsst_only",
        data_root / "data" / "lsst_only" / "nightly",
        data_root / "data" / "lsst_only" / "cumulative",
        data_root / "data" / "lsst_only" / "analysis",
        data_root / "analysis",
    ]
    if cache_root is not None:
        cache_path = Path(cache_root)
        if cache_path not in paths:
            paths.append(cache_path)
    for path in paths:
        ensure_storage_path(
            path, policy=policy, expected_group=expected_group
        )
    return paths


def _inspection_paths(data_root, cache_root=None):
    """Return configured paths to inspect without creating any of them."""
    data_root = Path(data_root)
    paths = [("data_root", data_root, True)]
    for name, path in [
        ("data_dir", data_root / "data"),
        ("survey_root", data_root / "data" / "lsst_only"),
        ("nightly_root", data_root / "data" / "lsst_only" / "nightly"),
        ("cumulative_root", data_root / "data" / "lsst_only" / "cumulative"),
        (
            "feature_analysis_root",
            data_root / "data" / "lsst_only" / "analysis",
        ),
        ("analysis_root", data_root / "analysis"),
    ]:
        paths.append((name, path, False))
    if cache_root is not None:
        paths.append(("cache_root", Path(cache_root), False))
    return paths


def check_storage_root(
    data_root,
    cache_root=None,
    policy=None,
    expected_group=None,
    write_test=False,
    require_group_writable=True,
    require_setgid=True,
):
    """Inspect configured storage without normalizing or creating any path."""
    policy = _storage_policy(policy)
    data_root = Path(data_root)
    expected_group = (
        _shared_group(expected_group) if policy == "shared-group" else None
    )
    report = {
        "ok": True,
        "policy": policy,
        "user": get_current_user(),
        "groups": get_current_groups(),
        "expected_group": expected_group,
        "umask": check_umask_recommendation(policy),
        "paths": [],
        "test_file": {},
        "failures": [],
    }

    if policy == "shared-group" and expected_group not in report["groups"]:
        report["failures"].append(
            f"Current user is not in expected shared group {expected_group}."
        )

    for name, path, required in _inspection_paths(data_root, cache_root):
        summary = path_permission_summary(path)
        summary["name"] = name
        summary["required"] = required
        report["paths"].append(summary)
        if not summary["exists"]:
            if required:
                report["failures"].append(f"Missing data root: {path}")
            continue
        if not summary["is_directory"]:
            report["failures"].append(
                f"Configured {name} is not a directory: {path}"
            )
            continue
        for flag in ("readable", "writable", "executable"):
            if not summary[flag]:
                report["failures"].append(f"{path} is not {flag}.")
        if (
            name in {"data_root", "cache_root"}
            and policy == "private"
            and not summary["private"]
        ):
            report["failures"].append(
                f"Private {name} has group/world or setgid bits: {path}."
            )
        if (
            name in {"data_root", "cache_root"}
            and policy == "private"
            and summary["owner"] != report["user"]
        ):
            report["failures"].append(
                f"Private {name} owner is {summary['owner']}, expected "
                f"{report['user']}."
            )
        if policy == "shared-group":
            if summary["group"] != expected_group:
                report["failures"].append(
                    f"{path} group is {summary['group']}, expected "
                    f"{expected_group}."
                )
            if require_group_writable and not summary["group_writable"]:
                report["failures"].append(f"{path} is not group-writable.")
            if require_setgid and path.is_dir() and not summary["setgid"]:
                report["failures"].append(f"{path} is missing setgid.")

    if write_test and data_root.is_dir() and not report["failures"]:
        try:
            with tempfile.TemporaryDirectory(
                dir=data_root, prefix="preflight_"
            ) as tmp:
                tmp_path = Path(tmp)
                if policy == "shared-group":
                    mark_directory_group_shared(tmp_path)
                test_file = tmp_path / f"permission_test_{get_current_user()}.txt"
                test_file.write_text("ok\n", encoding="utf-8")
                mark_file_for_storage(
                    test_file,
                    policy=policy,
                    expected_group=expected_group,
                )
                file_summary = path_permission_summary(test_file)
                report["test_file"] = file_summary
                if policy == "private" and not file_summary["private"]:
                    report["failures"].append(
                        "New private-mode test file is not owner-only."
                    )
                if (
                    policy == "shared-group"
                    and require_group_writable
                    and not file_summary["group_writable"]
                ):
                    report["failures"].append(
                        "New shared-mode test file is not group-writable."
                    )
                if (
                    policy == "shared-group"
                    and file_summary["group"] != expected_group
                ):
                    report["failures"].append(
                        "New shared-mode test file group is "
                        f"{file_summary['group']}, expected {expected_group}."
                    )
        except Exception as exc:
            report["failures"].append(
                f"Could not write temporary preflight file under {data_root}: {exc}"
            )

    report["ok"] = not report["failures"]
    return report


def print_storage_root_report(data_root, **kwargs):
    """Print and return the platform-neutral storage preflight report."""
    report = check_storage_root(data_root, **kwargs)
    print("ANTARES storage-root preflight")
    print("=" * 72)
    print(f"policy         : {report['policy']}")
    print(f"user           : {report['user']}")
    print(f"groups         : {', '.join(report['groups'])}")
    print(f"expected group : {report['expected_group'] or 'not used'}")
    print(
        "umask          : "
        f"{report['umask']['umask']} "
        f"(recommended {report['umask']['recommended']})"
    )
    print("\nPaths")
    for item in report["paths"]:
        status = item["mode"] if item["exists"] else "MISSING"
        print(
            f"  {status} {item['owner'] or '-'} {item['group'] or '-'} "
            f"g+w={item['group_writable']} setgid={item['setgid']} "
            f"{item['path']}"
        )
    if report["test_file"]:
        item = report["test_file"]
        print(
            f"\nTest file     : {item['mode']} {item['owner']} "
            f"{item['group']} {item['path']}"
        )
    if report["failures"]:
        print("\nFailures")
        for failure in report["failures"]:
            print(f"  - {failure}")
    else:
        print("\nPreflight passed: configured storage is safe for this policy.")
    return report


def require_storage_root(data_root, **kwargs):
    """Raise unless the non-mutating storage preflight passes."""
    report = print_storage_root_report(data_root, **kwargs)
    if not report["ok"]:
        joined = "\n".join(f"- {item}" for item in report["failures"])
        raise RuntimeError(
            "Configured ANTARES storage root is not safe for the selected "
            f"policy.\n{joined}"
        )
    return report


# ---------------------------------------------------------------------------
# Historical shared-only preflight API retained as an explicit wrapper
# ---------------------------------------------------------------------------
def check_shared_data_root(
    data_root,
    expected_group=None,
    require_group_writable=True,
    require_setgid=True,
):
    return check_storage_root(
        data_root,
        policy="shared-group",
        expected_group=expected_group,
        write_test=True,
        require_group_writable=require_group_writable,
        require_setgid=require_setgid,
    )


def print_shared_data_root_report(
    data_root,
    expected_group=None,
    require_setgid=True,
):
    return print_storage_root_report(
        data_root,
        policy="shared-group",
        expected_group=expected_group,
        write_test=True,
        require_setgid=require_setgid,
    )


def require_shared_data_root(
    data_root,
    expected_group=None,
    require_setgid=True,
):
    return require_storage_root(
        data_root,
        policy="shared-group",
        expected_group=expected_group,
        write_test=True,
        require_setgid=require_setgid,
    )
