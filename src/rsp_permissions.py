"""RSP shared-directory permission checks and helpers.

The ANTARES Analysis production workflow writes shared Parquet, manifests,
caches, and figures under a Rubin Science Platform data root. These helpers
keep those outputs group-readable/writable without making anything
world-writable.
"""

from __future__ import annotations

import grp
import os
import pwd
import stat
import tempfile
from pathlib import Path

from . import config


DEFAULT_EXPECTED_GROUP = config.EXPECTED_SHARED_GROUP


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


def path_permission_summary(path):
    """Return an ls -ld style permission summary for one path."""
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "mode": None,
            "owner": None,
            "group": None,
            "group_writable": False,
            "setgid": False,
            "readable": False,
            "writable": False,
            "executable": False,
        }
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mode": _mode_string(st.st_mode),
        "owner": _owner_name(st.st_uid),
        "group": _group_name(st.st_gid),
        "group_writable": bool(st.st_mode & stat.S_IWGRP),
        "setgid": bool(st.st_mode & stat.S_ISGID),
        "readable": os.access(path, os.R_OK),
        "writable": os.access(path, os.W_OK),
        "executable": os.access(path, os.X_OK),
    }


def check_umask_recommendation():
    """Return current umask and whether it is compatible with group writes."""
    current = os.umask(0)
    os.umask(current)
    return {
        "umask": f"{current:03o}",
        "group_write_preserved": not bool(current & 0o020),
        "recommended": "002",
    }


def mark_file_group_writable(path):
    """Add group read/write bits to a file without changing world permissions."""
    path = Path(path)
    if not path.exists():
        return
    path.chmod(path.stat().st_mode | stat.S_IRGRP | stat.S_IWGRP)


def mark_directory_group_shared(path):
    """Make a directory group-shared and setgid without world-write."""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID)


def ensure_group_shared_path(path, expected_group=DEFAULT_EXPECTED_GROUP):
    """Create a directory and mark every new ancestor from the path as shared."""
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


def ensure_project_directories(data_root, expected_group=DEFAULT_EXPECTED_GROUP):
    """Create standard project directories with group-shared permissions."""
    data_root = Path(data_root)
    paths = [
        data_root,
        data_root / "cache",
        data_root / "data",
        data_root / "data" / "lsst_only",
        data_root / "data" / "lsst_only" / "nightly",
        data_root / "data" / "lsst_only" / "cumulative",
        data_root / "data" / "lsst_only" / "analysis",
        data_root / "analysis",
    ]
    for path in paths:
        ensure_group_shared_path(path, expected_group=expected_group)
    return paths


def _standard_paths(data_root):
    data_root = Path(data_root)
    paths = []
    production_parent = Path("/home/ivezic/AntaresAlerts")
    try:
        is_production_tree = data_root.is_relative_to(production_parent)
    except AttributeError:
        is_production_tree = str(data_root).startswith(str(production_parent) + os.sep)
    if is_production_tree:
        paths.extend([Path("/home/ivezic"), production_parent])
    else:
        paths.append(data_root.parent)
    paths.extend([
        data_root,
        data_root / "cache",
        data_root / "data" / "lsst_only",
        data_root / "data" / "lsst_only" / "nightly",
        data_root / "data" / "lsst_only" / "cumulative",
        data_root / "analysis",
    ])
    return paths


def check_shared_data_root(
    data_root,
    expected_group=DEFAULT_EXPECTED_GROUP,
    require_group_writable=True,
    require_setgid=True,
):
    """Return a machine-readable preflight report for the shared data root."""
    data_root = Path(data_root)
    report = {
        "ok": True,
        "user": get_current_user(),
        "groups": get_current_groups(),
        "expected_group": expected_group,
        "umask": check_umask_recommendation(),
        "paths": [],
        "test_file": {},
        "failures": [],
    }
    if expected_group not in report["groups"]:
        report["failures"].append(
            f"Current user is not in expected RSP group {expected_group}. "
            "Add the user in Comanage and re-enter the Notebook Aspect."
        )

    try:
        ensure_project_directories(data_root, expected_group=expected_group)
    except Exception as exc:
        report["failures"].append(f"Could not create project directories: {exc}")

    for path in _standard_paths(data_root):
        summary = path_permission_summary(path)
        report["paths"].append(summary)
        if not summary["exists"]:
            report["failures"].append(f"Missing path: {path}")
            continue
        if path == data_root or str(path).startswith(str(data_root)):
            for flag in ["readable", "writable", "executable"]:
                if not summary[flag]:
                    report["failures"].append(f"{path} is not {flag}.")
            if require_group_writable and not summary["group_writable"]:
                report["failures"].append(
                    f"{path} is not group-writable; run chmod g+rwX {path}."
                )
            if require_setgid and path.is_dir() and not summary["setgid"]:
                report["failures"].append(
                    f"{path} is missing setgid; run chmod g+s {path}."
                )

    try:
        with tempfile.TemporaryDirectory(dir=data_root, prefix="preflight_") as tmp:
            tmp_path = Path(tmp)
            mark_directory_group_shared(tmp_path)
            test_file = tmp_path / f"permission_test_{get_current_user()}.txt"
            test_file.write_text("ok\n", encoding="utf-8")
            mark_file_group_writable(test_file)
            file_summary = path_permission_summary(test_file)
            report["test_file"] = file_summary
            if expected_group in report["groups"] and file_summary["group"] != expected_group:
                report["failures"].append(
                    f"New test file group is {file_summary['group']}, expected {expected_group}. "
                    "Run chgrp -R and setgid on the shared data root."
                )
            if require_group_writable and not file_summary["group_writable"]:
                report["failures"].append(
                    "New test file is not group-writable; set os.umask(0o002) "
                    "and ensure writers call chmod g+rw."
                )
    except Exception as exc:
        report["failures"].append(f"Could not create/write test file under {data_root}: {exc}")

    report["ok"] = not report["failures"]
    return report


def print_shared_data_root_report(
    data_root,
    expected_group=DEFAULT_EXPECTED_GROUP,
    require_setgid=True,
):
    """Print and return the shared data-root preflight report."""
    report = check_shared_data_root(
        data_root,
        expected_group=expected_group,
        require_setgid=require_setgid,
    )
    print("RSP shared data-root preflight")
    print("=" * 72)
    print(f"user           : {report['user']}")
    print(f"groups         : {', '.join(report['groups'])}")
    print(f"expected group : {expected_group}")
    print(
        "umask          : "
        f"{report['umask']['umask']} "
        f"(recommended {report['umask']['recommended']}, "
        f"group-write OK={report['umask']['group_write_preserved']})"
    )
    print("\nPaths")
    for item in report["paths"]:
        status = item["mode"] if item["exists"] else "MISSING"
        print(
            f"  {status} {item['owner'] or '-'} {item['group'] or '-'} "
            f"g+w={item['group_writable']} setgid={item['setgid']} "
            f"rwx={item['readable']}/{item['writable']}/{item['executable']} "
            f"{item['path']}"
        )
    if report["test_file"]:
        item = report["test_file"]
        print(
            f"\nTest file     : {item['mode']} {item['owner']} {item['group']} "
            f"g+w={item['group_writable']} {item['path']}"
        )
    if report["failures"]:
        print("\nFailures")
        for failure in report["failures"]:
            print(f"  - {failure}")
    else:
        print("\nPreflight passed: shared root is safe for production writes.")
    return report


def require_shared_data_root(
    data_root,
    expected_group=DEFAULT_EXPECTED_GROUP,
    require_setgid=True,
):
    """Raise RuntimeError unless the shared data root passes preflight."""
    report = print_shared_data_root_report(
        data_root,
        expected_group=expected_group,
        require_setgid=require_setgid,
    )
    if not report["ok"]:
        joined = "\n".join(f"- {failure}" for failure in report["failures"])
        raise RuntimeError(
            "Shared RSP data root is not safe for production writes.\n"
            f"{joined}\n\n"
            "Likely setup: create/add users to the Comanage group, then run:\n"
            f"  chgrp -R {expected_group} {data_root}\n"
            f"  chmod -R g+rwX {data_root}\n"
            f"  find {data_root} -type d -exec chmod g+s {{}} \\;\n"
        )
    return report
