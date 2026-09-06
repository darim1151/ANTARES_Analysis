#!/usr/bin/env python3
"""Compatibility preflight for the explicit RSP shared-group mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, rsp_permissions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate group membership, directory modes, setgid inheritance, "
            "and write access for the ANTARES Analysis shared RSP data root."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=config.DATA_ROOT,
        help=f"Data root to check. Default: {config.DATA_ROOT}",
    )
    parser.add_argument(
        "--expected-group",
        default=config.SHARED_GROUP or config.DEFAULT_SHARED_GROUP,
        help=(
            "Shared Unix group expected on RSP. Default: "
            f"{config.SHARED_GROUP or config.DEFAULT_SHARED_GROUP}"
        ),
    )
    parser.add_argument(
        "--set-umask",
        action="store_true",
        help="Set process umask to 002 before running the write test.",
    )
    parser.add_argument(
        "--allow-missing-setgid",
        action="store_true",
        help=(
            "Do not fail when setgid cannot be observed. Intended only for "
            "local/macOS smoke tests; leave unset on RSP."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.set_umask:
        rsp_permissions.configure_process_umask(policy="shared-group")
    report = rsp_permissions.print_storage_root_report(
        args.data_root,
        policy="shared-group",
        expected_group=args.expected_group,
        write_test=True,
        require_setgid=not args.allow_missing_setgid,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
