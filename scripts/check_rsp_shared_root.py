#!/usr/bin/env python3
"""Check that the shared RSP ANTARES data root is safe for production writes."""

from __future__ import annotations

import argparse
import os
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
        default=config.EXPECTED_SHARED_GROUP,
        help=f"Shared Unix group expected on RSP. Default: {config.EXPECTED_SHARED_GROUP}",
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
        os.umask(0o002)
    report = rsp_permissions.print_shared_data_root_report(
        args.data_root,
        expected_group=args.expected_group,
        require_setgid=not args.allow_missing_setgid,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
