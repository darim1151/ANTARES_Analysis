"""Command-line entry point for the packaged ANTARES Analysis project.

Phase 1 deliberately exposes only package identity and help. Operational
writer commands will be added after their safety contracts, locking, and
transactional publication behavior are implemented and tested.
"""

from __future__ import annotations

import argparse
from importlib import metadata
from typing import Optional, Sequence


DIST_NAME = "antares-analysis"
SOURCE_VERSION = "0.1.0"


def package_version() -> str:
    """Return installed distribution metadata, or the source-tree version."""
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        return SOURCE_VERSION


def build_parser() -> argparse.ArgumentParser:
    """Build the stable top-level parser used by the console entry point."""
    parser = argparse.ArgumentParser(
        prog=DIST_NAME,
        description=(
            "Portable ANTARES/LSST analysis workflows. Production ingestion "
            "commands are not enabled in this release scaffold."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the package CLI and return a process exit status."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
