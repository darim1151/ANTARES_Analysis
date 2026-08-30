#!/usr/bin/env python3
"""Fail unless ANTARES Analysis imports exclusively from an installed prefix."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
from importlib import metadata
from pathlib import Path


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid-root", type=Path, required=True)
    args = parser.parse_args()

    forbidden_root = args.forbid_root.resolve()
    installed_prefix = Path(sys.prefix).resolve()
    package = importlib.import_module("src")
    if "config" in vars(package):
        raise RuntimeError("src.config was imported eagerly")

    module_names = ["src"] + [f"src.{name}" for name in package.__all__]
    module_names.extend(
        [
            "src.operations.context",
            "src.operations.journal",
            "src.operations.locking",
            "src.operations.plan",
            "src.operations.recovery",
            "src.operations.report",
            "src.operations.science",
            "src.operations.state",
            "src.operations.storage",
            "src.operations.transaction",
            "src.operations.validation",
            "src.operations.writer",
            "src.operations.canary",
            "src.operations.commissioning",
            "src.operations.fetch_checkpoint",
            "src.operations.live_antares",
            "src.operations.query_checkpoint",
        ]
    )
    # Exercise the broker client's real import closure (including its BSON
    # dependency) without making a network request. Project modules import the
    # client lazily, so importing only src.* would not prove this closure.
    module_names.extend(["antares_client", "antares_client.search", "bson"])
    origins = {}
    for module_name in module_names:
        module = importlib.import_module(module_name)
        origin_text = getattr(module, "__file__", None)
        if not origin_text:
            raise RuntimeError(f"{module_name} has no filesystem origin")
        origin = Path(origin_text).resolve()
        if is_within(origin, forbidden_root):
            raise RuntimeError(f"{module_name} imported from checkout: {origin}")
        if not is_within(origin, installed_prefix):
            raise RuntimeError(f"{module_name} imported outside venv: {origin}")
        origins[module_name] = origin

    cli = importlib.import_module("src.cli")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli.main([])
    if status != 0 or "usage: antares-analysis" not in output.getvalue():
        raise RuntimeError("installed CLI help smoke test failed")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli.main(["profile", "list", "--json"])
    if status != 0 or '"middle-earth"' not in output.getvalue():
        raise RuntimeError("installed Phase 2 profile smoke test failed")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli.main(
            [
                "night",
                "plan",
                "2026-06-27",
                "--data-root",
                str(installed_prefix / "missing-data"),
                "--cache-root",
                str(installed_prefix / "missing-cache"),
                "--json",
            ]
        )
    if status != 0 or '"writer_not_enabled_in_this_release"' not in output.getvalue():
        raise RuntimeError("installed Phase 3 planning smoke test failed")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli.main(["night", "ingest", "2026-06-27", "--json"])
    if (
        status != 4
        or '"production_authorization_unavailable"' not in output.getvalue()
        or '"provider_constructed": false' not in output.getvalue()
    ):
        raise RuntimeError("installed Phase 5 authorization smoke test failed")

    version = metadata.version("antares-analysis")
    print(f"Installed package verified: antares-analysis {version}")
    for module_name in sorted(origins):
        print(f"{module_name}: {origins[module_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
