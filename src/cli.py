"""Command-line control plane for ANTARES Analysis.

The Phase 2 command surface is intentionally read-only. It resolves storage
profiles, inspects the migrated dataset, diagnoses the execution environment,
and renders Jupyter commands without starting processes or modifying storage.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional, Sequence

from src.cli_diagnostics import (
    collect_data_status,
    collect_doctor_checks,
    doctor_result,
)
from src.cli_notebooks import (
    NOTEBOOKS,
    discover_repo_root,
    notebook_spec,
    render_jupyter_command,
)
from src.cli_profiles import (
    BUILTIN_PROFILES,
    VALID_STORAGE_POLICIES,
    StorageProfile,
    render_shell_environment,
    resolve_profile,
)


DIST_NAME = "antares-analysis"
SOURCE_VERSION = "0.2.0"
PROFILE_CHOICES = ("auto", "environment", *sorted(BUILTIN_PROFILES))


def package_version() -> str:
    """Return installed distribution metadata, or the source-tree version."""
    try:
        return metadata.version(DIST_NAME)
    except metadata.PackageNotFoundError:
        return SOURCE_VERSION


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _help_handler(parser: argparse.ArgumentParser):
    def show_help(_args: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    return show_help


def _add_profile_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="auto",
        help="storage profile to resolve (default: auto)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="read-only override for the durable ANTARES data root",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="override the external cache root (the CLI will not create it)",
    )
    parser.add_argument(
        "--storage-policy",
        choices=VALID_STORAGE_POLICIES,
        help="override the profile storage policy",
    )
    parser.add_argument(
        "--shared-group",
        help="Unix group used only with the shared-group storage policy",
    )


def _profile_from_args(args: argparse.Namespace) -> StorageProfile:
    return resolve_profile(
        args.profile,
        data_root=getattr(args, "data_root", None),
        cache_root=getattr(args, "cache_root", None),
        storage_policy=getattr(args, "storage_policy", None),
        shared_group=getattr(args, "shared_group", None),
    )


def _profile_lines(profile: StorageProfile) -> list[str]:
    return [
        f"Profile:        {profile.name}",
        f"Source:         {profile.source}",
        f"Data root:      {profile.data_root}",
        f"Cache root:     {profile.cache_root}",
        f"Storage policy: {profile.storage_policy}",
        f"Shared group:   {profile.shared_group or '-'}",
        f"Description:    {profile.description}",
    ]


def _handle_profile_list(args: argparse.Namespace) -> int:
    profiles = [BUILTIN_PROFILES[name].as_dict() for name in sorted(BUILTIN_PROFILES)]
    if args.json:
        _print_json(
            {
                "profiles": profiles,
                "selectors": list(PROFILE_CHOICES),
                "read_only": True,
            }
        )
        return 0
    for index, profile in enumerate(profiles):
        if index:
            print()
        print(f"{profile['name']}: {profile['description']}")
        print(f"  data:   {profile['data_root']}")
        print(f"  cache:  {profile['cache_root']}")
        print(f"  policy: {profile['storage_policy']}")
    print("\nDynamic selectors: auto, environment")
    return 0


def _handle_profile_show(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    if args.json:
        _print_json(profile.as_dict())
    else:
        print("\n".join(_profile_lines(profile)))
    return 0


def _handle_profile_export(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    if args.format == "json":
        _print_json(profile.environment())
    else:
        print(render_shell_environment(profile))
    return 0


def _handle_data_status(args: argparse.Namespace) -> int:
    result = collect_data_status(_profile_from_args(args))
    if args.json:
        _print_json(result)
    else:
        profile = result["profile"]
        summary = result["summary"]
        print(f"Dataset:  {'PASS' if result['ok'] else 'FAIL'} (read-only)")
        print(f"Profile:  {profile['name']}")
        print(f"Root:     {profile['data_root']}")
        print(
            "Nights:   "
            f"{summary['manifest_count']} manifests; "
            f"{summary['append_ready_nights']} append-ready"
        )
        print(f"Dates:    {summary['first_date'] or '-'} to {summary['last_date'] or '-'}")
        print(f"Loci:     {summary['total_loci']:,}")
        print(f"Alerts:   {summary['total_alerts']:,}")
        zero_rows = summary["zero_row_nights"]
        print(f"Zero-row: {', '.join(zero_rows) if zero_rows else '-'}")
        for error in result["errors"]:
            print(f"[ERROR] {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    checks = collect_doctor_checks(
        profile,
        repo_root=args.repo_root,
        check_dependencies=not args.no_dependencies,
        check_jupyter=not args.no_jupyter,
    )
    result = doctor_result(profile, checks)
    if args.json:
        _print_json(result)
    else:
        for check in checks:
            print(f"[{check.status.upper():4}] {check.code}: {check.summary}")
            if check.detail:
                print(f"       {check.detail}")
        counts = result["counts"]
        print(
            "\nDoctor result: "
            f"{'PASS' if result['ok'] else 'FAIL'} "
            f"({counts['pass']} pass, {counts['warn']} warn, "
            f"{counts['info']} info, {counts['fail']} fail; read-only)"
        )
    return 0 if result["ok"] else 1


def _handle_jupyter_list(args: argparse.Namespace) -> int:
    root = discover_repo_root(args.repo_root)
    notebooks = [spec.as_dict(root) for spec in NOTEBOOKS]
    if args.json:
        _print_json({"repo_root": str(root), "notebooks": notebooks, "read_only": True})
    else:
        print(f"Repository: {root}")
        for item in notebooks:
            marker = "available" if item["exists"] else "missing"
            print(f"\n{item['alias']} [{marker}]")
            print(f"  {item['purpose']}")
            print(f"  {item['path']}")
    return 0 if all(item["exists"] for item in notebooks) else 1


def _handle_jupyter_env(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    if args.format == "json":
        _print_json(profile.environment())
    else:
        print(render_shell_environment(profile))
    return 0


def _handle_jupyter_command(args: argparse.Namespace) -> int:
    profile = _profile_from_args(args)
    root = discover_repo_root(args.repo_root)
    spec = notebook_spec(args.notebook)
    command = render_jupyter_command(profile, root, spec.alias, args.launcher)
    if args.json:
        _print_json(
            {
                "profile": profile.as_dict(),
                "notebook": spec.as_dict(root),
                "command": command,
                "executed": False,
                "read_only": True,
            }
        )
    else:
        print(command)
        print("# Command rendered only; no Jupyter process was started.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the stable, navigable parser used by the console entry point."""
    parser = argparse.ArgumentParser(
        prog=DIST_NAME,
        description=(
            "Read-only ANTARES Analysis control plane for Middle Earth storage, "
            "dataset diagnostics, and Jupyter navigation."
        ),
        epilog=(
            "Examples: antares-analysis doctor --profile middle-earth; "
            "antares-analysis data status --profile middle-earth; "
            "antares-analysis jupyter command setup --profile middle-earth"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    parser.set_defaults(handler=_help_handler(parser))

    profile_parser = commands.add_parser(
        "profile", help="inspect storage profiles and render environment variables"
    )
    profile_commands = profile_parser.add_subparsers(dest="profile_command", metavar="COMMAND")
    profile_parser.set_defaults(handler=_help_handler(profile_parser))

    profile_list = profile_commands.add_parser("list", help="list built-in profiles")
    profile_list.add_argument("--json", action="store_true", help="emit JSON")
    profile_list.set_defaults(handler=_handle_profile_list)

    profile_show = profile_commands.add_parser("show", help="show a resolved profile")
    _add_profile_options(profile_show)
    profile_show.add_argument("--json", action="store_true", help="emit JSON")
    profile_show.set_defaults(handler=_handle_profile_show)

    profile_export = profile_commands.add_parser(
        "export", help="render environment variables without changing the shell"
    )
    _add_profile_options(profile_export)
    profile_export.add_argument(
        "--format", choices=("shell", "json"), default="shell", help="output format"
    )
    profile_export.set_defaults(handler=_handle_profile_export)

    doctor = commands.add_parser("doctor", help="run read-only environment diagnostics")
    _add_profile_options(doctor)
    doctor.add_argument("--repo-root", type=Path, help="explicit project checkout")
    doctor.add_argument(
        "--no-dependencies", action="store_true", help="skip scientific dependency discovery"
    )
    doctor.add_argument(
        "--no-jupyter", action="store_true", help="skip Jupyter launcher discovery"
    )
    doctor.add_argument("--json", action="store_true", help="emit JSON")
    doctor.set_defaults(handler=_handle_doctor)

    data_parser = commands.add_parser("data", help="inspect migrated ANTARES data")
    data_commands = data_parser.add_subparsers(dest="data_command", metavar="COMMAND")
    data_parser.set_defaults(handler=_help_handler(data_parser))
    data_status = data_commands.add_parser(
        "status", help="summarize manifests and required cumulative products"
    )
    _add_profile_options(data_status)
    data_status.add_argument("--json", action="store_true", help="emit JSON")
    data_status.set_defaults(handler=_handle_data_status)

    jupyter = commands.add_parser("jupyter", help="discover notebooks and render launch commands")
    jupyter_commands = jupyter.add_subparsers(dest="jupyter_command", metavar="COMMAND")
    jupyter.set_defaults(handler=_help_handler(jupyter))

    jupyter_list = jupyter_commands.add_parser("list", help="list supported notebooks")
    jupyter_list.add_argument("--repo-root", type=Path, help="explicit project checkout")
    jupyter_list.add_argument("--json", action="store_true", help="emit JSON")
    jupyter_list.set_defaults(handler=_handle_jupyter_list)

    jupyter_env = jupyter_commands.add_parser(
        "env", help="render notebook environment variables"
    )
    _add_profile_options(jupyter_env)
    jupyter_env.add_argument(
        "--format", choices=("shell", "json"), default="shell", help="output format"
    )
    jupyter_env.set_defaults(handler=_handle_jupyter_env)

    jupyter_command = jupyter_commands.add_parser(
        "command", help="render (but do not execute) a Jupyter Lab command"
    )
    jupyter_command.add_argument("notebook", help="notebook alias from `jupyter list`")
    _add_profile_options(jupyter_command)
    jupyter_command.add_argument("--repo-root", type=Path, help="explicit project checkout")
    jupyter_command.add_argument(
        "--launcher", default="jupyter", help="Jupyter executable name (default: jupyter)"
    )
    jupyter_command.add_argument("--json", action="store_true", help="emit JSON")
    jupyter_command.set_defaults(handler=_handle_jupyter_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the package CLI and return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler = args.handler
        return int(handler(args))
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
