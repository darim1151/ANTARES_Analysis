"""Command-line control plane for ANTARES Analysis.

Ordinary inspection, planning, recovery, and Jupyter-navigation commands are
read-only.  ``night qualify`` is the single explicit Phase 6 exception: with
deliberate LIVE_ANTARES_READ authority it may query ANTARES and retain a
non-authoritative candidate under one exact canary run root.  Production
publication, reconciliation, and cache mutation remain unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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
from src.operations import context_from_profile, plan_backfill, plan_night
from src.operations.recovery import (
    RecoveryDisposition,
    inspect_recovery,
)
from src.operations.writer import WriterError, production_ingest_refusal


DIST_NAME = "antares-analysis"
SOURCE_VERSION = "0.4.1"
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


def _render_operation_report(report, *, json_output: bool) -> int:
    if json_output:
        print(report.to_json())
    else:
        print(report.render_human())
        plan_id = report.details.get("plan_id")
        if plan_id:
            print(f"Plan ID:   {plan_id}")
    return int(report.exit_code)


def _operation_context_from_args(args: argparse.Namespace):
    profile = _profile_from_args(args)
    return context_from_profile(
        profile,
        execution_metadata={"interface": "cli", "version": package_version()},
    )


def _handle_night_plan(args: argparse.Namespace) -> int:
    report = plan_night(_operation_context_from_args(args), args.date)
    return _render_operation_report(report, json_output=args.json)


def _handle_night_ingest(args: argparse.Namespace) -> int:
    report = production_ingest_refusal(args.date)
    return _render_operation_report(report, json_output=args.json)


def _handle_night_qualify(args: argparse.Namespace) -> int:
    """Run the explicit Phase 6 live-read, non-publication path."""
    from astropy.time import Time

    from src.cli_profiles import MIDDLE_EARTH_CANARY_ROOT
    from src.operations.commissioning import TARGET_DATE_UTC, qualify_live_night
    from src.operations.live_antares import (
        LIVE_ANTARES_READ,
        LiveAntaresProvider,
        LiveAntaresReadCapability,
    )
    from src.operations.science import NightScienceRequest
    from src.operations.storage import SyntheticWriteCapability
    from src.operations.writer import NightExecutionSpec

    profile = _profile_from_args(args)
    if args.date != TARGET_DATE_UTC:
        raise ValueError(f"Phase 6 qualification is restricted to {TARGET_DATE_UTC}.")
    if not args.authorize_live_read:
        raise ValueError("--authorize-live-read is required for the live provider.")
    run_root = MIDDLE_EARTH_CANARY_ROOT / args.run_id
    write_capability = SyntheticWriteCapability.for_arnor_canary_root(
        run_root, args.run_id
    )
    live_capability = LiveAntaresReadCapability.for_arnor_commissioning(
        run_root,
        run_id=args.run_id,
        target_date_utc=args.date,
        release_sha=args.release_sha,
        authority=LIVE_ANTARES_READ,
    )
    start = float(
        Time(f"{args.date}T00:00:00", format="isot", scale="utc").mjd
    )
    request = NightScienceRequest(
        args.date,
        start,
        start + 1.0,
        target_loci=None,
        ingested_at_utc=datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        range_label=f"ANTARES commissioning {args.date}",
    )
    provider = LiveAntaresProvider(
        live_capability,
        max_query_attempts=args.query_attempts,
        max_fetch_attempts=args.fetch_attempts,
        max_fetch_workers=args.fetch_workers,
    )
    spec = NightExecutionSpec(
        args.attempt_id or args.run_id,
        f"phase6-{args.date}",
        args.release_sha,
        "phase6-live-antares-commissioning-v1",
        request,
    )
    result = qualify_live_night(
        write_capability,
        live_capability,
        provider,
        spec,
        production_data_root=profile.data_root,
        production_cache_root=profile.cache_root,
    )
    return _render_operation_report(result.report, json_output=args.json)


def _handle_recovery_inspect(args: argparse.Namespace) -> int:
    assessment = inspect_recovery(
        args.journal,
        target_path=args.target,
        stage_path=args.stage,
        lock_path=args.lock,
    )
    if args.json:
        _print_json(assessment.as_dict())
    else:
        print(f"Recovery: {assessment.summary}")
        print("Disposition: " + ", ".join(
            item.value for item in assessment.dispositions
        ))
        print(f"Lock owner: {assessment.lock_owner_status}")
        for disagreement in assessment.disagreements:
            print(f"[WARN] {disagreement}")
    return (
        1
        if RecoveryDisposition.REQUIRES_OPERATOR_DECISION
        in assessment.dispositions
        else 0
    )


def _handle_backfill_plan(args: argparse.Namespace) -> int:
    report = plan_backfill(
        _operation_context_from_args(args), args.start_date, args.end_date
    )
    return _render_operation_report(report, json_output=args.json)


def build_parser() -> argparse.ArgumentParser:
    """Build the stable, navigable parser used by the console entry point."""
    parser = argparse.ArgumentParser(
        prog=DIST_NAME,
        description=(
            "ANTARES Analysis control plane for Middle Earth diagnostics, planning, "
            "Jupyter navigation, and explicit non-publication live qualification."
        ),
        epilog=(
            "Examples: antares-analysis doctor --profile middle-earth; "
            "antares-analysis data status --profile middle-earth; "
            "antares-analysis night plan 2026-06-27 --profile middle-earth; "
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

    night_parser = commands.add_parser(
        "night", help="plan nights or run the sealed non-publication qualification"
    )
    night_commands = night_parser.add_subparsers(dest="night_command", metavar="COMMAND")
    night_parser.set_defaults(handler=_help_handler(night_parser))
    night_plan = night_commands.add_parser(
        "plan", help="build a side-effect-free future-writer plan for one UTC date"
    )
    night_plan.add_argument("date", help="UTC night in canonical YYYY-MM-DD form")
    _add_profile_options(night_plan)
    night_plan.add_argument("--json", action="store_true", help="emit versioned JSON")
    night_plan.set_defaults(handler=_handle_night_plan)
    night_ingest = night_commands.add_parser(
        "ingest",
        help="show the fail-closed production authorization refusal",
    )
    night_ingest.add_argument("date", help="UTC night in canonical YYYY-MM-DD form")
    night_ingest.add_argument("--json", action="store_true", help="emit versioned JSON")
    night_ingest.set_defaults(handler=_handle_night_ingest)
    night_qualify = night_commands.add_parser(
        "qualify",
        help="run the explicit Phase 6 live-read commissioning path without publication",
    )
    night_qualify.add_argument("date", help="UTC night in canonical YYYY-MM-DD form")
    night_qualify.add_argument("--run-id", required=True, help="exact pre-created canary run id")
    night_qualify.add_argument(
        "--attempt-id",
        help=(
            "unique transaction identity inside the same run root; use a new value "
            "to resume validated Phase 6 checkpoints after a failed attempt"
        ),
    )
    night_qualify.add_argument(
        "--release-sha", required=True, help="full committed candidate SHA"
    )
    night_qualify.add_argument(
        "--authorize-live-read",
        action="store_true",
        help="explicitly issue LIVE_ANTARES_READ for this run",
    )
    night_qualify.add_argument(
        "--query-attempts",
        type=int,
        default=2,
        help="whole-tile attempts (maximum 2)",
    )
    night_qualify.add_argument(
        "--fetch-attempts",
        type=int,
        default=3,
        help="attempts per locus (maximum 3)",
    )
    night_qualify.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        help="parallel locus fetches (maximum 4)",
    )
    _add_profile_options(night_qualify)
    night_qualify.add_argument("--json", action="store_true", help="emit versioned JSON")
    night_qualify.set_defaults(handler=_handle_night_qualify)

    backfill_parser = commands.add_parser(
        "backfill", help="plan sequential backlog handling without executing it"
    )
    backfill_commands = backfill_parser.add_subparsers(
        dest="backfill_command", metavar="COMMAND"
    )
    backfill_parser.set_defaults(handler=_help_handler(backfill_parser))
    backfill_plan = backfill_commands.add_parser(
        "plan", help="plan an inclusive sequential UTC-date range"
    )
    backfill_plan.add_argument("start_date", help="inclusive start YYYY-MM-DD")
    backfill_plan.add_argument("end_date", help="inclusive end YYYY-MM-DD")
    _add_profile_options(backfill_plan)
    backfill_plan.add_argument("--json", action="store_true", help="emit versioned JSON")
    backfill_plan.set_defaults(handler=_handle_backfill_plan)

    recovery = commands.add_parser(
        "recovery", help="inspect durable interrupted-writer evidence read-only"
    )
    recovery_commands = recovery.add_subparsers(
        dest="recovery_command", metavar="COMMAND"
    )
    recovery.set_defaults(handler=_help_handler(recovery))
    recovery_inspect = recovery_commands.add_parser(
        "inspect", help="classify one journal/target/stage/lock evidence set"
    )
    recovery_inspect.add_argument("journal", type=Path, help="transaction journal JSON")
    recovery_inspect.add_argument("--target", type=Path, required=True)
    recovery_inspect.add_argument("--stage", type=Path, required=True)
    recovery_inspect.add_argument("--lock", type=Path, required=True)
    recovery_inspect.add_argument("--json", action="store_true", help="emit JSON")
    recovery_inspect.set_defaults(handler=_handle_recovery_inspect)

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
    except (OSError, ValueError, WriterError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
