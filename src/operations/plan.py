"""Deterministic, side-effect-free night and backfill planning."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from .context import OperationContext
from .report import Artifact, Evidence, ExitCode, Issue, OperationReport
from .storage import StorageContractError, StorageLayout
from .validation import PreflightCheck, inspect_writer_preflight


PLAN_SCHEMA_VERSION = "1.1"
ACCEPTED_BASELINE_LAST_NIGHT = date(2026, 6, 26)
EXPECTED_STAGES = (
    "plan",
    "precheck",
    "acquire_lock",
    "query",
    "fetch",
    "stage",
    "validate",
    "publish",
    "reconcile_derived_products",
    "complete",
)
DERIVED_PRODUCTS = (
    "data/lsst_only/cumulative/loci_index.parquet",
    "data/lsst_only/cumulative/nightly_summary.parquet",
    "data/lsst_only/analysis/locus_feature_snapshots.parquet",
    "analysis products",
)
VALIDATION_GATES = (
    "query_completed_without_errors",
    "fetch_completed_without_errors",
    "zero_row_success_evidence_if_empty",
    "schema_and_count_validation",
    "coordinate_and_lsst_identity_validation",
    "manifest_and_sibling_path_validation",
    "staged_product_checksum_validation",
    "independent_post_publication_validation",
)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid UTC date {value!r}; expected YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"Invalid UTC date {value!r}; expected canonical YYYY-MM-DD.")
    return parsed


def _plan_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "plan-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]


def _evidence(checks: Sequence[PreflightCheck]) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(check.code, check.status, check.summary, check.details)
        for check in checks
    )


def _refusals(checks: Sequence[PreflightCheck], state: str) -> tuple[Issue, ...]:
    values = [
        Issue(check.code, check.summary)
        for check in checks
        if check.status == "fail"
    ]
    if state == "complete":
        values.append(
            Issue("target_already_complete", "Published partition is already complete.")
        )
    return tuple(values)


def _failed_report(
    context: OperationContext,
    operation: str,
    started: datetime,
    code: str,
    message: str,
    *,
    exit_code: ExitCode,
) -> OperationReport:
    finished = context.now()
    return OperationReport(
        operation=operation,
        success=False,
        status="invalid_request" if exit_code == ExitCode.INVALID_REQUEST else "refused",
        started_at_utc=_iso(started),
        finished_at_utc=_iso(finished),
        elapsed_seconds=max(0.0, (finished - started).total_seconds()),
        exit_code=exit_code,
        errors=(Issue(code, message),) if exit_code == ExitCode.INVALID_REQUEST else (),
        refusal_reasons=(Issue(code, message),) if exit_code != ExitCode.INVALID_REQUEST else (),
        next_actions=("Correct the request or configuration and plan again.",),
    )


def plan_night(context: OperationContext, night: str) -> OperationReport:
    """Plan one UTC night without creating paths, locks, cache, or artifacts."""
    started = context.now()
    try:
        parsed = _parse_date(night)
    except ValueError as exc:
        return _failed_report(
            context, "night.plan", started, "invalid_date", str(exc),
            exit_code=ExitCode.INVALID_REQUEST,
        )

    try:
        layout = StorageLayout.from_context(context)
        target = layout.night(night)
        inspection = layout.inspect_night(target)
        checks = inspect_writer_preflight(context, layout, target, inspection)
    except (StorageContractError, OSError, ValueError) as exc:
        return _failed_report(
            context, "night.plan", started, "storage_contract_invalid", str(exc),
            exit_code=ExitCode.REFUSED,
        )

    identity = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "night": night,
        "profile": context.profile_name,
        "data_root": str(layout.data_root),
        "cache_root": str(layout.cache_root),
        "storage_policy": context.storage_policy,
        "target": target.relative_directory.as_posix(),
        "observed_state": inspection.state,
    }
    plan_id = _plan_id(identity)
    blockers = [
        check.code
        for check in checks
        if check.status == "fail" and check.code != "execution-authorization"
    ]
    if parsed <= ACCEPTED_BASELINE_LAST_NIGHT and inspection.state != "complete":
        blockers.append("accepted_historical_date_requires_scientific_review")
    refusals = list(_refusals(checks, inspection.state))
    if "accepted_historical_date_requires_scientific_review" in blockers:
        refusals.append(
            Issue(
                "accepted_historical_date_requires_scientific_review",
                "Missing or incomplete accepted-history dates require scientific review.",
            )
        )

    finished = context.now()
    status = "planned" if not blockers else "planned_with_blockers"
    next_actions = (
        ("Inspect the existing complete partition; no ingestion is authorized.",)
        if inspection.state == "complete"
        else (
            "Run read-only doctor/status checks and retain this plan as evidence.",
            "Do not ingest until a later release explicitly enables the writer.",
        )
    )
    return OperationReport(
        operation="night.plan",
        success=True,
        status=status,
        started_at_utc=_iso(started),
        finished_at_utc=_iso(finished),
        elapsed_seconds=max(0.0, (finished - started).total_seconds()),
        run_id=None,
        refusal_reasons=tuple(refusals),
        counts={
            "known_loci": inspection.actual_loci,
            "known_alert_rows": inspection.alert_rows,
            "blockers": len(blockers),
        },
        artifacts=(
            Artifact(
                "nightly_partition",
                "durable_science",
                target.relative_directory.as_posix(),
                inspection.state,
            ),
        ),
        evidence=_evidence(checks),
        next_actions=next_actions,
        details={
            "plan_version": PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            "request": {"night": night},
            "configuration": context.configuration(include_paths=True),
            "target": {
                "destination": target.relative_directory.as_posix(),
                "manifest": (target.relative_directory / "manifest.json").as_posix(),
                "lock_resource": None,
                "operations_root_status": "unconfigured",
                "lock_location_enabled_for_production": False,
            },
            "current_partition": inspection.as_dict(),
            "expected_stages": list(EXPECTED_STAGES),
            "prerequisites": [check.code for check in checks],
            "blockers": blockers,
            "storage": {
                "estimated_bytes": None,
                "estimated_inodes": None,
                "staging_same_filesystem_required": True,
                "cache_separate": True,
            },
            "future_execution": {
                "requires_network": True,
                "mutates_durable_science": True,
                "authorized": False,
                "refusal_reason": "writer_not_enabled_in_this_release",
            },
            "derived_reconciliation": {
                "separate_from_publication": True,
                "products": list(DERIVED_PRODUCTS),
            },
            "validation_gates": list(VALIDATION_GATES),
        },
    )


def plan_backfill(
    context: OperationContext,
    start_night: str,
    end_night: str,
) -> OperationReport:
    """Plan an inclusive sequential backlog without executing any night."""
    started = context.now()
    try:
        start = _parse_date(start_night)
        end = _parse_date(end_night)
    except ValueError as exc:
        return _failed_report(
            context, "backfill.plan", started, "invalid_date", str(exc),
            exit_code=ExitCode.INVALID_REQUEST,
        )
    if end < start:
        return _failed_report(
            context,
            "backfill.plan",
            started,
            "invalid_range",
            "Backfill end date must not precede its start date.",
            exit_code=ExitCode.INVALID_REQUEST,
        )
    total = (end - start).days + 1
    if total > 3660:
        return _failed_report(
            context,
            "backfill.plan",
            started,
            "range_too_large",
            "Backfill planning is limited to 3660 inclusive nights.",
            exit_code=ExitCode.INVALID_REQUEST,
        )

    nights = []
    complete = missing = blocked = 0
    cursor = start
    while cursor <= end:
        night = cursor.isoformat()
        report = plan_night(context, night)
        if not report.success:
            state = "blocked"
            blocked += 1
            reason = report.status
            plan_id = None
        else:
            current = report.details["current_partition"]
            state = str(current["state"])
            plan_id = report.details["plan_id"]
            reason = str(current["reason"])
            if state == "complete":
                complete += 1
            elif state == "missing":
                missing += 1
            else:
                blocked += 1
        nights.append(
            {
                "night": night,
                "state": state,
                "reason": reason,
                "plan_id": plan_id,
            }
        )
        cursor += timedelta(days=1)

    identity = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "start": start_night,
        "end": end_night,
        "profile": context.profile_name,
        "data_root": str(context.data_root.resolve(strict=False)),
        "nights": nights,
    }
    finished = context.now()
    return OperationReport(
        operation="backfill.plan",
        success=True,
        status="planned" if blocked == 0 else "planned_with_blockers",
        started_at_utc=_iso(started),
        finished_at_utc=_iso(finished),
        elapsed_seconds=max(0.0, (finished - started).total_seconds()),
        refusal_reasons=(
            Issue(
                "writer_not_enabled_in_this_release",
                "Backfill execution is not enabled in this release.",
            ),
        ),
        counts={
            "requested_nights": total,
            "already_complete": complete,
            "missing": missing,
            "blocked": blocked,
        },
        next_actions=(
            "Review nights in sequential order and stop at the first anomaly.",
            "Do not execute until a later release explicitly enables the writer.",
        ),
        details={
            "plan_version": PLAN_SCHEMA_VERSION,
            "plan_id": _plan_id(identity),
            "inclusive_range": {"start": start_night, "end": end_night},
            "ordering": "sequential_ascending",
            "stop_on_first_anomaly": True,
            "writer_authorization_required": True,
            "writer_enabled": False,
            "reconciliation_strategy": "separate_after_each_published_night",
            "recommended_ramp": [1, 3, 7, "remainder"],
            "nights": nights,
        },
    )
