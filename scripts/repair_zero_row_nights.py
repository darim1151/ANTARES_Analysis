#!/usr/bin/env python3
"""Safely revalidate valid empty LSST nights and rebuild cumulative products."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import history, rsp_permissions  # noqa: E402


class RepairError(RuntimeError):
    """Raised when the repair cannot be completed without ambiguity."""


DEFAULT_MIN_HEADROOM_MIB = 500


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path = Path(path)
    rsp_permissions.ensure_storage_path(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    rsp_permissions.mark_file_for_storage(path)


def _mark_file_for_storage(path):
    """Apply the configured private/shared policy to a generated file."""
    path = Path(path)
    if path.exists():
        rsp_permissions.mark_file_for_storage(path)


def _parse_human_bytes(value):
    value = value.strip()
    if not value:
        raise ValueError("empty size")
    suffix = value[-1].upper()
    factors = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
        "P": 1024**5,
    }
    if suffix in factors:
        return int(float(value[:-1]) * factors[suffix])
    # quota(1) reports unsuffixed block counts in KiB.
    return int(value) * 1024


def _quota_headroom_bytes(path):
    """Return the narrowest available quota/filesystem headroom."""
    candidates = [("filesystem", shutil.disk_usage(path).free)]
    try:
        result = subprocess.run(
            ["quota", "-s"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError):
        result = None
    if result is not None:
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            try:
                used = _parse_human_bytes(fields[0])
                soft = _parse_human_bytes(fields[1])
                hard = _parse_human_bytes(fields[2])
            except (TypeError, ValueError):
                continue
            limits = [limit for limit in (soft, hard) if limit > 0]
            if limits:
                candidates.append(("user-quota", max(0, min(limits) - used)))
    return min(candidates, key=lambda item: item[1])


def _ensure_headroom(path, minimum_mib):
    minimum_bytes = int(minimum_mib) * 1024**2
    source, available = _quota_headroom_bytes(path)
    if available < minimum_bytes:
        raise RepairError(
            "Insufficient write headroom: "
            f"{available / 1024**2:.1f} MiB available via {source}; "
            f"{minimum_mib} MiB required."
        )
    return {
        "source": source,
        "available_bytes": int(available),
        "available_mib": available / 1024**2,
        "required_mib": int(minimum_mib),
    }


def _is_within(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _source_targets(data_root, dates):
    targets = {}
    for date_utc in dates:
        targets[f"manifest:{date_utc}"] = history.nightly_paths(
            data_root, date_utc
        )["manifest"]
    cumulative = history.cumulative_paths(data_root)
    targets["cumulative:loci_index"] = cumulative["loci_index"]
    targets["cumulative:nightly_summary"] = cumulative["nightly_summary"]
    return targets


def _stage_repair(data_root, dates, staging_dir):
    overrides = {
        date_utc: history.revalidate_zero_row_night(data_root, date_utc)
        for date_utc in dates
    }
    staging_dir = Path(staging_dir)
    staged_manifests = {}
    for date_utc, payload in overrides.items():
        path = staging_dir / "manifests" / date_utc / "manifest.json"
        _write_json(path, payload)
        staged_manifests[f"manifest:{date_utc}"] = path

    staged_cumulative = staging_dir / "cumulative"
    loci_index, nightly_summary = history.update_cumulative_indexes(
        data_root,
        output_dir=staged_cumulative,
        manifest_overrides=overrides,
    )
    staged_paths = {
        **staged_manifests,
        "cumulative:loci_index": staged_cumulative / "loci_index.parquet",
        "cumulative:nightly_summary": (
            staged_cumulative / "nightly_summary.parquet"
        ),
    }

    accepted_dates = set(nightly_summary["date_utc"].astype(str))
    for date_utc in dates:
        if date_utc not in accepted_dates:
            raise RepairError(
                f"Revalidated date {date_utc} is absent from staged "
                "nightly_summary.parquet."
            )
    if len(nightly_summary) != len(accepted_dates):
        raise RepairError(
            "Staged nightly_summary.parquet contains duplicate dates."
        )

    expected_loci_rows = 0
    nightly_root = history.survey_data_root(data_root) / "nightly"
    for manifest_path in sorted(nightly_root.glob("*/*/*/manifest.json")):
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest = overrides.get(manifest.get("date_utc"), manifest)
        if manifest.get("status") not in history.APPENDABLE_STATUSES:
            continue
        if manifest.get("validation", {}).get("append_ready") is not True:
            continue
        expected_loci_rows += int(manifest["actual_loci"])
    if len(loci_index) != expected_loci_rows:
        raise RepairError(
            "Staged loci_index.parquet row count does not match accepted "
            f"nightly manifests: staged={len(loci_index)}, "
            f"expected={expected_loci_rows}."
        )

    # Force a fresh independent read of both staged products before promotion.
    staged_loci = pd.read_parquet(staged_paths["cumulative:loci_index"])
    staged_summary = pd.read_parquet(
        staged_paths["cumulative:nightly_summary"]
    )
    if len(staged_loci) != len(loci_index):
        raise RepairError("Staged loci_index.parquet did not round-trip.")
    if len(staged_summary) != len(nightly_summary):
        raise RepairError("Staged nightly_summary.parquet did not round-trip.")
    return overrides, staged_paths, {
        "loci_index_rows": int(len(staged_loci)),
        "nightly_summary_rows": int(len(staged_summary)),
        "first_date": (
            str(staged_summary["date_utc"].min())
            if not staged_summary.empty
            else None
        ),
        "last_date": (
            str(staged_summary["date_utc"].max())
            if not staged_summary.empty
            else None
        ),
    }


def _fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    descriptor = os.open(str(Path(path)), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_promotable(label, path, expected_sha256, staged_summary):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RepairError(f"Promotable file is missing or empty: {label}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RepairError(
            f"Promotable SHA-256 mismatch for {label}: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )
    if label.startswith("manifest:"):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validation = payload.get("validation", {})
        if payload.get("status") not in history.APPENDABLE_STATUSES:
            raise RepairError(f"Promotable manifest is not complete: {label}")
        if validation.get("append_ready") is not True:
            raise RepairError(
                f"Promotable manifest is not append-ready: {label}"
            )
        if validation.get("lsst_only_pass") is not True:
            raise RepairError(
                f"Promotable manifest did not pass LSST-only validation: "
                f"{label}"
            )
    elif label == "cumulative:loci_index":
        frame = pd.read_parquet(path)
        if len(frame) != staged_summary["loci_index_rows"]:
            raise RepairError(
                "Promotable loci_index.parquet row count changed."
            )
    elif label == "cumulative:nightly_summary":
        frame = pd.read_parquet(path)
        if len(frame) != staged_summary["nightly_summary_rows"]:
            raise RepairError(
                "Promotable nightly_summary.parquet row count changed."
            )
        first_date = (
            str(frame["date_utc"].min()) if not frame.empty else None
        )
        last_date = (
            str(frame["date_utc"].max()) if not frame.empty else None
        )
        if (
            first_date != staged_summary["first_date"]
            or last_date != staged_summary["last_date"]
        ):
            raise RepairError(
                "Promotable nightly_summary.parquet date range changed."
            )


def _prepare_atomic_promotions(
    source_targets,
    staged_paths,
    staged_hashes,
    staged_summary,
):
    prepared = {}
    created = []
    try:
        for label, destination in source_targets.items():
            destination = Path(destination)
            temporary = destination.with_name(
                f".{destination.name}.zero-row-repair-{uuid.uuid4().hex}.tmp"
            )
            created.append(temporary)
            shutil.copy2(staged_paths[label], temporary)
            _mark_file_for_storage(temporary)
            _fsync_file(temporary)
            if temporary.stat().st_size != Path(
                staged_paths[label]
            ).stat().st_size:
                raise RepairError(
                    f"Promotable size mismatch for {label}."
                )
            _verify_promotable(
                label,
                temporary,
                staged_hashes[label],
                staged_summary,
            )
            prepared[label] = temporary
    except Exception:
        for temporary in created:
            temporary.unlink(missing_ok=True)
        raise
    return prepared


def repair_zero_row_nights(
    data_root,
    dates,
    backup_dir,
    apply=False,
    min_headroom_mib=DEFAULT_MIN_HEADROOM_MIB,
):
    data_root = Path(data_root).resolve()
    dates = tuple(dict.fromkeys(dates))
    if not data_root.is_dir():
        raise RepairError(f"Data root does not exist: {data_root}")
    if not dates:
        raise RepairError("At least one --date is required.")
    if int(min_headroom_mib) < 0:
        raise RepairError("Minimum headroom cannot be negative.")

    preview = {
        date_utc: history.revalidate_zero_row_night(data_root, date_utc)
        for date_utc in dates
    }
    if not apply:
        return {
            "applied": False,
            "data_root": str(data_root),
            "dates": list(dates),
            "policy": history.ZERO_ROW_REVALIDATION_POLICY,
            "append_ready": {
                date_utc: payload["validation"]["append_ready"]
                for date_utc, payload in preview.items()
            },
        }

    if backup_dir is None:
        raise RepairError("--backup-dir is required with --apply.")
    quota_preflight = _ensure_headroom(data_root, min_headroom_mib)
    backup_dir = Path(backup_dir).resolve()
    if backup_dir.exists():
        raise RepairError(
            f"Backup directory already exists; refusing overwrite: {backup_dir}"
        )
    if _is_within(backup_dir, data_root):
        raise RepairError("Backup directory must be outside the data root.")
    rsp_permissions.ensure_storage_path(backup_dir)

    source_targets = _source_targets(data_root, dates)
    missing = [
        f"{label}: {path}"
        for label, path in source_targets.items()
        if not path.is_file()
    ]
    if missing:
        raise RepairError(
            "Required source files are missing:\n" + "\n".join(missing)
        )

    original_dir = backup_dir / "original"
    original_paths = {}
    for label, source in source_targets.items():
        relative = source.relative_to(data_root)
        backup = original_dir / relative
        rsp_permissions.ensure_storage_path(backup.parent)
        shutil.copy2(source, backup)
        _mark_file_for_storage(backup)
        original_paths[label] = backup

    before_hashes = {
        label: _sha256(path) for label, path in source_targets.items()
    }
    overrides, staged_paths, staged_summary = _stage_repair(
        data_root, dates, backup_dir / "staged"
    )
    staged_hashes = {
        label: _sha256(path) for label, path in staged_paths.items()
    }

    unchanged_hashes = {
        label: _sha256(path) for label, path in source_targets.items()
    }
    if unchanged_hashes != before_hashes:
        raise RepairError(
            "A source target changed while staging; no repair was promoted."
        )

    quota_before_promotion = _ensure_headroom(
        data_root, min_headroom_mib
    )
    prepared = _prepare_atomic_promotions(
        source_targets,
        staged_paths,
        staged_hashes,
        staged_summary,
    )
    final_unchanged_hashes = {
        label: _sha256(path) for label, path in source_targets.items()
    }
    if final_unchanged_hashes != before_hashes:
        for temporary in prepared.values():
            temporary.unlink(missing_ok=True)
        raise RepairError(
            "A source target changed while preparing atomic promotions; "
            "no repair was promoted."
        )

    promoted = []
    try:
        for label, destination in source_targets.items():
            os.replace(prepared[label], destination)
            promoted.append(label)
            _mark_file_for_storage(destination)
            _fsync_directory(Path(destination).parent)
            _verify_promotable(
                label,
                destination,
                staged_hashes[label],
                staged_summary,
            )
        after_hashes = {
            label: _sha256(path) for label, path in source_targets.items()
        }
        if after_hashes != staged_hashes:
            raise RepairError(
                "A promoted source file does not match its validated staged "
                "SHA-256."
            )
    except Exception as exc:
        for label, temporary in prepared.items():
            if label not in promoted:
                temporary.unlink(missing_ok=True)
        raise RepairError(
            f"Atomic promotion failed after {promoted}: {exc}. "
            "No rollback was attempted, so no valid destination was "
            "truncated. Verified originals remain in the backup directory."
        ) from exc

    report = {
        "applied": True,
        "data_root": str(data_root),
        "dates": list(dates),
        "policy": history.ZERO_ROW_REVALIDATION_POLICY,
        "backup_dir": str(backup_dir),
        "before_sha256": before_hashes,
        "staged_sha256": staged_hashes,
        "after_sha256": after_hashes,
        "staged_cumulative": staged_summary,
        "quota_preflight": quota_preflight,
        "quota_before_promotion": quota_before_promotion,
        "promotion_mode": "same_filesystem_atomic_replace_no_rollback",
        "validation": {
            date_utc: payload["validation"]
            for date_utc, payload in overrides.items()
        },
    }
    _write_json(backup_dir / "repair_report.json", report)
    return report


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate completed, schema-valid, error-free zero-row LSST "
            "nights and rebuild cumulative products through the history "
            "pipeline."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--date",
        action="append",
        required=True,
        help="UTC date to revalidate (repeat for multiple dates).",
    )
    parser.add_argument(
        "--backup-dir",
        help="Fresh directory outside the data root for originals and staging.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Promote the validated staged repair; otherwise perform dry-run.",
    )
    parser.add_argument(
        "--min-quota-headroom-mib",
        type=int,
        default=DEFAULT_MIN_HEADROOM_MIB,
        help=(
            "Minimum required per-user quota/filesystem headroom before "
            "staging promotions (default: 500 MiB)."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    print(f"ANTARES data root: {Path(args.data_root).resolve()}")
    print(f"Zero-row policy: {history.ZERO_ROW_REVALIDATION_POLICY}")
    try:
        report = repair_zero_row_nights(
            args.data_root,
            args.date,
            args.backup_dir,
            apply=args.apply,
            min_headroom_mib=args.min_quota_headroom_mib,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
