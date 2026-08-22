#!/usr/bin/env python3
"""Create a non-destructive, migration-safe audit of an ANTARES data root.

The script deliberately does not import the project package.  It is intended
to be copied with the data and run in both the source and destination
environments, even when the rest of the analysis dependencies are absent.

Exit codes:

* 0: the audit completed and no integrity issues were found;
* 1: the audit completed, all reports were written, and integrity issues exist;
* 2: preflight, dependency, usage, or runtime failure prevented a trustworthy
  complete audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import socket
import stat
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path


EXIT_OK = 0
EXIT_INTEGRITY_ISSUES = 1
EXIT_ERROR = 2
AUDIT_SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 65_536
SOURCE_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

OUTPUT_NAMES = (
    "summary.json",
    "nightly_manifest_table.csv",
    "nightly_manifest_table.txt",
    "file_counts.json",
    "size_summary.txt",
    "nightly_parquet_sha256.txt",
    "cumulative_parquet_sha256.txt",
    "all_science_files_sha256.txt",
    "feature_coverage.csv",
    "feature_coverage_summary.txt",
)

EVIDENCE_INCOMPLETE_CODES = frozenset(
    {
        "analysis_path_not_directory",
        "checksum_read_error",
        "cumulative_file_unreadable",
        "cumulative_semantic_column_missing",
        "cumulative_semantic_scan_error",
        "durable_path_set_changed",
        "feature_scan_batch_error",
        "feature_scan_parquet_error",
        "feature_scan_row_count_mismatch",
        "file_changed_during_audit",
        "invalid_alerts_parquet",
        "invalid_cumulative_parquet",
        "invalid_loci_parquet",
        "invalid_manifest_json",
        "inventory_scan_error",
        "inventory_stat_error",
        "missing_alerts_parquet",
        "missing_canonical_cumulative_product",
        "missing_cumulative_directory",
        "missing_loci_parquet",
        "missing_manifest_json",
        "missing_nightly_directory",
        "missing_source_identity",
        "nightly_loci_semantic_column_missing",
        "nightly_summary_semantic_column_missing",
        "no_cumulative_parquet",
        "no_nightly_manifests",
        "nonnumeric_feature_column",
        "parquet_data_decode_error",
        "parquet_decoded_row_count_mismatch",
        "science_path_outside_root",
        "science_path_resolution_error",
        "science_path_stat_error",
        "science_path_symlink",
        "science_traversal_error",
        "source_identity_changed",
        "source_identity_stat_error",
    }
)

FEATURE_COLUMNS = (
    "feature_chi2_magn_r",
    "feature_standard_deviation_magn_r",
    "feature_weighted_mean_magn_u",
    "feature_weighted_mean_magn_g",
    "feature_weighted_mean_magn_r",
    "feature_weighted_mean_magn_i",
    "feature_weighted_mean_magn_z",
    "feature_weighted_mean_magn_y",
)

PAIRWISE_SPECS = (
    (
        "variability_r",
        ("feature_chi2_magn_r", "feature_standard_deviation_magn_r"),
    ),
    (
        "weighted_mean_g_i",
        ("feature_weighted_mean_magn_g", "feature_weighted_mean_magn_i"),
    ),
    (
        "weighted_mean_u_g_r",
        (
            "feature_weighted_mean_magn_u",
            "feature_weighted_mean_magn_g",
            "feature_weighted_mean_magn_r",
        ),
    ),
    (
        "weighted_mean_r_i_g",
        (
            "feature_weighted_mean_magn_r",
            "feature_weighted_mean_magn_i",
            "feature_weighted_mean_magn_g",
        ),
    ),
)

CUMULATIVE_LOCUS_SCIENCE_COLUMNS = (
    "locus_id",
    "night_date_utc",
    "night_mjd_min",
    "night_mjd_max",
    "ingested_at_utc",
    "source_query_mode",
    "newest_alert_observation_time",
    "ra",
    "dec",
    "tags",
    "survey",
    "ztf_object_id",
    "dia_object_id",
    "ss_object_id",
    "brightest_alert_magnitude",
    "num_mag_values",
)

NIGHTLY_SUMMARY_CORE_FIELDS = (
    "date_utc",
    "mjd_min",
    "mjd_max",
    "query_tag",
    "target_loci",
    "actual_loci",
    "alert_rows",
    "chunk_count",
    "split_count",
    "saturated_chunk_count",
    "status",
    "append_ready",
    "mjd_pass",
    "duplicate_locus_count",
    "coordinate_pass",
    "overlap_count",
    "alert_locus_link_pass",
    "survey_mode",
    "lsst_filter_used",
    "parallel_shards",
    "lsst_dia_count",
    "lsst_ss_count",
    "ztf_object_id_count",
    "lsst_only_pass",
    "history_start_pass",
    "loci_path",
    "alerts_path",
    "manifest_path",
)

MANIFEST_TABLE_FIELDS = (
    "date_utc",
    "folder_date_utc",
    "manifest_path",
    "manifest_exists",
    "declared_manifest_path",
    "declared_loci_path",
    "declared_alerts_path",
    "status",
    "actual_loci",
    "alert_rows",
    "append_ready",
    "effective_append_ready",
    "zero_row_policy_accepted",
    "mjd_min",
    "mjd_max",
    "loci_path",
    "loci_exists",
    "loci_rows",
    "loci_size_bytes",
    "alerts_path",
    "alerts_exists",
    "alerts_rows",
    "alerts_size_bytes",
    "science_files_complete",
    "operationally_complete",
    "integrity_issue_count",
    "integrity_issue_codes",
)

ZERO_ROW_LOCI_REQUIRED_COLUMNS = frozenset(
    {
        "locus_id",
        "ra",
        "dec",
        "newest_alert_observation_time",
        "night_date_utc",
        "night_mjd_min",
        "night_mjd_max",
        "ingested_at_utc",
        "source_query_mode",
    }
)
ZERO_ROW_ALERTS_REQUIRED_COLUMNS = frozenset(
    {"locus_id", "night_date_utc", "range_label"}
)


class AuditPreflightError(RuntimeError):
    """Raised when a complete audit cannot safely be produced."""


class IntegrityIssueCollector:
    """Collect stable, de-duplicated machine-readable integrity issues."""

    def __init__(self):
        self.items = []
        self._seen = set()

    def add(self, code, message, path=None):
        item = {"code": str(code), "message": str(message)}
        if path is not None:
            item["path"] = str(path)
        key = (item["code"], item["message"], item.get("path"))
        if key not in self._seen:
            self._seen.add(key)
            self.items.append(item)

    def sorted_items(self):
        return sorted(
            self.items,
            key=lambda item: (
                item.get("path", ""),
                item["code"],
                item["message"],
            ),
        )


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_columnar_dependencies():
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise AuditPreflightError(
            "The audit requires numpy and pyarrow. Install them in the "
            f"environment used for the audit ({exc})."
        ) from exc
    return np, pa, ds, pq


def _path_is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _relative(path, data_root):
    """Return a stable, source/destination-independent POSIX path."""
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise AuditPreflightError(
            f"Science path escaped the data root: {path}"
        ) from exc


def _science_path_problems(path, data_root):
    """Return containment/symlink problems without opening ``path``.

    ``data_root`` is already resolved by preflight.  Every existing component
    below it is checked with ``lstat`` so neither an ancestor symlink nor a
    final-file symlink can be hidden by normal ``Path`` methods that follow
    links.
    """
    problems = []
    try:
        relative = path.relative_to(data_root)
    except ValueError:
        return [
            (
                "science_path_outside_root",
                f"Science path is lexically outside the data root: {path}",
            )
        ]

    cursor = data_root
    for part in relative.parts:
        cursor = cursor / part
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            problems.append(
                (
                    "science_path_stat_error",
                    f"Could not lstat science-path component {cursor}: {exc}",
                )
            )
            break
        if stat.S_ISLNK(metadata.st_mode):
            problems.append(
                (
                    "science_path_symlink",
                    f"Durable science path contains symlink component: {cursor}",
                )
            )
            break

    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        problems.append(
            (
                "science_path_resolution_error",
                f"Could not resolve science path {path}: {exc}",
            )
        )
    else:
        if not _path_is_within(resolved, data_root):
            problems.append(
                (
                    "science_path_outside_root",
                    "Resolved science path escapes the data root: "
                    f"{path} -> {resolved}",
                )
            )
    return problems


def _record_science_path_problems(
    path, data_root, issues, row_codes=None
):
    """Record path-safety problems and return whether the path is safe."""
    try:
        display_path = _relative(path, data_root)
    except AuditPreflightError:
        display_path = str(path)
    problems = _science_path_problems(path, data_root)
    for code, message in problems:
        issues.add(code, message, display_path)
        if row_codes is not None:
            row_codes.append(code)
    return not problems


def _source_identity(path):
    metadata = path.stat()
    return tuple(
        int(getattr(metadata, field)) for field in SOURCE_IDENTITY_FIELDS
    )


def _capture_source_identity(
    path, data_root, issues, phase, identities
):
    """Capture one durable file's identity for later phase comparison."""
    if not _record_science_path_problems(path, data_root, issues):
        return None
    rel = _relative(path, data_root)
    try:
        identity = _source_identity(path)
    except OSError as exc:
        issues.add(
            "source_identity_stat_error",
            f"Could not record {phase} source identity: {exc}",
            rel,
        )
        return None
    identities[rel] = identity
    return identity


def _record_source_identity_change(
    path, data_root, issues, expected, current, phase
):
    if expected == current:
        return False
    issues.add(
        "source_identity_changed",
        "Durable source identity changed between phases "
        f"(expected from {phase}: {expected}; current: {current}).",
        _relative(path, data_root),
    )
    return True


def _validate_paths(data_root, out):
    root = Path(data_root).expanduser().resolve()
    raw_audit_out = Path(out).expanduser()
    if os.path.lexists(str(raw_audit_out)):
        raise AuditPreflightError(
            "Audit output already exists; refusing to overwrite it: "
            f"{raw_audit_out.resolve()}"
        )
    lexical_audit_out = Path(os.path.abspath(str(raw_audit_out)))
    audit_out = raw_audit_out.resolve()

    if not root.exists():
        raise AuditPreflightError(f"Data root does not exist: {root}")
    if not root.is_dir():
        raise AuditPreflightError(f"Data root is not a directory: {root}")
    if (
        _path_is_within(lexical_audit_out, root)
        or _path_is_within(audit_out, root)
    ):
        raise AuditPreflightError(
            "Audit output must be entirely outside the data root: "
            f"{audit_out}"
        )

    protected_roots = (
        root / "cache",
        root / "data" / "lsst_only" / "nightly",
        root / "data" / "lsst_only" / "cumulative",
    )
    for protected in protected_roots:
        resolved_protected = protected.resolve(strict=False)
        if (
            _path_is_within(lexical_audit_out, protected)
            or _path_is_within(audit_out, resolved_protected)
        ):
            raise AuditPreflightError(
                "Audit output cannot be placed inside cache or a durable "
                f"science-product directory: {audit_out}"
            )
    return root, audit_out


def _empty_inventory(present):
    return {
        "present": bool(present),
        "file_count": 0,
        "directory_count": 0,
        "symlink_count": 0,
        "size_bytes": 0,
        "file_counts_by_extension": {},
        "file_bytes_by_extension": {},
    }


def _extension_label(path):
    return path.suffix.lower() or "[no extension]"


def _inventory_add_regular(inventory, path, size):
    extension = _extension_label(path)
    inventory["file_count"] += 1
    inventory["size_bytes"] += int(size)
    inventory["_extension_counts"][extension] += 1
    inventory["_extension_bytes"][extension] += int(size)


def _inventory_add_symlink(inventory):
    inventory["symlink_count"] += 1


def _finish_inventory(inventory):
    inventory["file_counts_by_extension"] = dict(
        sorted(inventory.pop("_extension_counts").items())
    )
    inventory["file_bytes_by_extension"] = dict(
        sorted(inventory.pop("_extension_bytes").items())
    )
    return inventory


def _new_working_inventory(present):
    inventory = _empty_inventory(present)
    inventory["_extension_counts"] = Counter()
    inventory["_extension_bytes"] = Counter()
    return inventory


def _scan_root_inventory(data_root, issues):
    """Scan the root once and report root, non-cache, and cache inventories."""
    cache_root = data_root / "cache"
    inventories = {
        "root": _new_working_inventory(True),
        "root_excluding_cache": _new_working_inventory(True),
        "cache": _new_working_inventory(
            cache_root.is_dir() and not cache_root.is_symlink()
        ),
    }

    def onerror(exc):
        raw_path = getattr(exc, "filename", None)
        display_path = str(raw_path) if raw_path else str(data_root)
        try:
            display_path = _relative(Path(display_path), data_root)
        except (AuditPreflightError, ValueError):
            pass
        issues.add(
            "inventory_scan_error",
            f"Could not inspect part of the data root: {exc}",
            display_path,
        )

    for current, dirnames, filenames in os.walk(
        data_root, topdown=True, onerror=onerror, followlinks=False
    ):
        current_path = Path(current)
        in_cache = current_path == cache_root or _path_is_within(
            current_path, cache_root
        )
        inventories["root"]["directory_count"] += 1
        target = "cache" if in_cache else "root_excluding_cache"
        inventories[target]["directory_count"] += 1

        # os.walk exposes symlinked directories in dirnames.  Count them as
        # links and remove them explicitly so the inventory never escapes root.
        retained_dirnames = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            try:
                is_link = child.is_symlink()
            except OSError as exc:
                issues.add(
                    "inventory_stat_error",
                    f"Could not stat directory entry: {exc}",
                    _relative(child, data_root),
                )
                is_link = False
            if is_link:
                _inventory_add_symlink(inventories["root"])
                _inventory_add_symlink(inventories[target])
            else:
                retained_dirnames.append(dirname)
        dirnames[:] = retained_dirnames

        for filename in sorted(filenames):
            path = current_path / filename
            try:
                metadata = path.lstat()
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    _inventory_add_symlink(inventories["root"])
                    _inventory_add_symlink(inventories[target])
                elif stat.S_ISREG(mode):
                    _inventory_add_regular(
                        inventories["root"], path, metadata.st_size
                    )
                    _inventory_add_regular(
                        inventories[target], path, metadata.st_size
                    )
            except OSError as exc:
                issues.add(
                    "inventory_stat_error",
                    f"Could not stat file: {exc}",
                    _relative(path, data_root),
                )

    return {
        name: _finish_inventory(inventory)
        for name, inventory in inventories.items()
    }


def _valid_iso_date(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    canonical = parsed.isoformat()
    return canonical if canonical == value else None


def _date_from_manifest_path(path, nightly_root):
    try:
        parts = path.relative_to(nightly_root).parts
    except ValueError:
        return None
    if len(parts) != 4 or parts[-1] != "manifest.json":
        return None
    year, month, day, _ = parts
    if not (
        re.fullmatch(r"\d{4}", year)
        and re.fullmatch(r"\d{2}", month)
        and re.fullmatch(r"\d{2}", day)
    ):
        return None
    return _valid_iso_date(f"{year}-{month}-{day}")


def _nonnegative_integer(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        integer = int(value)
        return integer if integer >= 0 else None
    return None


def _meaningful_error_value(value):
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _recorded_query_fetch_errors(payload):
    findings = {}

    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                lowered = key_text.lower()
                is_error_field = (
                    lowered == "error"
                    or (
                        "error" in lowered
                        and any(
                            token in lowered
                            for token in ("query", "fetch", "lightcurve")
                        )
                    )
                )
                if is_error_field and _meaningful_error_value(child):
                    findings[path] = child
                if isinstance(child, (dict, list, tuple)):
                    visit(child, path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{prefix}[{index}]")

    visit(payload)
    return findings


def _valid_legacy_zero_row_night(row, payload, product_schema_names):
    if not isinstance(payload, dict):
        return False
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return False
    chunk_count = payload.get("chunk_count")
    return bool(
        row["science_files_complete"]
        and row["status"] == "complete"
        and row["actual_loci"] == 0
        and row["alert_rows"] == 0
        and row["loci_rows"] == 0
        and row["alerts_rows"] == 0
        and isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count > 0
        and payload.get("saturated_chunk_count") in (0, 0.0)
        and payload.get("finished_at_utc")
        and payload.get("lsst_filter_used") is True
        and not _recorded_query_fetch_errors(payload)
        and validation.get("mjd_pass") is True
        and validation.get("duplicate_locus_count") == 0
        and validation.get("coordinate_pass") is True
        and validation.get("alert_locus_link_pass") is True
        and validation.get("history_start_pass") is True
        and ZERO_ROW_LOCI_REQUIRED_COLUMNS.issubset(
            product_schema_names.get("loci", set())
        )
        and ZERO_ROW_ALERTS_REQUIRED_COLUMNS.issubset(
            product_schema_names.get("alerts", set())
        )
    )


def _normalized_date_value(value):
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return str(value)


def _stream_validate_parquet(
    path,
    data_root,
    pq,
    issues,
    batch_size,
    open_issue_code,
    collect_columns=(),
):
    """Decode every Parquet data page in bounded batches.

    Metadata-only reads cannot detect damaged column pages.  This routine
    iterates all columns, counts decoded rows, and optionally collects bounded
    date counters needed for cumulative-index validation.
    """
    rel = _relative(path, data_root)
    try:
        try:
            parquet_file = pq.ParquetFile(
                path, page_checksum_verification=True
            )
        except TypeError:
            # Compatibility with older PyArrow releases that predate page
            # checksum verification; full page decoding still occurs.
            parquet_file = pq.ParquetFile(path)
        metadata_rows = int(parquet_file.metadata.num_rows)
        schema_names = set(parquet_file.schema_arrow.names)
    except Exception as exc:
        issues.add(
            open_issue_code,
            f"Could not read Parquet metadata: {type(exc).__name__}: {exc}",
            rel,
        )
        return None

    collected = {column: Counter() for column in collect_columns}
    decoded_rows = 0
    decode_ok = True
    try:
        for batch in parquet_file.iter_batches(
            batch_size=max(1, min(int(batch_size), 8_192))
        ):
            decoded_rows += int(batch.num_rows)
            for column in collect_columns:
                index = batch.schema.get_field_index(column)
                if index < 0:
                    continue
                collected[column].update(
                    _normalized_date_value(value)
                    for value in batch.column(index).to_pylist()
                )
    except Exception as exc:
        decode_ok = False
        issues.add(
            "parquet_data_decode_error",
            f"Could not decode all Parquet data pages: "
            f"{type(exc).__name__}: {exc}",
            rel,
        )

    if decoded_rows != metadata_rows:
        decode_ok = False
        issues.add(
            "parquet_decoded_row_count_mismatch",
            f"Parquet metadata reports {metadata_rows} rows but full-page "
            f"decoding produced {decoded_rows}.",
            rel,
        )
    return {
        "metadata_rows": metadata_rows,
        "decoded_rows": decoded_rows,
        "decode_ok": decode_ok,
        "schema_names": schema_names,
        "column_counts": collected,
    }


def _add_scoped_issue(issues, row_codes, code, message, path):
    issues.add(code, message, path)
    row_codes.append(code)


def _inspect_nightly(data_root, pq, issues, batch_size):
    nightly_root = data_root / "data" / "lsst_only" / "nightly"
    nightly_root_safe = _record_science_path_problems(
        nightly_root, data_root, issues
    )
    if not nightly_root_safe:
        manifest_paths = []
        partition_dirs = []
    elif not nightly_root.is_dir():
        issues.add(
            "missing_nightly_directory",
            "Expected nightly directory is missing.",
            _relative(nightly_root, data_root),
        )
        manifest_paths = []
        partition_dirs = []
    else:
        expected_product_paths = sorted(
            path
            for path in _files_under(nightly_root, data_root, issues)
            if path.name in {"manifest.json", "loci.parquet", "alerts.parquet"}
        )
        manifest_paths = [
            path for path in expected_product_paths if path.name == "manifest.json"
        ]
        partition_dirs = sorted({path.parent for path in expected_product_paths})
        if not manifest_paths:
            issues.add(
                "no_nightly_manifests",
                "No nightly manifest.json files were found.",
                _relative(nightly_root, data_root),
            )

    rows = []
    nightly_parquet_paths = []
    nightly_science_paths = []
    loci_paths = []
    source_identities = {}
    manifest_payloads = {}

    for partition_dir in partition_dirs:
        manifest_path = partition_dir / "manifest.json"
        manifest_rel = _relative(manifest_path, data_root)
        row_codes = []
        folder_date = _date_from_manifest_path(manifest_path, nightly_root)
        if folder_date is None:
            _add_scoped_issue(
                issues,
                row_codes,
                "noncanonical_nightly_partition",
                "Nightly science products are not under "
                "nightly/YYYY/MM/DD/.",
                _relative(partition_dir, data_root),
            )

        row = {
            "date_utc": folder_date,
            "folder_date_utc": folder_date,
            "manifest_path": manifest_rel,
            "manifest_exists": False,
            "declared_manifest_path": None,
            "declared_loci_path": None,
            "declared_alerts_path": None,
            "status": None,
            "actual_loci": None,
            "alert_rows": None,
            "append_ready": None,
            "effective_append_ready": None,
            "zero_row_policy_accepted": False,
            "mjd_min": None,
            "mjd_max": None,
            "loci_path": _relative(manifest_path.parent / "loci.parquet", data_root),
            "loci_exists": False,
            "loci_rows": None,
            "loci_size_bytes": None,
            "alerts_path": _relative(
                manifest_path.parent / "alerts.parquet", data_root
            ),
            "alerts_exists": False,
            "alerts_rows": None,
            "alerts_size_bytes": None,
            "science_files_complete": False,
            "operationally_complete": False,
            "integrity_issue_count": 0,
            "integrity_issue_codes": "",
        }

        payload = None
        manifest_counts_valid = False
        manifest_safe = _record_science_path_problems(
            manifest_path, data_root, issues, row_codes
        )
        manifest_exists = manifest_safe and manifest_path.is_file()
        row["manifest_exists"] = bool(manifest_exists)
        if not manifest_exists:
            if manifest_safe:
                _add_scoped_issue(
                    issues,
                    row_codes,
                    "missing_manifest_json",
                    "Nightly partition has loci.parquet or alerts.parquet but "
                    "no readable sibling manifest.json.",
                    manifest_rel,
                )
        else:
            manifest_identity = _capture_source_identity(
                manifest_path,
                data_root,
                issues,
                "nightly inspection",
                source_identities,
            )
            if manifest_identity is not None:
                try:
                    with manifest_path.open("r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    if not isinstance(loaded, dict):
                        raise ValueError("top-level JSON value is not an object")
                    payload = loaded
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    _add_scoped_issue(
                        issues,
                        row_codes,
                        "invalid_manifest_json",
                        f"Could not parse manifest JSON: "
                        f"{type(exc).__name__}: {exc}",
                        manifest_rel,
                    )
                if payload is not None:
                    manifest_payloads[manifest_rel] = payload
                nightly_science_paths.append(manifest_path)

        if payload is not None:
            manifest_counts_valid = True
            manifest_date = _valid_iso_date(payload.get("date_utc"))
            if manifest_date is None:
                _add_scoped_issue(
                    issues,
                    row_codes,
                    "invalid_manifest_date",
                    "Manifest date_utc is absent or is not canonical YYYY-MM-DD.",
                    manifest_rel,
                )
            else:
                row["date_utc"] = manifest_date
                if folder_date is not None and manifest_date != folder_date:
                    _add_scoped_issue(
                        issues,
                        row_codes,
                        "manifest_date_path_mismatch",
                        f"Manifest date {manifest_date} does not match folder "
                        f"date {folder_date}.",
                        manifest_rel,
                    )

            declared_paths = payload.get("paths")
            if declared_paths is not None and not isinstance(
                declared_paths, dict
            ):
                _add_scoped_issue(
                    issues,
                    row_codes,
                    "invalid_declared_paths",
                    "Manifest paths must be a JSON object when present.",
                    manifest_rel,
                )
                declared_paths = {}
            if isinstance(declared_paths, dict):
                for product in ("manifest", "loci", "alerts"):
                    value = declared_paths.get(product)
                    field = f"declared_{product}_path"
                    if value is None:
                        continue
                    if not isinstance(value, str):
                        _add_scoped_issue(
                            issues,
                            row_codes,
                            "invalid_declared_path",
                            f"Manifest paths.{product} is not a string.",
                            manifest_rel,
                        )
                        continue
                    row[field] = value
                    physical_name = (
                        "manifest.json"
                        if product == "manifest"
                        else f"{product}.parquet"
                    )
                    expected_suffix = _relative(
                        partition_dir / physical_name, data_root
                    )
                    normalized_declared = value.replace("\\", "/")
                    if not (
                        normalized_declared == expected_suffix
                        or normalized_declared.endswith("/" + expected_suffix)
                    ):
                        _add_scoped_issue(
                            issues,
                            row_codes,
                            "declared_path_suffix_mismatch",
                            f"Manifest paths.{product} does not end with its "
                            "portable data-root-relative science path. The "
                            "declared path was recorded but not followed.",
                            manifest_rel,
                        )

            status_value = payload.get("status")
            row["status"] = status_value if isinstance(status_value, str) else None
            if row["status"] not in {
                "complete",
                "under_target",
                "saturated_unresolved",
            }:
                _add_scoped_issue(
                    issues,
                    row_codes,
                    "nonappendable_manifest_status",
                    f"Manifest status is not appendable: {status_value!r}.",
                    manifest_rel,
                )

            for field in ("actual_loci", "alert_rows"):
                parsed_count = _nonnegative_integer(payload.get(field))
                row[field] = parsed_count
                if parsed_count is None:
                    manifest_counts_valid = False
                    _add_scoped_issue(
                        issues,
                        row_codes,
                        "invalid_manifest_count",
                        f"Manifest {field} is not a non-negative integer.",
                        manifest_rel,
                    )

            row["mjd_min"] = payload.get("mjd_min")
            row["mjd_max"] = payload.get("mjd_max")
            validation = payload.get("validation")
            if isinstance(validation, dict):
                append_ready = validation.get("append_ready")
                row["append_ready"] = (
                    append_ready if isinstance(append_ready, bool) else None
                )
            recorded_errors = _recorded_query_fetch_errors(payload)
            if recorded_errors:
                _add_scoped_issue(
                    issues,
                    row_codes,
                    "manifest_recorded_query_fetch_error",
                    "Manifest records query/fetch error evidence: "
                    f"{sorted(recorded_errors)}.",
                    manifest_rel,
                )

        product_specs = (
            ("loci", "actual_loci", "loci_rows", "loci_size_bytes"),
            ("alerts", "alert_rows", "alerts_rows", "alerts_size_bytes"),
        )
        products_readable = True
        product_schema_names = {}
        for product, count_field, rows_field, size_field in product_specs:
            product_path = manifest_path.parent / f"{product}.parquet"
            product_rel = _relative(product_path, data_root)
            product_safe = _record_science_path_problems(
                product_path, data_root, issues, row_codes
            )
            exists = product_safe and product_path.is_file()
            row[f"{product}_exists"] = bool(exists)
            if not exists:
                products_readable = False
                if product_safe:
                    _add_scoped_issue(
                        issues,
                        row_codes,
                        f"missing_{product}_parquet",
                        f"Expected sibling {product}.parquet is missing.",
                        product_rel,
                    )
                continue

            product_identity = _capture_source_identity(
                product_path,
                data_root,
                issues,
                "nightly inspection",
                source_identities,
            )
            if product_identity is None:
                products_readable = False
                continue
            row[size_field] = product_identity[2]
            nightly_parquet_paths.append(product_path)
            nightly_science_paths.append(product_path)
            if product == "loci":
                loci_paths.append(product_path)

            parquet_report = _stream_validate_parquet(
                product_path,
                data_root,
                pq,
                issues,
                batch_size,
                f"invalid_{product}_parquet",
            )
            if parquet_report is None:
                products_readable = False
                parquet_rows = None
                row_codes.append(f"invalid_{product}_parquet")
            else:
                parquet_rows = parquet_report["metadata_rows"]
                product_schema_names[product] = parquet_report["schema_names"]
                if not parquet_report["decode_ok"]:
                    products_readable = False
                    row_codes.append("parquet_data_decode_error")
            row[rows_field] = parquet_rows

            manifest_count = row.get(count_field)
            if (
                manifest_count is not None
                and parquet_rows is not None
                and manifest_count != parquet_rows
            ):
                products_readable = False
                _add_scoped_issue(
                    issues,
                    row_codes,
                    f"{product}_row_count_mismatch",
                    f"Manifest {count_field}={manifest_count} but sibling "
                    f"{product}.parquet has {parquet_rows} rows.",
                    product_rel,
                )

        row["science_files_complete"] = bool(
            payload is not None
            and manifest_exists
            and manifest_counts_valid
            and products_readable
            and row["loci_exists"]
            and row["alerts_exists"]
        )
        row["zero_row_policy_accepted"] = _valid_legacy_zero_row_night(
            row, payload, product_schema_names
        )
        row["effective_append_ready"] = bool(
            row["append_ready"] is not False
            or row["zero_row_policy_accepted"]
        )
        if (
            row["append_ready"] is False
            and not row["zero_row_policy_accepted"]
        ):
            _add_scoped_issue(
                issues,
                row_codes,
                "manifest_not_append_ready",
                "Manifest validation.append_ready is false and the night "
                "does not satisfy the valid zero-row LSST policy.",
                manifest_rel,
            )
        row["operationally_complete"] = bool(
            row["science_files_complete"]
            and row["status"]
            in {"complete", "under_target", "saturated_unresolved"}
            and row["effective_append_ready"]
        )
        row["integrity_issue_count"] = len(row_codes)
        row["integrity_issue_codes"] = ";".join(sorted(set(row_codes)))
        rows.append(row)

    seen_dates = {}
    for row in rows:
        date_value = row.get("date_utc")
        if date_value is None:
            continue
        if date_value in seen_dates:
            issues.add(
                "duplicate_nightly_date",
                f"More than one manifest claims nightly date {date_value}.",
                row["manifest_path"],
            )
            for affected in (seen_dates[date_value], row):
                affected["science_files_complete"] = False
                affected["operationally_complete"] = False
                codes = set(filter(None, affected["integrity_issue_codes"].split(";")))
                codes.add("duplicate_nightly_date")
                affected["integrity_issue_codes"] = ";".join(sorted(codes))
                affected["integrity_issue_count"] = len(codes)
        else:
            seen_dates[date_value] = row

    return {
        "rows": rows,
        "manifest_paths": manifest_paths,
        "nightly_parquet_paths": nightly_parquet_paths,
        "nightly_science_paths": nightly_science_paths,
        "loci_paths": loci_paths,
        "source_identities": source_identities,
        "manifest_payloads": manifest_payloads,
    }


def _files_under(root, data_root, issues):
    """Yield safe lexical paths without following or accepting symlinks."""
    if not _record_science_path_problems(root, data_root, issues):
        return
    if not root.is_dir():
        return

    def onerror(exc):
        raw_path = Path(getattr(exc, "filename", root))
        try:
            display_path = _relative(raw_path, data_root)
        except AuditPreflightError:
            display_path = str(raw_path)
        issues.add(
            "science_traversal_error",
            f"Could not traverse durable science path: {exc}",
            display_path,
        )

    for current, dirnames, filenames in os.walk(
        root, topdown=True, onerror=onerror, followlinks=False
    ):
        current_path = Path(current)
        if not _record_science_path_problems(
            current_path, data_root, issues
        ):
            dirnames[:] = []
            continue

        retained_dirnames = []
        for name in sorted(dirnames):
            child = current_path / name
            if _record_science_path_problems(
                child, data_root, issues
            ):
                retained_dirnames.append(name)
        dirnames[:] = retained_dirnames

        for filename in sorted(filenames):
            path = current_path / filename
            if _record_science_path_problems(
                path, data_root, issues
            ):
                yield path


def _canonical_scalar(value):
    """Normalize only representation-safe scalar/container differences."""
    if value is None:
        return ["null", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["number", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["null", None]
        if math.isinf(value):
            return ["float", "inf" if value > 0 else "-inf"]
        if value.is_integer():
            return ["number", str(int(value))]
        return ["float", value.hex()]
    if isinstance(value, (date, datetime)):
        return ["datetime", value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, dict):
        return [
            "object",
            [
                [str(key), _canonical_scalar(item)]
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )
            ],
        ]
    if isinstance(value, (list, tuple)):
        return ["array", [_canonical_scalar(item) for item in value]]
    if hasattr(value, "item"):
        try:
            return _canonical_scalar(value.item())
        except (TypeError, ValueError):
            pass
    return ["repr", repr(value)]


def _canonical_json(value):
    return json.dumps(
        _canonical_scalar(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_row_digest(row, columns):
    payload = [
        [column, _canonical_scalar(row.get(column))]
        for column in columns
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _batch_rows(batch, columns):
    values = [
        batch.column(batch.schema.get_field_index(column)).to_pylist()
        for column in columns
    ]
    for index in range(batch.num_rows):
        yield {
            column: values[position][index]
            for position, column in enumerate(columns)
        }


def _manifest_summary_expected(payload):
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        validation = {}
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    expected = {
        field: payload.get(field)
        for field in NIGHTLY_SUMMARY_CORE_FIELDS
    }
    for field in (
        "append_ready",
        "mjd_pass",
        "duplicate_locus_count",
        "coordinate_pass",
        "overlap_count",
        "alert_locus_link_pass",
        "lsst_only_pass",
        "history_start_pass",
    ):
        expected[field] = validation.get(field)
    expected["loci_path"] = paths.get("loci")
    expected["alerts_path"] = paths.get("alerts")
    expected["manifest_path"] = paths.get("manifest")
    return expected


def _reconcile_nightly_summary_semantics(
    summary_path,
    operational_rows,
    manifest_payloads,
    data_root,
    pq,
    issues,
    batch_size,
):
    rel = _relative(summary_path, data_root)
    try:
        parquet_file = pq.ParquetFile(summary_path)
        schema_names = set(parquet_file.schema_arrow.names)
    except Exception as exc:
        issues.add(
            "cumulative_semantic_scan_error",
            f"Could not open nightly summary for semantic reconciliation: "
            f"{type(exc).__name__}: {exc}",
            rel,
        )
        return

    missing_columns = [
        field
        for field in NIGHTLY_SUMMARY_CORE_FIELDS
        if field not in schema_names
    ]
    if missing_columns:
        issues.add(
            "nightly_summary_semantic_column_missing",
            "nightly_summary.parquet is missing semantic fields: "
            + ", ".join(missing_columns),
            rel,
        )

    columns = [
        field
        for field in NIGHTLY_SUMMARY_CORE_FIELDS
        if field in schema_names
    ]
    expected_by_date = {}
    for nightly_row in operational_rows:
        date_utc = nightly_row.get("date_utc")
        payload = manifest_payloads.get(nightly_row["manifest_path"])
        if date_utc is not None and payload is not None:
            expected_by_date[date_utc] = _manifest_summary_expected(payload)

    try:
        for batch in parquet_file.iter_batches(
            batch_size=max(1, min(int(batch_size), 8_192)),
            columns=columns,
        ):
            for row in _batch_rows(batch, columns):
                date_value = _normalized_date_value(row.get("date_utc"))
                expected = expected_by_date.get(date_value)
                if expected is None:
                    # Row/date cardinality is reported by canonical checks.
                    continue
                mismatches = [
                    field
                    for field in NIGHTLY_SUMMARY_CORE_FIELDS
                    if field in row
                    and _canonical_scalar(row.get(field))
                    != _canonical_scalar(expected.get(field))
                ]
                if mismatches:
                    issues.add(
                        "cumulative_nightly_summary_semantic_mismatch",
                        f"nightly_summary.parquet differs from manifest for "
                        f"{date_value} in fields: "
                        f"{', '.join(mismatches)}.",
                        rel,
                    )
    except Exception as exc:
        issues.add(
            "cumulative_semantic_scan_error",
            f"Could not stream nightly summary semantics: "
            f"{type(exc).__name__}: {exc}",
            rel,
        )


def _dataset_date_scalar(pa, field_type, date_utc):
    if pa.types.is_date32(field_type) or pa.types.is_date64(field_type):
        return pa.scalar(date.fromisoformat(date_utc), type=field_type)
    if pa.types.is_timestamp(field_type):
        return pa.scalar(
            datetime.fromisoformat(date_utc), type=field_type
        )
    return pa.scalar(date_utc, type=field_type)


def _reconcile_loci_index_semantics(
    index_path,
    operational_rows,
    data_root,
    pa,
    ds,
    pq,
    issues,
    batch_size,
):
    rel = _relative(index_path, data_root)
    try:
        index_dataset = ds.dataset(index_path, format="parquet")
        index_schema_names = set(index_dataset.schema.names)
    except Exception as exc:
        issues.add(
            "cumulative_semantic_scan_error",
            f"Could not open loci index for semantic reconciliation: "
            f"{type(exc).__name__}: {exc}",
            rel,
        )
        return

    required = {"night_date_utc", "locus_id"}
    if not required.issubset(index_schema_names):
        issues.add(
            "cumulative_semantic_column_missing",
            "loci_index.parquet must contain night_date_utc and locus_id "
            "for exact semantic reconciliation.",
            rel,
        )
        return

    date_type = index_dataset.schema.field("night_date_utc").type
    semantic_batch_size = max(1, min(int(batch_size), 8_192))
    for nightly_row in operational_rows:
        date_utc = nightly_row.get("date_utc")
        if date_utc is None:
            continue
        nightly_path = data_root / Path(nightly_row["loci_path"])
        nightly_rel = _relative(nightly_path, data_root)
        try:
            nightly_file = pq.ParquetFile(nightly_path)
            nightly_schema_names = set(
                nightly_file.schema_arrow.names
            )
        except Exception as exc:
            issues.add(
                "cumulative_semantic_scan_error",
                f"Could not open nightly loci for semantic reconciliation: "
                f"{type(exc).__name__}: {exc}",
                nightly_rel,
            )
            continue
        if not required.issubset(nightly_schema_names):
            issues.add(
                "nightly_loci_semantic_column_missing",
                "Nightly loci must contain night_date_utc and locus_id for "
                "exact semantic reconciliation.",
                nightly_rel,
            )
            continue

        shared_columns = [
            column
            for column in CUMULATIVE_LOCUS_SCIENCE_COLUMNS
            if column in nightly_schema_names
            and column in index_schema_names
        ]
        nightly_ids = Counter()
        index_ids = Counter()
        nightly_digests = Counter()
        index_digests = Counter()
        try:
            for batch in nightly_file.iter_batches(
                batch_size=semantic_batch_size,
                columns=shared_columns,
            ):
                for row in _batch_rows(batch, shared_columns):
                    pair = (
                        _canonical_json(
                            _normalized_date_value(
                                row.get("night_date_utc")
                            )
                        ),
                        _canonical_json(row.get("locus_id")),
                    )
                    nightly_ids[pair] += 1
                    nightly_digests[
                        _canonical_row_digest(row, shared_columns)
                    ] += 1

            date_scalar = _dataset_date_scalar(
                pa, date_type, date_utc
            )
            scanner = index_dataset.scanner(
                columns=shared_columns,
                filter=ds.field("night_date_utc") == date_scalar,
                batch_size=semantic_batch_size,
            )
            for batch in scanner.to_batches():
                for row in _batch_rows(batch, shared_columns):
                    pair = (
                        _canonical_json(
                            _normalized_date_value(
                                row.get("night_date_utc")
                            )
                        ),
                        _canonical_json(row.get("locus_id")),
                    )
                    index_ids[pair] += 1
                    index_digests[
                        _canonical_row_digest(row, shared_columns)
                    ] += 1
        except Exception as exc:
            issues.add(
                "cumulative_semantic_scan_error",
                f"Could not reconcile loci semantics for {date_utc}: "
                f"{type(exc).__name__}: {exc}",
                rel,
            )
            continue

        if nightly_ids != index_ids:
            issues.add(
                "cumulative_loci_id_multiset_mismatch",
                f"loci_index.parquet (night_date_utc, locus_id) multiset "
                f"differs from nightly loci for {date_utc}.",
                rel,
            )
        if nightly_digests != index_digests:
            issues.add(
                "cumulative_loci_science_digest_mismatch",
                f"loci_index.parquet shared science-column multiset differs "
                f"from nightly loci for {date_utc}; compared columns: "
                f"{', '.join(shared_columns)}.",
                rel,
            )


def _inspect_cumulative(
    data_root,
    pa,
    ds,
    pq,
    issues,
    batch_size,
    nightly_rows,
    manifest_payloads,
):
    cumulative_root = data_root / "data" / "lsst_only" / "cumulative"
    cumulative_root_safe = _record_science_path_problems(
        cumulative_root, data_root, issues
    )
    if not cumulative_root_safe:
        return {
            "all_paths": [],
            "parquet_paths": [],
            "parquet_size_bytes": 0,
            "source_identities": {},
            "parquet_reports": {},
        }
    if not cumulative_root.is_dir():
        issues.add(
            "missing_cumulative_directory",
            "Expected cumulative directory is missing.",
            _relative(cumulative_root, data_root),
        )
        return {
            "all_paths": [],
            "parquet_paths": [],
            "parquet_size_bytes": 0,
            "source_identities": {},
            "parquet_reports": {},
        }

    all_paths = []
    parquet_paths = []
    parquet_size_bytes = 0
    source_identities = {}
    parquet_reports = {}
    canonical_collect_columns = {
        "loci_index.parquet": ("night_date_utc",),
        "nightly_summary.parquet": ("date_utc",),
    }
    for path in _files_under(cumulative_root, data_root, issues):
        rel = _relative(path, data_root)
        if not _record_science_path_problems(path, data_root, issues):
            continue
        if not path.is_file():
            issues.add(
                "cumulative_file_unreadable",
                "Cumulative entry is not a readable file.",
                rel,
            )
            continue
        identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "cumulative inspection",
            source_identities,
        )
        if identity is None:
            continue
        all_paths.append(path)
        if path.suffix.lower() != ".parquet":
            continue
        parquet_paths.append(path)
        parquet_size_bytes += identity[2]
        parquet_reports[rel] = _stream_validate_parquet(
            path,
            data_root,
            pq,
            issues,
            batch_size,
            "invalid_cumulative_parquet",
            canonical_collect_columns.get(path.name, ()),
        )

    if not parquet_paths:
        issues.add(
            "no_cumulative_parquet",
            "No cumulative Parquet files were found.",
            _relative(cumulative_root, data_root),
        )

    operational_rows = [
        row for row in nightly_rows if row["operationally_complete"]
    ]
    expected_summary_dates = Counter()
    expected_loci_dates = Counter()
    for row in operational_rows:
        date_utc = row.get("date_utc")
        if date_utc is None:
            issues.add(
                "operational_night_missing_date",
                "Operationally complete nightly row has no usable date.",
                row["manifest_path"],
            )
            continue
        expected_summary_dates[date_utc] += 1
        loci_rows = int(row.get("loci_rows") or 0)
        if loci_rows:
            expected_loci_dates[date_utc] += loci_rows

    canonical_specs = (
        (
            "loci_index.parquet",
            sum(expected_loci_dates.values()),
            "night_date_utc",
            expected_loci_dates,
        ),
        (
            "nightly_summary.parquet",
            len(operational_rows),
            "date_utc",
            expected_summary_dates,
        ),
    )
    all_paths_by_rel = {
        _relative(path, data_root): path for path in all_paths
    }
    for filename, expected_rows, date_column, expected_dates in canonical_specs:
        canonical_path = cumulative_root / filename
        canonical_rel = _relative(canonical_path, data_root)
        if canonical_rel not in all_paths_by_rel:
            issues.add(
                "missing_canonical_cumulative_product",
                f"Required cumulative product is missing: {filename}.",
                canonical_rel,
            )
            continue
        report = parquet_reports.get(canonical_rel)
        if report is None:
            continue
        if report["metadata_rows"] != expected_rows:
            issues.add(
                "stale_cumulative_row_count",
                f"{filename} has {report['metadata_rows']} rows; expected "
                f"{expected_rows} from operationally complete nightly rows.",
                canonical_rel,
            )
        if date_column not in report["schema_names"]:
            issues.add(
                "cumulative_date_column_missing",
                f"{filename} is missing required date column {date_column}.",
                canonical_rel,
            )
            continue
        actual_dates = report["column_counts"][date_column]
        if actual_dates != expected_dates:
            issues.add(
                "stale_cumulative_date_coverage",
                f"{filename} date coverage does not match operationally "
                "complete nightly rows.",
                canonical_rel,
            )

    index_path = cumulative_root / "loci_index.parquet"
    summary_path = cumulative_root / "nightly_summary.parquet"
    index_rel = _relative(index_path, data_root)
    summary_rel = _relative(summary_path, data_root)
    if (
        index_rel in all_paths_by_rel
        and parquet_reports.get(index_rel) is not None
        and parquet_reports[index_rel]["decode_ok"]
    ):
        _reconcile_loci_index_semantics(
            index_path,
            operational_rows,
            data_root,
            pa,
            ds,
            pq,
            issues,
            batch_size,
        )
    if (
        summary_rel in all_paths_by_rel
        and parquet_reports.get(summary_rel) is not None
        and parquet_reports[summary_rel]["decode_ok"]
    ):
        _reconcile_nightly_summary_semantics(
            summary_path,
            operational_rows,
            manifest_payloads,
            data_root,
            pq,
            issues,
            batch_size,
        )
    return {
        "all_paths": all_paths,
        "parquet_paths": parquet_paths,
        "parquet_size_bytes": parquet_size_bytes,
        "source_identities": source_identities,
        "parquet_reports": parquet_reports,
    }


def _inspect_optional_analysis(data_root, issues):
    """Inventory optional verified analysis artifacts without requiring them."""
    analysis_roots = (
        data_root / "analysis",
        data_root / "data" / "lsst_only" / "analysis",
    )
    paths = []
    source_identities = {}
    for analysis_root in analysis_roots:
        if not os.path.lexists(str(analysis_root)):
            continue
        if not _record_science_path_problems(
            analysis_root, data_root, issues
        ):
            continue
        if not analysis_root.is_dir():
            issues.add(
                "analysis_path_not_directory",
                "Optional analysis path exists but is not a directory.",
                _relative(analysis_root, data_root),
            )
            continue
        for path in _files_under(analysis_root, data_root, issues):
            if not path.is_file():
                continue
            identity = _capture_source_identity(
                path,
                data_root,
                issues,
                "analysis inspection",
                source_identities,
            )
            if identity is not None:
                paths.append(path)
    return {"paths": paths, "source_identities": source_identities}


def _finite_mask(array, np):
    values = array.to_numpy(zero_copy_only=False)
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        numeric = np.asarray(array.to_pylist(), dtype=np.float64)
    return np.isfinite(numeric)


def _scan_feature_coverage(
    loci_paths,
    data_root,
    np,
    pq,
    issues,
    batch_size,
    inspection_identities,
):
    metric_specs = [
        ("feature", feature, (feature,)) for feature in FEATURE_COLUMNS
    ]
    metric_specs.extend(
        ("pairwise", name, columns) for name, columns in PAIRWISE_SPECS
    )
    metrics = {
        (kind, name): {
            "kind": kind,
            "name": name,
            "columns": ";".join(columns),
            "source_file_count": len(loci_paths),
            "files_with_all_columns": 0,
            "files_missing_any_column": 0,
            "total_locus_rows": 0,
            "finite_count": 0,
            "finite_fraction": 0.0,
        }
        for kind, name, columns in metric_specs
    }

    total_rows = 0
    readable_files = 0
    conversion_issue_keys = set()
    feature_identities = {}
    for path in sorted(set(loci_paths), key=lambda item: _relative(item, data_root)):
        rel = _relative(path, data_root)
        if not _record_science_path_problems(path, data_root, issues):
            continue
        feature_identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "feature scan",
            feature_identities,
        )
        if feature_identity is None:
            continue
        inspection_identity = inspection_identities.get(rel)
        if inspection_identity is None:
            issues.add(
                "missing_source_identity",
                "No nightly-inspection identity exists for feature source.",
                rel,
            )
            continue
        if _record_source_identity_change(
            path,
            data_root,
            issues,
            inspection_identity,
            feature_identity,
            "nightly inspection",
        ):
            continue
        try:
            parquet_file = pq.ParquetFile(path)
            schema_names = set(parquet_file.schema_arrow.names)
            row_count = int(parquet_file.metadata.num_rows)
        except Exception as exc:
            issues.add(
                "feature_scan_parquet_error",
                f"Could not prepare loci Parquet for feature scan: "
                f"{type(exc).__name__}: {exc}",
                rel,
            )
            continue

        readable_files += 1
        total_rows += row_count
        present_features = [
            feature for feature in FEATURE_COLUMNS if feature in schema_names
        ]
        for kind, name, columns in metric_specs:
            metric = metrics[(kind, name)]
            metric["total_locus_rows"] += row_count
            if all(column in schema_names for column in columns):
                metric["files_with_all_columns"] += 1
            else:
                metric["files_missing_any_column"] += 1

        if not present_features or row_count == 0:
            continue

        scanned_rows = 0
        try:
            batches = parquet_file.iter_batches(
                batch_size=batch_size, columns=present_features
            )
            for batch in batches:
                batch_rows = int(batch.num_rows)
                scanned_rows += batch_rows
                masks = {}
                for feature in present_features:
                    try:
                        index = batch.schema.get_field_index(feature)
                        masks[feature] = _finite_mask(batch.column(index), np)
                    except Exception as exc:
                        masks[feature] = np.zeros(batch_rows, dtype=bool)
                        key = (rel, feature)
                        if key not in conversion_issue_keys:
                            conversion_issue_keys.add(key)
                            issues.add(
                                "nonnumeric_feature_column",
                                f"Could not interpret {feature} as numeric: "
                                f"{type(exc).__name__}: {exc}",
                                rel,
                            )

                for kind, name, columns in metric_specs:
                    if not all(column in masks for column in columns):
                        continue
                    combined = np.ones(batch_rows, dtype=bool)
                    for column in columns:
                        combined &= masks[column]
                    metrics[(kind, name)]["finite_count"] += int(combined.sum())
        except Exception as exc:
            issues.add(
                "feature_scan_batch_error",
                f"Could not stream feature columns: {type(exc).__name__}: {exc}",
                rel,
            )
        if scanned_rows != row_count:
            issues.add(
                "feature_scan_row_count_mismatch",
                f"Feature scan read {scanned_rows} of {row_count} rows.",
                rel,
            )
        post_scan_identities = {}
        post_scan_identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "feature scan completion",
            post_scan_identities,
        )
        if post_scan_identity is not None:
            _record_source_identity_change(
                path,
                data_root,
                issues,
                feature_identity,
                post_scan_identity,
                "feature scan start",
            )

    rows = []
    for kind, name, _ in metric_specs:
        metric = metrics[(kind, name)]
        denominator = metric["total_locus_rows"]
        metric["finite_fraction"] = (
            metric["finite_count"] / denominator if denominator else 0.0
        )
        rows.append(metric)
    return {
        "rows": rows,
        "source_file_count": len(loci_paths),
        "readable_file_count": readable_files,
        "total_locus_rows": total_rows,
        "source_identities": feature_identities,
    }


def _sha256(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        path_after = path.stat()
    finally:
        os.close(descriptor)
    stable_fields = SOURCE_IDENTITY_FIELDS
    stable = all(
        getattr(before, field) == getattr(after, field)
        for field in stable_fields
    ) and all(
        getattr(after, field) == getattr(path_after, field)
        for field in stable_fields
    )
    return digest.hexdigest(), stable


def _checksum_records(
    paths,
    data_root,
    issues,
    inspection_identities,
    feature_identities=None,
):
    by_relative_path = {}
    for path in paths:
        rel = _relative(path, data_root)
        by_relative_path.setdefault(rel, path)

    records = []
    for rel, path in sorted(by_relative_path.items()):
        if not _record_science_path_problems(path, data_root, issues):
            continue
        inspection_identity = inspection_identities.get(rel)
        if inspection_identity is None:
            issues.add(
                "missing_source_identity",
                "No inspection identity exists for checksum source.",
                rel,
            )
            continue
        checksum_identities = {}
        checksum_identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "checksum",
            checksum_identities,
        )
        if checksum_identity is None:
            continue
        if _record_source_identity_change(
            path,
            data_root,
            issues,
            inspection_identity,
            checksum_identity,
            "inspection",
        ):
            continue
        if (
            feature_identities is not None
            and rel in feature_identities
            and _record_source_identity_change(
                path,
                data_root,
                issues,
                feature_identities[rel],
                checksum_identity,
                "feature scan",
            )
        ):
            continue
        try:
            digest, stable = _sha256(path)
        except OSError as exc:
            issues.add(
                "checksum_read_error",
                f"Could not checksum file: {exc}",
                rel,
            )
            continue
        if not _record_science_path_problems(path, data_root, issues):
            continue
        post_checksum_identities = {}
        post_checksum_identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "checksum completion",
            post_checksum_identities,
        )
        if post_checksum_identity is None:
            continue
        if _record_source_identity_change(
            path,
            data_root,
            issues,
            checksum_identity,
            post_checksum_identity,
            "checksum start",
        ):
            continue
        if not stable:
            issues.add(
                "file_changed_during_audit",
                "File metadata changed while it was being checksummed; the "
                "digest must not be trusted for migration verification.",
                rel,
            )
            continue
        records.append({"sha256": digest, "path": rel})
    return records


def _discover_current_durable_paths(data_root, issues):
    """Rediscover the complete durable file set for final coherence checks."""
    discovered = {}
    roots = (
        (
            data_root / "data" / "lsst_only" / "nightly",
            {"manifest.json", "loci.parquet", "alerts.parquet"},
        ),
        (data_root / "data" / "lsst_only" / "cumulative", None),
        (data_root / "analysis", None),
        (data_root / "data" / "lsst_only" / "analysis", None),
    )
    for root, allowed_names in roots:
        if not os.path.lexists(str(root)):
            continue
        if not _record_science_path_problems(root, data_root, issues):
            continue
        if not root.is_dir():
            continue
        for path in _files_under(root, data_root, issues):
            if allowed_names is not None and path.name not in allowed_names:
                continue
            if not path.is_file():
                continue
            discovered[_relative(path, data_root)] = path
    return discovered


def _finalize_source_snapshot(
    data_root, issues, initial_paths, inspection_identities
):
    """Require a stable path set and identities after all checksum reads."""
    current_paths = _discover_current_durable_paths(data_root, issues)
    initial_names = set(initial_paths)
    current_names = set(current_paths)
    for rel in sorted(current_names - initial_names):
        issues.add(
            "durable_path_set_changed",
            "Durable science file was added after initial inspection.",
            rel,
        )
    for rel in sorted(initial_names - current_names):
        issues.add(
            "durable_path_set_changed",
            "Durable science file was removed after initial inspection.",
            rel,
        )

    for rel, expected_identity in sorted(inspection_identities.items()):
        path = current_paths.get(rel)
        if path is None:
            continue
        final_identities = {}
        final_identity = _capture_source_identity(
            path,
            data_root,
            issues,
            "final source snapshot",
            final_identities,
        )
        if final_identity is None:
            continue
        _record_source_identity_change(
            path,
            data_root,
            issues,
            expected_identity,
            final_identity,
            "initial inspection",
        )
    return current_paths


def _render_checksum_records(records):
    return "".join(f"{record['sha256']}  {record['path']}\n" for record in records)


def _csv_text(rows, fields):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in fields
            }
        )
    return buffer.getvalue()


def _manifest_table_text(rows, data_root):
    fields = (
        "date_utc",
        "status",
        "actual_loci",
        "alert_rows",
        "loci_rows",
        "alerts_rows",
        "science_files_complete",
        "zero_row_policy_accepted",
        "operationally_complete",
        "integrity_issue_codes",
        "manifest_path",
    )
    lines = [
        "ANTARES nightly manifest audit",
        f"Data root: {data_root}",
        f"Nightly partition count: {len(rows)}",
        f"Manifest count: {sum(row['manifest_exists'] for row in rows)}",
        "",
        "\t".join(fields),
    ]
    for row in rows:
        lines.append(
            "\t".join(
                "" if row.get(field) is None else str(row.get(field))
                for field in fields
            )
        )
    return "\n".join(lines) + "\n"


def _human_bytes(value):
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} B"
    return f"{amount:.2f} {unit}"


def _size_summary_text(summary):
    values = (
        ("Root regular-file bytes", summary["root_size_bytes"]),
        (
            "Root excluding cache regular-file bytes",
            summary["root_excluding_cache_size_bytes"],
        ),
        ("Cache regular-file bytes", summary["cache_size_bytes"]),
        ("Nightly loci.parquet bytes", summary["total_loci_parquet_bytes"]),
        ("Nightly alerts.parquet bytes", summary["total_alerts_parquet_bytes"]),
        (
            "Cumulative Parquet bytes",
            summary["total_cumulative_parquet_bytes"],
        ),
    )
    lines = [
        "ANTARES data-root size summary",
        f"Data root: {summary['data_root']}",
        "Sizes are logical bytes of regular files (not allocated du blocks).",
        "",
    ]
    for label, value in values:
        lines.append(f"{label}: {value} ({_human_bytes(value)})")
    lines.extend(
        [
            "",
            f"Bytes per alert row: {summary['bytes_per_alert_row']}",
            f"Bytes per locus: {summary['bytes_per_locus']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _feature_summary_text(feature_scan, data_root):
    lines = [
        "ANTARES locus-feature finite coverage",
        f"Data root: {data_root}",
        f"Loci Parquet files discovered: {feature_scan['source_file_count']}",
        f"Loci Parquet files readable: {feature_scan['readable_file_count']}",
        f"Total locus rows scanned: {feature_scan['total_locus_rows']}",
        (
            "Fractions use all locus rows as the denominator; a missing feature "
            "column contributes zero finite rows."
        ),
        (
            "Feature-space plot sample sizes are limited by sparse finite and "
            "pairwise-complete feature coverage; they are not the full locus "
            "population unless the reported coverage supports that claim."
        ),
        "",
    ]
    for row in feature_scan["rows"]:
        lines.append(
            f"{row['kind']} {row['name']}: "
            f"{row['finite_count']}/{row['total_locus_rows']} finite "
            f"({row['finite_fraction']:.6%}); "
            f"all columns present in {row['files_with_all_columns']}/"
            f"{row['source_file_count']} files; columns={row['columns']}"
        )
    return "\n".join(lines) + "\n"


def _build_summary(
    data_root,
    generated_at,
    nightly,
    cumulative,
    inventories,
    feature_scan,
    checksum_sets,
    issues,
):
    rows = nightly["rows"]
    valid_dates = sorted(
        row["date_utc"]
        for row in rows
        if _valid_iso_date(row.get("date_utc")) is not None
    )
    total_loci = sum(
        row["actual_loci"] for row in rows if row["actual_loci"] is not None
    )
    total_alert_rows = sum(
        row["alert_rows"] for row in rows if row["alert_rows"] is not None
    )
    loci_bytes = sum(
        row["loci_size_bytes"]
        for row in rows
        if row["loci_size_bytes"] is not None
    )
    alerts_bytes = sum(
        row["alerts_size_bytes"]
        for row in rows
        if row["alerts_size_bytes"] is not None
    )
    physical_loci_rows = sum(
        row["loci_rows"] for row in rows if row["loci_rows"] is not None
    )
    physical_alert_rows = sum(
        row["alerts_rows"] for row in rows if row["alerts_rows"] is not None
    )
    status_counts = Counter(
        row["status"] if row["status"] is not None else "[missing]" for row in rows
    )
    sorted_issues = issues.sorted_items()
    incomplete_evidence_codes = sorted(
        {
            item["code"]
            for item in sorted_issues
            if item["code"] in EVIDENCE_INCOMPLETE_CODES
        }
    )
    evidence_complete = not incomplete_evidence_codes
    feature_counts = {
        f"{row['kind']}:{row['name']}": {
            "columns": row["columns"].split(";"),
            "finite_count": row["finite_count"],
            "finite_fraction": row["finite_fraction"],
        }
        for row in feature_scan["rows"]
    }

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_complete": evidence_complete,
        "audit_complete_definition": (
            "True only when durable-source traversal, full Parquet decoding, "
            "checksumming, and the final coherent source snapshot completed "
            "without evidence gaps. It is independent of whether integrity "
            "checks passed."
        ),
        "report_set_complete": True,
        "report_set_complete_definition": (
            "True means summary.json was published only after the other nine "
            "required reports were persisted."
        ),
        "audit_status": "PASS" if not sorted_issues else "FAIL",
        "data_root": str(data_root),
        "hostname": socket.gethostname(),
        "timestamp_utc": generated_at,
        "nightly_manifest_count": len(nightly["manifest_paths"]),
        "nightly_partition_count": len(rows),
        "first_date": valid_dates[0] if valid_dates else None,
        "last_date": valid_dates[-1] if valid_dates else None,
        "complete_nights": sum(row["operationally_complete"] for row in rows),
        "zero_row_policy_accepted_nights": sum(
            row["zero_row_policy_accepted"] for row in rows
        ),
        "status_complete_nights": sum(
            row["status"] == "complete" for row in rows
        ),
        "physically_complete_nights": sum(
            row["science_files_complete"] for row in rows
        ),
        "status_complete_nights_with_all_science_files": sum(
            row["status"] == "complete" and row["science_files_complete"]
            for row in rows
        ),
        "nightly_count_definitions": {
            "complete_nights": (
                "Parseable manifest plus readable sibling loci.parquet and "
                "alerts.parquet with matching manifest row counts, an "
                "appendable/non-failed status, and either append_ready not "
                "false or a completed, schema-valid, error-free LSST-only "
                "zero-row night."
            ),
            "status_complete_nights": (
                "Manifests whose literal status value is 'complete'."
            ),
            "physically_complete_nights": (
                "Parseable manifest plus readable sibling loci.parquet and "
                "alerts.parquet with matching manifest row counts, regardless "
                "of manifest status."
            ),
        },
        "nightly_status_counts": dict(sorted(status_counts.items())),
        "total_actual_loci": total_loci,
        "total_alert_rows": total_alert_rows,
        "total_loci_parquet_rows": physical_loci_rows,
        "total_alerts_parquet_rows": physical_alert_rows,
        "total_loci_parquet_bytes": loci_bytes,
        "total_alerts_parquet_bytes": alerts_bytes,
        "total_cumulative_parquet_bytes": cumulative["parquet_size_bytes"],
        "bytes_per_alert_row": (
            alerts_bytes / total_alert_rows if total_alert_rows else None
        ),
        "bytes_per_locus": loci_bytes / total_loci if total_loci else None,
        "root_file_count": inventories["root"]["file_count"],
        "root_size_bytes": inventories["root"]["size_bytes"],
        "root_excluding_cache_file_count": inventories["root_excluding_cache"][
            "file_count"
        ],
        "root_excluding_cache_size_bytes": inventories[
            "root_excluding_cache"
        ]["size_bytes"],
        "cache_present": inventories["cache"]["present"],
        "cache_file_count": inventories["cache"]["file_count"],
        "cache_size_bytes": inventories["cache"]["size_bytes"],
        "cumulative_parquet_file_count": len(cumulative["parquet_paths"]),
        "feature_coverage": feature_counts,
        "checksum_file_counts": {
            name: len(records) for name, records in checksum_sets.items()
        },
        "integrity": {
            "ok": not sorted_issues,
            "issue_count": len(sorted_issues),
            "issues": sorted_issues,
            "incomplete_evidence_issue_codes": incomplete_evidence_codes,
        },
    }


def _write_outputs_exclusively(out, outputs):
    missing = [name for name in OUTPUT_NAMES if name not in outputs]
    if missing:
        raise AuditPreflightError(
            "Internal error: required outputs were not prepared: "
            + ", ".join(missing)
        )
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise AuditPreflightError(
            f"Could not create audit output directory {out}: {exc}"
        ) from exc

    # A summary declaring audit_complete=true is published only after every
    # other required report has been closed successfully.
    for name in (item for item in OUTPUT_NAMES if item != "summary.json"):
        path = out / name
        try:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(outputs[name])
        except OSError as exc:
            raise AuditPreflightError(
                f"Could not write complete audit output {path}: {exc}"
            ) from exc

    summary_path = out / "summary.json"
    staged_summary_path = out / ".summary.json.tmp"
    try:
        with staged_summary_path.open(
            "x", encoding="utf-8", newline=""
        ) as handle:
            handle.write(outputs["summary.json"])
        # Hard-link publication is atomic and fails rather than overwriting an
        # unexpectedly created summary path.
        os.link(staged_summary_path, summary_path, follow_symlinks=False)
        staged_summary_path.unlink()
    except OSError as exc:
        raise AuditPreflightError(
            f"Could not publish complete audit summary {summary_path}: {exc}"
        ) from exc


def audit_data_root(data_root, out, batch_size=DEFAULT_BATCH_SIZE):
    """Audit ``data_root``, write all ten reports to a new ``out`` directory.

    The returned object is the same dictionary written to ``summary.json``.
    ``AuditPreflightError`` is raised rather than reusing or overwriting an
    existing output directory.
    """
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise AuditPreflightError("batch_size must be a positive integer.")
    batch_size = int(batch_size)
    root, audit_out = _validate_paths(data_root, out)
    np, pa, ds, pq = _load_columnar_dependencies()
    generated_at = _now_utc()
    issues = IntegrityIssueCollector()

    # Inventory is deliberately captured before the external output directory
    # is created, so the source snapshot always precedes report persistence.
    inventories = _scan_root_inventory(root, issues)
    nightly = _inspect_nightly(root, pq, issues, batch_size)
    cumulative = _inspect_cumulative(
        root,
        pa,
        ds,
        pq,
        issues,
        batch_size,
        nightly["rows"],
        nightly["manifest_payloads"],
    )
    analysis = _inspect_optional_analysis(root, issues)
    inspection_identities = {
        **nightly["source_identities"],
        **cumulative["source_identities"],
        **analysis["source_identities"],
    }
    feature_scan = _scan_feature_coverage(
        nightly["loci_paths"],
        root,
        np,
        pq,
        issues,
        batch_size,
        inspection_identities,
    )

    nightly_checksums = _checksum_records(
        nightly["nightly_parquet_paths"],
        root,
        issues,
        inspection_identities,
        feature_scan["source_identities"],
    )
    cumulative_checksums = _checksum_records(
        cumulative["parquet_paths"],
        root,
        issues,
        inspection_identities,
    )
    all_science_checksums = _checksum_records(
        nightly["nightly_science_paths"]
        + cumulative["all_paths"]
        + analysis["paths"],
        root,
        issues,
        inspection_identities,
        feature_scan["source_identities"],
    )
    checksum_sets = {
        "nightly_parquet_sha256": nightly_checksums,
        "cumulative_parquet_sha256": cumulative_checksums,
        "all_science_files_sha256": all_science_checksums,
    }
    initial_paths = {
        _relative(path, root): path
        for path in (
            nightly["nightly_science_paths"]
            + cumulative["all_paths"]
            + analysis["paths"]
        )
    }
    _finalize_source_snapshot(
        root, issues, initial_paths, inspection_identities
    )

    file_counts = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "data_root": str(root),
        "timestamp_utc": generated_at,
        **inventories,
        "science_products": {
            "nightly_manifest_count": len(nightly["manifest_paths"]),
            "nightly_loci_parquet_count": sum(
                row["loci_exists"] for row in nightly["rows"]
            ),
            "nightly_alerts_parquet_count": sum(
                row["alerts_exists"] for row in nightly["rows"]
            ),
            "analysis_file_count": len(analysis["paths"]),
        },
    }
    summary = _build_summary(
        root,
        generated_at,
        nightly,
        cumulative,
        inventories,
        feature_scan,
        checksum_sets,
        issues,
    )
    outputs = {
        "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        "nightly_manifest_table.csv": _csv_text(
            nightly["rows"], MANIFEST_TABLE_FIELDS
        ),
        "nightly_manifest_table.txt": _manifest_table_text(
            nightly["rows"], root
        ),
        "file_counts.json": json.dumps(
            file_counts, indent=2, sort_keys=True
        )
        + "\n",
        "size_summary.txt": _size_summary_text(summary),
        "nightly_parquet_sha256.txt": _render_checksum_records(
            nightly_checksums
        ),
        "cumulative_parquet_sha256.txt": _render_checksum_records(
            cumulative_checksums
        ),
        "all_science_files_sha256.txt": _render_checksum_records(
            all_science_checksums
        ),
        "feature_coverage.csv": _csv_text(
            feature_scan["rows"],
            (
                "kind",
                "name",
                "columns",
                "source_file_count",
                "files_with_all_columns",
                "files_missing_any_column",
                "total_locus_rows",
                "finite_count",
                "finite_fraction",
            ),
        ),
        "feature_coverage_summary.txt": _feature_summary_text(
            feature_scan, root
        ),
    }
    _write_outputs_exclusively(audit_out, outputs)
    return summary


def _positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit durable ANTARES nightly/cumulative science products and "
            "summarize cache separately without modifying the data root."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ANTARES_Analysis_Data root to audit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "New output directory outside the data root. Existing paths are "
            "never overwritten."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Rows per PyArrow feature-scan batch "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    raw_audit_out = args.out.expanduser()
    display_audit_out = Path(os.path.abspath(str(raw_audit_out)))
    print(f"ANTARES data root: {data_root}", flush=True)
    print(f"Audit output directory: {display_audit_out}", flush=True)
    try:
        summary = audit_data_root(
            data_root, raw_audit_out, batch_size=args.batch_size
        )
    except AuditPreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:
        print(
            "ERROR: unexpected audit failure; the output must not be treated "
            f"as complete: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_ERROR

    issue_count = summary["integrity"]["issue_count"]
    if issue_count:
        print(
            f"AUDIT FAIL: wrote all {len(OUTPUT_NAMES)} reports; found "
            f"{issue_count} integrity issue(s).",
            flush=True,
        )
        return EXIT_INTEGRITY_ISSUES
    print(
        f"AUDIT PASS: wrote all {len(OUTPUT_NAMES)} reports; no integrity "
        "issues found.",
        flush=True,
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
