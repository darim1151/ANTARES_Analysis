"""
Cumulative nightly history pipeline for ANTARES/LSST analyses.

The existing notebook still does the quick "last night vs sampled history"
comparison. This module adds a platform-backed research store that can be
built night by night and resumed after disconnects:

    data/lsst_only/nightly/YYYY/MM/DD/loci.parquet
    data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet
    data/lsst_only/nightly/YYYY/MM/DD/manifest.json
    data/lsst_only/cumulative/loci_index.parquet
    data/lsst_only/cumulative/nightly_summary.parquet

Each nightly partition is saved immediately after ingestion. The cumulative
tables are compact indexes rebuilt from those manifests and parquet files.
"""

import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from astropy.time import Time

from . import chunked_query, lightcurves, query
try:
    from . import rsp_permissions
except ImportError:  # Backward compatibility with older clean RSP checkouts.
    rsp_permissions = None
from .config import (
    CHUNK_INITIAL_DAYS,
    CHUNK_MAX_RESULTS,
    CHUNK_MIN_SECONDS,
    CHUNK_PARALLEL_SHARDS,
    CHUNK_SPLIT_THRESHOLD,
    HISTORY_DATA_ROOT,
    HISTORY_DATA_SUBDIR,
    HISTORY_FETCH_ALL_LIGHTCURVES,
    HISTORY_MAX_LIGHTCURVE_WORKERS,
    HISTORY_RESUME_EXISTING_NIGHTS,
    HISTORY_TARGET_LOCI,
    LSST_HISTORY_START_MJD,
    LSST_ONLY,
    QUERY_TAG,
    SURVEY_MODE,
)


MJD_COL = "newest_alert_observation_time"
LOCUS_ID_COL = "locus_id"
SOURCE_QUERY_MODE = "adaptive_chunked"
APPENDABLE_STATUSES = {"complete", "under_target", "saturated_unresolved"}

CUMULATIVE_INDEX_COLUMNS = [
    LOCUS_ID_COL,
    "night_date_utc",
    "night_mjd_min",
    "night_mjd_max",
    "ingested_at_utc",
    "source_query_mode",
    MJD_COL,
    "ra",
    "dec",
    "tags",
    "survey",
    "ztf_object_id",
    "dia_object_id",
    "ss_object_id",
    "brightest_alert_magnitude",
    "num_mag_values",
]

ZERO_ROW_LOCI_REQUIRED_COLUMNS = {
    LOCUS_ID_COL,
    "ra",
    "dec",
    MJD_COL,
    "night_date_utc",
    "night_mjd_min",
    "night_mjd_max",
    "ingested_at_utc",
    "source_query_mode",
}
ZERO_ROW_ALERTS_REQUIRED_COLUMNS = {
    LOCUS_ID_COL,
    "night_date_utc",
    "range_label",
}
ZERO_ROW_REVALIDATION_POLICY = "valid_zero_row_lsst_only_v1"


class ProductionWriterUnavailable(RuntimeError):
    """Direct live-query publication is absent from the Phase 5 release."""


def _ensure_storage_path(path):
    """Create a storage directory using the configured portability policy."""
    path = Path(path)
    if rsp_permissions is not None:
        helper = getattr(rsp_permissions, "ensure_storage_path", None)
        if helper is not None:
            return helper(path)
        # Older RSP checkouts predate the policy-neutral helper and are
        # explicitly shared-group deployments.
        return rsp_permissions.ensure_group_shared_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mark_file_for_storage(path):
    """Apply the configured storage policy to one newly written file."""
    path = Path(path)
    if rsp_permissions is not None:
        helper = getattr(rsp_permissions, "mark_file_for_storage", None)
        if helper is not None:
            helper(path)
        else:
            # Backward compatibility with older shared-group RSP checkouts.
            rsp_permissions.mark_file_group_writable(path)
    elif path.exists():
        path.chmod(path.stat().st_mode | 0o600)


def _now_utc():
    """Return an ISO-8601 UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mjd_to_utc_date(mjd):
    """Return the zero-padded UTC calendar date for an MJD boundary."""
    return Time(float(mjd), format="mjd", scale="utc").iso[:10]


def display_date(date_utc):
    """Return a compact display date such as 2026/4/28 from YYYY-MM-DD."""
    year, month, day = date_utc.split("-")
    return f"{int(year)}/{int(month)}/{int(day)}"


def date_folder(date_utc):
    """Return the sortable nested folder parts for YYYY-MM-DD."""
    year, month, day = date_utc.split("-")
    return year, month, day


def survey_data_root(data_root, survey_subdir=HISTORY_DATA_SUBDIR):
    """Return the data root for one survey mode under the persistent store."""
    root = Path(data_root) / "data"
    return root / survey_subdir if survey_subdir else root


def nightly_dir(data_root, date_utc):
    """Return the persistent directory for one UTC night."""
    year, month, day = date_folder(date_utc)
    return survey_data_root(data_root) / "nightly" / year / month / day


def nightly_paths(data_root, date_utc):
    """Return the standard parquet/manifest paths for one UTC night."""
    root = nightly_dir(data_root, date_utc)
    return {
        "dir": root,
        "loci": root / "loci.parquet",
        "alerts": root / "alerts.parquet",
        "manifest": root / "manifest.json",
    }


def cumulative_paths(data_root):
    """Return the standard cumulative index paths."""
    root = survey_data_root(data_root) / "cumulative"
    return {
        "dir": root,
        "loci_index": root / "loci_index.parquet",
        "nightly_summary": root / "nightly_summary.parquet",
    }


def iter_night_windows(mjd_start, mjd_stop):
    """
    Yield one-day MJD windows from `mjd_start` up to `mjd_stop`.

    The final window is shortened if `mjd_stop` is not an integer day.
    """
    cursor = float(mjd_start)
    stop = float(mjd_stop)
    while cursor < stop:
        end = min(cursor + 1.0, stop)
        yield mjd_to_utc_date(cursor), cursor, end
        cursor = end


def read_manifest(data_root, date_utc):
    """Load a nightly manifest if it exists; otherwise return None."""
    path = nightly_paths(data_root, date_utc)["manifest"]
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    """Write a JSON file with stable indentation."""
    _ensure_storage_path(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    _mark_file_for_storage(path)


def _safe_to_parquet(df, path):
    """Write a DataFrame to parquet, creating parent directories first."""
    _ensure_storage_path(path.parent)
    df.to_parquet(path, index=False)
    _mark_file_for_storage(path)


def _empty_alerts_frame():
    """Return an empty alerts table with enough schema to parquet round-trip."""
    return pd.DataFrame({
        LOCUS_ID_COL: pd.Series(dtype="object"),
        "night_date_utc": pd.Series(dtype="object"),
        "range_label": pd.Series(dtype="object"),
    })


def prepare_loci(df_loci, date_utc, mjd_min, mjd_max, ingested_at_utc):
    """Attach nightly provenance columns to a loci table."""
    df = df_loci.copy()
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame({
            LOCUS_ID_COL: pd.Series(dtype="object"),
            "ra": pd.Series(dtype="float64"),
            "dec": pd.Series(dtype="float64"),
            MJD_COL: pd.Series(dtype="float64"),
        })
    df["night_date_utc"] = date_utc
    df["night_mjd_min"] = float(mjd_min)
    df["night_mjd_max"] = float(mjd_max)
    df["ingested_at_utc"] = ingested_at_utc
    df["source_query_mode"] = SOURCE_QUERY_MODE
    return df.reset_index(drop=True)


def prepare_alerts(df_alerts, date_utc, range_label):
    """Attach nightly provenance columns to an alert/lightcurve table."""
    if df_alerts is None or (df_alerts.empty and len(df_alerts.columns) == 0):
        df = _empty_alerts_frame()
    else:
        df = df_alerts.copy()
        if LOCUS_ID_COL not in df.columns:
            df[LOCUS_ID_COL] = pd.Series(dtype="object")
    df["night_date_utc"] = date_utc
    df["range_label"] = range_label
    return df.reset_index(drop=True)


def validation_summary(
    df_loci,
    df_alerts,
    mjd_min,
    mjd_max,
    prior_locus_ids=None,
    lsst_only=LSST_ONLY,
    query_completed=True,
    query_fetch_clean=True,
):
    """Return machine-readable validation fields for a nightly manifest."""
    zero_row_night = bool(
        df_loci is not None
        and df_loci.empty
        and df_alerts is not None
        and df_alerts.empty
    )
    zero_row_schema_pass = bool(
        zero_row_night
        and ZERO_ROW_LOCI_REQUIRED_COLUMNS.issubset(df_loci.columns)
        and ZERO_ROW_ALERTS_REQUIRED_COLUMNS.issubset(df_alerts.columns)
    )
    validation = {
        "mjd_pass": True,
        "mjd_missing_column": MJD_COL not in df_loci.columns,
        "mjd_below_count": 0,
        "mjd_above_count": 0,
        "duplicate_locus_count": 0,
        "coordinate_pass": True,
        "coordinate_missing_columns": not {"ra", "dec"}.issubset(df_loci.columns),
        "bad_ra_count": 0,
        "bad_dec_count": 0,
        "overlap_count": 0,
        "alert_locus_link_pass": True,
        "alert_rows_without_locus": 0,
        "lsst_only_pass": True,
        "lsst_identifier_count": 0,
        "lsst_dia_count": 0,
        "lsst_ss_count": 0,
        "ztf_object_id_count": 0,
        "history_start_pass": True,
        "query_completed_pass": bool(query_completed),
        "query_fetch_clean": bool(query_fetch_clean),
        "zero_row_night": zero_row_night,
        "zero_row_schema_pass": (
            zero_row_schema_pass if zero_row_night else None
        ),
    }

    survey_counts = query.lsst_identifier_counts(df_loci)
    validation.update(survey_counts)
    if lsst_only:
        if zero_row_night:
            validation["lsst_only_pass"] = zero_row_schema_pass
        else:
            validation["lsst_only_pass"] = (
                not df_loci.empty
                and validation["lsst_identifier_count"] == int(len(df_loci))
            )
        validation["history_start_pass"] = float(mjd_min) >= float(LSST_HISTORY_START_MJD)

    if not df_loci.empty and MJD_COL in df_loci.columns:
        mjds = df_loci[MJD_COL].dropna()
        validation["mjd_below_count"] = int((mjds < float(mjd_min)).sum())
        validation["mjd_above_count"] = int((mjds > float(mjd_max)).sum())
        validation["mjd_pass"] = (
            validation["mjd_below_count"] == 0 and validation["mjd_above_count"] == 0
        )
    elif not df_loci.empty:
        validation["mjd_pass"] = False

    if LOCUS_ID_COL in df_loci.columns:
        validation["duplicate_locus_count"] = int(df_loci[LOCUS_ID_COL].duplicated().sum())
        locus_ids = set(df_loci[LOCUS_ID_COL].dropna().astype(str))
    else:
        locus_ids = set()

    if not df_loci.empty and {"ra", "dec"}.issubset(df_loci.columns):
        validation["bad_ra_count"] = int((~df_loci["ra"].between(0, 360, inclusive="left")).sum())
        validation["bad_dec_count"] = int((~df_loci["dec"].between(-90, 90)).sum())
        validation["coordinate_pass"] = (
            validation["bad_ra_count"] == 0 and validation["bad_dec_count"] == 0
        )
    elif not df_loci.empty:
        validation["coordinate_pass"] = False

    if prior_locus_ids is not None and locus_ids:
        prior = {str(value) for value in prior_locus_ids}
        validation["overlap_count"] = int(len(locus_ids & prior))

    if df_alerts is not None and not df_alerts.empty:
        if LOCUS_ID_COL not in df_alerts.columns:
            validation["alert_locus_link_pass"] = False
            validation["alert_rows_without_locus"] = int(len(df_alerts))
        else:
            alert_ids = df_alerts[LOCUS_ID_COL]
            missing_mask = alert_ids.isna() | ~alert_ids.astype(str).isin(locus_ids)
            validation["alert_rows_without_locus"] = int(missing_mask.sum())
            validation["alert_locus_link_pass"] = validation["alert_rows_without_locus"] == 0

    validation["append_ready"] = bool(
        validation["mjd_pass"]
        and validation["duplicate_locus_count"] == 0
        and validation["coordinate_pass"]
        and validation["alert_locus_link_pass"]
        and validation["lsst_only_pass"]
        and validation["history_start_pass"]
        and validation["query_completed_pass"]
        and validation["query_fetch_clean"]
        and validation["zero_row_schema_pass"] is not False
    )
    return validation


def _meaningful_error_value(value):
    """Return whether a recorded error field represents an actual error."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def recorded_query_fetch_errors(manifest):
    """Return non-empty recorded query/fetch error fields with dotted paths."""
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

    visit(manifest)
    return findings


def revalidate_zero_row_night(data_root, date_utc):
    """Regenerate one valid empty-night manifest from its durable products.

    This is intentionally conservative: it accepts only a completed,
    error-free, structurally valid LSST-only night whose manifest and both
    sibling Parquet products all report zero rows. It returns a regenerated
    manifest but performs no write; callers can validate an entire repair set
    before persisting any source change.
    """
    paths = nightly_paths(data_root, date_utc)
    manifest = read_manifest(data_root, date_utc)
    if manifest is None:
        raise FileNotFoundError(f"Missing nightly manifest: {paths['manifest']}")
    if manifest.get("date_utc") != date_utc:
        raise ValueError(
            f"Manifest date {manifest.get('date_utc')!r} does not match "
            f"requested date {date_utc}."
        )
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Zero-row revalidation requires status='complete'; found "
            f"{manifest.get('status')!r} for {date_utc}."
        )
    if manifest.get("actual_loci") != 0 or manifest.get("alert_rows") != 0:
        raise ValueError(
            f"Zero-row revalidation requires actual_loci=0 and alert_rows=0 "
            f"for {date_utc}."
        )
    chunk_count = manifest.get("chunk_count")
    if (
        isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count <= 0
    ):
        raise ValueError(
            f"Zero-row revalidation requires a positive chunk_count for "
            f"{date_utc}; found {chunk_count!r}."
        )
    if manifest.get("saturated_chunk_count") not in (0, 0.0):
        raise ValueError(
            f"Zero-row revalidation rejects saturated query chunks for "
            f"{date_utc}."
        )
    if not manifest.get("finished_at_utc"):
        raise ValueError(
            f"Zero-row revalidation requires finished_at_utc for {date_utc}."
        )
    recorded_errors = recorded_query_fetch_errors(manifest)
    if recorded_errors:
        raise ValueError(
            f"Zero-row revalidation found recorded query/fetch errors for "
            f"{date_utc}: {recorded_errors}"
        )

    for name in ("loci", "alerts"):
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"Missing zero-row science product: {paths[name]}"
            )
    try:
        df_loci = pd.read_parquet(paths["loci"])
        df_alerts = pd.read_parquet(paths["alerts"])
    except Exception as exc:
        raise ValueError(
            f"Could not read zero-row Parquet products for {date_utc}: {exc}"
        ) from exc
    if not df_loci.empty or not df_alerts.empty:
        raise ValueError(
            f"Zero-row revalidation found non-empty Parquet data for "
            f"{date_utc}: loci={len(df_loci)}, alerts={len(df_alerts)}."
        )

    validation = validation_summary(
        df_loci,
        df_alerts,
        mjd_min=manifest.get("mjd_min"),
        mjd_max=manifest.get("mjd_max"),
        prior_locus_ids=None,
        lsst_only=bool(manifest.get("lsst_filter_used", True)),
        query_completed=True,
        query_fetch_clean=True,
    )
    if not validation.get("append_ready"):
        raise ValueError(
            f"Zero-row validation did not pass for {date_utc}: {validation}"
        )

    regenerated = deepcopy(manifest)
    regenerated.update({
        "actual_loci": 0,
        "alert_rows": 0,
        "lsst_dia_count": 0,
        "lsst_ss_count": 0,
        "ztf_object_id_count": 0,
        "status": "complete",
        "validation": validation,
        "paths": {
            "loci": str(paths["loci"]),
            "alerts": str(paths["alerts"]),
            "manifest": str(paths["manifest"]),
        },
        "revalidated_at_utc": _now_utc(),
        "revalidation_policy": ZERO_ROW_REVALIDATION_POLICY,
    })
    return regenerated


def _status_from_counts(actual_loci, target_loci, saturated_chunk_count, failed=False):
    """Convert count/report outcomes into the manifest status label."""
    if failed:
        return "failed"
    if saturated_chunk_count:
        return "saturated_unresolved"
    if int(actual_loci) >= int(target_loci):
        return "complete"
    return "under_target"


def _report_counts(report_df):
    """Summarise adaptive chunk report rows for the nightly manifest."""
    if report_df is None or report_df.empty or "status" not in report_df.columns:
        return {"chunk_count": 0, "split_count": 0, "saturated_chunk_count": 0}
    statuses = report_df["status"].fillna("")
    return {
        "chunk_count": int(statuses.str.startswith("accepted").sum()),
        "split_count": int((statuses == "split").sum()),
        "saturated_chunk_count": int((statuses == "accepted_at_minimum_saturated").sum()),
    }


def _manifest_to_summary_row(manifest):
    """Flatten one manifest into a row for nightly_summary.parquet."""
    validation = manifest.get("validation", {})
    return {
        "date_utc": manifest.get("date_utc"),
        "display_date": display_date(manifest["date_utc"]) if manifest.get("date_utc") else None,
        "mjd_min": manifest.get("mjd_min"),
        "mjd_max": manifest.get("mjd_max"),
        "query_tag": manifest.get("query_tag"),
        "target_loci": manifest.get("target_loci"),
        "actual_loci": manifest.get("actual_loci"),
        "alert_rows": manifest.get("alert_rows"),
        "chunk_count": manifest.get("chunk_count"),
        "split_count": manifest.get("split_count"),
        "saturated_chunk_count": manifest.get("saturated_chunk_count"),
        "status": manifest.get("status"),
        "append_ready": validation.get("append_ready"),
        "mjd_pass": validation.get("mjd_pass"),
        "duplicate_locus_count": validation.get("duplicate_locus_count"),
        "coordinate_pass": validation.get("coordinate_pass"),
        "overlap_count": validation.get("overlap_count"),
        "alert_locus_link_pass": validation.get("alert_locus_link_pass"),
        "survey_mode": manifest.get("survey_mode"),
        "lsst_filter_used": manifest.get("lsst_filter_used"),
        "parallel_shards": manifest.get("parallel_shards"),
        "lsst_dia_count": manifest.get("lsst_dia_count"),
        "lsst_ss_count": manifest.get("lsst_ss_count"),
        "ztf_object_id_count": manifest.get("ztf_object_id_count"),
        "lsst_only_pass": validation.get("lsst_only_pass"),
        "history_start_pass": validation.get("history_start_pass"),
        "runtime_seconds": manifest.get("runtime_seconds"),
        "started_at_utc": manifest.get("started_at_utc"),
        "finished_at_utc": manifest.get("finished_at_utc"),
        "loci_path": manifest.get("paths", {}).get("loci"),
        "alerts_path": manifest.get("paths", {}).get("alerts"),
        "manifest_path": manifest.get("paths", {}).get("manifest"),
    }


def _resume_ready(paths):
    """Return True when a nightly partition has all expected outputs."""
    return paths["manifest"].exists() and paths["loci"].exists() and paths["alerts"].exists()


def ingest_night(
    data_root=HISTORY_DATA_ROOT,
    mjd_min=None,
    mjd_max=None,
    target_loci=HISTORY_TARGET_LOCI,
    query_tag=QUERY_TAG,
    fetch_lightcurves=HISTORY_FETCH_ALL_LIGHTCURVES,
    resume=HISTORY_RESUME_EXISTING_NIGHTS,
    range_label=None,
    chunk_cache_dir=None,
    initial_chunk_days=CHUNK_INITIAL_DAYS,
    min_chunk_seconds=CHUNK_MIN_SECONDS,
    max_results_per_chunk=CHUNK_MAX_RESULTS,
    split_threshold=CHUNK_SPLIT_THRESHOLD,
    max_lightcurve_workers=HISTORY_MAX_LIGHTCURVE_WORKERS,
    parallel_shards=CHUNK_PARALLEL_SHARDS,
    lsst_only=LSST_ONLY,
    prior_locus_ids=None,
    update_indexes=True,
    verbose=True,
):
    """
    Ingest one UTC night, save parquet outputs, and write a manifest.

    Returns a dict containing `manifest`, `df_loci`, `df_alerts`, `report`,
    and `skipped`. In resume mode, existing complete partitions are loaded and
    returned without live ANTARES calls.
    """
    # This legacy notebook-era entry point combined live ANTARES access and
    # direct filesystem publication without the transaction capability.  It
    # is deliberately sealed before validating paths or invoking any provider.
    # The guarded operations writer uses extracted preparation/validation
    # functions and cannot construct a live provider in this release.
    raise ProductionWriterUnavailable(
        "Direct history.ingest_night execution is disabled. Use the operations "
        "planner; production execution requires a future authorization release."
    )

    if mjd_min is None or mjd_max is None:
        raise ValueError("mjd_min and mjd_max are required")

    date_utc = mjd_to_utc_date(mjd_min)
    label = range_label or f"Night {display_date(date_utc)}"
    paths = nightly_paths(data_root, date_utc)

    if resume and _resume_ready(paths):
        manifest = read_manifest(data_root, date_utc)
        if manifest and manifest.get("status") != "failed":
            if verbose:
                print(f"  Resume: found {date_utc}; loading existing nightly partition.")
            df_loci = pd.read_parquet(paths["loci"])
            df_alerts = pd.read_parquet(paths["alerts"])
            if update_indexes:
                update_cumulative_indexes(data_root)
            return {
                "manifest": manifest,
                "df_loci": df_loci,
                "df_alerts": df_alerts,
                "report": pd.DataFrame(),
                "skipped": True,
            }

    started_at = _now_utc()
    t0 = time.time()
    manifest = {
        "date_utc": date_utc,
        "mjd_min": float(mjd_min),
        "mjd_max": float(mjd_max),
        "query_tag": query_tag,
        "target_loci": int(target_loci),
        "actual_loci": 0,
        "alert_rows": 0,
        "chunk_count": 0,
        "split_count": 0,
        "saturated_chunk_count": 0,
        "status": "failed",
        "survey_mode": SURVEY_MODE,
        "lsst_filter_used": bool(lsst_only),
        "lsst_filter": query.lsst_identifier_filter() if lsst_only else None,
        "parallel_shards": int(parallel_shards or 1),
        "lsst_dia_count": 0,
        "lsst_ss_count": 0,
        "ztf_object_id_count": 0,
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "runtime_seconds": None,
        "validation": {},
        "paths": {
            "loci": str(paths["loci"]),
            "alerts": str(paths["alerts"]),
            "manifest": str(paths["manifest"]),
        },
    }

    try:
        df_raw, report_df = chunked_query.query_range_adaptive(
            label=label,
            mjd_min=mjd_min,
            mjd_max=mjd_max,
            tag=query_tag,
            target_loci=target_loci,
            initial_chunk_days=initial_chunk_days,
            min_chunk_seconds=min_chunk_seconds,
            max_results_per_chunk=max_results_per_chunk,
            split_threshold=split_threshold,
            chunk_cache_dir=chunk_cache_dir,
            use_chunk_cache=True,
            verbose=verbose,
            lsst_only=lsst_only,
            parallel_shards=parallel_shards,
        )
        ingested_at = _now_utc()
        df_loci = prepare_loci(df_raw, date_utc, mjd_min, mjd_max, ingested_at)
        _safe_to_parquet(df_loci, paths["loci"])

        if fetch_lightcurves and not df_loci.empty:
            df_raw_alerts = lightcurves.load_lightcurves(
                df_loci,
                len(df_loci),
                label,
                max_workers=max_lightcurve_workers,
            )
        else:
            if verbose and not fetch_lightcurves:
                print("  Lightcurve fetch disabled for this run.")
            df_raw_alerts = pd.DataFrame()

        df_alerts = prepare_alerts(df_raw_alerts, date_utc, label)
        _safe_to_parquet(df_alerts, paths["alerts"])

        counts = _report_counts(report_df)
        validation = validation_summary(
            df_loci,
            df_alerts,
            mjd_min=mjd_min,
            mjd_max=mjd_max,
            prior_locus_ids=prior_locus_ids,
            lsst_only=lsst_only,
        )
        manifest.update(counts)
        survey_counts = query.lsst_identifier_counts(df_loci)
        manifest.update({
            "actual_loci": int(len(df_loci)),
            "alert_rows": int(len(df_alerts)),
            "lsst_dia_count": survey_counts["lsst_dia_count"],
            "lsst_ss_count": survey_counts["lsst_ss_count"],
            "ztf_object_id_count": survey_counts["ztf_object_id_count"],
            "status": _status_from_counts(
                len(df_loci),
                target_loci,
                counts["saturated_chunk_count"],
            ),
            "validation": validation,
        })
    except Exception as exc:
        df_loci = pd.DataFrame()
        df_alerts = _empty_alerts_frame()
        report_df = pd.DataFrame()
        manifest["error"] = str(exc)
        if verbose:
            print(f"  [ERROR] Nightly ingest failed for {date_utc}: {exc}")
        raise
    finally:
        manifest["finished_at_utc"] = _now_utc()
        manifest["runtime_seconds"] = round(time.time() - t0, 2)
        _write_json(paths["manifest"], manifest)

    if update_indexes:
        if manifest["validation"].get("append_ready"):
            update_cumulative_indexes(data_root)
        elif verbose:
            print("  Validation did not pass append gate; cumulative indexes left unchanged.")

    return {
        "manifest": manifest,
        "df_loci": df_loci,
        "df_alerts": df_alerts,
        "report": report_df,
        "skipped": False,
    }


def _manifest_paths(data_root):
    """Yield all nightly manifest paths under a data root."""
    root = survey_data_root(data_root) / "nightly"
    if not root.exists():
        return []
    return sorted(root.glob("*/*/*/manifest.json"))


def _load_manifest_path(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_required_nightly_parquet(
    manifest_path, manifest, product, count_field
):
    """Read and count-check a product beside its discovered manifest."""
    product_path = Path(manifest_path).parent / f"{product}.parquet"
    if not product_path.is_file():
        raise FileNotFoundError(
            f"Missing sibling {product}.parquet for nightly manifest "
            f"{manifest_path}"
        )
    try:
        frame = pd.read_parquet(product_path)
    except Exception as exc:
        raise ValueError(
            f"Could not read sibling {product}.parquet for nightly manifest "
            f"{manifest_path}: {exc}"
        ) from exc

    expected_rows = manifest.get(count_field)
    if (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows < 0
    ):
        raise ValueError(
            f"Invalid {count_field} in nightly manifest {manifest_path}: "
            f"{expected_rows!r}."
        )
    if int(len(frame)) != expected_rows:
        raise ValueError(
            f"Manifest/Parquet {product} row mismatch for {manifest_path}: "
            f"manifest={expected_rows}, parquet={len(frame)}."
        )
    return frame


def update_cumulative_indexes(
    data_root=HISTORY_DATA_ROOT,
    require_append_ready=True,
    output_dir=None,
    manifest_overrides=None,
):
    """
    Rebuild cumulative loci and nightly summary parquet files from platform data.

    By default, manifests with `validation.append_ready == False` are not
    included in the cumulative loci index.
    """
    if output_dir is None:
        paths = cumulative_paths(data_root)
    else:
        output_root = Path(output_dir)
        paths = {
            "dir": output_root,
            "loci_index": output_root / "loci_index.parquet",
            "nightly_summary": output_root / "nightly_summary.parquet",
        }
    _ensure_storage_path(paths["dir"])

    overrides = dict(manifest_overrides or {})
    manifests = []
    loci_frames = []
    for manifest_path in _manifest_paths(data_root):
        manifest = _load_manifest_path(manifest_path)
        manifest_date = manifest.get("date_utc")
        if manifest_date in overrides:
            manifest = deepcopy(overrides.pop(manifest_date))
        # Preserve the cumulative summary's historical semantics: every
        # discovered manifest is represented, even when its science rows are
        # not eligible for the loci index.
        manifests.append(manifest)

        if manifest.get("status") not in APPENDABLE_STATUSES:
            continue
        append_ready = manifest.get("validation", {}).get("append_ready")
        if require_append_ready and append_ready is False:
            continue

        # Absolute paths embedded in copied manifests are provenance only.
        df_loci = _read_required_nightly_parquet(
            manifest_path, manifest, "loci", "actual_loci"
        )
        keep = [col for col in CUMULATIVE_INDEX_COLUMNS if col in df_loci.columns]
        if keep:
            loci_frames.append(df_loci[keep].copy())

    if overrides:
        raise ValueError(
            f"Manifest overrides did not match source dates: "
            f"{sorted(overrides)}"
        )

    if manifests:
        summary_df = pd.DataFrame(_manifest_to_summary_row(m) for m in manifests)
        summary_df = summary_df.sort_values(["mjd_min", "date_utc"]).reset_index(drop=True)
    else:
        summary_df = pd.DataFrame(columns=list(_manifest_to_summary_row({"date_utc": None}).keys()))

    if loci_frames:
        loci_index = pd.concat(loci_frames, ignore_index=True, sort=False)
        subset = [c for c in ["night_date_utc", LOCUS_ID_COL] if c in loci_index.columns]
        if subset:
            loci_index = loci_index.drop_duplicates(subset=subset, keep="last")
        if "night_mjd_min" in loci_index.columns:
            loci_index = loci_index.sort_values(["night_mjd_min", LOCUS_ID_COL]).reset_index(drop=True)
    else:
        loci_index = pd.DataFrame(columns=CUMULATIVE_INDEX_COLUMNS)

    _safe_to_parquet(summary_df, paths["nightly_summary"])
    _safe_to_parquet(loci_index, paths["loci_index"])
    return loci_index, summary_df


def load_cumulative_loci_index(data_root=HISTORY_DATA_ROOT, before_mjd=None, before_date=None):
    """Load the cumulative loci index, optionally excluding current/future nights."""
    path = cumulative_paths(data_root)["loci_index"]
    if not path.exists():
        return pd.DataFrame(columns=CUMULATIVE_INDEX_COLUMNS)

    df = pd.read_parquet(path)
    if before_mjd is not None and "night_mjd_max" in df.columns:
        df = df[df["night_mjd_max"] <= float(before_mjd)].copy()
    if before_date is not None and "night_date_utc" in df.columns:
        df = df[df["night_date_utc"] < before_date].copy()
    return df.reset_index(drop=True)


def load_cumulative_alerts(data_root=HISTORY_DATA_ROOT, before_mjd=None, before_date=None,
                           max_nights=None):
    """
    Load alert/lightcurve rows from prior nightly partitions.

    Empty alert parquet files represent valid zero-row nights. A missing,
    unreadable, or count-mismatched sibling for an append-ready manifest raises
    instead of silently omitting scientific rows. Use `max_nights` for a
    bounded plotting sample from a very large platform store.
    """
    manifests = []
    for manifest_path in _manifest_paths(data_root):
        manifest = _load_manifest_path(manifest_path)
        if manifest.get("status") not in APPENDABLE_STATUSES:
            continue
        if manifest.get("validation", {}).get("append_ready") is False:
            continue
        if before_mjd is not None and manifest.get("mjd_max") is not None:
            if float(manifest["mjd_max"]) > float(before_mjd):
                continue
        if before_date is not None and manifest.get("date_utc"):
            if manifest["date_utc"] >= before_date:
                continue
        manifests.append((manifest_path, manifest))

    manifests = sorted(
        manifests, key=lambda item: item[1].get("mjd_min", 0.0)
    )
    if max_nights is not None:
        manifests = manifests[-int(max_nights):]

    frames = []
    for manifest_path, manifest in manifests:
        # Resolve from the discovered manifest, never from declared provenance.
        df_alerts = _read_required_nightly_parquet(
            manifest_path, manifest, "alerts", "alert_rows"
        )
        if not df_alerts.empty:
            frames.append(df_alerts)

    if not frames:
        return _empty_alerts_frame()
    return pd.concat(frames, ignore_index=True, sort=False)


def compare_night_to_cumulative(df_night, df_cumulative):
    """Return compact comparison statistics for newest night vs prior history."""
    result = {
        "night_loci": int(len(df_night)),
        "cumulative_loci_rows": int(len(df_cumulative)),
        "cumulative_unique_loci": 0,
        "overlap_loci": 0,
        "overlap_fraction_of_night": 0.0,
        "new_loci": int(len(df_night)),
    }
    if df_night.empty or LOCUS_ID_COL not in df_night.columns:
        return result
    if df_cumulative.empty or LOCUS_ID_COL not in df_cumulative.columns:
        return result

    night_ids = set(df_night[LOCUS_ID_COL].dropna().astype(str))
    cumulative_ids = set(df_cumulative[LOCUS_ID_COL].dropna().astype(str))
    overlap = night_ids & cumulative_ids
    result["cumulative_unique_loci"] = int(len(cumulative_ids))
    result["overlap_loci"] = int(len(overlap))
    result["new_loci"] = int(len(night_ids - cumulative_ids))
    result["overlap_fraction_of_night"] = (
        len(overlap) / len(night_ids) if night_ids else 0.0
    )
    return result


def backfill_history(
    data_root=HISTORY_DATA_ROOT,
    mjd_start=None,
    mjd_stop=None,
    target_loci=HISTORY_TARGET_LOCI,
    query_tag=QUERY_TAG,
    max_nights=None,
    fetch_lightcurves=HISTORY_FETCH_ALL_LIGHTCURVES,
    resume=HISTORY_RESUME_EXISTING_NIGHTS,
    chunk_cache_dir=None,
    parallel_shards=CHUNK_PARALLEL_SHARDS,
    lsst_only=LSST_ONLY,
    verbose=True,
):
    """
    Build nightly partitions from `mjd_start` to `mjd_stop`.

    Returns the refreshed nightly summary table. Use `max_nights` for smoke
    tests before launching the full historical backfill.
    """
    if mjd_start is None or mjd_stop is None:
        raise ValueError("mjd_start and mjd_stop are required")

    completed = 0
    for date_utc, lo, hi in iter_night_windows(mjd_start, mjd_stop):
        if max_nights is not None and completed >= int(max_nights):
            break
        if verbose:
            print("=" * 72)
            print(f"Backfill {display_date(date_utc)}  MJD [{lo:.6f}, {hi:.6f}]")
        ingest_night(
            data_root=data_root,
            mjd_min=lo,
            mjd_max=hi,
            target_loci=target_loci,
            query_tag=query_tag,
            fetch_lightcurves=fetch_lightcurves,
            resume=resume,
            range_label=f"History {display_date(date_utc)}",
            chunk_cache_dir=chunk_cache_dir,
            parallel_shards=parallel_shards,
            lsst_only=lsst_only,
            update_indexes=True,
            verbose=verbose,
        )
        completed += 1

    _, summary_df = update_cumulative_indexes(data_root)
    return summary_df


def run_nightly_update(
    data_root=HISTORY_DATA_ROOT,
    mjd_min=None,
    mjd_max=None,
    target_loci=HISTORY_TARGET_LOCI,
    query_tag=QUERY_TAG,
    fetch_lightcurves=HISTORY_FETCH_ALL_LIGHTCURVES,
    resume=HISTORY_RESUME_EXISTING_NIGHTS,
    chunk_cache_dir=None,
    parallel_shards=CHUNK_PARALLEL_SHARDS,
    lsst_only=LSST_ONLY,
    verbose=True,
):
    """
    Ingest the newest night, compare it to prior cumulative history, then append.

    The comparison is computed against cumulative rows strictly before the new
    night, so the current night never compares against itself.
    """
    if mjd_min is None or mjd_max is None:
        raise ValueError("mjd_min and mjd_max are required")

    date_utc = mjd_to_utc_date(mjd_min)
    prior = load_cumulative_loci_index(data_root, before_mjd=mjd_min, before_date=date_utc)
    prior_ids = prior[LOCUS_ID_COL].dropna().astype(str).tolist() if LOCUS_ID_COL in prior.columns else []

    result = ingest_night(
        data_root=data_root,
        mjd_min=mjd_min,
        mjd_max=mjd_max,
        target_loci=target_loci,
        query_tag=query_tag,
        fetch_lightcurves=fetch_lightcurves,
        resume=resume,
        range_label=f"Last Night {display_date(date_utc)}",
        chunk_cache_dir=chunk_cache_dir,
        parallel_shards=parallel_shards,
        lsst_only=lsst_only,
        prior_locus_ids=prior_ids,
        update_indexes=False,
        verbose=verbose,
    )
    comparison = compare_night_to_cumulative(result["df_loci"], prior)
    result["comparison"] = comparison

    if result["manifest"].get("validation", {}).get("append_ready"):
        update_cumulative_indexes(data_root)
    elif verbose:
        print("  Validation did not pass append gate; newest night was not appended.")

    return result
