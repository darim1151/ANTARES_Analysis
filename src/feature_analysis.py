"""Feature-space diagnostics for ANTARES nightly locus snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import history


SCHEMA_VERSION = 1
LOCUS_ID = "locus_id"
MJD_COL = "newest_alert_observation_time"
TAG_COL = "tags"
CHI2_R = "feature_chi2_magn_r"
STD_R = "feature_standard_deviation_magn_r"
WEIGHTED_MEAN_COLUMNS = {
    band: f"feature_weighted_mean_magn_{band}" for band in "ugrizy"
}
FEATURE_COLUMNS = [CHI2_R, STD_R, *WEIGHTED_MEAN_COLUMNS.values()]
PROVENANCE_COLUMNS = [
    LOCUS_ID,
    MJD_COL,
    "night_date_utc",
    "night_mjd_min",
    "night_mjd_max",
    TAG_COL,
]
REQUESTED_COLUMNS = PROVENANCE_COLUMNS + FEATURE_COLUMNS
OPERATIONAL_TAGS = {"lc_feature_extractor", "high_snr", "in_LSSTDDF"}

PLANE_SPECS = {
    "variability_r": {
        "x": CHI2_R,
        "y": STD_R,
        "x_label": r"$\chi^2_r$",
        "y_label": r"$\sigma_r$ (mag)",
        "positive": True,
        "invert_y": False,
    },
    "r_vs_g_minus_i": {
        "x": "color_g_minus_i",
        "y": WEIGHTED_MEAN_COLUMNS["r"],
        "x_label": r"$g-i$ (mag)",
        "y_label": r"weighted mean $r$ (mag)",
        "positive": False,
        "invert_y": True,
    },
    "g_minus_r_vs_u_minus_g": {
        "x": "color_u_minus_g",
        "y": "color_g_minus_r",
        "x_label": r"$u-g$ (mag)",
        "y_label": r"$g-r$ (mag)",
        "positive": False,
        "invert_y": False,
    },
    "r_minus_i_vs_g_minus_r": {
        "x": "color_g_minus_r",
        "y": "color_r_minus_i",
        "x_label": r"$g-r$ (mag)",
        "y_label": r"$r-i$ (mag)",
        "positive": False,
        "invert_y": False,
    },
}


def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _analysis_root(data_root):
    return history.survey_data_root(data_root) / "analysis"


def _snapshot_paths(data_root):
    root = _analysis_root(data_root)
    return {
        "root": root,
        "parquet": root / "locus_feature_snapshots.parquet",
        "manifest": root / "locus_feature_snapshots_manifest.json",
    }


def _manifest_files(data_root):
    root = history.survey_data_root(data_root) / "nightly"
    if not root.exists():
        return []
    return sorted(root.glob("*/*/*/manifest.json"))


def _source_inventory(data_root):
    inventory = []
    for manifest_path in _manifest_files(data_root):
        manifest = _read_manifest(manifest_path)
        if manifest.get("status") not in history.APPENDABLE_STATUSES:
            continue
        if manifest.get("validation", {}).get("append_ready") is False:
            continue
        loci_path = Path(manifest.get("paths", {}).get("loci", ""))
        if not loci_path.exists():
            continue
        stat = loci_path.stat()
        inventory.append(
            {
                "date_utc": manifest.get("date_utc"),
                "mjd_min": manifest.get("mjd_min"),
                "mjd_max": manifest.get("mjd_max"),
                "path": str(loci_path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return sorted(inventory, key=lambda row: (row.get("mjd_min") or 0, row["path"]))


def _inventory_hash(inventory):
    payload = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _coverage_rows(source, schema_names, row_count, frame=None):
    present = set(schema_names)
    rows = []
    for feature in FEATURE_COLUMNS:
        finite_count = 0
        if frame is not None and feature in frame.columns:
            finite_count = int(
                np.isfinite(pd.to_numeric(frame[feature], errors="coerce")).sum()
            )
        rows.append({
            "date_utc": source.get("date_utc"),
            "mjd_min": source.get("mjd_min"),
            "mjd_max": source.get("mjd_max"),
            "source_path": source["path"],
            "source_rows": int(row_count),
            "feature": feature,
            "column_present": feature in present,
            "finite_count": finite_count,
            "finite_fraction": finite_count / row_count if row_count else 0.0,
        })
    return rows


def _normalize_snapshot_frame(df, source):
    frame = df.copy()
    for col in REQUESTED_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    if source.get("date_utc") is not None:
        frame["night_date_utc"] = frame["night_date_utc"].fillna(
            source.get("date_utc")
        )
    frame["night_mjd_min"] = pd.to_numeric(
        frame["night_mjd_min"], errors="coerce"
    ).fillna(source.get("mjd_min"))
    frame["night_mjd_max"] = pd.to_numeric(
        frame["night_mjd_max"], errors="coerce"
    ).fillna(source.get("mjd_max"))
    return frame[REQUESTED_COLUMNS]


def build_or_load_feature_snapshots(data_root, force=False):
    """Build or load the compact locus-feature snapshot table.

    Returns ``(snapshots, coverage, manifest)``. The builder reads only
    requested columns that are actually present in each nightly parquet.
    """
    paths = _snapshot_paths(data_root)
    inventory = _source_inventory(data_root)
    inventory_hash = _inventory_hash(inventory)
    cached_manifest = _read_manifest(paths["manifest"])
    cache_current = (
        not force
        and paths["parquet"].exists()
        and cached_manifest.get("schema_version") == SCHEMA_VERSION
        and cached_manifest.get("source_inventory_hash") == inventory_hash
    )

    if cache_current:
        snapshots = pd.read_parquet(paths["parquet"])
        coverage = pd.DataFrame(cached_manifest.get("coverage", []))
        return snapshots, coverage, cached_manifest

    frames = []
    coverage_rows = []
    for source in inventory:
        parquet_file = pq.ParquetFile(source["path"])
        schema_names = parquet_file.schema_arrow.names
        columns = [col for col in REQUESTED_COLUMNS if col in schema_names]
        if LOCUS_ID not in columns:
            coverage_rows.extend(
                _coverage_rows(source, schema_names, parquet_file.metadata.num_rows)
            )
            continue
        frame = pd.read_parquet(source["path"], columns=columns)
        coverage_rows.extend(
            _coverage_rows(
                source, schema_names, parquet_file.metadata.num_rows, frame=frame
            )
        )
        frames.append(_normalize_snapshot_frame(frame, source))

    snapshots = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame(columns=REQUESTED_COLUMNS)
    )
    if not snapshots.empty:
        snapshots = snapshots.drop_duplicates(
            subset=["night_date_utc", LOCUS_ID], keep="last"
        )
        snapshots = snapshots.sort_values(
            ["night_mjd_min", LOCUS_ID], na_position="last"
        ).reset_index(drop=True)

    coverage = pd.DataFrame(coverage_rows)
    non_null_counts = {
        col: int(pd.to_numeric(snapshots[col], errors="coerce").notna().sum())
        for col in FEATURE_COLUMNS
        if col in snapshots.columns
    }
    feature_fractions = {
        col: (count / len(snapshots) if len(snapshots) else 0.0)
        for col, count in non_null_counts.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "built_at_utc": _now_utc(),
        "source_inventory_hash": inventory_hash,
        "source_inventory": inventory,
        "requested_columns": REQUESTED_COLUMNS,
        "snapshot_rows": int(len(snapshots)),
        "feature_non_null_counts": non_null_counts,
        "feature_non_null_fractions": feature_fractions,
        "coverage": coverage.to_dict(orient="records"),
    }

    paths["root"].mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(paths["parquet"], index=False)
    _write_json(paths["manifest"], manifest)
    return snapshots, coverage, manifest


def add_color_columns(df):
    """Return a copy with requested colors computed from weighted means."""
    frame = df.copy()
    g = pd.to_numeric(frame.get(WEIGHTED_MEAN_COLUMNS["g"]), errors="coerce")
    r = pd.to_numeric(frame.get(WEIGHTED_MEAN_COLUMNS["r"]), errors="coerce")
    i = pd.to_numeric(frame.get(WEIGHTED_MEAN_COLUMNS["i"]), errors="coerce")
    u = pd.to_numeric(frame.get(WEIGHTED_MEAN_COLUMNS["u"]), errors="coerce")
    frame["color_g_minus_i"] = g - i
    frame["color_u_minus_g"] = u - g
    frame["color_g_minus_r"] = g - r
    frame["color_r_minus_i"] = r - i
    return frame


def select_comparison_cohorts(snapshots, current_loci, current_mjd):
    """Select unique historical loci and current-night feature snapshots."""
    current_mjd = float(current_mjd)
    frame = snapshots.copy()
    frame["night_mjd_min"] = pd.to_numeric(frame["night_mjd_min"], errors="coerce")
    historical = frame[frame["night_mjd_min"] < current_mjd].copy()
    historical = historical.sort_values(
        ["night_mjd_min", MJD_COL], na_position="first"
    ).drop_duplicates(subset=[LOCUS_ID], keep="last")

    current_ids = set(
        current_loci.get(LOCUS_ID, pd.Series(dtype="object")).dropna().astype(str)
    )
    current = frame[
        np.isclose(frame["night_mjd_min"], current_mjd, equal_nan=False)
        & frame[LOCUS_ID].astype(str).isin(current_ids)
    ].copy()
    current = current.sort_values(MJD_COL, na_position="first").drop_duplicates(
        subset=[LOCUS_ID], keep="last"
    )

    if current.empty and current_ids:
        available = [col for col in REQUESTED_COLUMNS if col in current_loci.columns]
        current = _normalize_snapshot_frame(
            current_loci[available],
            {
                "date_utc": None,
                "mjd_min": current_mjd,
                "mjd_max": current_mjd + 1,
            },
        )
        current = current[current[LOCUS_ID].astype(str).isin(current_ids)]

    return add_color_columns(historical.reset_index(drop=True)), add_color_columns(
        current.reset_index(drop=True)
    )


def _split_tags(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        text = str(value).replace(";", ",")
        items = text.split(",")
    return {str(item).strip() for item in items if str(item).strip()}


def _tag_mask(df, tag):
    if TAG_COL not in df.columns:
        return pd.Series(False, index=df.index)
    return df[TAG_COL].map(lambda value: tag in _split_tags(value))


def rank_tag_subsets(
    historical,
    current,
    min_historical=500,
    min_current=30,
    max_tags=6,
    excluded_tags=OPERATIONAL_TAGS,
):
    """Rank multi-label ANTARES tags and identify analyzable subsets."""
    all_tags = set()
    for value in current.get(TAG_COL, pd.Series(dtype="object")):
        all_tags.update(_split_tags(value))
    all_tags -= set(excluded_tags)

    rows = []
    for tag in sorted(all_tags):
        h_tag = historical[_tag_mask(historical, tag)]
        c_tag = current[_tag_mask(current, tag)]
        row = {
            "tag": tag,
            "historical_loci": int(len(h_tag)),
            "current_loci": int(len(c_tag)),
        }
        eligible_planes = []
        for plane, spec in PLANE_SPECS.items():
            h_valid = len(_finite_pair(h_tag, spec["x"], spec["y"], spec["positive"]))
            c_valid = len(_finite_pair(c_tag, spec["x"], spec["y"], spec["positive"]))
            row[f"{plane}_historical_valid"] = h_valid
            row[f"{plane}_current_valid"] = c_valid
            if h_valid >= min_historical and c_valid >= min_current:
                eligible_planes.append(plane)
        row["eligible_planes"] = ",".join(eligible_planes)
        row["eligible"] = bool(eligible_planes)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return [], table
    table = table.sort_values(
        ["eligible", "current_loci", "historical_loci", "tag"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = table.loc[table["eligible"], "tag"].head(max_tags).tolist()
    table["selected"] = table["tag"].isin(selected)
    return selected, table


def feature_plane_coverage(historical, current):
    """Return pairwise-complete counts for every configured feature plane."""
    rows = []
    for plane, spec in PLANE_SPECS.items():
        h_pair = _finite_pair(historical, spec["x"], spec["y"], spec["positive"])
        c_pair = _finite_pair(current, spec["x"], spec["y"], spec["positive"])
        rows.append(
            {
                "plane": plane,
                "historical_unique_loci": int(len(historical)),
                "historical_pairwise_complete": int(len(h_pair)),
                "current_loci": int(len(current)),
                "current_pairwise_complete": int(len(c_pair)),
                "available_in_both_cohorts": bool(len(h_pair) and len(c_pair)),
            }
        )
    return pd.DataFrame(rows)


def _finite_pair(df, x_col, y_col, positive=False):
    x = pd.to_numeric(df.get(x_col), errors="coerce")
    y = pd.to_numeric(df.get(y_col), errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if positive:
        mask &= (x > 0) & (y > 0)
    return pd.DataFrame({"x": x[mask], "y": y[mask]}).reset_index(drop=True)


def _median_mad(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    median = float(np.median(values))
    return median, float(np.median(np.abs(values - median)))


def _ks_2samp(x, y):
    """Two-sided two-sample KS statistic with asymptotic p-value."""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if not len(x) or not len(y):
        return np.nan, np.nan
    values = np.concatenate([x, y])
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    statistic = float(np.max(np.abs(cdf_x - cdf_y)))
    if statistic == 0:
        return 0.0, 1.0
    effective_n = len(x) * len(y) / (len(x) + len(y))
    if effective_n <= 0:
        return statistic, np.nan
    root_n = math.sqrt(effective_n)
    lam = (root_n + 0.12 + 0.11 / root_n) * statistic
    terms = [(-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101)]
    return statistic, float(min(1.0, max(0.0, 2 * sum(terms))))


def _spearman(x, y):
    if len(x) < 2:
        return np.nan
    ranked_x = pd.Series(x).rank()
    ranked_y = pd.Series(y).rank()
    if ranked_x.nunique() < 2 or ranked_y.nunique() < 2:
        return np.nan
    return float(ranked_x.corr(ranked_y))


def _histogram_limits(x, y):
    x_limits = np.quantile(x, [0.005, 0.995])
    y_limits = np.quantile(y, [0.005, 0.995])
    if x_limits[0] == x_limits[1]:
        x_limits += np.array([-0.5, 0.5])
    if y_limits[0] == y_limits[1]:
        y_limits += np.array([-0.5, 0.5])
    return x_limits, y_limits


def _js_divergence_from_counts(hist_a, hist_b):
    hist_a = hist_a + 1e-12
    hist_b = hist_b + 1e-12
    p = hist_a / hist_a.sum()
    q = hist_b / hist_b.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))


def _point_bin_ids(points, x_edges, y_edges):
    x_bins = np.searchsorted(x_edges, points[:, 0], side="right") - 1
    y_bins = np.searchsorted(y_edges, points[:, 1], side="right") - 1
    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    x_bins[points[:, 0] == x_edges[-1]] = n_x - 1
    y_bins[points[:, 1] == y_edges[-1]] = n_y - 1
    valid = (
        (x_bins >= 0)
        & (x_bins < n_x)
        & (y_bins >= 0)
        & (y_bins < n_y)
    )
    return x_bins[valid] * n_y + y_bins[valid]


def _permutation_js(
    historical,
    current,
    seed,
    permutations=500,
    sample_cap=10000,
    bins=40,
):
    if min(len(historical), len(current)) < 2:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    n = min(len(historical), len(current), int(sample_cap))
    h_idx = rng.choice(len(historical), n, replace=False)
    c_idx = rng.choice(len(current), n, replace=False)
    h = historical.iloc[h_idx][["x", "y"]].to_numpy(dtype=float)
    c = current.iloc[c_idx][["x", "y"]].to_numpy(dtype=float)
    x_limits, y_limits = _histogram_limits(h[:, 0], h[:, 1])
    x_edges = np.linspace(*x_limits, bins + 1)
    y_edges = np.linspace(*y_limits, bins + 1)
    h_ids = _point_bin_ids(h, x_edges, y_edges)
    c_ids = _point_bin_ids(c, x_edges, y_edges)
    n_cells = bins * bins
    observed = _js_divergence_from_counts(
        np.bincount(h_ids, minlength=n_cells),
        np.bincount(c_ids, minlength=n_cells),
    )
    combined_ids = np.concatenate([h_ids, c_ids])
    split = len(h_ids)
    exceedances = 0
    for _ in range(int(permutations)):
        order = rng.permutation(len(combined_ids))
        permuted = _js_divergence_from_counts(
            np.bincount(combined_ids[order[:split]], minlength=n_cells),
            np.bincount(combined_ids[order[split:]], minlength=n_cells),
        )
        exceedances += permuted >= observed
    p_value = (exceedances + 1) / (int(permutations) + 1)
    return observed, float(p_value), n


def _bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted
    valid_idx = np.flatnonzero(valid)
    order = valid_idx[np.argsort(p_values[valid])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _plane_statistics(historical, current, plane, group, seed, permutations, sample_cap):
    spec = PLANE_SPECS[plane]
    h_pair = _finite_pair(historical, spec["x"], spec["y"], spec["positive"])
    c_pair = _finite_pair(current, spec["x"], spec["y"], spec["positive"])
    row = {
        "group": group,
        "plane": plane,
        "historical_total": int(len(historical)),
        "current_total": int(len(current)),
        "historical_complete": int(len(h_pair)),
        "current_complete": int(len(c_pair)),
        "historical_coverage": len(h_pair) / len(historical) if len(historical) else 0.0,
        "current_coverage": len(c_pair) / len(current) if len(current) else 0.0,
    }
    for axis in ["x", "y"]:
        h_values = h_pair[axis].to_numpy()
        c_values = c_pair[axis].to_numpy()
        h_median, h_mad = _median_mad(h_values)
        c_median, c_mad = _median_mad(c_values)
        ks_stat, ks_p = _ks_2samp(h_values, c_values)
        row.update(
            {
                f"{axis}_historical_median": h_median,
                f"{axis}_historical_mad": h_mad,
                f"{axis}_historical_q05": np.quantile(h_values, 0.05) if len(h_values) else np.nan,
                f"{axis}_historical_q95": np.quantile(h_values, 0.95) if len(h_values) else np.nan,
                f"{axis}_current_median": c_median,
                f"{axis}_current_mad": c_mad,
                f"{axis}_current_q05": np.quantile(c_values, 0.05) if len(c_values) else np.nan,
                f"{axis}_current_q95": np.quantile(c_values, 0.95) if len(c_values) else np.nan,
                f"{axis}_robust_effect": (
                    (c_median - h_median) / (1.4826 * h_mad)
                    if np.isfinite(h_mad) and h_mad > 0
                    else np.nan
                ),
                f"{axis}_ks_statistic": ks_stat,
                f"{axis}_ks_p_value": ks_p,
            }
        )
    row["historical_spearman"] = _spearman(h_pair["x"], h_pair["y"])
    row["current_spearman"] = _spearman(c_pair["x"], c_pair["y"])
    limits_source = h_pair if not h_pair.empty else c_pair
    if limits_source.empty:
        x_limits, y_limits = (np.nan, np.nan), (np.nan, np.nan)
        row["historical_clipped_count"] = 0
        row["current_clipped_count"] = 0
    else:
        x_limits, y_limits = _histogram_limits(
            limits_source["x"], limits_source["y"]
        )
        h_inside = h_pair["x"].between(*x_limits) & h_pair["y"].between(
            *y_limits
        )
        c_inside = c_pair["x"].between(*x_limits) & c_pair["y"].between(
            *y_limits
        )
        row["historical_clipped_count"] = int((~h_inside).sum())
        row["current_clipped_count"] = int((~c_inside).sum())
    row["plot_x_min"] = x_limits[0]
    row["plot_x_max"] = x_limits[1]
    row["plot_y_min"] = y_limits[0]
    row["plot_y_max"] = y_limits[1]

    h_for_js = h_pair
    c_for_js = c_pair
    if spec["positive"]:
        h_for_js = np.log10(h_pair[["x", "y"]])
        c_for_js = np.log10(c_pair[["x", "y"]])
    js, js_p, sample_n = _permutation_js(
        h_for_js,
        c_for_js,
        seed=seed,
        permutations=permutations,
        sample_cap=sample_cap,
    )
    row["js_divergence"] = js
    row["js_permutation_p_value"] = js_p
    row["js_sample_per_cohort"] = sample_n
    return row


def _cohort_summary(historical, current):
    rows = []
    for name, frame in [("historical", historical), ("current", current)]:
        row = {"cohort": name, "loci": int(len(frame))}
        for col in FEATURE_COLUMNS:
            values = pd.to_numeric(frame.get(col), errors="coerce")
            count = int(np.isfinite(values).sum()) if values is not None else 0
            row[f"{col}_finite"] = count
            row[f"{col}_coverage"] = count / len(frame) if len(frame) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_feature_diagnostics(
    historical,
    current,
    seed=42,
    permutations=500,
    sample_cap=10000,
    min_tag_historical=500,
    min_tag_current=30,
    max_tags=6,
):
    """Compute all-population and qualifying multi-label tag diagnostics."""
    selected_tags, tag_counts = rank_tag_subsets(
        historical,
        current,
        min_historical=min_tag_historical,
        min_current=min_tag_current,
        max_tags=max_tags,
    )
    all_rows = [
        _plane_statistics(
            historical,
            current,
            plane,
            "all",
            seed + index,
            permutations,
            sample_cap,
        )
        for index, plane in enumerate(PLANE_SPECS)
    ]
    tag_rows = []
    for tag_index, tag in enumerate(selected_tags):
        h_tag = historical[_tag_mask(historical, tag)]
        c_tag = current[_tag_mask(current, tag)]
        for plane_index, plane in enumerate(PLANE_SPECS):
            tag_rows.append(
                _plane_statistics(
                    h_tag,
                    c_tag,
                    plane,
                    tag,
                    seed + 1000 + tag_index * 20 + plane_index,
                    permutations,
                    sample_cap,
                )
            )

    feature_statistics = pd.DataFrame(all_rows)
    tag_statistics = pd.DataFrame(tag_rows)
    combined = pd.concat(
        [
            feature_statistics.assign(_table="feature"),
            tag_statistics.assign(_table="tag"),
        ],
        ignore_index=True,
    )
    if not combined.empty:
        for p_col in ["x_ks_p_value", "y_ks_p_value", "js_permutation_p_value"]:
            combined[f"{p_col}_bh"] = _bh_adjust(combined[p_col])
        feature_statistics = combined[combined["_table"] == "feature"].drop(
            columns="_table"
        )
        tag_statistics = combined[combined["_table"] == "tag"].drop(columns="_table")

    return {
        "historical": historical,
        "current": current,
        "cohort_summary": _cohort_summary(historical, current),
        "feature_statistics": feature_statistics.reset_index(drop=True),
        "tag_statistics": tag_statistics.reset_index(drop=True),
        "tag_counts": tag_counts,
        "selected_tags": selected_tags,
        "settings": {
            "seed": seed,
            "permutations": permutations,
            "sample_cap": sample_cap,
            "ks_method": "two-sided asymptotic two-sample KS",
            "js_histogram_bins_per_axis": 40,
            "js_variability_transform": "log10",
            "multiple_testing": "Benjamini-Hochberg across all all-population and tag tests",
            "min_tag_historical": min_tag_historical,
            "min_tag_current": min_tag_current,
            "max_tags": max_tags,
            "operational_tags_excluded": sorted(OPERATIONAL_TAGS),
        },
    }


def _plot_density(ax, pair, limits, title, spec, historical):
    if pair.empty:
        ax.text(0.5, 0.5, "Insufficient pairwise data", ha="center", va="center")
        ax.set_title(title)
        return
    x_limits, y_limits = limits
    clipped = pair[
        pair["x"].between(*x_limits) & pair["y"].between(*y_limits)
    ]
    if historical or len(clipped) > 10000:
        kwargs = {}
        if spec["positive"]:
            kwargs.update(
                {
                    "xscale": "log",
                    "yscale": "log",
                    "extent": (
                        math.log10(x_limits[0]),
                        math.log10(x_limits[1]),
                        math.log10(y_limits[0]),
                        math.log10(y_limits[1]),
                    ),
                }
            )
        else:
            kwargs["extent"] = (*x_limits, *y_limits)
        ax.hexbin(
            clipped["x"],
            clipped["y"],
            gridsize=70,
            mincnt=1,
            norm=mcolors.LogNorm(),
            cmap="viridis",
            **kwargs,
        )
    else:
        ax.scatter(clipped["x"], clipped["y"], s=8, alpha=0.35, edgecolors="none")
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    if spec["positive"] and not (historical or len(clipped) > 10000):
        ax.set_xscale("log")
        ax.set_yscale("log")
    if spec["invert_y"]:
        ax.invert_yaxis()
    ax.set_title(f"{title} (n={len(pair):,})")
    ax.set_xlabel(spec["x_label"])
    ax.set_ylabel(spec["y_label"])
    ax.grid(alpha=0.15)


def _shared_limits(h_pair, c_pair, positive=False):
    source = h_pair if not h_pair.empty else c_pair
    if source.empty:
        return (0.0, 1.0), (0.0, 1.0)
    x_limits, y_limits = _histogram_limits(source["x"], source["y"])
    if positive:
        min_x = float(source.loc[source["x"] > 0, "x"].min())
        min_y = float(source.loc[source["y"] > 0, "y"].min())
        x_limits[0] = max(x_limits[0], min_x * 0.8)
        y_limits[0] = max(y_limits[0], min_y * 0.8)
    return x_limits, y_limits


def plot_variability_plane(historical, current, title_suffix=""):
    spec = PLANE_SPECS["variability_r"]
    h_pair = _finite_pair(historical, spec["x"], spec["y"], True)
    c_pair = _finite_pair(current, spec["x"], spec["y"], True)
    limits = _shared_limits(h_pair, c_pair, positive=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
    _plot_density(axes[0], h_pair, limits, "Historical unique loci", spec, True)
    _plot_density(axes[1], c_pair, limits, "Current-night loci", spec, False)
    fig.suptitle(f"ANTARES r-band variability feature space{title_suffix}")
    fig.tight_layout()
    return fig


def plot_color_diagnostics(historical, current, title_suffix=""):
    planes = [
        "r_vs_g_minus_i",
        "g_minus_r_vs_u_minus_g",
        "r_minus_i_vs_g_minus_r",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for row_index, plane in enumerate(planes):
        spec = PLANE_SPECS[plane]
        h_pair = _finite_pair(historical, spec["x"], spec["y"], False)
        c_pair = _finite_pair(current, spec["x"], spec["y"], False)
        limits = _shared_limits(h_pair, c_pair)
        _plot_density(
            axes[row_index, 0], h_pair, limits, "Historical unique loci", spec, True
        )
        _plot_density(
            axes[row_index, 1], c_pair, limits, "Current-night loci", spec, False
        )
    fig.suptitle(f"ANTARES weighted-mean color diagnostics{title_suffix}", y=1.0)
    fig.tight_layout()
    return fig


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def save_feature_coverage_audit(
    coverage,
    cohort_summary,
    pairwise_coverage,
    output_dir,
    snapshot_manifest=None,
    analysis_context=None,
):
    """Persist a coverage-only result when no requested plane is usable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    cohort_summary.to_csv(output_dir / "cohort_summary.csv", index=False)
    pairwise_coverage.to_csv(output_dir / "feature_statistics.csv", index=False)
    pd.DataFrame(columns=["group", "plane"]).to_csv(
        output_dir / "tag_statistics.csv", index=False
    )
    metadata = {
        "created_at_utc": _now_utc(),
        "git_commit_sha": _git_sha(),
        "status": "coverage_only_no_usable_feature_plane",
        "output_directory": str(output_dir),
        "analysis_context": analysis_context or {},
        "feature_columns": FEATURE_COLUMNS,
        "snapshot_manifest": snapshot_manifest or {},
        "scientific_caveat": (
            "ANTARES locus features can summarize accumulated multi-survey history; "
            "they are not measurements restricted to the selected night."
        ),
    }
    _write_json(output_dir / "analysis_metadata.json", metadata)
    return metadata


def save_feature_products(
    results,
    output_dir,
    coverage=None,
    snapshot_manifest=None,
    analysis_context=None,
):
    """Save tables, metadata, and main/tag figure sets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = coverage if coverage is not None else pd.DataFrame()
    coverage.to_csv(output_dir / "feature_coverage.csv", index=False)
    results["cohort_summary"].to_csv(output_dir / "cohort_summary.csv", index=False)
    results["feature_statistics"].to_csv(
        output_dir / "feature_statistics.csv", index=False
    )
    tag_output = results["tag_statistics"].copy()
    if tag_output.empty:
        tag_output = results["tag_counts"].copy()
    elif not results["tag_counts"].empty:
        tag_output = tag_output.merge(
            results["tag_counts"], left_on="group", right_on="tag", how="outer"
        )
    tag_output.to_csv(output_dir / "tag_statistics.csv", index=False)

    figure_paths = []
    figures = [
        ("variability_plane.png", plot_variability_plane(results["historical"], results["current"])),
        ("color_diagnostics.png", plot_color_diagnostics(results["historical"], results["current"])),
    ]
    for tag in results["selected_tags"]:
        h_tag = results["historical"][_tag_mask(results["historical"], tag)]
        c_tag = results["current"][_tag_mask(results["current"], tag)]
        safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag)
        figures.extend(
            [
                (
                    f"tag_{safe_tag}_variability_plane.png",
                    plot_variability_plane(h_tag, c_tag, f" - {tag}"),
                ),
                (
                    f"tag_{safe_tag}_color_diagnostics.png",
                    plot_color_diagnostics(h_tag, c_tag, f" - {tag}"),
                ),
            ]
        )
    for filename, figure in figures:
        path = output_dir / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        figure_paths.append(str(path))
        plt.close(figure)

    metadata = {
        "created_at_utc": _now_utc(),
        "git_commit_sha": _git_sha(),
        "output_directory": str(output_dir),
        "analysis_context": analysis_context or {},
        "settings": results["settings"],
        "selected_tags": results["selected_tags"],
        "feature_columns": FEATURE_COLUMNS,
        "plane_specs": PLANE_SPECS,
        "snapshot_manifest": snapshot_manifest or {},
        "coverage_below_80_percent": {
            cohort: [
                feature
                for feature in FEATURE_COLUMNS
                if float(
                    results["cohort_summary"]
                    .set_index("cohort")
                    .loc[cohort, f"{feature}_coverage"]
                )
                < 0.8
            ]
            for cohort in ["historical", "current"]
        },
        "figures": figure_paths,
        "scientific_caveat": (
            "ANTARES locus features can summarize accumulated multi-survey history; "
            "they are not measurements restricted to the selected night."
        ),
    }
    _write_json(output_dir / "analysis_metadata.json", metadata)
    return metadata
