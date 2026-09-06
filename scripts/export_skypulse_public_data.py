"""Export static SkyPulse public data files.

Architecture:
    RSP parquet outputs -> this static exporter -> web/public/data/*.json
    -> SkyPulse frontend.

The public app must not query ANTARES, Rubin Butler, TAP, official Rubin
production systems, or RSP at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "web" / "public" / "data"
DEFAULT_SAMPLE = ROOT / "data" / "antares_raw_data.csv"
DEFAULT_MANIFEST = ROOT / "data" / "manifest_example.json"
SCHEMA_VERSION = 2
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)
SURVEY_SUBDIR = "lsst_only"
USABLE_STATUSES = {"complete", "under_target", "saturated_unresolved"}
PREFERRED_LATEST_STATUSES = {"complete", "under_target"}
RSP_LOCI_COLUMNS = [
    "locus_id",
    "ra",
    "dec",
    "tags",
    "all_tags",
    "newest_alert_observation_time",
    "brightest_alert_magnitude",
    "num_mag_values",
    "night_date_utc",
    "night_mjd_min",
    "night_mjd_max",
    "survey",
    "ztf_object_id",
    "dia_object_id",
    "ss_object_id",
]
ALERT_REQUIRED_COLUMNS = ["locus_id"]
ALERT_MAG_COLUMNS = ["ant_mag", "ztf_magpsf", "magpsf", "psf_mag", "magnitude", "mag"]
ALERT_MAGERR_COLUMNS = ["ant_magerr", "ztf_sigmapsf", "magerr", "magnitude_error", "mag_error"]
ALERT_FILTER_COLUMNS = ["ant_passband", "ztf_fid", "fid", "filter", "band"]
ALERT_TIME_COLUMNS = [
    "ant_mjd",
    "time",
    "mjd",
    "obs_mjd",
    "ztf_mjd",
    "alert_mjd",
    "observation_time",
    "jd",
    "ztf_jd",
]

PUBLIC_SOURCE_FILES = {
    "nightly_manifest": "data/lsst_only/nightly/YYYY/MM/DD/manifest.json",
    "nightly_loci": "data/lsst_only/nightly/YYYY/MM/DD/loci.parquet",
    "nightly_alerts": "data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet",
    "cumulative_loci_index": "data/lsst_only/cumulative/loci_index.parquet",
    "cumulative_nightly_summary": "data/lsst_only/cumulative/nightly_summary.parquet",
}


class ExportError(RuntimeError):
    """Raised for clear user-facing export failures."""


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mjd_to_utc_date(mjd: float) -> str:
    """Return the UTC calendar date containing a finite Modified Julian Date."""
    if not math.isfinite(mjd):
        raise ExportError(f"MJD must be finite, received {mjd!r}.")
    try:
        return (MJD_EPOCH + timedelta(days=mjd)).date().isoformat()
    except OverflowError as exc:
        raise ExportError(f"MJD {mjd!r} is outside the supported calendar range.") from exc


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    safe_payload = json_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def stable_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def parse_tags(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except TypeError:
        pass
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = str(value).replace(";", ",").split(",")
    return [str(tag).strip() for tag in raw if str(tag).strip()]


def number_or_none(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def int_or_none(value: Any) -> int | None:
    numeric = number_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def finite_or_default(value: Any, default: float) -> float:
    numeric = number_or_none(value)
    return default if numeric is None else float(numeric)


def parquet_columns(path: Path) -> list[str] | None:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema_arrow.names
    except Exception:
        return None


def read_parquet_selected(path: Path, wanted: Iterable[str] | None = None) -> pd.DataFrame:
    if wanted is None:
        return pd.read_parquet(path)

    available = parquet_columns(path)
    if available is None:
        return pd.read_parquet(path)

    columns = [column for column in wanted if column in available]
    if not columns:
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def source_caveats(mode: str, demo_synthetic_lightcurves: bool, saturated: bool) -> list[str]:
    caveats = [
        "SkyPulse shows LSST-associated ANTARES loci and public export samples; it is not a direct Rubin Butler, TAP, or official Rubin production-system query.",
        "ANTARES is the broker/source for these alert-analysis records; Rubin Science Platform was used for compute and storage.",
        "ANTARES can merge multi-survey histories into one locus, so some LSST-associated loci may also carry older survey identifiers.",
        "Brightness history comes from ANTARES alert records, with photometry columns such as ant_mag, ant_magerr, and ant_passband when available.",
        "Highlighted objects use a transparent exploration ranking, not a classification and not a claim of scientific importance.",
        "This is a static export of saved parquet outputs, not live data and not direct Rubin Butler/TAP access.",
    ]
    if mode == "demo":
        caveats.append(
            "Demo mode uses repository sample rows and synthetic brightness stories; use --data-root on RSP or copied parquet files for real alert-record lightcurves."
        )
    if demo_synthetic_lightcurves:
        caveats.append("Demo lightcurve samples are synthetic and must not be presented as ANTARES alert-record photometry.")
    if saturated:
        caveats.append("The selected nightly manifest reports saturated_unresolved; density and counts should be read with that warning visible.")
    return caveats


def common_payload(
    mode: str,
    generated_at: str,
    source_range: dict[str, Any],
    caveats: list[str],
    selected_night_date: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "export_mode": mode,
        "generated_at_utc": generated_at,
        "selected_night_date": selected_night_date,
        "source_data_range": source_range,
        "scientific_caveats": caveats,
    }


def source_files_payload(real_paths: dict[str, Path | None], include_source_paths: bool) -> dict[str, str | None]:
    if include_source_paths:
        return {key: str(path) if path is not None else None for key, path in real_paths.items()}
    return {key: PUBLIC_SOURCE_FILES.get(key, "path omitted") for key, path in real_paths.items() if path is not None}


def load_demo_frame(sample_csv: Path, max_points: int, seed: int) -> pd.DataFrame:
    if not sample_csv.exists():
        raise ExportError(f"Demo sample CSV is missing: {sample_csv}")
    frame = pd.read_csv(sample_csv)
    if "locus_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["locus_id"], keep="first")
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=seed)
    frame = frame.reset_index(drop=True)
    frame["tags"] = frame.get("all_tags", "").map(parse_tags)
    frame["obs_count"] = pd.to_numeric(frame.get("num_mag_values"), errors="coerce").fillna(1).clip(lower=1)
    frame["brightness_mag"] = (
        pd.to_numeric(frame.get("brightest_alert_magnitude"), errors="coerce")
        .fillna(20.5)
        .clip(lower=13.5, upper=25.5)
    )
    frame["ra"] = pd.to_numeric(frame["ra"], errors="coerce")
    frame["dec"] = pd.to_numeric(frame["dec"], errors="coerce")
    frame = frame[frame["ra"].between(0, 360, inclusive="left") & frame["dec"].between(-90, 90)]
    return frame.reset_index(drop=True)


def demo_reason(row: pd.Series, seen_before: bool, is_last_night: bool) -> str:
    tags = set(row["tags"])
    if is_last_night and not seen_before:
        return "new in the saved sky memory"
    if row["brightness_mag"] <= 17:
        return "bright public sample"
    if row["obs_count"] >= 150:
        return "many brightness measurements"
    if "young_extragalactic_candidate" in tags:
        return "young extragalactic candidate tag"
    return "representative sky object"


def make_demo_points(
    frame: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latest_min = float(manifest.get("mjd_min", 61102.0))
    latest_max = float(manifest.get("mjd_max", latest_min + 1.0))
    historical_min = max(61095.0, latest_min - 32.0)
    historical_max = latest_min
    latest_date = str(manifest.get("date_utc") or mjd_to_utc_date(latest_min))
    expected_latest_date = mjd_to_utc_date(latest_min)
    if latest_date != expected_latest_date:
        raise ExportError(
            f"Demo manifest date_utc {latest_date!r} does not match "
            f"mjd_min {latest_min}, whose UTC date is {expected_latest_date}."
        )

    points: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        locus_id = str(row["locus_id"])
        unit = stable_unit(locus_id)
        is_last_night = unit > 0.72
        seen_before = stable_unit(f"{locus_id}:seen") > 0.36
        if is_last_night:
            mjd = latest_min + stable_unit(f"{locus_id}:mjd") * (latest_max - latest_min)
            date_utc = latest_date
            group = "last_night"
        else:
            mjd = historical_min + stable_unit(f"{locus_id}:history") * (historical_max - historical_min)
            date_utc = mjd_to_utc_date(mjd)
            group = "historical"

        obs_count = int(row["obs_count"])
        brightness_mag = float(row["brightness_mag"])
        obs_score = min(1.0, math.log10(obs_count + 1) / 2.7)
        bright_score = max(0.0, min(1.0, (23.5 - brightness_mag) / 8.0))
        interest = (
            (30 if is_last_night else 0)
            + (18 if is_last_night and not seen_before else 0)
            + obs_score * 25
            + bright_score * 22
            + (5 if "young_extragalactic_candidate" in set(row["tags"]) else 0)
        )
        reason = demo_reason(row, seen_before, is_last_night)
        points.append(
            {
                "id": locus_id,
                "locus_id": locus_id,
                "label": locus_id,
                "group": group,
                "ra": round(float(row["ra"]), 5),
                "dec": round(float(row["dec"]), 5),
                "date_utc": date_utc,
                "mjd": round(float(mjd), 6),
                "newest_alert_observation_time": round(float(mjd), 6),
                "brightness_mag": round(brightness_mag, 3),
                "brightest_alert_magnitude": round(brightness_mag, 3),
                "obs_count": obs_count,
                "num_mag_values": obs_count,
                "tags": row["tags"][:6],
                "is_last_night": bool(is_last_night),
                "seen_before": bool(seen_before),
                "has_lightcurve": bool(is_last_night),
                "is_highlighted": False,
                "interest_score": round(float(interest), 3),
                "reason": reason,
                "public_description": f"{locus_id} is a demo sky object highlighted as {reason}.",
            }
        )

    source_range = {
        "latest_night_utc": latest_date,
        "latest_mjd_min": latest_min,
        "latest_mjd_max": latest_max,
        "historical_mjd_min": historical_min,
        "historical_mjd_max": historical_max,
    }
    return points, source_range


def score_point(point: dict[str, Any]) -> float:
    obs_score = min(1.0, math.log10(float(point.get("obs_count") or 1) + 1) / 2.7)
    bright_score = max(0.0, min(1.0, (23.5 - float(point.get("brightness_mag") or 20.5)) / 8.0))
    return (
        (30 if point.get("is_last_night") else 0)
        + (18 if point.get("is_last_night") and not point.get("seen_before") else 0)
        + obs_score * 25
        + bright_score * 22
        + (8 if point.get("has_lightcurve") else 0)
    )


def make_demo_candidates(points: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        [point for point in points if point["group"] == "last_night"],
        key=lambda item: item["interest_score"],
        reverse=True,
    )[:limit]
    candidates = []
    for rank, point in enumerate(ordered, start=1):
        point["is_highlighted"] = True
        summary = (
            f"{point['id']} is highlighted for exploration because it is {point['reason']} "
            f"with {point['obs_count']} brightness measurements in the public sample."
        )
        candidates.append(
            {
                "id": point["id"],
                "locus_id": point["id"],
                "label": point["id"],
                "rank": rank,
                "ra": point["ra"],
                "dec": point["dec"],
                "brightness_mag": point["brightness_mag"],
                "brightest_alert_magnitude": point["brightest_alert_magnitude"],
                "num_mag_values": point["num_mag_values"],
                "obs_count": point["obs_count"],
                "score": round(float(point["interest_score"]), 3),
                "reason": point["reason"],
                "public_summary": summary,
                "caveat": "Demo ranking is for interaction design only; it is not a classification.",
            }
        )
    return candidates


def make_demo_lightcurves(candidates: list[dict[str, Any]], points_by_id: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lightcurves: dict[str, list[dict[str, Any]]] = {}
    filters = ["g", "r", "i"]
    for candidate in candidates:
        point = points_by_id[candidate["id"]]
        base = float(point["brightness_mag"])
        latest_mjd = float(point["mjd"])
        rows = []
        for index in range(18):
            phase = index / 17
            jitter = (stable_unit(f"{point['id']}:{index}") - 0.5) * 0.18
            mag = base + math.sin(phase * math.pi * 2.2) * 0.34 + (phase - 0.5) * 0.22 + jitter
            rows.append(
                {
                    "mjd": round(latest_mjd - 17 + index, 6),
                    "magnitude": round(mag, 3),
                    "filter": filters[index % len(filters)],
                    "source": "synthetic_demo",
                }
            )
        lightcurves[point["id"]] = rows
    return lightcurves


def status_priority(status: str) -> int:
    if status == "complete":
        return 0
    if status == "under_target":
        return 1
    if status == "saturated_unresolved":
        return 2
    return 9


def discover_nights(data_root: Path) -> list[dict[str, Any]]:
    nightly_root = data_root / "data" / SURVEY_SUBDIR / "nightly"
    if not nightly_root.exists():
        raise ExportError(f"Nightly RSP directory is missing: {nightly_root}")

    nights: list[dict[str, Any]] = []
    for manifest_path in sorted(nightly_root.glob("*/*/*/manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable manifest {manifest_path}: {exc}", file=sys.stderr)
            continue

        folder = manifest_path.parent
        date_utc = manifest.get("date_utc")
        if not date_utc:
            parts = folder.parts[-3:]
            date_utc = "-".join(parts)
        loci_path = folder / "loci.parquet"
        alerts_path = folder / "alerts.parquet"
        status = str(manifest.get("status", "unknown"))
        nights.append(
            {
                "date_utc": date_utc,
                "mjd_min": manifest.get("mjd_min"),
                "mjd_max": manifest.get("mjd_max"),
                "status": status,
                "manifest": manifest,
                "manifest_path": manifest_path,
                "loci_path": loci_path,
                "alerts_path": alerts_path,
                "has_loci": loci_path.exists(),
                "has_alerts": alerts_path.exists(),
            }
        )
    return sorted(nights, key=lambda item: (str(item["date_utc"]), float(item.get("mjd_min") or 0)))


def choose_rsp_night(args: argparse.Namespace) -> dict[str, Any]:
    nights = discover_nights(args.data_root)
    if not nights:
        raise ExportError(f"No nightly manifests found under {args.data_root}/data/{SURVEY_SUBDIR}/nightly")

    if args.date:
        matches = [night for night in nights if night["date_utc"] == args.date]
        if not matches:
            raise ExportError(f"No nightly manifest found for --date {args.date}")
        selected = matches[-1]
        if not selected["has_loci"]:
            raise ExportError(f"Required loci parquet is missing for {args.date}: {selected['loci_path']}")
        if selected["status"] not in USABLE_STATUSES:
            raise ExportError(f"Night {args.date} has unusable manifest status '{selected['status']}'")
        if selected["status"] == "saturated_unresolved" and not args.allow_saturated:
            raise ExportError(
                f"Night {args.date} is saturated_unresolved. Re-run with --allow-saturated to export it with caveats."
            )
        return selected

    usable = [night for night in nights if night["has_loci"] and night["status"] in PREFERRED_LATEST_STATUSES]
    if not usable and args.allow_saturated:
        usable = [night for night in nights if night["has_loci"] and night["status"] in USABLE_STATUSES]
    if not usable:
        raise ExportError(
            "No usable nightly partitions found. Need manifest.json plus loci.parquet with status complete or under_target. "
            "Use --allow-saturated only if you intentionally want saturated_unresolved nights."
        )
    return sorted(usable, key=lambda item: (str(item["date_utc"]), -status_priority(item["status"])))[-1]


def cumulative_paths(data_root: Path) -> dict[str, Path]:
    root = data_root / "data" / SURVEY_SUBDIR / "cumulative"
    return {
        "loci_index": root / "loci_index.parquet",
        "nightly_summary": root / "nightly_summary.parquet",
    }


def load_rsp_frames(selected: dict[str, Any], data_root: Path) -> dict[str, Any]:
    paths = cumulative_paths(data_root)
    if not paths["loci_index"].exists():
        raise ExportError(f"Required cumulative loci index is missing: {paths['loci_index']}")
    if not paths["nightly_summary"].exists():
        raise ExportError(f"Required cumulative nightly summary is missing: {paths['nightly_summary']}")

    loci = read_parquet_selected(selected["loci_path"], RSP_LOCI_COLUMNS)
    if loci.empty:
        raise ExportError(f"Selected night loci parquet has no readable rows/columns: {selected['loci_path']}")
    cumulative = read_parquet_selected(paths["loci_index"], RSP_LOCI_COLUMNS)
    if cumulative.empty:
        raise ExportError(f"Cumulative loci index has no readable rows/columns: {paths['loci_index']}")

    alerts = pd.DataFrame()
    alerts_available = False
    alert_warnings: list[str] = []
    if selected["alerts_path"].exists():
        wanted = ALERT_REQUIRED_COLUMNS + ALERT_MAG_COLUMNS + ALERT_MAGERR_COLUMNS + ALERT_FILTER_COLUMNS + ALERT_TIME_COLUMNS
        alerts = read_parquet_selected(selected["alerts_path"], wanted)
        alerts_available = not alerts.empty and "locus_id" in alerts.columns
        if not alerts_available:
            alert_warnings.append("alerts.parquet exists but lacks usable public export columns.")
    else:
        alert_warnings.append("Optional alerts.parquet is missing for the selected night.")

    nightly_summary = pd.read_parquet(paths["nightly_summary"])

    selected_mjd_min = float(selected["manifest"].get("mjd_min"))
    selected_date = selected["date_utc"]
    history = cumulative.copy()
    if "night_mjd_max" in history.columns:
        history = history[pd.to_numeric(history["night_mjd_max"], errors="coerce") <= selected_mjd_min]
    if "night_date_utc" in history.columns:
        history = history[history["night_date_utc"].astype(str) < selected_date]
    if "locus_id" in loci.columns and "locus_id" in history.columns:
        history = history.drop_duplicates(subset=["locus_id"], keep="last")

    return {
        "loci": loci.reset_index(drop=True),
        "alerts": alerts.reset_index(drop=True),
        "alerts_available": alerts_available,
        "alert_warnings": alert_warnings,
        "cumulative": cumulative.reset_index(drop=True),
        "history": history.reset_index(drop=True),
        "nightly_summary": nightly_summary.reset_index(drop=True),
        "paths": paths,
    }


def deterministic_sample(df: pd.DataFrame, cap: int, seed: int, required_ids: set[str] | None = None) -> pd.DataFrame:
    if cap <= 0 or len(df) <= cap:
        return df.copy().reset_index(drop=True)
    required_ids = required_ids or set()
    if "locus_id" not in df.columns or not required_ids:
        return df.sample(cap, random_state=seed).reset_index(drop=True)

    required = df[df["locus_id"].astype(str).isin(required_ids)].copy()
    rest = df[~df["locus_id"].astype(str).isin(required_ids)].copy()
    remaining = max(0, cap - len(required))
    if len(rest) > remaining:
        rest = rest.sample(remaining, random_state=seed)
    return pd.concat([required, rest], ignore_index=True, sort=False).drop_duplicates(subset=["locus_id"], keep="first")


def normalize_loci_points(
    df: pd.DataFrame,
    group: str,
    selected_date: str,
    selected_mjd_min: float,
    historical_ids: set[str],
    alert_ids: set[str],
    highlighted_ids: set[str],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    if df.empty:
        return points

    for _, row in df.iterrows():
        locus_id = str(row.get("locus_id") or "").strip()
        if not locus_id:
            continue
        ra = number_or_none(row.get("ra"))
        dec = number_or_none(row.get("dec"))
        if ra is None or dec is None or not (0 <= ra < 360) or not (-90 <= dec <= 90):
            continue

        newest = number_or_none(row.get("newest_alert_observation_time"))
        if newest is None:
            newest = number_or_none(row.get("night_mjd_min")) or selected_mjd_min
        brightest = number_or_none(row.get("brightest_alert_magnitude"))
        obs_count = int_or_none(row.get("num_mag_values")) or 1
        tags = parse_tags(row.get("tags") if "tags" in row else row.get("all_tags"))
        is_last_night = group == "last_night"
        seen_before = (locus_id in historical_ids) if is_last_night else True
        has_lightcurve = locus_id in alert_ids
        reason = "latest processed night" if is_last_night else "saved sky memory"
        if brightest is not None:
            reason = "bright latest object" if is_last_night and brightest <= 18 else reason
        if has_lightcurve and is_last_night:
            reason = "has ANTARES alert-record brightness history"

        brightness_mag = brightest if brightest is not None else 20.5
        point = {
            "id": locus_id,
            "locus_id": locus_id,
            "label": locus_id,
            "group": group,
            "ra": round(float(ra), 5),
            "dec": round(float(dec), 5),
            "date_utc": selected_date if is_last_night else str(row.get("night_date_utc") or ""),
            "mjd": round(float(newest), 6),
            "newest_alert_observation_time": round(float(newest), 6),
            "brightness_mag": round(float(brightness_mag), 3),
            "brightest_alert_magnitude": round(float(brightest), 3) if brightest is not None else None,
            "obs_count": int(obs_count),
            "num_mag_values": int(obs_count),
            "tags": tags[:6],
            "is_last_night": bool(is_last_night),
            "seen_before": bool(seen_before),
            "has_lightcurve": bool(has_lightcurve),
            "is_highlighted": locus_id in highlighted_ids,
            "reason": reason,
            "public_description": f"{locus_id} is a {group.replace('_', ' ')} sky object from LSST-associated ANTARES analysis data.",
        }
        point["interest_score"] = round(score_point(point), 3)
        points.append(point)
    return points


def find_alert_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def alert_id_set(alerts: pd.DataFrame) -> set[str]:
    if alerts.empty or "locus_id" not in alerts.columns:
        return set()
    return set(alerts["locus_id"].dropna().astype(str))


def make_rsp_candidates(
    loci: pd.DataFrame,
    alert_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows = []
    for _, row in loci.iterrows():
        locus_id = str(row.get("locus_id") or "").strip()
        if not locus_id:
            continue
        ra = number_or_none(row.get("ra"))
        dec = number_or_none(row.get("dec"))
        if ra is None or dec is None or not (0 <= ra < 360) or not (-90 <= dec <= 90):
            continue
        mag = number_or_none(row.get("brightest_alert_magnitude"))
        obs = int_or_none(row.get("num_mag_values")) or 0
        has_lc = locus_id in alert_ids
        valid_mag = mag is not None
        brightness_score = 0.0 if mag is None else max(0.0, 30.0 - mag) * 5.0
        score = (100 if valid_mag else 0) + brightness_score + (12 if obs >= 3 else 0) + (10 if has_lc else 0)
        if obs:
            score += min(12.0, math.log10(obs + 1) * 6.0)
        if has_lc and valid_mag and obs >= 3:
            reason = "bright object with ANTARES alert-record brightness history"
        elif has_lc:
            reason = "has matching ANTARES alert records"
        elif valid_mag and obs >= 3:
            reason = "bright object with multiple magnitude values"
        elif valid_mag:
            reason = "bright latest-night object"
        else:
            reason = "latest-night object with public coordinates"
        rows.append(
            {
                "id": locus_id,
                "locus_id": locus_id,
                "label": locus_id,
                "rank": 0,
                "ra": round(float(ra), 5),
                "dec": round(float(dec), 5),
                "brightness_mag": round(float(mag), 3) if mag is not None else 20.5,
                "brightest_alert_magnitude": round(float(mag), 3) if mag is not None else None,
                "num_mag_values": int(obs),
                "obs_count": int(obs or 1),
                "score": round(float(score), 3),
                "reason": reason,
                "public_summary": (
                    f"{locus_id} is highlighted for exploration by a transparent ranking "
                    f"that favors bright latest-night objects, repeated magnitude values, "
                    f"and available ANTARES alert-record brightness history."
                ),
                "caveat": "This is an exploration ranking, not a classification or a claim of scientific importance.",
            }
        )
    ordered = sorted(rows, key=lambda item: (item["score"], -float(item["brightness_mag"])), reverse=True)[:limit]
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered


def filter_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    try:
        fid = int(value)
    except Exception:
        return str(value)
    return {1: "g", 2: "r", 3: "i"}.get(fid, str(fid))


def normalize_time(value: Any, source_column: str) -> float | None:
    numeric = number_or_none(value)
    if numeric is None:
        return None
    if source_column in {"jd", "ztf_jd"} and numeric > 2_000_000:
        return numeric - 2_400_000.5
    return numeric


def make_rsp_lightcurves(
    alerts: pd.DataFrame,
    candidates: list[dict[str, Any]],
    max_points_per_object: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], dict[str, Any]]:
    if alerts.empty or "locus_id" not in alerts.columns:
        return {}, {candidate["id"]: "No usable alerts.parquet rows were available." for candidate in candidates}, {}

    mag_col = find_alert_column(alerts, ALERT_MAG_COLUMNS)
    magerr_col = find_alert_column(alerts, ALERT_MAGERR_COLUMNS)
    time_col = find_alert_column(alerts, ALERT_TIME_COLUMNS)
    filter_col = find_alert_column(alerts, ALERT_FILTER_COLUMNS)
    source_columns = {"magnitude": mag_col, "magnitude_error": magerr_col, "time": time_col, "filter": filter_col}
    if mag_col is None or time_col is None:
        reason = f"Missing required alert columns: magnitude={mag_col}, time={time_col}."
        return {}, {candidate["id"]: reason for candidate in candidates}, source_columns

    lightcurves: dict[str, list[dict[str, Any]]] = {}
    unavailable: dict[str, str] = {}
    alerts_by_id = alerts[alerts["locus_id"].astype(str).isin([candidate["id"] for candidate in candidates])]
    for candidate in candidates:
        locus_id = candidate["id"]
        rows = alerts_by_id[alerts_by_id["locus_id"].astype(str) == locus_id].copy()
        if rows.empty:
            unavailable[locus_id] = "No alert rows matched this highlighted object."
            continue

        samples = []
        for _, row in rows.iterrows():
            mjd = normalize_time(row.get(time_col), time_col)
            mag = number_or_none(row.get(mag_col))
            magerr = number_or_none(row.get(magerr_col)) if magerr_col else None
            if mjd is None or mag is None:
                continue
            sample = {
                "mjd": round(float(mjd), 6),
                "magnitude": round(float(mag), 4),
                "filter": filter_name(row.get(filter_col)) if filter_col else "unknown",
                "source": "alerts_parquet",
            }
            if magerr is not None:
                sample["magnitude_error"] = round(float(magerr), 4)
            samples.append(sample)
        samples = sorted(samples, key=lambda item: item["mjd"])
        if len(samples) > max_points_per_object:
            step = max(1, len(samples) // max_points_per_object)
            samples = samples[::step][:max_points_per_object]
        if samples:
            lightcurves[locus_id] = samples
        else:
            unavailable[locus_id] = "Alert rows existed, but no rows had both time and magnitude values."
    return lightcurves, unavailable, source_columns


def make_density(points: list[dict[str, Any]], ra_bin_size: float, dec_bin_size: float) -> list[dict[str, Any]]:
    if not math.isfinite(ra_bin_size) or ra_bin_size <= 0:
        raise ExportError("RA bin size must be a positive finite number.")
    if not math.isfinite(dec_bin_size) or dec_bin_size <= 0:
        raise ExportError("Declination bin size must be a positive finite number.")
    ra_bin_count = math.ceil(360 / ra_bin_size)
    dec_bin_count = math.ceil(180 / dec_bin_size)
    rows = []
    for point in points:
        ra = number_or_none(point.get("ra"))
        dec = number_or_none(point.get("dec"))
        if ra is None or dec is None:
            continue
        rows.append(
            {
                "ra_bin": min(max(int(ra // ra_bin_size), 0), ra_bin_count - 1),
                "dec_bin": min(
                    max(int((dec + 90) // dec_bin_size), 0),
                    dec_bin_count - 1,
                ),
                "is_last_night": bool(point.get("is_last_night")),
            }
        )
    if not rows:
        return []

    grouped = pd.DataFrame(rows).groupby(["ra_bin", "dec_bin"], as_index=False)
    total_last = max(1, sum(row["is_last_night"] for row in rows))
    total_history = max(1, len(rows) - sum(row["is_last_night"] for row in rows))
    tiles = []
    for _, group in grouped:
        ra_bin = int(group["ra_bin"].iloc[0])
        dec_bin = int(group["dec_bin"].iloc[0])
        last_count = int(group["is_last_night"].sum())
        count = int(len(group))
        historical_count = count - last_count
        difference_score = (last_count / total_last) - (historical_count / total_history)
        tiles.append(
            {
                "id": f"ra{ra_bin:02d}_dec{dec_bin:02d}",
                "ra_min": round(ra_bin * ra_bin_size, 4),
                "ra_max": round(min((ra_bin + 1) * ra_bin_size, 360), 4),
                "dec_min": round(dec_bin * dec_bin_size - 90, 4),
                "dec_max": round(min((dec_bin + 1) * dec_bin_size - 90, 90), 4),
                "count": count,
                "last_night_count": last_count,
                "historical_count": historical_count,
                "difference_score": round(float(difference_score), 8),
                "difference_note": "Simple normalized bin-count difference; not a statistical significance claim.",
            }
        )
    return sorted(tiles, key=lambda item: item["count"], reverse=True)


def validate_payloads(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    points = payloads["sky_points.json"]["points"]
    candidates = payloads["top_candidates.json"]["candidates"]
    lightcurves = payloads["lightcurve_samples.json"]["lightcurves"]
    point_ids = [point["id"] for point in points]
    point_id_set = set(point_ids)
    candidate_ids = {candidate["id"] for candidate in candidates}
    lightcurve_ids = set(lightcurves)
    bad_coords = [
        point["id"]
        for point in points
        if not (0 <= float(point["ra"]) < 360 and -90 <= float(point["dec"]) <= 90)
    ]
    validation = {
        "ra_dec_bounds_pass": not bad_coords,
        "bad_coordinate_count": len(bad_coords),
        "top_candidates_in_sky_points": candidate_ids.issubset(point_id_set),
        "lightcurves_refer_to_candidates": lightcurve_ids.issubset(candidate_ids),
        "duplicate_sky_point_id_count": len(point_ids) - len(point_id_set),
        "json_serializable": True,
    }
    try:
        json.dumps(json_safe(payloads), allow_nan=False)
    except Exception:
        validation["json_serializable"] = False
    return validation


def comparison_payload(
    last_points: list[dict[str, Any]],
    historical_points: list[dict[str, Any]],
    alert_rows: int,
    candidates: list[dict[str, Any]],
    tiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe exactly the sampled points shipped to the public frontend."""

    overlap = sum(1 for point in last_points if point.get("seen_before") is True)
    night_count = len(last_points)
    return {
        "night_loci": night_count,
        "historical_loci": len(historical_points),
        "new_loci": night_count - overlap,
        "overlap_loci": overlap,
        "overlap_fraction_of_night": overlap / night_count if night_count else 0,
        "alert_rows": int(alert_rows),
        "highlighted_objects": len(candidates),
        "density_tiles": len(tiles),
    }


def build_demo_payloads(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    manifest_example = read_json(args.manifest)
    generated_at = now_utc()
    points, source_range = make_demo_points(load_demo_frame(args.sample_csv, args.max_points, args.seed), manifest_example)
    candidates = make_demo_candidates(points, args.top_candidates)
    points_by_id = {point["id"]: point for point in points}
    lightcurves = make_demo_lightcurves(candidates, points_by_id)
    tiles = make_density(points, args.ra_bin_size, args.dec_bin_size)
    last_points = [point for point in points if point["is_last_night"]]
    historical_points = [point for point in points if not point["is_last_night"]]
    comparison = comparison_payload(last_points, historical_points, 0, candidates, tiles)
    last_count = comparison["night_loci"]
    new_loci = comparison["new_loci"]
    historical_count = comparison["historical_loci"]
    caveats = source_caveats("demo", demo_synthetic_lightcurves=True, saturated=False)
    common = common_payload("demo", generated_at, source_range, caveats, source_range["latest_night_utc"])
    payloads = {
        "public_manifest.json": {
            **common,
            "dataset_name": "SkyPulse Eye-Candy Demo",
            "data_root_used": None,
            "source_files": (
                {"sample_csv": str(args.sample_csv), "manifest": str(args.manifest)}
                if args.include_source_paths
                else {"sample_csv": "data/antares_raw_data.csv", "manifest": "data/manifest_example.json"}
            ),
            "source_caveats": caveats,
            "alerts_available": False,
            "lightcurve_sample_source": "synthetic_demo",
            "nightly_manifest_validation": manifest_example.get("validation", {}),
            "counts": {
                "sky_points": len(points),
                "last_night_points": last_count,
                "historical_points": historical_count,
                "density_tiles": len(tiles),
                "top_candidates": len(candidates),
                "lightcurve_objects": len(lightcurves),
                "alert_rows": 0,
            },
            "validation": {},
        },
        "public_summary.json": {
            **common,
            "public_data_note": "Demo export from repository sample rows; run RSP mode for real parquet-backed data.",
            "promise": "Every night, the sky changes. SkyPulse shows where, how, and why.",
            "metrics": [
                {"label": "public sky points", "value": f"{len(points):,}", "detail": "sampled loci"},
                {"label": "latest night", "value": f"{last_count:,}", "detail": "latest processed points"},
                {"label": "new to memory", "value": f"{new_loci:,}", "detail": "not seen in prior sample"},
            ],
            "comparison": comparison,
        },
        "sky_points.json": {**common, "points": points},
        "density_tiles.json": {**common, "tiles": tiles},
        "top_candidates.json": {
            **common,
            "public_label": "Highlighted Objects",
            "ranking_note": "Transparent exploration ranking; not a classification.",
            "candidates": candidates,
        },
        "lightcurve_samples.json": {
            **common,
            "public_label": "Demo brightness story",
            "sample_source": "synthetic_demo",
            "lightcurves": lightcurves,
            "unavailable": {},
        },
    }
    payloads["public_manifest.json"]["validation"] = validate_payloads(payloads)
    return payloads


def build_rsp_payloads(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    selected = choose_rsp_night(args)
    frames = load_rsp_frames(selected, args.data_root)
    selected_manifest = selected["manifest"]
    generated_at = now_utc()
    selected_date = selected["date_utc"]
    selected_mjd_min = float(selected_manifest.get("mjd_min"))
    selected_mjd_max = float(selected_manifest.get("mjd_max"))
    expected_selected_date = mjd_to_utc_date(selected_mjd_min)
    if selected_date != expected_selected_date:
        raise ExportError(
            f"Selected manifest date_utc {selected_date!r} does not match "
            f"mjd_min {selected_mjd_min}, whose UTC date is {expected_selected_date}."
        )
    history = frames["history"]
    loci = frames["loci"]
    alerts = frames["alerts"]
    alert_ids = alert_id_set(alerts)
    historical_ids = set(history["locus_id"].dropna().astype(str)) if "locus_id" in history.columns else set()
    candidates = make_rsp_candidates(loci, alert_ids, args.top_candidates)
    highlighted_ids = {candidate["id"] for candidate in candidates}

    sampled_last = deterministic_sample(loci, args.max_last_night, args.seed, highlighted_ids)
    sampled_last_ids = set(sampled_last.get("locus_id", pd.Series(dtype="object")).dropna().astype(str))
    history_display = history.copy()
    if "locus_id" in history_display.columns:
        history_display = history_display[~history_display["locus_id"].astype(str).isin(sampled_last_ids)]
    sampled_history = deterministic_sample(history_display, args.max_history, args.seed + 1)

    last_points = normalize_loci_points(
        sampled_last,
        "last_night",
        selected_date,
        selected_mjd_min,
        historical_ids,
        alert_ids,
        highlighted_ids,
    )
    historical_points = normalize_loci_points(
        sampled_history,
        "historical",
        selected_date,
        selected_mjd_min,
        historical_ids,
        alert_ids,
        highlighted_ids,
    )
    points = last_points + historical_points
    point_by_id = {point["id"]: point for point in points}
    for candidate in candidates:
        if candidate["id"] in point_by_id:
            point_by_id[candidate["id"]]["is_highlighted"] = True
            point_by_id[candidate["id"]]["reason"] = candidate["reason"]
            point_by_id[candidate["id"]]["public_description"] = candidate["public_summary"]

    lightcurves, unavailable, source_columns = make_rsp_lightcurves(alerts, candidates, args.max_lightcurve_points)
    tiles = make_density(points, args.ra_bin_size, args.dec_bin_size)
    comparison = comparison_payload(
        last_points,
        historical_points,
        len(alerts),
        candidates,
        tiles,
    )
    saturated = selected["status"] == "saturated_unresolved"
    caveats = source_caveats("rsp_parquet", demo_synthetic_lightcurves=False, saturated=saturated)
    source_range = {
        "latest_night_utc": selected_date,
        "latest_mjd_min": selected_mjd_min,
        "latest_mjd_max": selected_mjd_max,
        "historical_mjd_min": float(history["night_mjd_min"].min()) if not history.empty and "night_mjd_min" in history.columns else None,
        "historical_mjd_max": float(history["night_mjd_max"].max()) if not history.empty and "night_mjd_max" in history.columns else selected_mjd_min,
    }
    common = common_payload("rsp_parquet", generated_at, source_range, caveats, selected_date)
    source_files = source_files_payload(
        {
            "nightly_manifest": selected["manifest_path"],
            "nightly_loci": selected["loci_path"],
            "nightly_alerts": selected["alerts_path"] if selected["alerts_path"].exists() else None,
            "cumulative_loci_index": frames["paths"]["loci_index"],
            "cumulative_nightly_summary": frames["paths"]["nightly_summary"],
        },
        args.include_source_paths,
    )
    historical_excludes_selected = True
    if not history.empty:
        if "night_date_utc" in history.columns:
            historical_excludes_selected = bool((history["night_date_utc"].astype(str) < selected_date).all())
        if "night_mjd_max" in history.columns:
            historical_excludes_selected = historical_excludes_selected and bool(
                (pd.to_numeric(history["night_mjd_max"], errors="coerce") <= selected_mjd_min).all()
            )

    payloads = {
        "public_manifest.json": {
            **common,
            "dataset_name": "SkyPulse RSP Parquet Export",
            "data_root_used": str(args.data_root) if args.include_source_paths else "provided --data-root (path omitted from public JSON)",
            "source_type_summary": "Saved RSP parquet outputs under data/lsst_only; absolute source paths omitted by default.",
            "source_files": source_files,
            "source_caveats": caveats,
            "selected_night_status": selected["status"],
            "alerts_available": frames["alerts_available"],
            "alert_warnings": frames["alert_warnings"],
            "lightcurve_sample_source": "alerts_parquet" if lightcurves else "unavailable",
            "alert_source_columns": source_columns,
            "nightly_manifest_validation": selected_manifest.get("validation", {}),
            "counts": {
                "sky_points": len(points),
                "last_night_points": len(last_points),
                "historical_points": len(historical_points),
                "density_tiles": len(tiles),
                "top_candidates": len(candidates),
                "lightcurve_objects": len(lightcurves),
                "alert_rows": int(len(alerts)),
                "nightly_loci_rows": int(len(loci)),
                "historical_loci_rows_before_selected": int(len(history)),
            },
            "validation": {
                "historical_excludes_selected_night": historical_excludes_selected,
                "manifest_append_ready": selected_manifest.get("validation", {}).get("append_ready"),
                "nightly_manifest_status": selected["status"],
                "alerts_available": frames["alerts_available"],
            },
        },
        "public_summary.json": {
            **common,
            "public_data_note": "Static public export from saved RSP parquet outputs; the frontend does not query RSP or ANTARES.",
            "promise": "Every night, the sky changes. SkyPulse shows where, how, and why.",
            "metrics": [
                {"label": "last-night loci", "value": f"{len(loci):,}", "detail": "selected nightly parquet rows"},
                {"label": "history before night", "value": f"{len(history):,}", "detail": "prior cumulative rows"},
                {"label": "alert rows", "value": f"{len(alerts):,}", "detail": "available alert-record rows"},
            ],
            "comparison": comparison,
        },
        "sky_points.json": {**common, "points": points},
        "density_tiles.json": {**common, "tiles": tiles},
        "top_candidates.json": {
            **common,
            "public_label": "Highlighted Objects",
            "ranking_note": "Transparent exploration ranking; not a classification.",
            "candidates": candidates,
        },
        "lightcurve_samples.json": {
            **common,
            "public_label": "Brightness history from ANTARES alert records.",
            "sample_source": "alerts_parquet" if lightcurves else "unavailable",
            "source_columns": source_columns,
            "lightcurves": lightcurves,
            "unavailable": unavailable,
        },
    }
    payloads["public_manifest.json"]["validation"].update(validate_payloads(payloads))
    return payloads


def print_summary(payloads: dict[str, dict[str, Any]], out_dir: Path) -> None:
    manifest = payloads["public_manifest.json"]
    counts = manifest["counts"]
    print("SkyPulse export complete")
    print(f"  mode              : {manifest['export_mode']}")
    print(f"  selected night    : {manifest['selected_night_date']}")
    print(f"  output directory  : {out_dir}")
    print(f"  sky points        : {counts['sky_points']:,}")
    print(f"  last-night points : {counts['last_night_points']:,}")
    print(f"  historical points : {counts['historical_points']:,}")
    print(f"  density tiles     : {counts['density_tiles']:,}")
    print(f"  highlighted       : {counts['top_candidates']:,}")
    print(f"  lightcurve objects: {counts['lightcurve_objects']:,}")
    if manifest.get("alert_warnings"):
        for warning in manifest["alert_warnings"]:
            print(f"  [WARN] {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="Use local CSV/sample manifest demo mode.")
    mode.add_argument(
        "--data-root",
        type=Path,
        help="RSP export root containing data/lsst_only (for example, a shared Arnor root).",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--latest", action="store_true", help="Select the latest usable nightly partition in RSP mode.")
    selection.add_argument("--date", help="Select a specific UTC night, e.g. 2026-05-30.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample-csv", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-points", type=int, default=1800, help="Demo mode point cap.")
    parser.add_argument("--max-last-night", type=int, default=2500, help="RSP mode selected-night display cap.")
    parser.add_argument("--max-history", type=int, default=2500, help="RSP mode historical display cap.")
    parser.add_argument("--top-candidates", type=int, default=10)
    parser.add_argument("--max-lightcurve-points", type=int, default=160)
    parser.add_argument("--ra-bin-size", type=float, default=15.0)
    parser.add_argument("--dec-bin-size", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-saturated", action="store_true", help="Allow saturated_unresolved RSP nights with public caveats.")
    parser.add_argument("--include-source-paths", action="store_true", help="Include absolute source paths in JSON for private debugging only.")
    args = parser.parse_args()

    if args.data_root is None and not args.demo:
        parser.error("Choose --data-root for RSP production export or --demo explicitly.")
    if args.data_root is not None and not args.latest and not args.date:
        args.latest = True
    args.out = args.out.resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.resolve()
    return args


def main() -> int:
    args = parse_args()
    try:
        payloads = build_demo_payloads(args) if args.demo else build_rsp_payloads(args)
        for filename, payload in payloads.items():
            write_json(args.out / filename, payload)
        print_summary(payloads, args.out)
        return 0
    except ExportError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
