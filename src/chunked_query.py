"""
Adaptive chunked ANTARES ingestion helpers.

ANTARES/ElasticSearch cannot safely return more than about 10,000 loci from
one query window. The old sampled path in `query.py` is still useful for quick
notebook experiments, but it is not the right tool for building a complete
nightly history.

This module turns one large MJD range into smaller MJD chunks:

    1. Start with readable chunks, usually one day or one hour.
    2. Query each chunk with the normal `query.query_range` helper.
    3. If a chunk comes back close to the ElasticSearch cap, split it in half.
    4. Keep splitting until each accepted chunk is below the cap, with
       30 seconds as the default minimum chunk size.

The result is still a normal loci DataFrame, so the existing summary, plotting,
lightcurve, cache, and validation helpers can keep working unchanged.
"""

import os
import re
import time

import pandas as pd

from . import query


SECONDS_PER_DAY = 86400.0
DEFAULT_ES_RESULT_LIMIT = 10000
DEFAULT_SPLIT_THRESHOLD = 9500
DEFAULT_INITIAL_CHUNK_DAYS = 1.0
DEFAULT_MIN_CHUNK_SECONDS = 30.0

MJD_COL = "newest_alert_observation_time"
LOCUS_ID_COL = "locus_id"


def seconds_to_mjd(seconds):
    """Convert seconds into the equivalent number of MJD days."""
    return float(seconds) / SECONDS_PER_DAY


def iter_fixed_mjd_chunks(mjd_min, mjd_max, chunk_seconds=DEFAULT_MIN_CHUNK_SECONDS):
    """
    Yield fixed-width MJD chunks as `(start, end, include_upper)` tuples.

    This is the direct "30-second bins" version of the idea. The adaptive
    query function below usually needs far fewer requests, but this iterator
    is handy when you explicitly want fixed bins for a nightly manifest.
    """
    if mjd_min > mjd_max:
        return

    step = seconds_to_mjd(chunk_seconds)
    cursor = float(mjd_min)
    final = float(mjd_max)
    while cursor < final:
        end = min(cursor + step, final)
        yield cursor, end, end >= final
        cursor = end


def _safe_part(value):
    """Return a filesystem-safe identifier component."""
    if value is None:
        return "all"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "all"


def _mjd_part(value):
    """Format an MJD for stable cache filenames."""
    return f"{float(value):.8f}".replace(".", "p")


def _chunk_cache_path(cache_dir, label, start, end, tag):
    """Build the parquet path for one accepted time chunk."""
    safe_label = _safe_part(label)
    safe_tag = _safe_part(tag)
    filename = f"chunk_{_mjd_part(start)}_{_mjd_part(end)}__tag_{safe_tag}.parquet"
    return os.path.join(cache_dir, safe_label, filename)


def _read_cached_chunk(path):
    """Load one cached chunk if it exists; otherwise return None."""
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"  [WARN] Could not read chunk cache {path}: {exc}")
        return None


def _write_cached_chunk(path, df):
    """Write one accepted chunk to disk."""
    if not path:
        return
    if df.empty and len(df.columns) == 0:
        # Pandas cannot always infer a parquet schema for a fully empty frame.
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _dedupe_loci(df):
    """Drop duplicate locus rows while keeping the newest copy when possible."""
    if df.empty or LOCUS_ID_COL not in df.columns:
        return df.reset_index(drop=True)

    if MJD_COL in df.columns:
        df = df.sort_values([LOCUS_ID_COL, MJD_COL])
    return df.drop_duplicates(subset=[LOCUS_ID_COL], keep="last").reset_index(drop=True)


def _make_initial_chunks(mjd_min, mjd_max, initial_chunk_days):
    """Split the requested range into first-pass chunks."""
    chunks = []
    cursor = float(mjd_min)
    final = float(mjd_max)
    while cursor < final:
        end = min(cursor + initial_chunk_days, final)
        chunks.append((cursor, end))
        cursor = end
    return chunks


def query_range_adaptive(label, mjd_min, mjd_max,
                         tag=None,
                         initial_chunk_days=DEFAULT_INITIAL_CHUNK_DAYS,
                         min_chunk_seconds=DEFAULT_MIN_CHUNK_SECONDS,
                         max_results_per_chunk=DEFAULT_ES_RESULT_LIMIT,
                         split_threshold=DEFAULT_SPLIT_THRESHOLD,
                         chunk_cache_dir=None,
                         use_chunk_cache=True,
                         verbose=True):
    """
    Query a full MJD range by adaptively splitting saturated time chunks.

    Returns `(df_loci, report_df)`.

    `df_loci` is the de-duplicated combined result. `report_df` has one row
    per attempted chunk so you can see which windows were accepted, split,
    loaded from cache, or still saturated at the minimum chunk size.
    """
    if mjd_min > mjd_max:
        if verbose:
            print(f"  Skipping '{label}': MJDmin > MJDmax.")
        return pd.DataFrame(), pd.DataFrame()

    min_chunk_days = seconds_to_mjd(min_chunk_seconds)
    initial_chunk_days = max(float(initial_chunk_days), min_chunk_days)
    split_threshold = min(int(split_threshold), int(max_results_per_chunk))

    pending = _make_initial_chunks(mjd_min, mjd_max, initial_chunk_days)
    accepted_frames = []
    report_rows = []
    t0 = time.time()
    attempts = 0

    if verbose:
        print(f"  Chunked query '{label}'  MJD [{mjd_min:.6f}, {mjd_max:.6f}]")
        print(f"    ES limit={max_results_per_chunk:,}, split at >= {split_threshold:,}, "
              f"minimum chunk={min_chunk_seconds:g}s")

    while pending:
        start, end = pending.pop(0)
        include_upper = end >= mjd_max
        width_seconds = (end - start) * SECONDS_PER_DAY
        attempts += 1

        cache_path = None
        df_chunk = None
        source = "live"
        if use_chunk_cache and chunk_cache_dir:
            cache_path = _chunk_cache_path(chunk_cache_dir, label, start, end, tag)
            df_chunk = _read_cached_chunk(cache_path)
            if df_chunk is not None:
                source = "cache"

        if df_chunk is None:
            try:
                df_chunk = query.query_range(
                    label=f"{label} chunk",
                    mjd_min=start,
                    mjd_max=end,
                    n_samples=max_results_per_chunk,
                    tag=tag,
                    seed=None,
                    verbose=False,
                    include_upper=include_upper,
                    raise_on_error=True,
                )
            except Exception as exc:
                report_rows.append({
                    "label": label,
                    "mjd_min": start,
                    "mjd_max": end,
                    "width_seconds": width_seconds,
                    "n_loci": 0,
                    "status": "error",
                    "source": source,
                    "include_upper": include_upper,
                    "error": str(exc),
                })
                print(f"  [ERROR] Chunk query failed for {start:.6f}-{end:.6f}: {exc}")
                raise

        n_rows = len(df_chunk)
        saturated = n_rows >= split_threshold
        can_split = (end - start) > (min_chunk_days * 1.01)

        if saturated and can_split:
            mid = (start + end) / 2.0
            pending.insert(0, (mid, end))
            pending.insert(0, (start, mid))
            status = "split"
        else:
            status = "accepted"
            if saturated:
                status = "accepted_at_minimum_saturated"
            accepted_frames.append(df_chunk)
            if source == "live" and use_chunk_cache and cache_path:
                _write_cached_chunk(cache_path, df_chunk)

        report_rows.append({
            "label": label,
            "mjd_min": start,
            "mjd_max": end,
            "width_seconds": width_seconds,
            "n_loci": n_rows,
            "status": status,
            "source": source,
            "include_upper": include_upper,
        })

        if verbose:
            queue = len(pending)
            print(f"    {attempts:>4d}. {start:.6f}-{end:.6f}  "
                  f"{width_seconds:>8.1f}s  {n_rows:>6,} loci  "
                  f"{status}  ({source}; {queue} queued)")

    report_df = pd.DataFrame(report_rows)
    if accepted_frames:
        df_loci = _dedupe_loci(pd.concat(accepted_frames, ignore_index=True))
    else:
        df_loci = pd.DataFrame()

    if verbose:
        elapsed = time.time() - t0
        n_split = (report_df["status"] == "split").sum() if not report_df.empty else 0
        n_sat = (report_df["status"] == "accepted_at_minimum_saturated").sum() if not report_df.empty else 0
        print(f"  {label}: {len(df_loci):,} unique loci from "
              f"{len(accepted_frames):,} accepted chunks  ({n_split} splits, {elapsed:.1f}s)")
        if n_sat:
            print(f"  [WARN] {n_sat} minimum-size chunks still reached the result cap. "
                  "Use smaller chunks or add another partition such as RA slices.")

    return df_loci, report_df


def load_cumulative_range(cumulative_path, mjd_min, mjd_max, mjd_col=MJD_COL):
    """
    Load the stored cumulative loci table and filter it to one MJD range.

    Missing stores are treated as empty. That lets a fresh checkout fall back
    to the old sampled query path while future nightly runs gradually build a
    complete historical record.
    """
    if not os.path.exists(cumulative_path):
        print(f"  Cumulative store not found: {cumulative_path}")
        return pd.DataFrame()

    df = pd.read_parquet(cumulative_path)
    if df.empty or mjd_col not in df.columns:
        return pd.DataFrame()

    mask = df[mjd_col].between(mjd_min, mjd_max, inclusive="both")
    return df.loc[mask].copy().reset_index(drop=True)


def update_cumulative_loci(cumulative_path, df_new,
                           id_col=LOCUS_ID_COL, mjd_col=MJD_COL):
    """
    Append new loci into a persistent cumulative parquet store.

    Existing rows with the same `locus_id` are replaced by the newest copy.
    Returns `(df_all, stats)` where `stats` records how many rows were loaded,
    added, and retained after de-duplication.
    """
    if os.path.exists(cumulative_path):
        df_old = pd.read_parquet(cumulative_path)
    else:
        df_old = pd.DataFrame()

    old_rows = len(df_old)
    new_rows = len(df_new)

    if df_old.empty and df_new.empty:
        return pd.DataFrame(), {
            "old_rows": 0,
            "new_rows": 0,
            "stored_rows": 0,
            "path": cumulative_path,
        }

    df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)
    if id_col in df_all.columns:
        if mjd_col in df_all.columns:
            df_all = df_all.sort_values([id_col, mjd_col])
        df_all = df_all.drop_duplicates(subset=[id_col], keep="last")
    df_all = df_all.reset_index(drop=True)

    os.makedirs(os.path.dirname(cumulative_path) or ".", exist_ok=True)
    df_all.to_parquet(cumulative_path, index=False)

    return df_all, {
        "old_rows": old_rows,
        "new_rows": new_rows,
        "stored_rows": len(df_all),
        "path": cumulative_path,
    }
