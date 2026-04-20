"""
Parquet-based on-disk cache for the four query DataFrames.

Why cache at all?
    Re-running every cell from scratch hits ANTARES with two ~5000-locus
    queries plus thousands of lightcurve HTTP calls. That can take many
    minutes on a slow link AND puts unnecessary load on the public ANTARES
    service. With caching, a re-run of the notebook with the SAME
    parameters loads everything from disk in well under a second.

Why parquet?
    - Columnar, compressed, fast to round-trip with pandas.
    - Preserves dtypes (CSV would lose datetimes/ints/strings).
    - Schema-flexible: ANTARES `properties` add/remove columns over time
      and parquet handles that gracefully.

Cache invalidation strategy:
    The filename encodes every parameter that would change the result -
    MJD bounds for each range and N_SAMPLES. If the user edits any of
    those in `config.py`, a new tag is generated and the old cache is
    naturally ignored (and orphaned on disk for them to clean up if they
    care). No timestamp-based invalidation - we trust the filename.
"""

import os
import time

import pandas as pd


def cache_paths(cache_dir, mjd1_min, mjd1_max, mjd2_min, mjd2_max, n_samples):
    """
    Return a dict of {dataframe_name: filepath} for the four cache files.

    The tag is a deterministic function of every parameter that affects
    the result, so the same inputs always produce the same paths.
    """
    tag = (
        f"r1_{mjd1_min:.0f}_{mjd1_max:.0f}"
        f"__r2_{mjd2_min:.0f}_{mjd2_max:.0f}"
        f"__n{n_samples}"
    )
    return {
        'df1':        os.path.join(cache_dir, f"cache_{tag}_r1_loci.parquet"),
        'df2':        os.path.join(cache_dir, f"cache_{tag}_r2_loci.parquet"),
        'df1_alerts': os.path.join(cache_dir, f"cache_{tag}_r1_alerts.parquet"),
        'df2_alerts': os.path.join(cache_dir, f"cache_{tag}_r2_alerts.parquet"),
    }


def try_load_cache(paths, label1, label2, use_cache=True):
    """
    Attempt to load all four DataFrames from disk.

    Returns `(loaded, df1, df2, df1_alerts, df2_alerts)`.
    `loaded` is True only if EVERY file was present and successfully read -
    a partial cache is treated as no cache so the pipeline doesn't wind up
    mixing fresh and stale data.

    Empty DataFrames are returned in the not-loaded case so the caller can
    unconditionally reference the names without `if df1 is None` guards.
    """
    all_present = all(os.path.exists(p) for p in paths.values())
    if not (use_cache and all_present):
        if use_cache and not all_present:
            print("No cache found - will run live ANTARES queries.")
        else:
            print("USE_CACHE = False - running live ANTARES queries.")
        return False, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print("Cache found - loading from disk (ANTARES queries skipped)")
    print("=" * 55)
    t0 = time.time()
    df1 = pd.read_parquet(paths['df1'])
    df2 = pd.read_parquet(paths['df2'])
    df1_alerts = pd.read_parquet(paths['df1_alerts'])
    df2_alerts = pd.read_parquet(paths['df2_alerts'])
    print(f"  {label1:20s}: {len(df1):>5d} loci,  {len(df1_alerts):>6,} alert rows")
    print(f"  {label2:20s}: {len(df2):>5d} loci,  {len(df2_alerts):>6,} alert rows")
    print(f"  Loaded in {time.time() - t0:.2f}s")
    print("  To force a fresh query: delete cache_*.parquet files or set USE_CACHE = False.")
    return True, df1, df2, df1_alerts, df2_alerts


def save_cache(paths, df1, df2, df1_alerts, df2_alerts):
    """
    Write the four DataFrames to their cache paths.

    Empty DataFrames are SKIPPED rather than saved as zero-row parquet
    files. The reason is subtle but important: if the user runs once with
    `LOAD_LIGHTCURVES = False`, df1_alerts is empty. Saving that empty
    cache would later fool `try_load_cache` into thinking the lightcurve
    fetch had legitimately produced no data, blocking a future re-run from
    populating it. Skipping the save preserves the option to top up later.
    """
    print("Saving results to Parquet cache ...")
    print("=" * 55)
    for name, df in [
        ('df1',        df1),
        ('df2',        df2),
        ('df1_alerts', df1_alerts),
        ('df2_alerts', df2_alerts),
    ]:
        path = paths[name]
        if not df.empty:
            # Ensure target directory exists - matters on a fresh checkout
            # where /data/ may not be present yet.
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            df.to_parquet(path, index=False)
            print(f"  Saved {name:12s} -> {path}  ({len(df):,} rows)")
        else:
            print(f"  Skipped {name:12s} (empty DataFrame)")
    print("Cache written.  Next run will load from disk automatically.")
