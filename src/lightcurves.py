"""
Lightcurve fetching helpers.

A "lightcurve" in ANTARES is the per-alert photometric history for one
locus: every detection (and its filter, magnitude, error, MJD, ...). A
locus-level query (`query.py`) does NOT include this; we have to make a
separate `get_by_id` HTTP call for each locus we want photometry on.

That makes lightcurve loading the SLOW step in the pipeline. We address
that with a `ThreadPoolExecutor` (one HTTP request per worker thread).
The work is I/O-bound, so threads outperform sequential code by roughly
the worker count - typical speedup is 10-16x.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


# ---------------------------------------------------------------------------
# COLUMN ALIASES inside the lightcurve DataFrame.
# ---------------------------------------------------------------------------
# ANTARES exposes ZTF photometry under fixed column names. Aliasing them
# here means downstream code (figures, summary, validation) doesn't have
# to repeat the magic string and can be retargeted to a different survey
# by changing one constant.
LC_MAG = 'ztf_magpsf'   # PSF-fit magnitude per alert
LC_FID = 'ztf_fid'      # Filter ID: 1=g, 2=r, 3=i


# ---------------------------------------------------------------------------
# FILTER -> (band name, plot color).
# ---------------------------------------------------------------------------
# Colors picked to be (a) distinct on a dark background and (b) loosely
# evocative of the actual filter wavelength (g=green, r=red, i=orange).
FILTER_INFO = {
    1: ('g', '#2ca02c'),
    2: ('r', '#d62728'),
    3: ('i', '#ff7f0e'),
}


# Conservative default for parallel HTTP workers. Tune up cautiously: too
# many concurrent connections will trigger ANTARES rate limits (HTTP 429)
# and timeouts. 16 is a safe starting point on the public API.
DEFAULT_MAX_LC_WORKERS = 16


def _fetch_one_lc(lid, label):
    """
    Fetch a single locus's lightcurve and tag it with its provenance.

    Returns None if the locus has no lightcurve data (rare but possible
    for very recent alerts that haven't been processed yet).

    The `label` and `locus_id` columns are added so we can later
    `pd.concat` lightcurves from many loci and still know which row came
    from where without losing that linkage in the merge.
    """
    # Imported lazily to avoid paying the antares_client import cost at
    # module load time, especially relevant in caches-only runs where this
    # helper is never invoked.
    from antares_client.search import get_by_id

    locus = get_by_id(lid)
    lc = locus.lightcurve
    if lc is not None and not lc.empty:
        # Copy before mutating - the antares_client may hand back a view
        # into an internal cache that we don't want to alter.
        lc = lc.copy()
        lc['locus_id'] = lid
        lc['range_label'] = label
        return lc
    return None


def load_lightcurves(df_loci, n_samples, label,
                     max_workers=DEFAULT_MAX_LC_WORKERS):
    """
    Fetch lightcurves in parallel for the first `n_samples` loci of `df_loci`.

    Returns a single tall DataFrame: one row per ALERT (so a locus with
    50 detections contributes 50 rows). The `locus_id` and `range_label`
    columns let downstream code group/filter back to the locus level.

    Why threads (not processes)?
        Each `get_by_id` is a network call - the GIL is released while
        Python waits on the socket, so multiple threads truly run in
        parallel. Processes would add startup overhead and complicate
        result aggregation with no speed benefit.

    Progress is printed every 100 fetches so the user can watch a long
    run advance and abort if something is clearly wrong.
    """
    if df_loci.empty:
        return pd.DataFrame()

    # `head(n_samples)` keeps sampling deterministic - we always pull the
    # same N loci from the same ordered DataFrame.
    sample_ids = df_loci['locus_id'].head(n_samples).tolist()
    n = len(sample_ids)
    print(f"  Fetching {n} lightcurves for '{label}' "
          f"(up to {max_workers} parallel workers) ...", flush=True)

    alert_rows, errors, done = [], 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Submit everything up front, then drain results as they complete.
        # Using `as_completed` gives accurate progress reporting even when
        # individual fetches finish out of order.
        futures = {pool.submit(_fetch_one_lc, lid, label): lid for lid in sample_ids}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result is not None:
                    alert_rows.append(result)
            except Exception:
                # Network blips and rare malformed responses are expected
                # at scale - count them but keep going.
                errors += 1
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    {done}/{n}  ({errors} errors)  "
                      f"{elapsed:.0f}s elapsed  ({rate:.1f} loci/s)")

    if not alert_rows:
        print(f"  No lightcurve data returned for '{label}'.")
        return pd.DataFrame()

    df_alerts = pd.concat(alert_rows, ignore_index=True)
    total_time = time.time() - t0
    print(f"  {label}: {len(df_alerts):,} alert rows from {len(alert_rows)} loci  "
          f"({errors} errors)  [{total_time:.1f}s total]")
    return df_alerts
