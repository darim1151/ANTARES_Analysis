"""
ANTARES locus-level query helpers.

The ANTARES broker exposes an ElasticSearch-backed search API. This module
contains everything we need to:

    1. Translate an ANTARES `Locus` object into a plain dict / DataFrame row
       (`locus_to_record`).
    2. Build an ElasticSearch query that filters on an MJD window AND
       returns the rows in a deterministic-but-random order
       (`build_query`).
    3. Execute that query, harvest up to N loci, and gracefully fall back
       to an unrandomised query if the cluster rejects random_score
       (`query_range`).

The functions here intentionally do NOT touch lightcurves - those are
handled in `lightcurves.py`. Splitting locus-level metadata from
per-alert photometry keeps the cheap query (seconds) separable from the
expensive one (one HTTP request per locus).
"""

import pandas as pd
from antares_client.search import search as antares_search


def locus_to_record(locus):
    """
    Flatten an ANTARES `Locus` object into a single dict (one DataFrame row).

    Why merge `locus.properties`?
        ANTARES stores most useful per-object metadata
        (`brightest_alert_magnitude`, `num_mag_values`,
        `newest_alert_observation_time`, ...) inside `properties`. Spreading
        them into top-level keys means a downstream `pd.DataFrame(records)`
        call gives us one column per ANTARES field automatically - no
        bespoke schema mapping required.
    """
    record = {
        'locus_id': locus.locus_id,
        'ra':       locus.ra,
        'dec':      locus.dec,
        # Tags are a list; flatten to a comma string so the column has a
        # consistent scalar dtype that parquet can round-trip.
        'tags':     ', '.join(locus.tags) if locus.tags else '',
    }
    if locus.properties:
        record.update(locus.properties)
    return record


def build_query(mjd_min, mjd_max, tag=None, seed=None, include_upper=True):
    """
    Build the ElasticSearch query body for an MJD window.

    Two query shapes are produced:

      - `seed is None`  -> plain bool/filter query. ES returns hits ordered
        by relevance (which, for a pure filter query, collapses to
        newest-first by `newest_alert_observation_time`). Fine for "latest N"
        usage; BAD for sampling, because two non-overlapping windows would
        each return their respective newest objects with no shuffling.

      - `seed is not None` -> wrap the bool query in a `function_score`
        with `random_score`. Each document gets a deterministic random
        score derived from `hash(seed, _seq_no)`, and `boost_mode="replace"`
        throws away any relevance score so ordering is purely random.

    Why `"field": "_seq_no"` is REQUIRED in ES 7+:
        Without it, ES 7+ silently ignores the seed and reverts to
        relevance-based ordering. The symptom is two "random" samples that
        look identical because they are actually both newest-first lists.
        `_seq_no` is a per-shard monotonic ID guaranteed present on every
        document, so it gives the random hash a stable input.
    `include_upper=False` switches the upper MJD bound from `lte` to `lt`.
    Chunked ingestion uses that half-open form for intermediate chunks so
    loci exactly on a boundary are not fetched twice.
    """
    # ANTARES indexes the locus-level "last seen" time as
    # properties.newest_alert_observation_time. Filtering on this field
    # (rather than per-alert times) lets us pull a snapshot without
    # joining alert tables.
    upper_op = "lte" if include_upper else "lt"
    mjd_filter = {"range": {
        "properties.newest_alert_observation_time": {"gte": mjd_min, upper_op: mjd_max}
    }}
    filters = [mjd_filter]
    if tag:
        # ANTARES tags live on the locus; a `term` clause is an exact match.
        filters.append({"term": {"tags": tag}})

    bool_clause = {"bool": {"filter": filters}}

    if seed is not None:
        return {
            "query": {
                "function_score": {
                    "query":      bool_clause,
                    "functions":  [{"random_score": {"seed": int(seed), "field": "_seq_no"}}],
                    "boost_mode": "replace",
                }
            }
        }

    return {"query": bool_clause}


def query_range(label, mjd_min, mjd_max, n_samples,
                tag=None, seed=None, verbose=True, include_upper=True,
                raise_on_error=False):
    """
    Execute the ANTARES query and return up to `n_samples` loci as a DataFrame.

    Behaviour notes:
      - If the MJD range is invalid (min > max) we bail out with an empty
        DataFrame. This is the project's "skip-don't-fail" rule.
      - If the random_score query is rejected by the ES cluster (older
        ANTARES deployments without function_score support), we retry once
        with a plain query and warn. The fallback returns newest-first
        objects, which is non-ideal for sampling but better than no data.
      - Per-locus parsing errors are tolerated: we count them but don't
        re-raise, because one badly-formed locus shouldn't kill a 5000-row
        query.
      - `raise_on_error=True` is intended for chunked ingestion. It prevents
        transient network/API errors from being mistaken for genuinely empty
        time chunks.
    """
    if mjd_min > mjd_max:
        if verbose:
            print(f"  Skipping '{label}': MJDmin > MJDmax.")
        return pd.DataFrame()

    def _collect(q, limit):
        """Iterate the ANTARES generator until we have `limit` records."""
        recs, errs = [], 0
        for locus in antares_search(q):
            try:
                recs.append(locus_to_record(locus))
            except Exception:
                # Tolerate malformed loci - one bad row shouldn't kill the run.
                errs += 1
            if len(recs) >= limit:
                break
        return recs, errs

    query = build_query(mjd_min, mjd_max, tag=tag, seed=seed, include_upper=include_upper)
    mode = f"random (seed={seed})" if seed is not None else "newest-first"
    if verbose:
        print(f"  Querying '{label}'  MJD [{mjd_min:.1f}, {mjd_max:.1f}]  "
              f"n={n_samples}  [{mode}] ...", end=" ", flush=True)

    records, errors = [], 0
    try:
        records, errors = _collect(query, n_samples)
    except Exception as exc:
        # Most likely cause: the ES cluster rejected `random_score` (older
        # ANTARES deployments don't support function_score on every index).
        # Retry once without randomisation rather than aborting the run.
        if seed is not None:
            if verbose:
                print(f"\n  [WARN] random_score query failed ({exc}); "
                      "retrying without randomisation ...")
            fallback = build_query(
                mjd_min, mjd_max, tag=tag, seed=None, include_upper=include_upper
            )
            try:
                records, errors = _collect(fallback, n_samples)
            except Exception as exc2:
                if raise_on_error:
                    raise
                if verbose:
                    print(f"  [ERROR] Fallback also failed: {exc2}")
                return pd.DataFrame()
        else:
            if raise_on_error:
                raise
            if verbose:
                print(f"  [ERROR] Query failed: {exc}")
            return pd.DataFrame()

    df = pd.DataFrame(records)
    if verbose:
        extra = f"  ({errors} parse errors)" if errors else ""
        print(f"retrieved {len(df)} loci.{extra}")
    return df


def query_both_ranges_parallel(range1_args, range2_args):
    """
    Run the two range queries concurrently with a 2-worker thread pool.

    The two queries are independent network calls, so wall-clock time
    drops by ~2x compared to running them serially. Threads (not
    processes) are correct here because each call spends almost all its
    time blocked on I/O - the GIL is released during the ANTARES HTTP
    request, so true parallelism is achieved despite Python's threading
    model.

    Each `*_args` is a dict of kwargs for `query_range`; pass `None` to
    skip a range entirely (e.g. when its MJD window was invalid).
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(query_range, **range1_args) if range1_args else None
        f2 = pool.submit(query_range, **range2_args) if range2_args else None
        df1 = f1.result() if f1 is not None else pd.DataFrame()
        df2 = f2.result() if f2 is not None else pd.DataFrame()
    return df1, df2
