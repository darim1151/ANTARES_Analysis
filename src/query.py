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

import math
from numbers import Integral, Rational, Real

import numpy as np
import pandas as pd

LSST_DIA_FIELD = "properties.survey.lsst.dia_object_id"
LSST_SS_FIELD = "properties.survey.lsst.ss_object_id"
ZTF_OBJECT_ID_COL = "ztf_object_id"
_ABSENT_SURVEY_IDENTIFIER = ("absence",)


def lsst_identifier_filter():
    """
    Return the ANTARES/ElasticSearch filter for LSST-associated loci.

    ANTARES can merge multi-survey history into one locus. This filter means
    "has an LSST DIA-object or Solar-System identifier", not "has no ZTF data".
    """
    return {
        "bool": {
            "should": [
                {"exists": {"field": LSST_DIA_FIELD}},
                {"exists": {"field": LSST_SS_FIELD}},
            ],
            "minimum_should_match": 1,
        }
    }


def _identifier_sequence(value):
    """Return supported identifier-container items without testing truthiness."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return (value[()],)
        return value
    if isinstance(value, (list, tuple)):
        return value
    return None


def _missing_identifier_scalar(value):
    """Return whether one non-container identifier value is a missing sentinel."""
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _nonempty(value):
    """Return True for an identifier scalar/container carrying real content."""
    items = _identifier_sequence(value)
    if items is not None:
        return any(_nonempty(item) for item in items)
    if isinstance(value, set):
        return any(_nonempty(item) for item in value)
    if _missing_identifier_scalar(value):
        return False
    return str(value).strip() != ""


def _canonical_identifier_leaf(value):
    """Encode one scalar with deterministic, type-safe equality."""
    if value is None or value is pd.NA or value is pd.NaT:
        return ("missing", type(value))
    if isinstance(value, np.generic):
        if not isinstance(
            value, (np.bool_, np.integer, np.floating, np.str_, np.bytes_)
        ) or isinstance(value, np.timedelta64):
            raise TypeError(f"Unsupported survey identifier scalar type: {type(value)!r}")
        converted = value.item()
        if not isinstance(converted, np.generic):
            value = converted
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, Integral):
        return ("number", int(value), 1)
    if isinstance(value, Rational):
        return ("number", int(value.numerator), int(value.denominator))
    if isinstance(value, (float, np.floating, Real)):
        if _missing_identifier_scalar(value):
            return ("missing", type(value))
        infinite = (
            bool(np.isinf(value))
            if isinstance(value, np.floating)
            else math.isinf(value)
        )
        if infinite:
            return ("infinity", 1 if value > 0 else -1)
        try:
            numerator, denominator = value.as_integer_ratio()
        except (AttributeError, OverflowError, ValueError) as exc:
            raise TypeError(
                f"Unsupported real survey identifier type: {type(value)!r}"
            ) from exc
        return ("number", int(numerator), int(denominator))
    raise TypeError(f"Unsupported survey identifier scalar type: {type(value)!r}")


def _canonical_identifier_sequence_item(value):
    """Encode one item without losing its scientific value or nesting."""
    items = _identifier_sequence(value)
    if items is not None:
        return (
            "sequence",
            tuple(_canonical_identifier_sequence_item(item) for item in items),
        )
    return _canonical_identifier_leaf(value)


def canonical_survey_identifier_value(value):
    """Normalize only declared nested survey-identifier representations.

    Sequence and equal-valued numeric types are representational after a Parquet
    reopen, while identifier values, boolean-vs-number distinctions, nesting,
    order, and multiplicity remain scientific. Wholly empty containers share
    the same absent representation as null and blank scalar values.
    """
    items = _identifier_sequence(value)
    if items is not None:
        # Validate every leaf before collapsing a wholly absent container.
        # Otherwise unsupported NaN-like scalars could bypass the type gate.
        encoded = tuple(_canonical_identifier_sequence_item(item) for item in items)
        if not any(_nonempty(item) for item in items):
            return _ABSENT_SURVEY_IDENTIFIER
        return (
            "sequence",
            encoded,
        )
    canonical = _canonical_identifier_leaf(value)
    if canonical[0] == "missing" or (
        canonical[0] == "string" and not canonical[1].strip()
    ):
        return _ABSENT_SURVEY_IDENTIFIER
    return canonical


def survey_identifier_value_is_absent(value):
    """Return whether a whole identifier value has approved absence semantics."""
    return canonical_survey_identifier_value(value) == _ABSENT_SURVEY_IDENTIFIER


def _survey_lsst_dict(value):
    """Extract the nested LSST survey dict from an ANTARES survey property."""
    if not isinstance(value, dict):
        return {}
    lsst = value.get("lsst", {})
    return lsst if isinstance(lsst, dict) else {}


def lsst_identifier_counts(df):
    """Count LSST DIA, LSST Solar-System, and ZTF IDs in a loci DataFrame."""
    counts = {
        "lsst_dia_count": 0,
        "lsst_ss_count": 0,
        "lsst_identifier_count": 0,
        "ztf_object_id_count": 0,
    }
    if df is None or df.empty:
        return counts

    dia_mask = pd.Series(False, index=df.index)
    ss_mask = pd.Series(False, index=df.index)

    if "survey" in df.columns:
        dia_mask = df["survey"].map(
            lambda survey: _nonempty(_survey_lsst_dict(survey).get("dia_object_id"))
        )
        ss_mask = df["survey"].map(
            lambda survey: _nonempty(_survey_lsst_dict(survey).get("ss_object_id"))
        )

    for col in ["survey.lsst.dia_object_id", "lsst_dia_object_id", "dia_object_id"]:
        if col in df.columns:
            dia_mask = dia_mask | df[col].map(_nonempty)
    for col in ["survey.lsst.ss_object_id", "lsst_ss_object_id", "ss_object_id"]:
        if col in df.columns:
            ss_mask = ss_mask | df[col].map(_nonempty)

    ztf_mask = pd.Series(False, index=df.index)
    if ZTF_OBJECT_ID_COL in df.columns:
        ztf_mask = ztf_mask | df[ZTF_OBJECT_ID_COL].map(_nonempty)
    if "survey" in df.columns:
        ztf_mask = ztf_mask | df["survey"].map(
            lambda survey: _nonempty(survey.get("ztf", {}).get("id"))
            if isinstance(survey, dict) and isinstance(survey.get("ztf", {}), dict)
            else False
        )

    counts["lsst_dia_count"] = int(dia_mask.sum())
    counts["lsst_ss_count"] = int(ss_mask.sum())
    counts["lsst_identifier_count"] = int((dia_mask | ss_mask).sum())
    counts["ztf_object_id_count"] = int(ztf_mask.sum())
    return counts


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


def build_query(mjd_min, mjd_max, tag=None, seed=None, include_upper=True,
                lsst_only=True):
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
    if lsst_only:
        filters.append(lsst_identifier_filter())
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
                raise_on_error=False, lsst_only=True):
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

    # Keep the broker client optional for offline operations such as
    # validating existing Parquet partitions and rebuilding cumulative
    # indexes. Live query calls still fail clearly when the client is absent.
    try:
        from antares_client.search import search as antares_search
    except ImportError as exc:
        raise RuntimeError(
            "antares_client is required for live ANTARES queries but is not "
            "installed in this environment."
        ) from exc

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

    query = build_query(
        mjd_min,
        mjd_max,
        tag=tag,
        seed=seed,
        include_upper=include_upper,
        lsst_only=lsst_only,
    )
    mode = f"random (seed={seed})" if seed is not None else "newest-first"
    survey_mode = "LSST-only" if lsst_only else "all ANTARES"
    if verbose:
        print(f"  Querying '{label}'  MJD [{mjd_min:.1f}, {mjd_max:.1f}]  "
              f"n={n_samples}  [{mode}; {survey_mode}] ...", end=" ", flush=True)

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
                mjd_min,
                mjd_max,
                tag=tag,
                seed=None,
                include_upper=include_upper,
                lsst_only=lsst_only,
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


def find_latest_populated_mjd_window(label, mjd_min, mjd_max,
                                     max_days=5, tag=None, verbose=True,
                                     lsst_only=True):
    """
    Return the newest recent 1-day MJD window with at least one ANTARES locus.

    This is a cheap guard against broker indexing lag: probe `n_samples=1`
    before the expensive full chunked extraction. If every probe is empty,
    keep the original window so downstream validation reports the real issue.
    """
    base_min = float(mjd_min)
    base_max = float(mjd_max)
    checked = []

    for days_back in range(max(1, int(max_days))):
        candidate_min = base_min - days_back
        candidate_max = base_max - days_back
        try:
            probe = query_range(
                label=f"{label} probe",
                mjd_min=candidate_min,
                mjd_max=candidate_max,
                n_samples=1,
                tag=tag,
                seed=None,
                verbose=False,
                lsst_only=lsst_only,
            )
            n_probe = len(probe)
        except Exception as exc:
            n_probe = 0
            if verbose:
                print(f"  Probe failed for MJD [{candidate_min:.1f}, {candidate_max:.1f}]: {exc}")

        checked.append({
            "mjd_min": candidate_min,
            "mjd_max": candidate_max,
            "n_loci": n_probe,
        })
        if verbose:
            print(f"  Probe MJD [{candidate_min:.1f}, {candidate_max:.1f}] -> {n_probe} loci")
        if n_probe > 0:
            return candidate_min, candidate_max, pd.DataFrame(checked)

    if verbose:
        print("  [WARN] No populated recent night found; keeping configured window.")
    return base_min, base_max, pd.DataFrame(checked)
