"""
Configuration constants and MJD-range validation.

This module is the single place where the user picks WHICH two MJD windows
the rest of the pipeline will compare. Centralising it here means the
notebook stays free of magic numbers and a future user can re-run a
different comparison just by editing this file (or by importing the module
and overriding the constants in the notebook).

WHY MJD (Modified Julian Date)?
    ANTARES stores every alert observation time as an MJD float, so the
    queries we send and the validation we run all speak in MJD. Converting
    to/from calendar dates only happens at the edges (titles, log lines).
"""

from astropy.time import Time

# ---------------------------------------------------------------------------
# CURRENT TIME ANCHOR
# ---------------------------------------------------------------------------
# MJD_NOW is the "today" anchor used to derive Range 1 ("last night").
# Hard-coding it instead of using astropy.time.Time.now() means the analysis
# is REPRODUCIBLE: the exact same MJD window is queried every time the
# notebook is run, regardless of when it is executed.
MJD_NOW = 61103.0

# ---------------------------------------------------------------------------
# RANGE 1 - "Last night" snapshot of what LSST observed most recently.
# ---------------------------------------------------------------------------
# A 1-day window ending at MJD_NOW. This represents the freshest slice of
# the survey - useful for spotting new transients while they are still
# bright/active.
MJD1_MIN = MJD_NOW - 1
MJD1_MAX = MJD_NOW
LABEL1 = "Last Night"

# ---------------------------------------------------------------------------
# RANGE 2 - Cumulative LSST history (everything BEFORE last night).
# ---------------------------------------------------------------------------
# WHY MJD2_MAX = MJD1_MIN (and not MJD_NOW)?
#   ANTARES indexes loci by `newest_alert_observation_time`, which is a
#   LOCUS-LEVEL "last seen" field. Any object that was active last night
#   has its newest_alert_observation_time falling inside Range 1. If we
#   let Range 2 extend up to MJD_NOW we would re-include those same objects
#   and inflate the apparent overlap between the two samples. By cutting
#   Range 2 off at MJD1_MIN the two windows are STRICTLY DISJOINT, which
#   is what we want for an honest "tonight vs everything before tonight"
#   comparison.
MJD2_MIN = 60200.0   # ~Oct 2024, approximate start of LSST science ops
MJD2_MAX = MJD1_MIN
LABEL2 = "LSST History"

# ---------------------------------------------------------------------------
# SAMPLING / QUERY PARAMETERS
# ---------------------------------------------------------------------------
# N_SAMPLES caps the number of loci pulled per range. ANTARES can return
# millions of loci; 5000 is enough to make the histograms statistically
# meaningful while keeping the queries quick (~seconds, not minutes).
N_SAMPLES = 5000

# QUERY_TAG optionally restricts the query to loci carrying a specific
# ANTARES tag (e.g. 'in_LSSTDDF' for Deep Drilling Fields). None means
# "no tag filter, give me everything in the MJD window".
QUERY_TAG = None

# RANDOM_SEED makes the random sampling reproducible. ANTARES uses
# ElasticSearch random_score with this seed so the same N_SAMPLES are
# returned across reruns - critical for caching to be meaningful.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# CHUNKED INGESTION PARAMETERS
# ---------------------------------------------------------------------------
# USE_CHUNKED_INGEST switches the notebook from "sample up to N_SAMPLES loci"
# to an adaptive chunked query designed to avoid the ElasticSearch ~10,000
# result cap. The old sampled path remains in the notebook as a fallback.
USE_CHUNKED_INGEST = True

# Backfilling the full historical range can require many ANTARES requests.
# Keep this False for routine nightly runs: Range 1 is ingested chunk-by-chunk,
# appended to the cumulative store, and Range 2 is read from that store. Turn
# it on only when you intentionally want to build history from scratch.
CHUNKED_BACKFILL_HISTORY = False

# Start with 1-day chunks, then split only dense chunks. This usually needs
# far fewer requests than always using 30-second bins.
CHUNK_INITIAL_DAYS = 1.0

# 30 seconds is the minimum chunk size requested for dense windows.
CHUNK_MIN_SECONDS = 30.0

# ANTARES/ElasticSearch can truncate around 10,000 hits. We ask for no more
# than that and split a little early so "almost capped" chunks are not trusted.
CHUNK_MAX_RESULTS = 10000
CHUNK_SPLIT_THRESHOLD = 9500

# Approximate MJD when LSST science operations began. Used by the
# validation suite to flag pre-LSST observations.
LSST_START_MJD = 60200.0


def validate_mjd_range(label, mjd_min, mjd_max):
    """
    Decide whether an MJD window is queryable and print a one-line summary.

    The "core project rule" is: if MJDmin > MJDmax the range is INVALID
    and must be skipped silently by every downstream cell. Returning a
    boolean here lets the caller short-circuit queries and plots without
    raising exceptions, which keeps the notebook flowing even when the
    user intentionally disables one of the two ranges.
    """
    if mjd_min > mjd_max:
        # Print rather than raise: an invalid range is a configuration
        # choice (e.g. "I only want Range 2 today"), not an error.
        print(f"  [{label}]  INVALID - MJDmin ({mjd_min}) > MJDmax ({mjd_max}). Range skipped.")
        return False

    # Convert MJD floats to ISO calendar dates only for human-readable output.
    # The pipeline itself never needs this - only the printed summary does.
    t_min = Time(mjd_min, format='mjd').iso[:10]
    t_max = Time(mjd_max, format='mjd').iso[:10]
    span = mjd_max - mjd_min
    print(f"  [{label}]  {t_min}  ->  {t_max}  ({span:.1f} days)")
    return True


def print_config_summary():
    """Pretty-print the active configuration so the notebook log is self-documenting."""
    print("Configuration")
    print("=" * 55)
    for label, lo, hi in [(LABEL1, MJD1_MIN, MJD1_MAX), (LABEL2, MJD2_MIN, MJD2_MAX)]:
        span = hi - lo
        valid = lo <= hi
        status = "OK" if valid else "INVALID (will be skipped)"
        print(f"  {label}: MJD {lo:.1f} - {hi:.1f}  ({span:.1f} days)  [{status}]")
    print(f"  Samples per range : {N_SAMPLES}")
    print(f"  Tag filter        : {QUERY_TAG if QUERY_TAG else 'none (all alerts)'}")
    print(f"  Random seed       : {RANDOM_SEED}")
    print(f"  Chunked ingest    : {'ON' if USE_CHUNKED_INGEST else 'OFF'}")
    if USE_CHUNKED_INGEST:
        print(f"  Chunk start size  : {CHUNK_INITIAL_DAYS:g} day(s)")
        print(f"  Chunk min size    : {CHUNK_MIN_SECONDS:g} sec")
        print(f"  Chunk split at    : {CHUNK_SPLIT_THRESHOLD:,}/{CHUNK_MAX_RESULTS:,} loci")
        print(f"  History backfill  : {'ON' if CHUNKED_BACKFILL_HISTORY else 'OFF'}")
    # Spell out the disjointness check so a reviewer can verify the
    # "no overlap" property at a glance.
    overlap = "NON-overlapping" if MJD1_MIN >= MJD2_MAX else "OVERLAPPING"
    print(f"\n  Ranges are {overlap}  (MJD2_MAX={MJD2_MAX:.1f}, MJD1_MIN={MJD1_MIN:.1f})")
