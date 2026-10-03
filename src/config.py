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

import math
import os
from pathlib import Path

from astropy.time import Time

from .cli_profiles import MIDDLE_EARTH_CACHE_ROOT, MIDDLE_EARTH_DATA_ROOT

# ---------------------------------------------------------------------------
# STORAGE / REPOSITORY PATHS
# ---------------------------------------------------------------------------
# Generated ANTARES products are large research artifacts.  The historical
# default remains the RSP location for compatibility, while deployments should
# select their data root, cache root, and permission policy explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data")
DEFAULT_SHARED_GROUP = "g_antares_analysis"
DEFAULT_STORAGE_POLICY = "private"
VALID_STORAGE_POLICIES = frozenset({"private", "shared-group"})


def _configured_data_root():
    """Return the configured durable data root.

    ANTARES_ANALYSIS_DATA_ROOT is the canonical override. ANTARES_DATA_ROOT is
    still accepted for older notebooks and shell sessions.
    """
    override = os.getenv("ANTARES_ANALYSIS_DATA_ROOT") or os.getenv("ANTARES_DATA_ROOT")
    return Path(override).expanduser() if override else DEFAULT_DATA_ROOT


def _configured_cache_root(data_root):
    """Return the independently configurable cache root without creating it."""
    override = os.getenv("ANTARES_ANALYSIS_CACHE_ROOT")
    if override:
        return Path(override).expanduser()
    if Path(data_root) == MIDDLE_EARTH_DATA_ROOT:
        return MIDDLE_EARTH_CACHE_ROOT
    return Path(data_root) / "cache"


def normalize_storage_policy(value):
    """Validate and normalize one explicit storage-policy value."""
    policy = str(value).strip().lower()
    if policy not in VALID_STORAGE_POLICIES:
        choices = ", ".join(sorted(VALID_STORAGE_POLICIES))
        raise ValueError(
            f"Invalid ANTARES_STORAGE_POLICY {value!r}; expected one of: {choices}."
        )
    return policy


def _configured_storage_policy():
    return normalize_storage_policy(
        os.getenv("ANTARES_STORAGE_POLICY", DEFAULT_STORAGE_POLICY)
    )


def _configured_shared_group(storage_policy):
    """Return the configured Unix group only when shared-group mode is active."""
    if storage_policy != "shared-group":
        return None
    group = os.getenv("ANTARES_SHARED_GROUP", DEFAULT_SHARED_GROUP).strip()
    if not group:
        raise ValueError(
            "ANTARES_SHARED_GROUP must be non-empty in shared-group storage mode."
        )
    return group


DATA_ROOT = _configured_data_root()
CACHE_ROOT = _configured_cache_root(DATA_ROOT)
STORAGE_POLICY = _configured_storage_policy()
SHARED_GROUP = _configured_shared_group(STORAGE_POLICY)
LSST_ONLY_ROOT = DATA_ROOT / "data" / "lsst_only"
NIGHTLY_ROOT = LSST_ONLY_ROOT / "nightly"
CUMULATIVE_ROOT = LSST_ONLY_ROOT / "cumulative"
ANALYSIS_ROOT = DATA_ROOT / "analysis"
FEATURE_SNAPSHOT_PATH = LSST_ONLY_ROOT / "analysis" / "locus_feature_snapshots.parquet"
FEATURE_SNAPSHOT_MANIFEST_PATH = (
    LSST_ONLY_ROOT / "analysis" / "locus_feature_snapshots_manifest.json"
)
# Backward-compatible name used by older notebooks and scripts.  It is
# deliberately None in private mode so ANTARES_SHARED_GROUP cannot silently
# activate group-specific behavior.
EXPECTED_SHARED_GROUP = SHARED_GROUP

# ---------------------------------------------------------------------------
# SURVEY / PLATFORM MODE
# ---------------------------------------------------------------------------
# The science path now targets LSST-associated ANTARES loci. ANTARES can merge
# histories across surveys, so "lsst" means the locus has at least one LSST
# survey identifier, not that it necessarily has no older ZTF association.
SURVEY_MODE = os.getenv("ANTARES_SURVEY_MODE", "lsst").strip().lower()
LSST_ONLY = os.getenv("ANTARES_LSST_ONLY", "1") != "0"


# Default Rubin alert-history start. MJD 61095.0 is 2026-02-24.
LSST_HISTORY_START_MJD = float(os.getenv("ANTARES_LSST_HISTORY_START_MJD", "61095.0"))

# Store LSST products separately from older broad/ZTF-era products.
HISTORY_DATA_SUBDIR = os.getenv(
    "ANTARES_HISTORY_DATA_SUBDIR",
    "lsst_only" if LSST_ONLY else "all_antares",
)

# ---------------------------------------------------------------------------
# CURRENT TIME ANCHOR
# ---------------------------------------------------------------------------
# AUTO_REALTIME_LAST_NIGHT controls whether Range 1 tracks the latest complete
# UTC MJD day at runtime. Keep it on for operational "last night" comparisons.
# Set the environment variable ANTARES_AUTO_REALTIME=0 if you need a frozen,
# exactly reproducible historical rerun.
AUTO_REALTIME_LAST_NIGHT = os.getenv("ANTARES_AUTO_REALTIME", "1") != "0"
FROZEN_MJD_NOW = 61103.0

# ANTARES can lag the newest completed UTC day while broker products are still
# being indexed. A one-day lookback makes "Last Night" mean the most recent
# completed day that is likely to be queryable, avoiding zero-row fresh-night
# windows such as 61164-61165 when 61163-61164 is the latest populated night.
ANTARES_INDEXING_LOOKBACK_DAYS = float(os.getenv("ANTARES_INDEXING_LOOKBACK_DAYS", "1"))

# Before the expensive full extraction, the comparison notebook can probe a
# few recent 1-day windows and use the newest one that actually has ANTARES
# loci. This protects the analysis when the broker indexing lag is longer than
# the static lookback above.
AUTO_SELECT_POPULATED_LAST_NIGHT = os.getenv("ANTARES_AUTO_SELECT_POPULATED", "1") != "0"
ANTARES_LAST_NIGHT_SEARCH_DAYS = int(os.getenv("ANTARES_LAST_NIGHT_SEARCH_DAYS", "5"))


def latest_completed_mjd_day():
    """Return the integer MJD boundary for the latest completed UTC day."""
    return float(math.floor(Time.now().mjd))


# MJD_NOW is the "today" anchor used to derive Range 1 ("last night").
MJD_NOW = (
    latest_completed_mjd_day() - ANTARES_INDEXING_LOOKBACK_DAYS
    if AUTO_REALTIME_LAST_NIGHT
    else FROZEN_MJD_NOW
)

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
MJD2_MIN = LSST_HISTORY_START_MJD
MJD2_MAX = MJD1_MIN
LABEL2 = "Cumulative LSST History"

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

# Approximate MJD when Rubin/LSST alert products should be treated as valid
# for this project. Used by validation to reject accidental ZTF-era backfills.
LSST_START_MJD = LSST_HISTORY_START_MJD

# ---------------------------------------------------------------------------
# CUMULATIVE NIGHTLY HISTORY PARAMETERS
# ---------------------------------------------------------------------------
# Backward-compatible alias used by older modules and notebooks.
HISTORY_DATA_ROOT = DATA_ROOT

# Research target for historical backfill. If a night has fewer available
# loci, the manifest records `under_target` rather than failing the run.
HISTORY_TARGET_LOCI = 100000

# Keep all lightcurves for accepted loci. This is scientifically richer but
# slow/storage-heavy, so notebooks expose an easy override for smoke tests.
HISTORY_FETCH_ALL_LIGHTCURVES = True

# Parallelism for the per-locus lightcurve fetch in history workflows.
HISTORY_MAX_LIGHTCURVE_WORKERS = 16

# Parallel parent shards for adaptive ANTARES locus ingestion. Each parent
# shard is a non-overlapping MJD interval; adaptive splitting happens inside
# the shard. Reduce to 2 if ANTARES begins returning timeouts/rate limits.
CHUNK_PARALLEL_SHARDS = int(os.getenv("ANTARES_CHUNK_PARALLEL_SHARDS", "3"))

# Resume interrupted Colab runs by skipping nightly partitions that already
# have a manifest plus parquet outputs.
HISTORY_RESUME_EXISTING_NIGHTS = True

# Nightly comparison notebook behavior. Prefer the platform-backed cumulative
# history built by notebooks/historical_backfill.ipynb, and load historical
# alert parquet only when it exists. This avoids re-querying history.
USE_STORED_CUMULATIVE_HISTORY = True
LOAD_CUMULATIVE_HISTORY_ALERTS = True

# Backward-compatible alias used by older notebook cells.
USE_DRIVE_CUMULATIVE_HISTORY = USE_STORED_CUMULATIVE_HISTORY


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
    print(f"  Survey mode       : {SURVEY_MODE}")
    print(f"  LSST-only filter  : {'ON' if LSST_ONLY else 'OFF'}")
    print(f"  LSST history start: MJD {LSST_HISTORY_START_MJD:.1f}")
    print(f"  Tag filter        : {QUERY_TAG if QUERY_TAG else 'none (all alerts)'}")
    print(f"  Random seed       : {RANDOM_SEED}")
    print(f"  Realtime night    : {'ON' if AUTO_REALTIME_LAST_NIGHT else 'OFF'}")
    if AUTO_REALTIME_LAST_NIGHT:
        print(f"  ANTARES lookback  : {ANTARES_INDEXING_LOOKBACK_DAYS:g} day(s)")
        print(f"  Populated search  : {'ON' if AUTO_SELECT_POPULATED_LAST_NIGHT else 'OFF'}")
        if AUTO_SELECT_POPULATED_LAST_NIGHT:
            print(f"  Search depth      : {ANTARES_LAST_NIGHT_SEARCH_DAYS} day(s)")
    print(f"  Chunked ingest    : {'ON' if USE_CHUNKED_INGEST else 'OFF'}")
    if USE_CHUNKED_INGEST:
        print(f"  Chunk start size  : {CHUNK_INITIAL_DAYS:g} day(s)")
        print(f"  Chunk min size    : {CHUNK_MIN_SECONDS:g} sec")
        print(f"  Chunk split at    : {CHUNK_SPLIT_THRESHOLD:,}/{CHUNK_MAX_RESULTS:,} loci")
        print(f"  Parallel shards   : {CHUNK_PARALLEL_SHARDS}")
        print(f"  History backfill  : {'ON' if CHUNKED_BACKFILL_HISTORY else 'OFF'}")
    print(f"  History data root : {HISTORY_DATA_ROOT}")
    print(f"  History data set  : {HISTORY_DATA_SUBDIR}")
    print(f"  History target    : {HISTORY_TARGET_LOCI:,} loci/night")
    print(f"  History LC fetch  : {'ON' if HISTORY_FETCH_ALL_LIGHTCURVES else 'OFF'}")
    print(f"  Use stored history: {'ON' if USE_DRIVE_CUMULATIVE_HISTORY else 'OFF'}")
    print(f"  Project root       : {PROJECT_ROOT}")
    print(f"  Data root          : {DATA_ROOT}")
    print(f"  Cache root         : {CACHE_ROOT}")
    print(f"  Storage policy     : {STORAGE_POLICY}")
    print(f"  Shared group       : {SHARED_GROUP or 'not used'}")
    print(f"  Nightly root       : {NIGHTLY_ROOT}")
    print(f"  Cumulative root    : {CUMULATIVE_ROOT}")
    print(f"  Analysis root      : {ANALYSIS_ROOT}")
    # Spell out the disjointness check so a reviewer can verify the
    # "no overlap" property at a glance.
    overlap = "NON-overlapping" if MJD1_MIN >= MJD2_MAX else "OVERLAPPING"
    print(f"\n  Ranges are {overlap}  (MJD2_MAX={MJD2_MAX:.1f}, MJD1_MIN={MJD1_MIN:.1f})")
