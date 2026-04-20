"""
ANTARES / LSST alert-distribution comparison helpers.

This package contains all the logic backing
`notebooks/alerts_time_comparison.ipynb`. Splitting the code into modules
keeps the notebook a thin storyboard while each helper stays unit-testable
and reusable from other scripts.

Module map:
    config       - MJD windows, sample sizes, validation of ranges
    query        - ANTARES locus-level queries (with random_score)
    lightcurves  - Parallel per-locus lightcurve fetching
    cache        - Parquet load/save keyed by query parameters
    summary      - Human-readable summary statistics
    figures      - The three matplotlib plots produced by the analysis
    validation   - Eight-test data-integrity suite
"""

from . import cache, config, figures, lightcurves, query, summary, validation

__all__ = [
    "cache",
    "config",
    "figures",
    "lightcurves",
    "query",
    "summary",
    "validation",
]
