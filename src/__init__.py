"""
ANTARES / LSST alert-distribution comparison helpers.

This package contains all the logic backing
`notebooks/alerts_time_comparison.ipynb`. Splitting the code into modules
keeps the notebook a thin storyboard while each helper stays unit-testable
and reusable from other scripts.

Module map:
    config       - MJD windows, sample sizes, validation of ranges
    query        - ANTARES locus-level queries (with random_score)
    chunked_query - Adaptive chunked ingestion for complete nightly updates
    history      - Platform-backed cumulative nightly history pipeline
    feature_analysis - Locus feature snapshots, statistics, and figures
    rsp_permissions - Portable private/shared storage preflight and permissions
    lightcurves  - Parallel per-locus lightcurve fetching
    cache        - Parquet load/save keyed by query parameters
    summary      - Human-readable summary statistics
    figures      - The three matplotlib plots produced by the analysis
    validation   - Eight-test data-integrity suite
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

__all__ = [
    "cache",
    "chunked_query",
    "cli",
    "config",
    "feature_analysis",
    "figures",
    "history",
    "lightcurves",
    "operations",
    "query",
    "rsp_permissions",
    "summary",
    "validation",
]


def __getattr__(name: str) -> ModuleType:
    """Load public submodules on demand.

    Keeping package initialization lightweight lets metadata-only commands such
    as ``antares-analysis --version`` run without importing the scientific stack
    or evaluating storage configuration. Existing ``from src import config``
    style imports remain compatible through this lazy loader.
    """
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Include lazily exposed modules in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
