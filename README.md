# ANTARES Analysis

LSST-only ANTARES workflows for building a cumulative historical alert-era data store and comparing a real-time completed night against that stored history.

This project still uses ANTARES as the broker/source. Rubin Science Platform is used for compute and storage, not as a direct Rubin Butler/TAP replacement.

## What This Project Does

The pipeline has two main jobs:

1. Build historical LSST-associated ANTARES nightly partitions.
2. Compare the newest completed ANTARES/LSST night against prior cumulative history.

Use these notebooks:

- `notebooks/historical_backfill.ipynb`: full-night historical backfill and cumulative index maintenance.
- `notebooks/alerts_time_comparison.ipynb`: real-time last-night extraction and comparison against the historical backfill.
- `notebooks/rsp_setup.ipynb`: optional setup/probe checks for imports, storage, and ANTARES connectivity.

## LSST-Only Rule

The project keeps ANTARES loci where at least one Rubin/LSST survey identifier exists:

```json
{
  "bool": {
    "should": [
      {"exists": {"field": "properties.survey.lsst.dia_object_id"}},
      {"exists": {"field": "properties.survey.lsst.ss_object_id"}}
    ],
    "minimum_should_match": 1
  }
}
```

This means "LSST-associated ANTARES loci." It does not mean direct Rubin Science Platform catalog data, and it does not guarantee the locus has no older ZTF history. ANTARES can merge multi-survey histories into one locus, so some LSST-associated loci can still carry ZTF identifiers. The manifests record those counts for transparency.

## Storage Layout

Do not store parquet or manifest products in GitHub. On RSP, use:

```text
/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
```

Expected layout:

```text
/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data/
  data/lsst_only/nightly/YYYY/MM/DD/loci.parquet
  data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet
  data/lsst_only/nightly/YYYY/MM/DD/manifest.json
  data/lsst_only/cumulative/loci_index.parquet
  data/lsst_only/cumulative/nightly_summary.parquet
  cache/
```

Avoid `/project` for this workflow unless your RSP session definitely has writable project storage.

### Shared RSP Permissions

The production data root is shared between collaborators. Both accounts must
belong to the RSP/Comanage group:

```text
g_antares_analysis
```

Before running a backfill or nightly comparison, run:

```bash
python scripts/check_rsp_shared_root.py
```

The check verifies group membership, group-writable directories, setgid
inheritance, and a real test write under the shared data root. The notebooks
run the same preflight before they rebuild indexes or query ANTARES.

If the preflight fails, fix the shared directory once instead of manually
`chmod`-ing individual outputs after each run. The usual owner/admin repair is:

```bash
chgrp -R g_antares_analysis /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
chmod -R g+rwX /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
find /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data -type d -exec chmod g+s {} \;
```

The code deliberately avoids world-writable permissions. It sets a cooperative
`umask(0o002)` in RSP notebooks and marks generated Parquet, JSON, CSV, cache,
and figure files group-readable/group-writable.

## Loci vs Alert/Lightcurve Rows

- A locus is one ANTARES object/sky position.
- A lightcurve is the per-alert photometric history for one locus.
- One locus can produce many alert/lightcurve rows.

Sky plots usually show loci, so the number of plotted points can be much smaller than the number of alert rows.

## Recommended RSP Workflow

1. Start an RSP Large server.
2. Open a terminal and go to the repo:

   ```bash
   cd /home/mdarim/notebooks/ANTARES_Analysis
   ```

3. Pull the notebook or docs you need from GitHub.
4. Run the shared-root preflight:

   ```bash
   python scripts/check_rsp_shared_root.py
   ```

5. Optional: run `notebooks/rsp_setup.ipynb` to verify imports and an LSST-only ANTARES probe.
6. Run `notebooks/historical_backfill.ipynb` to continue historical full-night extraction.
7. Run `notebooks/alerts_time_comparison.ipynb` to compare the newest completed night against prior historical backfill.

For local smoke tests, override the shared root explicitly:

```bash
ANTARES_ANALYSIS_DATA_ROOT=/tmp/ANTARES_Analysis_Data python -m unittest discover -s tests
```

## Historical Backfill

Open `notebooks/historical_backfill.ipynb`.

Edit only the **User Settings** cell for normal use:

```python
MJD_START = 61103.0
MJD_STOP = 61104.0
RESUME_EXISTING_NIGHT = True
FETCH_LIGHTCURVES = True
```

The MJD window is half-open: `[MJD_START, MJD_STOP)`.

Examples:

- `61096.0 -> 61097.0` is UTC date `2026-02-25`.
- `61102.0 -> 61103.0` is UTC date `2026-03-03`.

The notebook uses the current working probe-first strategy:

- Query small time/RA/Dec tiles.
- Accept a tile only when it returns fewer than the probe threshold.
- Split dense tiles automatically.
- Refuse to mark a night complete if a final tile is still saturated.
- Cache tile and lightcurve requests so interrupted runs can resume.

## Real-Time Comparison

Open `notebooks/alerts_time_comparison.ipynb`.

The notebook:

1. Reads the stored historical backfill.
2. Selects or accepts a real-time completed night.
3. Extracts that night with the same LSST-only probe-first strategy.
4. Fetches full lightcurves.
5. Compares the night against prior cumulative history.

Do not use `historical_backfill.ipynb` for real-time comparison; keep the roles separate.

## Feature-Space Diagnostics

The real-time comparison notebook also contains an optional feature-space
analysis for ANTARES locus properties:

- `feature_chi2_magn_r` versus `feature_standard_deviation_magn_r`.
- Weighted-mean `ugrizy` color-magnitude and color-color diagrams.
- Robust summaries, KS statistics, two-dimensional Jensen-Shannon divergence,
  and a deterministic permutation test.
- Multi-label subsets for sufficiently populated ANTARES tags.

The analysis first audits the schemas of saved nightly `loci.parquet` files.
It reads only the requested columns and builds a compact, rebuildable table at:

```text
/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data/
  data/lsst_only/analysis/locus_feature_snapshots.parquet
  data/lsst_only/analysis/locus_feature_snapshots_manifest.json
```

No ANTARES query or alert refetch is performed by this feature analysis.
If the requested properties were not saved in the nightly locus files, the
notebook writes and displays the coverage audit and skips unavailable panels.

Historical distributions use one row per unique locus: the latest saved
snapshot strictly before the current comparison night. Generated tables,
metadata, and PNG figures are written outside Git under:

```text
/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data/
  analysis/nightly_comparison/YYYY-MM-DD/feature_diagnostics/
```

These are ANTARES locus-level broker features. They may summarize accumulated
multi-survey history associated with a locus and must not be described as
measurements restricted to the selected UTC night. Features with less than
80% finite-value coverage are marked as exploratory in the saved metadata.

## Pull Only the Historical Backfill Notebook on RSP

After changes are pushed to GitHub:

```bash
cd /home/mdarim/notebooks/ANTARES_Analysis
git fetch origin
git checkout origin/main -- notebooks/historical_backfill.ipynb
```

If you also want the README update:

```bash
git checkout origin/main -- README.md
```

Do not pull `notebooks/alerts_time_comparison.ipynb` unless you intentionally want to update the comparison workflow.

## Operational Notes

- Do not run old Colab-only cells such as `google.colab` drive mounting on RSP.
- Do not use `/content` paths on RSP.
- Do not hard-code local data paths in notebooks; use `src.config.DATA_ROOT`, `CACHE_ROOT`, `NIGHTLY_ROOT`, `CUMULATIVE_ROOT`, and `ANALYSIS_ROOT`.
- Keep `RESUME_EXISTING_NIGHT=True` for normal historical work.
- The cumulative indexes are safe to rebuild; they are derived from saved nightly manifests and parquet files.
- If a notebook reports an unexpected date in the data store, inspect it before using it in science comparisons. The backfill notebook does not remove or relocate data.
- If the shared-root preflight fails, the correct outcome is to stop before the expensive query. Fix Comanage group membership or setgid/group-write inheritance first, then restart the RSP Notebook Aspect session.
