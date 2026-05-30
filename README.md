# ANTARES Analysis

LSST-only ANTARES workflows for building a cumulative historical data store and comparing a newly completed observing night against prior history.

This project uses ANTARES as the broker/source of alert and locus data. The Rubin Science Platform (RSP) is used for compute and storage; this is not a direct Rubin Butler or TAP pipeline.

## Project Structure

- `notebooks/historical_backfill.ipynb` builds and validates full-night historical LSST-associated ANTARES partitions.
- `notebooks/alerts_time_comparison.ipynb` extracts a completed comparison night and compares it against prior cumulative history.
- `notebooks/rsp_setup.ipynb` provides setup and connectivity checks.
- `src/` contains shared configuration, query, chunking, and history helpers used by the notebooks.

## LSST-Only Selection

The analysis keeps ANTARES loci that have at least one Rubin/LSST survey identifier:

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

This means **LSST-associated ANTARES loci**. ANTARES may merge multi-survey histories into one locus, so an LSST-associated locus can still contain older ZTF identifiers or ZTF-era lightcurve history. The manifests track LSST and ZTF identifier counts for validation.

## Data Storage

Large parquet products are intentionally kept outside GitHub. On RSP, the default data root is:

```text
/home/mdarim/ANTARES_Analysis_Data
```

Expected layout:

```text
ANTARES_Analysis_Data/
  data/lsst_only/nightly/YYYY/MM/DD/loci.parquet
  data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet
  data/lsst_only/nightly/YYYY/MM/DD/manifest.json
  data/lsst_only/cumulative/loci_index.parquet
  data/lsst_only/cumulative/nightly_summary.parquet
  cache/
```

For shared RSP use, copy the `ANTARES_Analysis_Data` directory to a shared writable project location and set `DATA_ROOT` in the notebooks to that path. The `data/` directory is required for analysis; `cache/` is useful for resuming extraction but is not required for reading completed results.

## Loci and Alert Rows

- A **locus** is one ANTARES object record.
- A **lightcurve** is the alert-level photometric history associated with a locus.
- One locus can correspond to many alert/lightcurve rows.

Most sky-position plots show loci, not individual alert rows, so plotted point counts can be much smaller than the number of extracted alert rows.

## Historical Backfill Workflow

Use `notebooks/historical_backfill.ipynb` to continue the LSST-only historical store.

In normal use, edit only the user settings cell:

```python
DATA_ROOT = Path("/home/mdarim/ANTARES_Analysis_Data")
MJD_START = 61103.0
MJD_STOP = 61104.0
RESUME_EXISTING_NIGHT = True
FETCH_LIGHTCURVES = True
```

MJD ranges are half-open: `[MJD_START, MJD_STOP)`.

The current extraction strategy is probe-first tiling:

- Query time/RA/Dec tiles.
- Accept a tile only when it returns fewer than the probe threshold.
- Split dense tiles automatically.
- Stop rather than silently accepting an incomplete saturated final tile.
- Use cache files to resume repeated ANTARES and lightcurve requests.

After each successful night, cumulative indexes are rebuilt from saved nightly parquet and manifest files.

## Real-Time Comparison Workflow

Use `notebooks/alerts_time_comparison.ipynb` to compare a completed comparison night against prior cumulative history.

The comparison workflow:

1. Reads the existing historical backfill from `DATA_ROOT`.
2. Selects or accepts a completed comparison night.
3. Extracts LSST-associated ANTARES loci for that night.
4. Fetches lightcurves when enabled.
5. Compares the current night against prior cumulative loci and alert rows.

Keep historical backfill and real-time comparison as separate notebook workflows.

## Running on RSP

Clone or update the repository on RSP:

```bash
cd /home/mdarim/notebooks
git clone https://github.com/darim1151/ANTARES_Analysis.git
cd ANTARES_Analysis
git pull --ff-only
```

Open notebooks from:

```text
/home/mdarim/notebooks/ANTARES_Analysis/notebooks
```

The notebooks set the project import path explicitly so they can run whether Jupyter starts in the repo root or inside the `notebooks/` directory.

## Shared Data Use

To let another RSP user run analysis without repeating the backfill:

1. Copy the completed data store to shared storage.
2. Preserve the `ANTARES_Analysis_Data` directory layout.
3. Set `DATA_ROOT` in the notebooks to the shared path.
4. Rebuild cumulative indexes from the shared nightly files:

```python
from pathlib import Path
from src import history

DATA_ROOT = Path("/shared/path/ANTARES_Analysis_Data")
loci_index, nightly_summary = history.update_cumulative_indexes(DATA_ROOT)
```

This rebuild step reads saved parquet and manifest files; it does not query ANTARES.

## Operational Notes

- Do not commit parquet data, manifests, or cache products to GitHub.
- Do not use Colab-only paths or `google.colab` drive mounting on RSP.
- Avoid `/content` paths on RSP.
- Avoid `/project` unless a writable shared project allocation is available.
- Keep `RESUME_EXISTING_NIGHT=True` for normal historical backfill continuation.
- Treat saved nightly parquet and manifests as the source of truth for completed nights.
