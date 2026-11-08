# ANTARES Analysis

Night-by-night extraction of LSST-associated loci from the [ANTARES](https://antares.noirlab.edu) broker, with each night certified only when no search tile is saturated.

ANTARES's search caps each query at about 10,000 results. A night of LSST-associated loci can exceed that, so a single query can return only part of the night. This project splits each night into time × RA × Dec tiles, probes each tile, splits any tile that reaches the cap, and refuses to mark a night complete while a tile is still saturated. Accepted nights are saved as Parquet with a manifest, and the cumulative index is rebuilt from those manifests.

> [!NOTE]
> **Which branch to read.** `main` is the notebook workflow used on the Rubin Science Platform (RSP), documented below. The packaged pipeline (resumable acquisition, transactional publication and an offline test suite of 570 test functions) is on [`release/g664-r3-v0.4.10`](https://github.com/darim1151/ANTARES_Analysis/tree/e7fd9f95c30b8e456def49d4e25a570125d1ff08) (commit `e7fd9f9`). Earlier versions are [tagged](https://github.com/darim1151/ANTARES_Analysis/tags) v0.4.1–v0.4.9.

**Data so far.** 90 processed nights between 25 February and 26 June 2026 (not every night in that range; two returned zero rows), with 993,218 locus rows and 13,579,707 light-curve rows. A locus's light curve is its full ANTARES history, so these rows include earlier ZTF detections and are not a count of LSST alerts. The data is stored outside Git.

**What the data is.** ANTARES loci with at least one Rubin/LSST identifier (see [LSST-Only Rule](#lsst-only-rule)). RSP is used for compute and storage; nothing here is Rubin Butler or TAP catalog data, and a locus can also carry older ZTF history.

**Public view.** [SkyPulse](https://sky-pulse-theta.vercel.app) is a static map built from one export of this pipeline's saved products. It reads JSON files and does not query ANTARES or Rubin.

**Credits.** Nirav Nemali started the repository and contributed the initial data pull, the first real-time comparison notebook and the original `src/` layout (February–April 2026). Parts of the later code and documentation were written with AI coding agents: commits authored or co-authored by `Claude`, and work on the `claude/*` and `codex/*` branches.

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
4. Optional: run `notebooks/rsp_setup.ipynb` to verify imports and an LSST-only ANTARES probe.
5. Run `notebooks/historical_backfill.ipynb` to continue historical full-night extraction.
6. Run `notebooks/alerts_time_comparison.ipynb` to compare the newest completed night against prior historical backfill.

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
- Keep `RESUME_EXISTING_NIGHT=True` for normal historical work.
- The cumulative indexes are safe to rebuild; they are derived from saved nightly manifests and parquet files.
- If a notebook reports an unexpected date in the data store, inspect it before using it in science comparisons. The backfill notebook does not remove or relocate data.
