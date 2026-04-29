# ANTARES_Analysis

ANTARES/LSST time-domain comparison notebooks and helper functions.

## Main Workflows

- `notebooks/alerts_time_comparison.ipynb` remains the normal nightly comparison notebook.
- `notebooks/historical_backfill.ipynb` builds the cumulative Google Drive history store.

The historical store is intentionally kept out of GitHub. In Colab it writes:

- `/content/drive/MyDrive/ANTARES_Analysis/data/nightly/YYYY/MM/DD/loci.parquet`
- `/content/drive/MyDrive/ANTARES_Analysis/data/nightly/YYYY/MM/DD/alerts.parquet`
- `/content/drive/MyDrive/ANTARES_Analysis/data/nightly/YYYY/MM/DD/manifest.json`
- `/content/drive/MyDrive/ANTARES_Analysis/data/cumulative/loci_index.parquet`
- `/content/drive/MyDrive/ANTARES_Analysis/data/cumulative/nightly_summary.parquet`

## Safe Run Order

1. Open `notebooks/historical_backfill.ipynb` in Colab.
2. Run the bootstrap, install, Drive mount, and import cells.
3. Run the tiny smoke backfill first: `target_loci=1000`, one night, no lightcurves.
4. Re-run the same smoke cell and confirm it resumes/skips the existing night.
5. Turn on the three-night test.
6. Run one full `100000`-target night.
7. Launch the full historical backfill only after those checks look good.
