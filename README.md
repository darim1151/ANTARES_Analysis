# ANTARES_Analysis

ANTARES/LSST time-domain comparison notebooks and helper functions.

## Main Workflows

- `notebooks/alerts_time_comparison.ipynb` remains the normal nightly comparison notebook.
- `notebooks/historical_backfill.ipynb` builds the cumulative LSST-only history store.
- `notebooks/rsp_setup.ipynb` checks Rubin Science Platform paths, imports, and a 5-locus LSST-only ANTARES probe.

The historical store is intentionally kept out of GitHub. On Rubin Science Platform it defaults to:

- `/project/$USER/ANTARES_Analysis/data/lsst_only/nightly/YYYY/MM/DD/loci.parquet`
- `/project/$USER/ANTARES_Analysis/data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet`
- `/project/$USER/ANTARES_Analysis/data/lsst_only/nightly/YYYY/MM/DD/manifest.json`
- `/project/$USER/ANTARES_Analysis/data/lsst_only/cumulative/loci_index.parquet`
- `/project/$USER/ANTARES_Analysis/data/lsst_only/cumulative/nightly_summary.parquet`
- `/project/$USER/ANTARES_Analysis/data/ztf_archive_do_not_use_for_lsst/`

Set `ANTARES_DATA_ROOT` before starting the kernel if your RSP project storage is elsewhere.

## LSST-Only Rule

The default query requires at least one ANTARES LSST survey identifier:

- `properties.survey.lsst.dia_object_id`
- `properties.survey.lsst.ss_object_id`

Some LSST-associated loci can also carry older ZTF IDs because ANTARES merges multi-survey histories. The pipeline records those counts in the manifest but does not reject them automatically.

## Safe Run Order

1. Start an RSP Large server and clone this repo into `$HOME/notebooks/ANTARES_Analysis`.
2. Open `notebooks/rsp_setup.ipynb` and run the setup/probe cells.
3. Open `notebooks/historical_backfill.ipynb`.
3. Run the tiny smoke backfill first: `target_loci=1000`, one night, no lightcurves.
4. Re-run the same smoke cell and confirm it resumes/skips the existing night.
5. Turn on the three-night test.
6. Run one full `100000`-target night.
7. Launch the full historical backfill only after those checks look good.
