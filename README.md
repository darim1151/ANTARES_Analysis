# ANTARES Analysis

LSST-only ANTARES workflows for building a cumulative historical alert-era data store and comparing a real-time completed night against that stored history.

This project still uses ANTARES as the broker/source. The workflows can run on
Middle Earth or Rubin Science Platform (RSP); neither environment is a direct
Rubin Butler/TAP replacement.

## What This Project Does

The pipeline has two main jobs:

1. Build historical LSST-associated ANTARES nightly partitions.
2. Compare the newest completed ANTARES/LSST night against prior cumulative history.

Use these notebooks:

- `notebooks/historical_backfill.ipynb`: full-night historical backfill and cumulative index maintenance.
- `notebooks/alerts_time_comparison.ipynb`: real-time last-night extraction and comparison against the historical backfill.
- `notebooks/rsp_setup.ipynb`: optional portable setup/probe checks for imports, storage, and ANTARES connectivity.

## Reproducible Development and Release Checks

The project is packaged as `antares-analysis`. Python 3.11 is the production
candidate, while Python 3.9 remains in the compatibility test matrix. The
direct dependency versions in `pyproject.toml` and `environment.yml` are the
versions exercised by the current regression environment.

For local development in an isolated environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip==26.1.2
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m unittest discover -s tests -v
```

The console is a read-only control plane for the migrated Middle Earth dataset,
future-writer planning, and the repository's Jupyter workflows:

```bash
antares-analysis --version
antares-analysis --help
antares-analysis profile list
antares-analysis profile show --profile middle-earth
antares-analysis profile export --profile middle-earth
antares-analysis doctor --profile middle-earth
antares-analysis data status --profile middle-earth
antares-analysis night plan 2026-06-27 --profile middle-earth
antares-analysis night ingest 2026-06-27 --json
antares-analysis recovery inspect JOURNAL --target TARGET --stage STAGE --lock LOCK --json
antares-analysis backfill plan 2026-06-27 2026-06-29 --profile middle-earth
antares-analysis jupyter list
antares-analysis jupyter command setup --profile middle-earth
```

Add `--json` to profile inspection, `doctor`, `data status`, `night plan`,
`night ingest`, `recovery inspect`, `backfill plan`, `jupyter list`, or
`jupyter command` for stable
machine-readable output. `profile export` and
`jupyter env` render copy/paste-safe shell exports. `jupyter command` renders a
shell-safe launch command but deliberately does not execute it. Exit status 0
means the requested inspection passed, 1 means a health/status gate failed,
and 2 means the request or configuration is invalid.

Every supported CLI command is non-mutating: it does not create data or cache
directories, launch Jupyter, query ANTARES, update manifests, or rebuild
cumulative products. Planning reports the current partition state, prerequisites,
blockers, the explicitly unconfigured production operations/lock root,
validation gates, unknown estimates, and
separate derived reconciliation. Production writer commands remain disabled;
plans say `writer_not_enabled_in_this_release` rather than implying execution.
`night ingest` is a structural surface that returns exit code `4` before any
provider or write capability can be constructed. `recovery inspect` classifies
explicit journal, target, stage, and lock evidence without modifying it.
The operations decision and acceptance gates are in
`docs/architecture/ADR-0001-operations-and-writer-contracts.md`, the empirical
Arnor filesystem/publication contract is in
`docs/architecture/ADR-0002-arnor-filesystem-qualification.md`, and sequencing
is tracked in
`docs/cli/PHASED_ROADMAP.md`.

Phase 5 adds one production-shaped transactional writer implementation for
synthetic qualification. It is available to Python/Jupyter clients only when
they hold an exact sealed `SyntheticWriteCapability` and use the exact
deterministic synthetic provider. There is no production capability factory
and no live-provider adapter. The durable journal/recovery decision is recorded
in `docs/architecture/ADR-0004-transactional-writer-and-recovery.md`; operator
interpretation is in `docs/operations/PHASE5_RECOVERY_RUNBOOK.md`.

The reusable Python API is the same path used by the CLI and is safe to import
from Jupyter without invoking a subprocess:

```python
from src import operations

ctx = operations.context_from_environment()
report = operations.plan_night(ctx, "2026-06-27")
```

Plan schema `1.1` and operation-report schema `1.0` are deterministic for the
same explicit context and filesystem fixture. Phase 2 exit codes remain `0`
success, `1` failed health/validation gate, and `2` invalid
request/configuration. The operations
contract reserves `3` for operational failure, `4` for refusal/authorization,
and `5` for unexpected internal failure. A valid read-only plan returns `0`
while carrying the separate writer refusal in its report.

See `requirements/README.md` for the mandatory Linux x86_64 hashed-lock and
fresh-wheel verification procedure. Neither a macOS `pip freeze` nor
`environment.yml` alone is a production deployment lock.

The wheel contains the importable `src` package and the read-only CLI.
Operational notebooks and repository scripts remain versioned companion
material; the CLI discovers notebooks from a checkout and does not package or
execute them. When `doctor` runs from an installed wheel without a discoverable
checkout, it reports the skipped repository/notebook check as informational and
continues to enforce the installed runtime, profile, storage, dataset, and
dependency gates. Passing `--repo-root` makes the requested checkout mandatory;
every discovered checkout must still contain all supported notebooks.
`environment.yml` is an interactive convenience input; the
accepted Linux pip locks and verified wheelhouse are the supported release
artifacts.

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

## Storage and Configuration

Do not store Parquet or manifest products in GitHub. Configure the durable data
root and the rebuildable cache root independently. Set environment variables
**before** importing `src.config`; after changing them in a Jupyter environment,
restart the kernel so the module is imported with the new values.

The selected Middle Earth profile is:

```bash
export ANTARES_ANALYSIS_DATA_ROOT=/astro/store/shire/ANTARES/data
export ANTARES_ANALYSIS_CACHE_ROOT=/astro/store/shire/ANTARES/cache
export ANTARES_STORAGE_POLICY=private
```

The migrated Middle Earth dataset is private (`mdarim:mdarim`, mode `0700`).
Private mode does not require an RSP group and does not add group-write, setgid,
`chgrp`, or ACL changes. Configuring `ANTARES_ANALYSIS_CACHE_ROOT` only selects a
path: it does not create, authorize, copy, rebuild, warm, or delete a cache. The
durable root must not acquire an `ANTARES/data/cache` directory on
Middle Earth.

ANTARES owns one top-level Shire namespace: `/astro/store/shire/ANTARES`.
Authoritative science is beneath `data/`, historical migration evidence is
beneath `migration_audits/`, and any future disposable qualification run must
use `work/canary/<RUN_ID>`. Project code must not create another top-level
`/astro/store/shire/ANTARES_*` sibling. The reserved `cache/` path remains
absent until cache rollout is separately authorized.

RSP remains available as an explicit shared-group compatibility profile:

```bash
export ANTARES_ANALYSIS_DATA_ROOT=/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
export ANTARES_STORAGE_POLICY=shared-group
export ANTARES_SHARED_GROUP=g_antares_analysis
```

Set `ANTARES_ANALYSIS_CACHE_ROOT` too when the RSP cache should live outside the
data root. When the cache override is unset, `DATA_ROOT/cache` remains the
backward-compatible default.

The durable and cache layouts are therefore separate:

```text
$ANTARES_ANALYSIS_DATA_ROOT/
  data/lsst_only/nightly/YYYY/MM/DD/loci.parquet
  data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet
  data/lsst_only/nightly/YYYY/MM/DD/manifest.json
  data/lsst_only/cumulative/loci_index.parquet
  data/lsst_only/cumulative/nightly_summary.parquet

$ANTARES_ANALYSIS_CACHE_ROOT/
  <versioned query caches>
```

Avoid `/project` for this workflow unless your RSP session definitely has writable project storage.

### Shared RSP Compatibility Permissions

In `shared-group` mode, collaborators must belong to the RSP/Comanage group:

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

The code deliberately avoids world-writable permissions. In `shared-group`
mode it selects a cooperative umask and marks generated Parquet, JSON, CSV,
cache, and figure files group-readable/group-writable. These operations are
not taken in `private` mode.

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
4. Select the explicit RSP compatibility policy, then run the shared-root
   preflight:

   ```bash
   export ANTARES_ANALYSIS_DATA_ROOT=/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
   export ANTARES_STORAGE_POLICY=shared-group
   export ANTARES_SHARED_GROUP=g_antares_analysis
   python scripts/check_rsp_shared_root.py
   ```

5. Optional: run `notebooks/rsp_setup.ipynb` to verify imports and an LSST-only ANTARES probe.
6. Run `notebooks/historical_backfill.ipynb` to continue historical full-night extraction.
7. Run `notebooks/alerts_time_comparison.ipynb` to compare the newest completed night against prior historical backfill.

For local smoke tests, override both roots explicitly and use private mode:

```bash
ANTARES_ANALYSIS_DATA_ROOT=/tmp/ANTARES_Analysis_Data \
ANTARES_ANALYSIS_CACHE_ROOT=/tmp/ANTARES_Analysis_cache \
ANTARES_STORAGE_POLICY=private \
python3 -m unittest discover -s tests
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

The historical and comparison notebooks intentionally retain their existing
distinct cache-version namespaces. Do not merge them unless identical query
and cache-key semantics are demonstrated and covered by tests.

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

The analysis first audits the schemas and fully streams saved nightly
`loci.parquet` files to catch unreadable data pages. The compact snapshot itself
materializes only the requested columns and is written at:

```text
$ANTARES_ANALYSIS_DATA_ROOT/data/lsst_only/analysis/locus_feature_snapshots.parquet
$ANTARES_ANALYSIS_DATA_ROOT/data/lsst_only/analysis/locus_feature_snapshots_manifest.json
```

No ANTARES query or alert refetch is performed by this feature analysis.
If the requested properties were not saved in the nightly locus files, the
notebook writes and displays the coverage audit and skips unavailable panels.

Historical distributions use one row per unique locus: the latest saved
snapshot strictly before the current comparison night. Generated tables,
metadata, and PNG figures are written outside Git under:

```text
$ANTARES_ANALYSIS_DATA_ROOT/analysis/nightly_comparison/YYYY-MM-DD/feature_diagnostics/
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
- Set storage environment variables before importing `src.config`; restart the Jupyter kernel after changing them.
- Merely configuring a cache root does not authorize or perform cache creation or rebuilding.
- Keep `RESUME_EXISTING_NIGHT=True` for normal historical work.
- The cumulative indexes are safe to rebuild; they are derived from saved nightly manifests and parquet files.
- If a notebook reports an unexpected date in the data store, inspect it before using it in science comparisons. The backfill notebook does not remove or relocate data.
- If the shared-root preflight fails, the correct outcome is to stop before the expensive query. Fix Comanage group membership or setgid/group-write inheritance first, then restart the RSP Notebook Aspect session.
- The notebooks are interactive operational workflows, not yet an unattended production writer or scheduler.

## Current Safety Boundary

The Phase 3 foundation adds reusable operation contexts/reports, deterministic
night/backfill planning, legal writer states, contained storage paths,
single-writer ownership, strict query/fetch evidence, guarded staging,
validation-before-publication, and reconciliation-after-publication semantics.
Executable lock/transaction proofs are restricted to local temporary fixtures.
There is no production capability and no ingestion, backfill-run, reconcile, or
deployment command.

`historical_backfill.ipynb` and `alerts_time_comparison.ipynb` still contain
operational publication/reconciliation logic. They are retained unchanged to
avoid silently migrating accepted science behavior, but they are not the
future production-writer authority. Read-only Arnor acceptance must verify
`/astro/store/shire` locking and atomic-rename behavior before transactional
writer implementation or canary preparation begins. Cache rebuilding/warming
and managed Jupyter execution remain separately authorized later work.
