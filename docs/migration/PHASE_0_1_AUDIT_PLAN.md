# Phase 0–1 RSP Audit Plan

## Purpose and scope

This run establishes which RSP data root is authoritative before any migration
or configuration change. It is inventory and audit only:

- read the repository and the four candidate data roots;
- write new audit products only under
  `/deleted-sundays/mdarim/antares_migration_audits`;
- do not copy, transfer, merge, rename, modify, or delete source data;
- do not run `rsync`, `scp`, `rclone`, or a backfill;
- do not clean up source roots or partial/failed audit directories.

`/deleted-sundays` is being used because `/home/mdarim` was reported at
`40001M / 40000M`. It is scratch, not a durable science destination or a new
source of truth. Every audit output must remain outside all four source roots.

## Verified Phase 0 facts

These facts were re-confirmed on RSP on 2026-07-29. They are evidence for the
audit, not yet an authority decision:

| Candidate root | Verified observation |
| --- | --- |
| `/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data` | `data/` is 1.1 GB; 90 manifests span 2026-02-25 through 2026-06-26. This remains the strongest candidate. |
| `/home/mdarim/ANTARES_Analysis_Data` | `data/` is 540 MB; 55 manifests span 2026-02-25 through 2026-05-30. |
| `/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data_backup_mdarim_group_issue` | `data/` is 740 MB; 80 manifests span 2026-02-25 through 2026-06-03. |
| `/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data` | `data/` is 740 MB; 80 manifests span 2026-02-25 through 2026-06-03. |

The RSP session is `mdarim@mdarim-nb`. The lab started in an abnormal state
because `/home/mdarim` is at `40001M / 40000M` on
`10.231.144.6:/lcv-home-share`. In this session, `HOME` and the initial working
directory are both `/deleted-sundays/mdarim`. Group ownership by
`g_antares_analysis` does not move bytes out of the `/home/mdarim` quota.

The clean RSP git root is
`/home/mdarim/notebooks/ANTARES_Analysis/ANTARES_Analysis`, on `main` at
`f451a17` (`Add RSP LSST-only ANTARES pipeline`). Both
`ANTARES_ANALYSIS_DATA_ROOT` and `ANTARES_DATA_ROOT` are unset. That checkout's
configured default resolves to `/project/mdarim/ANTARES_Analysis`, which does
not exist. The base terminal has Python 3.13.9, PyArrow 21.0.0, and NumPy
2.3.5; it does not have `antares_client`, which is why configuration was
inspected directly with `runpy` instead of importing the `src` package.
The hardened script staged at
`/deleted-sundays/mdarim/audit_antares_data_root_551b94ce.py` matches the local source
at SHA-256
`551b94ce462edda608f2adb719364cc829af29eb49e5034430bf8ea141e2c356`.
Neither a configured default, group ownership, nor cache size proves
scientific authority.

## Audit contract

The canonical interface is:

```bash
python3 /path/to/audit_antares_data_root.py \
  --data-root /absolute/source/root \
  --out /absolute/new/audit/directory
```

The RSP checkout predates the audit script, so the exact RSP procedure below
uses the SHA-256-pinned staged copy at
`/deleted-sundays/mdarim/audit_antares_data_root_551b94ce.py`.

`--out` must be new, must not be under the source root or its cache, and is
never overwritten. The audit produces exactly:

1. `summary.json`
2. `nightly_manifest_table.csv`
3. `nightly_manifest_table.txt`
4. `file_counts.json`
5. `size_summary.txt`
6. `nightly_parquet_sha256.txt`
7. `cumulative_parquet_sha256.txt`
8. `all_science_files_sha256.txt`
9. `feature_coverage.csv`
10. `feature_coverage_summary.txt`

The summary must record the resolved root, host, UTC timestamp, manifest and
complete-night counts, first/last date, total loci and alert rows, parquet
sizes, cumulative size, bytes-per-row metrics, and cache size/file count.
Extension counts and bytes are recorded in `file_counts.json`. Cache is
summarized only; the roughly 1.16 million cache files are not checksummed.

Science checksums cover durable nightly parquet/manifests and cumulative
parquet products. Checksum paths are root-relative POSIX paths so roots can be
compared without path-prefix noise. Nightly files are resolved beside each
manifest rather than trusted from stale absolute paths embedded in a
manifest.

Feature coverage must count finite values for the eight audited feature
columns and pairwise/multi-column completeness for:

- `feature_chi2_magn_r` with `feature_standard_deviation_magn_r`;
- `g` with `i`;
- `u`, `g`, and `r`;
- `r`, `i`, and `g`.

Exit status is part of the contract:

- `0`: complete audit with no integrity issue;
- `1`: all ten outputs were written, but integrity issues were found;
- `2`: usage, dependency, preflight, or runtime failure; no complete audit may
  be trusted.

## Exact RSP run commands

Run the following in one Bash terminal session. They create audit metadata
only on `/deleted-sundays`; there is no transfer command.

### 1. Establish the repository, audited script, and external output directory

```bash
bash
set -euo pipefail

if cd /home/mdarim/notebooks/ANTARES_Analysis/ANTARES_Analysis 2>/dev/null; then
  :
elif cd /home/mdarim/notebooks/ANTARES_Analysis 2>/dev/null; then
  :
else
  echo "STOP: expected RSP repository path is inaccessible"
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
cd "$REPO_ROOT" || exit 1

# The tested script is staged on scratch because the RSP checkout is older and
# the personal home quota cannot accept another write.
AUDIT_SCRIPT=/deleted-sundays/mdarim/audit_antares_data_root_551b94ce.py
test -r "$AUDIT_SCRIPT" || {
  echo "STOP: staged audit script is absent or unreadable: $AUDIT_SCRIPT"
  exit 1
}
EXPECTED_AUDIT_SCRIPT_SHA256=551b94ce462edda608f2adb719364cc829af29eb49e5034430bf8ea141e2c356
ACTUAL_AUDIT_SCRIPT_SHA256="$(
  sha256sum "$AUDIT_SCRIPT" | awk '{print $1}'
)" || {
  echo "STOP: could not hash staged audit script"
  exit 1
}
test "$ACTUAL_AUDIT_SCRIPT_SHA256" = "$EXPECTED_AUDIT_SCRIPT_SHA256" || {
  echo "STOP: staged audit script does not match the tested SHA-256"
  echo "expected=$EXPECTED_AUDIT_SCRIPT_SHA256"
  echo "actual=$ACTUAL_AUDIT_SCRIPT_SHA256"
  exit 1
}

CANDIDATES=(
  /home/mdarim/AntaresAlerts/ANTARES_Analysis_Data
  /home/mdarim/ANTARES_Analysis_Data
  /home/mdarim/AntaresAlerts/ANTARES_Analysis_Data_backup_mdarim_group_issue
  /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data
)

AUDIT_PARENT=/deleted-sundays/mdarim
test -d "$AUDIT_PARENT" && test -w "$AUDIT_PARENT" || {
  echo "STOP: expected scratch parent is absent or not writable: $AUDIT_PARENT"
  exit 1
}
AUDIT_PARENT_RESOLVED="$(readlink -f "$AUDIT_PARENT")" || exit 1
for ROOT in "${CANDIDATES[@]}"; do
  test -d "$ROOT" || continue
  ROOT_RESOLVED="$(readlink -f "$ROOT")" || exit 1
  case "$AUDIT_PARENT_RESOLVED/" in
    "$ROOT_RESOLVED/"*)
      echo "STOP: scratch output parent resolves inside source root: $ROOT"
      exit 1
      ;;
  esac
done

AUDIT_BASE="$AUDIT_PARENT/antares_migration_audits"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="$AUDIT_BASE/phase_0_1_$RUN_ID"
test ! -e "$RUN_ROOT" || {
  echo "STOP: run directory already exists: $RUN_ROOT"
  exit 1
}
mkdir -p "$RUN_ROOT/logs" || exit 1
test -w "$RUN_ROOT" || {
  echo "STOP: scratch audit directory is not writable"
  exit 1
}
echo "$RUN_ROOT"
```

### 2. Capture environment, quota, repository, and configured-root facts

```bash
{
  date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
  whoami
  hostname
  pwd
  pwd -P
  echo "HOME=$HOME"
  echo "ANTARES_ANALYSIS_DATA_ROOT=${ANTARES_ANALYSIS_DATA_ROOT-<unset>}"
  echo "ANTARES_DATA_ROOT=${ANTARES_DATA_ROOT-<unset>}"
  quota -s || true
  df -h /home/mdarim || true
  df -h /deleted-sundays || true
  git rev-parse --show-toplevel
  git status --short
  git branch --show-current
  git log -1 --oneline
  git rev-parse HEAD
  echo "expected_audit_script_sha256=$EXPECTED_AUDIT_SCRIPT_SHA256"
  sha256sum "$AUDIT_SCRIPT"
  python3 - <<'PY'
import runpy
values = runpy.run_path("src/config.py")
print(f"config.HISTORY_DATA_ROOT={values['HISTORY_DATA_ROOT']}")
PY
  grep -R \
    "DATA_ROOT\|ANTARES_ANALYSIS_DATA_ROOT\|ANTARES_Analysis_Data" \
    -n notebooks src scripts --include='*.py' --include='*.ipynb' \
    2>/dev/null | head -120 || true
} 2>&1 | tee "$RUN_ROOT/logs/phase0_environment.txt"
test "${PIPESTATUS[0]}" -eq 0 || {
  echo "STOP: Phase 0 environment capture failed"
  exit 1
}
```

### 3. Confirm all four candidate roots before auditing

```bash
RUN_ROOT_RESOLVED="$(readlink -f "$RUN_ROOT")"
CANDIDATE_LOG="$RUN_ROOT/logs/phase0_candidates.txt"
: > "$CANDIDATE_LOG"

for ROOT in "${CANDIDATES[@]}"; do
  if ! {
    echo "candidate=$ROOT"
    if test ! -d "$ROOT"; then
      echo "status=MISSING_OR_NOT_A_DIRECTORY"
      exit 1
    fi
    if test ! -r "$ROOT"; then
      echo "status=NOT_READABLE"
      exit 1
    fi
    ROOT_RESOLVED="$(readlink -f "$ROOT")" || {
      echo "status=RESOLUTION_FAILED"
      exit 1
    }
    echo "resolved=$ROOT_RESOLVED"
    case "$RUN_ROOT_RESOLVED/" in
      "$ROOT_RESOLVED/"*)
        echo "status=OUTPUT_IS_INSIDE_SOURCE"
        exit 1
        ;;
    esac
    DATA_DIR="$ROOT/data"
    if test ! -d "$DATA_DIR" || test ! -r "$DATA_DIR"; then
      echo "status=DATA_DIRECTORY_INACCESSIBLE"
      exit 1
    fi
    # The full audit performs the only complete root/cache walk. A separate
    # du over the million-file cache would duplicate expensive NFS work.
    du -sh "$DATA_DIR" || {
      echo "status=DATA_SIZE_SCAN_FAILED"
      exit 1
    }
  } 2>&1 | tee -a "$CANDIDATE_LOG"; then
    echo "STOP: candidate root is missing, unreadable, or unscannable: $ROOT"
    exit 1
  fi
done
```

### 4. Audit the roots sequentially

Sequential execution avoids multiplying load on the same RSP filesystem.
Each root is a checkpoint: only status `0` unlocks the next candidate.
Status `1`, `2`, or any undocumented exit stops the run immediately.

```bash
printf "label\troot\texit_code\n" | tee "$RUN_ROOT/audit_status.tsv"

run_audit () {
  LABEL="$1"
  ROOT="$2"
  OUT="$RUN_ROOT/$LABEL"
  LOG="$RUN_ROOT/logs/$LABEL.log"

  test ! -e "$OUT" || {
    echo "STOP: refusing existing audit output: $OUT"
    return 2
  }

  set +e
  python3 -u "$AUDIT_SCRIPT" \
    --data-root "$ROOT" \
    --out "$OUT" 2>&1 | tee "$LOG"
  RC="${PIPESTATUS[0]}"
  set -e

  printf "%s\t%s\t%s\n" "$LABEL" "$ROOT" "$RC" |
    tee -a "$RUN_ROOT/audit_status.tsv"

  case "$RC" in
    0) ;;
    1)
      echo "STOP: complete audit reported integrity issues for $ROOT"
      return 2
      ;;
    *)
      echo "STOP: incomplete/untrustworthy audit for $ROOT (exit $RC)"
      return 2
      ;;
  esac
}

set -e
run_audit mdarim_active \
  /home/mdarim/AntaresAlerts/ANTARES_Analysis_Data
run_audit mdarim_older \
  /home/mdarim/ANTARES_Analysis_Data
run_audit mdarim_group_issue_backup \
  /home/mdarim/AntaresAlerts/ANTARES_Analysis_Data_backup_mdarim_group_issue
run_audit ivezic_shared \
  /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data

for SUMMARY in "$RUN_ROOT"/*/summary.json; do
  echo "===== $SUMMARY"
  python3 -m json.tool "$SUMMARY"
done | tee "$RUN_ROOT/root_summary_comparison.txt"

echo "AUDITS COMPLETE; authoritative-root review is still required"
```

Do not remove an output directory after a failed run. Preserve its log and
partial products, report the failure, and use a new `RUN_ID` for any rerun.

## Authoritative-root decision

Choose a source only after all four audits are available and trustworthy.
The authoritative root must:

1. have audit exit status `0`, readable manifests/parquet, and no missing
   required nightly or cumulative products;
2. have the newest valid last date and the greatest coherent set of completed
   nights, not merely the greatest directory size;
3. reconcile manifest totals, parquet metadata/rows, dates, and cumulative
   products without unexplained discrepancies;
4. match the expected full-data envelope—currently 90 completed nights through
   2026-06-26, 993,218 loci, and 13,579,707 alert rows—or have a documented,
   evidence-backed reason for a newer coherent total;
5. have stable root-relative checksums for all durable science files.

Use qualified, path-set-aware checksum classifications:

- two audited durable-science sets are byte-identical only when their complete
  relative path sets are equal and every SHA-256 matches;
- an older nightly-Parquet set is a verified proper subset only when it has
  zero older-only paths, every one of its paths exists with the same SHA-256
  in the larger set, and the larger set has additional paths;
- matching hashes only on the intersection establish byte-identical overlap,
  not a subset or duplicate.

Never call whole roots duplicates: cache and other non-science content are not
content-compared. Compare nightly parquet separately from raw manifests and
cumulative products. Cumulative files at fixed relative names normally change
when snapshots end on different dates, and manifests can differ because they
store absolute source paths. Stop on an unexplained overlapping nightly
Parquet difference. Record manifest or cumulative differences and stop only
when coherent snapshot coverage or path-only metadata cannot explain them.
Cache completeness, ownership, permissions, and the configured default path
are not authority criteria.

The expected outcome is
`/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data`, but it remains provisional
until these conditions pass. The other roots remain untouched even after a
decision.

## Stop conditions

Stop and report, without transfer or cleanup, if any of the following occurs:

- the RSP repository, audit script, a candidate root, or `/deleted-sundays`
  output base is inaccessible;
- the resolved repository or configured data root differs unexpectedly and
  cannot be explained from captured environment variables;
- an audit output path already exists or resolves inside a source root;
- the audit exits `1`, `2`, or any undocumented status;
- required manifests/parquet are missing, unreadable, malformed, or changing
  during the audit;
- complete-night, date, locus, alert-row, feature-coverage, or checksum
  comparisons conflict;
- an overlapping nightly Parquet file differs, or a manifest/cumulative
  difference remains unexplained after accounting for stored paths and
  coherent snapshot coverage;
- the expected 2026-06-26/90-night/full-total envelope is not reproduced and
  no newer internally consistent dataset explains the difference;
- RSP quota/storage behavior would cause writes outside the fresh
  `/deleted-sundays` audit directory.

Phase 2 planning and all RSP-to-UW transfer work begin only after an explicit,
documented authoritative-root decision.
