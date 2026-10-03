# Middle Earth Migration Plan

## Post-promotion addendum — 2026-08-19

Run `20260816T153235Z` was subsequently promoted after the pre-promotion gates
documented below passed. The private staging root
`/astro/store/shire/ANTARES_Analysis_Data.incoming_20260816T153235Z` was
atomically renamed to the durable final root
`/astro/store/shire/ANTARES_Analysis_Data`. The staging path is now absent and
the final path exists.

The complete post-promotion audit is preserved outside the dataset at
`/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z/uw_destination_final`.
It produced all ten required reports with `audit_status=PASS`; the independent
strict verifier, source/destination comparator, and all 324 durable-file
checksum checks also passed. The accepted final dataset remains 90 complete
and append-ready nights (2026-02-25 through 2026-06-26), 993,218 loci,
13,579,707 alerts, 324 durable non-cache files, and 1,141,241,743 durable
non-cache bytes, with the accepted cumulative checksums, zero-row nights,
schemas, and feature coverage unchanged.

The final root is still private (`mdarim:mdarim`, mode `0700`); no ACL change
was made, and it contains no `cache/`. The RSP source and its authoritative
cache remain untouched. The separate Middle Earth cache root remains a future
rebuild decision and was not created or populated by this migration. The SSH
ControlMaster session used for migration verification was closed.

Everything below preserves the pre-promotion plan and chronology as historical
evidence. Statements that the final path was absent, the stage existed, or
promotion still required approval describe that earlier checkpoint and are
not the current operational state. Do not rerun the historical transfer or
promotion commands against the verified final root.

## Status and fixed paths

Status: **durable non-cache staging, destination audit, and exact comparison
passed; stopped before promotion and cache handling**.

| Role | Path |
| --- | --- |
| Authoritative RSP source | `/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data` |
| Fresh RSP source audit | `/home/mdarim/antares_migration_audits/rsp_active_pretransfer_20260816T153235Z` |
| Pinned RSP auditor | `/home/mdarim/antares_migration_tools_20260816/audit_antares_data_root.py` |
| UW host and NetID | `mdarim@arnor.astro.washington.edu` |
| Verified private staging root | `/astro/store/shire/ANTARES_Analysis_Data.incoming_20260816T153235Z` |
| External audit run | `/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z` |
| Proposed durable final root (currently absent) | `/astro/store/shire/ANTARES_Analysis_Data` |
| Proposed cache final root | `/astro/store/shire/ANTARES_Analysis_cache` |
| Complete destination audit | `/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z/uw_destination_stage_complete` |

Middle Earth documentation identifies Arnor and Gondor as SSH entry points,
states that `/astro/store` is available on both, and recommends
`/astro/store/shire` for new projects. Arnor is preferred for transfer because
the published host specification lists a 10 Gbps connection, versus 1 Gbps
for Gondor.

References:

- [Middle Earth documentation](https://middle-earth-docs.readthedocs.io/en/latest/index.html)
- [Middle Earth specifications](https://middle-earth-docs.readthedocs.io/en/latest/specs.html)

The account can create entries under `/astro/store/shire`. The verified stage
is deliberately private (`mdarim:mdarim`, mode 700). Before promotion, Darim
must confirm whether the final root should remain private or receive a
specific UW group/ACL and whether the top-level final path is the desired
shared location.

## Invariants

- Never delete or mutate the RSP source.
- Never use `rsync --delete`.
- Never overwrite an existing final or staging destination.
- Transfer durable science data first.
- Include root-level `analysis/` and the source recovery archive, not only
  `data/`.
- Treat cache as non-authoritative and transfer or rebuild it separately.
- Audit the complete staging root with the same auditor before promotion.
- Compare root-relative source and destination checksum reports.
- Stop on any checksum, manifest, row-count, date-range, feature-coverage,
  schema, or audit mismatch.
- Promote staging to the final name only after a clean audit/comparison and a
  separate user authorization.
- Keep audit output outside the dataset root so the audit does not change the
  science-file inventory it is evaluating.
- Leave the verified staging root private until final group/ACL policy is
  explicitly selected.

## Phase 1: read-only destination preflight — complete

The August 16 preflight used these fixed values:

```bash
set -euo pipefail

SOURCE=/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data
SOURCE_AUDIT=/home/mdarim/antares_migration_audits/rsp_active_pretransfer_20260816T153235Z
AUDITOR=/home/mdarim/antares_migration_tools_20260816/audit_antares_data_root.py
EXPECTED_AUDITOR_SHA=d1967e280f904129b7fa1d3e95218857a6cb67d4e099c729e294b90ecf850862

UW_NETID=mdarim
UW_HOST=arnor.astro.washington.edu
UW_FINAL=/astro/store/shire/ANTARES_Analysis_Data
UW_CACHE=/astro/store/shire/ANTARES_Analysis_cache
UW_AUDIT_BASE=/astro/store/shire/ANTARES_Analysis_Data_migration_audits
RUN_ID=20260816T153235Z
UW_STAGE="/astro/store/shire/ANTARES_Analysis_Data.incoming_${RUN_ID}"
UW_AUDIT_RUN="${UW_AUDIT_BASE}/${RUN_ID}"
```

The preflight confirmed `mdarim` on Arnor, `/astro/store/shire` writable with
467 TB available, remote `rsync`/`scp`/`tar`, and absence of the final,
staging, audit-run, and cache-final paths. The RSP image lacked local `rsync`.
Arnor's system Python 3.9.25 lacked NumPy and PyArrow, so a run-scoped virtual
environment was created under the external audit directory with NumPy 2.0.2
and PyArrow 21.0.0. The environment does not modify the science dataset or
global Python installation.

Do not rerun same-run absence checks now: the staging and audit-run paths
exist by design. The final and cache-final paths remain absent.

## Phase 2: create new staging and audit paths — complete

The run-scoped audit directory and a mode-700 staging directory were created
only after repeated absence checks. Both are owned by `mdarim:mdarim`. The
final destination was not created or modified.

Pinned tools and the fresh source audit were copied outside the staging
dataset. Their verified SHA-256 values are:

| Tool | SHA-256 |
| --- | --- |
| `audit_antares_data_root.py` | `d1967e280f904129b7fa1d3e95218857a6cb67d4e099c729e294b90ecf850862` |
| `verify_pretransfer_audit.py` | `ec747785ca69554035e65b657abfd60343ea28f264e64c12ac9f38118e31c569` |
| `compare_source_destination_audits_v2.py` | `278592773d85ef67ae1e680bd58089a3f9c89b588df5caa925f965af5cdd5a18` |

## Phase 3: transfer durable non-cache data — complete

The complete transfer scope is `data/`, root-level `analysis/`, and
`skypulse_rsp_export_2026_06_03.tar.gz`. The archive is valid gzip, contains
seven members, and has SHA-256
`6e433de0ee86ccce82a9442d3224b711366a22998ae03b28b701f4b47c8e605d7`.
The active `cache/` was deliberately excluded.

Because the RSP image had no local `rsync`, the executed bulk transfer was a
GNU tar stream over the authenticated SSH ControlMaster into the empty,
private staging root. The recovery archive was then copied only after its
target path was confirmed absent. Transfer ran from 16:36:03Z through
16:36:54Z and returned zero.

A complete non-cache SHA-256 manifest was generated through code, not edited
by hand. It covers 324 files, has SHA-256
`594ced3dfc7ed2d36f1e5562defabb6e9475f2a72f6ebeab54c09020f6db5908`,
and passed against source and destination. The `data/` plus `analysis/`
sub-manifest covers 323 files. The auditor's science subset covers 304 files,
including 30 under root-level `analysis/`; all three gates passed.

For a future transfer from an environment that actually has local `rsync`,
the equivalent commands below are correct only for a newly created, empty
staging path. **Do not run them against the verified current stage.**

```bash
rsync -aH --no-owner --no-group \
  --partial --partial-dir=.rsync-partial --info=progress2 --protect-args \
  "$SOURCE/data/" \
  "$UW_NETID@$UW_HOST:$UW_STAGE/data/"

rsync -aH --no-owner --no-group \
  --partial --partial-dir=.rsync-partial --info=progress2 --protect-args \
  "$SOURCE/analysis/" \
  "$UW_NETID@$UW_HOST:$UW_STAGE/analysis/"

rsync -a --no-owner --no-group --protect-args \
  "$SOURCE/skypulse_rsp_export_2026_06_03.tar.gz" \
  "$UW_NETID@$UW_HOST:$UW_STAGE/"
```

No transfer command used `--delete`; no existing final or staging dataset was
overwritten.

## Phase 4: audit the destination staging root — complete

The authoritative complete-stage audit used the isolated run-scoped Python:

```bash
ssh "$UW_NETID@$UW_HOST" \
  "'$UW_AUDIT_RUN/venv/bin/python' \
     '$UW_AUDIT_RUN/tools/audit_antares_data_root.py' \
     --data-root '$UW_STAGE' \
     --out '$UW_AUDIT_RUN/uw_destination_stage_complete'"
```

Required outcome:

- auditor exit 0;
- `audit_status=PASS`;
- `audit_complete=true`;
- `report_set_complete=true`;
- 90 complete and append-ready nights;
- date range 2026-02-25 through 2026-06-26;
- 993,218 loci;
- 13,579,707 alerts;
- no integrity issues; and
- unchanged feature coverage.

Every required outcome passed. The independent strict verifier also decoded
and schema-checked both zero-row Parquets, rechecked their manifests for
query/fetch errors, verified the accepted cumulative hashes and all 304
science-file hashes, and returned `PRETRANSFER_SOURCE_GATE: PASS` for the
staged root. The verifier's historical message label refers to the fixed
accepted baseline; it was run with the destination root and destination audit
arguments.

## Phase 5: compare source and destination evidence — complete

All 324 non-cache files passed a source-generated SHA-256 manifest on staging.
The source was checked against the same manifest again after comparison. The
machine-readable nightly manifest table, all three auditor checksum reports,
and feature-coverage CSV match byte-for-byte.

The two human-readable text summaries contain the audit root path. Their only
bytewise difference was the expected `Data root:` line; after excluding that
single environment-specific line, both match. A standalone fail-closed
comparator then matched 30 scientific summary fields,
`file_counts.json.science_products`, the complete non-cache inventory, and
the intended source-present/destination-absent cache state. It returned:

```text
NONCACHE_INVENTORY 324 1141241743
CACHE_SEPARATION_PASS 1160798 14609242085 0 0
AUDIT_EVIDENCE_COMPARISON: PASS
```

These commands rerun the current read-only validation without changing the
stage:

```bash
ssh "$UW_NETID@$UW_HOST" \
  "set -eu
   A='$UW_AUDIT_RUN'
   S='$UW_STAGE'
   test ! -e '$UW_FINAL'
   test ! -e \"\$S/cache\"
   cd \"\$S\"
   sha256sum --check \"\$A/source_complete_noncached_sha256.txt\" >/dev/null
   \"\$A/venv/bin/python\" \"\$A/tools/verify_pretransfer_audit.py\" \
     --audit-dir \"\$A/uw_destination_stage_complete\" \
     --data-root \"\$S\"
   \"\$A/venv/bin/python\" \
     \"\$A/tools/compare_source_destination_audits_v2.py\" \
     --source-audit \"\$A/rsp_source\" \
     --destination-audit \"\$A/uw_destination_stage_complete\" \
     --source-root '$SOURCE' \
     --destination-root \"\$S\""
```

Do not promote if any command returns nonzero or any checksum, manifest,
row-count, date-range, schema, feature, or integrity value differs.

## Phase 6: promote only after a separate approval

Promotion is an atomic same-filesystem rename. Run only after the destination
audit and all comparisons pass, the final group/ACL policy is selected, and
the user explicitly approves promotion. The current stage is mode 700; do not
change its ownership, group, mode, or ACL by assumption.

```bash
ssh "$UW_NETID@$UW_HOST" \
  "set -eu
   test ! -e '$UW_FINAL'
   test -d '$UW_STAGE/data'
   test -d '$UW_STAGE/analysis'
   test -f '$UW_STAGE/skypulse_rsp_export_2026_06_03.tar.gz'
   test -f '$UW_AUDIT_RUN/uw_destination_stage_complete/summary.json'
   grep -qx 'AUDIT_EVIDENCE_COMPARISON: PASS' \
     '$UW_AUDIT_RUN/source_destination_semantic_comparison.log'
   mv '$UW_STAGE' '$UW_FINAL'"
```

After promotion, rerun the destination audit and repeat the comparisons:

```bash
ssh "$UW_NETID@$UW_HOST" \
  "set -eu
   test -d '$UW_FINAL/data'
   test ! -e '$UW_AUDIT_RUN/uw_destination_final'
   '$UW_AUDIT_RUN/venv/bin/python' \
     '$UW_AUDIT_RUN/tools/audit_antares_data_root.py' \
     --data-root '$UW_FINAL' \
     --out '$UW_AUDIT_RUN/uw_destination_final'"
```

For the second comparison, run the strict verifier and comparator with
`--data-root '$UW_FINAL'`, `--destination-root '$UW_FINAL'`, and
`--destination-audit '$UW_AUDIT_RUN/uw_destination_final'`. Also rerun the
324-file checksum manifest from `$UW_FINAL`. Renaming must not be treated as a
substitute for the post-promotion check.

## Phase 7: cache is a separate decision

The authoritative RSP cache currently contains 1,160,798 files and
14,609,242,085 bytes. It is not included in the durable transfer above.

Preferred policy: rebuild cache on Middle Earth from the verified durable
science products if the cache is reproducible. If a full cache copy is
required, use another new staging directory and audit/count it separately.
The following reference command requires a local `rsync` binary and therefore
cannot currently run in the RSP image:

```bash
UW_CACHE_STAGE="/astro/store/shire/ANTARES_Analysis_cache.incoming_${RUN_ID}"

ssh "$UW_NETID@$UW_HOST" \
  "set -eu
   test ! -e '$UW_CACHE'
   test ! -e '$UW_CACHE_STAGE'
   mkdir '$UW_CACHE_STAGE'"

rsync -aH \
  --no-owner \
  --no-group \
  --partial \
  --partial-dir=.rsync-partial \
  --info=progress2 \
  --protect-args \
  "$SOURCE/cache/" \
  "$UW_NETID@$UW_HOST:$UW_CACHE_STAGE/"
```

Do not run these cache commands until the durable dataset is promoted and the
user selects copy versus rebuild. Do not remove the active RSP cache.

## Remaining user decisions

1. Confirm that `/astro/store/shire/ANTARES_Analysis_Data` is the desired final
   path rather than a lab/project-specific parent.
2. Select final ownership/group/mode or ACL. The verified stage is currently
   private (`mdarim:mdarim`, mode 700).
3. Explicitly authorize the atomic staging-to-final rename after reviewing
   the passed audit and comparison evidence.
4. Choose cache rebuild, full cache transfer, or selected-cache transfer.
5. After promotion, decide how long to retain the timestamped external audit
   directory and the pre-promotion staging audit evidence.

Repository portability changes, commit, and push are later work and require
separate authorization.
