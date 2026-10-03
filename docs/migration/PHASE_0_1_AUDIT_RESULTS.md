# Phase 0–1 RSP Audit Results

> **Post-promotion status — 2026-08-19:** the verified staging root from run
> `20260816T153235Z` was atomically promoted to
> `/astro/store/shire/ANTARES_Analysis_Data`, and the complete final audit at
> `/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z/uw_destination_final`
> passed. See `MIDDLE_EARTH_MIGRATION_PLAN.md` for the full addendum. The text
> below intentionally preserves the earlier pre-promotion audit chronology.

## Decision

Phase 0–1 is complete. The authoritative migration source is:

`/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data`

The active root passed the full integrity audit after the zero-row-night
policy was implemented through the pipeline and the affected manifests and
cumulative products were regenerated. The other three roots are older or
duplicate legacy states. None contains a later date, a candidate-only science
night, a candidate-only nightly Parquet file, or a conflicting same-path
nightly Parquet checksum.

| Decision gate | State |
| --- | --- |
| Phase 0 environment and provenance | `COMPLETE` |
| Active-root integrity | `PASS` |
| Candidate-root classification | `COMPLETE` |
| Authoritative source selected | `YES` |
| Durable-data staging and destination audit | `PASS` |
| Actual RSP-to-UW transfer | `324 NON-CACHE FILES STAGED; NOT PROMOTED` |
| Final destination | `ABSENT` |
| Cache transfer | `NOT STARTED` |
| Source deletion, destination overwrite, commit, or push | `NOT PERFORMED` |

No older candidate root was repaired, merged, promoted, or otherwise
modified. No source or recovery data was deleted. The only earlier
space-recovery action was removal of cache directories from two confirmed
legacy duplicates; the active-root cache remains present. The authoritative
non-cache source is now copied to a private timestamped staging root on
Arnor, but the final destination has not been created.

## Zero-row-night policy

A successfully queried LSST-only night with zero loci and zero alerts is
complete and append-ready only when all of the following hold:

- `loci.parquet` and `alerts.parquet` both exist;
- both Parquet schemas are readable and valid;
- the reconciled counts are `actual_loci == 0` and `alert_rows == 0`;
- no query or fetch errors are recorded; and
- the night is otherwise structurally complete.

The policy does not waive missing files, invalid schemas, query/fetch errors,
or other structural failures. It was implemented in validation code rather
than by hand-editing JSON. The pipeline regenerated:

- `data/lsst_only/nightly/2026/03/05/manifest.json`
- `data/lsst_only/nightly/2026/03/11/manifest.json`
- `data/lsst_only/cumulative/loci_index.parquet`
- `data/lsst_only/cumulative/nightly_summary.parquet`

The two accepted empty nights are 2026-03-05 and 2026-03-11.

## Authoritative active-root audit

Fresh pre-transfer audit evidence:

`/home/mdarim/antares_migration_audits/rsp_active_pretransfer_20260816T153235Z`

The same ten-report source snapshot is archived on Arnor at:

`/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z/rsp_source`

The earlier August 2 evidence under `/deleted-sundays` had expired under that
filesystem's retention policy by August 16, so the source was audited again
with the pinned repository auditor before any destination path was created.

| Metric | Audited value |
| --- | ---: |
| Audit status | `PASS` |
| Complete report set | `true` |
| Integrity issue count | 0 |
| Manifest / physical / complete-and-append-ready nights | 90 / 90 / 90 |
| Date range | 2026-02-25 through 2026-06-26 |
| Total loci | 993,218 |
| Total alerts | 13,579,707 |
| Nightly science Parquet files | 180 |
| Durable files excluding cache | 324 |
| Durable bytes excluding cache | 1,141,241,743 |
| Cache files | 1,160,798 |
| Cache bytes | 14,609,242,085 |
| Entire-root bytes | 15,750,483,828 |

The nightly products, `loci_index.parquet`, and
`nightly_summary.parquet` reconcile on rows and dates. Current cumulative
checksums are:

| Product | SHA-256 |
| --- | --- |
| `data/lsst_only/cumulative/loci_index.parquet` | `f75196d18690e610ab6e79231b244c3fddca396a68eea08dd2d0408e91d8b587` |
| `data/lsst_only/cumulative/nightly_summary.parquet` | `85c5fac9c242fa2e7993155036ada649336b0affe8ffc8843d2c5733ea765114` |

The regenerated zero-row manifest checksums are:

| Night | SHA-256 |
| --- | --- |
| 2026-03-05 | `c4528942b0c967a350e2c99c7c64922bef9cb4fa60100d12b4c4b723c094891a` |
| 2026-03-11 | `a57b9ec917d0944910df3c465f9e9e96f9da569eb0df745880bbcd8160646624` |

## Candidate-root classification

| Root | Audit | Nights | Date range | Loci | Alerts | Classification |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data` | `PASS` | 90 | 2026-02-25–2026-06-26 | 993,218 | 13,579,707 | **Authoritative** |
| `/home/mdarim/ANTARES_Analysis_Data` | `FAIL` under stale pre-policy metadata | 55 | 2026-02-25–2026-05-30 | 366,279 | 7,052,946 | Legacy pre-zero-row-policy subset |
| `/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data_backup_mdarim_group_issue` | `PASS` | 80 | 2026-02-25–2026-06-03 | 635,511 | 9,511,355 | Older byte-identical science subset |
| `/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data` | `PASS` | 80 | 2026-02-25–2026-06-03 | 635,511 | 9,511,355 | Older duplicate of the 80-night backup |

The four findings in `/home/mdarim/ANTARES_Analysis_Data` are retained as
expected legacy evidence: two zero-row manifests predate the accepted policy,
and its cumulative summary reflects stale legacy inclusion semantics. They do
not block migration because the candidate is strictly older and contributes
no unique or conflicting durable nightly science.

The 80-night backup audit was originally written to the following retention
path, which had expired by the August 16 pre-transfer run:

`/deleted-sundays/mdarim/antares_migration_audits/rsp_candidate_backup_group_issue_20260803T085500Z`

It passed with zero integrity issues. It has 262 durable files and
770,799,024 durable bytes excluding cache. Its legacy cache is absent.

The Ivezić-root audit was originally written to the following retention path,
which had also expired by the August 16 pre-transfer run:

`/deleted-sundays/mdarim/antares_migration_audits/rsp_candidate_ivezic_20260803T091000Z`

It also passed with zero integrity issues and has the same durable counts as
the backup. Its cache contains 608,088 files and 5,377,717,843 bytes; cache
presence is not a scientific-authority criterion.

## Candidate checksum and date-set comparisons

All comparisons used root-relative nightly science paths and SHA-256 values.

| Candidate | Candidate files | Active-only files | Candidate-only files | Conflicting shared paths |
| --- | ---: | ---: | ---: | ---: |
| `/home/mdarim/ANTARES_Analysis_Data` | 110 | 70 | 0 | 0 |
| Backup group-issue root | 160 | 20 | 0 | 0 |
| Ivezić root | 160 | 20 | 0 | 0 |

The 80-night candidates have no dates outside the active set. The ten
active-only dates relative to both 80-night candidates are:

`2026-06-04`, `2026-06-05`, `2026-06-06`, `2026-06-07`,
`2026-06-08`, `2026-06-09`, `2026-06-10`, `2026-06-11`,
`2026-06-25`, and `2026-06-26`.

The backup and Ivezić roots have:

- 160 versus 160 nightly science Parquet files;
- zero files unique to either root; and
- zero same-path checksum conflicts.

Historical comparison evidence was written to these retention paths. Their
results are preserved in this document even though the files had expired by
August 16:

- `/deleted-sundays/mdarim/rsp_candidate_old_md_home_comparison_20260803.txt`
- `/deleted-sundays/mdarim/rsp_candidate_backup_comparison_20260803.txt`
- `/deleted-sundays/mdarim/rsp_candidate_ivezic_comparison_20260803.txt`

## Scientific feature-coverage sanity check

Feature coverage was unchanged by the zero-row policy repair:

| Coverage metric | Finite rows |
| --- | ---: |
| `feature_chi2_magn_r` | 17,743 |
| `feature_standard_deviation_magn_r` | 17,743 |
| `feature_weighted_mean_magn_u` | 767 |
| `feature_weighted_mean_magn_g` | 9,980 |
| `feature_weighted_mean_magn_r` | 23,740 |
| `feature_weighted_mean_magn_i` | 133,822 |
| `feature_weighted_mean_magn_z` | 248,639 |
| `feature_weighted_mean_magn_y` | 592 |
| variability pair (`chi2_r`, `stddev_r`) | 17,743 |
| weighted-mean pair (`g`, `i`) | 2,090 |
| weighted-mean triple (`r`, `i`, `g`) | 1,866 |
| weighted-mean triple (`u`, `g`, `r`) | 140 |

The sparse plotting samples remain a stored-feature coverage property, not a
side effect of the zero-row policy change.

## Audit implementation provenance

The historical post-policy active-root audit used the hardened auditor at
SHA-256
`551b94ce462edda608f2adb719364cc829af29eb49e5034430bf8ea141e2c356`.
The older 55-night candidate was audited with the same version.

The current repository auditor was uploaded to
`/home/mdarim/antares_migration_tools_20260816/audit_antares_data_root.py`,
verified at SHA-256
`d1967e280f904129b7fa1d3e95218857a6cb67d4e099c729e294b90ecf850862`,
and used for the fresh source and Arnor staging audits. Both produced all ten
reports, returned exit 0, and reported no integrity issues. The independent
strict verifier was SHA-256
`ec747785ca69554035e65b657abfd60343ea28f264e64c12ac9f38118e31c569`.

The local zero-row validation tests pass, and the full local test selection
completed with 40 passing tests. Focused RSP tests completed with 7 passing
tests. No commit or push was made.

## Arnor staging and destination audit

Run ID: `20260816T153235Z`

| Role | Path |
| --- | --- |
| Private staging root | `/astro/store/shire/ANTARES_Analysis_Data.incoming_20260816T153235Z` |
| External audit run | `/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z` |
| Complete destination audit | `/astro/store/shire/ANTARES_Analysis_Data_migration_audits/20260816T153235Z/uw_destination_stage_complete` |
| Proposed final root | `/astro/store/shire/ANTARES_Analysis_Data` |

Arnor preflight confirmed the `mdarim` account, a writable Shire filesystem,
and 467 TB available. The final root and same-run staging path were absent.
The remote system Python lacked NumPy and PyArrow, so the audit run contains
an isolated virtual environment using Python 3.9.25, NumPy 2.0.2, and PyArrow
21.0.0. The science data and audit tools are not installed globally.

The RSP image did not provide a local `rsync` executable. The transfer
therefore used a GNU tar stream over the authenticated SSH ControlMaster into
a newly created mode-700 staging directory. It ran from 16:36:03Z through
16:36:54Z and returned zero. It copied:

- all of `data/`;
- all of root-level `analysis/`; and
- the valid seven-member recovery archive
  `skypulse_rsp_export_2026_06_03.tar.gz`.

The recovery archive has SHA-256
`6e433de0ee86ccce82a9442d3224b711366a22998ae03b28b701f4b47c8e605d7`.
The complete non-cache source manifest contains 324 files and has SHA-256
`594ced3dfc7ed2d36f1e5562defabb6e9475f2a72f6ebeab54c09020f6db5908`.
All 324 staged files passed `sha256sum --check`. The auditor's 304-file
science subset also matched exactly.

The complete staging audit and independent strict verifier both passed with:

- 90 complete and append-ready nights;
- 2026-02-25 through 2026-06-26;
- 993,218 loci and 13,579,707 alerts;
- both zero-row nights readable, schema-valid, 0/0, and error-free;
- 30 root-level `analysis/` science files;
- identical cumulative checksums and feature coverage; and
- zero integrity issues.

The source/destination comparator matched all machine-readable manifest,
checksum, and feature reports, 30 scientific summary fields,
`file_counts.json.science_products`, and the complete 324-file non-cache
inventory of 1,141,241,743 bytes. The two human-readable summaries differed
only in their expected `Data root:` header; after excluding that one
environment-specific line, both matched exactly. The destination contains no
cache files.

## Migration gate

The source-integrity, durable-transfer, destination-audit, and exact-comparison
gates are clear. Work is intentionally stopped before promotion. The proposed
final path is still absent, the verified staging root remains mode 700 and
owned by `mdarim:mdarim`, and the active RSP source and cache remain intact.

Promotion requires a separate user decision on final ownership/group/access
and explicit authorization for the atomic same-filesystem rename. Cache copy
versus rebuild remains a separate decision. No command used `--delete`, no
existing destination was overwritten, and no source data was removed.
