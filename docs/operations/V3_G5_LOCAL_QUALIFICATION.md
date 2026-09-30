# V3-G5 local qualification and later Control gate

This change implements range acquisition and publication derivation. It does
not authorize or execute a production canary. Qualification uses temporary
fixtures, including production capabilities whose roots are explicitly mapped
to temporary directories. No ANTARES request, Arnor access, production/cache
namespace creation, production publication, push, tag, or release occurred.

Baseline: `0226753288de0dd1a1f917ad8aa735e764d2c534`, branch
`codex/feature-space-diagnostics`, tag `v0.4.4`; clean before editing. HEAD and
tag remain unchanged. Implementation changes are uncommitted. Package version
remains `0.4.4`; source hashes and the later Control-approved wheel/release
identities distinguish this local candidate. No release artifact was changed.

## Resulting architecture

- Canonical UTC nights derive integer MJD midnight bounds from the civil date
  and MJD epoch. Live requests must match the exact `[min, max)` day, untagged
  exhaustive LSST selection and their sealed capability's night.
- The accepted sequential probe-first time/RA/Dec extractor remains intact:
  50-row saturation discards provisional rows, deterministic binary splitting
  preserves the frontier, and only exhausted sub-threshold tiles contribute.
- `QueryProgress` adds versioned per-night, non-authoritative durability below
  `checkpoints/query-progress-v1`. Request, science, implementation, client,
  configuration, initial tiling and release identities must match on reopen.
- Hash-chained events record terminal tile records, split decisions and retry
  evidence. An atomic head binds the committed sequence, detecting tail loss
  as well as corruption, missing interior events and reference drift.
- Existing `query-result` sealing and `SegmentedFetchCheckpoint` remain the
  acquisition boundaries. Fetch uses the existing bounded per-object method;
  construction reopens completed segments through the read-only interface.
- `RangeWorkCapability` owns only the explicit
  `/astro/store/shire/ANTARES/work/backfill/<run>` root. `PublicationRoots`
  supplies observation addresses. A production publication capability cannot
  be passed as a backfill work capability.
- Acquisition is bounded across independent nights. Construction chains prior
  loci; publication has one writer in chronological order. The writer waits
  for brief overlapping construction reads under the existing authority lock.
- `ProductionRangeAuthorization` binds at most 31 nights, exact work root,
  initial Sentinel, scientific/configuration identities, reviewed prior-free
  provider/adapter, all installed Python source hashes, releases, wheel,
  host/service UID, expiry and Control-token digest. Shared cache is disabled.
- Each candidate receives the existing exact publication authorization. A new
  one-shot production capability is derived for only that candidate/night and
  its verified COMPLETE predecessor. Existing Sentinel V2, gate, journal,
  reconciliation and single-success rules remain in force.
- The operator module provides offline planning, read-only inspection,
  explicitly gated execution/resume, and recovery limited to an existing
  journaled publication. It never provisions infrastructure.

## Exact resume behavior

An event is written to a new private file, fsynced, linked exclusively under its
sequence name, and the directory is fsynced. The head is atomically replaced
and fsynced. A valid linked event stranded before the head update can be
verified and rolled forward on reopen; orphan temporary files are ignored.
Both replay and the independent artifact verifier reconstruct exact tile
lineage. Verified terminal tiles and committed saturated parents are skipped.
An interrupted tile contributes no partial rows and can be queried again.
The final query seal requires complete coverage, no pending tiles and normal
exhaustion of every accepted iterator. `keep='last'` deduplication, row order,
column order, and nested missing-versus-null semantics are preserved.

A durably recorded retry resumes at the next permitted attempt. Malformed or
exhausted retry evidence remains blocked; resume does not invent extra
attempts beyond the accepted policy. Process interruptions alone never mark a
tile complete. Operational timestamps/runtime describe the resumed invocation;
deterministic fixtures with a fixed clock reproduce the full query evidence.

Fetch recovery reopens the sealed ordered query and existing segment receipts.
Only absent segments invoke the provider. Read-only construction requires an
already-complete ledger and invokes no provider callbacks. A completed range
resume performs neither query nor fetch again.

## Exact production authority behavior

The range authorization is permission to derive narrow authorities, not a
production write capability. Its token must match its digest, host/UID, exact
immutable release interpreter, release/wheel markers and installed sources.
Science and configuration must match the supported live adapter before reads.
Issuance verifies the candidate's range/configuration/science provenance,
candidate record and artifact hashes, exact night/predecessor, releases,
nonce/expiry, roots, authority-lock identity and current Sentinel binding.
The first night must match the range's initial Sentinel; later nights must
match their COMPLETE predecessor's resulting fingerprint. A blocked or
unfinished predecessor stops publication even when later fetches are complete.

Each per-night binding and its baseline Sentinel are persisted under the work
root before issuance. Token-free recovery requires the exact matching durable
gate/journal, or the already-committed matching journal. Pre-gate residue is
insufficient. `recover --resume` can finish that transaction after expiry but
cannot query, fetch, reconstruct a candidate or initiate another publication.
Repeated resume is idempotent. Historical June-27 bindings omit the new optional
range field, preserving their original digests and G4 recovery contract.

Canonical publication roots remain `data` and `work/publication/{staging,
control,evidence}`. `/astro/store/shire/ANTARES/cache` must remain absent.
The optional `work/segment-cache` remains inactive and uncreated.

## CI triage and verification

All three workflow files were inspected; none was changed. Web CI runs its own
web checks; Python CI and Linux-lock both run repository unittest discovery.
The baseline local run reported 411 tests with two failures and two errors:

1. The distribution test fixture omitted the now-required `MANIFEST.in` from
   its source and archive. This is a reproducible test defect, repaired in the
   fixture without weakening the distribution verifier.
2. Backfill's publication could refuse the controller's own overlapping
   construction reader with `publication_lock_busy`. The race was separately
   reproduced from durable temporary range evidence. Bounded lock waiting
   repairs it using the existing single-writer lock.
3. The available local release smoke environment lacked `nbformat`. Its exact
   `5.10.4` installation and dependencies were supplied from another existing
   local Python 3.9 environment through `PYTHONPATH`; nothing was downloaded.

The GitHub statuses in the request are user-supplied evidence. No GitHub API or
network was accessed. Reproducible Python defects exist, but the specific
push-run failures with no job evidence cannot be attributed to them. An
external workflow/platform/startup cause remains unresolved. No scientific
code was changed to conceal an external CI status.

Local Python is CPython 3.9.25, a supported repository compatibility lane.
Linux x86_64 / CPython 3.11.16 locked installation, Linux lock regeneration,
and external Actions startup were not reproduced offline on this macOS host.
The later installed-release gate still requires exact CPython 3.11.16.

Commands were run from the repository root. The interpreter was
`release-artifact/0.4.2/committed-smoke/bin/python`, with that environment's
site-packages preceding the existing `Documents/Fink Analysis/.venv` packages.
Final qualification added a temporary `sitecustomize` that rejects socket
connection and DNS calls in Python subprocesses, and configured data/cache
roots to absent `/private/tmp/antares-g5-forbidden-{data,cache}` paths.

```text
python -m unittest discover -s tests -v                 # baseline triage
python -m unittest test_operations_phase6                # 58 passed
python -m unittest test_backfill_v3 test_publication_authority_v3
                   test_fetch_checkpoint test_phase6_query_checkpoint
                   test_python_distributions            # focused iteration
python -m unittest test_production_range_g5 -v           # 18 passed
python -m compileall -q src scripts tests                # passed
git diff --check                                        # passed
python -m unittest discover -s tests -v                 # final result below
```

SIGKILL qualification covers pre-split, post-split, post-terminal and in-flight
tile death. Additional tests cover retry resumption, exact fixture equivalence,
idempotence, corruption/ref/tiling drift, tail truncation, date/interval rejection,
canonical two-night production-capability chaining under temporary roots,
blocked predecessors, distinct one-shot identities and token-free recovery.
Existing fetch-segment and publication crash suites remain part of the complete
suite. Intermediate fixture failures (column order, read-only API routing,
synthetic-versus-live schema and timestamp ordering) were corrected; production
schema and chronology checks were preserved.

Final suite: **445 tests passed in 110.990 seconds**, with the network guard
enabled. Compilation, whitespace validation and absent configured data/cache
root checks passed. There are no unresolved local test failures. The macOS
sandbox emits Arrow CPU-probe warnings; Astropy warns about distant fixture
years. Neither warning is a test failure. Raw local logs are in
`/private/tmp/antares-g5-*.log`, including `antares-g5-baseline.log`,
`antares-g5-authority6.log` and `antares-g5-full.log`.

Final working tree: six tracked files modified, four new files, no commit;
HEAD remains the baseline SHA. Changes:

| File | Purpose |
| --- | --- |
| `src/operations/live_antares.py` | Date-aware live contract, durable tile replay, shared existing segmented fetch body and read-only reconstruction. |
| `src/operations/science.py` | Date-aware independent artifact interval/domain and canonical traversal verification; scientific rules retained. |
| `src/operations/query_progress.py` (new) | Versioned private, hash-bound event journal and semantic record codec. |
| `src/operations/backfill.py` | Explicit work/publication boundaries, progress-aware acquisition, range provenance, bounded lock wait and recovery-only mode. |
| `src/operations/storage.py` | Separate sealed work capability and read-only publication addresses. |
| `src/operations/publication.py` | Optional exact range binding and narrow per-night issuance, preserving legacy G4 digests. |
| `src/operations/production_range.py` (new) | Bound range authorization, prior-free per-night adapter, one-shot publisher derivation and operator commands. |
| `tests/test_production_range_g5.py` (new) | 18 focused local qualification tests, including real SIGKILL and mapped production-capability flows. |
| `tests/test_python_distributions.py` | Repair the baseline fixture's missing manifest input. |
| This document (new) | Qualification evidence and proposed later commands. |

The exact final discovery command was:

```sh
PYTHONPATH='/private/tmp/antares-g5-no-network:/Users/darim_shamsi/Documents/ANTARES REVAMP/ANTARES_Analysis/release-artifact/0.4.2/committed-smoke/lib/python3.9/site-packages:/Users/darim_shamsi/Documents/Fink Analysis/.venv/lib/python3.9/site-packages' \
ANTARES_ANALYSIS_DATA_ROOT=/private/tmp/antares-g5-forbidden-data \
ANTARES_ANALYSIS_CACHE_ROOT=/private/tmp/antares-g5-forbidden-cache \
ANTARES_STORAGE_POLICY=private MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/antares-g5-mpl \
release-artifact/0.4.2/committed-smoke/bin/python \
  -m unittest discover -s tests -v > /private/tmp/antares-g5-full.log 2>&1
```

The focused G5 command used the same environment plus `:tests` in
`PYTHONPATH`, with `-m unittest test_production_range_g5 -v` and output to
`/private/tmp/antares-g5-authority6.log`. The focused existing-suite command
listed above ran 122 tests during iteration; its instrumentation compatibility
failure was fixed and both existing concurrency tests passed on rerun. The
complete final discovery includes all those tests. Existing synthetic cache
tests created only disposable temporary fixture caches; production cache and
shared-cache activation never occurred.

## Later canary commands — NOT EXECUTED

These are proposed commands for a separately approved installed release.
`G5_PYTHON`, `G5_WORK`, `G5_SHA`, `G5_WHEEL`, `G5_AUTHORIZATION` and
`G5_CONTROL_TOKEN_FILE` must be resolved by Control. The work root must be an
existing canonical `work/backfill/<run>` directory, outside publication roots.
Control must provision/qualify infrastructure separately; these commands do not.

```sh
# NOT EXECUTED: offline plan
"$G5_PYTHON" -m src.operations.production_range plan \
  --start 2026-06-28 --end 2026-06-29 \
  --work-root "$G5_WORK" --candidate-release "$G5_SHA"

# NOT EXECUTED: inspect existing evidence and authority
"$G5_PYTHON" -m src.operations.production_range inspect \
  --start 2026-06-28 --end 2026-06-29 \
  --work-root "$G5_WORK" --candidate-release "$G5_SHA"

# NOT EXECUTED: explicitly authorized acquisition and ordered publication
"$G5_PYTHON" -m src.operations.production_range execute \
  --start 2026-06-28 --end 2026-06-29 \
  --work-root "$G5_WORK" --candidate-release "$G5_SHA" \
  --publisher-wheel-sha256 "$G5_WHEEL" \
  --authorization "$G5_AUTHORIZATION" \
  --control-token-file "$G5_CONTROL_TOKEN_FILE" \
  --acquisition-concurrency 2 --execute-authorized-range

# NOT EXECUTED: same execute command with --resume reuses existing acquisition.

# NOT EXECUTED: only recover an already-journaled publication, without a token
"$G5_PYTHON" -m src.operations.production_range recover \
  --start 2026-06-28 --end 2026-06-29 \
  --work-root "$G5_WORK" --candidate-release "$G5_SHA" \
  --publisher-wheel-sha256 "$G5_WHEEL" \
  --authorization "$G5_AUTHORIZATION" \
  --execute-authorized-range --resume
```

Recommended next Control gate: independent source/evidence review; freeze the
candidate SHA and wheel; qualify the installed Linux 3.11.16 release; capture
fresh Sentinel V2 proving June 27 COMPLETE, exact authority tail and absent
cache; approve the exact two-night authorization, service UID, roots, expiry
and token digest. Only that later gate may authorize the live canary.
