# ANTARES Analysis CLI Phased Roadmap

## Mission

Build one safe, predictable command-line control plane for ANTARES Analysis on
Middle Earth. The CLI must make migrated data, Jupyter workflows, environment
diagnostics, and future scheduled operations easy to discover without hiding
scientific or operational state.

This roadmap is limited to that mission. Public websites, SkyPulse, unrelated
data products, and unrelated deployment work are outside scope.

## Non-negotiable contracts

Every phase preserves these rules:

1. `/astro/store/shire/ANTARES_Analysis_Data` is durable migrated data, and
   `/astro/store/shire/ANTARES_Analysis_cache` is a separate rebuildable cache.
2. Middle Earth defaults to private storage. No command silently changes owner,
   group, ACLs, permission policy, or cache placement.
3. Read-only commands never create paths or update access/content timestamps by
   opening project files for writing.
4. A command fails closed when configuration, data integrity, authentication,
   or publication validation is ambiguous.
5. Human output remains concise; `--json` provides stable automation output.
6. Exit codes are consistent: `0` success, `1` completed health/validation gate
   failed, `2` invalid request/configuration, and a later reserved code for
   interrupted or locked writer operations.
7. Writer operations require explicit intent, a dry-run plan, a lock, a unique
   run identity, staged output, complete validation, and atomic publication.
8. No phase is complete until source tests, installed-wheel tests, failure-mode
   tests, documentation, and the relevant Middle Earth acceptance run pass.

## Phase 1 — portability and release foundation

Status: complete in the preceding local commit.

Delivered package metadata, a lightweight package initializer, explicit Python
compatibility, release validation, storage configuration, migration evidence,
and the initial safe console entry point. This phase established the boundary
that scientific dependencies and storage configuration are not imported merely
to display CLI metadata.

## Phase 2 — read-only Middle Earth control plane

Status: complete in local commit `c74977c`.

### User surface

| Command | Result | Writes or process launch |
| --- | --- | --- |
| `profile list` | Lists built-in Middle Earth and RSP compatibility profiles | None |
| `profile show` | Resolves auto, environment, or explicit configuration | None |
| `profile export` | Renders copy/paste-safe environment variables | None |
| `doctor` | Checks Python, storage, policy, dataset layout, dependencies, notebooks, and Jupyter discovery | None |
| `data status` | Summarizes nightly manifests and cumulative-product presence | None |
| `jupyter list` | Discovers supported notebooks in a checkout | None |
| `jupyter env` | Renders the environment a notebook process should inherit | None |
| `jupyter command` | Renders a shell-safe Jupyter Lab command | None; it is never executed |

All storage-aware commands support explicit profile, data-root, cache-root,
policy, and shared-group selection. All health/inventory commands return a
non-zero status when required conditions fail.

### Phase 2 exit gate

- CLI startup imports only Python standard-library modules.
- Default Middle Earth paths and private policy exactly match the migration.
- Manifest inventory validates dates, directory membership, statuses, counts,
  append-ready types, duplicate dates, cumulative products, and cache separation.
- Diagnostics never create a missing cache or data root.
- Rendered shell commands quote spaces and shell metacharacters.
- Tests cover human help, JSON, overrides, environment selection, missing data,
  malformed manifests, private permissions, path safety, no-write behavior,
  notebook discovery, command rendering, and invalid requests.
- The full Python suite, source compilation, build, installed-wheel smoke, and
  exact staged-diff review pass before commit.

## Phase 3 — operations architecture and writer-contract foundation

Status: complete locally; production writer intentionally disabled pending
read-only Arnor acceptance.

Delivered foundation:

- immutable/effectively immutable operation context with one-time profile
  resolution, explicit clock/logger/run metadata, and no path creation;
- versioned unified operation reports with deterministic JSON and human output;
- explicit writer state machine and legal transitions;
- side-effect-free `night plan` and inclusive sequential `backfill plan`;
- contained storage layout, cache separation, sibling-manifest science paths,
  and symlink/path-injection refusal;
- conservative single-writer lock ownership and stale-lock inspection contracts;
- same-filesystem staging, validation-before-publication, overwrite refusal,
  manifest-last staging, atomic directory promotion, and pre-publication abort
  behavior restricted to local temporary fixtures;
- strict query/fetch and zero-row evidence contracts;
- publication preserved when separate derived reconciliation fails;
- Jupyter-facing Python API shared with the CLI.

Current supported surface:

```text
antares-analysis night plan YYYY-MM-DD
antares-analysis backfill plan START END
```

Both commands are read-only. Reports explicitly state
`writer_not_enabled_in_this_release`; there is no production writer command or
capability.

The original guarded-engine proposal remains the next implementation work,
after read-only Arnor acceptance:

Proposed surface:

```text
antares-analysis data plan last-night
antares-analysis data plan backfill --from YYYY-MM-DD --to YYYY-MM-DD
antares-analysis data run PLAN_ID
antares-analysis data resume RUN_ID
antares-analysis data verify RUN_ID
antares-analysis data rebuild-index --dry-run
```

Future execution order:

1. Extract pure planning and validation functions from notebooks; do not expose
   writes yet.
2. Define a versioned immutable run-plan schema containing profile, query
   bounds, source identity, dependency/configuration fingerprint, expected
   targets, output paths, and plan hash.
3. Implement a data-root lock with owner, host, PID, run ID, start time, and
   stale-lock inspection. Never auto-break an ambiguous lock.
4. Stage every nightly/cumulative artifact outside its published location.
5. Propagate every query/fetch error; incomplete or saturated states are explicit
   scientific outcomes, never silent success.
6. Validate schema, counts, coordinates, locus/alert linkage, LSST association,
   manifest paths, checksums, and cumulative reconciliation before publication.
7. Publish with same-filesystem atomic renames and a journal that makes
   interruption recovery deterministic.
8. Add idempotent resume and evidence-backed rollback that cannot delete a
   pre-existing valid night.

Local foundation exit gate: context/report/state/planner/storage/lock/transaction
tests; failure injection; zero-row regressions; full existing suite; reproducible
build; installed-wheel import and CLI smoke; and exact diff review.

Arnor acceptance gate: verify root containment, private ownership, disk/inode
observability, lock-directory atomicity, same-filesystem staging, atomic rename,
and interruption semantics read-only before any writer activation.

Transactional writer exit gate: concurrent-writer, stale-lock, network failure, partial fetch, disk
full, SIGINT, corrupt stage, duplicate run, resume, and rollback tests all pass
in temporary stores; a dry-run on Middle Earth matches the expected plan; one
authorized canary night passes validation before broader use.

## Phase 4 — Middle Earth execution and Jupyter integration

Goal: offer one interface for interactive and scheduled execution while keeping
platform-specific details behind inspected adapters.

Proposed surface:

```text
antares-analysis runtime probe
antares-analysis job submit PLAN_ID
antares-analysis job status JOB_ID
antares-analysis job logs JOB_ID
antares-analysis job cancel JOB_ID
antares-analysis jupyter open NOTEBOOK
antares-analysis jupyter kernel install --user
```

Execution order:

1. Record the supported Middle Earth host, scheduler/service, authentication,
   resource, filesystem, and network contracts from authoritative platform
   evidence.
2. Define a runtime adapter so local foreground execution and Middle Earth jobs
   share the same plan, validation, logging, and exit semantics.
3. Add resource estimates and explicit CPU, memory, wall-time, and concurrency
   controls. Reject unsupported resource requests before submission.
4. Store job IDs and immutable provenance outside the durable science tree.
5. Stream or retrieve logs with secrets and host-private paths redacted from any
   shareable export.
6. Add an explicit Jupyter launcher that shows the exact profile, kernel,
   notebook, URL-binding behavior, and working directory before execution.
7. Install kernels idempotently and verify the kernel uses the accepted Python
   environment and CLI version.

Exit gate: probe, submit, status, log, cancellation, authentication failure,
resource rejection, reconnect, kernel identity, and notebook-open flows pass on
Middle Earth without changing durable data except through an authorized Phase 3
transaction.

## Phase 5 — science workflow and UX completion

Goal: make common analysis tasks discoverable, composable, and observable.

- Add command aliases and guided prompts only where they map to the same
  underlying plan schema; automation never depends on prompts.
- Add `--explain`, `--dry-run`, progress, structured logs, run history, and
  machine-readable schemas with compatibility tests.
- Expose read-only comparison and feature-analysis commands that produce
  provenance-bearing artifacts in an explicit analysis-output root.
- Add completion scripts, concise examples, troubleshooting, and operator
  runbooks for first use, daily use, incidents, and recovery.
- Add performance budgets for startup, manifest inventory, planning, and
  large-run memory use; optimize only against recorded profiles.

Exit gate: representative new users can discover profiles, diagnose setup,
open the correct notebook, plan a run, inspect its status, and recover a failed
run from the CLI documentation alone. JSON compatibility and performance budgets
pass in CI and on Middle Earth.

## Phase 6 — controlled production rollout

Goal: convert the validated CLI into a supportable service/operator workflow.

1. Produce hashed Linux locks and an offline wheelhouse for the accepted Middle
   Earth Python/runtime target.
2. Sign or checksum release artifacts and record source commit, build evidence,
   environment fingerprint, and schema versions.
3. Run shadow mode, one-night canaries, and monitored expansion with explicit
   stop/rollback criteria.
4. Define ownership, alerts, retention, backup/restore drills, incident response,
   upgrade/rollback procedure, and compatibility windows.
5. Promote only after science reconciliation and operational service-level gates
   pass for the agreed observation period.

## Immediate next decision after Phase 2

Phase 3 begins with the immutable run-plan and locking contracts, not with a
writer command. The first implementation milestone should end at a fully tested
`data plan`/`--dry-run` path. No durable Middle Earth write should be enabled
until that plan can be reviewed, reproduced, locked, staged, validated, resumed,
and rolled back under failure injection.
