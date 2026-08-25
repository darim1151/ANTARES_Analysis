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

1. ANTARES owns exactly one top-level Shire namespace,
   `/astro/store/shire/ANTARES`. Durable migrated data is beneath `data/`,
   migration evidence is beneath `migration_audits/`, operational work is
   beneath `work/`, and the separate rebuildable cache is reserved at `cache/`.
   No project code may create a top-level `/astro/store/shire/ANTARES_*` sibling.
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
| `doctor` | Checks Python, storage, policy, dataset layout, dependencies, optional checkout notebooks, and Jupyter discovery | None |
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
- An installed runtime without a discoverable checkout reports notebook checks
  as informational; an explicit or discovered checkout remains fully validated.
- Rendered shell commands quote spaces and shell metacharacters.
- Tests cover human help, JSON, overrides, environment selection, missing data,
  malformed manifests, private permissions, path safety, no-write behavior,
  notebook discovery, command rendering, and invalid requests.
- The full Python suite, source compilation, build, installed-wheel smoke, and
  exact staged-diff review pass before commit.

## Phase 3 — operations architecture and writer-contract foundation

Status: complete locally; its read-only Arnor acceptance gate was satisfied in
Phase 4, while the production writer remains intentionally disabled.

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
  manifest-last staging, proof-only directory promotion, and pre-publication
  abort behavior restricted to local temporary fixtures; Phase 4 superseded
  the promotion primitive with ADR-0002's empirically qualified contract;
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

With the read-only Arnor acceptance gate now satisfied by Phase 4, the original
guarded-engine proposal remains the next local implementation work:

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
7. Publish with the empirically qualified Shire manifest-commit protocol and a
   journal that makes interruption recovery deterministic.
8. Add idempotent resume and evidence-backed rollback that cannot delete a
   pre-existing valid night.

Local foundation exit gate: context/report/state/planner/storage/lock/transaction
tests; failure injection; zero-row regressions; full existing suite; reproducible
build; installed-wheel import and CLI smoke; and exact diff review.

Arnor acceptance gate: verify root containment, private ownership, disk/inode
observability, lock-directory atomicity, same-filesystem staging, publication
primitives, durability calls, and interruption semantics before any writer
activation.

Transactional writer exit gate: concurrent-writer, stale-lock, network failure,
partial fetch, disk full, SIGINT, corrupt stage, duplicate run, recovery, and
manifest-commit failure tests all pass in temporary and explicitly authorized
synthetic canary stores. Production execution remains a later authorization.

## Phase 4 — Arnor qualification and environment hardening

Status: filesystem and runtime qualification complete in evidence run
`phase4-20260823T103930Z`; production activation remains disabled.

Delivered gates:

- exact immutable installed-release execution on Arnor;
- read-only production inventory, checksum, permission, ACL, CLI, and planner
  acceptance;
- real NFS mount/device/capacity/inode/quota characterization;
- controlled Shire canary rename, fsync, contention, process-death, containment,
  failure-injection, zero-row, and recovery-state tests;
- empirical rejection of race-prone plain rename and unsupported
  `renameat2(RENAME_NOREPLACE)`;
- empirical qualification of atomic target reservation, same-filesystem hard
  links, and manifest-last logical commit;
- installed-runtime `doctor` hardening without weakening explicit-checkout
  validation;
- plan schema `1.1`, which reports the production operations/lock root as
  explicitly unconfigured instead of exposing a superseded in-data-root path;
- preservation of the writer-disabled CLI and production/cache boundaries.

The authoritative filesystem/publication decision is
`docs/architecture/ADR-0002-arnor-filesystem-qualification.md`.

## Phase 5 — transactional writer and recovery implementation

Goal: implement the proven contract locally and in synthetic canaries without
exposing a production write capability.

Execution order:

1. Persist an immutable, versioned run plan with science/configuration,
   dependency, release, target, baseline, and resource fingerprints.
2. Model a separately configured operations root on the qualified Shire mount;
   do not provision its production path yet.
3. Configure an operations root explicitly, unify planner, writer, journal, and
   recovery lock identity, and add durable, PID-reuse-safe lock metadata. The
   read-only planner reports the production lock location as unconfigured until
   then; its in-data-root path is legacy test-only.
4. Stream and fsync staged artifacts, write the manifest last, and independently
   validate every scientific and query/fetch invariant.
5. Implement descriptor-pinned atomic final-target reservation, re-prove
   validated data hard links, then commit via a verified pending-manifest inode
   linked to `manifest.json` with ADR-0002's fsync and ambiguity rules.
6. Persist every transition and distinguish uncommitted partial targets from
   committed-but-durability-uncertain and published-unreconciled states.
7. Implement a read-only recovery inspector before any destructive recovery
   action. No stale, ambiguous, or canonical-target state is auto-deleted.
8. Add independent post-commit verification and idempotent derived
   reconciliation under its own lock.
9. Exercise every crash boundary, short write, disk/inode exhaustion, malformed
   lock, target race, link/fsync failure, SIGINT, zero-row outcome, and
   reconciliation failure using synthetic data.
10. Build and deploy an immutable writer-disabled Arnor release and repeat
    installed-package, CLI, Jupyter-import, canary, and production-sentinel
    acceptance.

Exit gate: all local and synthetic-Arnor writer/recovery tests pass; no ANTARES
query, production operations/cache-root creation, science publication,
reconciliation, scheduler, or production writer command is enabled.

## Phase 6 — Middle Earth execution and Jupyter integration

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
Middle Earth without changing durable data except through a separately
authorized Phase 5 transaction.

## Phase 7 — science workflow and UX completion

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

## Phase 8 — controlled production rollout

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

## Immediate next decision after Phase 4

Phase 5 begins with the durable journal and read-only recovery inspector, not a
production writer command. No durable science write is enabled until a plan can
be reviewed, reproduced, locked, staged, validated, manifest-committed,
independently revalidated, and recovered under every qualified failure boundary.
