# ADR-0001: Operations Architecture and Writer Contracts

Status: accepted for local Phase 3 development. Production writing remains
disabled.

## Decision

ANTARES Analysis has one reusable operations layer between user interfaces and
the existing science/storage implementations:

```text
              CLI / Jupyter
                    |
                    v
             src.operations
        context / report / plan
        state / validation
        storage / lock / transaction
                    |
          +---------+---------+
          |                   |
    existing science APIs   explicit profiles
          |
   validated storage/manifests
```

The CLI is an adapter. Notebooks call Python operations directly and are not a
second implementation path. Validated nightly storage and manifests remain the
scientific authority; CLI state, notebook memory, and cache are not scientific
truth.

## Contracts

- `OperationContext` captures a resolved profile, roots, policy, run identity,
  clock, metadata, and logger once. Construction and import do not create paths.
- `OperationReport` schema `1.0` has fixed fields, deterministic JSON encoding,
  stable issues/evidence/actions, human rendering, and explicit exit classes.
- Planning is read-only. It does not create a lock or directory, touch cache,
  query ANTARES, write a manifest, alter mtimes intentionally, or estimate
  values that are not knowable before a query.
- Writer states are explicit: `PLANNED`, `PRECHECKED`, `LOCKED`, `QUERYING`,
  `FETCHING`, `STAGED`, `VALIDATED`, `PUBLISHED`, `RECONCILING`, `COMPLETE`, and
  `FAILED`. Illegal transitions raise.
- Paths are root-relative and contained after symlink resolution. `..`,
  absolute injection, symlink escape, and durable/cache overlap fail closed.
- Current science files are manifest siblings. Embedded absolute manifest paths
  are provenance and never redirect current reads.
- A future writer must hold a single-writer lock, stage a complete partition on
  the publication filesystem, preserve query/fetch failures, validate before
  publication, refuse overwrite, publish atomically, release the nightly lock,
  and only then reconcile derived products.
- Lock age is evidence, not permission to steal. Only the acquiring operation
  may release a lock. `/astro/store/shire` lock and rename semantics require
  explicit read-only Arnor acceptance before production activation.
- A failed reconciliation never deletes or invalidates an already published
  nightly partition. It produces published science with reconciliation still
  required.
- Zero-row science requires a completed, error-free query/fetch workflow and
  explicit schema/validation evidence. An exception or swallowed fetch failure
  can never become an append-ready empty night.
- Cache remains optional, rebuildable, and non-authoritative. Existing distinct
  cache namespaces and keys are unchanged.

## Authorization boundary

The only executable publication/lock proof in this phase requires a
`DevelopmentWriteCapability` restricted to an existing child of the local
temporary directory. There is no production capability constructor and no CLI
writer command. `night plan` and `backfill plan` report
`writer_not_enabled_in_this_release`.

## CLI and automation

Phase 2 codes remain unchanged: `0` success, `1` failed health/validation gate,
and `2` invalid request/configuration. Operations reserve `3` for operational
environment failure, `4` for authorization/refusal, and `5` for unexpected
internal failure. Read-only planning succeeds with code `0` when it produced a
valid plan; execution refusal is explicit inside the report.

## Notebook debt

`historical_backfill.ipynb` and `alerts_time_comparison.ipynb` still contain
authoritative-looking orchestration and direct publication/reconciliation
calls. They were not rewritten in Phase 3 because preserving accepted science
behavior takes priority. Before a production writer is enabled, that logic must
move behind the operations contracts and notebooks must become supervised
clients rather than publishers.

## Consequences and next gates

This foundation supports Jupyter with:

```python
from src import operations

ctx = operations.context_from_environment()
report = operations.plan_night(ctx, "2026-06-27")
```

Production ingestion, lock placement, stale-lock recovery, resumability, and
production execution remain disabled. Phase 4's Arnor evidence and the
manifest-commit publication decision superseding the proof-only rename detail
are recorded in `ADR-0002-arnor-filesystem-qualification.md`. The next gate is
local transactional-writer and recovery implementation under synthetic-only
authority; production activation remains separately authorized.
