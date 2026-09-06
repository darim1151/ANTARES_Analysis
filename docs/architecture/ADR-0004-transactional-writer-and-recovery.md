# ADR-0004: Transactional Writer and Durable Recovery Contract

Status: accepted for the Phase 5 synthetic-only candidate. Production
authorization, live ANTARES access, production reconciliation, cache creation,
and scheduling remain unavailable.

## Decision

ANTARES Analysis has one nightly-writer coordinator in
`src.operations.writer`. CLI, Jupyter, tests, and an eventual scheduler must be
adapters over this operations implementation, not independent publishers.

The Phase 5 executable path accepts only:

1. an exact, sealed `SyntheticWriteCapability`; and
2. the exact deterministic `SyntheticScienceProvider` type.

Subclasses, the legacy development capability, arbitrary providers, and normal
configuration cannot cross this boundary. No production-capability factory or
live-provider adapter exists. `history.ingest_night`, the notebook-era function
that combined live query and direct publication, raises before it validates a
path, invokes a provider, or writes a file.

## Architecture

```text
CLI / Jupyter / future scheduler
               |
               v
        src.operations API
               |
       plan and preflight
               |
   transactional night writer
       /       |        \
  journal     lock     science adapter
       \       |        /
        staged validation
               |
       pre-commit reproof
               |
 manifest-last publication
               |
 independent canonical reopen
               |
   separate reconciliation lock
               |
            complete

read-only journal + physical evidence --> recovery inspector
```

## Synthetic run layout

Local qualification uses an existing direct child of the operating-system
temporary directory. Arnor qualification uses exactly one existing direct
child of:

```text
/astro/store/shire/ANTARES/work/canary/<RUN_ID>/
  .canary-root.json
  published/
    data/lsst_only/nightly/YYYY/MM/DD/
    derived/nightly/
  staging/<TRANSACTION_ID>/<TARGET_LOCK_ID>/
  control/
    journals/<TRANSACTION_ID>.json
    locks/<TARGET_LOCK_ID>.lock/owner.json
  evidence/
```

Authoritative production science remains at
`/astro/store/shire/ANTARES/data`. Operational journals never live inside a
nightly science partition. The reserved canonical cache path remains absent.

## Lifecycle and state legality

The enforced lifecycle is:

```text
PLANNED -> PRECHECKED -> LOCKED -> QUERYING -> FETCHING -> STAGED
        -> VALIDATED -> PUBLISHED -> RECONCILING -> COMPLETE
```

Every nonterminal state may transition to `FAILED`. Illegal transitions raise
immediately. A failure after `PUBLISHED` retains a durable published flag and
requires reconciliation. A generic failure status cannot erase the publication
boundary.

## Journal

Journal schema `1.0` is deterministic UTF-8 JSON. Creation uses an exclusive
temporary file and hard-link no-clobber. Updates write a new private file,
`fsync` it, atomically replace the prior journal, and `fsync` the parent.
Readers reject duplicate JSON keys, malformed UTF-8/JSON, unknown fields,
unknown schema/state/outcome values, illegal transition histories, inconsistent
publication flags, and outcome/evidence disagreement.

The journal retains transaction, plan, release, configuration, provider,
target-night, target/stage/lock/operations paths, transition history, expected
artifacts, device/inode/size/mode/SHA-256 identities, query/fetch and science
validation, publication attempt and boundary, durability status,
reconciliation status, recovery metadata, and structured failure evidence. It
contains no credential. It is operational evidence, not scientific truth.

Publication-aware outcomes are:

- `ACTIVE`
- `UNPUBLISHED_FAILURE`
- `PUBLISHED`
- `PUBLISHED_DURABILITY_UNCERTAIN`
- `PUBLISHED_RECONCILIATION_REQUIRED`
- `COMPLETE`

## Lock ownership

Writers acquire a target-specific lock through atomic private-directory
creation. Metadata contains canonical root and target, transaction/run,
release, plan/config and artifact hashes, effective user/UID, host, PID,
process-start identity, acquisition time, and an ownership token. Metadata and
lock directories are fsynced.

Release re-proves directory identity, exact metadata, ownership token, and
directory contents. Liveness requires host, PID, and process-start identity.
PID existence alone is insufficient. Stale and ambiguous locks are never
stolen or removed by the writer or recovery inspector. Only single-host Arnor
writing is qualified.

## Query and fetch

The synthetic adapter preserves the established LSST-only preparation and
validation functions from `src.history`. Query and fetch are separate stages
with structured completion evidence. Query failure/interruption, fetch failure,
partial fetch, malformed results, and validation failure are nonpublishable.

A valid zero-row result requires completed clean query and fetch stages, zero
rows in both tables, the explicit `completed_successful_query` proof, required
empty schemas, and append-ready validation. Absence of rows is never treated as
completion evidence.

## Pre-commit and publication

Immediately before publication the writer re-proves:

- exact artifact names and canonical relative paths;
- staged-directory identity;
- regular-file device, inode, size, mode, and SHA-256;
- equality with the durable journal and lock artifact hashes;
- held lock ownership;
- target absence;
- same-device staging and publication.

Publication follows ADR-0002. It atomically reserves the absent target with a
descriptor-relative `mkdir`, pins the directory, hard-links validated loci and
alerts, fsyncs and re-proves them, links the validated manifest to a pending
name, and then links that inode to `manifest.json`. `manifest.json` is the
logical commit boundary. A late target is a conflict and is never replaced.

After commit the writer re-proves the target, fsyncs target and parent, removes
the pending link, removes only proven staged links, and independently reopens
canonical paths. Client `fsync` success is recorded without claiming
storage-controller crash survival.

If commit visibility is proven or the NFS link outcome is indeterminate,
science is treated as published. Later cleanup, fsync, reopen, unlock, or
reconciliation failure never deletes it.

## Reconciliation

Night publication and synthetic derived reconciliation are separate
transactions. The nightly lock is released before a separate derived lock is
acquired. Derived bytes are deterministic, staged and validated, promoted with
hard-link no-clobber, reopened, and accepted on replay only when byte-identical.
A conflicting derived artifact fails closed. Reconciliation failure leaves the
night published and records `PUBLISHED_RECONCILIATION_REQUIRED`.

## Recovery

The inspector is read-only and combines journal evidence with independently
observed no-follow target, stage, lock, JSON, and artifact identity evidence.
It reports one or more of:

- `SAFE_TO_DISCARD`: only proven unpublished operational/staging state;
- `SAFE_TO_RESUME`: reserved by the schema, never emitted by this release;
- `REQUIRES_REVALIDATION`: complete or canonical partial bytes need proof;
- `REQUIRES_RECONCILIATION`: committed science lacks completed derived state;
- `REQUIRES_OPERATOR_DECISION`: ambiguity, conflict, active/stale lock, or
  journal/storage disagreement prevents automation;
- `MUST_NOT_AUTO_DELETE`: published, canonical, locked, malformed, or ambiguous
  evidence must be preserved.

The inspector never deletes, repairs, unlocks, resumes, reconciles, or writes a
journal. Malformed journals, orphan journal temporaries, unexpected entries,
symlinks, unsafe file types, checksum/identity mismatches, path disagreement,
and publication disagreement fail closed.

## Consequences

The production-shaped lifecycle can be qualified with deterministic synthetic
science on local storage and the approved Arnor/Shire canary. This candidate
cannot perform an ordinary supported production ingestion. A later explicit
decision must add and review both a production capability issuer and a live
provider adapter; neither can be enabled through a boolean or environment
value in this release.
