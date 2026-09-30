# V3 Publication and Resumable Backfill

This note covers the 0.4.4 candidate (V3-G3.2). It does not authorize live
publication, live acquisition, or cache creation. Production still has no
write-capability factory, so every path below runs only against a sealed
`SyntheticWriteCapability`, which is either temporary or an Arnor canary.

## Candidate state and authoritative state

- **Candidate.** A validated nightly product with `authoritative=false`. It is
  either an offline-recovery root (`status/RECOVERY_FINAL.json` with status
  `RECOVERY_COMPLETE_UNPUBLISHED`) or a backfill night
  (`candidate/candidate-record.json`). Candidate bytes are never rewritten.
- **Authority.** Authority exists only through
  `NightPublisher.publish(candidate, authorization)`. Loci and alert Parquet
  bytes are published unchanged, hard-linked manifest-last. The published
  `manifest.json` is new metadata derived from the candidate manifest, with an
  `authority` block (transaction, authorization digest, predecessor, candidate
  hashes and provenance, chronology, and provenance identities).
- **Three identities that differ.** The scientific artifact contract (for June
  27, `phase6f.offline-recovery.0.4.1-to-0.4.2.v1`), the candidate execution
  release (for June 27, 0.4.3 `4378bce…`), and the publisher release are
  recorded separately under `authority.provenance_identities`.

## Authority commit model

POSIX cannot replace the nightly partition and both cumulative files in one
operation. The publisher therefore brackets the change with one production
marker, `data/lsst_only/PUBLICATION_TRANSACTION_IN_PROGRESS.json` (the
*gate*). This is a gated transition, not an atomic multi-file replace.

| Phase | Production | Readers see | State |
|---|---|---|---|
| Staging, planning, validation, fresh Sentinel V2 qualification | byte-identical baseline | previous generation | `NOT_COMMITTED` |
| Gate hard-linked (`O_EXCL`) before any other mutation; nightly partition (manifest-last) and both cumulative files installed idempotently and verified | changing | explicit refusal (`history.PublicationInProgress`) | `RECONCILIATION_REQUIRED` |
| Gate removed (unlink + directory fsync): **the single authority commit** | new generation | complete new generation | `COMPLETE` |

- Every canonical multi-file reader holds the universal shared authority lock
  for its complete logical operation. This includes history readers and
  rebuild inputs, feature snapshots, Sentinel/commissioning reads, SkyPulse,
  data audit, CLI status/diagnostics, backfill authority status, and zero-row
  repair validation. The publisher and canonical rebuild writers take the
  same object exclusively. Readers also refuse while the durable gate exists
  and retain the generation-token check as a fail-closed detector for a legacy
  writer that bypasses the lock. `data status` reports only the unresolved
  recovery condition; audit and scientific readers refuse without scanning a
  non-authoritative generation.
- The gate name contains `transaction`, so the existing Sentinel V2
  `transaction_artifacts` predicate already fails any qualification while a
  transition is unresolved.
- Classification is evidence-derived (`classify_night_authority`): gate
  present → `RECONCILIATION_REQUIRED`; gate absent and the night's manifest
  SHA and cumulative hashes equal the journal's plan → `COMPLETE` (even if the
  journal was not yet finalized); gate absent and production equal to the
  baseline → `NOT_COMMITTED`; anything else → `CONTRADICTION` (fail closed).
  A visible manifest alone is never `COMPLETE`.
- Resume always rolls a gated transition **forward** with the same
  authorization; staged material is deleted only after durable `COMPLETE`.
  A pre-gate unfinished transaction is closed as `NOT_COMMITTED` and a new
  attempt (`-a2`) starts.
- The gate is recovery evidence, not a cross-host mutex. Reader/writer
  serialization does not depend on prompt NFS directory-entry visibility.

## Exclusion

- One server-coordinated kernel `flock` on
  `control/locks/publication-authority.flock` is shared by readers and exclusive
  to the publisher. Shared leases span the whole logical read. The exclusive
  lease spans the whole `publish()` call: authorization/preflight, staging,
  nightly transaction, cumulative installation, authority commit,
  verification, and terminal evidence. A crash releases it. A reader or second
  writer prevents publication from entering (`publication_lock_busy`,
  transient); readers wait for the bounded read policy.
- The authority lock is outermost; target and reconciliation locks are acquired
  only beneath it. Nested shared reads are reentrant, and reads invoked by the
  publisher borrow its exclusive lease. Shared-to-exclusive upgrades and
  nested publishers fail closed rather than deadlocking.
- Readers never create the production lock. Deployment must first provision
  the real, non-symlink lock at
  `/astro/store/shire/ANTARES/work/publication/control/locks/publication-authority.flock`
  on the same filesystem as production. A missing, linked, non-regular,
  unsupported, differently mounted, or identity-changing lock fails closed.
- The gate (`O_EXCL` hard link) excludes any other authority transition even
  from another control root, and it survives a crash. Only the same
  authorization can resume it; others get `authority_transition_pending`.
- Per-target and reconciliation writer locks are kept. On resume, the
  transaction's own stranded locks are adopted only while the global lock is
  held (no live owner can exist).
- Every journal in the journal root is loaded, of every transaction class. Any
  corrupt journal, or unfinished non-publication transaction, blocks new
  publication.

## Authorization

A `PublicationAuthorization` (schema v2) is integrity-bound by its digest to:
operation `publish-night`; night and predecessor; candidate directory, record,
provenance and artifact hashes; candidate and publisher releases; the full
Sentinel V2 production binding (canonical root, logical mount binding,
durable fingerprint, manifest count, cumulative hashes, and the exact
target/cache/transaction predicates); the planned cumulative output hashes;
a nonce; and an expiry. Runtime `st_dev` is never persisted: immediately
before the gate a fresh full Sentinel capture is compared field by field,
and the data root, every durable file, and staging must be one live
filesystem. Resume is permitted only for the same authorization and
transaction and is allowed after expiry only once the gate exists. The
repository has no trusted signing primitive, so authenticity rests on
filesystem ownership; a signature is a Control requirement.

`ProductionPublicationBinding` defines what a Control-authorized June 27
canary must bind (host, UID, roots, mount, Sentinel, candidate root and hashes,
releases, cumulative baseline and plan, authorization digest, nonce, expiry,
single use). `issue_production_publication_capability` always refuses in
0.4.4.

## Publication chronology

The accepted June 27 candidate manifest has `started_at_utc=2026-09-03T20:13:33Z`
but a placeholder `finished_at_utc=2026-06-27T00:00:00Z`. It is left unchanged
and kept only under `authority.chronology.candidate_manifest_original`, with its
defects listed. Machine-readable, order-checked fields:

- manifest `authority.chronology`: `query_request_started_at_utc`,
  `candidate_completed_at_utc` (recovery `RECOVERY_FINAL.finished_at_utc`),
  `publication_authorized_at_utc`, `publication_transaction_started_at_utc`;
- journal and terminal record: `authority_commit_intent_at_utc`,
  `authority_committed_at_utc` (observed after the gate unlink), and
  `terminal_record_at_utc`.

If the process dies between the gate unlink and recording the commit time,
the commit time is `null` with `authority_commit_observed=false` and explicit
`authority_commit_not_before_utc` / `authority_commit_not_after_utc` bounds.
No time is invented.

## Failure categories

`FailureCategory` distinguishes transient lock contention, interrupted work
before the gate, interrupted reconciliation, transient network or I/O,
authorization refusal, permanent filesystem/permission failure (`EACCES`,
`EROFS`, errno-less `OSError`), candidate corruption, Sentinel drift,
predecessor gaps, journal contradiction, and unclassified failures. Only the
first four are retryable. Wrapped exceptions are classified by their cause.

## Checkpoint and cache

- **Checkpoints are authority.** Each night owns its run root:
  `backfill/nights/night-YYYY-MM-DD/`. Valid sealed segments are never fetched
  again; corrupt evidence fails closed.
- **The cache is disposable.** Every hit is re-verified: object hash, header
  identity, schema identity, row accounting, and per-object locus ownership
  and order. A hit only answers the fetch callback; the checkpoint still
  commits its own evidence.
- **Confinement.** The cache root must be a real directory that neither equals
  nor nests with production data, the Sentinel cache path
  (`/astro/store/shire/ANTARES/cache`, which must stay absent), canary and
  recovery evidence, migration audits, or any capability root. On shire the
  only acceptable root is exactly
  `/astro/store/shire/ANTARES/work/cache/fetch-segments-v1`; it is never
  created implicitly. Internal directories are re-proven as real, contained
  directories at every write.

## Per-night and range state

Stage precedence:

```text
RECONCILIATION_REQUIRED (gate, or an unfinished publication journal)
  > BLOCKED (authority contradiction)
  > PUBLISHED (COMPLETE authority)
  > BLOCKED (last event is a failure) > WAITING_FOR_PUBLICATION
  > FETCH_COMPLETE > FETCHING / QUERY_COMPLETE > PLANNED
```

```text
antares-analysis backfill status START END --run-root RUN_ROOT [--data-root DATA]
    [--journal-root JOURNALS] [--evidence-root EVIDENCE] [--json]
```

This command exits 1 when a night is blocked or needs reconciliation.

## Concurrency, resume, and prior-free acquisition

| Setting | Default | Notes |
|---|---|---|
| `acquisition_concurrency` | 3 | Bounded 1–8. |
| `construction_concurrency` | 1 | Bounded 1–2. |
| `publication_concurrency` | 1 | Only 1 is accepted. |

- `run(start, end, resume=True)` first resolves any `RECONCILIATION_REQUIRED`
  night with the journaled authorization: no acquisition, no rebuild, and no
  later night advances until `COMPLETE` verifies. A `COMPLETE` night whose
  terminal record was lost is replayed to emit it.
- A `RangePublicationAuthorization` (schema v2) binds the science contract,
  configuration, exact provider and adapter implementation hashes, the
  prior-free attestation, the initial Sentinel state, the cache root, and
  publication concurrency 1. No production range capability is issued.
- Acquisition is prior-free. That is equivalent only for providers whose
  selection ignores prior loci. `PRIOR_FREE_ACQUISITION_ATTESTATIONS` pins both
  the exact reviewed provider and the exact `LiveAntaresProvider` adapter class
  to `live_antares.py` SHA-256 `5dbda5e4…`; any other or modified provider or
  adapter is refused before query/fetch. The returned attestation record also
  binds the range's science-contract and configuration hashes that
  `RangePublicationAuthorization` carries.

## Not yet available

- No production publication capability or range capability is issued.
- No general live multi-night acquisition adapter exists; the Arnor live-read
  factory remains restricted to 2026-06-27.
- Offline recovery remains pinned to its 0.4.3 consumer contract and refuses
  to run under 0.4.4. Its output stays publishable through
  `load_offline_recovery_candidate`.
