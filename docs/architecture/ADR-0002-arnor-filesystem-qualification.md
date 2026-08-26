# ADR-0002: Arnor Filesystem Qualification and Publication Contract

Status: accepted for Phase 5 local implementation. Production writing,
production operations-root provisioning, and cache creation remain disabled.

Administrative update (2026-08-26): the single-root correction in ADR-0003 was
accepted and Phase 5 resumed. Phase 4 qualification evidence remains valid
historical evidence; current project paths are governed by ADR-0003.

## Context

Phase 3 established proof-only writer contracts. Phase 4 exercised those
contracts on Arnor with synthetic artifacts in a disposable Shire canary while
the accepted science root remained read-only.

Canonical evidence:

```text
/astro/users/mdarim/.local/state/antares-analysis/phase4/
    phase4-20260823T103930Z
```

The qualified host is `arnor.astro.washington.edu`, Rocky Linux 9.8,
kernel `5.14.0-687.33.1.el9_8.x86_64`, x86_64, with system Python 3.9.25.

`/astro/store/shire` is NFSv4.2 on device 55, mounted from
`shire.infiniband:/data/shire`. User release/state space is a different NFS
export on device 54. Staging in user state therefore cannot be atomically
published into Shire and cannot be hard-linked into a final partition.

## Empirical findings

- File and directory rename to an absent target succeeded on Shire.
- An observer sampled only absent or complete states during an absent-target
  directory rename; no partial directory was observed.
- Plain rename replaced an existing regular file and an existing empty
  directory. It refused a non-empty destination with `EEXIST`.
- A source on device 54 could not be renamed to device 55 (`EXDEV`).
- Linux `renameat2(RENAME_NOREPLACE)` was available in libc but the Shire mount
  rejected an absent-target operation with `EINVAL`. It is not a usable Shire
  publication primitive.
- `flush`, file `fsync`, directory `fsync`, and parent-directory `fsync`
  succeeded from the Arnor client. This proves client/protocol support, not
  storage-server or controller crash durability.
- Atomic directory creation admitted exactly one writer under eight-process
  contention. A lock survived abrupt process death, was classifiable as stale,
  and was not stolen automatically.
- An atomic target-directory reservation followed by same-filesystem hard
  links for `loci.parquet`, `alerts.parquet`, and finally `manifest.json`
  succeeded. The links had the same inodes as the validated stage. Observers
  never saw a manifest without both data files. Re-reservation and manifest
  relinking refused existing targets with `EEXIST` and preserved them.

## Evidence classification

| Classification | Phase 4 evidence |
| --- | --- |
| `VERIFIED_EMPIRICALLY` | Synthetic same-device rename, overwrite, `EXDEV`, `renameat2(RENAME_NOREPLACE)`, hard-link publication, observer visibility, client `fsync`, lock contention, and abrupt-owner-death behavior in the exact Shire canary. |
| `VERIFIED_BY_READ_ONLY_INSPECTION` | Arnor identity and runtime, mount/export/device topology, production inventory/checksums/permissions, capacity and inode counters, and the fact that the cache path was absent. |
| `SUPPORTED_BY_DOCUMENTATION` | Linux/NFS interfaces and error semantics informed the probes, but documentation alone was not accepted as proof of Shire server/controller crash durability. |
| `PLAUSIBLE_BUT_UNVERIFIED` | Survival across storage-server or controller failure, behavior from another NFS client host, and any inference that an unreported user quota means unlimited capacity. |

These classifications are intentionally non-transitive: successful client
calls do not promote a storage-crash claim to `VERIFIED_EMPIRICALLY`.

## Decision: logical publication

Plain `os.rename()` is rejected for publication because an existence check and
rename have a target-race window, and Shire replaces an empty directory.
`renameat2(RENAME_NOREPLACE)` is rejected because Shire does not support it.

Phase 5 will use a manifest-commit protocol:

1. Hold the deterministic target writer lock and re-prove its on-disk
   directory identity, target/run identity, ownership token, and exact metadata
   immediately before publication. A memory-only `held` flag is insufficient.
2. Stage all three artifacts on the qualified Shire filesystem.
3. Flush and `fsync` `loci.parquet`, then `alerts.parquet`, then the manifest.
4. Validate the closed staged files, including query/fetch evidence, schemas,
   counts, linkage, checksums, and the manifest-last contract; capture and
   persist each validated device, inode, size, mode, and SHA-256 identity.
5. `fsync` the staged directory and persist validated journal state.
6. Open no-follow descriptors for the stage and target parent. Atomically
   reserve the absent final target with descriptor-relative `mkdir`, open and
   pin that exact directory, and reject any later parent-entry substitution.
   Any existing file, directory, or symlink is a conflict; nothing may be
   removed or replaced.
7. Descriptor-relative hard-link loci and alerts into the pinned target,
   re-prove every link against its validated identity, and `fsync` the target
   directory.
8. Link the validated manifest inode first to a non-commit pending name, verify
   that link, `fsync` it, and hard-link that verified inode to `manifest.json`
   last. The `manifest.json` link is the logical commit.
9. Re-prove the pinned target entry, `fsync` the target directory and its
   parent, remove the exact pending manifest link, and `fsync` the target again.
10. Prove every staged/final pair is the validated regular-file inode, remove only
    the three exact staged links, remove the empty stage, and `fsync` the
    staging parent.
11. Independently reopen and verify the final partition before releasing the
    nightly writer lock.

Consumers must treat a target without `manifest.json` as uncommitted and must
never infer publication from directory or Parquet-file presence alone.

## Commit-boundary failures

Before the manifest link is issued, a failure is unpublished. A reserved or
partially linked target, including a pending-manifest link, is not safe for
automatic deletion; it requires revalidation and operator recovery because it
occupies the canonical final path.

After the manifest link succeeds, science is logically published. Any later
file-system, `fsync`, staged-link cleanup, final validation, or reconciliation
failure must record `published=true`, preserve both target and available stage,
and require operator action. It must never be represented as rollback-safe or
cause published science to be deleted.

An indeterminate NFS error from the manifest-link operation is classified
conservatively as committed-or-unknown even when an immediate lookup cannot
confirm visibility. An exact visible `manifest.json` matching the validated
identity proves commit independently of the temporary pending link. Recovery,
not rollback, resolves either case.

Client-side `fsync` success does not establish storage-server crash durability.
That property remains documentation-dependent and must not be claimed by the
writer.

## Lock and recovery contract

The production lock and staging location must be explicitly configured outside
the accepted science root but on the same qualified Shire filesystem. Phase 4
did not create that location.

Every writer, planner, journal, and recovery inspector must derive the same
lock identity. Lock metadata must include the canonical root/target, plan and
run identity, release and artifact hashes, user/UID, host, PID plus PID-reuse
defense, acquisition time, and an ownership token. Metadata and containing
directories must be fsynced.

Phase 4 qualified independent processes and SSH sessions on Arnor only. Writer
execution from another client host or a multi-host writer topology remains
refused until separately qualified.

Initial recovery classifications are:

| Observed state | Classification |
| --- | --- |
| Lock only | `REQUIRES_OPERATOR_DECISION`, `MUST_NOT_AUTO_DELETE` |
| Partial stage | `SAFE_TO_DISCARD` only after exact containment and target-absence proof |
| Complete unvalidated stage | `REQUIRES_REVALIDATION` |
| Validated unpublished stage | `REQUIRES_REVALIDATION` |
| Reserved partial canonical target | `REQUIRES_REVALIDATION`, `REQUIRES_OPERATOR_DECISION`, `MUST_NOT_AUTO_DELETE` |
| Published unreconciled target | `REQUIRES_RECONCILIATION`, `MUST_NOT_AUTO_DELETE` |
| Stale metadata with no active process | `REQUIRES_OPERATOR_DECISION`, `MUST_NOT_AUTO_DELETE` |

No state is automatically resumable in the first Phase 5 implementation.
Published nightly science and derived reconciliation are separate transactions;
reconciliation failure never rolls back the night.

## Preflight and deployment contract

- The production operations root is unconfigured. Read-only plans report no
  production lock resource; `.antares-operations` beneath the data root is a
  legacy temporary-fixture layout only.
- Measure filesystem bytes and inodes separately on the target mount.
- Report quota separately. Arnor exposed no user quota through `quota`; this is
  not evidence of unlimited capacity.
- Re-prove canonical containment, ownership, mount identity, device, cache/data
  separation, target absence, lock state, and baseline identity immediately
  before mutation.
- Deploy immutable release-specific environments under
  `/astro/users/mdarim/opt/antares-analysis/releases/<git-sha>/venv`.
- CLI and Jupyter must import the same installed operations package. A source
  checkout is optional for installed-runtime health and required only when
  checkout notebooks are explicitly validated.

## Permission contract

The Phase 4 private synthetic writer creates lock, staging, and final
directories as `0700` and metadata/science files as `0600`, using explicit
`fchmod` rather than ambient umask. Hard links retain the validated staged-file
mode. The accepted production root itself is `0700`, while its migrated
descendants remain the separately observed `2775`/`0664`; Phase 4 did not
change them. Production activation must explicitly approve its final ownership
and mode policy and must not infer it from the recorded Arnor umask `0022`.

## Authorization boundary

The current capability remains private and restricted to synthetic temporary or
exact canary roots. There is no production capability constructor or writer CLI
command. Phase 5 may implement and test the journal, recovery inspector, and
manifest-commit writer locally, but it may not contact ANTARES, provision the
production operations/cache roots, publish production science, reconcile real
derived data, run a scheduler, or enable backfill execution.

Production activation requires a later explicit authorization, a fresh
before/after production sentinel, and a reviewed recovery/operator runbook.
