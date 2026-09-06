# Phase 5 Recovery Inspection Runbook

This runbook interprets interrupted synthetic-writer evidence. Phase 5 recovery
is inspection-only: it does not authorize deletion, lock breaking, resume,
production publication, or production reconciliation.

## Safety boundary

Before inspection, preserve the journal, target, stage, and lock paths exactly
as observed. Do not rename, edit, chmod, unlink, truncate, or recreate them.
Do not infer that a PID belongs to the recorded writer unless host and process
start identity also match.

The scientific authority is the canonical partition plus its validated
`manifest.json`; the journal is operational evidence. A visible
`manifest.json` may prove publication even when the last journal update did not
complete. Conversely, a journal claim cannot make incomplete physical storage
valid.

## Read-only command

```bash
antares-analysis recovery inspect JOURNAL \
  --target TARGET \
  --stage STAGE \
  --lock LOCK \
  --json
```

Use the exact paths from the transaction descriptor or preserved incident
evidence. The command performs no mutation. Exit `0` means no operator-decision
classification was required; exit `1` means the evidence requires operator
review. An invalid request or unreadable path returns the normal CLI error
class.

## Dispositions

### `SAFE_TO_DISCARD`

The inspector positively proved an unpublished state with no canonical target
and no surviving lock. This is a classification, not a delete command. Phase 5
ships no general cleanup action. Preserve evidence until an authorized operator
procedure names the exact objects and validates containment and ownership.

### `SAFE_TO_RESUME`

The value is reserved for schema compatibility. Phase 5 never emits it and
ships no resume operation. Existing files alone are insufficient proof of safe
reuse.

### `REQUIRES_REVALIDATION`

Complete staging or a reserved/partial canonical target exists, but publication
or reusable identity is not fully proven. Preserve all bytes. Re-prove the
target night, provider/configuration/release identities, expected artifacts,
query/fetch completion, schemas, counts, modes, inodes, and checksums before any
future authorized action.

### `REQUIRES_RECONCILIATION`

The logical publication boundary is visible, but derived reconciliation is not
complete. Preserve the night. Never roll it back because a cumulative or
derived operation failed. Phase 5 does not run production reconciliation.

### `REQUIRES_OPERATOR_DECISION`

Automation lacks sufficient proof. Typical causes include an active, stale, or
ambiguous lock; partial canonical target; malformed journal; orphan journal
temporary; unexpected file; path mismatch; checksum mismatch; PID reuse; or
journal/storage disagreement. Do not steal the lock or delete state.

### `MUST_NOT_AUTO_DELETE`

The evidence is published, canonical, locked, malformed, or ambiguous. Preserve
it. This classification is deliberately combined with other dispositions to
make the no-delete requirement explicit.

## Publication-boundary interpretation

```text
no manifest.json
  -> not proven published; partial canonical targets still require operator review

manifest.json linked and matching validated identity
  -> logically published; preserve science

manifest link returned an indeterminate NFS error
  -> committed-or-unknown; preserve and inspect

published + client fsync not durably recorded
  -> PUBLISHED_DURABILITY_UNCERTAIN

published + derived failure/incompletion
  -> PUBLISHED_RECONCILIATION_REQUIRED
```

Never describe a post-commit failure as simply unpublished.

## Lock interpretation

- `ACTIVE`: the local hostname, PID, and process-start identity all match.
- `STALE`: the process is absent or the PID now has a different start identity.
- `AMBIGUOUS`: metadata is malformed, the owner is on another host, liveness is
  permission-blocked, or a trustworthy start identity is unavailable.
- `ABSENT`: no lock directory is present.

All present locks require preservation in this release. Age is evidence only;
it is never permission to steal.

## Evidence to retain for escalation

Retain:

- transaction and plan IDs;
- release SHA and configuration/provider identity;
- journal bytes and SHA-256;
- canonical target, stage, and lock paths;
- recovery JSON output;
- target/stage/lock directory inventory with device, inode, mode, owner, size,
  and SHA-256 for regular files;
- host, PID, process-start identity, UTC timestamps, and relevant filesystem
  errors;
- whether `manifest.json` and `.manifest.pending` are visible;
- whether client fsync and independent reopen were durably recorded.

Do not include credentials, ANTARES tokens, or private science in a shareable
report.

## Phase 5 canary cleanup exception

The installed canary harness has a separate, narrowly authorized cleanup for
one exact `/astro/store/shire/ANTARES/work/canary/<RUN_ID>` tree. It requires a
matching bootstrap identity, successful external qualification evidence,
sealed inventory digest, private ownership/modes, same-device and hardlink
closure, a synthetic path allowlist, no nested mount, no active or ambiguous
lock, and descriptor-pinned bottom-up deletion. It is not a general recovery
or production-cleanup command.

## Escalation threshold

Stop and obtain a reviewed, transaction-specific operator decision if any path
is outside the expected run, any identity differs between observations,
production science is involved, a lock is active or ambiguous, publication is
indeterminate, or accepted science/checksum sentinels change.
