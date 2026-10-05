# ANTARES V3 — G6.6.3A CLOUD EXECUTION HANDOFF

## Purpose

This document preserves the exact execution state of G6.6.3A across a
local-Claude-to-Claude-Cloud transition.

The local Claude session stopped because its usage allowance was exhausted,
NOT because the scientific or engineering task completed.

This document is continuity evidence, not scientific acceptance evidence.

---

## Exact Git baseline

Repository:

darim1151/ANTARES_Analysis

Branch:

claude/g663-transport-canary-ready

G6.6.3A started from exact accepted G6.6.2B candidate:

8f4ee9bcf4a8be64cfabfa94af50ebb2fc35bc47

That candidate itself follows accepted production baseline:

812c545e14693cdce7ff7458f1d2b50b0804dcd8
(v0.4.7)

---

## Persisted implementation state at local-session interruption

At the handoff point Git showed exactly TWO modified runtime files:

src/operations/live_antares.py
src/operations/science.py

There were NO untracked files.

`git diff --check` passed.

Approximate current diff:

- live_antares.py: ~403 changed/added lines
- science.py: ~15 changed lines
- total: 407 insertions / 11 deletions

IMPORTANT:

No transport-test file had yet been persisted when the local session stopped.

Therefore the exact practical resume point is:

    guarded transport implementation exists
        ->
    transport guard tests still need to be written/completed
        ->
    exact canary profile still needs to be frozen
        ->
    final qualification still needs to run
        ->
    packet/runbook still need to be produced

Do NOT infer that tests have already been completed merely because the local
Claude UI had reached "Writing transport guard test suite."

---

## Local Claude progress before interruption

The local Claude session reported that it had:

1. created a fresh worktree from 8f4ee9b;
2. inspected inherited G6.6.2B code and evidence;
3. inspected pinned antares-client 1.14.0;
4. independently investigated pagination behavior offline;
5. implemented guarded P2 transport changes;
6. changed live_antares.py;
7. changed science.py where transport proof/evidence validation was needed;
8. begun the transport-test phase.

These are executor progress claims.

VERIFY them against the repository before relying on them.

Repository/runtime evidence outranks this narrative.

---

## Gate objective

Gate:

V3-G6.6.3A — TRANSPORT GUARD + CANARY-READY QUALIFICATION

Primary purpose:

Remove:

LIVE_CANARY_BLOCKED_TRANSPORT_GUARD

without changing historical P1 scientific semantics or weakening P2
completeness semantics.

---

## Current scientific architecture

P1 remains historical/default behavior.

P2 is explicit opt-in only.

P2 exists to refine a saturated P1 minimum cell further using bounded
deterministic subdivision.

A scientific leaf may be accepted only when positive completeness is proven
through natural exhaustion below the 50-row probe threshold.

Transport/resource limits are NEVER completeness certificates.

If page, byte, deadline, cycle, origin, node, search, depth, crash or other
finite bounds are reached:

    FAIL CLOSED.

Partial or ambiguous results are NON-SCIENCE.

---

## Known transport problem

Pinned:

antares-client == 1.14.0

G6.6.2B established that one logical iterator advancement may internally
consume multiple HTTP JSON:API pages before yielding a locus.

Therefore P2 node/search/depth budgets alone do not bound actual HTTP work.

The new implementation is intended to add a provider-owned bounded transport
path for live P2 without changing P1 semantics.

Independently verify that this is what the current diff actually does.

---

## Current Arnor production state

Most recent read-only operator verification found:

authoritative nights = 100
authoritative tail   = 2026-07-06
publication gate     = none
production cache     = absent

Authority:

2026-07-06 COMPLETE / finalized
2026-07-07 NOT_COMMITTED
2026-07-08 NOT_COMMITTED
2026-07-09 NOT_COMMITTED
2026-07-10 NOT_COMMITTED
2026-07-11 NOT_COMMITTED
2026-07-12 NOT_COMMITTED
2026-07-13 NOT_COMMITTED

Original G6.6 acquisition:

Jul07 = BLOCKED at QUERYING; saturation unresolved; 0 segments
Jul08 = FETCH_COMPLETE
Jul09 = FETCH_COMPLETE
Jul10 = FETCH_COMPLETE
Jul11 = FETCH_COMPLETE
Jul12 = FETCH_COMPLETE
Jul13 = BLOCKED at QUERYING; saturation unresolved; 0 segments

No separate Jul7 or Jul13 canary roots were found.

Jul8–12 MUST NOT be requeried.

---

## Preserved compatibility result

G6.6.2B already performed real old-vs-new A/B verification for Jul8–12.

All five passed.

Their complete source inventories remained unchanged.

Do NOT repeat this extremely expensive qualification unless the G6.6.3A
implementation changes:

- P1 runtime semantics;
- historical source verification;
- adoption compatibility;
- scientific-selection descriptor semantics.

Prove whether those areas changed using exact diff/hash analysis.

---

## Local evidence paths

On the original Mac:

/Users/darim_shamsi/Documents/ANTARES REVAMP/recovery-evidence/g66-20261004/

/Users/darim_shamsi/Documents/ANTARES REVAMP/recovery-evidence/g66.2a-20261004/

/Users/darim_shamsi/Documents/ANTARES REVAMP/recovery-evidence/g66.2b-20261004/

/Users/darim_shamsi/Documents/ANTARES REVAMP/recovery-evidence/g66.3a-20261004/

The cloud environment may NOT have these arbitrary local paths.

Do not invent their contents.

Use repository evidence and this handoff first.

If an unavailable external artifact becomes necessary to establish a material
claim, STOP and identify the exact artifact required.

---

## Intended write boundary

Primary runtime file:

src/operations/live_antares.py

Allowed only when scientifically justified:

src/operations/science.py

Current interrupted implementation has modified exactly these two runtime files.

Do not broaden runtime scope casually.

Especially do not modify publication, transaction, storage, authority,
commissioning or checkpoint installer semantics.

---

## Immediate next task

FIRST reconstruct continuity.

Then continue from:

    WRITE / COMPLETE TRANSPORT GUARD TESTS

Expected focused test location:

tests/test_g663_transport_guard.py

After that:

1. run focused transport qualification;
2. run P1 firewall;
3. run P2 proof suite;
4. run relevant query-transport regression;
5. freeze one exact finite P2 canary profile;
6. freeze exact finite transport limits;
7. compute/profile-bind their canonical identity;
8. run final qualification against FINAL runtime bytes;
9. prove P1/historical compatibility code remained unchanged or rerun only
   qualification actually invalidated by the diff;
10. write the final evidence packet;
11. write exact future Arnor Jul7/Jul13 manual commands;
12. commit/push final candidate;
13. STOP for Control adjudication.

---

## Final expected packet

Local intended evidence destination:

/Users/darim_shamsi/Documents/ANTARES REVAMP/recovery-evidence/g66.3a-20261004/

Required packet:

G6.6.3A_TRANSPORT_CANARY_READY_PACKET.md

If cloud cannot write the external Mac path, create the packet in a clearly
identified repository/scratch location and report exactly where it resides so
it can later be copied into canonical evidence storage.

Final marker:

AWAITING_CONTROL_G6.6.3A_ADJUDICATION

---

## Hard prohibitions for this gate

NO live ANTARES query.

NO July 7 query.

NO July 13 query.

NO production mutation.

NO publication.

NO release.

NO tag.

NO production P2 attestation.

NO Jul8–12 re-query.

NO change that silently alters P1 semantics.

NO treating transport limits as scientific completeness.

NO weakening fail-closed behavior.

---

## Future path — not authorized yet

If G6.6.3A later passes Control:

immutable release
    ->
fresh Jul7 non-authoritative P2 canary
    ->
offline verification
    ->
fresh Jul13 non-authoritative P2 canary
    ->
offline verification
    ->
fresh Jul7–13 ALL-ADOPTION production range

Final publication must perform ZERO ANTARES query/fetch and use:

Jul07 -> new P2 saved acquisition
Jul08 -> preserved P1 acquisition
Jul09 -> preserved P1 acquisition
Jul10 -> preserved P1 acquisition
Jul11 -> preserved P1 acquisition
Jul12 -> preserved P1 acquisition
Jul13 -> new P2 saved acquisition

---

## Continuity rule

Do not trust this document over current repository evidence.

At cloud startup:

1. verify branch;
2. verify HEAD;
3. inspect complete diff from 8f4ee9b;
4. reopen changed implementation;
5. reconstruct what is complete vs incomplete;
6. only then continue.

If repository state contradicts this handoff:

STOP:

CLOUD_CONTINUITY_MISMATCH
