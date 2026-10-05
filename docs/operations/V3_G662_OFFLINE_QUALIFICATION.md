# V3-G6.6.2B offline implementation qualification

Control authorization: G6.6.2A ACCEPT_WITH_FOLLOWUP; G6.6.2B offline implementation only.
Baseline: `812c545e14693cdce7ff7458f1d2b50b0804dcd8` (v0.4.7).
Candidate branch: `codex/g662b-p2-compatibility`.
This is a candidate implementation, with no version bump, release, production attestation, live canary, production execution, or production publication.

## Scope and frozen P1

Runtime changes are confined to `live_antares.py`, `science.py`, `backfill.py`, and `production_range.py` under `src/operations`.
P1 remains the default. Its extraction contract, scientific contract, initial grid, ordinary splitter, tile-query construction, membership predicate, and independent P1 replay helpers match the immutable baseline source exactly.

The firewall executes the immutable provider under a test-only alias after checking its SHA256
`afe11a1b0846ed20293d503b393d477f3b4309fcfa195b13320170ed4e18d14c`.
It compares contracts, policies, all 6912 initial tiles, splitter lineage/ties, inner journal events, replay, complete query evidence, returned frame/order, keep-last survivor, and fetch completion.
New candidate bytes are explicitly refused by the unchanged production attestations.

## Closed source-profile verification

The default source registry contains only historical P1 and its two existing exact provider SHAs.
P2 source and artifact registries have no production approval; qualification tests inject one explicit profile.
Source profiles require a nonempty finite frozenset of exact lowercase 64-character provider SHAs and an exact optional P2ProofProfile.

An untrusted source's canonical journal header, schema, full scientific contract, and execution policy must select exactly one trusted profile. Unsupported and ambiguous dispatch refuse; there is no verifier fallback.
The selected provider SHA is pinned through final journal verification.
Historical configuration, request, query checkpoint, complete journal/HEAD/partition/frame, and complete pinned fetch evidence are verified before compatibility metadata is derived.
P2 additionally requires independent complete query proof and exact equality between the sealed P2 event/budget history and the actual journal replay.

## Additive selection compatibility

The canonical schema is `v3.qualified-scientific-selection.v1`.
The descriptor enumerates exact UTC date and interval, the observation-time field and half-open bounds, RA/Dec fields and degree units, exact spatial boundaries including the north pole, DIA-or-SS EXISTS filter, untagged LSST exhaustive target, prior-free intent, qualified normalization/identity, and accepted-tile locus_id keep-last semantics.
Order is lower-child-first, then within-leaf qualified service order, then keep-last and reset-index.
It does not assert equality of mutable live payloads.

New candidate provenance binds `selection_descriptor_schema` and `selection_descriptor_sha256` into its existing binding identity.
Historical adoption-entry fields and all original source bytes/hashes are preserved.
Descriptor equality cannot override failed full source proof.

## Explicit bounded P2

P2 enters only when a 50-row P1 probe reaches a leaf where the frozen ordinary splitter returns no children.
It reuses that saturated probe; it does not repeat the root query merely to enter P2.
Binary64 midpoint subdivision selects the largest width normalized by the historical time/RA/Dec floors, with time, RA, Dec tie order.
Children must have strict interior midpoints and positive widths, preserve exact parent containment, and traverse lower child first.
Only natural exhaustion with 0–49 validated records is accepted; 50 is provisional saturation.
Malformed normalization, identity conversion, numeric overflow, missing LSST association, out-of-domain rows, codec failures, partial responses, and known iterator-close lookup/invocation failures remain non-science.

The explicit profile binds additional depth, secondary logical nodes per root and night, global search-iterator starts, crash reserve, and event-byte ceilings.
The saturated floor root counts as one secondary node, and each successful split reserves both children.
These are qualification values supplied by tests. No production D/B_parent/B_night or page limit is chosen.

## Durable reservation and failure grammar

Every search-iterator attempt has a committed reservation before the callback.
Each committed outcome binds profile identity, stable binary node/path/parent/root/depth, exact tile/query hash, reservation, observed/discarded counts, exhaustion, typed error/retry, dimension/midpoint/paired children, and before/after budgets.
Journal replay restores the frontier and every counter; restarting cannot renew a bound.
A committed reservation without an outcome remains unknown, consumes global starts and crash reserve, and is closed durably before a retry.
Known close errors are recorded as known terminal failures, including when 50 rows were observed.
Exception messages and raw HTTP bodies are not retained.

Under a stationary deterministic mock service, within finite reserved bounds, clean commit hooks and a stale-HEAD event boundary reproduce accepted science/order/partition.
Operational reservation and retry histories may differ.
Exhausted depth/node/search/crash bounds fail closed.

The existing query_progress storage API is unchanged.
Physical interruption inside its file installer can leave ignored temporary residue; strict saved-source verification refuses residue.
No automatic source cleanup or promise of universally successful arbitrary-crash adoption is added.

## Independent science proof

Science independently constructs the initial grid, deterministic secondary split geometry, node lineage, reservation transitions, complete frontier, binary leaf-count identity, counters/budgets, accepted rows, traversal order, and keep-last frame.
It does not use the live P2 reducer or splitter as its proof oracle.
Canonical metadata comparison distinguishes booleans from numeric values.
Complete query identity, request, outcome, completion flags, bounded policy, exact offline client identity, and timing are checked for saved sources and P2 artifacts.
P2 artifact loci must equal the independently reconstructed and prepared keep-last frame; only the existing declared survey-null representation alignment is allowed.

## Narrow range qualification

Unadopted acquisition keeps exact equality to the current selected profile's full contract.
Adopted acquisition keeps the authorized historical source full contract and all exact source/query/fetch/completion provenance.
Initial issuance re-proves the source; its verified selection must equal intended selection for the exact bound night.
New mixed/P2 candidates require both descriptor identities.
Already-bound legacy P1 recovery may retain its original missing additive fields only with exact P1 source/current full-contract equality.

Range release/configuration, detached approval, token/host/UID, Sentinel, predecessor, source entry equality, and gated recovery remain enforced.
ProductionRangePublisher transaction semantics and forbidden runtime files are unchanged.

## Local qualification

Python 3.11 isolated test environment; scientific dependencies pinned to the accepted project versions.

| Suite | Cases |
|---|---:|
| test_g662_p1_firewall | 8 |
| test_g662_p2_proof | 21 |
| test_operations_phase6 | 58 |
| test_phase6_query_checkpoint | 11 |
| test_fetch_checkpoint | 15 |
| test_g64b_saved_acquisition_adoption | 18 |
| test_production_range_g5 | 25 |
| test_backfill_v3 | 14 |
| test_g62_query_transport_retry | 8 |
| test_publication_authority_v3, candidate cases | 79 |
| test_publication_v3 | 24 |
| test_operations_phase5 | 38 |
| test_g3r_regressions | 2 |
| test_offline_recovery | 29 |
| Immutable-baseline provider approval assertion | 1 |

351 unique cases pass: 350 against final candidate runtime bytes plus one immutable-baseline approval assertion.
The final combined firewall/P2/acquisition run passed 178 cases in 418.688 seconds.
Publication/authority regressions passed 172 cases in 57.150 seconds.
The final absolute-URL repeated/cyclic/advancing empty-pagination fixture passed separately.
The original immutable-release approval assertion runs on immutable baseline source; candidate refusal is covered separately. The saved regression launcher preserves macOS multiprocessing spawn behavior.

Required negative categories are covered by the existing checkpoint/fetch/adoption/authority matrices and new P2/profile cases: broken and coherent journal mutations, sequence/hash/HEAD, request/filter/date/config/contract, partition gaps/orphans, full-frame/order changes, missing/altered/extra fetch data, completion identity, unsupported/ambiguous profiles, cross-profile grammar, substitution/symlinks, swap-and-restore/issuance mutation, forged counters/reservations, and zero-row proof.
No source refusal triggers network, repair, or alternate verification.

The synthetic seven-night Jun28–Jul4 flow uses six P1 sources and one genuinely floor-refined P2 source.
It verifies source-specific proof and compatible selection, constructs against current predecessor state, qualifies range authority, and publishes only temporary synthetic roots through the unchanged publisher.
All seven source digests and historical entries remain exact; query/fetch callbacks are unused during adoption.

## Real Jul8–12 protocol

The test-only helper `tests/g662_real_p1_differential.py` runs separate A/B processes under the immutable v0.4.7 Python.
A uses the immutable installed package. B prepends a private scratch copy of baseline source with the four candidate runtime files.
Both check immutable release/version/wheel markers and all 43 baseline/candidate Python source hashes.
Both force package imports to execute source by setting an absent private bytecode prefix before imports, retaining -B, and checking that the prefix remains absent.
B requires a completed guarded A receipt from the same helper and exact immutable implementation.
Both block network callbacks and audit filesystem writes outside private scratch.

Original full inventories come from preserved G6.6.1 phase1.jsonl and are cross-checked against its retained summary and adoption entries.
Each process inventories before/after; B also requires its starting inventory to equal A's final inventory.
Frame records and semantic hashes are recomputed from the entire returned ordered frame using the frozen lossless codec.
A derives selection independently from already verified P1 semantics; B must produce the exact same canonical descriptor.
Full request, contract, independent partition, query evidence, historical adoption entry, query integrity, trace/order hashes, HEAD, fetch binding/completion, segment count, and alert rows must agree.

Real differential status: all five final source-forced A/B pairs PASS. Original before/between/after complete inventories match exactly for every night, with zero network callbacks and zero forbidden-write attempts.

| Night | A | B | Loci | Fetch segments | Alert rows |
|---|---|---|---:|---:|---:|
| 2026-07-08 | PASS | PASS | 179 | 1 | 62911 |
| 2026-07-09 | PASS | PASS | 478719 | 1870 | 1457100 |
| 2026-07-10 | PASS | PASS | 186358 | 728 | 555956 |
| 2026-07-11 | PASS | PASS | 90565 | 354 | 516674 |
| 2026-07-12 | PASS | PASS | 58869 | 230 | 296019 |

Full receipts, exact verifier hashes, complete inventory/provenance comparisons, and the final candidate commit are recorded in the external G6.6.2B adjudication packet. Earlier unguarded/interrupted results are interim only.

## Transport follow-up and next gate

`LIVE_CANARY_BLOCKED_TRANSPORT_GUARD`.
A single next() on pinned antares-client 1.14.0 can consume arbitrarily many empty or cyclic HTTP pages without yielding a locus.
The finite mocked pagination test demonstrates six HTTP callbacks inside one iterator advance, stopped only by a test-owned sentinel.
Search reservations and node/depth limits therefore do not establish a page-count or total HTTP bound.

The smallest viable follow-up within live_antares.py is a provider-owned Session/paginator using the pinned client schema, exact query/sort/order semantics, explicit page/response-byte/deadline limits, and repeated/cyclic-next-link detection before each request.
It must cover empty pages and periods with no yielded locus.
Global requests/client monkeypatching across concurrent nights is unsuitable.
This design is not implemented or qualified for live use in this gate.

Control must independently review the exact candidate commit, adjudicate the physical-crash residue limitation, choose production proof/transport limits, and authorize any later activation/canary/release.
