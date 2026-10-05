# V3-G6.6.3A guarded P2 transport and canary-ready qualification

Gate: V3-G6.6.3A — transport guard plus canary-ready qualification (offline only).
Base: accepted G6.6.2B candidate `8f4ee9bcf4a8be64cfabfa94af50ebb2fc35bc47`.
Continuity checkpoint: `7d7e6654ffdc7b7b84ed684fd5cf095b6d3cf630`.
Branch: `claude/g663-transport-canary-ready`.

This gate makes no version bump, release, tag, production attestation or
registry entry. It performs no live ANTARES request, Jul07/Jul13 query,
Jul08–12 re-query, production mutation or publication. The final
qualification results, source hashes and Control verdict are recorded in
[`G6.6.3A_TRANSPORT_CANARY_READY_PACKET.md`](G6.6.3A_TRANSPORT_CANARY_READY_PACKET.md).

**G6.6.3D update (release blocker RB1).** A G6.6.3A page thread blocked in
name resolution or in urllib3's chunked-trailer loop could not be stopped at
the iterator deadline. G6.6.3D moves each P2 page's blocking HTTP exchange
into one dedicated, OS-killable child process per night query
(`src/operations/p2_transport.py`). Pagination, decoding, deadlines and
every completeness decision stay in the scientific process. The transport
and profile identities, the runbook and the risks below are updated to the
G6.6.3D bytes. The remediation evidence is
[`G6.6.3D_PROCESS_TRANSPORT_REMEDIATION.md`](G6.6.3D_PROCESS_TRANSPORT_REMEDIATION.md).

## 1. Transport root cause

The pinned `antares-client==1.14.0` `search()` is `_list_all_resources`. One
loop does `requests.get(url, params, timeout=60)`, raises on status >= 400,
loads the whole page with `_LocusListingSchema(many=True, partial=True)`,
yields it, then reads `response.json().get("links", {}).get("next")`. A
`None` value stops the loop. There is no page, byte, whole-iterator time,
cycle, scheme or origin bound. `requests.get` also follows up to 30
redirects to any origin. One `next()` can therefore consume unboundedly many
empty or cyclic pages. The finite P2 node, depth and search budgets bound
iterators, not HTTP.

Pagination is real in production. The preserved v0.4.5 Jun30 and Jul06
journals (G6.2) failed on a later page after 40 and 30 rows. Page
boundaries therefore fall on a divisor of 10 rows. A 50-row probe crosses
five or more pages, and the client followed those links. The client can
only request absolute URLs: `requests` raises `MissingSchema` on a relative
one. Every followed production link was therefore absolute.

`search` is the only pinned-client function that paginates. `get_by_id`
(fetch), `get_available_tags` and the lazy locus attributes each perform a
single request. Fetch is unchanged and identical to P1.

## 2. Architecture

* **P2 only.** `LiveAntaresProvider._load_client` sends only providers that
  carry a `P2ProofProfile` to `_load_p2_client`. The P1 branch and every
  other P1 definition are byte-identical to G6.6.2B (section 6).
* **Opt-in.** `_load_p2_client` returns injected callbacks only for a
  `local-mock` capability (the G6.6.2B offline qualification). A live search
  requires an exact `P2TransportLimits` on the profile, a sealed
  `LiveAntaresReadCapability` in `arnor-commissioning`, no injected
  callables, the pinned client 1.14.0, the official base URL and the 60 s
  client timeout. Otherwise it raises `LiveCapabilityError`. The G6.6.2B
  offline profile (`transport=None`) can never search live.
* **Provider-owned paginator.** `_GuardedListing` uses the client's own URL
  (`urljoin(base, "loci")`), parameters (`sort`, `json.dumps(query)`), schema,
  page order, termination (a complete page whose `links.next` is null or
  missing) and laziness. The next page is requested only when the consumer
  asks past the current page. Each page uses a fresh `requests.Session`, the
  same as `requests.get`. Nothing global is patched.
* **Per logical iterator bounds** (one durable P2 search reservation):
  * pages (`max_pages`);
  * consecutive empty pages that still continue (`max_consecutive_empty_pages`);
  * decoded body bytes per page and per iterator (`max_page_bytes`,
    `max_iterator_bytes`), enforced on the streamed body, plus a refusal
    whenever a declared `Content-Length` exceeds the budget;
  * whole-iterator wall clock (`iterator_deadline_seconds`), enforced before
    every page and during it: the page runs in the transport child, and at
    the deadline the parent kills that child (SIGTERM, 2 s grace, SIGKILL,
    5 s join) and requires a reaped exit status (G6.6.3D);
  * the client's connect and read timeouts, exactly (60 s each);
  * redirects per page (`max_redirects`).
* **Continuations.** A `links.next` must be one printable-ASCII string with
  no whitespace, control character or backslash. It must be absolute
  `https`, on the exact API host and port 443, carry no credentials or
  fragment, and name exactly `/v1/loci`. Its `(path, query)` must never repeat
  an already requested page. Relative continuations are refused, matching
  the pinned client, which cannot request them either.
* **Redirects.** Every `Location` is validated before any request: plain
  ASCII, same origin, `https`, no credentials or fragment, under `/v1/`.
* **Process boundary (G6.6.3D).** One child per P2 night query, started on
  the first page and reused for every page of every search, runs only
  `p2_transport.py` (standard library and `requests`; never the `src`
  package). It performs one page exchange per request (resolution, connect,
  TLS, redirects, bounded body read) and returns the exact body bytes and the
  charset `requests` derived, or a typed failure, over a versioned,
  pickle-free, bounded pipe protocol. Any deadline, IPC violation, early end
  of stream, crash or startup failure kills the child and fails the
  iterator; the transport is never restarted. A child that cannot be proven
  dead is fatal (`P2TransportUnkillableError`).
* **Bodies.** Every HTTP exchange goes through the session's transport
  adapter with exactly the prepared request and environment settings that
  `Session.request` derives. Only `Session.send`'s redirect bookkeeping is
  skipped (D2 below). Bodies are never read for redirects or for statuses
  >= 400.
* **Status classification.**
  * HTTP >= 400 raises `P2TransportHTTPError` (`RuntimeError`), which is
    retryable within the existing per-node attempts, like the client's
    `AntaresException`.
  * A non-200 status below 400 that is not a redirect raises a refusal.
* **Refusals are terminal.** Every bound or rule violation raises a
  `P2TransportGuardError` subclass (`ValueError`), which the unchanged
  classifier treats as non-retryable. The P2 reducer records the outcome as
  `attempt_error` with reason `p2_transport_guard` and the exact exception
  type. The node discards every row, the night fails, and the science
  verifier refuses any non-retryable error. A limit is never a completeness
  certificate.

Where the guard is stricter than the pinned client, it refuses rather than
interprets:

* non-200 2xx and 304 responses;
* a non-object `links` member;
* insecure, foreign or relative continuations;
* foreign or unsafe redirects.

## 3. Defects found and fixed in this gate

| Id | Defect at checkpoint `7d7e665` | Fix |
|---|---|---|
| D1 | `science._validate_phase6_manifest_evidence` accepted only the unguarded client identity for `arnor-commissioning`. Every guarded-P2 Arnor night therefore failed artifact reopen (`phase6_client_identity_invalid`) at construction and publication, after passing the P2 proof. Reproduced end to end. | Only for a trusted P2 profile with transport limits on Arnor, the expected identity is the guarded pagination contract. `validate_p2_query_result` already proved the exact identity, including `transport_sha256`. P1 manifests take the unchanged path. |
| D2 | `requests.Session.send` pre-computes `Response.next` even with `allow_redirects=False`, so it read every redirect body completely, outside any byte ceiling. The pinned client has the same exposure. A non-UTF-8 `Location` raised inside requests. | `_GuardedListing._send` calls the session's transport adapter directly with identical request and settings. |
| H1 | `Location` lacked the control-character, backslash and fragment checks that continuations had. Backslash is a known `urlsplit`/urllib3 parsing difference. | Shared `_p2_plain_url` (G6.6.3D: `p2_transport.plain_url`) for continuations and `Location`. Fragments in redirects are refused. |
| H2 | The canary profile docstring referred to a hash constant that did not exist. | `G663_CANARY_P2_PROFILE_SHA256` and `G663_CANARY_TRANSPORT_SHA256` are defined. The profile refuses to exist unless its canonical bytes still hash to them. |

## 4. The frozen canary profile (one profile, both nights)

```text
P2ProofProfile(max_depth=18, max_nodes_per_root=511, max_nodes_per_night=4095,
               max_search_attempts=250000, crash_reserve=4, max_event_bytes=33554432,
               transport=P2TransportLimits(max_pages=64, max_consecutive_empty_pages=2,
                   max_page_bytes=16777216, max_iterator_bytes=67108864,
                   iterator_deadline_seconds=600, connect_timeout_seconds=60,
                   read_timeout_seconds=60, max_redirects=2,
                   ipc_max_header_bytes=65536, ipc_max_chunk_bytes=1048576,
                   child_startup_seconds=30, child_shutdown_seconds=5,
                   terminate_grace_seconds=2, kill_join_seconds=5))
transport schema v3.p2-transport-limits.v2, IPC g663d.p2-transport-ipc.v1
child implementation (p2_transport.py) sha256 2e661e528899c6a6ac02eb6af38efbcc23ad9e5cad2e083963c9f9a24cea5bee
profile identity  G663_CANARY_P2_PROFILE_SHA256 = 6ce4b3a29d79154513bfe20213cd956475e20bc224795e54724c6af44c6a105c
transport identity G663_CANARY_TRANSPORT_SHA256 = 6e0291f3dd203dbccc9a00239072f6501b06dbafc9f1d37c97498be1d7052ddd
2026-07-07 scientific contract sha256 = f1df451787cc22728fb5bf66ebd124f818b303d27e74f74b01c05aa550fdf036
2026-07-13 scientific contract sha256 = df812a2ec232e331f33e1f0020ffd46591cae1775335e067eb77f859e1a24b45
```

G6.6.3A's identities (profile `d1dfee3b…9684`, transport `41ad48a1…0650`,
contracts `e1e28e86…1fe5` and `08ab9c77…eb13`) are superseded: the transport
contract now also binds the process boundary and the child's bytes.

The profile digest is every P2 event's `profile_sha256`. The transport
digest is the client identity's `transport_sha256`. Both are bound into the
extraction method, the scientific contract, the query policy, the
configuration identity and every checkpoint. The selection descriptor is
unchanged, so Jul07/Jul13 remain selection-compatible with Jul08–12.

### Geometry

The geometry comes from the code itself (`FrozenProfileTests`).

* The P1 floor is reached after 25 splits from a 30 min × 15° × 30° initial
  tile. It measures 28.125 s × 0.029297° × 0.029297° (105.5″ × 105.5″).
  Jul07 and Jul13 failed because such a floor cell held ≥ 50 LSST loci.
* P2 splits the largest floor-normalized width, so it cycles time, RA, Dec
  and halves the area every three levels. At depth 18 cells are
  0.44 s × 1.65″ × 1.65″, with no binary64 midpoint collapse at any domain
  extreme (RA 0/360, Dec ±90, first and last time bin).

### Per-root demand

Simulated with the provider's own `_p2_split` over one floor cell, as
(nodes, deepest depth):

| Loci in one floor cell | uniform, one visit | uniform, mixed times | ~10″ cluster | ~2″ cluster | RA streak |
|---:|---|---|---|---|---|
| 50 | 5, 2 | 3, 1 | 5, 2 | 5, 2 | 5, 2 |
| 500 | 41, 6 | 29, 4 | 115, 17 | 169, 23 | 67, 8 |
| 1000 | 101, 8 | 59, 5 | 173, 18 | 189, 24 | 163, 11 |
| 3000 | 257, 11 | 169, 7 | 373, 21 | 445, 27 | 487, 17 |
| 5000 | 437, 12 | 261, 8 | 503, 21 | 671, 29 | 763, 20 |

* `max_depth=18`. Fifty distinct loci in a 1.65″ × 1.65″ cell would need
  about 0.23″ spacing, far below the seeing and any distinct-locus
  association scale. Deeper saturation means coincident or duplicate loci,
  which must fail closed (`p2_depth_exhausted`).
* `max_nodes_per_root=511` (255 splits) covers about 5,000 co-temporal
  loci uniformly spread in one 105″ floor cell (437 nodes). It also stops
  any single pathological root from consuming the night's secondary budget.
* `max_nodes_per_night=4095` is about eight maximal roots, or about 800
  roots just over 50 loci (5 nodes each).
  * Jul08–12 includes Jul09's 478,719 loci, and none of those nights reached
    the floor. Floor saturation is a rare local pathology, not a density
    regime.
  * Extra service load is at most about 4,095 searches (≤ 8,190 with
    retries), against an estimated 40–60k searches for a Jul09-scale
    night.
* `max_search_attempts=250000` bounds every reservation of the night,
  primary and secondary.
  * Leaves × 49 must cover the night's loci. For Jul09, about 25k leaves
    gives about 6,912 + 2 × 18k ≈ 43k searches, so the cap is about 5–6×
    that.
  * Control can check this read-only against the sealed Jul09
    `search_request_count` before authorizing the canary.
* `crash_reserve=4` allows up to four interrupted reservations, each an
  operator resume after a kill.
* `max_event_bytes=32 MiB` allows a 671 KB normalized-record ceiling (≥ 100×
  any realistic LSST locus listing record). It bounds every journal event at
  commit and at replay.

### Transport

* `max_pages=64` is at least the 50 rows a probe needs plus one terminal
  page at any page size ≥ 1, plus 13 spare empty pages. At the evidenced
  10-row pages a node needs ≤ 6 pages. A legitimate sequence with no empty
  continuing pages can never reach it (tested at page size 1: 50 pages).
* `max_consecutive_empty_pages=2`. Offset pagination yields no empty
  continuing pages, and a terminal empty page is natural exhaustion, not
  counted. Two tolerates a service quirk. Three or more, or any cycle, is
  pathological.
* `max_page_bytes=16 MiB` and `max_iterator_bytes=64 MiB`. Ten listing
  resources are about 10–100 KB, so the margins are ≥ 160×. Memory stays
  bounded per iterator, roughly 16 MiB raw plus decoding. Bytes are counted
  after content decoding, so compression cannot bypass them.
* `connect/read_timeout_seconds=60` is exactly the pinned client's 60 s.
* `iterator_deadline_seconds=600` is 10× the client's per-request timeout
  and about 100× a normal node. It is the only bound on a slow-drip body,
  which defeats per-read timeouts.
* `max_redirects=2` allows a trailing-slash or similar same-origin
  normalization. Each logical page costs at most 3 HTTP requests.
* **Worst case per iterator:** 64 pages × 3 requests, ≤ 64 MiB and ≤ 600 s.
  The night's reservations are finite, so total HTTP is finite.

## 5. Qualification tests

`tests/test_g663_transport_guard.py` drives the real pinned client and the
guarded paginator through `requests`' own transport-adapter seam against one
deterministic offline JSON:API service. It also uses a real loopback HTTP
server for timing. A socket guard refuses any non-loopback connection or
name lookup in every test.

* **Equivalence.**
  * One page, many pages, empty intermediate pages and terminal empty pages.
  * `next` null, missing, or no `links` at all.
  * Absolute, `:443` and upper-case-host links, and same-origin redirects.
  * 60 randomized sequences with unicode, nested and binary64 payloads.
  * Each case yields the client's own `Locus` objects with identical
    records and order, after the identical HTTP request sequence. With P2
    consumption, the same number of pages is used before close at 50.
* **Whole nights.**
  * A floor-saturated night completes through the real Arnor wiring with
    journal events byte-identical to the G6.6.2B-qualified callback path.
    It fetches through the unchanged `get_by_id`, and builds and reopens as
    science (D1).
  * A non-floor night issues exactly the P1 pinned client's request sequence
    and produces identical science, trace, order and keep-last survivor.
* **Negative.**
  * Self-loop, two-page and longer cycles, including other spellings of the
    same URL.
  * Ever-advancing empty pages; page, page-byte, iterator-byte and gzip
    ceilings at the frozen values.
  * Deadlines before a page, during a stalled page, during a slow-drip body,
    and across fast pages.
  * Read, connect and request timeouts.
  * Malformed JSON and JSON:API (same exception types as the client).
  * Malformed, relative, insecure, foreign, credentialed, fragment and
    other-path continuations (37 vectors).
  * Unsafe redirects (15 vectors) and redirect loops.
  * Unexpected statuses.
  * Transport failure after earlier valid pages (503, non-JSON, reset).
* **Fail-closed matrix.** Each of 22 refusals runs inside a full P2 night,
  which must end:
  * incomplete, with exactly the typed exception, reason and discarded rows;
  * with no accepted row, and refused by `require_completed`,
    `validate_p2_query_result`, the independent science replay, checkpoint
    sealing and fetch;
  * reproducible from its journal alone, with zero requests.
* **Identity and firewall.** Section 6 and the frozen-profile goldens.
* **Process boundary (G6.6.3D).** Every whole-night, socket and deadline test
  above runs the real transport child over loopback.
  `tests/test_g663d_process_transport.py` qualifies the boundary itself:
  pre-socket (resolver) and chunked-trailer kills with the bytes stopping,
  equivalence and 50-row ordering, crashes, malformed, foreign, out-of-order
  and oversized frames, end of stream before COMPLETE, deadlines during the
  IPC transfer, startup failures, clean shutdown, persistence, the
  finalizer and orphan watchdog, the fail-closed night matrix, the P1
  firewall and the child's import boundary.
* **Runbook dry run.** The two scripts in section 7 are extracted verbatim
  from this document and executed offline, on success and failure nights.

## 6. P1 non-regression and the Jul08–12 A/B decision

`G663FirewallTests` proves this from Git (base `8f4ee9b`):

* Only `live_antares.py`, `science.py` and the new P2-only
  `p2_transport.py` (G6.6.3D) differ under `src/`. Every forbidden runtime
  file, plus `backfill.py` and `production_range.py`, is byte-identical.
* `live_antares.py`:
  * Changed top-level definitions are only `LiveAntaresProvider`,
    `P2ProofProfile`, `_P2State` and (G6.6.3D) `_run_p2_query`, whose only
    change is a `try`/`finally` that ends the night's transport child;
    everything added is the named G6.6.3A and G6.6.3D set.
  * In the provider class, only `_load_client` differs, and only in its
    first statement, the P2-only `if proof_profile is not None` branch.
  * In `_P2State`, only `outcome()`'s terminal reason differs
    (`p2_transport_guard` for guard types).
  * P1 `query`, the tiles, splitter, query builder, membership, replay,
    dedup and fetch are unchanged.
* `science.py`: only `validate_p2_query_result` (P2-only) and
  `_validate_phase6_manifest_evidence` differ. Removing the one D1 block,
  which is gated on a trusted P2 transport profile, reproduces the base
  function exactly.
* The G6.6.2 immutable-provider differential (`test_g662_p1_firewall`) runs
  unchanged on the final bytes.

The Jul08–12 real A/B verification is **not** repeated. This gate leaves
these unchanged:

* historical P1 runtime semantics;
* historical source verification (`backfill.py`, the P1 science paths,
  source registries);
* adoption compatibility (`production_range.py`, `backfill.py`);
* selection-descriptor semantics.

The provider file's SHA-256 changes, but the trusted historical source
providers (`ADOPTABLE_SOURCE_PROVIDERS`) and the replay code that verifies
Jul08–12 are unchanged.

## 7. Future Arnor canary runbook (prepared, NOT EXECUTED)

Execute only after Control accepts G6.6.3A, closes RB1 (G6.6.3D) **and**
approves an immutable release built from the accepted commit. That release
must carry exactly these runtime bytes:

```text
src/operations/live_antares.py  sha256 c8557ff6b6d80842b0c5f4be4a8be21d66f7f36852c0f57af60507e5827284ba
src/operations/p2_transport.py  sha256 2e661e528899c6a6ac02eb6af38efbcc23ad9e5cad2e083963c9f9a24cea5bee
src/operations/science.py       sha256 bb3e377bdee1132cc90cb2890b24d31b209a499cac5ee0e77c329705476be106
```

Run Jul07 first. Run Jul13 only after Jul07 passes verification and Control
authorizes it. Both nights use the identical profile and the same procedure.
The canary roots are non-authoritative.

* Nothing here publishes, adopts or constructs production authority.
* Nothing creates a cache.
* Nothing writes outside the canary root and the operator evidence directory.

### 7.1 Session, identity and preflight (read-only)

```bash
tmux new -s g663-p2-canary          # re-attach: tmux attach -t g663-p2-canary
umask 077
unset ANTARES_API_BASE_URL API_TIMEOUT
export RELEASE_SHA=<Control-approved release SHA carrying the G6.6.3D bytes>
export WHEEL_SHA256=<that release's wheel SHA-256>
export RELEASE_ROOT=/astro/users/mdarim/opt/antares-analysis/releases/$RELEASE_SHA
export PY=$RELEASE_ROOT/venv/bin/python
export PROVIDER_SHA256=c8557ff6b6d80842b0c5f4be4a8be21d66f7f36852c0f57af60507e5827284ba
export P2_TRANSPORT_SHA256=2e661e528899c6a6ac02eb6af38efbcc23ad9e5cad2e083963c9f9a24cea5bee
export SCIENCE_SHA256=bb3e377bdee1132cc90cb2890b24d31b209a499cac5ee0e77c329705476be106
export PROFILE_SHA256=6ce4b3a29d79154513bfe20213cd956475e20bc224795e54724c6af44c6a105c
export TRANSPORT_SHA256=6e0291f3dd203dbccc9a00239072f6501b06dbafc9f1d37c97498be1d7052ddd
export NIGHT=2026-07-07            # 2026-07-13 only for the second, separately authorized canary
export RUN_ID=g663-p2-$NIGHT-v1
export CANARY_ROOT=/astro/store/shire/ANTARES/work/canary/$RUN_ID
export EVIDENCE_DIR=/astro/users/mdarim/antares-control/g663/$RUN_ID

test "$(hostname -f)" = arnor.astro.washington.edu && test "$(id -u)" = 1533564 || echo "STOP: host/UID"
test "$(cat "$RELEASE_ROOT/RELEASE_SHA")" = "$RELEASE_SHA" || echo "STOP: release marker"
test "$(cat "$RELEASE_ROOT/WHEEL_SHA256")" = "$WHEEL_SHA256" || echo "STOP: wheel marker"
"$PY" -I -B -c 'import sys; from importlib import metadata; import src.operations.live_antares as L
print(sys.version.split()[0], metadata.version("antares-analysis"), metadata.version("antares-client"), L.__file__)'
# Expect: 3.11.16, the release's package version, 1.14.0, a path under $RELEASE_ROOT/venv/
SITE=$("$PY" -I -B -c 'import pathlib, src.operations as o; print(pathlib.Path(o.__file__).parent)')
echo "$PROVIDER_SHA256  $SITE/live_antares.py" | sha256sum -c && echo "$SCIENCE_SHA256  $SITE/science.py" | sha256sum -c
echo "$P2_TRANSPORT_SHA256  $SITE/p2_transport.py" | sha256sum -c
"$PY" -I -B -c 'from src.operations import live_antares as L, science as S, backfill as B
L.g663_canary_p2_profile()
print(L.G663_CANARY_P2_PROFILE_SHA256, L.G663_CANARY_TRANSPORT_SHA256, S.QUALIFIED_P2_PROFILES, [p.name for p in B.QUALIFIED_SOURCE_PROFILES])'
# Expect: the two digests above, (), ['historical-P1']
test ! -e /astro/store/shire/ANTARES/cache || echo "STOP: cache exists"
pgrep -u "$(id -u)" -af 'src\.operations|production_range|production_canary|antares-analysis' || echo "no writer"
"$PY" -I -B -c 'from src.operations.publication import authoritative_nights, read_publication_gate
from src.operations.storage import PRODUCTION_DATA_ROOT
print(authoritative_nights(PRODUCTION_DATA_ROOT)[-1], read_publication_gate(PRODUCTION_DATA_ROOT))'
# Expect: 2026-07-06 None (re-check after the canary: unchanged)
unshare --user --map-root-user --net -- true && echo "netns available" || echo "STOP: no empty network namespace"
# The transport child must start, prove its bytes and stop cleanly in this release (no request, no network).
unshare --user --map-root-user --net -- "$PY" -I -B - <<'EOF'
# g663-child-preflight v1 (offline; starts and stops one transport child, sends no request)
import json, os, time
from src.operations import live_antares as L, p2_transport as T

limits = L.g663_canary_p2_profile().transport
assert T.implementation_sha256() == os.environ["P2_TRANSPORT_SHA256"], "transport child bytes"
assert T.CHILD_TEST_HOOK is None, "test hook set"
child = T.P2ProcessTransport(L._p2_child_config(limits, L.OFFICIAL_API_BASE_URL))
child._start(time.monotonic() + limits.child_startup_seconds)  # READY proves the bytes and no src import
record = child.close()
assert child.state == T.CLOSED and record["reaped"] and record["returncode"] == 0, record
print(json.dumps({"stage": "child-preflight", "startup_seconds": child.stats["startup_seconds"],
                  "child_implementation_sha256": T.implementation_sha256(), "exit": record}, sort_keys=True))
EOF
test ! -e "$CANARY_ROOT" && test ! -e "$EVIDENCE_DIR" || echo "STOP: root/evidence exists; never reuse"
mkdir -m 0700 "$CANARY_ROOT" && mkdir -p -m 0700 "$EVIDENCE_DIR"
declare -px RELEASE_SHA WHEEL_SHA256 RELEASE_ROOT PY PROVIDER_SHA256 P2_TRANSPORT_SHA256 SCIENCE_SHA256 \
  PROFILE_SHA256 TRANSPORT_SHA256 NIGHT RUN_ID CANARY_ROOT EVIDENCE_DIR > "$EVIDENCE_DIR/canary.env"
```

Any line printing `STOP`, or any unexpected value, stops the procedure. Do
not improvise; ask Control.

### 7.2 Live acquisition of one UTC night (the only live step)

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" -I -B - > "$EVIDENCE_DIR/acquire-$TS.json" 2> "$EVIDENCE_DIR/acquire-$TS.log" <<'EOF'
# g663-canary-acquire v2 (NOT EXECUTED in G6.6.3A or G6.6.3D)
import hashlib, json, os
from pathlib import Path
from src.operations import backfill as B, live_antares as L, production_range as R
from src.operations.fetch_checkpoint import SegmentedFetchCheckpoint
from src.operations.query_checkpoint import load_query_result_checkpoint, seal_query_result_checkpoint

env = os.environ
night, run_id, release = env["NIGHT"], env["RUN_ID"], env["RELEASE_SHA"]
assert night in ("2026-07-07", "2026-07-13") and run_id == f"g663-p2-{night}-v1", (night, run_id)
assert hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest() == env["PROVIDER_SHA256"], "provider bytes"
assert L._p2_transport.implementation_sha256() == env["P2_TRANSPORT_SHA256"], "transport child bytes"
profile = L.g663_canary_p2_profile()  # refuses unless its canonical bytes hash to the frozen identity
assert (L.G663_CANARY_P2_PROFILE_SHA256, L.G663_CANARY_TRANSPORT_SHA256) == (
    env["PROFILE_SHA256"], env["TRANSPORT_SHA256"]), "profile identity"
root = L.ARNOR_CANARY_ROOT / run_id
capability = L.LiveAntaresReadCapability.for_arnor_commissioning(
    root, run_id=run_id, target_date_utc=night, release_sha=release, authority=L.LIVE_ANTARES_READ)
adapter = R.LiveRangeAdapter(root, release, None, proof_profile=profile)
provider = L.LiveAntaresProvider(capability, proof_profile=profile)
assert provider.execution_policy() == adapter.execution_policy()
request = adapter.acquisition_request(night)
segment_size = B.BackfillSettings().segment_size
configuration = B.configuration_sha256(release, adapter, segment_size)
bindings = B.query_checkpoint_bindings(run_id, release, configuration, adapter, request)
if (root / "request.json").exists():  # resuming this same root only
    assert json.loads((root / "request.json").read_text()) == B._request_document(request), "request"
else:
    B._write_json_new(root / "request.json", B._request_document(request))
query_dir = root / "checkpoints" / "query-result"
if (query_dir / "COMMITTED.json").is_file():
    loaded = load_query_result_checkpoint(root, request, bindings)
else:
    assert not query_dir.exists(), "uncommitted query checkpoint residue: STOP and ask Control"
    result = provider.query_resumable(request, bindings)
    details = result.evidence.details
    print(json.dumps({"stage": "query", "clean": result.clean,
        "errors": [issue.code for issue in result.evidence.errors],
        **{key: details.get(key) for key in (
            "completion_classification", "terminal_evidence", "p2_budget", "search_request_count",
            "split_count", "accepted_tile_count", "returned_loci", "retry_count",
            "retry_exception_types", "client")}}, sort_keys=True), flush=True)
    result.require_completed()  # any refusal or limit stops here: nothing is sealed
    seal_query_result_checkpoint(root, result, bindings)
    loaded = load_query_result_checkpoint(root, request, bindings)
ids = B.BackfillController._ordered_ids(loaded)
binding = B.fetch_checkpoint_binding(run_id, release, configuration, adapter, request, loaded, segment_size)
completion = SegmentedFetchCheckpoint.open(capability, binding).fetch_missing(
    ids, lambda locus_ids: provider.fetch_segment(request, locus_ids))
details = loaded.query_result.evidence.details
print(json.dumps({"stage": "acquired", "night": night, "run_id": run_id, "release_sha": release,
    "configuration_sha256": configuration, "query_integrity_sha256": loaded.integrity_sha256,
    "query_contract_sha256": details["query_contract_sha256"], "terminal_evidence": details["terminal_evidence"],
    "p2_budget": details["p2_budget"], "search_request_count": details["search_request_count"],
    "split_count": details["split_count"], "accepted_tile_count": details["accepted_tile_count"],
    "loci": len(ids), "fetch": completion.as_dict()}, sort_keys=True, indent=2))
EOF
echo "exit=$?"
```

An interrupted run (dropped session, kill or network error during fetch)
resumes by re-running exactly this block in the same root. The sealed query
is reused and only missing fetch segments are fetched.

A P2 refusal is terminal for the root. The query journal records a failure
reason, and a re-run reproduces it from the journal without contacting
ANTARES. The possible reasons are:

* `p2_transport_guard`;
* `p2_retry_exhausted`;
* `p2_*_exhausted`;
* `p2_malformed`;
* `p2_midpoint_collapse`.

Never delete, edit or reuse a root. A new attempt needs a new `-v2` root and
Control authorization.

### 7.3 Network-disabled verification (no ANTARES access possible)

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
unshare --user --map-root-user --net -- "$PY" -I -B - > "$EVIDENCE_DIR/verify-$TS.json" \
  2> "$EVIDENCE_DIR/verify-$TS.log" <<'EOF'
# g663-canary-verify v1 (NOT EXECUTED in G6.6.3A or G6.6.3D)
import hashlib, json, os, socket
from pathlib import Path
from unittest import mock
from src.operations import backfill as B, live_antares as L, production_range as R, science as S

env = os.environ
night, run_id, release = env["NIGHT"], env["RUN_ID"], env["RELEASE_SHA"]
assert [name for _index, name in socket.if_nameindex()] == ["lo"], "not an empty network namespace"
assert hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest() == env["PROVIDER_SHA256"], "provider bytes"
profile = L.g663_canary_p2_profile()
root = L.ARNOR_CANARY_ROOT / run_id


def inventory(top):
    rows = []
    for path in sorted(top.rglob("*")):
        assert not path.is_symlink(), path
        if path.is_file():
            rows.append([str(path.relative_to(top)), path.stat().st_size,
                         hashlib.sha256(path.read_bytes()).hexdigest()])
    return rows


def refuse(*_args, **_kwargs):
    raise RuntimeError("network access during offline verification")


before = inventory(root)
# Verification-only trust, in this process only: never a production registry entry.
verifier = B.SourceProfile("g663-canary-verification-only", frozenset({env["PROVIDER_SHA256"]}), profile)
consumer = R.LiveRangeAdapter(R.RANGE_WORK_PARENT / "g663-verification-unused", release, None)
with mock.patch.object(socket.socket, "connect", refuse), mock.patch.object(socket, "getaddrinfo", refuse), \
        mock.patch.object(B, "QUALIFIED_SOURCE_PROFILES", (*B.QUALIFIED_SOURCE_PROFILES, verifier)), \
        mock.patch.object(S, "QUALIFIED_P2_PROFILES", (profile,)):
    saved = B.describe_saved_acquisition(root, night, consumer, B.BackfillSettings().segment_size)
    assert saved.source_profile is verifier
    descriptor = B.selection_descriptor_for_request(consumer.acquisition_request(night))
    assert saved.selection_descriptor == descriptor
    replay = S.validate_p2_query_result(saved.request, saved.loaded.query_result, profile)
    science = R.LiveRangeAdapter(root, release, None, proof_profile=profile).construct_checkpoint(
        saved.request, saved.loaded.query_result, saved.checkpoint)
    artifacts = S.build_night_artifacts(science)
    S.reopen_and_validate_artifacts(artifacts, expected=science)
after = inventory(root)
assert after == before, "verification changed the canary root"
details = saved.loaded.query_result.evidence.details
print(json.dumps({"stage": "verified", "night": night, "run_id": run_id,
    "adoption_entry": saved.entry, "selection_descriptor": B.selection_descriptor_identity(descriptor),
    "terminal_evidence": details["terminal_evidence"], "p2_budget": replay["budget"],
    "accepted_tiles": replay["accepted"], "splits": replay["splits"], "search_attempts": replay["search_attempts"],
    "client": details["client"], "loci": len(science.loci), "alert_rows": len(science.alerts),
    "artifact_sha256": {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(artifacts.items())},
    "root_files": len(after), "root_inventory_sha256": hashlib.sha256(json.dumps(after).encode()).hexdigest()},
    sort_keys=True, indent=2))
EOF
echo "exit=$?"
```

### 7.4 Canary PASS criteria (each night)

All of the following must hold, and all evidence stays in `$EVIDENCE_DIR`.

1. **Preflight (7.1)** printed no `STOP` and only the expected identities,
   and the transport-child preflight exited 0 with `returncode` 0.
   The authority tail (`2026-07-06`) and publication gate (`None`) are
   unchanged afterwards. The cache is still absent.
2. **Acquisition (7.2)** exited 0. Its final document has
   `terminal_evidence` = `natural-exhaustion-below-50` and
   `p2_budget.secondary_nodes` > 0 (the floor saturation was actually
   resolved by P2). `client` equals the guarded identity with
   `transport_sha256` = `6e0291f3…`. The fetch completed every object.
3. **Verification (7.3)**, run in an empty network namespace, exited 0. It
   re-proved the sealed query, the journal, P2 replay, the fetch
   checkpoint, the selection descriptor and artifact reopen. The canary
   root inventory was unchanged. `adoption_entry` is the exact identity a
   later, separately authorized production adoption would bind.

### 7.5 Canary FAIL criteria (each night)

Any of the following fails the night. On any FAIL: stop, keep everything,
report the acquisition and verification evidence to Control, and never
retry in the same root.

* Any `STOP`, non-zero exit or traceback.
* Any P2 failure reason:
  * `p2_transport_guard`, with its `exception_type` in the journal;
  * `p2_retry_exhausted`;
  * `p2_depth_exhausted`, `p2_root_nodes_exhausted`,
    `p2_night_nodes_exhausted`, `p2_search_budget_exhausted` or
    `p2_crash_reserve_exhausted`;
  * `p2_midpoint_collapse` or `p2_malformed`.
* An incomplete fetch.
* Any verification refusal.
* Any change to production or the root during verification.

## 8. Remaining risks

* **Real link shape is unknown offline.** If ANTARES emits `http://`,
  foreign-origin, relative or other-path `links.next`, the canary fails
  closed on the first multi-page node (`P2Insecure…`,
  `P2CrossOrigin…`, `P2RelativeContinuationError`, …). That would be
  evidence for Control, never science.
* **Transient outages are terminal for a P2 root.** G6.6.2B's P2 grammar
  has no invocation-boundary renewal. A gateway outage that spans both
  attempts of one node (0.5 s apart) ends the canary with
  `p2_retry_exhausted`, and the next attempt needs a new root. The G6.2
  evidence shows non-JSON 502 pages do occur.
* **Floor-saturation scale on Jul07/Jul13 is unknown.** The P1 failure
  stopped at the first saturated floor. Node budgets that bind fail closed
  (`p2_*_nodes_exhausted`). They never truncate science.
* **Transport child on Arnor (G6.6.3D).** Each P2 night query runs one
  child process of the release's own interpreter (`-I -B`), in its own
  session, with stdio on `/dev/null`. The 7.1 preflight proves that it
  starts, proves its bytes and stops cleanly in the release environment
  before any live step. A child failure during the canary (startup, IPC,
  kill) fails the root closed with a typed `P2Transport…Error`, never
  science. A child stuck in uninterruptible kernel sleep (for example on a
  hung filesystem) may not be reapable within the 5 s join; that is fatal
  for the night (`P2TransportUnkillableError`) and is reported, never
  ignored.
* **Server-defined continuation content.** The guard constrains a
  continuation's origin and path, not its query string. Pagination content
  remains as trusted as it is for the pinned client and every historical P1
  night. Completeness still requires natural exhaustion below 50 at every
  leaf.
* **Journal size.** About two P2 journal events per search (for example
  ~86k for a Jul09-scale night), and the sealed query evidence embeds all
  events. Memory and journal cost scale like P1's per-tile journal, at
  about twice the event count.
* **Redirect cookies.** A same-origin redirect hop inside one page does not
  replay a `Set-Cookie` from the 3xx response, as `requests`' own redirect
  loop would. The public search API is not known to use cookies. A
  cookie-dependent redirect would surface as a refusal or a transport
  failure, never as science.
* **Decompression.** Byte ceilings count decoded bytes, read in 8 KiB
  chunks. Per-read decompression amplification depends on the release's
  pinned urllib3 (≥ 2.6 bounds it).
* **No production approval.** Both registries stay empty for P2. Adopting a
  verified canary into the Jul07–13 ALL-ADOPTION range requires a later
  Control-approved release that registers exactly this profile and
  provider SHA.
