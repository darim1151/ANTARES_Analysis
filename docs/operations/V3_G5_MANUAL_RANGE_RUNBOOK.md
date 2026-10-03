# V3-G5 manual production range runbook

Release **v0.4.5** (`7211b5c2bce60edd24ca92309d33f0e81b412821`), qualified on Arnor on 2026-09-30. Evidence:
[`V3_G5_PLATFORM_RELEASE_QUALIFICATION.md`](V3_G5_PLATFORM_RELEASE_QUALIFICATION.md).

This procedure acquires ANTARES data for a Control-authorized range of UTC nights,
builds and validates one candidate per night in date order, and publishes each
night into production authority. A human starts every live run. Nothing in the
package creates a Control token or a Control approval.

- **Publication is sequential**, even though acquisition runs concurrently.
  Exactly one writer publishes, in date order. A night is published only after
  its predecessor is verified COMPLETE, and nothing crosses a blocked night.
- **Cache: none.** `/astro/store/shire/ANTARES/cache` must stay absent, because
  it is a Sentinel V2 predicate. No shared or segment cache is enabled: the
  authorization binds `cache_root: null`. Never create a cache directory.
- **Scope.** At most **31 nights** per authorization. Execution runs only on
  `arnor.astro.washington.edu`, as UID `1533564`, from the immutable release
  interpreter.

## 1. Preconditions

Stop and ask Control if any of these does not hold:

1. Section 4 `inspect` shows every night in the range as `PLANNED`, with
   authority `NOT_COMMITTED`.
2. The section 6 script reports the start date's predecessor as `COMPLETE`
   and finalized, with no publication gate and no later authoritative night.
   The current Sentinel V2 must equal the predecessor's resulting fingerprint.
3. `/astro/store/shire/ANTARES/cache` is absent, and
   `/astro/store/shire/ANTARES/work/publication/staging/` is empty.
4. There is no running range controller, publisher or other writer. Use
   `ps -u mdarim -f`.
5. Control has approved this exact range, `RUN_ID` and expiry.

## 2. Release and interpreter

Always use these exact values. Never use a mutable alias, a different venv, or
`python -m src.operations.production_range`. The `-m` form executes the module
as `__main__`, and the adapter attestation then refuses to run. Do every step
inside one `tmux` session, so that a dropped SSH connection cannot interrupt a
live run. Re-attach with `tmux attach -t antares-range`.

```bash
tmux new -s antares-range
export RELEASE_SHA=7211b5c2bce60edd24ca92309d33f0e81b412821
export RELEASE_ROOT=/astro/users/mdarim/opt/antares-analysis/releases/$RELEASE_SHA
export PY=$RELEASE_ROOT/venv/bin/python
export WHEEL_SHA256=e2461eee1f84c85d176045bce7fcc789898f045d491ab4aaa6f92231b376aff7
export ENTRY='import sys; from src.operations.production_range import main; sys.exit(main())'
cat "$RELEASE_ROOT/RELEASE_SHA" "$RELEASE_ROOT/WHEEL_SHA256" "$RELEASE_ROOT/PACKAGE_VERSION"
"$PY" -I -c 'import sys; from importlib import metadata; import src.operations.production_range as r; print(sys.version.split()[0], metadata.version("antares-analysis"), r.__file__)'
# Expect: the two hashes above, 0.4.5, "3.11.16 0.4.5 $RELEASE_ROOT/venv/lib/python3.11/site-packages/src/operations/production_range.py"
```

## 3. Choose the range and names

- `START_DATE` must be the day after the current authority tail. Check it with
  `inspect` or the section 6 script. `END_DATE` is inclusive, at or after
  `START_DATE`, and at most 30 days later (31 nights). Choose only nights that
  have completely ended in UTC.
- Give `RUN_ID` a new, unique name that is never reused, for example
  `g5-range-2026-06-28_2026-06-29-v1`.

```bash
export START_DATE=YYYY-MM-DD
export END_DATE=YYYY-MM-DD
export RUN_ID=g5-range-${START_DATE}_${END_DATE}-v1
export WORK_ROOT=/astro/store/shire/ANTARES/work/backfill/$RUN_ID
export CONTROL_DIR=/astro/users/mdarim/antares-control/ranges/$RUN_ID
export CONTROL_TOKEN_FILE=/astro/users/mdarim/antares-control/tokens/$RUN_ID.token
export AUTHORIZATION=$CONTROL_DIR/authorization.json
export APPROVAL=$CONTROL_DIR/approval.json
umask 077; mkdir -p "$CONTROL_DIR" "$(dirname "$CONTROL_TOKEN_FILE")"
declare -px RELEASE_SHA RELEASE_ROOT PY WHEEL_SHA256 ENTRY START_DATE END_DATE RUN_ID WORK_ROOT \
  CONTROL_DIR CONTROL_TOKEN_FILE AUTHORIZATION APPROVAL > "$CONTROL_DIR/range.env"
# In any new shell or tmux window: source /astro/users/mdarim/antares-control/ranges/<RUN_ID>/range.env
```

## 4. Plan and inspect (offline and read-only)

`plan` touches neither the network nor production. It only hashes the
installed sources. `inspect` only reads, under a shared authority lock.
Neither command creates `WORK_ROOT`.

```bash
"$PY" -I -B -c "$ENTRY" plan --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA > "$CONTROL_DIR/plan.json"
"$PY" -I -B -c "$ENTRY" inspect --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA
```

`plan.json` must show:

- `"execution": "NOT EXECUTED"`
- `"publication_concurrency": 1`
- `range_binding.cache_root: null`
- `acquisition_identity.adapter == "src.operations.production_range.LiveRangeAdapter"`
- For 2026-06-28 → 2026-06-29, `range_binding` equals:
  - `science_contract_sha256`: `9de004e6e8bc3af0e029c3adfb78f4cc34e937075ba5377de7891017cb9da801`
  - `prior_free_attestation_sha256`: `ed3ccca6589b475dbec62777fc8f4da5c4ac2857ce740819736945450d358f26`
  - `configuration_sha256`: `8e5111a55e4999aca869c04ffb818d1ded1fd1a736f325f62fa6f2cad0abcba7`

  For other ranges, the science and attestation values differ. Section 6
  re-derives and checks them.

## 4a. Optional: adopt saved acquisitions instead of querying

A night that already has complete, sealed query and fetch evidence from an
earlier run can be adopted instead of queried again. Adoption never contacts
ANTARES and never writes the source. The adoptable sources are:

- a direct child of `/astro/store/shire/ANTARES/work/canary/`;
- a `nights/night-YYYY-MM-DD` root of another range under
  `/astro/store/shire/ANTARES/work/backfill/`.

Name each adopted night on `plan` with `--adopt DATE=SOURCE_ROOT`:

```bash
"$PY" -I -B -c "$ENTRY" plan --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA \
  --adopt 2026-06-30=/astro/store/shire/ANTARES/work/canary/<run-id> > "$CONTROL_DIR/plan.json"
```

For each adopted night, `plan` re-proves the source read-only and fails on
any difference:

- the sealed query checkpoint and every fetch blob;
- the request and the scientific contract;
- the configuration, including segment size;
- the journal identity;
- a reviewed prior-free provider.

It then writes the exact evidence identities into
`range_binding.adopted_acquisitions`. Section 6 binds them into the
authorization unchanged, and the section 7 approval covers them.

Execution re-proves each source at run start and again at construction. It
records `adoption.json` in the night's own workspace. It rebuilds the
candidate against the current prior, so later nights include earlier
published nights. It publishes through the normal gated path. Publication
refuses any candidate whose `acquisition_source` provenance differs from the
authorization. `execute` refuses `--adopt`: adoption exists only through an
approved authorization.

## 5. Control step: generate the Control token

Control generates a fresh 256-bit token for this `RUN_ID` only, keeps it private,
and never commits, pastes or reuses it.

```bash
umask 077
python3 -c 'import secrets; print(secrets.token_hex(32))' > "$CONTROL_TOKEN_FILE"
chmod 600 "$CONTROL_TOKEN_FILE"
export CONTROL_TOKEN_SHA256=$(tr -d '\n' < "$CONTROL_TOKEN_FILE" | sha256sum | cut -d' ' -f1)
echo "$CONTROL_TOKEN_SHA256"
```

## 6. Create the ProductionRangeAuthorization

The authorization uses the package's own classes. It captures a fresh, read-only
Sentinel V2 for the initial state. `ProductionRangeAuthorization` re-derives and
checks the science contract, configuration, adapter attestation, implementation
hashes, roots, host, UID, release and wheel, and refuses if anything differs.
The authorization grants no write capability by itself. It also writes no
approval.

```bash
export AUTHORIZED_BY="ANTARES-Control"   # the approving person or role, never "range:..."
export EXPIRES_HOURS=72                  # Control's choice (decimal allowed); execute/--resume refuse after expiry
"$PY" -I -B - <<'EOF'
import hashlib, json, os, socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.operations import backfill as B, production_range as R
from src.operations.commissioning import capture_production_sentinel
from src.operations.publication import (authoritative_nights, classify_night_authority,
    production_binding_from_sentinel, read_publication_gate)
from src.operations.storage import PRODUCTION_CONTROL_ROOT, PRODUCTION_DATA_ROOT
e = os.environ
start, end, release, wheel = e["START_DATE"], e["END_DATE"], e["RELEASE_SHA"], e["WHEEL_SHA256"]
work, control = R.RANGE_WORK_PARENT / e["RUN_ID"], Path(e["CONTROL_DIR"])
plan = json.loads((control / "plan.json").read_text())
assert plan["execution"] == "NOT EXECUTED" and plan["work_root"] == str(work)
assert len(B._dates(start, end)) <= R.MAX_PRODUCTION_RANGE_NIGHTS
predecessor = B._previous(start)
state = classify_night_authority(PRODUCTION_DATA_ROOT, PRODUCTION_CONTROL_ROOT / "journals", predecessor)
assert state["state"] == "COMPLETE" and state["finalized"] and not state["gate_present"], state
assert read_publication_gate(PRODUCTION_DATA_ROOT) is None
assert authoritative_nights(PRODUCTION_DATA_ROOT)[-1] == predecessor, "authority tail is not START_DATE-1"
sentinel = capture_production_sentinel(PRODUCTION_DATA_ROOT, R.SENTINEL_CACHE, start)
production = production_binding_from_sentinel(sentinel)
assert production["predicates"]["cache_absent"] and production["predicates"]["target_absent"]
assert production["predicates"]["transaction_artifacts"] == []
assert production["durable_fingerprint_sha256"] == state["resulting_production_fingerprint"], "Sentinel drift"
now = datetime.now(timezone.utc).replace(microsecond=0)
scope = B.RangePublicationAuthorization(
    start, end, predecessor, {key: production[key] for key in B._INITIAL_SENTINEL_FIELDS},
    candidate_release_sha=release, publisher_release_sha=release, authorized_by=e["AUTHORIZED_BY"],
    authorized_at_utc=now.isoformat(), expires_at_utc=(now + timedelta(hours=float(e["EXPIRES_HOURS"]))).isoformat(),
    **plan["range_binding"])
authority = R.ProductionRangeAuthorization(scope, str(work), socket.getfqdn(), os.geteuid(),
    e["CONTROL_TOKEN_SHA256"], R.implementation_identity(), publisher_wheel_sha256=wheel)
assert dict(authority.implementation_sha256) == plan["implementation_sha256"]
authority.verify()  # host, UID, expiry, release markers, 3.11.16, module origin (no token needed)
for name, document in (("initial-sentinel.json", sentinel), ("authorization.json", authority.as_dict())):
    fd = os.open(control / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(document, stream, sort_keys=True, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
print("initial Sentinel", production["durable_fingerprint_sha256"], "manifests", production["manifest_count"])
print("range scope digest", scope.digest)
print("authorization digest", authority.digest)
print("expires", scope.expires_at_utc)
EOF
```

Control reviews `authorization.json`. The range, `work_root`, `hostname`,
`service_uid`, `control_token_sha256`, `publisher_wheel_sha256`,
`scope.initial_sentinel` and `scope.expires_at_utc` must be exactly as intended.

## 7. Control step: detached approval (outside the package)

Only Control runs this, deliberately, after the section 6 review. It uses the
standard library only and does not import ANTARES. It reads the exact
authorization file and the private token, computes the authorization digest
and the canonical approval body expected by `ControlRangeApproval`
(schema `v3.control-range-approval.v1`, HMAC-SHA256 keyed by the token, over
sorted-key, `,`/`:`-separated UTF-8 JSON), and writes a new private file. It
never queries, acquires or publishes anything.

```bash
export APPROVED_BY="Darim Shamsi (ANTARES Control)"
umask 077
python3 - "$AUTHORIZATION" "$CONTROL_TOKEN_FILE" "$APPROVAL" "$APPROVED_BY" <<'EOF'
import hashlib, hmac, json, os, sys
from datetime import datetime, timezone
authorization_path, token_path, approval_path, approved_by = sys.argv[1:5]
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")
with open(authorization_path, encoding="utf-8") as stream:
    authorization = json.load(stream)
with open(token_path, encoding="ascii") as stream:
    token = stream.read().strip()
assert len(token) == 64 and set(token) <= set("0123456789abcdef"), "token must be 64 lowercase hex"
assert hashlib.sha256(token.encode("ascii")).hexdigest() == authorization["control_token_sha256"], "wrong token"
assert approved_by.strip() and not approved_by.startswith("range:")
now = datetime.now(timezone.utc).replace(microsecond=0)
expires = datetime.fromisoformat(authorization["scope"]["expires_at_utc"].replace("Z", "+00:00"))
assert now < expires, "authorization already expired"
body = {"schema_version": "v3.control-range-approval.v1",
        "range_authorization_sha256": hashlib.sha256(canonical(authorization)).hexdigest(),
        "approved_by": approved_by, "approved_at_utc": now.isoformat()}
approval = dict(body, approval_hmac_sha256=hmac.new(token.encode("ascii"), canonical(body), hashlib.sha256).hexdigest())
fd = os.open(approval_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, "wb") as stream:
    stream.write(canonical(approval)); stream.flush(); os.fsync(stream.fileno())
os.chmod(approval_path, 0o600)
print("authorization sha256", body["range_authorization_sha256"])
print("approval identity sha256", hashlib.sha256(canonical(approval)).hexdigest())
EOF
sha256sum "$APPROVAL"; ls -l "$APPROVAL"   # -rw------- ; file bytes are the canonical JSON, so both hashes agree
```

## 8. Read-only verification before starting

This check changes nothing. It must print the same two digests as sections 6 and 7:

```bash
"$PY" -I -B - "$AUTHORIZATION" "$CONTROL_TOKEN_FILE" "$APPROVAL" <<'EOF'
import json, sys
from pathlib import Path
from src.operations import production_range as R
authority = R.ProductionRangeAuthorization.from_dict(json.loads(Path(sys.argv[1]).read_text()))
token = Path(sys.argv[2]).read_text().strip()
approval = R.ControlRangeApproval.load(Path(sys.argv[3]), work_root=authority.work_root)
authority.verify(control_token=token)
approval.verify(authority, control_token=token)
print("authorization", authority.digest); print("approval", approval.sha256); print("expires", authority.scope.expires_at_utc)
EOF
```

## 9. Start the range (manual live command)

The work root must be a new, real, empty directory. Run inside `tmux`, so that a
dropped SSH session does not interrupt the controller. This is the `tmux`
session from section 2. Keep the output outside the work root.

```bash
umask 077; mkdir -p /astro/store/shire/ANTARES/work/backfill; chmod 700 /astro/store/shire/ANTARES/work/backfill
mkdir -m 0700 "$WORK_ROOT"                      # fails if it already exists: pick a new RUN_ID
TS=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" -I -B -c "$ENTRY" execute --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA \
  --publisher-wheel-sha256 $WHEEL_SHA256 --authorization "$AUTHORIZATION" \
  --control-token-file "$CONTROL_TOKEN_FILE" --control-approval "$APPROVAL" \
  --acquisition-concurrency 2 --execute-authorized-range \
  > "$CONTROL_DIR/execute-$TS.json" 2> "$CONTROL_DIR/execute-$TS.log"
echo "exit=$?"
```

The command prints one range document. Success means `summary.nights_published`
equals `summary.nights_total`, and `publication_frontier` is `null`. A non-zero
exit or a traceback in the `.log` file is a refusal or interruption. Read
section 11 before doing anything else.

## 10. Watch progress safely

`inspect` (section 4) is read-only and safe while the controller runs. Other
read-only views:

```bash
cat "$WORK_ROOT/ranges/${START_DATE}_${END_DATE}.json"        # last range document written by the controller
tail -n 5 "$WORK_ROOT/nights/night-$START_DATE/events.jsonl"   # per-night append-only events
```

Stage names are derived from durable evidence:

| Meaning | Stage in `inspect` | Evidence |
|---|---|---|
| planned, nothing acquired | `PLANNED` | no sealed query |
| query sealed / fetch in progress | `QUERY_COMPLETE` / `FETCHING` | `checkpoints/…/COMMITTED.json`, segment commits |
| **acquired** | `FETCH_COMPLETE` | `checkpoints/live-fetch-v1/fetch-complete.json` |
| **constructed** (validated candidate, not authority) | `WAITING_FOR_PUBLICATION` | `candidate/candidate-record.json` |
| **published** | `PUBLISHED` | authority `COMPLETE`, finalized journal |
| **blocked** | `BLOCKED` | last event `failure`; see `blocked.failure_category` and `blocked.retryable` |
| **reconciliation required** | `RECONCILIATION_REQUIRED` | publication gate present or unfinished publication journal |

Events also record transient stages: `QUERYING`, `CANDIDATE_BUILDING`,
`CANDIDATE_VALIDATED` and `PUBLISHING`.

## 11. Ordinary interruption: `--resume`

Use this after a dropped session or kill, a network error, or a `BLOCKED` night
with `retryable: true`. The retryable categories are `lock_contention`,
`interrupted_before_commit`, `reconciliation_interrupted` and
`transient_network`. Run the **same** section 9 command with `--resume`
added, before the authorization expires. Resume reuses every sealed query,
committed segment and valid candidate, and never re-queries or re-publishes
completed work. An unresolved publication is always rolled forward first.

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" -I -B -c "$ENTRY" execute --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA \
  --publisher-wheel-sha256 $WHEEL_SHA256 --authorization "$AUTHORIZATION" \
  --control-token-file "$CONTROL_TOKEN_FILE" --control-approval "$APPROVAL" \
  --acquisition-concurrency 2 --execute-authorized-range --resume \
  > "$CONTROL_DIR/resume-$TS.json" 2> "$CONTROL_DIR/resume-$TS.log"
```

Before resuming, check that no other controller is running.
`Another backfill controller is active` means one still is.

## 12. `recover`: only for an already-journaled publication

`recover` accepts neither a token nor an approval, and works after expiry. It
only finishes a publication transaction that is already journaled or gated
(`RECONCILIATION_REQUIRED`) for this exact authorization, or re-emits the
terminal record of a night that is already `COMPLETE`. It **cannot** query,
fetch, construct a candidate, or start any new publication. It stops at the
first night that is not in one of those states.

```bash
"$PY" -I -B -c "$ENTRY" recover --start $START_DATE --end $END_DATE \
  --work-root $WORK_ROOT --candidate-release $RELEASE_SHA \
  --publisher-wheel-sha256 $WHEEL_SHA256 --authorization "$AUTHORIZATION" \
  --execute-authorized-range --resume
```

## 13. Stop conditions: ask Control, do not improvise

- Any night is `BLOCKED` with `retryable: false`, for example
  `sentinel_drift`, `predecessor_gap`, `candidate_corruption`,
  `journal_contradiction`, `authorization_*`, or a `CONTRADICTION` authority.
- `RECONCILIATION_REQUIRED` remains after one `recover` or `--resume`.
- The authorization expires before all nights are published. Resume is then
  refused, and a new authorization and approval are required.
- Any refusal about release, wheel, host, UID, token, approval, implementation
  identity or adapter attestation.
- `/astro/store/shire/ANTARES/cache`, a stray gate, or staging residue appears.
- Any wish to delete, edit or move files under `data/`, `work/publication/`
  or a night's checkpoints. Never do this by hand.

## 14. Afterwards

- Keep `CONTROL_DIR`, `WORK_ROOT` and the logs as evidence.
- Retire the token: `shred -u "$CONTROL_TOKEN_FILE"`, once Control confirms no
  resume or recovery is needed.
- The next range starts at the new authority tail + 1, with a new `RUN_ID`,
  token, authorization and approval.

## Appendix: 2026-06-28 → 2026-06-29 (prepared, NOT EXECUTED)

Starting state verified read-only from the installed v0.4.5 release on
2026-09-30T17:49:26Z. Recheck it with sections 4 and 6 immediately before use.

| Item | Value |
|---|---|
| Release / candidate / publisher SHA | `7211b5c2bce60edd24ca92309d33f0e81b412821` (tag `v0.4.5`) |
| Interpreter | `/astro/users/mdarim/opt/antares-analysis/releases/7211b5c2bce60edd24ca92309d33f0e81b412821/venv/bin/python` (CPython 3.11.16) |
| Wheel SHA-256 | `e2461eee1f84c85d176045bce7fcc789898f045d491ab4aaa6f92231b376aff7` |
| Production root | `/astro/store/shire/ANTARES/data` (NFSv4.2 `shire.infiniband:/data/shire` on `/astro/store/shire`) |
| Initial predecessor | `2026-06-27`: `COMPLETE`, finalized, transaction `v3pub-2026-06-27-197eca1861cb-a1` |
| Starting Sentinel V2 | `50cf323d43f16ccb0d092a01b9df767b3297028f48b96ccbb61d5de5efd9cb05`: 91 manifests, 327 files, 1,271,751,077 bytes |
| Cumulative | `loci_index` `1cc0bad54423070cd8326db56409e618acb23123041070daab0b35314f1ca91f` (1,325,004 rows); `nightly_summary` `20a8230af18c3b5876ac778dc1747ffb91fe447caeea387da0d569fa1202d9f2` (91 rows) |
| Nights / MJD | `2026-06-28` [61219, 61220); `2026-06-29` [61220, 61221) |
| Plan binding | science `9de004e6…da801`, attestation `ed3ccca6…58f26`, configuration `8e5111a5…abcba7`, `cache_root: null`, publication concurrency 1 |
| Work root | `/astro/store/shire/ANTARES/work/backfill/g5-range-2026-06-28_2026-06-29-v1` |
| Authorization / approval | `/astro/users/mdarim/antares-control/ranges/g5-range-2026-06-28_2026-06-29-v1/{authorization,approval}.json` |
| Control token file | `/astro/users/mdarim/antares-control/tokens/g5-range-2026-06-28_2026-06-29-v1.token` |

The human still supplies only:

- the Control token, which Control generates in section 5;
- `AUTHORIZED_BY`, `APPROVED_BY` and `EXPIRES_HOURS`;
- a different `-vN` suffix, only if this `RUN_ID` was ever used before.

The authorization digest and approval identity are created at authorization
time.

```bash
# NOT EXECUTED — MANUAL LIVE COMMAND (run only after sections 4–8 pass and Control approves)
tmux new -s antares-range
export RELEASE_SHA=7211b5c2bce60edd24ca92309d33f0e81b412821
export RELEASE_ROOT=/astro/users/mdarim/opt/antares-analysis/releases/$RELEASE_SHA
export PY=$RELEASE_ROOT/venv/bin/python
export WHEEL_SHA256=e2461eee1f84c85d176045bce7fcc789898f045d491ab4aaa6f92231b376aff7
export ENTRY='import sys; from src.operations.production_range import main; sys.exit(main())'
export START_DATE=2026-06-28
export END_DATE=2026-06-29
export RUN_ID=g5-range-2026-06-28_2026-06-29-v1
export WORK_ROOT=/astro/store/shire/ANTARES/work/backfill/$RUN_ID
export CONTROL_DIR=/astro/users/mdarim/antares-control/ranges/$RUN_ID
export CONTROL_TOKEN_FILE=/astro/users/mdarim/antares-control/tokens/$RUN_ID.token
export AUTHORIZATION=$CONTROL_DIR/authorization.json
export APPROVAL=$CONTROL_DIR/approval.json
# ... sections 4–8 (plan, inspect, Control token, authorization, detached approval, verification) ...
umask 077; mkdir -p /astro/store/shire/ANTARES/work/backfill; chmod 700 /astro/store/shire/ANTARES/work/backfill
mkdir -m 0700 "$WORK_ROOT"
TS=$(date -u +%Y%m%dT%H%M%SZ)
"$PY" -I -B -c "$ENTRY" execute --start 2026-06-28 --end 2026-06-29 \
  --work-root /astro/store/shire/ANTARES/work/backfill/g5-range-2026-06-28_2026-06-29-v1 \
  --candidate-release 7211b5c2bce60edd24ca92309d33f0e81b412821 \
  --publisher-wheel-sha256 e2461eee1f84c85d176045bce7fcc789898f045d491ab4aaa6f92231b376aff7 \
  --authorization /astro/users/mdarim/antares-control/ranges/g5-range-2026-06-28_2026-06-29-v1/authorization.json \
  --control-token-file /astro/users/mdarim/antares-control/tokens/g5-range-2026-06-28_2026-06-29-v1.token \
  --control-approval /astro/users/mdarim/antares-control/ranges/g5-range-2026-06-28_2026-06-29-v1/approval.json \
  --acquisition-concurrency 2 --execute-authorized-range \
  > "$CONTROL_DIR/execute-$TS.json" 2> "$CONTROL_DIR/execute-$TS.log"
echo "exit=$?"
```

Expected result: both nights `PUBLISHED`, `summary.nights_published == 2`,
`publication_frontier: null`, authority tail `2026-06-29`, and a new Sentinel
V2 equal to the 2026-06-29 journal's `resulting_production_fingerprint`.
