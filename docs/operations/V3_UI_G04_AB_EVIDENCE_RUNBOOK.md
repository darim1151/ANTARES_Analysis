# V3-UI-G0.4A/B evidence harness — independent review required

This script consumes v0.4.7 read behavior, from source
`812c545e14693cdce7ff7458f1d2b50b0804dcd8`. It provides AP-0 through
AP-6 evidence for Control. It grants no authorization to access Arnor.
No Arnor command was executed during construction or local qualification.
No Console, production implementation, packaging, or configuration changes
are part of this branch. There is no Tier C CLI or execution path.

## Review inputs and release identity

Inspect these three files together:

- `scripts/v3_ui_g04_ab_evidence.py`: consumer, CLI, confinement and supervisor.
- `tests/test_v3_ui_g04_ab_evidence.py`: synthetic safety qualification.
- This runbook: proposed commands, interpretation and refusal contract.

The reviewed checkout must be a descendant of the exact baseline in
`darim1151/ANTARES_Analysis`. `verify_repository` checks origin, ancestry,
the complete `src/**/*.py` content digest, and `pyproject.toml` digest.
Only the harness, tests and document may differ from release source.
Tier A and plan import no ANTARES package. Tier B imports **installed release
modules**, from the exact release venv, checks version 0.4.7 / Python 3.11.16,
release markers and the complete installed source digest, and checks runtime
paths against the statically resolved paths. A checkout on `PYTHONPATH` cannot
replace the worker's installed release (`-I -B`).

`static_layout` reads the pinned source AST, without executing it. Storage's
`PRODUCTION_DATA_ROOT` is `/astro/store/shire/ANTARES/data`.
`history.survey_data_root(PRODUCTION_DATA_ROOT)` appends `data/lsst_only`:

```text
nightly:    /astro/store/shire/ANTARES/data/data/lsst_only/nightly/YYYY/MM/DD
cumulative: /astro/store/shire/ANTARES/data/data/lsst_only/cumulative
gate:       /astro/store/shire/ANTARES/data/data/lsst_only/PUBLICATION_TRANSACTION_IN_PROGRESS.json
journals:   /astro/store/shire/ANTARES/work/publication/control/journals
lock:       /astro/store/shire/ANTARES/work/publication/control/locks/publication-authority.flock
terminals:  /astro/store/shire/ANTARES/work/publication/evidence/publications/YYYY-MM-DD
events:     /astro/store/shire/ANTARES/work/backfill/RUN_ID/nights/night-YYYY-MM-DD/events.jsonl
```

These are source-derived addresses, **not observations of production**. The
older G5/G3 documents describe earlier releases; their historical inventories
are not current measurements. No material contradiction with the Control
packet's six expected authority semantics was found.

## Proposed commands — NOT EXECUTED

**DO NOT RUN UNTIL INDEPENDENT REVIEW + CONTROL AUTHORIZATION.**

Control must place the exact reviewed commit in the checkout below and confirm
that the exact baseline release is installed. These evidence directories have
an existing parent, are outside authority and Control secrets, and are private
mode 0700 if they already exist. They are evidence addresses, not Console-store
choices. Substitute a different absolute, safe evidence address if Control
requires it; protected roots and their ancestors are refused.

```bash
# 1. Static plan: no production metadata or contents read.
/astro/users/mdarim/opt/antares-analysis/releases/812c545e14693cdce7ff7458f1d2b50b0804dcd8/venv/bin/python -I -B \
  /astro/users/mdarim/antares-ui-g04-review/scripts/v3_ui_g04_ab_evidence.py \
  --plan --output-dir /astro/users/mdarim/antares-ui-g04-plan-evidence

# 2. AP-0 (60 seconds), then Tier A discovery.
/astro/users/mdarim/opt/antares-analysis/releases/812c545e14693cdce7ff7458f1d2b50b0804dcd8/venv/bin/python -I -B \
  /astro/users/mdarim/antares-ui-g04-review/scripts/v3_ui_g04_ab_evidence.py \
  --tier-a --output-dir /astro/users/mdarim/antares-ui-g04-tier-a-evidence

# 3. AP-0, fresh Tier A metadata, fresh AP-0 (another 60 seconds), then AP-6.
/astro/users/mdarim/opt/antares-analysis/releases/812c545e14693cdce7ff7458f1d2b50b0804dcd8/venv/bin/python -I -B \
  /astro/users/mdarim/antares-ui-g04-review/scripts/v3_ui_g04_ab_evidence.py \
  --tier-b --control-authorized-tier-b \
  --output-dir /astro/users/mdarim/antares-ui-g04-tier-b-evidence
```

No explicit tier means an argparse refusal before repository, host or
production inspection. The acknowledgement is an accidental-use barrier;
it does not replace independent review or Control authorization. There are no
CLI root overrides, retry flags, interval overrides or deadline overrides.
Tier B repeats Tier A discovery so its representative addresses come from
current metadata, without trusting an old evidence file.

## Protocol and interpretation

| Packet | Implementation | Focused test coverage |
|---|---|---|
| AP-0 | `ap0`, `generation`, `gate_clear`, `range_activity`, `event_tail`, `live_controllers` | `SafetyGateTests` |
| AP-1 | `ap1`, `mount_facts` | NFS protocol/cache options; UID refusal |
| AP-2 | `ap2`, `landlock_abi` | Fixed discovery reads; kernel unavailable refusal |
| AP-3 | `ap3`, `metadata`, `entries` | Actual directory tail, no content open, symlink/type STOP |
| AP-4 | `ap4` | Fixed interpreter probe and capacity outside authority |
| AP-5 | `POLICY_QUESTIONS`, `tier_a` | All six exact questions remain UNKNOWN |
| AP-6 | `run_tier_b`, `supervise`, `worker_main`, `ap6_locked`, `parquet_rows` | `ReleaseReadTests`, `SupervisorTests` |

AP-0 samples cumulative `(st_ino, st_size, st_mtime_ns)` twice, 60 seconds
apart. It observes the gate, activity and controllers at both ends. Only the
four individual true results allow `tier_b_safe_to_attempt=true`; missing
cumulative files are unsafe. A bounded event tail must cover the entire
30-minute window, or it refuses. Recent QUERYING through PUBLISHING remains
unsafe even if a later terminal event was appended. Missing/unknown process
discovery is unsafe. Only current-UID `/proc/PID/cmdline` files are read, and
full command lines remain in memory. Detection matches the actual production
range module, import-form CLI, execute/recover operations and plausible Python
wrappers. A quiet log never overrides a live controller.

AP-3 reports a **directory tail**, journal-name V3 candidates, and a legacy
candidate before June 27 without a V3 journal name. Metadata cannot prove
finalization or manifest authority. AP-6 confirms the three representative
roles with `classify_night_authority`; each must be COMPLETE and finalized,
and the legacy role must classify as legacy. If June 27 is also the newest V3
night, those two roles share a file; no third V3 night is invented.
Manifest sizes are measured for the full observed inventory and selected
partitions; no nightly or journal contents are opened in AP-3.

The nightly-summary projection uses the release's real `date_utc` column and
only the fifteen listed AP-6 fields. Missing fields are ABSENT; missing values
remain null, and invalid values are INVALID. Unselected columns never enter
evidence. The loci-index projection is exactly `night_date_utc` and
`ingested_at_utc`, with row counts and min/max timestamps per night. Incomplete
columns disable exclusion conclusions. Zero-row published nights can validly
be absent from the index. Evidence distinguishes sampled, confirmed published
exclusions from unclassified physical candidates. Legacy time-evidence classes
describe the summary for metadata candidates; only three roles receive release
authority classification. Times alone do not establish acquisition reliability.

AP-1 uses the kernel hostname and local `/etc/hosts` for FQDN evidence, with
UNKNOWN if the FQDN is not available locally. It performs no DNS/NSS network
lookup. Attribute-cache settings and host capabilities remain UNKNOWN when
not exposed. Cgroup-controller discovery does not infer admin permission.
The disk candidate is the explicitly selected evidence directory's filesystem.
All six policy questions require written human/admin answers.

## Confinement, deadline and bounds

Output paths must be absolute, contain no `..`, have no symlink component, and
neither equal, enter nor contain authority or Control-secret roots. Existing
roots must be owned by the invoking UID and private. The parent creates only
new mode-0600 JSON artifacts there, using exclusive/no-follow opens. No temp
file or existing artifact is replaced. Every generated document and its
reopened artifact undergo recursive forbidden-key scanning. Failure output
uses fixed refusal codes or exception class names, never source messages.

The audit guard precedes package import and prohibits Control-secret paths,
binding/authorization/approval/token files, all production writes, gate-content
reads, ANTARES-client import and networking. Parent production content reads
are confined to exact append-only event paths. Production data opens belong
only to the worker and are read-only.

Before imports/locking, the Linux/x86_64 worker installs Landlock ABI >=3 with
read rules only for runtime libraries and individually enumerated journals,
three-role manifests, cumulative files and the already provisioned lock. It
also grants directory-only access for the journal reader's parent `dir_fd`,
without granting content reads of unlisted/new files. It permits no native
filesystem writes. Seccomp denies native socket operations,
io_uring and child processes; native pthreads remain possible. Unsupported
boundaries cause refusal, with **no weaker fallback**. No namespace or timer
is created. Native Linux boundary execution is a platform-dependent test;
local macOS qualification checks the real rule builder and BPF decisions and
skips the irreversible Linux syscall test. Arnor support remains unmeasured.

The child alone holds the release shared lock, using
`history.authoritative_read(..., wait_seconds=5)`. The supervisor caps the
entire worker lifetime at **10 seconds**, including startup and acquisition,
with nonblocking request delivery. At overrun it sends SIGKILL, waits for
death, discards partial results and reports HOLD_TIMEOUT. It does not extend
the cap or retry. Successful and clean-refusal holds use the monotonic flock
syscall envelope; a killed holder without a release observation reports a
kill/reap **upper bound**, with exact hold duration unknown. No timeout
measurement is presented as exact. The reaping margin is recorded as
`kill_reap_margin_seconds`, alongside total worker lifetime.

All hashes, Arrow readers, batches and descriptors finish before release.
Linux checks `/proc/self/fd` inside the hold, excluding only the live lock;
local tests instrument handle closure and inject leakage. Release generation
and gate checks on context exit must pass before any result is serialized.

Bounds: 4,096 inventory entries/nights/summary rows, 1 MiB per event tail,
16 KiB event line, 64 KiB process command, 2 MiB per journal/representative
manifest, 64 MiB total journals, 20 million index rows, and 8 MiB serialized
worker/evidence packets. Exceeding a bound refuses; no full logs, journal
documents, authorization material or scientific rows are emitted.

## Refusal and later human retry

Tier B must refuse on a present/dangling-link gate, moving/missing cumulative
generation, live/unknown controller detection, recent/unknown nonterminal
activity, wrong UID, missing representatives, unsafe paths or file types,
source/release/path mismatch, unavailable kernel boundary, unavailable/shared
lock timeout, unfinalized/nonpublished representative, nonlegacy legacy role,
invalid/bounded evidence, descriptor leakage, exit gate/generation drift,
forbidden keys or hard timeout. A symlink/unexpected type causes STOP in either
tier. No override exists.

Only a later human-triggered second attempt is permitted, after at least two
minutes and a completely fresh AP-0, with Control approval for that attempt.
This script does not schedule or automatically perform one. Do not continue
to Arnor execution, Console G1, scratch NFS testing or remediation from this
review branch.

## Local qualification

Use a compatible local interpreter with v0.4.7 dependencies, without installing
or changing the release. From this isolated worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests "$LOCAL_PY" -B -m unittest \
  test_v3_ui_g04_ab_evidence -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests "$LOCAL_PY" -B -m unittest \
  test_v3_ui_g04_ab_evidence test_publication_authority_v3 \
  test_publication_v3 test_backfill_v3 test_production_range_g5 test_g3r_regressions -v
git diff --check
git status --short
```

The new suite installs an audit tripwire for local production/Control reads and
network calls, propagates it to regression children, and models protected stat
paths as absent before any syscall even if accidentally mounted. It uses real
temporary release journals and locks, without
constructing a publisher, provider, candidate or authorization. Existing
regression suites retain their own synthetic fixtures. Local qualification
cannot determine current production inventory, NFS timings/visibility,
Arnor kernel-boundary availability, policy permission, or identity-phase
feasibility; those remain future Control evidence questions.
