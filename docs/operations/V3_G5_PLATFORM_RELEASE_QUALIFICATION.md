# V3-G5.2 Linux/Arnor platform and v0.4.5 release qualification

Executed 2026-09-30 (UTC 16:50–17:55) against the accepted G5 source. Every
value below was observed directly in this gate. Raw evidence lives outside git
under the qualification root
`/astro/store/shire/ANTARES/work/qualification/g5.2-20260930T165014Z/`
(`out/`, `runs/`, `scripts/`). That root holds no authoritative science.

**Verdict: PASS WITH FOLLOW-UP.** Nothing needed a source change. The
follow-ups are all fail-closed; see the last section.

## Identity

| Item | Value |
|---|---|
| Accepted starting source | `a01947386fb7aaf9eb2a59609137b3a5cd79ed0f` (clean, equal to origin) |
| Release commit / tag | `7211b5c2bce60edd24ca92309d33f0e81b412821` / `v0.4.5` (annotated tag object `c7e941f0b81aec5d4dfb8a5e5637a73c869eccc0`) |
| Release diff vs accepted source | `pyproject.toml` version; `cli.SOURCE_VERSION`; `production_canary._verify_release` installed-version pin (reused by range `execute`); README note. `live_antares.py` (`f22578a5…`) and `production_range.py` (`81ce9209…`) are unchanged, so the adapter attestation still holds. |
| Host | `arnor.astro.washington.edu`, Rocky Linux 9.8, kernel `5.14.0-687.33.1.el9_8.x86_64`, x86_64, UID 1533564 |
| Python | CPython 3.11.16 (release chain base `/astro/users/mdarim/.local/state/antares-analysis/phase6-development/python-3.11.16-20260825`) |
| Mount | `shire.infiniband:/data/shire` on `/astro/store/shire`, `nfs4` `vers=4.2`, `proto=rdma`, `nconnect=8`, `hard`, `local_lock=none`; session `st_dev` 60 |

## Platform qualification (real `/astro/store`)

This test ran `scripts/g52_platform.py` below the qualification root. It
exercises the exact ANTARES helpers: `authority.AuthorityLock`, all five
`_fsync_directory` variants, `_ensure_private_tree`, `QueryProgress`, the
backfill controller lock, and `BackfillController._read_prior_index`. It ran
three times: with the accepted source, as a long 100-second variant, and with
the final installed release. Each run passed 25/25.

- **Directory fsync: PASS.** Every helper fsyncs directories around create,
  link, rename and unlink with no error, and content persists.
  `QueryProgress` exclusive-link/rename/fsync commit and replay work.
- **Cross-process flock: PASS.** SH+SH is granted. SH/EX, EX/SH and EX/EX
  block. An unrelated `close()` in the holder keeps the lock. A SIGKILLed
  exclusive holder's lock is released immediately (0.0 s). The QueryProgress
  night lock and the controller lock both exclude another process.
- **Same process / threads: PASS.**
  - Kernel semantics on this mount are per open-file-description, like local
    Linux `flock`. Two descriptors in one process conflict (SH vs EX) whether
    held from the same thread or different threads.
  - `AuthorityLock` adds its own in-process bookkeeping and never holds more
    than one kernel descriptor per process. A thread's shared lease blocks
    another thread's exclusive request. The reverse holds too. The bounded
    wait acquires the lock after release (3.0 s).
  - An in-process shared lease blocks another process's exclusive lease. An
    in-process exclusive lease blocks other processes' shared leases.
  - Nested shared locks are reentrant. A shared-to-exclusive upgrade raises
    `AuthorityLockOrderError`. A nested exclusive request is unavailable.
- **Gate and 120 s coordination: PASS.** The unmodified
  `PUBLICATION_QUIESCENCE_WAIT_SECONDS = 120` was used for all of these.
  - A present gate refuses readers.
  - Writer first: the writer holds an exclusive lease plus gate for 5 s and
    for 100 s. The construction reader waited 5.0 s and 100.0 s respectively,
    then read only the committed new generation (2 rows).
  - Reader first: the reader holds a shared lease for 5 s and for 100 s. The
    publisher, with its 120 s wait, acquired the lock only after release
    (5.0 s and 100.0 s).

**M4 disposition: PASS.** The same-process reader-versus-writer path is
enforced twice over. `AuthorityLock` bookkeeping enforces it independently of
the kernel, and the kernel's per-OFD semantics on this mount would enforce it
as well. No race lets a reader see a mixed generation.

The 120 s window is operationally sufficient. The G4 June 27 publish held the
exclusive lock for about 27 s (05:04:34 → 05:05:01). A publish longer than
roughly 120 s under heavy NFS load would make the overlapping construction
fail with `LOCK_CONTENTION`, which is retryable and resolved with
`execute --resume`. That outcome is not unsafe.

Cross-host lock behaviour was not repeated. The mount identity equals the
earlier qualification.

## Test harness

- **Confinement.** Tests, `verify`, `plan`, `inspect` and the baseline ran
  under `scripts/g52_confine.py`.
  - It uses an identity-mapped user namespace, an empty network namespace
    (only `lo`), and Landlock. Filesystem writes are allowed only beneath the
    run directory. Every TCP connect is denied.
  - Probes proved write-denial on production `loci_index.parquet`, the June 27
    manifest, the production authority lock, the G4 journal and the v0.4.4
    `RELEASE_SHA`. They also proved no routes and failed TCP/UDP.
- **Environment.** Fixture temp directories lived on the real NFS mount. A
  Python audit hook denied socket connect/DNS. `ANTARES_ANALYSIS_{DATA,CACHE}_ROOT`
  pointed at absent paths, and those stayed absent.
- **Installed-mode runs.** These use a test tree without `src/`, so tests and
  their child processes import only the installed wheel. Source-mode runs use
  the extracted tag archive.

## Candidate wheel (accepted source, provisional 0.4.4 metadata)

| Item | Value |
|---|---|
| Wheel | `f553d7d4c7030fb2cd68c01b9a10e6ffaf8aefced6a9d503ef84f825e8ee9cd7` (byte-identical across two builds) |
| Install | disposable venv under the qualification root; dependencies copied from v0.4.4; v0.4.4 tree digest unchanged |
| Identity | `implementation_identity()` = source over 43 files (canonical digest `d8d6b62a…`); 51 RECORD entries verified; adapter attested |
| `plan` / `inspect` | NOT EXECUTED; attestation `e576cc86…`; 2026-06-28/29 `PLANNED` / `NOT_COMMITTED` |
| Focused G5 | 24/24 |
| Full discovery | installed: 430 run, 426 pass; source: 435 run, 432 pass. Identical pass sets apart from `test_cli_packaging`, which needs `src/` (6 tests) |

## v0.4.5 release, artifacts and immutable installation

Local qualification of the release tree (CPython 3.9, network-denial
`sitecustomize`, forbidden data/cache roots):

- full discovery: 451 passed;
- focused G5 + packaging: 30/30;
- touched suites: 178 passed;
- `compileall` and `git diff --check` clean.

Artifacts were built from `git archive` of the tag with
`SOURCE_DATE_EPOCH=1790789576` (commit time) and
`release-artifact/0.4.2/build-tools` (CPython 3.11.16, build 1.4.4, setuptools
83.0.0, wheel 0.45.1). `scripts/verify_python_distributions.py` confirmed
byte-identical wheels and 79 content-equivalent sdist files. The wheel's 43
`.py` files equal the tag source.

| Artifact | Bytes | SHA-256 |
|---|---|---|
| `antares_analysis-0.4.5-py3-none-any.whl` | 315,934 | `e2461eee1f84c85d176045bce7fcc789898f045d491ab4aaa6f92231b376aff7` |
| `antares_analysis-0.4.5.tar.gz` | 406,208 | `e5bdeb166767a88951edc12b55201cbf3a68af605d538ebe852b9f8618987843` |
| `antares-analysis-0.4.5-source.tar.gz` | 3,885,636 | `9fb6b0974ceffa82e6cc336ae102e040622cc5f9fbc98b88355097f23a6b88c7` |

The release is installed at
`/astro/users/mdarim/opt/antares-analysis/releases/7211b5c2bce60edd24ca92309d33f0e81b412821`,
following the v0.4.4 convention:

- The venv was created with `venv --copies` by the v0.4.4 interpreter, so no
  bin symlinks point into it.
- Dependencies were copied from v0.4.4. `SOURCE_RELEASE_PRESERVATION.json`
  records the v0.4.4 tree digest `534af831…` over 14,278 entries, identical
  before and after.
- The wheel was installed with `--no-index --no-deps` and the version check
  disabled.
- Markers are present: `RELEASE_SHA`, `WHEEL_SHA256`, `SDIST_SHA256`,
  `SOURCE_ARCHIVE_SHA256` and `PACKAGE_VERSION` = 0.4.5. `pip check` reports
  no broken requirements.
- The whole tree was made read-only: root mode 0500, 0 writable entries. No
  other release was modified.

## Final installed-release qualification (v0.4.5)

- **Verification.** Package 0.4.5, CPython 3.11.16, every module loaded from
  the release venv. The real, unmocked `_verify_release(7211b5c…, e2461eee…)`
  passes.
- **Implementation identity.** Installed equals tag source over 43 files.
  Canonical digest:
  `4b29a0c0145333246e0a5f1f3597393b2603e05fb1c29b24c444840993bdd76e`. 51 RECORD
  entries verified.
- **`plan` (import-form entry).** NOT EXECUTED, publication concurrency 1,
  `cache_root: null`, and the attested adapter
  `src.operations.production_range.LiveRangeAdapter`. The binding is:
  - science `9de004e6e8bc3af0e029c3adfb78f4cc34e937075ba5377de7891017cb9da801`;
  - attestation `ed3ccca6589b475dbec62777fc8f4da5c4ac2857ce740819736945450d358f26`;
  - configuration `8e5111a55e4999aca869c04ffb818d1ded1fd1a736f325f62fa6f2cad0abcba7`.

  As a negative control, the `-m` form refuses as expected.
- **`inspect`.** 2026-06-28 and 06-29 are `PLANNED` / `NOT_COMMITTED`. No
  `work/backfill` was created.
- **Focused G5: 24/24.** The run used the release path: a transient symlink
  `/astro/users/mdarim/opt/antares-analysis/g52-final-focused-tests`, created
  and removed outside `releases/`, so that the fixture's `sys.prefix` mock is
  a true ancestor of the installed module.
- **Full discovery.**

  | Mode | Run | Pass |
  |---|---|---|
  | Installed | 430 | 422 |
  | Tag source | 435 | 433 |

  No test passes only when installed. The source-only passes are
  layout-explained: `test_cli_packaging` needs `src/`, and 5
  `ProductionAuthorityTests` use the `sys.prefix` mock and pass in the
  release-path run above. Two non-passes appear in both modes: the NFS teardown
  follow-up, and `nbformat` being absent from the production venv
  (`test_rsp_shared_workflow`).
- **Platform script** with the release interpreter: PASS 25/25.
- **Runbook Control procedure.** Sections 2–8 of
  `V3_G5_MANUAL_RANGE_RUNBOOK.md` ran verbatim against this release. The only
  overrides were the DUMMY labels, a 0.25 h expiry and Control paths inside
  the qualification run directory. This needed DNS-only confinement: Landlock
  write and TCP denial without a netns, because `getfqdn()` returns
  `localhost.localdomain` inside an empty netns.
  - **Authorization.** Section 6 built a valid authorization, digest
    `ad9e5f4c…`, with initial Sentinel `50cf323d…`.
  - **Approval.** The stdlib-only section 7 snippet reproduced the digest.
    Approval identity `a4da1a0f…` equals the file's SHA-256.
  - **Verification.** Section 8 `ControlRangeApproval.verify` and
    `authority.verify` accepted the pair, and a wrong token was refused.
  - **Retirement.** The dummy token was shredded and the dummy work root was
    never created. The inert, expired pair remains only as evidence under
    `runs/final-control-dummy2/`.

## Fresh production baseline (read-only, v0.4.5, 2026-09-30T17:49:26Z)

This baseline is identical to the Phase 1 capture made with v0.4.4 at
16:52:34Z.

| Item | Value |
|---|---|
| Authority | 91 nights, 2026-02-25 → **2026-06-27**, none later; 06-28 target absent |
| June 27 | `COMPLETE`, finalized; transaction `v3pub-2026-06-27-197eca1861cb-a1`; resulting fingerprint `50cf323d…`; manifest `4bc98e4e685441242327b03ce7af0b60af9ea5fca44adb4b524d76feaddea457`; 331,786 loci, 509,431 alerts |
| Sentinel V2 | `50cf323d43f16ccb0d092a01b9df767b3297028f48b96ccbb61d5de5efd9cb05`; 91 manifests; 327 files; 1,271,751,077 bytes; mount binding `nfs4` / `/astro/store/shire` / `shire.infiniband:/data/shire` |
| Cumulative | `loci_index` `1cc0bad54423070cd8326db56409e618acb23123041070daab0b35314f1ca91f` (1,325,004 rows); `nightly_summary` `20a8230af18c3b5876ac778dc1747ffb91fe447caeea387da0d569fa1202d9f2` (91 rows) |
| Gate / journal / lock / staging | gate absent; one journal, `COMPLETE`; authority lock `0600`, uid 1533564, size 0, inode 1825692724162, shared lease granted without waiting (idle); staging empty |
| Cache | `/astro/store/shire/ANTARES/cache`, `work/cache`, `work/segment-cache` and `work/backfill` all absent |

**No June 28 or later ANTARES query, candidate or publication occurred. June 27
and the production cache are unchanged.** Every run of package code was
network-contained in one of three ways, and none recorded a network attempt:

- Tests, `plan`, `inspect` and the baseline ran in an empty network namespace.
- The dummy Control procedure ran under Landlock TCP-connect denial.
- The inspection and platform scripts ran under the socket audit hook and
  never touch the provider.

No range `execute` or `recover` was run. No night work root exists.

## Deviations during this gate

- `pip list` in the disposable candidate venv ran outside confinement. It made
  pip's PyPI self-version check once, at 17:10:01Z. This was not ANTARES and
  touched no production state. Every later pip call used
  `--disable-pip-version-check` and `PIP_NO_INDEX=1`.
- The first harness runner lacked a `__main__` guard, so `multiprocessing`
  spawn children re-ran the suite. They stayed inside confinement and were
  killed. One orphaned lock-holder test child, holding only a synthetic
  fixture lock, was found and killed at ~17:52Z. Results come from the
  corrected runner.
- The candidate suites ran under umask 077, which made the fixture-mode
  tripwire subtest vacuous. A rerun under the default umask 022 passes. Final
  runs used 022.

## Follow-ups and remaining unknowns

1. **`python -m src.operations.production_range` fails closed.** Executed as
   `__main__`, the adapter identity becomes `__main__.LiveRangeAdapter` and
   the attestation refuses. The runbook uses the tested import form (`main()`
   on the normally imported module), which is the path the G5 tests exercise.
   A code fix needs a source change and re-attestation in a later gate.
2. **`AuthorityLock.__enter__` leaks one descriptor per refused acquisition**
   when `flock` raises anything other than `EWOULDBLOCK`. Reproduced on NFS:
   the leak causes `.nfs*` silly-rename and the `ENOTEMPTY` teardown of
   `test_lock_symlink_and_unsupported_protocol_fail_closed` (all its
   assertions pass, and it passes fully on local `/tmp`). The leaked
   descriptor holds no lock and the path is fail-closed. It is not reachable
   while `flock` works, which it does on this mount. Fix `src/authority.py`
   later.
3. **The range hostname binding uses `socket.getfqdn()`.** Execution fails
   closed if DNS/NSS stops resolving `arnor.astro.washington.edu`.
4. **`nbformat` is not in the production venv**, so `test_rsp_shared_workflow`
   runs only locally, where it is included in the 451.
5. The release chain's base CPython directory is owner-writable. This is
   pre-existing and shared by every release.
6. The live ANTARES acquisition path on Arnor is exercised only by the first
   manual range, by design.
7. The proposed `-m` commands in `V3_G5_LOCAL_QUALIFICATION.md` are
   superseded by `V3_G5_MANUAL_RANGE_RUNBOOK.md`.
