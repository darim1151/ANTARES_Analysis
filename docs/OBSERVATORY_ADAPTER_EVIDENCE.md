# V3-UI-U1.A ANTARES adapter evidence packet

## BASELINE

- Repository: `https://github.com/darim1151/ANTARES_Analysis.git`.
- Frozen source branch: `codex/feature-space-diagnostics`.
- Frozen source SHA: `812c545e14693cdce7ff7458f1d2b50b0804dcd8`, tag `v0.4.7`.
- `git fetch origin` completed before the worktree was created; the fetched
  source branch matched the requested SHA.
- Implementation branch: `codex/v3-ui-u1-antares-observatory-adapter`.
- Isolated worktree:
  `/Users/darim_shamsi/Documents/ANTARES REVAMP/ANTARES_Analysis-observatory-adapter`.
- The original source checkout's untracked operations note was left untouched.
- Read-only comparison with local G6.6.4 preparation (`9ed0ed2`) showed no
  differences in `history.py`, `authority.py`, `operations/publication.py`, the
  existing public exporter, or `web/`. No later acquisition changes were imported.
- Final commit/state are reported with the delivery. This packet participates
  in the evidence commit and therefore cannot contain its own commit hash.

## ANTARES SCIENTIFIC STATE MODEL

Frozen ANTARES-native records cover basis, capability manifest, temporal and
sky products, loci with per-night snapshots, feature definitions, input artifact
identities and provenance. Native semantic contract:
`antares.observatory.native.v1`; adapter version: `1.0.0`.
Every generated product identifies itself as DERIVED / NON-AUTHORITATIVE.

## AUTHORITY/NIGHT MODEL

The accepted native classifier retains COMPLETE, NOT_COMMITTED,
RECONCILIATION_REQUIRED and CONTRADICTION. Authority, availability, content,
acquisition status, scientific qualification, reason and evidence are separate.
Zero, missing, unknown, uncommitted, unavailable, rows and outside-basis states
are distinct. Legacy COMPLETE is explicitly partially qualified. A global
publication gate prevents science rows from every night. Missing counts remain
null; zero requires the existing conservative read-only proof.

## CAPABILITY + FEATURE REGISTRY

Thirteen definitions cover eight existing broker features (chi-square name,
standard-deviation name and six weighted-mean band names), one exact saved
alert/history row count, and four unavailable colors. Broker values are raw
native float64 features and remain PARTIALLY_QUALIFIED when finite: this source
repository has no generating broker estimator. Units, calibration, weights,
normalization and passband interpretation are not invented. All definitions
carry coverage/missingness, namespace, entity level, dtype, estimator semantics,
laboratory role and evidence. No RMS/amplitude/probability substitution occurs.
Tag state distinguishes missing from known empty labels; identifiers expose
actual content availability. Registry/selection axes remain machine-readable.

## TEMPORAL PRODUCT

UTC-calendar query-night convention is retained, with source interval bounds
and recorded upper inclusivity. Observation MJD time scale remains
UNESTABLISHED. Acquisition start/completion, qualified fetch completion, build
finish and publication start remain separate. Alert counts are saved source
rows, not observations during that night or unique alerts across partitions.
The legacy exporter's UTC-epoch/timedelta limitation is explicit in provenance.

## SKY PRODUCT

Isolated `uniform-ra-sin-dec.v1` backend, normally 72 x 36 equal-solid-angle
cells, with explicit bounds and deterministic ordering. One latest included
saved snapshot per native locus enters density accounting. Invalid, missing,
malformed and nonfinite coordinates are excluded and counted. Celestial frame
is UNESTABLISHED; survey footprint remains UNAVAILABLE. No HEALPix/common
encoding requirement is introduced.

## LOCUS / ENTITY MODEL

Native locus IDs, separate membership snapshots, qualified coordinates,
native multi-survey identifiers, tags, features, newest observation MJD, exact
linked saved source-row counts and provenance are preserved. Historical
membership is not replaced by a newer row. Nested identifier shape/order/
multiplicity/nulls and large integers survive as string leaves. Floating IDs
must retain their exact original integral value; already lossy values are
rejected. No universal Rubin identity or cross-broker match is created.

## PROVENANCE + DETERMINISM

Basis includes code SHA/state, optional release, adapter/native version,
explicit range and input-kind attestation, source artifact hashes and sizes,
authority references, generation digest and optional supplied Sentinel identity.
Sentinel is never recaptured or requalified. Source file hashes include the
native implementation and exporter, pinned dependency metadata, and relevant
existing scientific/read/authority code. Canonical ordering, UTF-8 JSON and
SHA256 are deterministic; unsafe JSON integers/nonfinite values are refused.
No runtime export timestamp enters the payload.

## SCIENTIFIC SAFEGUARDS

The adapter is read-only. Existing read/authority/publication/acquisition code
and `web/` are unchanged. Production roots were not executed, no live ANTARES
query or alert fetch ran, and no production publication, repair, Sentinel
update, manifest update, merge or Fink modification occurred. Synthetic
publication in tests is confined to the accepted temporary repository fixtures.
The existing temporary-lock provisioning exception remains unchanged.

The CLI builds fully before writing outside source/journal roots. Shared export
fails before any write. Atomic replacement uses a new output inode, including
when an output name was hardlinked to an input file.

## REAL DATA VS FIXTURE STATUS

No real production science was consumed. Tests use `tests/v3_fixtures.py` and
synthetic authority fixtures. The delivered JSON is METADATA / CONTRACT FIXTURE,
DERIVED / NON-AUTHORITATIVE, with zero loci, nights and sky science cells.
It contains thirteen feature definitions and sixteen code/input identities.
No SkyPulse demo or first-light science rows are promoted.

Fixture: `tests/fixtures/observatory_metadata_native_v1.json` (16,088 bytes).
Implementation pin: `664c8431921d8614008f70efc5d446bbc041f114`, CLEAN at generation.
Fixture SHA256:
`c8f60a0f9b45dbd1b4964d9c66ef94aa2a8a52f0bea7422e88594b2170af1622`.
An independent rebuild at that exact code pin/state produced identical
canonical bytes; zero science rows, unavailable feature values and UNBOUND
shared binding were verified from the saved fixture.

## TEST RESULTS

Final stable-source full offline suite: **513 tests passed in 295.593 seconds**,
including all **35 Observatory adapter tests** and the exact-value numeric
identifier regression assertion. Relevant existing read/authority/publication,
zero-row history, feature-analysis, public-exporter and exporter-validation
tests are included. Final failures/errors: **0**.

Command (captured log: `/private/tmp/antares-observatory-final-full-tests.log`):

```sh
MPLCONFIGDIR=/private/tmp/antares-observatory-mpl \
PYTHONPATH=/private/tmp/antares-observatory-no-network:. \
/private/tmp/g662b-venv/bin/python -m unittest discover -s tests -v
```

Runtime: Python 3.11.16; numpy 1.26.4; pandas 2.3.3; pyarrow 21.0.0; Astropy 6.0.1.
The network audit guard refuses socket connection, DNS and datagram operations.

Additional verification:

- Native wheel builds offline with no dependencies or build isolation.
- All six packaged native module bytes match the final source.
- Python 3.9 syntax compatibility is checked; no Python 3.9 runtime run is claimed.
- Wheel SHA256:
  `f3234fc0f3eb25185dac85975ea57d01e7d162518023f8dff739f7a90843d943`.
- Packaging assertion was updated specifically for the newly included namespace.
- No unrelated baseline failure was altered.
- `git diff --check` passes.

## OBSERVATORY CONTRACT RENDEZVOUS

Central serializer: `src/observatory/serialization.py`.
Binding: **UNBOUND**; `serialize_shared_bundle` fails closed even for a claimed
owner identity. Native serialization is independently usable.

| Common conceptual section | Native responsibility proposed |
| --- | --- |
| manifest | Product identity, derivation, versions and later verified owner identity |
| basis | Source/code/generation evidence |
| capabilities | Qualified registry and selection metadata |
| time | Independent night authority/availability/content/time semantics |
| sky | Backend geometry, density and exclusions |
| entities | Native loci with preserved membership snapshots |
| features | Definitions, qualifications, values and missingness |
| provenance | Artifact identities and evidence references |

The active Claude first-light branch was inspected read-only at
`5f0d76085a128eb23a57b0e8193f86b079c4cfb4`. It does not supply a binding acceptance
for this gate. Owner frozen revision, schema version/hash and acceptance are
still required. Unresolved common fields include independent night-state
encoding, source-row count meaning, historical entity cardinality, nested ID
encoding, partial features, time/frame limitations, spatial conversion,
entity/detail limits and checksum/shard layout. No competing common schema was
defined.

## FILES CHANGED

- `src/observatory/__init__.py`
- `src/observatory/model.py`
- `src/observatory/adapter.py`
- `src/observatory/features.py`
- `src/observatory/spatial.py`
- `src/observatory/serialization.py`
- `scripts/export_observatory_bundle.py`
- `tests/test_observatory_adapter.py`
- `tests/test_cli_packaging.py` (new package expectation only)
- `pyproject.toml` (package inclusion only)
- `docs/OBSERVATORY_ADAPTER.md`
- `docs/OBSERVATORY_ADAPTER_EVIDENCE.md`
- `tests/fixtures/observatory_metadata_native_v1.json` (delivery fixture)

## COMMITS

- `664c8431921d8614008f70efc5d446bbc041f114` — Add read-only ANTARES native
  Observatory adapter (native layer, CLI, tests, package inclusion and design
  documentation).
- The final evidence commit — Record Observatory adapter fixture and gate
  evidence (this packet and the fixture pinned to the implementation commit).
  Its exact SHA is reported with delivery because the packet participates in
  that commit. No merge is performed.
