# ANTARES native Observatory adapter

The adapter in `src/observatory/` describes saved ANTARES science and its
authority evidence. Every product is **DERIVED / NON-AUTHORITATIVE**. It neither
grants production authority nor implements the shared Observatory Bundle V1.
The baseline is `codex/feature-space-diagnostics` at
`812c545e14693cdce7ff7458f1d2b50b0804dcd8` (release tag `v0.4.7`).

## Native model and scientific boundary

`model.py` defines frozen Python records for basis, capability/feature registry,
temporal product, sky product, loci with membership snapshots, artifact
identities and provenance. These records contain no frontend TypeScript types.
ANTARES loci remain native ANTARES entities; survey identifiers are qualified
external identifiers, never a universal Rubin-object identity. Tags are broker
labels rather than astrophysical truth. A saved alert row and a saved locus
row have distinct count definitions. Snapshot tag state distinguishes missing
metadata from known empty tags. External identifier availability refers to
actual content, including in nested containers, rather than column existence.

Loci contain a snapshot for each included saved membership night. A later
snapshot cannot replace older membership, coordinates, tags or features. This
is preserved membership in the selected partitions, not a claim of complete
historical membership. No cumulative latest-row substitution is used. Entity
output is bounded by `max_loci` (default 10,000; maximum 100,000); exceeding the
bound refuses the build rather than truncating the science. Requested calendar
intervals are bounded to 3,661 nights. Full saved partitions are read and
verified in memory, so this implementation is intended for bounded offline
bases, not unbounded production ingestion.

The native products expose selection axes for membership intervals, native
sky cell sets, feature availability and qualified ranges, tags, identifier
namespaces and saved-alert-row thresholds. No universal query language is
introduced.

## Authority and night qualification

The reader reuses `publication.classify_night_authority`,
`history.authority_read_lock`, and `history.authoritative_read`. Their existing
shared-lock and publication behavior is unchanged. One shared lock covers the
whole product; an authoritative generation guard covers the whole logical
science read and the individual science partitions. Status evidence can be
read under a global publication gate, but no science rows are returned while
that gate exists, including rows from other COMPLETE nights.

Qualification independently records authority, availability, content,
acquisition status, scientific qualification, reason and provenance.

| Native authority / evidence | Availability / content | Ribbon interpretation |
| --- | --- | --- |
| COMPLETE, verified nonempty products | AVAILABLE / COMPLETE_WITH_ROWS | PUBLISHED_WITH_ROWS |
| COMPLETE, conservative native empty-night proof | AVAILABLE / ZERO_ROW_NIGHT | PUBLISHED_ZERO_ROW_NIGHT |
| NOT_COMMITTED, no manifest or pending journal | UNAVAILABLE / MISSING | MISSING; authority remains NOT_COMMITTED |
| NOT_COMMITTED, pending pre-gate journal | UNAVAILABLE / MISSING or UNKNOWN | NOT_COMMITTED |
| RECONCILIATION_REQUIRED | UNAVAILABLE / MISSING or UNKNOWN | RECONCILIATION_REQUIRED |
| CONTRADICTION | UNAVAILABLE / MISSING or UNKNOWN | CONTRADICTION |
| COMPLETE but missing/corrupt/unqualified product | UNAVAILABLE / MISSING or UNKNOWN | UNAVAILABLE |
| Outside explicit basis interval | OUTSIDE_BASIS / UNKNOWN | OUTSIDE_BASIS_COVERAGE |

Manifest existence never overrides the native classifier. Its accepted legacy
COMPLETE behavior is preserved and explicitly marked PARTIALLY_QUALIFIED; it
does not become accepted V3 publication evidence. A V3 manifest without its
journal remains a contradiction. Input-kind attestations do not grant
authority.

Required saved products are checked against manifest counts and supplied
artifact checksums, including V3 candidate artifact bindings. Known incomplete
query/fetch evidence, recorded errors, unsupported acquisition states, duplicate
IDs, unlinked alerts, incompatible membership dates/bounds and out-of-window
newest observation values reject the partition. Rejected partitions retain
authority evidence and a stable reason, and contribute no partial science.
Zero requires the existing read-only `history.revalidate_zero_row_night` proof:
empty schemas, completed clean query, positive chunk count and both products.
Its newly generated timestamp and regenerated manifest are discarded; nothing
is repaired or persisted.

## Feature and capability registry

The repository's `query.locus_to_record` copies broker properties into saved
locus columns. `feature_analysis.py` reads the columns below, audits their
finite coverage, and plots them. It does **not** implement their generating
broker estimators. Plot labels and names are insufficient estimator evidence.

| Feature | Qualification when finite values exist | Established semantics |
| --- | --- | --- |
| `feature_chi2_magn_r` | PARTIALLY_QUALIFIED | Raw native column; normalization, errors and degrees of freedom unestablished |
| `feature_standard_deviation_magn_r` | PARTIALLY_QUALIFIED | Raw native column; estimator unestablished; never renamed RMS or amplitude |
| `feature_weighted_mean_magn_{u,g,r,i,z,y}` | PARTIALLY_QUALIFIED | Raw native columns; weights, photometric system and input selection unestablished |
| `saved_alert_source_rows` | AVAILABLE for saved snapshots | Exact linked source-row count in the saved partition |
| `color_{g_minus_i,u_minus_g,g_minus_r,r_minus_i}` | UNAVAILABLE | Legacy code subtracts weighted means; no qualified color definition |

Without finite values, broker features are UNAVAILABLE. No broker-feature unit
is asserted. Band suffixes are preserved as native labels; passband/survey and
calibration semantics are unestablished. The adapter never derives color merely
because two columns exist, and emits no calibrated probability or photometric
summary. Existing `lightcurves.py` documents ZTF PSF magnitudes and filter IDs,
while saved LSST-associated loci can contain multi-survey history; that does
not justify silently treating arbitrary saved photometry as Rubin-band science.

Every definition has namespace, entity level, dtype, unit, band label, passband
semantics, definition, estimator limitations, capability state, source/finite/
missing/column-present row counts, missingness categories, laboratory role and
evidence references. Missingness distinguishes absent column, null, nonfinite,
malformed and undefined estimator. Coverage counts refer to included saved
snapshots rather than unique loci or the whole survey. Tags and sky density
remain partially qualified; photometric summaries, color and survey footprint
remain unavailable.

## Time semantics

Night dates follow the repository's UTC-calendar MJD query-window convention.
`history.mjd_to_utc_date` uses Astropy with `scale='utc'`; the existing public
exporter uses a UTC MJD epoch plus `timedelta`. That exporter implements a
calendar convention and does not qualify the broker observation time scale or
leap-second handling. The native observation time scale remains UNESTABLISHED.

Manifest query upper inclusivity is preserved per night when established;
otherwise it is UNKNOWN. Newest broker observation MJD is a separate optional
snapshot field. Acquisition start/completion, evidence-backed fetch completion,
build finish and publication start are separate fields. Missing fetch timing is
not inferred from a generic build-finished timestamp. Recorded UTC timestamps
must carry a UTC offset. Alert counts mean **saved alert/history source rows**,
not observations during that calendar night and not distinct alerts summed
across nights. No acquisition or observation timestamp is replaced by night
date.

## Spatial backend

`spatial.py` implements `uniform-ra-sin-dec.v1`: 72 RA bins and 36 uniform
sin(declination) bins by default. The longitude and sine-latitude boundaries
define equal solid angles, `4*pi/(ra_bins*sin_dec_bins)` steradians per cell.
Nonempty cells use `sin-dec-index:ra-index` IDs and deterministic order.
Southern and northern poles belong to the first and last latitude bins.
Valid input requires finite `0 <= ra < 360` and `-90 <= dec <= 90`, matching
native validation bounds. Missing, nonfinite, malformed and out-of-range
coordinates are separately counted; no placeholder position is emitted.

One latest included snapshot per native locus enters the density accounting.
Earlier coordinates remain in the inspector snapshots. Celestial reference
frame is UNESTABLISHED because committed source evidence establishes only the
RA/declination convention. Density is not survey coverage. The backend is
isolated and makes no HEALPix or frontend encoding commitment.

## Basis, provenance and determinism

Basis carries full code revision and CLEAN/DIRTY/UNVERIFIED state, optional
release, adapter/native-contract versions, frozen baseline, explicit source
interval and input-kind attestation, authority references, supplied Sentinel
identity and a content-addressed generation digest. Sentinel is optional and
never recaptured or requalified; absent evidence stays absent. Supplied
Sentinel bytes are hashed and identified as supplied identity only.

Code, nightly manifest/parquet, journal, gate and available cumulative artifacts
have SHA256 and byte-length identities. The cumulative files are generation
evidence, not substitutes for historical nightly membership. Evidence records
link consumed artifacts and semantic limitations. Host inode/mtime generation
tokens enforce read consistency but are not serialized as scientific identity.
Code artifact hashes capture dirty code too; callers of the Python API must
attest the supplied code SHA, or retain its default UNVERIFIED code state.

Canonical UTF-8 JSON sorts object keys and scientific record ordering, rejects
nonfinite/unsupported values and unsafe JSON integers, and adds no runtime
timestamp. Identical inputs and code give identical bytes and SHA256. Changed
source bytes change the artifact/generation identity even if row order alone
changed. Native identifiers become strings without losing leading zeros;
integer survey identifiers beyond JavaScript's safe range remain exact.
Nested identifier shape, order, multiplicity and nulls are preserved in
`value_json`. Unsafe pre-rounded floating IDs are rejected.

## Offline use and fixture status

No production data was used in this gate. Tests reuse synthetic repository
authority/Sentinel/publication fixtures in OS temporary directories. The
committed metadata fixture contains zero science rows and is explicitly
METADATA / CONTRACT FIXTURE and DERIVED / NON-AUTHORITATIVE. No
`web/public/data` or first-light demo rows are imported. Known synthetic
manifests cannot be relabeled SAVED_SCIENCE_SNAPSHOT.

Generate a zero-row native metadata artifact from a source checkout:

```sh
python scripts/export_observatory_bundle.py --metadata-fixture \
  --output /private/tmp/antares-observatory-native.json
```

For an explicitly authorized **offline copy**, provide `--offline-root`,
`--journal-root`, `--start`, `--end`, and an explicit `--input-kind`:
SYNTHETIC_AUTHORITY_FIXTURE or SAVED_SCIENCE_SNAPSHOT. The API requires canonical
real root paths. Production `/astro/store` execution is refused by this CLI.
Existing deployment lock requirements still apply outside temporary fixtures;
this adapter never changes them. The existing lock code can provision temporary
fixture lock files outside the authoritative data root; source products and
authority evidence are never mutated. Output must be outside input/journal
roots and cannot replace supplied Sentinel evidence. A new output inode is
installed atomically, protecting inputs even from hardlinked output names.

## Shared contract rendezvous

`serialization.py` is the only native-to-common compatibility boundary.
`canonical_bytes` and `native_sha256` support native use today.
`serialize_shared_bundle` fails closed unconditionally, including when a caller
supplies a claimed owner identity. `--format shared` fails before any output
file is written. Native payloads always report `shared_contract_binding=UNBOUND`.

The mapping proposal assigns native product identity to common manifest,
native basis to basis, registry/selection metadata to capabilities, independent
night states to time, backend geometry/accounting to sky, locus snapshots to
entities, definitions/missingness to features, and artifacts/evidence to
provenance. It defines no common fields.

The fetched Claude first-light branch was inspected read-only at
`5f0d76085a128eb23a57b0e8193f86b079c4cfb4`. Its versioned first-light fixture
does not itself authorize binding this native model to an accepted frozen
contract. Control must provide the owner contract's frozen revision, schema
version/content hash and acceptance reference before a later binding gate.
Unresolved mapping decisions include independent night-state encoding, saved
alert-row count semantics, per-night historical snapshots, lossless nested
identifiers, partial feature qualification, source time/frame limitations,
spatial scheme conversion, entity/detail limits and shared checksum/shard
layout. No owner frontend/schema files are modified here.

## Verification

`tests/test_observatory_adapter.py` covers native and V3 authority classes,
zero/missing/outside distinctions, pending and reconciliation states, global
gates, membership preservation, feature definitions and missingness, time
semantics, spatial exclusions/accounting, provenance, canonical serialization,
large/nested identifiers, fixture labeling, no source mutation, generation
drift and fail-closed shared export. Run it with:

```sh
python -m unittest discover -s tests -p 'test_observatory_*.py' -v
```

The gate evidence packet records broader offline suite results and any baseline
failures without modifying unrelated code.
