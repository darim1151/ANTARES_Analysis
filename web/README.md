# SkyPulse web

SkyPulse is a static-facing Next.js application. It reads a complete public-data
bundle from `public/data`; it must never query ANTARES, Rubin services, or private
analysis storage at runtime.

## Reproducible local check

Use exact Node.js 22.22.0 and the pnpm version declared in `package.json`:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm run validate:data:test
pnpm run validate:data
pnpm run typecheck
pnpm run lint
pnpm run build
```

The repository includes a sanitized, explicitly labelled demo bundle so a clean
checkout can be checked and built without credentials or access to scientific
infrastructure. Generated directories such as `.next`, `out`, and `node_modules`
are local artifacts and must not be committed.

## Public-data gate

`pnpm run validate:data` validates the six required JSON files as one coherent
bundle. It checks their schema/mode/timestamps, declared counts, object references,
coordinate bounds, validation flags, and absence of private filesystem paths or
secret-bearing fields.

The bundled demo is valid for development and CI. A real release pipeline must
instead run:

```bash
pnpm run validate:data:production
```

That stricter command rejects `export_mode: "demo"`. Passing either command is a
build-integrity check, not scientific approval of a newly exported dataset.

This is a Phase 1 demo/CI surface, not a public production deployment. The
current Next.js 14 line must be migrated to a supported release and pass a
dedicated UI/accessibility regression phase before public hosting. CI still
builds and starts the current application so its declared `start` command and
all static routes cannot silently regress while that migration remains pending.

## Unified Scientific Observatory (First Light)

`/observatory` is a research workspace over broker-native ANTARES and Fink
domains: one shared scientific state (basis + selection + focus + lens +
presentation), no shared ontology. It reads one integrity-sealed bundle from
`public/observatory/first-light/` and never queries a broker at runtime.

The First-Light bundle is a **fixture basis** (`bundle_class:
FIRST_LIGHT_FIXTURE`, `science_ready: false`):

- ANTARES: an adapter over the SkyPulse legacy demo above. Positions, tags and
  measurement counts are `LEGACY_SAMPLE`; dates and light curves are
  `SYNTHETIC_DEMO`; magnitudes the demo exporter clipped are null.
- Fink: real data from the accepted five-window cohort catalog
  `final-five-window-analytics-20261006-v1` (`accepted_five_window_20260225_20260714`,
  2026-02-25 → 2026-07-14, five delivery-validated, characterized and admitted
  acquisitions; build kind `QUALIFIED_COHORT_CATALOG`). Values are labelled
  `VALIDATED_TRANSPORT_EVIDENCE`, never accepted science.
  - Complete: delivered alert rows (DIA + SSO) for every UTC date, with
    zero-row dates as `ZERO`; DiaObject density on HEALPix order 6 over all
    3,487,175 derived DIA groups.
  - Sampled: a simple random sample of 5,000 DiaObjects (lowest md5 of the
    decimal id) with every delivered DiaSource and first/last-two source-time
    snapshots drives the entity layer, inspector and Lab. It is never
    aggregated into population counts: `fink:sky.filtered_density` is
    `UNAVAILABLE` and the time lane draws no sample subset.
- No cross-broker relation exists in this basis.

```bash
pnpm run test:observatory            # kernel + scientific-state tests
pnpm run validate:observatory        # contract, evidence and integrity gate
pnpm run validate:observatory:test   # validator mutation tests
pnpm run observatory:check           # bundle == deterministic generator output
pnpm run observatory:build           # regenerate after an input changes
node scripts/observatory/fetch-fink-catalog.mjs   # re-extract the Fink catalog (read-only, over SSH)
```

The Fink input `scripts/observatory/inputs/fink-catalog.<run id>.json.gz` is
produced by `extract-fink-catalog.py`, streamed to the data host over SSH. It
opens the catalog with Fink's native `open_catalog` (DuckDB `read_only`, raw
input fingerprints re-verified), checks the catalog SHA256 against its build
manifest and TAI − UTC with ERFA, disables DuckDB spilling and runs only
SELECTs. Nothing is written on the host.

Contract: `types/observatory.ts`. Each domain ships `time`, `sky`,
`entities`, `features` and lazily loaded `detail/NN.json` shards; `basis`,
`capabilities` and `provenance` are shared. Every file is listed with its
sha256 in `manifest.json`, and the reader refuses any payload whose digest
does not match. A real adapter replaces one domain's payloads and its basis
pin, and must keep native identifiers (int64 as decimal strings), declare
time scales, separate coverage from density and label evidence per field;
the validator enforces these rules.
