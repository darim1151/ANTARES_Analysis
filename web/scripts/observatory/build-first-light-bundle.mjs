#!/usr/bin/env node
// Build the Unified Scientific Observatory First-Light bundle.
//
//   node scripts/observatory/build-first-light-bundle.mjs           # write
//   node scripts/observatory/build-first-light-bundle.mjs --check   # verify committed bundle
//
// Deterministic: no wall clock, no network. Inputs:
//   - public/data/{sky_points,lightcurve_samples,public_manifest}.json
//     (the existing SkyPulse LEGACY DEMO contract; read only)
//   - scripts/observatory/inputs/fink-evidence.fd02c8e.json
//     (pinned Light Static schema excerpt; used to check broker field names)
//   - scripts/observatory/inputs/fink-catalog.<run id>.json.gz
//     (read-only extract of the accepted Fink five-window catalog; produced by
//     fetch-fink-catalog.mjs + extract-fink-catalog.py)
//
// Scientific honesty rules enforced here (and re-checked by the validator):
//   - ANTARES demo values are labelled LEGACY_SAMPLE or SYNTHETIC_DEMO per field;
//     values the demo exporter clipped are emitted as null.
//   - Fink values are real catalog values labelled as transport evidence, never
//     accepted science. Per-date delivered rows and DiaObject sky density are
//     complete; the Fink entity layer is a labelled simple random sample and is
//     never aggregated into population counts.
//   - No cross-broker relation is produced.

import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { radecToPix } from "../../lib/observatory/kernel/healpix.ts";
import { icrsToEcliptic, icrsToGalactic } from "../../lib/observatory/kernel/astro.ts";
import { addDays, utcDateRange, utcMjdToUtcDate } from "../../lib/observatory/kernel/dates.ts";
import { shardOf, shardPath } from "../../lib/observatory/kernel/shard.ts";

const CONTRACT = "uso.observatory-bundle";
const CONTRACT_VERSION = "1.0.0";
const GENERATOR = { name: "web/scripts/observatory/build-first-light-bundle.mjs", version: "1.2.0" };
// Adapter tag carried by every build id, so a changed adapter yields a new build identity.
const ADAPTER_TAG = `g${GENERATOR.version.split(".").slice(0, 2).join(".")}`;
const BUNDLE_ID = "uso-first-light-0002";
// The basis id is derived from what the basis pins (see basisIdFor), never a free constant.
const BASIS_PREFIX = "basis.uso.first-light";
const BASELINE_REVISION = "812c545e14693cdce7ff7458f1d2b50b0804dcd8";
const DENSITY_ORDER = 6;
// TAI - UTC on every date of the Fink cohort; the extractor checks it with ERFA.
const FINK_TAI_MINUS_UTC_S = 37;
const DETAIL_SHARDS = { antares: 4, fink: 64 };
const DEMO_MAG_CLIP = [13.5, 25.5];

const VERSIONS = {
  semantic_contract: { id: "uso.semantic-contract", version: "0.1.0-first-light" },
  feature_registry: { id: "uso.feature-registry", version: "fr-0.1.0" },
  analysis_kernel: { id: "uso.analysis-kernel", version: "ak-0.1.0" }
};
const KERNEL_TAG = `${VERSIONS.analysis_kernel.id}/${VERSIONS.analysis_kernel.version}`;

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(scriptDirectory, "..", "..");
const outputRoot = path.join(webRoot, "public", "observatory", "first-light");
const inputPaths = {
  skyPoints: "public/data/sky_points.json",
  lightcurves: "public/data/lightcurve_samples.json",
  demoManifest: "public/data/public_manifest.json",
  finkEvidence: "scripts/observatory/inputs/fink-evidence.fd02c8e.json",
  finkCatalog: "scripts/observatory/inputs/fink-catalog.final-five-window-analytics-20261006-v1.json.gz"
};

const check = process.argv.includes("--check");
const unknown = process.argv.slice(2).filter((a) => a !== "--check");
if (unknown.length) {
  console.error(`Unknown argument(s): ${unknown.join(", ")}`);
  process.exit(2);
}

/* ------------------------------------------------------------------ utils */

const sha256 = (data) => createHash("sha256").update(data).digest("hex");
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
/** basis_id = prefix + sha256(canonical pins)[:12]; the validator recomputes it. */
function basisIdFor(pins) {
  return `${BASIS_PREFIX}.${sha256(canonical(pins)).slice(0, 12)}`;
}
const round = (v, d) => (v === null || !Number.isFinite(v) ? null : Number(v.toFixed(d)));
const sig = (v, n = 4) => (v === null || !Number.isFinite(v) ? null : Number(v.toPrecision(n)));
function fail(message) {
  throw new Error(`First-Light generator: ${message}`);
}

function cap(scope, area, name, state, evidence, summary, reason, { codes = [], qualifications = [] } = {}) {
  return { id: `${scope}:${area}.${name}`, scope, area, name, state, evidence, summary, reason, codes, qualifications };
}

function dim(spec) {
  return {
    id: spec.id,
    domain: spec.domain,
    family: spec.family,
    label: spec.label,
    short: spec.short ?? spec.label,
    unit: spec.unit ?? null,
    scale: spec.scale ?? "linear",
    reversed: spec.reversed ?? false,
    extent: spec.extent ?? null,
    definition: spec.definition,
    definition_id: spec.definition_id ?? `${spec.id}@${VERSIONS.feature_registry.version}`,
    state: spec.state ?? "AVAILABLE",
    evidence: spec.state === "UNAVAILABLE" ? null : spec.evidence,
    qualifications: spec.qualifications ?? [],
    unavailable_reason: spec.unavailable_reason ?? null,
    snapshot: spec.snapshot ?? null,
    calibrated: spec.family === "model_output" ? false : null
  };
}

function positionDims(domain, evidence) {
  return [
    dim({
      id: `${domain}.galactic_latitude`,
      domain,
      family: "position",
      label: "Galactic latitude b",
      short: "b",
      unit: "deg",
      extent: [-90, 90],
      definition: "Galactic latitude of the native entity position (ICRS -> Galactic, Astropy/Hipparcos rotation).",
      definition_id: `derived.galactic_latitude_deg@${KERNEL_TAG}`,
      evidence
    }),
    dim({
      id: `${domain}.ecliptic_latitude`,
      domain,
      family: "position",
      label: "Ecliptic latitude β",
      short: "β",
      unit: "deg",
      extent: [-90, 90],
      definition: "Mean ecliptic latitude of J2000.0 of the native entity position (ICRS frame bias ignored).",
      definition_id: `derived.ecliptic_latitude_deg@${KERNEL_TAG}`,
      evidence
    })
  ];
}

function densityMap(records, quantity, unit, evidence) {
  const counts = new Map();
  for (const r of records) {
    const p = radecToPix(DENSITY_ORDER, r.ra, r.dec);
    counts.set(p, (counts.get(p) ?? 0) + 1);
  }
  const pixels = [...counts.keys()].sort((a, b) => a - b);
  return {
    order: DENSITY_ORDER,
    ordering: "NESTED",
    frame: "ICRS",
    quantity,
    unit,
    evidence,
    pixels,
    values: pixels.map((p) => counts.get(p))
  };
}

function nightStatesFromWindows(range, windows) {
  return utcDateRange(range.start, range.stop).map((date) => {
    const w = windows.find((x) => date >= x.start && date < x.stop);
    if (!w) return { date, state: "OUTSIDE_COVERAGE", reason: "OUTSIDE_BUILD_RANGE", window_id: null, evidence: [] };
    return { date, state: w.state, reason: w.status_codes[0], window_id: w.id, evidence: w.evidence };
  });
}

/* ---------------------------------------------------------------- inputs */

async function readInput(relative) {
  const bytes = await readFile(path.join(webRoot, relative));
  const text = relative.endsWith(".gz") ? gunzipSync(bytes).toString("utf8") : bytes.toString("utf8");
  return { relative, bytes, sha256: sha256(bytes), json: JSON.parse(text) };
}

const inputs = {
  skyPoints: await readInput(inputPaths.skyPoints),
  lightcurves: await readInput(inputPaths.lightcurves),
  demoManifest: await readInput(inputPaths.demoManifest),
  finkEvidence: await readInput(inputPaths.finkEvidence),
  finkCatalog: await readInput(inputPaths.finkCatalog)
};

/* --------------------------------------------------------------- ANTARES */

function buildAntares() {
  const demo = inputs.demoManifest.json;
  if (demo.export_mode !== "demo") fail("the First-Light ANTARES adapter only accepts the legacy demo export");
  const buildId = `antares.skypulse-demo.${demo.generated_at_utc.replace("+00:00", "Z")}.${ADAPTER_TAG}`;
  let clipped = 0;
  const records = inputs.skyPoints.json.points
    .map((p) => {
      const id = p.locus_id ?? p.id;
      const newest = p.newest_alert_observation_time ?? p.mjd;
      if (utcMjdToUtcDate(newest) !== p.date_utc) fail(`${id}: date_utc does not match its UTC-treated MJD`);
      const mag = p.brightest_alert_magnitude ?? p.brightness_mag;
      const isClipped = mag <= DEMO_MAG_CLIP[0] || mag >= DEMO_MAG_CLIP[1];
      if (isClipped) clipped += 1;
      return {
        kind: "antares.locus",
        id,
        ra: round(p.ra, 5),
        dec: round(p.dec, 5),
        entity_date: p.date_utc,
        newest_alert_observation_time: round(newest, 6),
        brightest_alert_magnitude: isClipped ? null : round(mag, 3),
        num_mag_values: p.num_mag_values ?? p.obs_count,
        tags: [...p.tags]
      };
    })
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  if (new Set(records.map((r) => r.id)).size !== records.length) fail("duplicate ANTARES locus ids");

  const range = { start: demo.source_data_range ? utcMjdToUtcDate(demo.source_data_range.historical_mjd_min) : null, stop: null };
  range.stop = addDays(demo.source_data_range.latest_night_utc, 1);
  const window = {
    id: "antares.legacy-demo-export",
    label: "SkyPulse legacy demo export",
    start: range.start,
    stop: range.stop,
    state: "AVAILABLE",
    source_state: `export_mode=${demo.export_mode}`,
    delivery_validation: "NOT_APPLICABLE",
    admission: "ADMITTED",
    status_codes: ["LEGACY_DEMO_EXPORT", "SYNTHETIC_DATE_ASSIGNMENT"],
    evidence: ["SYNTHETIC_DEMO"],
    facts: [
      { label: "Export mode", value: demo.export_mode, evidence: "SYNTHETIC_DEMO" },
      { label: "Exported at (UTC)", value: demo.generated_at_utc, evidence: "SYNTHETIC_DEMO" },
      { label: "Loci in demo sample", value: records.length, unit: "loci", evidence: "LEGACY_SAMPLE" },
      { label: "Selected demo date", value: demo.selected_night_date, evidence: "SYNTHETIC_DEMO" }
    ],
    rate_comparison: "PROHIBITED",
    caveat:
      "Positions, tags and measurement counts come from a legacy repository sample; the date of every locus was " +
      "assigned synthetically by the demo exporter. Per-date counts are sample allocations, not ANTARES nightly totals."
  };
  const nights = nightStatesFromWindows(range, [window]);
  const countValues = {};
  for (const r of records) countValues[r.entity_date] = (countValues[r.entity_date] ?? 0) + 1;
  for (const n of nights) {
    // An empty date in a *sample* is not a source-level zero (ZERO means the
    // source delivered zero records), so the adapter refuses rather than mislabel.
    if (n.state === "AVAILABLE" && !countValues[n.date]) fail(`demo sample has no loci on ${n.date}; a sample cannot express ZERO`);
  }

  const timeSemantics = {
    stored_field: "newest_alert_observation_time",
    format: "MJD",
    scale: "UTC",
    scale_label: "UTC (exporter-treated)",
    scale_basis:
      "The historical ANTARES exporter treats MJD as UTC. In this legacy demo the stored value was synthesized by the demo exporter.",
    date_binning: "UTC date of the UTC-treated MJD, half-open [00:00, 24:00) UTC.",
    entity_date_rule: "UTC date of newest_alert_observation_time (demo date_utc; synthetic assignment)."
  };

  const pin = {
    domain: "antares",
    build_id: buildId,
    build_kind: "LEGACY_DEMO_EXPORT",
    label: "ANTARES · SkyPulse legacy demo export",
    evidence: ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"],
    science_ready: false,
    native_ontology:
      "ANTARES locus: an alert-aggregated sky position. ANTARES tags are filter-pipeline memberships, not astrophysical classes.",
    source: { repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, paths: [inputPaths.skyPoints, inputPaths.lightcurves, inputPaths.demoManifest] },
    time: timeSemantics
  };

  const time = {
    contract: CONTRACT,
    domain: "antares",
    build_id: buildId,
    semantics: timeSemantics,
    range,
    nights,
    windows: [window],
    counts: {
      quantity: "Legacy demo-sample loci by synthetic entity date",
      unit: "loci",
      evidence: ["SYNTHETIC_DEMO"],
      values: Object.fromEntries(nights.filter((n) => n.state === "AVAILABLE" || n.state === "ZERO").map((n) => [n.date, countValues[n.date]]))
    }
  };

  const sky = {
    contract: CONTRACT,
    domain: "antares",
    build_id: buildId,
    density: densityMap(records, "ANTARES loci in the legacy demo sample", "loci", ["LEGACY_SAMPLE"]),
    coverage: null,
    coverage_unavailable_reason:
      "No survey footprint or exposure product is pinned for ANTARES in this basis. Locus density is not coverage, and empty cells are not zero-coverage cells."
  };

  const fields = [
    { key: "locus_id", label: "Locus ID", unit: null, evidence: "LEGACY_SAMPLE", description: "ANTARES locus identifier." },
    { key: "ra", label: "RA (ICRS)", unit: "deg", evidence: "LEGACY_SAMPLE", description: "Locus right ascension." },
    { key: "dec", label: "Dec (ICRS)", unit: "deg", evidence: "LEGACY_SAMPLE", description: "Locus declination." },
    {
      key: "newest_alert_observation_time",
      label: "Newest alert time",
      unit: "MJD (UTC-treated)",
      evidence: "SYNTHETIC_DEMO",
      description: "Synthesized by the legacy demo exporter; not the ANTARES value."
    },
    {
      key: "brightest_alert_magnitude",
      label: "Brightest alert magnitude",
      unit: "mag",
      evidence: "LEGACY_SAMPLE",
      description: `Legacy sample value; null where the demo exporter clipped to [${DEMO_MAG_CLIP.join(", ")}].`
    },
    { key: "num_mag_values", label: "Magnitude values", unit: "count", evidence: "LEGACY_SAMPLE", description: "Number of magnitude values recorded on the locus." },
    { key: "tags", label: "ANTARES tags", unit: null, evidence: "LEGACY_SAMPLE", description: "Filter-pipeline memberships; not astrophysical classes." }
  ];

  const entities = {
    contract: CONTRACT,
    domain: "antares",
    build_id: buildId,
    entity_kind: "antares.locus",
    native_label: "ANTARES locus",
    id_field: "locus_id",
    population: {
      complete: true,
      represented: records.length,
      total: records.length,
      sampling: "The complete legacy demo sample of this basis (a seeded sample of a legacy repository CSV); not an ANTARES nightly census."
    },
    fields,
    records,
    detail: { shard_count: DETAIL_SHARDS.antares, path_template: "detail/{shard}.json", shard_rule: "fnv1a32(id) mod shard_count" }
  };

  const gal = records.map((r) => icrsToGalactic(r.ra, r.dec));
  const ecl = records.map((r) => icrsToEcliptic(r.ra, r.dec));
  const dimensions = [
    dim({
      id: "antares.num_mag_values",
      domain: "antares",
      family: "multiplicity",
      label: "Magnitude values on locus",
      short: "N mag",
      unit: "count",
      scale: "log",
      definition: "num_mag_values: number of magnitude values recorded on the ANTARES locus (alert multiplicity proxy).",
      evidence: "LEGACY_SAMPLE",
      qualifications: ["Legacy repository sample; not a V3 published build."]
    }),
    dim({
      id: "antares.brightest_alert_magnitude",
      domain: "antares",
      family: "photometry",
      label: "Brightest alert magnitude",
      short: "m_bright",
      unit: "mag",
      reversed: true,
      definition: "brightest_alert_magnitude: brightest magnitude among the locus alerts (survey/band mix as stored by ANTARES).",
      state: "PARTIALLY_QUALIFIED",
      evidence: "LEGACY_SAMPLE",
      qualifications: [
        `${clipped} demo values were clipped by the legacy exporter and are null here.`,
        "Band and survey of the brightest alert are not carried in this basis."
      ]
    }),
    dim({
      id: "antares.newest_alert_mjd",
      domain: "antares",
      family: "time",
      label: "Newest alert time",
      short: "t_newest",
      unit: "MJD (UTC-treated)",
      definition: "newest_alert_observation_time as exported; MJD treated as UTC by the historical exporter.",
      state: "PARTIALLY_QUALIFIED",
      evidence: "SYNTHETIC_DEMO",
      qualifications: ["Synthesized by the legacy demo exporter (hash-derived); not the ANTARES value."]
    }),
    ...positionDims("antares", "LEGACY_SAMPLE"),
    dim({
      id: "antares.time_baseline_days",
      domain: "antares",
      family: "time_baseline",
      label: "Alert time baseline",
      definition: "Newest minus oldest alert time on the locus.",
      state: "UNAVAILABLE",
      unavailable_reason: "Oldest-alert time is not in this basis, and the demo newest-alert time is synthetic."
    }),
    dim({
      id: "antares.colour",
      domain: "antares",
      family: "colour",
      label: "Colour",
      definition: "Difference of contemporaneous per-band magnitudes.",
      state: "UNAVAILABLE",
      unavailable_reason: "No per-band photometry is exported for ANTARES loci in this basis."
    }),
    dim({
      id: "antares.lc_features",
      domain: "antares",
      family: "variability",
      label: "Variability statistics (χ², skewness, …)",
      definition: "ANTARES lc_feature_extractor statistics.",
      state: "UNAVAILABLE",
      unavailable_reason:
        "ANTARES lc_feature_extractor values are not exported in this basis; the lc_feature_extractor tag marks filter membership only."
    }),
    dim({
      id: "antares.model_scores",
      domain: "antares",
      family: "model_output",
      label: "Classifier scores",
      definition: "Model outputs attached to the locus.",
      state: "UNAVAILABLE",
      unavailable_reason: "ANTARES tags are filter memberships, not model scores; no classifier outputs are exported in this basis."
    })
  ];
  const columns = {
    "antares.num_mag_values": records.map((r) => r.num_mag_values),
    "antares.brightest_alert_magnitude": records.map((r) => r.brightest_alert_magnitude),
    "antares.newest_alert_mjd": records.map((r) => r.newest_alert_observation_time),
    "antares.galactic_latitude": gal.map(([, b]) => round(b, 4)),
    "antares.ecliptic_latitude": ecl.map(([, b]) => round(b, 4))
  };
  const features = {
    contract: CONTRACT,
    domain: "antares",
    build_id: buildId,
    registry: VERSIONS.feature_registry,
    kernel: VERSIONS.analysis_kernel,
    dimensions,
    lab_defaults: { x: "antares.num_mag_values", y: "antares.brightest_alert_magnitude" },
    columns
  };

  const lc = inputs.lightcurves.json;
  if (lc.sample_source !== "synthetic_demo") fail("legacy demo lightcurves must be synthetic_demo");
  const shards = Array.from({ length: DETAIL_SHARDS.antares }, (_, shard) => ({
    contract: CONTRACT,
    domain: "antares",
    build_id: buildId,
    shard,
    evidence: ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"],
    records: {}
  }));
  for (const r of records) {
    const points = lc.lightcurves[r.id];
    shards[shardOf(r.id, DETAIL_SHARDS.antares)].records[r.id] = {
      kind: "antares.locus",
      id: r.id,
      tags: r.tags,
      lightcurve: points
        ? {
            evidence: "SYNTHETIC_DEMO",
            time_scale: "UTC",
            label: "Synthetic demo brightness story; not ANTARES alert-record photometry.",
            points: points.map((pt) => ({ mjd: round(pt.mjd, 6), magnitude: round(pt.magnitude, 3), band: pt.filter }))
          }
        : null,
      lightcurve_unavailable_reason: points ? null : "No alert-record photometry is exported for this locus in this basis."
    };
  }

  const fieldEvidence = fields.map((f) => ({ field: f.key, evidence: f.evidence, note: f.description }));
  return { pin, time, sky, entities, features, shards, clipped, fieldEvidence, buildId };
}

/* ------------------------------------------------------------------ Fink */

function buildFink() {
  // Schema reference: the pinned Light Static schema excerpt (fd02c8e). The
  // population itself comes from the read-only catalog extract.
  const ev = inputs.finkEvidence.json;
  const schema = ev.light_static_schema;
  const cat = inputs.finkCatalog.json;
  const c = cat.catalog;
  if (cat.kind !== "uso.fink-catalog-extract" || cat.version !== 1) fail("Fink catalog extract has an unknown kind/version");
  if (c.contract_id !== "analysis_contract_v1" || c.qualification_status !== "PASS") fail("Fink catalog must be a PASS analysis_contract_v1 catalog");
  if (!cat.reader?.read_only || !cat.reader?.raw_fingerprints_reverified) fail("Fink catalog extract must come from the native read-only opener");
  if (cat.time_scale.tai_minus_utc_s !== FINK_TAI_MINUS_UTC_S) fail("TAI - UTC over the cohort window must be the ERFA-verified 37 s");
  for (const a of cat.acquisitions) {
    if (a.state !== "DELIVERY_VALIDATED" || a.expected_rows !== a.validated_rows || !a.characterization || !c.cohort_acquisitions.includes(a.acquisition_id)) {
      fail(`${a.acquisition_id} must be delivery-validated, reconciled, characterized and admitted to ${c.cohort_name}`);
    }
    if (a.schema_groups.length !== 1 || !schema.schema_groups[0].startsWith(`${a.schema_groups[0]}=`)) fail(`${a.acquisition_id} schema group differs from the pinned Light Static schema`);
  }
  const buildId = `fink.catalog.${c.run_id}.${inputs.finkCatalog.sha256.slice(0, 12)}.${ADAPTER_TAG}`;
  const TAI = "VALIDATED_TRANSPORT_EVIDENCE";
  const taiToUtcMjd = (mjdTai) => mjdTai - FINK_TAI_MINUS_UTC_S / 86400;

  /* --- acquisition windows from the catalog's build manifest --- */
  const acquisitions = [...cat.acquisitions].sort((a, b) => (a.start < b.start ? -1 : 1));
  const windows = acquisitions.map((a, i) => {
    const facts = [
      { label: "Acquisition state", value: a.state, evidence: "COMMITTED_OPERATIONAL_RECORD" },
      { label: "Science profile", value: a.profile, evidence: "COMMITTED_OPERATIONAL_RECORD" },
      { label: "Delivered rows (validated = expected)", value: a.validated_rows, unit: "rows", evidence: TAI },
      { label: "DIA rows", value: a.qc.dia_rows, unit: "rows", evidence: TAI },
      { label: "SSO rows", value: a.qc.sso_rows, unit: "rows", evidence: TAI },
      { label: "Ambiguous rows", value: a.qc.ambiguous_rows, unit: "rows", evidence: TAI },
      { label: "Distinct DIA objects in this window", value: a.qc.distinct_dia_objects, unit: "DiaObjects", evidence: TAI },
      { label: "Readable Parquet files", value: a.parquet_files, unit: "files", evidence: TAI },
      { label: "Delivered bytes", value: a.raw_total_bytes, unit: "bytes", evidence: TAI },
      { label: "Characterization run", value: a.characterization.run_id, evidence: "COMMITTED_OPERATIONAL_RECORD" },
      { label: "Analytical cohort", value: c.cohort_name, evidence: "COMMITTED_OPERATIONAL_RECORD" }
    ];
    return {
      id: `fink.month-${i + 1}`,
      label: `Month-${i + 1} · ${a.start} → ${a.stop}`,
      start: a.start,
      stop: a.stop,
      state: "AVAILABLE",
      source_state: a.state,
      delivery_validation: "DELIVERY_VALIDATED",
      admission: "ADMITTED",
      status_codes: ["DELIVERY_VALIDATED", "ADMITTED", "CHARACTERIZED"],
      evidence: [TAI, "COMMITTED_OPERATIONAL_RECORD"],
      facts,
      rate_comparison: "PROHIBITED",
      caveat:
        `Delivery-validated (expected = validated rows), characterized (${a.characterization.run_id}) and admitted to the accepted ` +
        `${c.cohort_name} cohort. Delivered rows are transport facts, not Rubin scientific completeness, and are never compared ` +
        "across windows as rates. A zero-row date had no delivered rows; it is not a survey non-observation."
    };
  });
  const range = { start: c.requested_start, stop: c.requested_stop };
  if (range.start !== windows[0].start || range.stop !== windows.at(-1).stop) fail("acquisition windows must tile the cohort range");
  const daily = new Map(cat.daily.map((d) => [d.date, d]));
  const nights = utcDateRange(range.start, range.stop).map((date) => {
    const w = windows.find((x) => date >= x.start && date < x.stop);
    const d = daily.get(date);
    if (!w || !d) fail(`cohort date ${date} has no window or daily coverage`);
    return d.delivered_rows > 0
      ? { date, state: "AVAILABLE", reason: "DELIVERED_ROWS", window_id: w.id, evidence: w.evidence }
      : { date, state: "ZERO", reason: "ZERO_ROWS_IN_VALIDATED_DELIVERY", window_id: w.id, evidence: w.evidence };
  });
  if (cat.rows_outside_requested_dates !== 0) fail("delivered rows outside the requested dates must be shown, not dropped");

  const timeSemantics = {
    stored_field: "midpointMjdTai",
    format: "MJD",
    scale: "TAI",
    scale_label: "TAI",
    scale_basis: "Fink analysis_contract_v1: midpointMjdTai is the authoritative observation time, MJD on the TAI scale.",
    date_binning:
      "UTC date from the catalog's own UTC-midnight boundaries expressed in TAI MJD, half-open [00:00, 24:00) UTC. " +
      "TAI − UTC = 37 s on every date of the cohort (ERFA leap-second table, checked at extraction).",
    entity_date_rule: "UTC date of the first delivered DiaSource of the DiaObject in this basis."
  };
  const time = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    semantics: timeSemantics,
    range,
    nights,
    windows,
    counts: {
      quantity: "Delivered alert rows per UTC date (every DIA and SSO DiaSource packet in the cohort)",
      unit: "alert rows",
      evidence: [TAI],
      values: Object.fromEntries(nights.map((n) => [n.date, daily.get(n.date).delivered_rows]))
    }
  };

  /* --- sky: complete DiaObject density from the extract --- */
  const dens = cat.density;
  if (dens.order !== DENSITY_ORDER) fail(`Fink density must be at order ${DENSITY_ORDER}`);
  const sky = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    density: {
      order: dens.order,
      ordering: "NESTED",
      frame: "ICRS",
      quantity: `DiaObjects, complete cohort population (${dens.objects.toLocaleString("en-US")} derived DIA groups)`,
      unit: "DiaObjects",
      evidence: [TAI],
      pixels: dens.pixels,
      values: dens.values
    },
    coverage: null,
    coverage_unavailable_reason:
      "Fink Light Static delivers alerts, not survey pointings: the Rubin observing footprint is not in this basis. " +
      "Cells with delivered DiaObjects are density, not coverage; an empty cell is not evidence that Rubin did not observe it."
  };

  /* --- entities: a simple random sample of DiaObjects --- */
  const sample = cat.sample;
  if (sample.population !== dens.objects) fail("sample population must equal the density population");
  const objects = sample.objects.map((o) => {
    if (radecToPix(DENSITY_ORDER, o.ra, o.dec) !== o.pix) fail(`HEALPix parity: ${o.id} differs between the extractor and the kernel`);
    if (!o.sources.length) fail(`${o.id} has no delivered sources`);
    return o;
  });
  const records = objects.map((o) => ({
    kind: "fink.diaObject",
    id: o.id,
    ra: o.ra,
    dec: o.dec,
    entity_date: utcMjdToUtcDate(taiToUtcMjd(o.sources[0].midpointMjdTai)),
    n_dia_sources: o.sources.length,
    first_midpoint_mjd_tai: o.sources[0].midpointMjdTai,
    last_midpoint_mjd_tai: o.sources.at(-1).midpointMjdTai,
    bands: [...new Set(o.sources.map((s) => s.band))].sort((a, b) => "ugrizy".indexOf(a) - "ugrizy".indexOf(b))
  }));
  const fields = [
    { key: "diaObjectId", label: "diaObjectId", unit: null, evidence: TAI, description: "Derived DIA grouping key (pred.is_sso false AND diaObjectId > 0); int64 carried as a decimal string." },
    { key: "ra", label: "RA (ICRS)", unit: "deg", evidence: TAI, description: "Unit-vector mean of delivered DiaSource positions, rounded to 1e-5 deg." },
    { key: "dec", label: "Dec (ICRS)", unit: "deg", evidence: TAI, description: "Unit-vector mean of delivered DiaSource positions, rounded to 1e-5 deg." },
    { key: "n_dia_sources", label: "Delivered DiaSources", unit: "rows", evidence: TAI, description: "Delivered DiaSource rows in this basis; not lifetime detections." },
    { key: "first_midpoint_mjd_tai", label: "First midpointMjdTai", unit: "MJD (TAI)", evidence: TAI, description: "Earliest delivered DiaSource midpoint, TAI." },
    { key: "last_midpoint_mjd_tai", label: "Last midpointMjdTai", unit: "MJD (TAI)", evidence: TAI, description: "Latest delivered DiaSource midpoint, TAI." },
    { key: "bands", label: "Bands", unit: null, evidence: TAI, description: "LSST bands among delivered DiaSources." }
  ];
  const entities = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    entity_kind: "fink.diaObject",
    native_label: "Rubin DiaObject (Fink-delivered DIA grouping)",
    id_field: "diaObjectId",
    population: {
      complete: false,
      represented: records.length,
      total: sample.population,
      sampling:
        `Simple random sample of ${records.length.toLocaleString("en-US")} of ${sample.population.toLocaleString("en-US")} DiaObjects ` +
        "(the lowest md5 of the decimal diaObjectId), with every delivered DiaSource of each. Sky density and per-date counts are complete; " +
        "this layer is not, so it is never aggregated into population counts."
    },
    fields,
    records,
    detail: { shard_count: DETAIL_SHARDS.fink, path_template: "detail/{shard}.json", shard_rule: "fnv1a32(id) mod shard_count" }
  };

  /* --- feature registry and columns (sample) --- */
  const latestSnapshot = (o) => o.snapshots.at(-1);
  const posMag = (o, band) => {
    const vals = o.sources.filter((s) => s.band === band && s.psfFlux > 0).map((s) => 31.4 - 2.5 * Math.log10(s.psfFlux));
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  };
  const positive = (v) => (v === null || v === undefined || !Number.isFinite(v) || v <= 0 ? null : v);
  const unitScore = (v) => (v === null || v === undefined || !Number.isFinite(v) || v < 0 || v > 1 ? null : v);
  const census = cat.clf_census;
  const censusNote = (key) => {
    const x = census[key];
    return `Population census over ${x.rows.toLocaleString("en-US")} DIA rows: ${x.unit_interval_rows.toLocaleString("en-US")} in [0, 1], ` +
      `${x.minus_one_rows.toLocaleString("en-US")} at −1, ${x.null_rows.toLocaleString("en-US")} null, ${x.other_rows.toLocaleString("en-US")} other.`;
  };
  const lcDim = (key, label, short, scale, unit) =>
    dim({
      id: `fink.lc_features.r.${key}`,
      domain: "fink",
      family: "variability",
      label: `${label} (r)`,
      short,
      unit,
      scale,
      definition:
        `lc_features["r"].${key} from the latest delivered source-time snapshot in this basis. Fink computes lc_features over the history it holds ` +
        "for the object at alert time, which can include detections that were not delivered in this basis.",
      evidence: TAI,
      snapshot: "Latest delivered snapshot; features summarize the history visible to Fink at that alert time, not a timeless object property.",
      qualifications: [
        "Null where Fink delivered no r-band features or a non-finite value" + (scale === "log" ? ", or a non-positive value on this log axis." : ".")
      ]
    });
  // Where CATS assigned no class (cats_class -1) Fink delivers cats_score 0.0; that
  // 0.0 is a placeholder, not a score (inferred from the paired values).
  const catsPlaceholder = (snap) => snap.clf.cats_class === -1;
  const catsPlaceholders = objects.filter((o) => catsPlaceholder(latestSnapshot(o))).length;
  const clfDim = (key, label, extra = []) =>
    dim({
      id: `fink.clf.${key}`,
      domain: "fink",
      family: "model_output",
      label,
      short: key,
      unit: "score",
      extent: [0, 1],
      definition: `clf.${key} from the latest delivered source-time snapshot in this basis.`,
      evidence: TAI,
      snapshot: "Latest delivered snapshot; classifier outputs change as alerts arrive and are not timeless classifications.",
      qualifications: [
        "Model score, not a calibrated probability.",
        "Values outside [0, 1] (Fink's −1 'not computed' sentinel) are shown as null here and kept verbatim in the inspector.",
        censusNote(key),
        ...extra
      ]
    });
  const dimensions = [
    dim({
      id: "fink.n_dia_sources",
      domain: "fink",
      family: "multiplicity",
      label: "Delivered DiaSources per DiaObject",
      short: "N src",
      unit: "rows",
      scale: "log",
      definition: "Count of delivered DiaSource rows grouped by diaObjectId within the cohort.",
      evidence: TAI,
      qualifications: ["Light Static carries no complete history; this is not a lifetime detection count."]
    }),
    dim({
      id: "fink.time_baseline_days",
      domain: "fink",
      family: "time_baseline",
      label: "Delivered time baseline",
      short: "Δt",
      unit: "days",
      definition: "last − first midpointMjdTai of delivered DiaSources (TAI day difference).",
      evidence: TAI,
      qualifications: ["Bounded by the cohort window; not the object's full baseline.", "Zero for single-source DiaObjects."]
    }),
    dim({
      id: "fink.max_snr",
      domain: "fink",
      family: "photometry",
      label: "Peak DiaSource S/N",
      short: "S/N max",
      unit: null,
      scale: "log",
      definition: "Maximum snr over delivered DiaSources.",
      evidence: TAI,
      qualifications: ["Null when no delivered DiaSource has a positive snr."]
    }),
    dim({
      id: "fink.peak_psf_mag",
      domain: "fink",
      family: "photometry",
      label: "Peak difference-flux magnitude",
      short: "m_peak",
      unit: "AB mag",
      reversed: true,
      definition: "−2.5·log10(max psfFlux / nJy) + 31.4 over delivered DiaSources.",
      evidence: TAI,
      qualifications: ["Difference-image flux, not a total magnitude; undefined without a positive psfFlux."]
    }),
    dim({
      id: "fink.g_minus_r",
      domain: "fink",
      family: "colour",
      label: "g − r (mean difference-flux mag)",
      short: "g−r",
      unit: "mag",
      definition: "Mean positive-psfFlux AB magnitude in g minus that in r over delivered DiaSources.",
      state: "PARTIALLY_QUALIFIED",
      evidence: TAI,
      qualifications: ["Non-contemporaneous bands.", "Difference-image fluxes; not a source colour.", "Requires positive flux in both bands."]
    }),
    lcDim("chi2", "Reduced χ² about weighted mean", "χ²(r)", "log", null),
    lcDim("skew", "Skewness", "skew(r)", "linear", null),
    lcDim("stetson_K", "Stetson K", "K(r)", "linear", null),
    lcDim("amplitude", "Half range of psfFlux", "amp(r)", "log", "nJy"),
    lcDim("linear_fit_reduced_chi2", "Linear-fit reduced χ²", "χ²_lin(r)", "log", null),
    clfDim("snnSnVsOthers_score", "SuperNNova SN-vs-others score"),
    clfDim("cats_score", "CATS score", [
      `Where cats_class is −1 (no CATS class) the delivered cats_score 0.0 is a placeholder and is shown as null (${catsPlaceholders.toLocaleString("en-US")} of ${objects.length.toLocaleString("en-US")} sampled latest snapshots; inferred from the paired values).`
    ]),
    clfDim("earlySNIa_score", "Early SN Ia score"),
    ...positionDims("fink", TAI),
    dim({
      id: "fink.forced_photometry",
      domain: "fink",
      family: "photometry",
      label: "Forced photometry / upper limits",
      definition: "Forced-source fluxes and non-detections.",
      state: "UNAVAILABLE",
      unavailable_reason: "analysis_contract_v1 lists forced photometry / upper limits as a known absent capability of Light Static."
    }),
    dim({
      id: "fink.lifetime_detections",
      domain: "fink",
      family: "multiplicity",
      label: "Lifetime detection count",
      definition: "All historical detections of the DiaObject.",
      state: "UNAVAILABLE",
      unavailable_reason: "analysis_contract_v1 lists complete historical detections as a known absent capability."
    })
  ];
  const columns = {
    "fink.n_dia_sources": objects.map((o) => o.sources.length),
    "fink.time_baseline_days": objects.map((o) => round(o.sources.at(-1).midpointMjdTai - o.sources[0].midpointMjdTai, 4)),
    "fink.max_snr": objects.map((o) => round(positive(Math.max(...o.sources.map((s) => s.snr ?? -Infinity))), 2)),
    "fink.peak_psf_mag": objects.map((o) => {
      const peak = Math.max(...o.sources.map((s) => s.psfFlux));
      return peak > 0 ? round(31.4 - 2.5 * Math.log10(peak), 3) : null;
    }),
    "fink.g_minus_r": objects.map((o) => {
      const g = posMag(o, "g");
      const rr = posMag(o, "r");
      return g !== null && rr !== null ? round(g - rr, 3) : null;
    })
  };
  for (const dm of dimensions.filter((x) => x.id.startsWith("fink.lc_features.r."))) {
    const key = dm.id.slice("fink.lc_features.r.".length);
    columns[dm.id] = objects.map((o) => {
      const v = latestSnapshot(o).lc_features.r?.[key] ?? null;
      return dm.scale === "log" ? positive(v) : v !== null && Number.isFinite(v) ? v : null;
    });
  }
  for (const key of ["snnSnVsOthers_score", "cats_score", "earlySNIa_score"]) {
    columns[`fink.clf.${key}`] = objects.map((o) => {
      const snap = latestSnapshot(o);
      return key === "cats_score" && catsPlaceholder(snap) ? null : unitScore(snap.clf[key]);
    });
  }
  columns["fink.galactic_latitude"] = records.map((rec) => round(icrsToGalactic(rec.ra, rec.dec)[1], 4));
  columns["fink.ecliptic_latitude"] = records.map((rec) => round(icrsToEcliptic(rec.ra, rec.dec)[1], 4));
  const features = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    registry: VERSIONS.feature_registry,
    kernel: VERSIONS.analysis_kernel,
    dimensions,
    lab_defaults: { x: "fink.peak_psf_mag", y: "fink.clf.snnSnVsOthers_score" },
    columns
  };

  /* --- detail shards --- */
  const shards = Array.from({ length: DETAIL_SHARDS.fink }, (_, shard) => ({
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    shard,
    evidence: [TAI],
    records: {}
  }));
  objects.forEach((o) => {
    shards[shardOf(o.id, DETAIL_SHARDS.fink)].records[o.id] = {
      kind: "fink.diaObject",
      id: o.id,
      sources: o.sources.map((s) => ({
        diaSourceId: s.diaSourceId,
        midpointMjdTai: s.midpointMjdTai,
        band: s.band,
        psfFlux: s.psfFlux,
        psfFluxErr: s.psfFluxErr,
        snr: s.snr,
        reliability: s.reliability
      })),
      snapshots: o.snapshots.map((s) => ({
        diaSourceId: s.diaSourceId,
        midpointMjdTai: s.midpointMjdTai,
        fink_science_version: s.fink_science_version,
        pred: s.pred,
        clf: s.clf,
        xm: s.xm,
        lc_features: s.lc_features
      })),
      snapshot_policy:
        "First and last two delivered source-time snapshots per DiaObject (the G4A policy). clf values are verbatim (−1 is Fink's " +
        `'not computed' sentinel). lc_features shows ${sample.lc_feature_keys.length} of the ${schema.lc_features_fields.length} Light Static keys; ` +
        "xm fields Fink delivered as null are omitted."
    };
  });

  const pin = {
    domain: "fink",
    build_id: buildId,
    build_kind: "QUALIFIED_COHORT_CATALOG",
    label: `Fink · accepted ${c.cohort_name} catalog (${c.run_id})`,
    evidence: [TAI, "COMMITTED_OPERATIONAL_RECORD"],
    science_ready: false,
    native_ontology:
      "Rubin DiaSource rows delivered by Fink (Light Static packet). DiaObject is a derived grouping key; broker fields are source-time snapshots.",
    source: {
      repository: ev.repository,
      revision: c.code_commit_sha,
      paths: [c.cohort_config, "src/fink_lsst/analytics/contract.py", "src/fink_lsst/analytics/catalog.py"]
    },
    time: timeSemantics
  };

  const provenanceAcquisitions = acquisitions.map((a, i) => ({
    domain: "fink",
    acquisition_id: a.acquisition_id,
    label: `Month-${i + 1}`,
    window: { start: a.start, stop: a.stop, semantics: "half-open UTC date window [start, stop)" },
    science_profile: a.profile,
    state: a.state,
    state_at_utc: c.finished_utc,
    delivery: {
      topic: a.topic,
      readable_rows: a.validated_rows,
      parquet_files: a.parquet_files,
      total_bytes: a.raw_total_bytes,
      reconciliation_passed: a.expected_rows === a.validated_rows,
      terminal_lag: null,
      meaning: "Validated rows equal expected topic messages; raw stat fingerprint re-verified when the catalog was opened. Transport evidence only."
    },
    characterization: a.characterization,
    delivery_validation: "DELIVERY_VALIDATED",
    admission: "ADMITTED",
    scientific_status: "CHARACTERIZED",
    cohort: c.cohort_name
  }));
  const fieldEvidence = [
    ...fields.map((f) => ({ field: f.key, evidence: f.evidence, note: f.description })),
    { field: "acquisition windows", evidence: "COMMITTED_OPERATIONAL_RECORD", note: "Catalog build manifest and cohort configuration." },
    { field: "per-date delivered rows", evidence: TAI, note: "Catalog daily_coverage, recomputed from sources at extraction. Transport evidence only." },
    { field: "sky density", evidence: TAI, note: "Every derived DIA group, HEALPix NESTED order 6." }
  ];
  return { pin, time, sky, entities, features, shards, windows, acquisitions: provenanceAcquisitions, fieldEvidence, buildId, ev, cat };
}

/* --------------------------------------------------------------- assemble */

const antares = buildAntares();
const fink = buildFink();

const capabilities = [];
const both = { antares, fink };
// Time
capabilities.push(
  cap("antares", "time", "date_states", "AVAILABLE", ["SYNTHETIC_DEMO"], "Per-date states of the legacy demo export",
    "Every demo date carries a state; the date assignment itself is synthetic.", { codes: ["SYNTHETIC_DATE_ASSIGNMENT"] }),
  cap("antares", "time", "date_counts", "PARTIALLY_QUALIFIED", ["SYNTHETIC_DEMO"], "Demo-sample loci per synthetic date",
    "Counts are allocations of a legacy sample to synthetic dates, not ANTARES nightly totals.", { codes: ["SAMPLE_ALLOCATION"] }),
  cap("antares", "time", "transport_totals", "UNAVAILABLE", [], "Published nightly totals",
    "No published ANTARES nightly manifests are pinned in this basis.", { codes: ["NOT_IN_BASIS"] }),
  cap("fink", "time", "date_states", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE", "COMMITTED_OPERATIONAL_RECORD"],
    "Per-date states from the accepted cohort catalog",
    "Each date inherits its acquisition window's state; dates with no delivered rows are ZERO, not missing.",
    { qualifications: ["A zero-row date is not a survey non-observation."] }),
  cap("fink", "time", "date_counts", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Delivered alert rows per UTC date (complete)",
    "Every delivered DIA and SSO DiaSource packet of the cohort, binned by the UTC date of midpointMjdTai.",
    { codes: ["TRANSPORT_ONLY"], qualifications: ["Delivered rows are not Rubin completeness and are never read as alert rates."] }),
  cap("fink", "time", "transport_totals", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Delivered row totals per acquisition",
    "Validated rows equal expected topic messages for every acquisition in the cohort.",
    { codes: ["TRANSPORT_ONLY"], qualifications: ["Transport totals are not Rubin completeness and must never be compared across windows as rates."] })
);
for (const w of fink.windows) {
  const name = `acquisition.${w.id.split(".")[1]}`;
  const state = w.state === "AVAILABLE" ? "AVAILABLE" : w.state === "UNQUALIFIED" ? "PARTIALLY_QUALIFIED" : "UNAVAILABLE";
  capabilities.push(
    cap("fink", "time", name, state, state === "UNAVAILABLE" ? [] : w.evidence, w.label, w.caveat, { codes: w.status_codes })
  );
}
// Sky
capabilities.push(
  cap("antares", "sky", "density", "AVAILABLE", ["LEGACY_SAMPLE"], "Locus density on HEALPix", "Counts of demo-sample loci per equal-area cell."),
  cap("antares", "sky", "coverage", "UNAVAILABLE", [], "Footprint / coverage", antares.sky.coverage_unavailable_reason, { codes: ["NOT_IN_BASIS"] }),
  cap("antares", "sky", "filtered_density", "AVAILABLE", ["LEGACY_SAMPLE"], "Cross-filtered density", "The entity table is the complete population of this basis."),
  cap("fink", "sky", "density", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "DiaObject density on HEALPix (complete)",
    "Counts of every derived DIA group in the cohort per equal-area cell."),
  cap("fink", "sky", "coverage", "UNAVAILABLE", [], "Footprint / coverage", fink.sky.coverage_unavailable_reason, { codes: ["NOT_IN_BASIS"] }),
  cap("fink", "sky", "filtered_density", "UNAVAILABLE", [], "Cross-filtered density",
    "The Fink entity layer is a random sample; filtering it would present sample counts as population counts. Filters apply to sampled points only.",
    { codes: ["SAMPLED_ENTITY_LAYER"] })
);
// Entities
capabilities.push(
  cap("antares", "entity", "inspector", "AVAILABLE", ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"], "Native ANTARES locus record", "Supported locus fields with per-field evidence."),
  cap("antares", "entity", "lightcurve", "PARTIALLY_QUALIFIED", ["SYNTHETIC_DEMO"], "Brightness history",
    "Only synthetic demo stories exist (10 loci); they are not ANTARES alert photometry.", { codes: ["SYNTHETIC_DEMO"] }),
  cap("antares", "entity", "broker_inference", "PARTIALLY_QUALIFIED", ["LEGACY_SAMPLE"], "ANTARES tags",
    "Filter-pipeline memberships only; not astrophysical classes and not scores.", { codes: ["TAGS_ARE_NOT_CLASSES"] }),
  cap("fink", "entity", "inspector", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Native Fink DiaObject / DiaSource record",
    "Delivered DiaSources and source-time broker snapshots of the sampled DiaObjects.", { qualifications: [fink.entities.population.sampling] }),
  cap("fink", "entity", "lightcurve", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Delivered psfFlux history",
    "Difference-image psfFlux of delivered DiaSources per band.", { qualifications: ["No forced photometry or upper limits (Light Static)."] }),
  cap("fink", "entity", "broker_inference", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Source-time classifier snapshots",
    "clf/pred/xm/lc_features as delivered with each DiaSource.",
    { qualifications: ["Model scores are not calibrated probabilities.", "Snapshots are not timeless classifications.", "fink_science_version changes within the cohort; scores from different versions are not strictly comparable."] })
);
// Features
for (const d of ["antares", "fink"]) {
  const dims = both[d].features.dimensions;
  const usable = dims.filter((x) => x.state !== "UNAVAILABLE").length;
  capabilities.push(
    d === "antares"
      ? cap(d, "features", "population", "AVAILABLE", ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"],
          "Parameter-space population", `${usable} of ${dims.length} registered dimensions are selectable; the rest are declared unavailable with reasons.`)
      : cap(d, "features", "population", "PARTIALLY_QUALIFIED", ["VALIDATED_TRANSPORT_EVIDENCE"], "Parameter-space population (random sample)",
          `${usable} of ${dims.length} registered dimensions are selectable over the sampled DiaObjects; the rest are declared unavailable with reasons.`,
          { codes: ["SAMPLED_ENTITY_LAYER"], qualifications: [fink.entities.population.sampling] })
  );
}
// Relation and workspace
capabilities.push(
  cap("relation", "relation", "cross_broker_association", "UNAVAILABLE", [], "Cross-broker association",
    "Cross-broker association is a future versioned scientific relation product. No ANTARES locus is matched to any Fink DiaObject in this basis.",
    { codes: ["NOT_IMPLEMENTED_IN_GATE"] }),
  cap("workspace", "compare", "side_by_side", "AVAILABLE", [], "Independent side-by-side lenses",
    "Each domain is drawn on its own normalization over synchronized geometry."),
  cap("workspace", "compare", "difference_map", "UNAVAILABLE", [], "Count difference map",
    "Raw ANTARES-minus-Fink counts are not meaningful: different entity ontologies, selection functions, time scales and evidence classes.",
    { codes: ["SCIENTIFICALLY_INVALID"] }),
  cap("workspace", "compare", "shared_axes", "PARTIALLY_QUALIFIED", [], "Shared Lab axes",
    "Only dimensions with an identical definition_id (kernel-derived Galactic/ecliptic latitude) may share axes across domains."),
  cap("workspace", "time", "shared_utc_date_axis", "AVAILABLE", [], "Shared UTC-date axis",
    "Lanes share UTC-date bins only. Each domain maps its own time scale to UTC dates; MJDs are never aligned across domains.",
    { qualifications: ["ANTARES MJD is treated as UTC by the historical exporter; Fink midpointMjdTai is TAI."] })
);

const BASIS_ID = basisIdFor({
  domains: { antares: antares.pin.build_id, fink: fink.pin.build_id },
  relation: null,
  ...VERSIONS
});
const basis = {
  contract: CONTRACT,
  contract_version: CONTRACT_VERSION,
  basis_id: BASIS_ID,
  label: "First Light · ANTARES demo + Fink accepted catalog",
  status: "FIRST_LIGHT_FIXTURE",
  science_ready: false,
  domains: { antares: antares.pin, fink: fink.pin },
  relation: null,
  ...VERSIONS,
  invariants: [
    "An ANTARES locus is not a Fink DiaObject.",
    "An ANTARES tag is not an astrophysical class.",
    "A Fink classification is a source-time snapshot, not a timeless object classification.",
    "A model score is not a calibrated probability.",
    "Fink lag-zero transport is not Rubin scientific completeness.",
    "Readable Parquet is not science readiness.",
    "A missing ANTARES night is not a zero-row night.",
    "Source density is not survey coverage.",
    "Fink midpointMjdTai is TAI; the historical ANTARES exporter treats MJD as UTC. They are never silently aligned.",
    "No cross-broker identity matching is performed in this basis.",
    "The Fink entity layer is a random sample; sample counts are never population counts."
  ]
};

const provenance = {
  contract: CONTRACT,
  basis_id: BASIS_ID,
  sources: [
    { id: "antares.sky_points", label: "SkyPulse demo sky points", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.skyPoints}`, sha256: inputs.skyPoints.sha256, role: "ANTARES legacy demo loci", evidence: "LEGACY_SAMPLE" },
    { id: "antares.lightcurve_samples", label: "SkyPulse demo lightcurves", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.lightcurves}`, sha256: inputs.lightcurves.sha256, role: "Synthetic demo brightness stories", evidence: "SYNTHETIC_DEMO" },
    { id: "antares.public_manifest", label: "SkyPulse demo manifest", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.demoManifest}`, sha256: inputs.demoManifest.sha256, role: "Demo export identity", evidence: "SYNTHETIC_DEMO" },
    { id: "fink.evidence_excerpt", label: "Fink evidence excerpt", repository: "darim1151/ANTARES_Analysis", revision: null, path: `web/${inputPaths.finkEvidence}`, sha256: inputs.finkEvidence.sha256, role: "Pinned Light Static schema (broker field names)", evidence: "COMMITTED_OPERATIONAL_RECORD" },
    { id: "fink.catalog_extract", label: "Fink catalog extract", repository: "darim1151/ANTARES_Analysis", revision: null, path: `web/${inputPaths.finkCatalog}`, sha256: inputs.finkCatalog.sha256, role: "Read-only extract of the accepted Fink catalog", evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
    { id: "fink.catalog", label: `${fink.cat.catalog.run_id} analytics catalog`, repository: "Fink data root (Arnor)", revision: fink.cat.catalog.code_commit_sha, path: fink.cat.catalog.catalog_relative_path, sha256: fink.cat.catalog.catalog_sha256, role: `Accepted ${fink.cat.catalog.cohort_name} DuckDB catalog (opened read-only)`, evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
    { id: "fink.catalog_manifest", label: "Catalog build manifest", repository: "Fink data root (Arnor)", revision: fink.cat.catalog.code_commit_sha, path: fink.cat.catalog.manifest_relative_path, sha256: fink.cat.catalog.manifest_sha256, role: `Build manifest (${fink.cat.catalog.completion_status})`, evidence: "COMMITTED_OPERATIONAL_RECORD" },
    ...fink.ev.extracted_files.map((f) => ({
      id: `fink.${f.path}`,
      label: path.basename(f.path),
      repository: fink.ev.repository,
      revision: fink.ev.revision,
      path: f.path,
      sha256: f.sha256,
      role: "Committed Fink evidence",
      evidence: f.path.includes("/evidence/delivery") ? "VALIDATED_TRANSPORT_EVIDENCE" : "COMMITTED_OPERATIONAL_RECORD"
    }))
  ],
  acquisitions: fink.acquisitions,
  derivations: [
    { id: "healpix.nested", description: `Entity positions indexed on HEALPix NESTED; density maps at order ${DENSITY_ORDER}.`, kernel: KERNEL_TAG },
    { id: "coords.galactic", description: "ICRS -> Galactic with the Astropy/Hipparcos rotation matrix.", kernel: KERNEL_TAG },
    { id: "coords.ecliptic", description: "ICRS -> mean ecliptic of J2000.0 (obliquity 23.4392911°).", kernel: KERNEL_TAG },
    { id: "fink.object_position", description: "DiaObject position = unit-vector mean of delivered DiaSource positions, rounded to 1e-5 deg.", kernel: KERNEL_TAG },
    { id: "fink.density", description: `Complete DiaObject density computed at extraction with a NumPy port of the kernel's ang2pixNest; every sampled object's cell is re-checked against the kernel.`, kernel: KERNEL_TAG },
    { id: "fink.time_scale", description: `UTC dates from TAI with TAI − UTC = ${FINK_TAI_MINUS_UTC_S} s, verified with ERFA over every cohort date.`, kernel: KERNEL_TAG },
    { id: "fink.sample", description: fink.cat.sample.method, kernel: KERNEL_TAG },
    { id: "antares.clipped_magnitudes", description: `${antares.clipped} demo magnitudes at the exporter clip bounds were nulled.`, kernel: KERNEL_TAG }
  ],
  field_evidence: { antares: antares.fieldEvidence, fink: fink.fieldEvidence }
};

/* ----------------------------------------------------------------- write */

const files = new Map();
const pretty = (v) => `${JSON.stringify(v, null, 2)}\n`;
const compact = (v) => `${JSON.stringify(v)}\n`;
files.set("basis.json", { text: pretty(basis), role: "basis" });
files.set("capabilities.json", {
  text: pretty({ contract: CONTRACT, contract_version: CONTRACT_VERSION, basis_id: BASIS_ID, capabilities }),
  role: "capabilities"
});
files.set("provenance.json", { text: pretty(provenance), role: "provenance" });
for (const [d, built] of Object.entries(both)) {
  files.set(`domains/${d}/time.json`, { text: pretty(built.time), role: `${d}.time` });
  files.set(`domains/${d}/sky.json`, { text: compact(built.sky), role: `${d}.sky` });
  files.set(`domains/${d}/entities.json`, { text: compact(built.entities), role: `${d}.entities` });
  files.set(`domains/${d}/features.json`, { text: compact(built.features), role: `${d}.features` });
  for (const shard of built.shards) {
    files.set(`domains/${d}/${shardPath(built.entities.detail.path_template, shard.shard)}`, { text: compact(shard), role: `${d}.detail` });
  }
}

const asOf = [inputs.demoManifest.json.generated_at_utc, fink.cat.catalog.finished_utc]
  .map((s) => s.replace(/(\.\d+)?\+00:00$/, "Z"))
  .sort()
  .at(-1);
const manifest = {
  contract: CONTRACT,
  contract_version: CONTRACT_VERSION,
  bundle_id: BUNDLE_ID,
  bundle_class: "FIRST_LIGHT_FIXTURE",
  science_ready: false,
  as_of_utc: asOf,
  evidence_policy:
    "ANTARES fixture and demo values are labelled per payload and per field and must never be presented as accepted science. " +
    "Fink values are read from the accepted five-window catalog and are transport/operational evidence, not accepted science. " +
    "Fink per-date counts and sky density are complete; the Fink entity layer is a labelled random sample.",
  generator: {
    ...GENERATOR,
    deterministic: true,
    seed: null,
    inputs: Object.values(inputs).map((i) => ({
      path: `web/${i.relative}`,
      sha256: i.sha256,
      role: i === inputs.finkEvidence ? "fink-evidence" : i === inputs.finkCatalog ? "fink-catalog-extract" : "antares-legacy-demo"
    }))
  },
  basis: "basis.json",
  capabilities: "capabilities.json",
  provenance: "provenance.json",
  domains: Object.fromEntries(
    Object.keys(both).map((d) => [
      d,
      { time: `domains/${d}/time.json`, sky: `domains/${d}/sky.json`, entities: `domains/${d}/entities.json`, features: `domains/${d}/features.json` }
    ])
  ),
  files: [...files.entries()]
    .map(([p, f]) => ({ path: p, bytes: Buffer.byteLength(f.text), sha256: sha256(f.text), role: f.role }))
    .sort((a, b) => a.path.localeCompare(b.path))
};
files.set("manifest.json", { text: pretty(manifest), role: "manifest" });

async function listTree(root, prefix = "") {
  const out = [];
  let entries = [];
  try {
    entries = await readdir(path.join(root, prefix), { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") return out;
    throw error;
  }
  for (const e of entries) {
    const rel = prefix ? `${prefix}/${e.name}` : e.name;
    if (e.isDirectory()) out.push(...(await listTree(root, rel)));
    else out.push(rel);
  }
  return out;
}

if (check) {
  const existing = new Set(await listTree(outputRoot));
  const problems = [];
  for (const [p, f] of files) {
    if (!existing.has(p)) problems.push(`missing ${p}`);
    else if ((await readFile(path.join(outputRoot, p), "utf8")) !== f.text) problems.push(`differs ${p}`);
    existing.delete(p);
  }
  for (const p of existing) problems.push(`unexpected ${p}`);
  if (problems.length) {
    console.error("Committed First-Light bundle is not the deterministic output of its inputs:");
    problems.forEach((p) => console.error(`- ${p}`));
    process.exit(1);
  }
  console.log(`First-Light bundle is reproducible: ${files.size} files match the generator output.`);
} else {
  await rm(outputRoot, { recursive: true, force: true });
  let bytes = 0;
  for (const [p, f] of files) {
    await mkdir(path.dirname(path.join(outputRoot, p)), { recursive: true });
    await writeFile(path.join(outputRoot, p), f.text, "utf8");
    bytes += Buffer.byteLength(f.text);
  }
  console.log(
    `Wrote ${files.size} files (${(bytes / 1e6).toFixed(2)} MB) to ${path.relative(webRoot, outputRoot)}: ` +
      `ANTARES ${antares.entities.records.length} loci (${antares.clipped} clipped magnitudes nulled), ` +
      `Fink ${fink.entities.records.length} sampled of ${fink.entities.population.total} DiaObjects, ${fink.sky.density.pixels.length} density cells.`
  );
}
