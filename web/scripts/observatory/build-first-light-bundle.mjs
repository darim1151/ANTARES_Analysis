#!/usr/bin/env node
// Build the Unified Scientific Observatory First-Light bundle.
//
//   node scripts/observatory/build-first-light-bundle.mjs           # write
//   node scripts/observatory/build-first-light-bundle.mjs --check   # verify committed bundle
//
// Deterministic: no wall clock, no network, seeded PRNG. Inputs:
//   - public/data/{sky_points,lightcurve_samples,public_manifest}.json
//     (the existing SkyPulse LEGACY DEMO contract; read only)
//   - scripts/observatory/inputs/fink-evidence.fd02c8e.json
//     (committed Fink acquisition evidence at the pinned revision)
//
// Scientific honesty rules enforced here (and re-checked by the validator):
//   - ANTARES demo values are labelled LEGACY_SAMPLE or SYNTHETIC_DEMO per field;
//     values the demo exporter clipped are emitted as null.
//   - The Fink population is a SYNTHETIC_FIXTURE confined to the Month-1 window.
//     Month-2 (transport-validated, uncharacterized) and Month-3 (producer
//     complete, not delivery-validated, not admitted) carry no counts at all.
//   - No cross-broker relation is produced.

import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { normalizeMoc, pixToRaDec, radecToPix } from "../../lib/observatory/kernel/healpix.ts";
import { icrsToEcliptic, icrsToGalactic } from "../../lib/observatory/kernel/astro.ts";
import { addDays, utcDateRange, utcDateToMs, utcMjdToUtcDate } from "../../lib/observatory/kernel/dates.ts";
import { fnv1a32, shardOf, shardPath } from "../../lib/observatory/kernel/shard.ts";

const CONTRACT = "uso.observatory-bundle";
const CONTRACT_VERSION = "1.0.0";
const GENERATOR = { name: "web/scripts/observatory/build-first-light-bundle.mjs", version: "1.1.0" };
// Adapter tag carried by every build id, so a changed adapter yields a new build identity.
const ADAPTER_TAG = `g${GENERATOR.version.split(".").slice(0, 2).join(".")}`;
const BUNDLE_ID = "uso-first-light-0001";
// The basis id is derived from what the basis pins (see basisIdFor), never a free constant.
const BASIS_PREFIX = "basis.uso.first-light";
const SEED = "uso-first-light/fink-fixture/v1";
const BASELINE_REVISION = "812c545e14693cdce7ff7458f1d2b50b0804dcd8";
const DENSITY_ORDER = 6;
const COVERAGE_ORDER = 6;
const TAI_MINUS_UTC_S = 37; // valid since 2017-01-01; fixture-only conversion
const FINK_FIXTURE_OBJECTS = 1800;
const FIXTURE_FOOTPRINT_DEC = [-72, 8];
const DETAIL_SHARDS = { antares: 4, fink: 16 };
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
  finkEvidence: "scripts/observatory/inputs/fink-evidence.fd02c8e.json"
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
function mjdUtcOfDate(date) {
  return utcDateToMs(date) / 86_400_000 + 40587;
}

function prng(seedText) {
  let a = fnv1a32(seedText);
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.uniform = (lo, hi) => lo + (hi - lo) * next();
  next.int = (lo, hiInclusive) => lo + Math.floor(next() * (hiInclusive - lo + 1));
  next.normal = () => {
    const u = Math.max(next(), 1e-12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next());
  };
  next.pick = (weights) => {
    const total = Object.values(weights).reduce((s, w) => s + w, 0);
    let x = next() * total;
    for (const [key, w] of Object.entries(weights)) {
      x -= w;
      if (x <= 0) return key;
    }
    return Object.keys(weights).at(-1);
  };
  return next;
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
  return { relative, bytes, sha256: sha256(bytes), json: JSON.parse(bytes.toString("utf8")) };
}

const inputs = {
  skyPoints: await readInput(inputPaths.skyPoints),
  lightcurves: await readInput(inputPaths.lightcurves),
  demoManifest: await readInput(inputPaths.demoManifest),
  finkEvidence: await readInput(inputPaths.finkEvidence)
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

function lcFeatures(points, allowedKeys) {
  // points: [{t, f, e}] sorted by t; returns a map of computed features.
  const n = points.length;
  if (n < 3) return null;
  const f = points.map((p) => p.f);
  const e = points.map((p) => p.e);
  const t = points.map((p) => p.t);
  const mean = f.reduce((s, v) => s + v, 0) / n;
  const w = e.map((x) => 1 / (x * x));
  const wsum = w.reduce((s, v) => s + v, 0);
  const wmean = f.reduce((s, v, i) => s + v * w[i], 0) / wsum;
  const sd = Math.sqrt(f.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1));
  const sorted = [...f].sort((a, b) => a - b);
  const q = (p) => {
    const pos = (n - 1) * p;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  };
  const median = q(0.5);
  const amplitude = (sorted[n - 1] - sorted[0]) / 2;
  const m2 = f.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const m3 = f.reduce((s, v) => s + (v - mean) ** 3, 0) / n;
  const m4 = f.reduce((s, v) => s + (v - mean) ** 4, 0) / n;
  const skew = m2 > 0 ? (Math.sqrt(n * (n - 1)) / (n - 2)) * (m3 / m2 ** 1.5) : null;
  const kurtosis = n >= 4 && m2 > 0 ? ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * (m4 / m2 ** 2 - 3) + 6) : null;
  const tm = t.reduce((s, v) => s + v, 0) / n;
  const stt = t.reduce((s, v) => s + (v - tm) ** 2, 0);
  const slope = stt > 0 ? t.reduce((s, v, i) => s + (v - tm) * (f[i] - mean), 0) / stt : null;
  const resid = slope === null ? null : f.map((v, i) => v - (mean + slope * (t[i] - tm)));
  const noise = resid && n > 2 ? Math.sqrt(resid.reduce((s, v) => s + v * v, 0) / (n - 2)) : null;
  const slopeSigma = noise !== null && stt > 0 ? noise / Math.sqrt(stt) : null;
  const twm = t.reduce((s, v, i) => s + v * w[i], 0) / wsum;
  const wtt = t.reduce((s, v, i) => s + w[i] * (v - twm) ** 2, 0);
  const wslope = wtt > 0 ? t.reduce((s, v, i) => s + w[i] * (v - twm) * (f[i] - wmean), 0) / wtt : null;
  const wfitChi2 =
    wslope === null || n <= 2
      ? null
      : f.reduce((s, v, i) => s + ((v - (wmean + wslope * (t[i] - twm))) / e[i]) ** 2, 0) / (n - 2);
  let maxSlope = 0;
  for (let i = 1; i < n; i += 1) {
    const dt = t[i] - t[i - 1];
    if (dt > 0) maxSlope = Math.max(maxSlope, Math.abs((f[i] - f[i - 1]) / dt));
  }
  const mad = (() => {
    const dev = f.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
    return dev[Math.floor((n - 1) / 2)];
  })();
  const chi2 = f.reduce((s, v, i) => s + ((v - wmean) / e[i]) ** 2, 0) / (n - 1);
  const delta = f.map((v, i) => Math.sqrt(n / (n - 1)) * ((v - wmean) / e[i]));
  const stetsonK =
    delta.reduce((s, v) => s + Math.abs(v), 0) / n / Math.sqrt(delta.reduce((s, v) => s + v * v, 0) / n);
  const values = {
    mean,
    weighted_mean: wmean,
    standard_deviation: sd,
    median,
    amplitude,
    beyond_1_std: f.filter((v) => Math.abs(v - mean) > sd).length / n,
    inter_percentile_range_10: q(0.9) - q(0.1),
    kurtosis,
    linear_trend: slope,
    linear_trend_sigma: slopeSigma,
    linear_trend_noise: noise,
    linear_fit_slope: wslope,
    linear_fit_slope_sigma: wtt > 0 ? 1 / Math.sqrt(wtt) : null,
    linear_fit_reduced_chi2: wfitChi2,
    maximum_slope: maxSlope,
    median_absolute_deviation: mad,
    median_buffer_range_percentage_10: f.filter((v) => Math.abs(v - median) < 0.1 * (sorted[n - 1] - sorted[0])).length / n,
    percent_amplitude: Math.max(Math.abs(sorted[n - 1] - median), Math.abs(sorted[0] - median)),
    mean_variance: Math.abs(mean) > 1e-9 ? sd / mean : null,
    chi2,
    skew,
    stetson_K: stetsonK
  };
  const out = {};
  for (const [key, value] of Object.entries(values)) {
    if (!allowedKeys.includes(key)) fail(`fixture lc_features key ${key} is not in the pinned Light Static schema`);
    const v = sig(value, 4);
    if (v !== null) out[key] = v;
  }
  return out;
}

function buildFink() {
  const ev = inputs.finkEvidence.json;
  if (ev.revision !== "fd02c8eabcaf3a1160e0ed2c5c1d6959ac11d23d") fail("Fink evidence excerpt is not at the pinned revision");
  const schema = ev.light_static_schema;
  const byLabel = Object.fromEntries(ev.acquisitions.map((a) => [a.label, a]));
  const m1 = byLabel["Month-1"];
  if (!m1 || m1.state !== "DELIVERY_VALIDATED" || !m1.characterization || !ev.cohort.acquisitions.includes(m1.acquisition_id)) {
    fail("Month-1 must be delivery-validated, characterized and admitted to the accepted cohort");
  }
  const buildId = `fink.first-light-fixture.${ev.revision.slice(0, 7)}.${ADAPTER_TAG}`;

  // Acquisition windows strictly from committed evidence.
  const windows = ev.acquisitions.map((a) => {
    const validated = a.state === "DELIVERY_VALIDATED" && a.delivery?.reconciliation_passed === true;
    const admitted = ev.cohort.acquisitions.includes(a.acquisition_id);
    const characterized = Boolean(a.characterization);
    let state;
    let codes;
    let caveat;
    if (validated && admitted && characterized) {
      state = "AVAILABLE";
      codes = ["DELIVERY_VALIDATED", "ADMITTED", "CHARACTERIZED"];
      caveat =
        "Delivery reconciled three ways (expected topic messages = terminal committed = local readable rows, lag 0) and characterized (G4A). " +
        "Transport reconciliation is not Rubin scientific completeness. Per-date delivered counts are not committed evidence and are not shown.";
    } else if (validated) {
      state = "UNQUALIFIED";
      codes = ["DELIVERY_VALIDATED", admitted ? "ADMITTED" : "NOT_ADMITTED", characterized ? "CHARACTERIZED" : "UNCHARACTERIZED"];
      caveat =
        "Transport-validated but scientifically uncharacterized and not admitted to the analytical cohort. " +
        "Its delivered row total must not be read as an alert rate or compared with other windows.";
    } else {
      state = "UNAVAILABLE";
      codes = ["NOT_DELIVERY_VALIDATED", "NOT_ADMITTED"];
      caveat = `Upstream state ${a.state}: not delivery-validated and not admitted to this basis. This is not missing data; it is not yet available.`;
    }
    const facts = [
      { label: "Acquisition state", value: a.state, evidence: "COMMITTED_OPERATIONAL_RECORD" },
      { label: "State recorded (UTC)", value: a.state_at_utc, evidence: "COMMITTED_OPERATIONAL_RECORD" },
      { label: "Science profile", value: a.science_profile, evidence: "COMMITTED_OPERATIONAL_RECORD" }
    ];
    if (a.delivery) {
      facts.push(
        { label: "Delivered readable rows (transport)", value: a.delivery.readable_rows, unit: "rows", evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
        { label: "Readable Parquet files", value: a.delivery.readable_parquet_files, unit: "files", evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
        { label: "Delivered bytes", value: a.delivery.total_bytes, unit: "bytes", evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
        { label: "Three-way reconciliation", value: a.delivery.reconciliation_passed, evidence: "VALIDATED_TRANSPORT_EVIDENCE" },
        { label: "Terminal consumer lag", value: a.delivery.terminal_lag, unit: "messages", evidence: "VALIDATED_TRANSPORT_EVIDENCE" }
      );
    }
    if (a.characterization) facts.push({ label: "Characterization run", value: a.characterization.run_id, evidence: "COMMITTED_OPERATIONAL_RECORD" });
    facts.push({ label: "Analytical cohort", value: admitted ? ev.cohort.cohort_name : "not admitted", evidence: "COMMITTED_OPERATIONAL_RECORD" });
    return {
      id: `fink.${a.label.toLowerCase()}`,
      label: `${a.label} · ${a.window.start} → ${a.window.stop}`,
      start: a.window.start,
      stop: a.window.stop,
      state,
      source_state: a.state,
      delivery_validation: validated ? "DELIVERY_VALIDATED" : "NOT_DELIVERY_VALIDATED",
      admission: admitted ? "ADMITTED" : "NOT_ADMITTED",
      status_codes: codes,
      evidence: a.delivery ? ["VALIDATED_TRANSPORT_EVIDENCE", "COMMITTED_OPERATIONAL_RECORD"] : ["COMMITTED_OPERATIONAL_RECORD"],
      facts,
      rate_comparison: "PROHIBITED",
      caveat
    };
  });
  const range = { start: windows[0].start, stop: windows.at(-1).stop };
  const nights = nightStatesFromWindows(range, windows);

  /* --- synthetic fixture population, confined to the Month-1 window --- */
  const r = prng(SEED);
  const m1Dates = utcDateRange(m1.window.start, m1.window.stop);
  const m1StopMjd = mjdUtcOfDate(m1.window.stop);
  const dateWeights = Object.fromEntries(m1Dates.map((d) => [d, 0.45 + 0.9 * r()]));
  const coverage = new Set();
  // Fixture footprint at COVERAGE_ORDER: cells whose center lies in the declination band.
  for (let p = 0; p < 12 * 4 ** COVERAGE_ORDER; p += 1) {
    const [, dec] = pixToRaDec(COVERAGE_ORDER, p);
    if (dec >= FIXTURE_FOOTPRINT_DEC[0] && dec <= FIXTURE_FOOTPRINT_DEC[1]) coverage.add(p);
  }
  const archetypeWeights = { transient: 0.3, periodic: 0.35, stochastic: 0.15, sparse: 0.2 };
  const bandWeights = { u: 0.04, g: 0.24, r: 0.28, i: 0.24, z: 0.14, y: 0.06 };
  const bandDepth = { u: 2.2, g: 1, r: 1, i: 1.2, z: 1.6, y: 2.4 };
  const bandColour = { u: 0.6, g: 1, r: 0.95, i: 0.85, z: 0.75, y: 0.65 };
  const flux = (mag) => 10 ** ((31.4 - mag) / 2.5);
  const lcKeys = schema.lc_features_fields;
  for (const key of ["snnSnVsOthers_score", "cats_class", "cats_score", "earlySNIa_score"]) {
    if (!schema.clf_fields.includes(key)) fail(`clf field ${key} absent from pinned schema`);
  }

  const objects = [];
  let sourceCounter = 0;
  for (let i = 0; i < FINK_FIXTURE_OBJECTS; i += 1) {
    const archetype = r.pick(archetypeWeights);
    let ra;
    let dec;
    for (;;) {
      ra = r() * 360;
      dec = (Math.asin(2 * r() - 1) * 180) / Math.PI;
      if (!coverage.has(radecToPix(COVERAGE_ORDER, ra, dec))) continue;
      const [l, b] = icrsToGalactic(ra, dec);
      const lw = l > 180 ? l - 360 : l;
      const absB = Math.abs(b);
      let weight = 1;
      if (archetype === "periodic") weight = 0.12 + 0.88 * Math.exp(-absB / 9) + 0.8 * Math.exp(-((lw / 14) ** 2) - (b / 9) ** 2);
      if (archetype === "transient" || archetype === "stochastic") weight = 0.1 + 0.9 * Math.min(1, absB / 35);
      if (r() < Math.min(1, weight)) break;
    }
    const firstDate = r.pick(dateWeights);
    const nTarget = {
      transient: r.int(2, 9),
      periodic: r.int(6, 30),
      stochastic: r.int(4, 18),
      sparse: r() < 0.3 ? 2 : 1
    }[archetype];
    const gap = {
      transient: () => r.uniform(0.6, 4.5),
      periodic: () => r.uniform(0.3, 2.6),
      stochastic: () => r.uniform(0.9, 4.2),
      sparse: () => r.uniform(0.012, 0.05)
    }[archetype];
    const m0 = { transient: r.uniform(20, 23.4), periodic: r.uniform(17.5, 22), stochastic: r.uniform(19, 22.5), sparse: r.uniform(21, 23.8) }[archetype];
    const peakT = r.uniform(1, 9);
    const rise = r.uniform(2, 6);
    const decline = r.uniform(8, 26);
    const period = 10 ** r.uniform(Math.log10(0.3), Math.log10(14));
    const ampFrac = r.uniform(0.15, 0.85);
    const phase = r.uniform(0, 2 * Math.PI);
    const drift = r.uniform(-0.03, 0.03);
    const stochastic = [r.uniform(4, 30), r.uniform(4, 30), r.uniform(0, 6.28), r.uniform(0, 6.28)];

    const t0 = mjdUtcOfDate(firstDate) + r.uniform(0.02, 0.4);
    const times = [t0];
    while (times.length < nTarget) {
      const next = times.at(-1) + gap();
      if (next >= m1StopMjd) break; // the fixture never extends past the admitted Month-1 window
      times.push(next);
    }
    const sources = times.map((tUtc) => {
      sourceCounter += 1;
      const band = r.pick(bandWeights);
      const dt = tUtc - t0;
      let model;
      if (archetype === "transient") {
        model = flux(m0) * (dt < peakT ? Math.exp(-(peakT - dt) / rise) : Math.exp(-(dt - peakT) / decline));
      } else if (archetype === "periodic") {
        model = flux(m0) * ampFrac * Math.sin((2 * Math.PI * dt) / period + phase);
      } else if (archetype === "stochastic") {
        model =
          flux(m0) * (0.25 * Math.sin((2 * Math.PI * dt) / stochastic[0] + stochastic[2]) + 0.18 * Math.sin((2 * Math.PI * dt) / stochastic[1] + stochastic[3]) + drift * dt);
      } else {
        model = flux(m0) * r.uniform(0.8, 1.25);
      }
      model *= bandColour[band];
      const err = 180 * r.uniform(0.7, 1.6) * bandDepth[band];
      const observed = model + err * r.normal();
      return {
        diaSourceId: `991${String(sourceCounter).padStart(15, "0")}`,
        midpointMjdTai: round(tUtc + TAI_MINUS_UTC_S / 86400, 6),
        tUtc,
        band,
        psfFlux: round(observed, 1),
        psfFluxErr: round(err, 1),
        snr: round(Math.abs(observed) / err, 2),
        reliability: round(archetype === "sparse" ? r.uniform(0.12, 0.8) : r.uniform(0.74, 0.995), 3),
        ra: ra + (r.normal() * 0.1) / 3600 / Math.max(0.05, Math.cos((dec * Math.PI) / 180)),
        dec: dec + (r.normal() * 0.1) / 3600
      };
    });

    const cataloged = archetype === "periodic" ? r() < 0.5 : archetype === "stochastic" ? r() < 0.3 : r() < 0.05;
    const otype =
      archetype === "periodic" && r() < 0.32
        ? ["RRLyr", "EB*", "LP*", "V*"][r.int(0, 3)]
        : archetype === "stochastic" && r() < 0.25
          ? "QSO"
          : null;
    const catsClass = { transient: [11, 12, 13], periodic: [21, 22], stochastic: [31], sparse: [11, 21, 31] }[archetype];
    const snapshotIdx = [...new Set([0, sources.length - 2, sources.length - 1].filter((k) => k >= 0))];
    const snapshots = snapshotIdx.map((k) => {
      const history = sources.slice(0, k + 1);
      const lc = {};
      for (const band of Object.keys(bandWeights)) {
        const pts = history.filter((s) => s.band === band).map((s) => ({ t: s.midpointMjdTai, f: s.psfFlux, e: s.psfFluxErr }));
        const feats = lcFeatures(pts, lcKeys);
        if (feats) lc[band] = feats;
      }
      const grow = 1 - Math.exp(-(k + 1) / 3);
      const snn = { transient: 0.35 + 0.55 * grow, periodic: 0.15, stochastic: 0.25, sparse: 0.4 }[archetype] + 0.12 * r.normal();
      const cats = { transient: 0.55 + 0.3 * grow, periodic: 0.6, stochastic: 0.5, sparse: 0.35 }[archetype] + 0.15 * r.normal();
      return {
        diaSourceId: sources[k].diaSourceId,
        midpointMjdTai: sources[k].midpointMjdTai,
        fink_science_version: "FIXTURE",
        pred: { is_sso: false, is_first: k === 0, is_cataloged: cataloged },
        clf: {
          snnSnVsOthers_score: k === 0 && r() < 0.3 ? null : round(Math.min(1, Math.max(0, snn)), 4),
          cats_class: catsClass[r.int(0, catsClass.length - 1)],
          cats_score: r() < 0.02 ? null : round(Math.min(1, Math.max(0, cats)), 4),
          earlySNIa_score: archetype === "transient" && history.length >= 3 ? round(Math.min(1, Math.max(0, 0.2 + 0.5 * r() * grow)), 4) : null
        },
        xm: { simbad_otype: otype },
        lc_features: lc
      };
    });

    // Unit-vector mean: safe across the RA = 0/360 seam and near the poles.
    const v = sources.reduce(
      (acc, x) => {
        const a = (x.ra * Math.PI) / 180;
        const b = (x.dec * Math.PI) / 180;
        return [acc[0] + Math.cos(b) * Math.cos(a), acc[1] + Math.cos(b) * Math.sin(a), acc[2] + Math.sin(b)];
      },
      [0, 0, 0]
    );
    const raMean = (Math.atan2(v[1], v[0]) * 180) / Math.PI;
    const decMean = (Math.atan2(v[2], Math.hypot(v[0], v[1])) * 180) / Math.PI;
    objects.push({ archetypeIndex: i, sources, snapshots, ra: ((raMean % 360) + 360) % 360, dec: decMean });
  }

  const records = objects.map((o, i) => ({
    kind: "fink.diaObject",
    id: `990${String(i + 1).padStart(15, "0")}`,
    ra: round(o.ra, 5),
    dec: round(o.dec, 5),
    entity_date: utcMjdToUtcDate(o.sources[0].tUtc),
    n_dia_sources: o.sources.length,
    first_midpoint_mjd_tai: o.sources[0].midpointMjdTai,
    last_midpoint_mjd_tai: o.sources.at(-1).midpointMjdTai,
    bands: [...new Set(o.sources.map((s) => s.band))].sort((a, b) => "ugrizy".indexOf(a) - "ugrizy".indexOf(b))
  }));

  const timeSemantics = {
    stored_field: "midpointMjdTai",
    format: "MJD",
    scale: "TAI",
    scale_label: "TAI",
    scale_basis: "Fink analysis_contract_v1: midpointMjdTai is the authoritative observation time, MJD on the TAI scale.",
    date_binning:
      "UTC date of the TAI time converted with a leap-second table, half-open [00:00, 24:00) UTC. Fixture conversion: TAI − UTC = 37 s (valid since 2017-01-01); a real adapter must use ERFA/Astropy.",
    entity_date_rule: "UTC date of the first delivered DiaSource of the DiaObject in this basis."
  };

  const countValues = {};
  for (const rec of records) countValues[rec.entity_date] = (countValues[rec.entity_date] ?? 0) + 1;
  const m1Window = windows.find((w) => w.source_state === "DELIVERY_VALIDATED" && w.state === "AVAILABLE");
  for (const d of m1Dates) if (!countValues[d]) fail(`fixture produced no DiaObjects on Month-1 date ${d}`);
  const time = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    semantics: timeSemantics,
    range,
    nights,
    windows,
    counts: {
      quantity: "Synthetic fixture DiaObjects by first-delivered-DiaSource UTC date (Month-1 window only)",
      unit: "DiaObjects",
      evidence: ["SYNTHETIC_FIXTURE"],
      values: Object.fromEntries(m1Dates.filter((d) => d >= m1Window.start && d < m1Window.stop).map((d) => [d, countValues[d]]))
    }
  };

  const sky = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    density: densityMap(records, "Synthetic fixture DiaObjects", "DiaObjects", ["SYNTHETIC_FIXTURE"]),
    coverage: {
      max_order: COVERAGE_ORDER,
      ordering: "NESTED",
      frame: "ICRS",
      meaning:
        `Synthetic fixture footprint (${FIXTURE_FOOTPRINT_DEC[0]}° ≤ Dec ≤ +${FIXTURE_FOOTPRINT_DEC[1]}° cell centers) used to generate the fixture population. ` +
        "NOT the Rubin footprint and not Fink delivery coverage.",
      evidence: ["SYNTHETIC_FIXTURE"],
      moc: normalizeMoc(coverage, COVERAGE_ORDER)
    },
    coverage_unavailable_reason: null
  };

  const fields = [
    { key: "diaObjectId", label: "diaObjectId", unit: null, evidence: "SYNTHETIC_FIXTURE", description: "Derived DIA grouping key (pred.is_sso false AND diaObjectId > 0); int64 carried as a decimal string. Fixture ids use the reserved 990… prefix." },
    { key: "ra", label: "RA (ICRS)", unit: "deg", evidence: "SYNTHETIC_FIXTURE", description: "Unit-vector mean of delivered DiaSource positions." },
    { key: "dec", label: "Dec (ICRS)", unit: "deg", evidence: "SYNTHETIC_FIXTURE", description: "Unit-vector mean of delivered DiaSource positions." },
    { key: "n_dia_sources", label: "Delivered DiaSources", unit: "rows", evidence: "SYNTHETIC_FIXTURE", description: "Delivered DiaSource rows in this basis; not lifetime detections." },
    { key: "first_midpoint_mjd_tai", label: "First midpointMjdTai", unit: "MJD (TAI)", evidence: "SYNTHETIC_FIXTURE", description: "Earliest delivered DiaSource midpoint, TAI." },
    { key: "last_midpoint_mjd_tai", label: "Last midpointMjdTai", unit: "MJD (TAI)", evidence: "SYNTHETIC_FIXTURE", description: "Latest delivered DiaSource midpoint, TAI." },
    { key: "bands", label: "Bands", unit: null, evidence: "SYNTHETIC_FIXTURE", description: "LSST bands among delivered DiaSources." }
  ];
  const entities = {
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    entity_kind: "fink.diaObject",
    native_label: "Rubin DiaObject (Fink-delivered DIA grouping)",
    id_field: "diaObjectId",
    population: {
      complete: true,
      represented: records.length,
      total: records.length,
      sampling: `Complete synthetic fixture population (${records.length} DiaObjects, seed "${SEED}"). Not drawn from any Fink delivery.`
    },
    fields,
    records,
    detail: { shard_count: DETAIL_SHARDS.fink, path_template: "detail/{shard}.json", shard_rule: "fnv1a32(id) mod shard_count" }
  };

  /* --- feature registry and columns --- */
  const latestSnapshot = (o) => o.snapshots.at(-1);
  const posMag = (o, band) => {
    const vals = o.sources.filter((s) => s.band === band && s.psfFlux > 0).map((s) => 31.4 - 2.5 * Math.log10(s.psfFlux));
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null;
  };
  const lcDim = (key, label, short, scale, unit, definitionExtra) =>
    dim({
      id: `fink.lc_features.r.${key}`,
      domain: "fink",
      family: "variability",
      label: `${label} (r)`,
      short,
      unit,
      scale,
      definition: `lc_features["r"].${key} from the latest delivered source-time snapshot in this basis.${definitionExtra ? ` ${definitionExtra}` : ""}`,
      evidence: "SYNTHETIC_FIXTURE",
      snapshot: "Latest delivered snapshot; features summarize the history visible to Fink at that alert time, not a timeless object property.",
      qualifications: ["Requires ≥ 3 r-band DiaSources in the snapshot history; otherwise undefined."]
    });
  const clfDim = (key, label) =>
    dim({
      id: `fink.clf.${key}`,
      domain: "fink",
      family: "model_output",
      label,
      short: key,
      unit: "score",
      extent: [0, 1],
      definition: `clf.${key} from the latest delivered source-time snapshot in this basis.`,
      evidence: "SYNTHETIC_FIXTURE",
      snapshot: "Latest delivered snapshot; classifier outputs change as alerts arrive and are not timeless classifications.",
      qualifications: ["Model score, not a calibrated probability."]
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
      definition: "Count of delivered DiaSource rows grouped by diaObjectId within the admitted window.",
      evidence: "SYNTHETIC_FIXTURE",
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
      evidence: "SYNTHETIC_FIXTURE",
      qualifications: ["Bounded by the admitted window; not the object's full baseline."]
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
      evidence: "SYNTHETIC_FIXTURE"
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
      evidence: "SYNTHETIC_FIXTURE",
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
      evidence: "SYNTHETIC_FIXTURE",
      qualifications: ["Non-contemporaneous bands.", "Difference-image fluxes; not a source colour.", "Requires positive flux in both bands."]
    }),
    lcDim("chi2", "Reduced χ² about weighted mean", "χ²(r)", "log", null),
    lcDim("skew", "Skewness", "skew(r)", "linear", null),
    lcDim("stetson_K", "Stetson K", "K(r)", "linear", null),
    lcDim("amplitude", "Half range of psfFlux", "amp(r)", "log", "nJy"),
    lcDim("linear_fit_reduced_chi2", "Linear-fit reduced χ²", "χ²_lin(r)", "log", null),
    clfDim("snnSnVsOthers_score", "SuperNNova SN-vs-others score"),
    clfDim("cats_score", "CATS score"),
    clfDim("earlySNIa_score", "Early SN Ia score"),
    ...positionDims("fink", "SYNTHETIC_FIXTURE"),
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
    "fink.time_baseline_days": objects.map((o) => round(o.sources.at(-1).tUtc - o.sources[0].tUtc, 4)),
    "fink.max_snr": objects.map((o) => round(Math.max(...o.sources.map((s) => s.snr)), 2)),
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
  for (const key of ["chi2", "skew", "stetson_K", "amplitude", "linear_fit_reduced_chi2"]) {
    columns[`fink.lc_features.r.${key}`] = objects.map((o) => latestSnapshot(o).lc_features.r?.[key] ?? null);
  }
  for (const key of ["snnSnVsOthers_score", "cats_score", "earlySNIa_score"]) {
    columns[`fink.clf.${key}`] = objects.map((o) => latestSnapshot(o).clf[key] ?? null);
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
    lab_defaults: { x: "fink.time_baseline_days", y: "fink.lc_features.r.chi2" },
    columns
  };

  const shards = Array.from({ length: DETAIL_SHARDS.fink }, (_, shard) => ({
    contract: CONTRACT,
    domain: "fink",
    build_id: buildId,
    shard,
    evidence: ["SYNTHETIC_FIXTURE"],
    records: {}
  }));
  objects.forEach((o, i) => {
    const id = records[i].id;
    shards[shardOf(id, DETAIL_SHARDS.fink)].records[id] = {
      kind: "fink.diaObject",
      id,
      sources: o.sources.map((s) => ({
        diaSourceId: s.diaSourceId,
        midpointMjdTai: s.midpointMjdTai,
        band: s.band,
        psfFlux: s.psfFlux,
        psfFluxErr: s.psfFluxErr,
        snr: s.snr,
        reliability: s.reliability
      })),
      snapshots: o.snapshots,
      snapshot_policy:
        "First and last two delivered source-time snapshots per DiaObject (mirrors the G4A sampling). The fixture computes a subset of the pinned lc_features keys from synthetic psfFlux; absent keys were not computed."
    };
  });

  const pin = {
    domain: "fink",
    build_id: buildId,
    build_kind: "SYNTHETIC_FIXTURE_WITH_COMMITTED_EVIDENCE",
    label: "Fink · committed acquisition evidence + synthetic fixture population",
    evidence: ["VALIDATED_TRANSPORT_EVIDENCE", "COMMITTED_OPERATIONAL_RECORD", "SYNTHETIC_FIXTURE"],
    science_ready: false,
    native_ontology:
      "Rubin DiaSource rows delivered by Fink (Light Static packet). DiaObject is a derived grouping key; broker fields are source-time snapshots.",
    source: { repository: ev.repository, revision: ev.revision, paths: ev.extracted_files.map((f) => f.path) },
    time: timeSemantics
  };

  const acquisitions = ev.acquisitions.map((a) => {
    const w = windows.find((x) => x.start === a.window.start);
    return {
      domain: "fink",
      acquisition_id: a.acquisition_id,
      label: a.label,
      window: a.window,
      science_profile: a.science_profile,
      state: a.state,
      state_at_utc: a.state_at_utc,
      delivery: a.delivery
        ? {
            topic: a.delivery.topic,
            readable_rows: a.delivery.readable_rows,
            parquet_files: a.delivery.parquet_files,
            total_bytes: a.delivery.total_bytes,
            reconciliation_passed: a.delivery.reconciliation_passed,
            terminal_lag: a.delivery.terminal_lag,
            meaning: a.delivery.meaning
          }
        : null,
      characterization: a.characterization,
      delivery_validation: w.delivery_validation,
      admission: w.admission,
      scientific_status: a.characterization ? "CHARACTERIZED" : a.delivery ? "UNCHARACTERIZED" : "NOT_DELIVERY_VALIDATED",
      cohort: ev.cohort.acquisitions.includes(a.acquisition_id) ? ev.cohort.cohort_name : null
    };
  });
  const fieldEvidence = [
    ...fields.map((f) => ({ field: f.key, evidence: f.evidence, note: f.description })),
    { field: "acquisition windows", evidence: "COMMITTED_OPERATIONAL_RECORD", note: "Requests and state logs at the pinned Fink revision." },
    { field: "delivery totals", evidence: "VALIDATED_TRANSPORT_EVIDENCE", note: "Three-way reconciliation; transport evidence only." }
  ];
  return { pin, time, sky, entities, features, shards, windows, acquisitions, fieldEvidence, buildId, ev };
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
    "Per-date states from committed acquisition windows",
    "Each date inherits the state of its acquisition window at the pinned Fink revision.",
    { qualifications: ["Per-date delivered counts (G4A daily coverage) are data-plane artifacts and are not in this basis."] }),
  cap("fink", "time", "date_counts", "PARTIALLY_QUALIFIED", ["SYNTHETIC_FIXTURE"], "Fixture DiaObjects per date (Month-1 only)",
    "Synthetic fixture counts inside the admitted Month-1 window; no counts exist for Month-2 or Month-3 by design.", { codes: ["SYNTHETIC_FIXTURE"] }),
  cap("fink", "time", "transport_totals", "AVAILABLE", ["VALIDATED_TRANSPORT_EVIDENCE"], "Delivered row totals per acquisition",
    "Three-way reconciled transport totals per validated delivery.",
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
  cap("fink", "sky", "density", "AVAILABLE", ["SYNTHETIC_FIXTURE"], "DiaObject density on HEALPix", "Counts of fixture DiaObjects per equal-area cell."),
  cap("fink", "sky", "coverage", "PARTIALLY_QUALIFIED", ["SYNTHETIC_FIXTURE"], "Fixture footprint", fink.sky.coverage.meaning, { codes: ["SYNTHETIC_FIXTURE"] }),
  cap("fink", "sky", "filtered_density", "AVAILABLE", ["SYNTHETIC_FIXTURE"], "Cross-filtered density", "The entity table is the complete population of this basis.")
);
// Entities
capabilities.push(
  cap("antares", "entity", "inspector", "AVAILABLE", ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"], "Native ANTARES locus record", "Supported locus fields with per-field evidence."),
  cap("antares", "entity", "lightcurve", "PARTIALLY_QUALIFIED", ["SYNTHETIC_DEMO"], "Brightness history",
    "Only synthetic demo stories exist (10 loci); they are not ANTARES alert photometry.", { codes: ["SYNTHETIC_DEMO"] }),
  cap("antares", "entity", "broker_inference", "PARTIALLY_QUALIFIED", ["LEGACY_SAMPLE"], "ANTARES tags",
    "Filter-pipeline memberships only; not astrophysical classes and not scores.", { codes: ["TAGS_ARE_NOT_CLASSES"] }),
  cap("fink", "entity", "inspector", "AVAILABLE", ["SYNTHETIC_FIXTURE"], "Native Fink DiaObject / DiaSource record", "Delivered DiaSources and source-time broker snapshots."),
  cap("fink", "entity", "lightcurve", "AVAILABLE", ["SYNTHETIC_FIXTURE"], "Delivered psfFlux history",
    "Difference-image psfFlux of delivered DiaSources per band.", { qualifications: ["No forced photometry or upper limits (Light Static)."] }),
  cap("fink", "entity", "broker_inference", "AVAILABLE", ["SYNTHETIC_FIXTURE"], "Source-time classifier snapshots",
    "clf/pred/xm/lc_features as delivered with each DiaSource.", { qualifications: ["Model scores are not calibrated probabilities.", "Snapshots are not timeless classifications."] })
);
// Features
for (const d of ["antares", "fink"]) {
  const dims = both[d].features.dimensions;
  const usable = dims.filter((x) => x.state !== "UNAVAILABLE").length;
  capabilities.push(
    cap(d, "features", "population", "AVAILABLE", d === "antares" ? ["LEGACY_SAMPLE", "SYNTHETIC_DEMO"] : ["SYNTHETIC_FIXTURE"],
      "Parameter-space population", `${usable} of ${dims.length} registered dimensions are selectable; the rest are declared unavailable with reasons.`)
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
  label: "First Light · fixture basis",
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
    "No cross-broker identity matching is performed in this basis."
  ]
};

const provenance = {
  contract: CONTRACT,
  basis_id: BASIS_ID,
  sources: [
    { id: "antares.sky_points", label: "SkyPulse demo sky points", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.skyPoints}`, sha256: inputs.skyPoints.sha256, role: "ANTARES legacy demo loci", evidence: "LEGACY_SAMPLE" },
    { id: "antares.lightcurve_samples", label: "SkyPulse demo lightcurves", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.lightcurves}`, sha256: inputs.lightcurves.sha256, role: "Synthetic demo brightness stories", evidence: "SYNTHETIC_DEMO" },
    { id: "antares.public_manifest", label: "SkyPulse demo manifest", repository: "darim1151/ANTARES_Analysis", revision: BASELINE_REVISION, path: `web/${inputPaths.demoManifest}`, sha256: inputs.demoManifest.sha256, role: "Demo export identity", evidence: "SYNTHETIC_DEMO" },
    { id: "fink.evidence_excerpt", label: "Fink evidence excerpt", repository: "darim1151/ANTARES_Analysis", revision: null, path: `web/${inputPaths.finkEvidence}`, sha256: inputs.finkEvidence.sha256, role: "Extracted at the pinned Fink revision", evidence: "COMMITTED_OPERATIONAL_RECORD" },
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
    { id: "fink.object_position", description: "DiaObject position = unit-vector mean of delivered DiaSource positions.", kernel: KERNEL_TAG },
    { id: "fink.fixture_time_scale", description: `Fixture UTC→TAI with a fixed ${TAI_MINUS_UTC_S} s offset (real adapters use ERFA/Astropy).`, kernel: KERNEL_TAG },
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

const asOf = [inputs.demoManifest.json.generated_at_utc, ...fink.ev.acquisitions.map((a) => a.state_at_utc)]
  .map((s) => s.replace(/\+00:00$/, "Z"))
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
    "Fixture and demo values are labelled per payload and per field and must never be presented as accepted science. " +
    "Only committed Fink acquisition evidence is real, and it is transport/operational evidence only.",
  generator: {
    ...GENERATOR,
    deterministic: true,
    seed: SEED,
    inputs: Object.values(inputs).map((i) => ({ path: `web/${i.relative}`, sha256: i.sha256, role: i === inputs.finkEvidence ? "fink-evidence" : "antares-legacy-demo" }))
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
      `Fink ${fink.entities.records.length} fixture DiaObjects.`
  );
}
