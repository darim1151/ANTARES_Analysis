#!/usr/bin/env node
// Validate the Unified Scientific Observatory bundle as one coherent,
// integrity-sealed scientific read product.
//
//   node scripts/observatory/validate-observatory-bundle.mjs [--bundle-dir=<dir>]
//
// Passing is a contract-integrity check. It is never scientific approval: in
// this gate only FIRST_LIGHT_FIXTURE bundles are admissible.

import { createHash } from "node:crypto";
import { lstat, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { mocCellsAtOrder, npix, radecToPix } from "../../lib/observatory/kernel/healpix.ts";
import { isUtcDate, utcDateRange, utcMjdToUtcDate } from "../../lib/observatory/kernel/dates.ts";
import { shardOf, shardPath } from "../../lib/observatory/kernel/shard.ts";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(scriptDirectory, "..", "..");
let bundleDir = path.join(webRoot, "public", "observatory", "first-light");
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith("--bundle-dir=")) bundleDir = path.resolve(argument.slice("--bundle-dir=".length));
  else {
    console.error(`Unknown argument: ${argument}`);
    process.exit(2);
  }
}

const CONTRACT = "uso.observatory-bundle";
const DOMAINS = ["antares", "fink"];
const KIND = { antares: "antares.locus", fink: "fink.diaObject" };
const EVIDENCE = new Set([
  "ACCEPTED_SCIENCE",
  "VALIDATED_TRANSPORT_EVIDENCE",
  "COMMITTED_OPERATIONAL_RECORD",
  "LEGACY_SAMPLE",
  "SYNTHETIC_DEMO",
  "SYNTHETIC_FIXTURE"
]);
const NIGHT_STATES = new Set(["AVAILABLE", "ZERO", "UNQUALIFIED", "UNAVAILABLE", "MISSING", "OUTSIDE_COVERAGE"]);
const CAP_STATES = new Set(["AVAILABLE", "PARTIALLY_QUALIFIED", "UNAVAILABLE"]);
const FAMILIES = new Set(["position", "time", "multiplicity", "time_baseline", "photometry", "colour", "variability", "model_output", "crossmatch"]);
const BANDS = new Set(["u", "g", "r", "i", "z", "y"]);
const INT64_MAX = 9223372036854775807n;
const FIXTURE_TAI_MINUS_UTC_DAYS = 37 / 86400;
const FORBIDDEN_CLAIMS = ["rubin live feed", "official rubin result", "direct rubin catalog query", "real-time lsst stream", "classified transient"];
const REQUIRED_DOMAIN_CAPS = [
  "time.date_states",
  "time.date_counts",
  "sky.density",
  "sky.coverage",
  "sky.filtered_density",
  "entity.inspector",
  "entity.lightcurve",
  "entity.broker_inference",
  "features.population"
];

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

const errors = [];
const err = (where, message) => errors.push(`${where}: ${message}`);
const check = (condition, where, message) => {
  if (!condition) err(where, message);
  return Boolean(condition);
};
const isRecord = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const nonEmpty = (v) => typeof v === "string" && v.trim().length > 0;
const sha256 = (b) => createHash("sha256").update(b).digest("hex");

function finish() {
  if (errors.length) {
    console.error("Observatory bundle validation failed:");
    for (const e of errors) console.error(`- ${e}`);
    process.exit(1);
  }
}

function evidenceList(value, where, { allowEmpty = false } = {}) {
  if (!check(Array.isArray(value), where, "evidence must be an array")) return [];
  if (!allowEmpty) check(value.length > 0, where, "evidence must not be empty");
  for (const e of value) check(EVIDENCE.has(e), where, `unknown evidence class ${e}`);
  return value;
}

/* ------------------------------------------------------------- integrity */

async function listTree(root, prefix = "") {
  const out = [];
  for (const entry of await readdir(path.join(root, prefix), { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
    const st = await lstat(path.join(root, rel));
    if (st.isSymbolicLink()) err(rel, "symlinks are not allowed in the bundle");
    else if (st.isDirectory()) out.push(...(await listTree(root, rel)));
    else out.push(rel);
  }
  return out;
}

let manifestText;
try {
  manifestText = await readFile(path.join(bundleDir, "manifest.json"), "utf8");
} catch (error) {
  err("manifest.json", `cannot be read (${error.message})`);
  finish();
}
const manifest = JSON.parse(manifestText);
const M = "manifest.json";
check(manifest.contract === CONTRACT, M, `contract must be ${CONTRACT}`);
check(/^1\.\d+\.\d+$/.test(manifest.contract_version ?? ""), M, "contract_version must be 1.x.y");
check(nonEmpty(manifest.bundle_id), M, "bundle_id required");
check(manifest.bundle_class === "FIRST_LIGHT_FIXTURE", M, "only FIRST_LIGHT_FIXTURE bundles are admissible in this gate; QUALIFIED requires Control adjudication");
check(manifest.science_ready === false, M, "a FIRST_LIGHT_FIXTURE bundle must declare science_ready=false");
check(/Z$/.test(manifest.as_of_utc ?? "") && Number.isFinite(Date.parse(manifest.as_of_utc)), M, "as_of_utc must be an ISO UTC timestamp");
check(nonEmpty(manifest.evidence_policy), M, "evidence_policy required");
check(manifest.generator?.deterministic === true, M, "generator must be deterministic");

for (const input of manifest.generator?.inputs ?? []) {
  const where = `${M}.generator.inputs[${input.path}]`;
  if (!check(typeof input.path === "string" && input.path.startsWith("web/"), where, "input path must be repository-relative under web/")) continue;
  try {
    const bytes = await readFile(path.join(webRoot, input.path.slice("web/".length)));
    check(sha256(bytes) === input.sha256, where, "input changed since the bundle was generated; regenerate the bundle");
  } catch {
    err(where, "input file is missing");
  }
}

const listed = new Map();
for (const entry of manifest.files ?? []) {
  check(!listed.has(entry.path), M, `duplicate file entry ${entry.path}`);
  listed.set(entry.path, entry);
}
const onDisk = await listTree(bundleDir);
for (const rel of onDisk) if (rel !== "manifest.json" && !listed.has(rel)) err(rel, "unlisted file in bundle");
const docs = new Map();
for (const [rel, entry] of listed) {
  try {
    const bytes = await readFile(path.join(bundleDir, rel));
    check(bytes.length === entry.bytes, rel, "byte length differs from manifest");
    check(sha256(bytes) === entry.sha256, rel, "sha256 differs from manifest");
    docs.set(rel, JSON.parse(bytes.toString("utf8")));
  } catch (error) {
    err(rel, `cannot be read as JSON (${error.message})`);
  }
}
for (const ref of [manifest.basis, manifest.capabilities, manifest.provenance]) check(listed.has(ref), M, `referenced payload ${ref} is not listed`);
for (const d of DOMAINS) {
  for (const part of ["time", "sky", "entities", "features"]) {
    check(listed.has(manifest.domains?.[d]?.[part]), M, `domains.${d}.${part} must reference a listed payload`);
  }
}
finish();

/* ------------------------------------------------------- global scanners */

function scan(value, where) {
  if (typeof value === "number") {
    // Any number beyond 2^53 has already lost precision (int64 ids must be strings).
    if (Math.abs(value) > Number.MAX_SAFE_INTEGER) err(where, "number exceeds 2^53; int64 values must be decimal strings");
    return;
  }
  if (typeof value === "string") {
    if (/(?:^|[\s("'=:[])\/(?!\/)[A-Za-z0-9._-]+(?:\/[^\s"'<>\]]*)?/i.test(value) || /file:\/\//i.test(value) || /(?:^|[\s("'=])[a-z]:[\\/]/i.test(value)) {
      err(where, "contains a host-local filesystem path");
    }
    const lower = value.toLowerCase();
    for (const claim of FORBIDDEN_CLAIMS) if (lower.includes(claim)) err(where, `uses forbidden claim "${claim}"`);
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((v, i) => scan(v, `${where}[${i}]`));
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, v] of Object.entries(value)) {
    if (/(?:^|_)(?:password|passwd|secret|api[_-]?key|access[_-]?token|token|authorization|cookie|credential|private[_-]?key|session[_-]?id)(?:_|$)/i.test(key)) {
      err(`${where}.${key}`, "secret-bearing field name");
    }
    if (key === "evidence" && (v === "ACCEPTED_SCIENCE" || (Array.isArray(v) && v.includes("ACCEPTED_SCIENCE")))) {
      err(`${where}.${key}`, "ACCEPTED_SCIENCE is reserved for a Control-adjudicated basis and is forbidden here");
    }
    scan(v, `${where}.${key}`);
  }
}
scan(manifest, M);
for (const [rel, doc] of docs) {
  scan(doc, rel);
  check(doc.contract === CONTRACT, rel, `contract must be ${CONTRACT}`);
}

/* ------------------------------------------------------------------ basis */

const basis = docs.get(manifest.basis);
const B = manifest.basis;
check(nonEmpty(basis.basis_id), B, "basis_id required");
{
  // A basis id names exactly what it pins: views recorded against it cannot
  // silently replay against different builds or versions.
  const pins = {
    domains: Object.fromEntries(Object.entries(basis.domains ?? {}).map(([d, pin]) => [d, pin.build_id])),
    relation: basis.relation === null ? null : basis.relation,
    semantic_contract: basis.semantic_contract,
    feature_registry: basis.feature_registry,
    analysis_kernel: basis.analysis_kernel
  };
  const digest = sha256(canonical(pins)).slice(0, 12);
  check(typeof basis.basis_id === "string" && basis.basis_id.endsWith(`.${digest}`), B, `basis_id must end with the pin digest .${digest}`);
}
check(basis.status === "FIRST_LIGHT_FIXTURE" && basis.science_ready === false, B, "basis must be a FIRST_LIGHT_FIXTURE with science_ready=false");
check(basis.relation === null, B, "relation must be null: cross-broker association is not implemented in this gate");
check(JSON.stringify(Object.keys(basis.domains ?? {}).sort()) === JSON.stringify(DOMAINS), B, "basis must pin exactly the antares and fink domains");
for (const key of ["semantic_contract", "feature_registry", "analysis_kernel"]) {
  check(nonEmpty(basis[key]?.id) && nonEmpty(basis[key]?.version), B, `${key} must pin id and version`);
}
check(Array.isArray(basis.invariants) && basis.invariants.length >= 8, B, "scientific invariants must be declared");
for (const d of DOMAINS) {
  const pin = basis.domains[d];
  const where = `${B}.domains.${d}`;
  check(pin.domain === d, where, "domain mismatch");
  check(nonEmpty(pin.build_id), where, "build_id required");
  check(pin.science_ready === false, where, "domain build must not be science_ready in a fixture basis");
  evidenceList(pin.evidence, where);
  check(nonEmpty(pin.native_ontology), where, "native_ontology required");
  for (const f of ["stored_field", "scale_label", "scale_basis", "date_binning", "entity_date_rule"]) check(nonEmpty(pin.time?.[f]), where, `time.${f} required`);
}
check(basis.domains.fink.time.scale === "TAI" && basis.domains.fink.time.stored_field === "midpointMjdTai", B, "Fink time must be midpointMjdTai on the TAI scale");
check(basis.domains.antares.time.scale === "UTC", B, "ANTARES historical exporter time is declared UTC-treated");
const synthetic = (pin) => pin.evidence.some((e) => e === "SYNTHETIC_DEMO" || e === "SYNTHETIC_FIXTURE");
for (const d of DOMAINS) {
  check(synthetic(basis.domains[d]), `${B}.domains.${d}`, "First-Light domain builds must declare their synthetic evidence");
}

/* ----------------------------------------------------------- capabilities */

const capsDoc = docs.get(manifest.capabilities);
const C = manifest.capabilities;
check(capsDoc.basis_id === basis.basis_id, C, "basis_id mismatch");
const caps = new Map();
for (const c of capsDoc.capabilities ?? []) {
  const where = `${C}[${c.id}]`;
  check(!caps.has(c.id), where, "duplicate capability id");
  caps.set(c.id, c);
  check(c.id === `${c.scope}:${c.area}.${c.name}`, where, "id must be scope:area.name");
  check(CAP_STATES.has(c.state), where, `invalid state ${c.state}`);
  check(nonEmpty(c.summary) && nonEmpty(c.reason), where, "summary and reason required");
  check(Array.isArray(c.codes) && Array.isArray(c.qualifications), where, "codes and qualifications must be arrays");
  evidenceList(c.evidence, where, { allowEmpty: true });
  if (c.state === "UNAVAILABLE") check(c.evidence.length === 0, where, "an UNAVAILABLE capability carries no evidence");
  else if (DOMAINS.includes(c.scope)) check(c.evidence.length > 0, where, "a domain capability must state its evidence");
}
for (const d of DOMAINS) for (const id of REQUIRED_DOMAIN_CAPS) check(caps.has(`${d}:${id}`), C, `missing required capability ${d}:${id}`);
check(caps.get("relation:relation.cross_broker_association")?.state === "UNAVAILABLE", C, "cross-broker association must be declared UNAVAILABLE");
check(caps.get("workspace:compare.difference_map")?.state === "UNAVAILABLE", C, "raw count difference maps must be declared UNAVAILABLE");

/* ------------------------------------------------------------ per domain */

const excerptPath = path.join(scriptDirectory, "inputs", "fink-evidence.fd02c8e.json");
const excerpt = JSON.parse(await readFile(excerptPath, "utf8"));
const provenance = docs.get(manifest.provenance);
const P = manifest.provenance;
check(provenance.basis_id === basis.basis_id, P, "basis_id mismatch");

const entityIndex = {};
for (const d of DOMAINS) {
  const refs = manifest.domains[d];
  const time = docs.get(refs.time);
  const sky = docs.get(refs.sky);
  const ents = docs.get(refs.entities);
  const feats = docs.get(refs.features);
  const pin = basis.domains[d];
  for (const [rel, doc] of [[refs.time, time], [refs.sky, sky], [refs.entities, ents], [refs.features, feats]]) {
    check(doc.domain === d, rel, "domain mismatch");
    check(doc.build_id === pin.build_id, rel, "build_id must equal the basis pin");
  }

  /* time */
  const T = refs.time;
  check(JSON.stringify(time.semantics) === JSON.stringify(pin.time), T, "time semantics must equal the basis pin");
  const dates = isUtcDate(time.range?.start) && isUtcDate(time.range?.stop) && time.range.start < time.range.stop ? utcDateRange(time.range.start, time.range.stop) : [];
  check(dates.length > 0, T, "range must be a non-empty half-open UTC date interval");
  check(JSON.stringify((time.nights ?? []).map((n) => n.date)) === JSON.stringify(dates), T, "nights must enumerate every UTC date of the range in order");
  const windows = new Map((time.windows ?? []).map((w) => [w.id, w]));
  const sortedWindows = [...windows.values()].sort((a, b) => (a.start < b.start ? -1 : 1));
  for (let i = 0; i < sortedWindows.length; i += 1) {
    const w = sortedWindows[i];
    const where = `${T}.windows[${w.id}]`;
    check(isUtcDate(w.start) && isUtcDate(w.stop) && w.start < w.stop, where, "window must be a half-open UTC date interval");
    check(w.start >= time.range.start && w.stop <= time.range.stop, where, "window must lie inside the lane range");
    if (i > 0) check(sortedWindows[i - 1].stop <= w.start, where, "windows must not overlap");
    check(NIGHT_STATES.has(w.state) && w.state !== "OUTSIDE_COVERAGE" && w.state !== "ZERO", where, `invalid window state ${w.state}`);
    evidenceList(w.evidence, where);
    check(nonEmpty(w.caveat) && nonEmpty(w.source_state), where, "caveat and source_state required");
    check(Array.isArray(w.status_codes) && w.status_codes.length > 0, where, "status_codes required");
    const transport = (w.facts ?? []).some((f) => f.evidence === "VALIDATED_TRANSPORT_EVIDENCE");
    if (transport || w.state === "UNQUALIFIED") check(w.rate_comparison === "PROHIBITED", where, "transport totals and unqualified windows must prohibit rate comparison");
    if (w.state === "AVAILABLE") {
      check(w.admission === "ADMITTED" && w.delivery_validation !== "NOT_DELIVERY_VALIDATED", where, "an AVAILABLE window must be admitted and not unvalidated");
    }
    if (w.state === "UNQUALIFIED") {
      check(w.delivery_validation === "DELIVERY_VALIDATED", where, "UNQUALIFIED is reserved for delivery-validated but uncharacterized/unadmitted data");
      check(w.admission === "NOT_ADMITTED" || w.status_codes.includes("UNCHARACTERIZED"), where, "an UNQUALIFIED window must be unadmitted or uncharacterized");
    }
    if (w.state === "UNAVAILABLE") {
      check(
        w.delivery_validation === "NOT_DELIVERY_VALIDATED" && w.admission === "NOT_ADMITTED" && w.status_codes.includes("NOT_DELIVERY_VALIDATED") && w.status_codes.includes("NOT_ADMITTED"),
        where,
        "an UNAVAILABLE window must be NOT_DELIVERY_VALIDATED and NOT_ADMITTED"
      );
    }
    if (w.state === "MISSING") {
      check(w.status_codes.includes("DATA_ABSENT") || w.status_codes.includes("EVIDENCE_ABSENT"), where, "MISSING is reserved for genuinely absent data or evidence (DATA_ABSENT/EVIDENCE_ABSENT)");
    }
  }
  for (const n of time.nights ?? []) {
    const where = `${T}.nights[${n.date}]`;
    check(NIGHT_STATES.has(n.state), where, `invalid state ${n.state}`);
    const w = n.window_id ? windows.get(n.window_id) : null;
    if (n.state === "OUTSIDE_COVERAGE") {
      check(!w, where, "OUTSIDE_COVERAGE dates belong to no window");
    } else if (check(w, where, "date must reference a window")) {
      check(n.date >= w.start && n.date < w.stop, where, "date lies outside its window");
      check(n.state === w.state || (n.state === "ZERO" && w.state === "AVAILABLE"), where, "date state must inherit its window state");
    }
  }
  const nightState = new Map((time.nights ?? []).map((n) => [n.date, n.state]));
  if (time.counts) {
    evidenceList(time.counts.evidence, `${T}.counts`);
    for (const [date, value] of Object.entries(time.counts.values ?? {})) {
      const state = nightState.get(date);
      const where = `${T}.counts[${date}]`;
      check(state === "AVAILABLE" || state === "ZERO", where, `counts are only allowed on AVAILABLE/ZERO dates, not ${state}`);
      check(Number.isInteger(value) && value >= 0, where, "count must be a non-negative integer");
      check((value === 0) === (state === "ZERO"), where, "a zero count must be a ZERO date and vice versa");
    }
    for (const [date, state] of nightState) {
      if (state === "AVAILABLE" || state === "ZERO") check(date in time.counts.values, `${T}.counts`, `missing count for ${state} date ${date}`);
    }
    const capCounts = caps.get(`${d}:time.date_counts`);
    check(JSON.stringify(capCounts?.evidence) === JSON.stringify(time.counts.evidence), C, `${d}:time.date_counts evidence must match the count series`);
  }

  /* entities */
  const E = refs.entities;
  check(ents.entity_kind === KIND[d], E, `entity_kind must be ${KIND[d]}`);
  const records = ents.records ?? [];
  check(ents.population?.represented === records.length, E, "population.represented must equal the record count");
  if (ents.population?.complete) check(ents.population.total === records.length, E, "a complete population has total == represented");
  else {
    // Client-side cross-filtering of a sample would present sample counts as the population.
    check(caps.get(`${d}:sky.filtered_density`)?.state !== "AVAILABLE", C, `${d}:sky.filtered_density cannot be AVAILABLE for an incomplete population`);
  }
  const ids = new Set();
  const perDate = {};
  for (const [i, r] of records.entries()) {
    const where = `${E}.records[${i}]`;
    check(r.kind === KIND[d], where, "record kind mismatch");
    check(typeof r.id === "string" && !ids.has(r.id), where, "ids must be unique strings");
    ids.add(r.id);
    check(Number.isFinite(r.ra) && r.ra >= 0 && r.ra < 360 && Number.isFinite(r.dec) && r.dec >= -90 && r.dec <= 90, where, "ICRS coordinates out of range");
    const state = nightState.get(r.entity_date);
    check(state === "AVAILABLE" || state === "ZERO", where, `entity_date ${r.entity_date} must fall on an AVAILABLE date, not ${state}`);
    perDate[r.entity_date] = (perDate[r.entity_date] ?? 0) + 1;
    if (d === "fink") {
      // A JS number would silently lose int64 precision: only decimal strings are valid.
      const isDecimal = typeof r.id === "string" && /^[1-9]\d{0,18}$/.test(r.id);
      check(isDecimal && BigInt(r.id) <= INT64_MAX, where, "diaObjectId must be a positive int64 decimal string");
      if (isDecimal && pin.build_kind.startsWith("SYNTHETIC_FIXTURE")) {
        check(r.id.startsWith("990"), where, "fixture diaObjectIds must use the reserved 990 prefix");
      }
      check(r.first_midpoint_mjd_tai <= r.last_midpoint_mjd_tai && r.n_dia_sources >= 1, where, "time order / multiplicity invalid");
      check((r.bands ?? []).every((b) => BANDS.has(b)), where, "unknown band");
      if (pin.build_kind.startsWith("SYNTHETIC_FIXTURE")) {
        check(utcMjdToUtcDate(r.first_midpoint_mjd_tai - FIXTURE_TAI_MINUS_UTC_DAYS) === r.entity_date, where, "entity_date must be the UTC date of the first TAI time");
      }
    } else {
      check(typeof r.id === "string" && /^ANT\d{4}[a-z0-9]+$/.test(r.id), where, "ANTARES locus id format");
      check(Array.isArray(r.tags) && r.tags.every(nonEmpty), where, "tags must be strings");
      check(r.brightest_alert_magnitude === null || Number.isFinite(r.brightest_alert_magnitude), where, "magnitude must be finite or null");
      check(utcMjdToUtcDate(r.newest_alert_observation_time) === r.entity_date, where, "entity_date must be the UTC date of the UTC-treated MJD");
    }
  }
  if (ents.population?.complete && time.counts) {
    for (const [date, value] of Object.entries(time.counts.values)) check((perDate[date] ?? 0) === value, `${T}.counts[${date}]`, "count must equal the complete entity table");
  }
  entityIndex[d] = ids;

  /* sky */
  const S = refs.sky;
  const dens = sky.density ?? {};
  check(Number.isInteger(dens.order) && dens.order >= 0 && dens.order <= 13 && dens.ordering === "NESTED" && dens.frame === "ICRS", S, "density must be a NESTED ICRS HEALPix map, order 0..13");
  evidenceList(dens.evidence, `${S}.density`);
  check(Array.isArray(dens.pixels) && Array.isArray(dens.values) && dens.pixels.length === dens.values.length, S, "pixels/values must align");
  for (let i = 0; i < (dens.pixels ?? []).length; i += 1) {
    if (!check(Number.isInteger(dens.pixels[i]) && dens.pixels[i] >= 0 && dens.pixels[i] < npix(dens.order), S, `pixel ${dens.pixels[i]} out of range`)) break;
    if (i > 0 && !check(dens.pixels[i] > dens.pixels[i - 1], S, "pixels must be strictly increasing")) break;
    if (!check(Number.isInteger(dens.values[i]) && dens.values[i] > 0, S, "density values must be positive counts (absent cells are not zeros)")) break;
  }
  if (ents.population?.complete) {
    const expect = new Map();
    for (const r of records) {
      const p = radecToPix(dens.order, r.ra, r.dec);
      expect.set(p, (expect.get(p) ?? 0) + 1);
    }
    const ok = expect.size === dens.pixels.length && dens.pixels.every((p, i) => expect.get(p) === dens.values[i]);
    check(ok, S, "density must equal the HEALPix aggregation of the complete entity table");
  }
  const capCoverage = caps.get(`${d}:sky.coverage`);
  if (sky.coverage === null) {
    check(nonEmpty(sky.coverage_unavailable_reason), S, "absent coverage needs a reason");
    check(capCoverage?.state === "UNAVAILABLE", C, `${d}:sky.coverage must be UNAVAILABLE when coverage is absent`);
  } else {
    const cov = sky.coverage;
    check(cov.ordering === "NESTED" && cov.frame === "ICRS" && Number.isInteger(cov.max_order) && cov.max_order <= 13, S, "coverage must be a NESTED ICRS MOC");
    evidenceList(cov.evidence, `${S}.coverage`);
    check(nonEmpty(cov.meaning), S, "coverage meaning required");
    check(capCoverage?.state !== "UNAVAILABLE", C, `${d}:sky.coverage must not be UNAVAILABLE when coverage exists`);
    const seen = new Set();
    for (const [order, cells] of Object.entries(cov.moc ?? {})) {
      const o = Number(order);
      check(Number.isInteger(o) && o >= 0 && o <= cov.max_order, S, `invalid MOC order ${order}`);
      const siblings = new Map();
      for (const cell of cells) {
        check(Number.isInteger(cell) && cell >= 0 && cell < npix(o), S, `MOC cell ${cell} out of range at order ${o}`);
        for (let a = o; a >= 0; a -= 1) {
          const key = `${a}/${Math.floor(cell / 4 ** (o - a))}`;
          if (a < o) check(!seen.has(key), S, `MOC cell ${o}/${cell} duplicates ancestor ${key}`);
        }
        seen.add(`${o}/${cell}`);
        if (o > 0) siblings.set(Math.floor(cell / 4), (siblings.get(Math.floor(cell / 4)) ?? 0) + 1);
      }
      for (const [parent, n] of siblings) check(n < 4, S, `MOC not normalized: four children of ${o - 1}/${parent}`);
    }
    if (dens.order <= cov.max_order) {
      const covered = mocCellsAtOrder(cov.moc, dens.order);
      check(dens.pixels.every((p) => covered.has(p)), S, "density must lie inside the declared coverage");
    }
  }

  /* features */
  const F = refs.features;
  check(JSON.stringify(feats.registry) === JSON.stringify(basis.feature_registry), F, "registry pin mismatch");
  check(JSON.stringify(feats.kernel) === JSON.stringify(basis.analysis_kernel), F, "kernel pin mismatch");
  const dimIds = new Set();
  const selectable = new Set();
  for (const dm of feats.dimensions ?? []) {
    const where = `${F}.dimensions[${dm.id}]`;
    check(!dimIds.has(dm.id), where, "duplicate dimension");
    dimIds.add(dm.id);
    check(typeof dm.id === "string" && dm.id.startsWith(`${d}.`) && dm.domain === d, where, "dimension must be domain-qualified");
    check(FAMILIES.has(dm.family) && CAP_STATES.has(dm.state), where, "invalid family/state");
    check(nonEmpty(dm.label) && nonEmpty(dm.definition) && nonEmpty(dm.definition_id), where, "label, definition and definition_id required");
    const col = feats.columns?.[dm.id];
    if (dm.state === "UNAVAILABLE") {
      check(col === undefined, where, "an UNAVAILABLE dimension must not ship a column");
      check(nonEmpty(dm.unavailable_reason) && dm.evidence === null, where, "an UNAVAILABLE dimension needs a reason and no evidence");
      continue;
    }
    selectable.add(dm.id);
    check(EVIDENCE.has(dm.evidence), where, "selectable dimension needs an evidence class");
    if (!check(Array.isArray(col) && col.length === records.length, where, "column must align with entity records")) continue;
    for (const v of col) {
      if (v === null) continue;
      if (!check(Number.isFinite(v), where, "values must be finite or null")) break;
      if (dm.scale === "log" && !check(v > 0, where, "log-scale values must be positive or null")) break;
      if (dm.extent && !check(v >= dm.extent[0] && v <= dm.extent[1], where, "value outside declared extent")) break;
    }
    if (dm.family === "model_output") {
      check(dm.calibrated === false && dm.unit === "score" && nonEmpty(dm.snapshot), where, "model outputs are uncalibrated source-time snapshot scores");
      check(dm.qualifications.some((q) => /not a calibrated probability/i.test(q)), where, "model outputs must state they are not calibrated probabilities");
    }
    if (dm.id.includes("lc_features") || dm.id.includes(".clf.")) check(nonEmpty(dm.snapshot), where, "broker fields must declare snapshot semantics");
  }
  for (const key of Object.keys(feats.columns ?? {})) check(dimIds.has(key), F, `column ${key} has no registered dimension`);
  check(selectable.has(feats.lab_defaults?.x) && selectable.has(feats.lab_defaults?.y), F, "lab_defaults must be selectable dimensions");
  entityIndex[`${d}:dims`] = feats.dimensions ?? [];

  /* detail shards */
  const shardCount = ents.detail?.shard_count;
  check(Number.isInteger(shardCount) && shardCount > 0 && ents.detail.path_template.includes("{shard}"), E, "detail sharding invalid");
  const seenDetail = new Set();
  const allSourceIds = new Set();
  const summaryById = new Map(records.map((r) => [r.id, r]));
  for (let shard = 0; shard < shardCount; shard += 1) {
    const rel = `domains/${d}/${shardPath(ents.detail.path_template, shard)}`;
    const doc = docs.get(rel);
    if (!check(doc, rel, "detail shard missing from manifest")) continue;
    check(doc.shard === shard && doc.domain === d && doc.build_id === pin.build_id, rel, "shard identity mismatch");
    evidenceList(doc.evidence, rel);
    for (const [id, detail] of Object.entries(doc.records ?? {})) {
      const where = `${rel}[${id}]`;
      check(shardOf(id, shardCount) === shard, where, "record routed to the wrong shard");
      if (!check(summaryById.has(id) && !seenDetail.has(id), where, "detail must map one-to-one onto entity records")) continue;
      seenDetail.add(id);
      if (!check(isRecord(detail) && detail.kind === KIND[d] && detail.id === id, where, "detail kind/id mismatch")) continue;
      if (d === "antares") {
        if (detail.lightcurve === null) check(nonEmpty(detail.lightcurve_unavailable_reason), where, "absent lightcurve needs a reason");
        else check(EVIDENCE.has(detail.lightcurve.evidence) && detail.lightcurve.time_scale === "UTC", where, "lightcurve evidence/time scale");
      } else {
        const s = summaryById.get(id);
        const sources = Array.isArray(detail.sources) ? detail.sources : [];
        check(sources.length === s.n_dia_sources, where, "source count must equal n_dia_sources");
        for (let i = 0; i < sources.length; i += 1) {
          const src = sources[i];
          check(typeof src.diaSourceId === "string" && /^[1-9]\d{0,18}$/.test(src.diaSourceId) && !allSourceIds.has(src.diaSourceId), where, "diaSourceId must be a unique int64 decimal string");
          allSourceIds.add(src.diaSourceId);
          check(BANDS.has(src.band) && src.psfFluxErr > 0 && Number.isFinite(src.psfFlux), where, "invalid DiaSource photometry");
          if (i > 0) check(src.midpointMjdTai >= sources[i - 1].midpointMjdTai, where, "sources must be time ordered");
          const utcDate = utcMjdToUtcDate(src.midpointMjdTai - FIXTURE_TAI_MINUS_UTC_DAYS);
          check(nightState.get(utcDate) === "AVAILABLE", where, `DiaSource on ${utcDate} lies outside the admitted window (${nightState.get(utcDate)})`);
        }
        if (sources.length) {
          check(sources[0].midpointMjdTai === s.first_midpoint_mjd_tai && sources.at(-1).midpointMjdTai === s.last_midpoint_mjd_tai, where, "summary times must match sources");
        }
        const sourceIds = new Set(sources.map((x) => x.diaSourceId));
        for (const snap of detail.snapshots ?? []) {
          check(sourceIds.has(snap.diaSourceId), where, "snapshot must reference a delivered DiaSource");
          check(snap.pred?.is_sso === false, where, "DIA objects require pred.is_sso == false");
          for (const key of Object.keys(snap.clf ?? {})) check(excerpt.light_static_schema.clf_fields.includes(key), where, `clf.${key} is not in the pinned schema`);
          for (const [band, featsMap] of Object.entries(snap.lc_features ?? {})) {
            check(BANDS.has(band), where, `lc_features band ${band}`);
            for (const key of Object.keys(featsMap)) check(excerpt.light_static_schema.lc_features_fields.includes(key), where, `lc_features.${key} is not in the pinned schema`);
          }
        }
        check(nonEmpty(detail.snapshot_policy), where, "snapshot_policy required");
      }
    }
  }
  check(seenDetail.size === records.length, E, "every entity needs exactly one detail record");
}

/* shared definitions across domains must be identical */
const byDefinition = new Map();
for (const d of DOMAINS) {
  for (const dm of entityIndex[`${d}:dims`]) {
    const prior = byDefinition.get(dm.definition_id);
    if (prior) check(prior.definition === dm.definition && prior.unit === dm.unit, `${dm.id}`, `shares definition_id with ${prior.id} but not its definition`);
    else byDefinition.set(dm.definition_id, dm);
  }
}

/* --------------------------------------------- Fink windows vs evidence */

const finkTime = docs.get(manifest.domains.fink.time);
for (const a of excerpt.acquisitions) {
  const w = finkTime.windows.find((x) => x.start === a.window.start && x.stop === a.window.stop);
  const where = `${manifest.domains.fink.time}[${a.label}]`;
  if (!check(w, where, "every pinned acquisition must appear as a window")) continue;
  const validated = a.state === "DELIVERY_VALIDATED" && a.delivery?.reconciliation_passed === true;
  const admitted = excerpt.cohort.acquisitions.includes(a.acquisition_id);
  const expected = validated && admitted && a.characterization ? "AVAILABLE" : validated ? "UNQUALIFIED" : "UNAVAILABLE";
  check(w.state === expected, where, `window state must be ${expected} for upstream state ${a.state} (admitted=${admitted})`);
  check(w.source_state === a.state, where, "source_state must equal the pinned acquisition state");
  const capability = caps.get(`fink:time.acquisition.${w.id.split(".")[1]}`);
  const capState = { AVAILABLE: "AVAILABLE", UNQUALIFIED: "PARTIALLY_QUALIFIED", UNAVAILABLE: "UNAVAILABLE" }[w.state];
  check(capability?.state === capState, C, `fink acquisition capability for ${a.label} must be ${capState}`);
  check(JSON.stringify(capability?.codes) === JSON.stringify(w.status_codes), C, `fink acquisition capability codes for ${a.label} must equal window status codes`);
  const prov = provenance.acquisitions.find((x) => x.acquisition_id === a.acquisition_id);
  if (prov) check(prov.domain === "fink", P, `${a.label} acquisition evidence must declare its domain (fink)`);
  if (check(prov, P, `acquisition ${a.acquisition_id} missing from provenance`)) {
    check(prov.state === a.state && prov.admission === w.admission && prov.delivery_validation === w.delivery_validation, P, `${a.label} provenance disagrees with its window`);
    check((prov.delivery?.readable_rows ?? null) === (a.delivery?.readable_rows ?? null), P, `${a.label} delivered rows disagree with pinned evidence`);
  }
}
check(provenance.sources.every((s) => s.sha256 === null || /^[0-9a-f]{64}$/.test(s.sha256)), P, "source digests must be sha256 hex");
for (const f of excerpt.extracted_files) {
  check(provenance.sources.some((s) => s.path === f.path && s.sha256 === f.sha256 && s.revision === excerpt.revision), P, `pinned Fink file ${f.path} must be cited with its digest`);
}

finish();
const counts = DOMAINS.map((d) => `${d} ${entityIndex[d].size}`).join(", ");
console.log(`Validated Observatory bundle ${manifest.bundle_id} (${listed.size + 1} files, ${counts}; class ${manifest.bundle_class}, not science-ready).`);
