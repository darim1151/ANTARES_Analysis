#!/usr/bin/env node
// Extract the committed, non-secret Fink acquisition evidence that the
// Observatory First-Light basis pins. READ ONLY with respect to the Fink
// repository: it reads files at the pinned revision and writes only the
// excerpt inside this repository.
//
//   node scripts/observatory/extract-fink-evidence.mjs --fink-root=<checkout> [--check]
//
// The checkout must be exactly at PINNED_REVISION. With --check the excerpt is
// regenerated in memory and compared byte-for-byte with the committed file.

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PINNED_REPOSITORY = "darim1151/Fink_Alerts-Analysis-LSST";
const PINNED_REVISION = "fd02c8eabcaf3a1160e0ed2c5c1d6959ac11d23d";
const SCHEMA_TOPIC = "ftransfer_lsst_2026-10-04_177446";
const ACQUISITIONS = [
  { id: "acq_lsst_ls_v1_2026-02-25_to_2026-03-25_b1c7b482b56b", label: "Month-1" },
  { id: "acq_lsst_ls_v1_2026-03-25_to_2026-04-25_0c852cd059fa", label: "Month-2" },
  { id: "acq_lsst_ls_v1_2026-04-25_to_2026-05-25_1ccf601aa29f", label: "Month-3" }
];

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const outputPath = path.join(scriptDirectory, "inputs", `fink-evidence.${PINNED_REVISION.slice(0, 7)}.json`);

let finkRoot = null;
let check = false;
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith("--fink-root=")) finkRoot = path.resolve(argument.slice("--fink-root=".length));
  else if (argument === "--check") check = true;
  else {
    console.error(`Unknown argument: ${argument}`);
    process.exit(2);
  }
}
if (!finkRoot) {
  console.error("--fink-root=<checkout of the pinned Fink revision> is required");
  process.exit(2);
}

const head = execFileSync("git", ["-C", finkRoot, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
if (head !== PINNED_REVISION) {
  console.error(`Fink checkout is at ${head}; the Observatory basis pins ${PINNED_REVISION}. Refusing.`);
  process.exit(1);
}
const dirty = execFileSync("git", ["-C", finkRoot, "status", "--porcelain"], { encoding: "utf8" }).trim();
if (dirty) {
  console.error("Fink checkout has local modifications; evidence must come from the pinned commit only.");
  process.exit(1);
}

const extracted = [];
async function readEvidence(relative, { json = true } = {}) {
  const bytes = await readFile(path.join(finkRoot, relative));
  extracted.push({ path: relative, sha256: createHash("sha256").update(bytes).digest("hex") });
  const text = bytes.toString("utf8");
  return json ? JSON.parse(text) : text;
}

function require(condition, message) {
  if (!condition) throw new Error(`Evidence assertion failed: ${message}`);
}

const cohort = await readEvidence("configs/analysis_cohorts/month1.json");
const characterizations = new Map(cohort.acquisitions.map((a) => [a.acquisition_id, a.characterization]));

const acquisitions = [];
for (const { id, label } of ACQUISITIONS) {
  const base = `configs/acquisitions/${id}`;
  const request = await readEvidence(`${base}/request.json`);
  const stateLog = (await readEvidence(`${base}/state_log.jsonl`, { json: false }))
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const last = stateLog[stateLog.length - 1];
  let delivery = null;
  try {
    const d = await readEvidence(`${base}/evidence/delivery_1.json`);
    require(d.acquisition_id === id && d.fingerprint === request.fingerprint, `${id} delivery identity`);
    const r = d.reconciliation;
    require(
      r.expected_topic_messages === r.terminal_committed && r.terminal_committed === r.local_readable_rows,
      `${id} three-way reconciliation`
    );
    delivery = {
      topic: d.topic,
      readable_rows: d.readable_rows,
      parquet_files: d.parquet_files,
      readable_parquet_files: d.readable_parquet_files,
      unreadable_files: d.unreadable_files,
      total_bytes: d.total_bytes,
      schema_groups: Object.keys(d.schema_groups),
      reconciliation_passed: r.passed === true,
      terminal_lag: r.terminal_lag,
      meaning: r.meaning
    };
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const window = request.scientific_identity.window;
  require(window.semantics === "half_open_utc_dates", `${id} window semantics`);
  acquisitions.push({
    acquisition_id: id,
    label,
    fingerprint: request.fingerprint,
    science_profile: request.scientific_identity.science_profile,
    survey: request.scientific_identity.survey,
    window: { start: window.start, stop: window.stop, semantics: window.semantics },
    portal_dates_inclusive: request.portal_dates_inclusive,
    expected_dates: request.expected_dates.length,
    state: last.to,
    state_at_utc: last.at_utc,
    state_sequence: stateLog.map((entry) => entry.to),
    delivery,
    characterization: characterizations.get(id) ?? null
  });
}

const contractSource = await readEvidence("src/fink_lsst/analytics/contract.py", { json: false });
for (const needle of [
  "CONTRACT_ID = 'analysis_contract_v1'",
  "'midpointMjdTai': ['double']",
  "time=dict(authoritative_field='midpointMjdTai', sql_name='observation_mjd_tai', scale='TAI'",
  "'diaObjectId': ['int64']",
  "DIA='pred.is_sso false AND diaObjectId > 0'",
  "known_absent_capabilities=['complete historical detections', 'forced photometry / upper limits',",
  "'canonical DiaObject record', 'SSO identity / orbits']"
]) {
  require(contractSource.includes(needle), `analysis contract contains ${needle}`);
}

const avro = await readEvidence(`configs/delivery_evidence/${SCHEMA_TOPIC}/avro_schema_${SCHEMA_TOPIC}.json`);
const deliveryEvidence = await readEvidence(`configs/delivery_evidence/${SCHEMA_TOPIC}/delivery_evidence.json`);
function recordFields(name) {
  const field = avro.fields.find((f) => f.name === name);
  const type = Array.isArray(field.type) ? field.type.find((t) => typeof t === "object") : field.type;
  return type.fields.map((f) => f.name);
}
const lcField = avro.fields.find((f) => f.name === "lc_features");
const lcMap = lcField.type.find((t) => typeof t === "object");
const lcRecord = lcMap.values.find((t) => typeof t === "object");

const excerpt = {
  kind: "uso.fink-evidence-excerpt",
  version: 1,
  repository: PINNED_REPOSITORY,
  revision: PINNED_REVISION,
  extractor: "web/scripts/observatory/extract-fink-evidence.mjs",
  statement:
    "Committed, non-secret acquisition and schema evidence read at the pinned revision. " +
    "Delivery reconciliation is transport evidence, not Rubin scientific completeness.",
  cohort: { cohort_name: cohort.cohort_name, contract_id: cohort.contract_id, acquisitions: cohort.acquisitions.map((a) => a.acquisition_id) },
  acquisitions,
  analysis_contract: {
    contract_id: "analysis_contract_v1",
    time: { authoritative_field: "midpointMjdTai", format: "MJD", scale: "TAI" },
    identity: {
      dia: "pred.is_sso false AND diaObjectId > 0",
      object: "derived grouping key for DIA; broker values remain source-time snapshots"
    },
    known_absent_capabilities: [
      "complete historical detections",
      "forced photometry / upper limits",
      "canonical DiaObject record",
      "SSO identity / orbits"
    ]
  },
  light_static_schema: {
    source_topic: SCHEMA_TOPIC,
    schema_groups: Object.keys(deliveryEvidence.schema_groups),
    top_level_fields: avro.fields.map((f) => f.name),
    pred_fields: recordFields("pred"),
    clf_fields: recordFields("clf"),
    xm_fields: recordFields("xm"),
    misc_fields: recordFields("misc"),
    lc_features_map: "map<band, struct>",
    lc_features_fields: lcRecord.fields.map((f) => f.name)
  },
  extracted_files: extracted.sort((a, b) => a.path.localeCompare(b.path))
};

const text = `${JSON.stringify(excerpt, null, 2)}\n`;
if (check) {
  const committed = await readFile(outputPath, "utf8");
  if (committed !== text) {
    console.error(`${path.relative(process.cwd(), outputPath)} differs from the pinned Fink evidence.`);
    process.exit(1);
  }
  console.log(`Fink evidence excerpt matches ${PINNED_REVISION} (${extracted.length} files).`);
} else {
  await writeFile(outputPath, text, "utf8");
  console.log(`Wrote ${path.relative(process.cwd(), outputPath)} from ${PINNED_REVISION} (${extracted.length} files).`);
}
