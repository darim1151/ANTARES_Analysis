#!/usr/bin/env node
// Mutation tests for the Observatory bundle validator. Each case corrupts a
// copy of the committed bundle, re-seals the manifest digests so the semantic
// check (not the integrity check) is exercised, and expects a specific error.

import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const validator = path.join(scriptDirectory, "validate-observatory-bundle.mjs");
const source = path.resolve(scriptDirectory, "..", "..", "public", "observatory", "first-light");

function run(dir) {
  return spawnSync(process.execPath, ["--disable-warning=MODULE_TYPELESS_PACKAGE_JSON", validator, `--bundle-dir=${dir}`], { encoding: "utf8" });
}

async function load(dir, rel) {
  return JSON.parse(await readFile(path.join(dir, rel), "utf8"));
}

async function save(dir, rel, doc, { reseal = true } = {}) {
  const text = rel === "manifest.json" || rel.endsWith("time.json") || !rel.startsWith("domains/") ? `${JSON.stringify(doc, null, 2)}\n` : `${JSON.stringify(doc)}\n`;
  await writeFile(path.join(dir, rel), text, "utf8");
  if (reseal && rel !== "manifest.json") {
    const manifest = await load(dir, "manifest.json");
    const entry = manifest.files.find((f) => f.path === rel);
    entry.bytes = Buffer.byteLength(text);
    entry.sha256 = createHash("sha256").update(text).digest("hex");
    await writeFile(path.join(dir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  }
}

async function edit(dir, rel, mutate, options) {
  const doc = await load(dir, rel);
  mutate(doc);
  await save(dir, rel, doc, options);
}

async function withCopy(fn) {
  const root = await mkdtemp(path.join(os.tmpdir(), "uso-bundle-"));
  const dir = path.join(root, "first-light");
  try {
    await cp(source, dir, { recursive: true });
    await fn(dir);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

const fail = (message) => {
  throw new Error(message);
};

await withCopy(async (dir) => {
  const result = run(dir);
  if (result.status !== 0) fail(`baseline bundle must validate:\n${result.stderr}`);
});

const FINK_TIME = "domains/fink/time.json";
const FINK_ENTITIES = "domains/fink/entities.json";
const FINK_FEATURES = "domains/fink/features.json";
const ANTARES_SKY = "domains/antares/sky.json";

const cases = [
  ["QUALIFIED relabel is refused", "only FIRST_LIGHT_FIXTURE", (d) => edit(d, "manifest.json", (m) => (m.bundle_class = "QUALIFIED"))],
  ["science_ready relabel is refused", "science_ready=false", (d) => edit(d, "manifest.json", (m) => (m.science_ready = true))],
  ["unsealed payload change is detected", "sha256 differs", (d) => edit(d, "basis.json", (b) => (b.label = "tampered"), { reseal: false })],
  ["unlisted files are rejected", "unlisted file", (d) => writeFile(path.join(d, "extra.json"), "{}\n")],
  ["generator inputs are pinned", "input changed", (d) => edit(d, "manifest.json", (m) => (m.generator.inputs[0].sha256 = "0".repeat(64)))],
  [
    "a cross-broker relation is refused",
    "relation must be null",
    (d) => edit(d, "basis.json", (b) => (b.relation = { relation_id: "x", version: "1", method: "cone 1 arcsec", evidence: ["SYNTHETIC_FIXTURE"] }))
  ],
  [
    "cross-broker association cannot be enabled",
    "cross-broker association must be declared UNAVAILABLE",
    (d) =>
      edit(d, "capabilities.json", (c) => {
        const x = c.capabilities.find((k) => k.id === "relation:relation.cross_broker_association");
        x.state = "AVAILABLE";
        x.evidence = ["SYNTHETIC_FIXTURE"];
      })
  ],
  [
    "count difference maps cannot be enabled",
    "difference maps must be declared UNAVAILABLE",
    (d) => edit(d, "capabilities.json", (c) => (c.capabilities.find((k) => k.id === "workspace:compare.difference_map").state = "PARTIALLY_QUALIFIED"))
  ],
  [
    "a delivered window cannot be labelled MISSING",
    "MISSING is reserved for genuinely absent data",
    (d) =>
      edit(d, FINK_TIME, (t) => {
        const w = t.windows.find((x) => x.id === "fink.month-3");
        w.state = "MISSING";
        for (const n of t.nights) if (n.window_id === w.id) n.state = "MISSING";
      })
  ],
  [
    "an AVAILABLE window must stay admitted",
    "an AVAILABLE window must be admitted",
    (d) => edit(d, FINK_TIME, (t) => (t.windows.find((x) => x.id === "fink.month-2").admission = "NOT_ADMITTED"))
  ],
  [
    "a validated catalog window cannot be relabelled unavailable",
    "window state must be AVAILABLE",
    (d) =>
      edit(d, FINK_TIME, (t) => {
        const w = t.windows.find((x) => x.id === "fink.month-2");
        Object.assign(w, { state: "UNAVAILABLE", delivery_validation: "NOT_DELIVERY_VALIDATED", admission: "NOT_ADMITTED", status_codes: ["NOT_DELIVERY_VALIDATED", "NOT_ADMITTED"] });
        for (const n of t.nights) if (n.window_id === w.id) n.state = "UNAVAILABLE";
        for (const date of Object.keys(t.counts.values)) if (date >= w.start && date < w.stop) delete t.counts.values[date];
      })
  ],
  [
    "per-date counts must equal the catalog's delivered rows",
    "count must equal the catalog's delivered rows",
    (d) => edit(d, FINK_TIME, (t) => (t.counts.values["2026-02-25"] += 1))
  ],
  [
    "a zero-row date cannot be shown as available",
    "night state must follow the catalog's delivered rows",
    (d) =>
      edit(d, FINK_TIME, (t) => {
        const n = t.nights.find((x) => x.state === "ZERO");
        n.state = "AVAILABLE";
        t.counts.values[n.date] = 1;
      })
  ],
  [
    "Month-2 transport totals cannot be compared as rates",
    "prohibit rate comparison",
    (d) => edit(d, FINK_TIME, (t) => (t.windows.find((x) => x.id === "fink.month-2").rate_comparison = "PERMITTED"))
  ],
  ["Fink time must stay TAI", "TAI scale", (d) => edit(d, "basis.json", (b) => (b.domains.fink.time.scale = "UTC"))],
  ["ACCEPTED_SCIENCE is forbidden", "ACCEPTED_SCIENCE is reserved", (d) => edit(d, "basis.json", (b) => b.domains.antares.evidence.push("ACCEPTED_SCIENCE"))],
  [
    "fixture evidence cannot be dropped",
    "must declare their synthetic evidence",
    (d) => edit(d, "basis.json", (b) => (b.domains.antares.evidence = ["LEGACY_SAMPLE"]))
  ],
  [
    "a catalog domain cannot carry synthetic evidence",
    "must not carry synthetic evidence",
    (d) => edit(d, "domains/fink/sky.json", (s) => s.density.evidence.push("SYNTHETIC_FIXTURE"))
  ],
  [
    "Fink density must equal the catalog extract",
    "complete DiaObject density",
    (d) => edit(d, "domains/fink/sky.json", (s) => (s.density.values[0] += 1))
  ],
  [
    "the Fink sample cannot claim client-side filtered density",
    "cannot be AVAILABLE for an incomplete population",
    (d) =>
      edit(d, "capabilities.json", (c) => {
        const x = c.capabilities.find((k) => k.id === "fink:sky.filtered_density");
        x.state = "AVAILABLE";
        x.evidence = ["VALIDATED_TRANSPORT_EVIDENCE"];
      })
  ],
  [
    "int64 identifiers must be strings",
    // Real diaObjectIds exceed 2^53, so the global unsafe-number scan fires first.
    "int64 values must be decimal strings",
    (d) => edit(d, FINK_ENTITIES, (e) => (e.records[0].id = Number(e.records[0].id)))
  ],
  [
    "sampled entities must be catalog DiaObjects",
    "not a sampled catalog DiaObject",
    (d) => edit(d, FINK_ENTITIES, (e) => (e.records[0].id = "123456789012345678"))
  ],
  [
    "sampled entities keep their catalog positions",
    "record disagrees with the catalog extract",
    (d) => edit(d, FINK_ENTITIES, (e) => (e.records[0].ra = (e.records[0].ra + 1) % 360))
  ],
  [
    "model scores must not become probabilities",
    "not calibrated probabilities",
    (d) => edit(d, FINK_FEATURES, (f) => (f.dimensions.find((x) => x.id === "fink.clf.cats_score").qualifications = []))
  ],
  ["columns must align with entities", "column must align", (d) => edit(d, FINK_FEATURES, (f) => f.columns["fink.max_snr"].pop())],
  [
    "unavailable dimensions ship no data",
    "must not ship a column",
    (d) => edit(d, FINK_FEATURES, (f) => (f.columns["fink.forced_photometry"] = f.columns["fink.max_snr"]))
  ],
  [
    "shared definition ids require identical definitions",
    "shares definition_id",
    (d) => edit(d, FINK_FEATURES, (f) => (f.dimensions.find((x) => x.id === "fink.galactic_latitude").definition = "something else"))
  ],
  [
    "absent coverage needs a reason",
    "absent coverage needs a reason",
    (d) => edit(d, ANTARES_SKY, (s) => (s.coverage_unavailable_reason = ""))
  ],
  [
    "density must equal the entity aggregation",
    "density must equal the HEALPix aggregation",
    (d) => edit(d, ANTARES_SKY, (s) => (s.density.values[0] += 1))
  ],
  [
    "sources cannot fall outside the cohort",
    "lies outside the admitted window",
    async (d) => {
      const entities = await load(d, FINK_ENTITIES);
      const id = entities.records[0].id;
      const { shardOf } = await import("../../lib/observatory/kernel/shard.ts");
      const rel = `domains/fink/detail/${String(shardOf(id, entities.detail.shard_count)).padStart(2, "0")}.json`;
      await edit(d, rel, (shard) => {
        const sources = shard.records[id].sources;
        sources[sources.length - 1].midpointMjdTai = 61300.2; // 2026-09-17, after the cohort
      });
    }
  ],
  [
    "broker fields are pinned to the Light Static schema",
    "is not in the pinned schema",
    async (d) => {
      const entities = await load(d, FINK_ENTITIES);
      const id = entities.records[0].id;
      const { shardOf } = await import("../../lib/observatory/kernel/shard.ts");
      const rel = `domains/fink/detail/${String(shardOf(id, entities.detail.shard_count)).padStart(2, "0")}.json`;
      await edit(d, rel, (shard) => (shard.records[id].snapshots[0].clf.sn_probability = 0.9));
    }
  ],
  [
    "host-local paths are rejected",
    "host-local filesystem path",
    (d) => edit(d, "provenance.json", (p) => (p.derivations[0].description = "read from /astro/store/shire/FINK/data"))
  ],
  [
    "forbidden public claims are rejected",
    "forbidden claim",
    (d) => edit(d, FINK_TIME, (t) => (t.windows[0].caveat = "A classified transient stream."))
  ],
  [
    "basis ids are bound to their pins",
    "basis_id must end with the pin digest",
    (d) => edit(d, "basis.json", (b) => (b.domains.fink.build_id = `${b.domains.fink.build_id}-other`))
  ],
  [
    "unsafe integers are rejected anywhere",
    "exceeds 2^53",
    (d) => edit(d, "provenance.json", (p) => (p.acquisitions[0].delivery.readable_rows = 9007199254740993))
  ],
  [
    "acquisition evidence is domain-scoped",
    "must declare its domain",
    (d) => edit(d, "provenance.json", (p) => delete p.acquisitions[0].domain)
  ],
  [
    "a sample population cannot claim client-side filtered density",
    "cannot be AVAILABLE for an incomplete population",
    (d) => edit(d, "domains/antares/entities.json", (e) => (e.population.complete = false))
  ],
  [
    "entities cannot fall on unavailable dates",
    "must fall on an AVAILABLE date",
    (d) => edit(d, FINK_ENTITIES, (e) => (e.records[0].entity_date = "2026-08-01"))
  ]
];

let passed = 0;
for (const [name, expected, mutate] of cases) {
  await withCopy(async (dir) => {
    await mutate(dir);
    const result = run(dir);
    if (result.status !== 1) fail(`${name}: validator unexpectedly exited ${result.status}\n${result.stdout}`);
    if (!result.stderr.includes(expected)) fail(`${name}: expected "${expected}" in:\n${result.stderr}`);
    passed += 1;
  });
}
console.log(`Observatory bundle validator tests passed: baseline + ${passed} rejected mutations.`);
