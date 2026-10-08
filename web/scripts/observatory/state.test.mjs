#!/usr/bin/env node
// Scientific-state tests against the committed First-Light bundle (node --test).
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { buildWorkspaceModel, capability } from "../../lib/observatory/model.ts";
import { adaptiveSkyOrder, buildViewManifest, defaultState, reduce, stateFromUrl, stateToUrl } from "../../lib/observatory/state.ts";
import { computeMasks } from "../../lib/observatory/kernel/selection.ts";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "public", "observatory", "first-light");
const read = async (rel) => JSON.parse(await readFile(path.join(root, rel), "utf8"));

async function loadBundle() {
  const manifest = await read("manifest.json");
  const domains = {};
  for (const [d, refs] of Object.entries(manifest.domains)) {
    domains[d] = { time: await read(refs.time), sky: await read(refs.sky), entities: await read(refs.entities), features: await read(refs.features) };
  }
  return {
    base: root,
    manifest,
    manifestSha256: null,
    basis: await read(manifest.basis),
    capabilities: await read(manifest.capabilities),
    provenance: await read(manifest.provenance),
    domains,
    integrity: { method: "unavailable", files: [] }
  };
}

const model = buildWorkspaceModel(await loadBundle());

test("default state is valid against the basis and adapts the sky order to the population", () => {
  const s = defaultState(model);
  assert.equal(s.basis_id, model.bundle.basis.basis_id);
  assert.equal(s.mode, "compare");
  for (const d of ["antares", "fink"]) {
    const ids = new Set(model.domains[d].selectable.map((x) => x.id));
    assert.ok(ids.has(s.lens.lab[d].x) && ids.has(s.lens.lab[d].y));
  }
  assert.equal(s.presentation.skyOrder, adaptiveSkyOrder(model));
  assert.equal(adaptiveSkyOrder(model), 3, "1,800 entities per domain support order 3, not finer");
});

test("source switching preserves compatible context and retains native predicates", () => {
  let s = defaultState(model);
  s = reduce(s, { type: "time", time: { kind: "utc_dates", start: "2026-02-26", stop: "2026-03-01" } });
  s = reduce(s, { type: "sky", sky: { kind: "cone", ra: 150, dec: -20, radius_deg: 15 } });
  s = reduce(s, { type: "feature", domain: "fink", predicate: { x: { dimension: "fink.time_baseline_days", min: 2, max: 10 }, y: { dimension: "fink.lc_features.r.chi2", min: 1, max: 1e4 } } });
  s = reduce(s, { type: "focus", focus: { domain: "fink", kind: "fink.diaObject", id: model.domains.fink.records[0].id } });
  const switched = reduce(s, { type: "mode", mode: "antares" });
  assert.deepEqual(switched.selection, s.selection, "time, sky and the Fink feature predicate survive a switch to ANTARES");
  assert.deepEqual(switched.focus, s.focus, "focus is retained, never re-mapped to the other broker");
  // The Fink brush never filters ANTARES entities.
  const a = computeMasks(model.domains.antares.cols, switched.selection);
  const aNoFeature = computeMasks(model.domains.antares.cols, { ...switched.selection, feature: {} });
  assert.equal(a.counts.all, aNoFeature.counts.all);
});

test("changing a Lab axis retires the brush drawn on the old axes", () => {
  let s = defaultState(model);
  s = reduce(s, { type: "feature", domain: "antares", predicate: { x: { dimension: s.lens.lab.antares.x, min: 10, max: 100 }, y: { dimension: s.lens.lab.antares.y, min: 15, max: 17 } } });
  assert.ok(s.selection.feature.antares);
  s = reduce(s, { type: "lab", domain: "antares", patch: { y: "antares.galactic_latitude" } });
  assert.equal(s.selection.feature.antares, undefined);
  s = reduce(s, { type: "lab", domain: "antares", patch: { statistic: "median", z: "antares.ecliptic_latitude" } });
  assert.equal(s.lens.lab.antares.statistic, "median");
});

test("HEALPix cell toggling: click selects/clears, shift adds/removes", () => {
  let s = defaultState(model);
  s = reduce(s, { type: "skyCell", order: 3, pixel: 10, additive: false });
  assert.deepEqual(s.selection.sky, { kind: "healpix", order: 3, pixels: [10] });
  s = reduce(s, { type: "skyCell", order: 3, pixel: 4, additive: true });
  assert.deepEqual(s.selection.sky.pixels, [4, 10]);
  s = reduce(s, { type: "skyCell", order: 3, pixel: 10, additive: true });
  assert.deepEqual(s.selection.sky.pixels, [4]);
  s = reduce(s, { type: "skyCell", order: 3, pixel: 4, additive: false });
  assert.equal(s.selection.sky, null);
  s = reduce(s, { type: "skyCell", order: 3, pixel: 4, additive: false });
  s = reduce(s, { type: "skyCell", order: 4, pixel: 99, additive: true });
  assert.deepEqual(s.selection.sky, { kind: "healpix", order: 4, pixels: [99] }, "a different order starts a new set");
});

test("URL round trip restores the full scientific state", () => {
  let s = defaultState(model);
  s = reduce(s, { type: "mode", mode: "fink" });
  s = reduce(s, { type: "time", time: { kind: "utc_dates", start: "2026-03-02", stop: "2026-03-05" } });
  s = reduce(s, { type: "skyCell", order: 3, pixel: 500, additive: false });
  s = reduce(s, { type: "focus", focus: { domain: "antares", kind: "antares.locus", id: model.domains.antares.records[5].id } });
  s = reduce(s, { type: "lens", primary: "population" });
  const back = stateFromUrl(model, stateToUrl(s));
  assert.deepEqual(back.warnings, []);
  assert.deepEqual(back.state, s);
});

test("Fink zero-row dates are ZERO, never MISSING, and sampled entities stay inside the cohort", () => {
  const fink = model.domains.fink;
  const states = new Map(fink.bundle.time.nights.map((n) => [n.date, n.state]));
  assert.equal(states.get("2026-02-25"), "AVAILABLE");
  assert.equal(states.get("2026-03-05"), "ZERO");
  assert.equal(fink.bundle.time.counts.values["2026-03-05"], 0);
  assert.ok([...states.values()].every((st) => st === "AVAILABLE" || st === "ZERO"), "every cohort date is delivered or zero-row");
  assert.equal(states.size, 139);
  assert.ok(fink.cols.dates.every((d) => states.get(d) === "AVAILABLE"), "sampled DiaObjects start on delivered dates");
  const after = { kind: "utc_dates", start: "2026-07-14", stop: "2026-08-01" };
  assert.equal(computeMasks(fink.cols, { version: 1, time: after, sky: null, feature: {} }).counts.all, 0);
  assert.equal(fink.complete, false, "the Fink entity layer is a sample");
});

test("view manifest pins the basis, carries the state and declares evidence in view", () => {
  let s = defaultState(model);
  s = reduce(s, { type: "mode", mode: "antares" });
  const vm = buildViewManifest(model, s);
  assert.equal(vm.kind, "uso.view-manifest");
  assert.equal(vm.basis.relation, null);
  assert.equal(vm.basis.science_ready, false);
  assert.deepEqual(vm.state, s);
  assert.ok(vm.caveats[0].includes("not science-ready"));
  // Evidence follows what is drawn: ANTARES sky/lab only, both time lanes (one as context).
  assert.deepEqual(vm.evidence_by_lens["sky.antares.density"], ["LEGACY_SAMPLE"]);
  assert.deepEqual(vm.evidence_by_lens["lab.antares.y"], ["LEGACY_SAMPLE"]);
  assert.equal(vm.evidence_by_lens["sky.fink.density"], undefined);
  assert.deepEqual(vm.evidence_by_lens["time.fink"], ["COMMITTED_OPERATIONAL_RECORD", "VALIDATED_TRANSPORT_EVIDENCE"]);
  // Real catalog values are transport evidence, never accepted science and never synthetic.
  const focused = buildViewManifest(model, reduce(s, { type: "focus", focus: { domain: "fink", kind: "fink.diaObject", id: model.domains.fink.records[0].id } }));
  assert.deepEqual(focused.evidence_by_lens["inspector.fink"], ["VALIDATED_TRANSPORT_EVIDENCE"]);
  const compare = buildViewManifest(model, reduce(s, { type: "mode", mode: "compare" }));
  assert.deepEqual(compare.evidence_by_lens["sky.fink.density"], ["VALIDATED_TRANSPORT_EVIDENCE"]);
  for (const [lens, evidence] of Object.entries(compare.evidence_by_lens)) {
    if (lens.includes("fink")) assert.ok(!evidence.some((e) => e.startsWith("SYNTHETIC")), `${lens} must not be synthetic`);
  }
});

test("time admission distinguishes not-admitted dates from zero", async () => {
  const { timeAdmission } = await import("../../lib/observatory/model.ts");
  // Zero-row dates are admitted (ZERO), not missing.
  const march = timeAdmission(model.domains.fink, { kind: "utc_dates", start: "2026-03-05", stop: "2026-03-12" });
  assert.deepEqual(march, { total: 7, admitted: 7, byState: { ZERO: 2, AVAILABLE: 5 }, status: "FULL" });
  const pastCohort = timeAdmission(model.domains.fink, { kind: "utc_dates", start: "2026-07-12", stop: "2026-07-16" });
  assert.equal(pastCohort.admitted, 2);
  assert.equal(pastCohort.status, "PARTIAL");
  const antaresOutside = timeAdmission(model.domains.antares, { kind: "utc_dates", start: "2026-03-02", stop: "2026-03-06" });
  assert.equal(antaresOutside.status, "PARTIAL");
  assert.deepEqual(antaresOutside.byState, { AVAILABLE: 2, OUTSIDE_COVERAGE: 2 });
  assert.equal(timeAdmission(model.domains.fink, null), null);
});

test("sky orders never exceed the density maps the basis provides", () => {
  const finest = Math.min(model.bundle.domains.antares.sky.density.order, model.bundle.domains.fink.sky.density.order);
  assert.ok(model.skyOrders.length > 0 && model.skyOrders.every((o) => o <= finest));
  assert.throws(() => model.domains.fink.densityAt(finest + 1), RangeError);
});

test("capability lookups never invent availability", () => {
  assert.equal(capability(model, "relation", "relation.cross_broker_association").state, "UNAVAILABLE");
  assert.equal(capability(model, "workspace", "compare.difference_map").state, "UNAVAILABLE");
  assert.equal(capability(model, "antares", "sky.coverage").state, "UNAVAILABLE");
  assert.equal(capability(model, "fink", "time.acquisition.month-3").state, "AVAILABLE");
  assert.deepEqual(capability(model, "fink", "time.acquisition.month-3").codes, ["DELIVERY_VALIDATED", "ADMITTED", "CHARACTERIZED"]);
  assert.equal(capability(model, "fink", "sky.coverage").state, "UNAVAILABLE");
  assert.equal(capability(model, "fink", "sky.filtered_density").state, "UNAVAILABLE");
  assert.deepEqual(capability(model, "fink", "sky.filtered_density").codes, ["SAMPLED_ENTITY_LAYER"]);
  assert.equal(capability(model, "fink", "something.undeclared").state, "UNAVAILABLE");
});
