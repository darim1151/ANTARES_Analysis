#!/usr/bin/env node
// Unit tests for the pure Observatory analysis kernel (node --test).
import assert from "node:assert/strict";
import test from "node:test";

import {
  ang2pixNest,
  degradeCounts,
  npix,
  pixBoundaryRaDec,
  pixelAreaDeg2,
  pixToRaDec,
  radecToPix
} from "../../lib/observatory/kernel/healpix.ts";
import {
  insideMollweide,
  projectMollweide,
  unprojectMollweide
} from "../../lib/observatory/kernel/projection.ts";
import {
  angularSeparationDeg,
  eclipticToIcrs,
  galacticToIcrs,
  GALACTIC_CENTER_ICRS,
  icrsToEcliptic,
  icrsToGalactic
} from "../../lib/observatory/kernel/astro.ts";
import { addDays, isUtcDate, utcDateRange, utcMjdToUtcDate } from "../../lib/observatory/kernel/dates.ts";
import { computeMasks, ENTITY_INDEX_ORDER } from "../../lib/observatory/kernel/selection.ts";
import { binOf, histogram2d, robustExtent } from "../../lib/observatory/kernel/population.ts";
import { decodeState, encodeState } from "../../lib/observatory/kernel/stateCodec.ts";
import { fnv1a32, shardOf } from "../../lib/observatory/kernel/shard.ts";

const DEG = Math.PI / 180;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomSkyPoint(r) {
  return [r() * 360, (Math.asin(2 * r() - 1) * 180) / Math.PI];
}

// Gnomonic projection about (ra0, dec0) for a local point-in-polygon test.
function gnomonic(ra, dec, ra0, dec0) {
  const a = ra * DEG;
  const d = dec * DEG;
  const a0 = ra0 * DEG;
  const d0 = dec0 * DEG;
  const cosc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a - a0);
  return [
    (Math.cos(d) * Math.sin(a - a0)) / cosc,
    (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a - a0)) / cosc
  ];
}

function pointInPolygon([x, y], poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

test("HEALPix reference pixels match the published NESTED layout", () => {
  assert.equal(npix(0), 12);
  assert.equal(npix(4), 3072);
  assert.equal(ang2pixNest(0, 0.1, 0), 0); // north polar face 0
  assert.equal(ang2pixNest(0, Math.PI / 2, 0), 4); // equatorial face 4 at phi = 0
  assert.equal(ang2pixNest(0, Math.PI - 0.1, 0.1), 8); // south polar face 8
  assert.equal(ang2pixNest(0, Math.PI / 2, Math.PI / 2), 5);
  // phi -> 2*pi wraps into equatorial face 4 (the `ifp | 4` case), continuously.
  assert.equal(ang2pixNest(0, Math.PI / 2 + 0.6, 2 * Math.PI - 1e-9), 4);
  // Longitude is 2*pi periodic (phi = 0 itself is a pixel boundary, so test off it).
  for (const order of [0, 3, 7]) {
    for (const theta of [0.3, 1.0, Math.PI / 2 + 0.013, 2.0, 2.9]) {
      for (const phi of [0.0137, 1.3, 4.0, 2 * Math.PI - 0.0137]) {
        assert.equal(ang2pixNest(order, theta, phi + 2 * Math.PI), ang2pixNest(order, theta, phi));
        assert.equal(ang2pixNest(order, theta, phi - 2 * Math.PI), ang2pixNest(order, theta, phi));
      }
    }
  }
  assert.ok(Math.abs(pixelAreaDeg2(0) * 12 - 41252.961249) < 1e-3);
});

test("HEALPix: every pixel center maps to its own pixel (orders 0-5)", () => {
  for (let order = 0; order <= 5; order += 1) {
    for (let p = 0; p < npix(order); p += 1) {
      const [ra, dec] = pixToRaDec(order, p);
      assert.equal(radecToPix(order, ra, dec), p, `order ${order} pixel ${p}`);
    }
  }
});

test("HEALPix: random points lie inside their pixel boundary polygon", () => {
  const r = rng(20261006);
  for (let order = 0; order <= 7; order += 1) {
    for (let k = 0; k < 600; k += 1) {
      const [ra, dec] = randomSkyPoint(r);
      const p = radecToPix(order, ra, dec);
      assert.ok(p >= 0 && p < npix(order));
      const [ra0, dec0] = pixToRaDec(order, p);
      const poly = pixBoundaryRaDec(order, p, 8).map(([a, d]) => gnomonic(a, d, ra0, dec0));
      // Gnomonic is undefined beyond 90 deg; orders 0-1 cells are too large for this check near edges.
      if (order >= 2) {
        assert.ok(pointInPolygon(gnomonic(ra, dec, ra0, dec0), poly), `order ${order} (${ra}, ${dec}) -> ${p}`);
      }
      assert.ok(angularSeparationDeg(ra, dec, ra0, dec0) < 2.2 * Math.sqrt(pixelAreaDeg2(order)));
    }
  }
});

test("HEALPix degradation aggregates NESTED children into parents", () => {
  const r = rng(7);
  const pixels = [];
  const values = [];
  const expected = new Map();
  for (let k = 0; k < 500; k += 1) {
    const [ra, dec] = randomSkyPoint(r);
    pixels.push(radecToPix(6, ra, dec));
    values.push(1);
    const parent = radecToPix(3, ra, dec);
    expected.set(parent, (expected.get(parent) ?? 0) + 1);
  }
  assert.deepEqual(new Map([...degradeCounts(pixels, values, 6, 3)].sort()), new Map([...expected].sort()));
});

test("Mollweide: forward/inverse round trip and astronomical orientation", () => {
  const r = rng(99);
  for (let k = 0; k < 2000; k += 1) {
    const [ra, dec] = randomSkyPoint(r);
    const { x, y } = projectMollweide(ra, dec, 0);
    assert.ok(insideMollweide(x, y));
    const back = unprojectMollweide(x, y, 0);
    assert.ok(back);
    assert.ok(angularSeparationDeg(ra, dec, back.ra, back.dec) < 1e-6);
  }
  const center = projectMollweide(0, 0, 0);
  assert.ok(Math.abs(center.x) < 1e-12 && Math.abs(center.y) < 1e-12);
  assert.ok(Math.abs(projectMollweide(0, 90, 0).y - Math.SQRT2) < 1e-9);
  assert.ok(projectMollweide(90, 0, 0).x < 0, "RA increases to the left (east left)");
  assert.equal(unprojectMollweide(2.9, 0, 0), null);
});

test("Galactic and ecliptic transforms reproduce reference positions", () => {
  const [l, b] = icrsToGalactic(266.40499, -28.93617);
  assert.ok(Math.min(l, 360 - l) < 0.01 && Math.abs(b) < 0.01, `GC -> (${l}, ${b})`);
  const [, bNgp] = icrsToGalactic(192.85948, 27.12825);
  assert.ok(Math.abs(bNgp - 90) < 0.01);
  assert.ok(angularSeparationDeg(...GALACTIC_CENTER_ICRS, 266.40499, -28.93617) < 0.01);
  const r = rng(3);
  for (let k = 0; k < 200; k += 1) {
    const [ra, dec] = randomSkyPoint(r);
    assert.ok(angularSeparationDeg(ra, dec, ...galacticToIcrs(...icrsToGalactic(ra, dec))) < 1e-8);
    assert.ok(angularSeparationDeg(ra, dec, ...eclipticToIcrs(...icrsToEcliptic(ra, dec))) < 1e-8);
  }
  // Ecliptic north pole lies at RA 270, Dec 90 - obliquity.
  const [, betaPole] = icrsToEcliptic(270, 90 - 23.4392911);
  assert.ok(Math.abs(betaPole - 90) < 1e-6);
});

test("UTC dates are half-open calendar bins and never time-scale conversions", () => {
  assert.ok(isUtcDate("2026-02-28"));
  assert.ok(!isUtcDate("2026-02-30"));
  assert.equal(addDays("2026-02-28", 1), "2026-03-01");
  assert.deepEqual(utcDateRange("2026-02-27", "2026-03-02"), ["2026-02-27", "2026-02-28", "2026-03-01"]);
  assert.equal(utcMjdToUtcDate(61102.0), "2026-03-03");
  assert.equal(utcMjdToUtcDate(61102.999), "2026-03-03");
});

function columns(domain, rows, features = {}) {
  return {
    domain,
    n: rows.length,
    ra: Float64Array.from(rows.map((r) => r.ra)),
    dec: Float64Array.from(rows.map((r) => r.dec)),
    dates: rows.map((r) => r.date),
    hpx: Int32Array.from(rows.map((r) => radecToPix(ENTITY_INDEX_ORDER, r.ra, r.dec))),
    features: new Map(Object.entries(features).map(([k, v]) => [k, Float64Array.from(v, (x) => (x === null ? NaN : x))]))
  };
}

test("Selection: cross-filter masks exclude each lens's own predicate", () => {
  const cols = columns(
    "fink",
    [
      { ra: 10, dec: -30, date: "2026-02-26" },
      { ra: 10.5, dec: -30.2, date: "2026-02-27" },
      { ra: 200, dec: 10, date: "2026-02-26" },
      { ra: 11, dec: -29.5, date: "2026-03-10" }
    ],
    { "fink.n": [1, 5, 9, null] }
  );
  const selection = {
    version: 1,
    time: { kind: "utc_dates", start: "2026-02-26", stop: "2026-02-28" },
    sky: { kind: "cone", ra: 10, dec: -30, radius_deg: 2 },
    feature: { fink: { x: { dimension: "fink.n", min: 2, max: 10 }, y: null }, antares: { x: { dimension: "antares.x", min: 0, max: 1 }, y: null } }
  };
  const m = computeMasks(cols, selection);
  assert.deepEqual([...m.all], [0, 1, 0, 0]);
  assert.deepEqual([...m.exceptFeature], [1, 1, 0, 0]);
  assert.deepEqual([...m.exceptTime], [0, 1, 0, 0]);
  assert.deepEqual([...m.exceptSky], [0, 1, 1, 0]);
  assert.equal(m.counts.total, 4);
  // A null feature value never satisfies a feature predicate.
  assert.equal(m.feature[3], 0);
  // HEALPix predicate at a coarser order matches by NESTED parent.
  const hp = { version: 1, time: null, sky: { kind: "healpix", order: 3, pixels: [radecToPix(3, 10, -30)] }, feature: {} };
  assert.equal(computeMasks(cols, hp).all[0], 1);
  assert.equal(computeMasks(cols, hp).all[2], 0);
});

test("Population binning: fixed axes, log axes and statistics", () => {
  const xs = Float64Array.from([1, 10, 100, 1000, -1, NaN]);
  const ys = Float64Array.from([0, 0.5, 1, 1, 1, 1]);
  const zs = Float64Array.from([1, 2, 3, 5, 7, 9]);
  const ax = { scale: "log", min: 1, max: 1000, bins: 3 };
  const ay = { scale: "linear", min: 0, max: 1, bins: 2 };
  assert.equal(binOf(1000, ax), 2);
  assert.equal(binOf(-1, ax), -1);
  const h = histogram2d(xs, ys, null, ax, ay, "median", zs);
  assert.equal(h.usableXY, 4);
  assert.equal(h.counts.reduce((s, v) => s + v, 0), 4);
  assert.equal(h.stat[1 * 3 + 2], 4); // median of 3 and 5 in the top-right bin
  const [lo, hi] = robustExtent(Float64Array.from([5, 6, 7, NaN]), "linear", null);
  assert.ok(lo < 5 && hi > 7);
  assert.deepEqual(robustExtent(xs, "linear", [-90, 90]), [-90, 90]);
});

test("State codec round-trips and refuses foreign bases and unknown dimensions", () => {
  const defaults = {
    basis_id: "basis-x",
    mode: "antares",
    selection: { version: 1, time: null, sky: null, feature: {} },
    focus: null,
    lens: {
      primary: "sky",
      lab: {
        antares: { x: "antares.a", y: "antares.b", statistic: "count", z: null },
        fink: { x: "fink.a", y: "fink.b", statistic: "count", z: null }
      }
    },
    presentation: { skyOrder: 4, skyLayer: "density", overlays: { graticule: true, galacticPlane: true, ecliptic: true } }
  };
  const ctx = {
    basisId: "basis-x",
    isSelectableDimension: (d, dim) => dim.startsWith(`${d}.`) && dim !== "fink.unavailable",
    hasEntity: (ref) => ref.id === "990000000000000123",
    skyOrders: [3, 4, 5]
  };
  const state = structuredClone(defaults);
  state.mode = "compare";
  state.selection.time = { kind: "utc_dates", start: "2026-02-26", stop: "2026-03-01" };
  state.selection.sky = { kind: "healpix", order: 4, pixels: [12, 3, 7] };
  state.selection.feature.fink = { x: { dimension: "fink.a", min: 1.5, max: 3 }, y: { dimension: "fink.b", min: -2, max: 2 } };
  state.focus = { domain: "fink", kind: "fink.diaObject", id: "990000000000000123" };
  state.lens.primary = "population";
  state.lens.lab.fink = { x: "fink.a", y: "fink.b", statistic: "median", z: "fink.c" };
  state.presentation.overlays.ecliptic = false;
  const decoded = decodeState(encodeState(state), defaults, ctx);
  assert.deepEqual(decoded.warnings, []);
  assert.deepEqual(decoded.state.selection.sky, { kind: "healpix", order: 4, pixels: [3, 7, 12] });
  assert.deepEqual({ ...decoded.state, selection: { ...decoded.state.selection, sky: state.selection.sky } }, state);
  // The focus id is an int64 decimal string and must survive exactly.
  assert.equal(decoded.state.focus.id, "990000000000000123");

  const foreign = decodeState(encodeState({ ...state, basis_id: "other" }), defaults, ctx);
  assert.deepEqual(foreign.state, defaults);
  assert.equal(foreign.warnings.length, 1);

  // A brush on axes other than the decoded Lab axes is dropped, never kept invisibly.
  const mismatched = structuredClone(state);
  mismatched.lens.lab.fink = { x: "fink.b", y: "fink.a", statistic: "count", z: null };
  const dropped = decodeState(encodeState(mismatched), defaults, ctx);
  assert.equal(dropped.state.selection.feature.fink, undefined);
  assert.ok(dropped.warnings.some((w) => w.includes("does not match its Lab axes")));
  // Ranges keep significant digits (small-scale dimensions survive the URL).
  const tiny = structuredClone(state);
  tiny.selection.feature.fink.x = { dimension: "fink.a", min: 1.234567e-9, max: 9.87654321e-8 };
  assert.deepEqual(decodeState(encodeState(tiny), defaults, ctx).state.selection.feature.fink.x, tiny.selection.feature.fink.x);

  const bad = structuredClone(state);
  bad.selection.feature.fink.x.dimension = "fink.unavailable";
  const rejected = decodeState(encodeState(bad), defaults, ctx);
  assert.equal(rejected.state.selection.feature.fink, undefined);
  assert.ok(rejected.warnings.some((w) => w.includes("does not support")));
});

test("Detail shard routing is stable", () => {
  assert.equal(fnv1a32(""), 0x811c9dc5);
  assert.equal(fnv1a32("a"), 0xe40c292c);
  assert.equal(shardOf("ANT2020jrbbk", 4), fnv1a32("ANT2020jrbbk") % 4);
});

test("MOC normalization collapses complete siblings and expands losslessly", async () => {
  const { normalizeMoc, mocCellsAtOrder } = await import("../../lib/observatory/kernel/healpix.ts");
  const r = rng(11);
  const base = new Set();
  for (let p = 0; p < npix(5); p += 1) {
    const [, dec] = pixToRaDec(5, p);
    if (dec > -40 && dec < 10 && r() > 0.02) base.add(p);
  }
  const moc = normalizeMoc(base, 5);
  const total = Object.values(moc).reduce((s, cells) => s + cells.length, 0);
  assert.ok(total < base.size, "normalized MOC must be smaller than the flat set");
  assert.deepEqual([...mocCellsAtOrder(moc, 5)].sort((a, b) => a - b), [...base].sort((a, b) => a - b));
  for (const [order, cells] of Object.entries(moc)) {
    if (Number(order) === 0) continue;
    const parents = new Map();
    for (const c of cells) parents.set(Math.floor(c / 4), (parents.get(Math.floor(c / 4)) ?? 0) + 1);
    assert.ok([...parents.values()].every((n) => n < 4), `uncollapsed siblings at order ${order}`);
  }
  // Any-overlap semantics at a coarser display order.
  const coarse = mocCellsAtOrder(moc, 3);
  for (const p of base) assert.ok(coarse.has(Math.floor(p / 16)));
});

test("HEALPix NESTED matches independent healpy 1.20.1 reference values (orders 3, 8, 12)", () => {
  // healpy.ang2pix(2**order, ra, dec, nest=True, lonlat=True), computed outside this repository.
  const reference = [
    [3, 10.0, -30.0, 257], [3, 266.40499, -28.93617, 450], [3, 359.99, 0.01, 304], [3, 0.0, 89.9, 63],
    [3, 123.456, -77.7, 578], [3, 200.0, 45.0, 172], [3, 45.0, 41.81, 15], [3, 315.0, -41.81, 752],
    [8, 10.0, -30.0, 263514], [8, 266.40499, -28.93617, 461282], [8, 359.99, 0.01, 311296], [8, 0.0, 89.9, 65535],
    [8, 123.456, -77.7, 592328], [8, 200.0, 45.0, 176281], [8, 45.0, 41.81, 16383], [8, 315.0, -41.81, 770048],
    [12, 10.0, -30.0, 67459733], [12, 266.40499, -28.93617, 118088310], [12, 359.99, 0.01, 79691776], [12, 0.0, 89.9, 16777151],
    [12, 123.456, -77.7, 151636063], [12, 200.0, 45.0, 45128089], [12, 45.0, 41.81, 4194303], [12, 315.0, -41.81, 197132288]
  ];
  for (const [order, ra, dec, pixel] of reference) assert.equal(radecToPix(order, ra, dec), pixel, `order ${order} (${ra}, ${dec})`);
  // healpy.pix2ang(2**order, pixel, nest=True, lonlat=True)
  for (const [order, pixel, ra, dec] of [
    [4, 1234, 8.4375, 14.477512186],
    [8, 500000, 238.359375, -6.429418463],
    [12, 123456789, 298.377685547, 3.209645264]
  ]) {
    const [r, d] = pixToRaDec(order, pixel);
    assert.ok(Math.abs(r - ra) < 1e-8 && Math.abs(d - dec) < 1e-8, `center of ${order}/${pixel}`);
  }
});
