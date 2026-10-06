/**
 * Shared-selection evaluation over one domain's native population.
 *
 * Time and sky predicates are geometric and evaluated through each domain's
 * own declared mappings (entity UTC date, ICRS position). Feature predicates
 * are native to one domain and only ever evaluated against that domain.
 *
 * Every lens uses a cross-filter mask that excludes its own predicate, so a
 * lens shows the population constrained by the *other* lenses.
 */

import type { DomainId, FeaturePredicate, SelectionSpec, SkyPredicate } from "@/types/observatory";
import { angularSeparationDeg } from "./astro.ts";
import { degradePix } from "./healpix.ts";
import { inUtcInterval } from "./dates.ts";

/** HEALPix order at which entity positions are indexed for sky predicates. */
export const ENTITY_INDEX_ORDER = 12;

export type DomainColumns = {
  domain: DomainId;
  n: number;
  ra: Float64Array;
  dec: Float64Array;
  dates: string[];
  /** NESTED pixel at ENTITY_INDEX_ORDER for each entity. */
  hpx: Int32Array;
  /** Feature columns; NaN marks undefined (null) values. */
  features: Map<string, Float64Array>;
};

export type MaskSet = {
  time: Uint8Array | null;
  sky: Uint8Array | null;
  feature: Uint8Array | null;
  all: Uint8Array;
  exceptTime: Uint8Array;
  exceptSky: Uint8Array;
  exceptFeature: Uint8Array;
  counts: { total: number; all: number; exceptTime: number; exceptSky: number; exceptFeature: number };
  /** Feature predicate present in the shared selection for this domain. */
  featureActive: boolean;
};

export function skyPredicateMask(cols: DomainColumns, sky: SkyPredicate): Uint8Array {
  const mask = new Uint8Array(cols.n);
  if (sky.kind === "healpix") {
    if (sky.order > ENTITY_INDEX_ORDER) return mask;
    const wanted = new Set(sky.pixels);
    for (let i = 0; i < cols.n; i += 1) {
      if (wanted.has(degradePix(cols.hpx[i], ENTITY_INDEX_ORDER, sky.order))) mask[i] = 1;
    }
    return mask;
  }
  for (let i = 0; i < cols.n; i += 1) {
    if (angularSeparationDeg(cols.ra[i], cols.dec[i], sky.ra, sky.dec) <= sky.radius_deg) mask[i] = 1;
  }
  return mask;
}

export function featurePredicateMask(cols: DomainColumns, predicate: FeaturePredicate): Uint8Array | null {
  const xs = cols.features.get(predicate.x.dimension);
  const ys = predicate.y ? cols.features.get(predicate.y.dimension) : undefined;
  if (!xs || (predicate.y && !ys)) return null;
  const mask = new Uint8Array(cols.n);
  for (let i = 0; i < cols.n; i += 1) {
    const x = xs[i];
    if (!(x >= predicate.x.min && x <= predicate.x.max)) continue;
    if (predicate.y && ys) {
      const y = ys[i];
      if (!(y >= predicate.y.min && y <= predicate.y.max)) continue;
    }
    mask[i] = 1;
  }
  return mask;
}

function and(n: number, parts: Array<Uint8Array | null>): Uint8Array {
  const out = new Uint8Array(n).fill(1);
  for (const part of parts) {
    if (!part) continue;
    for (let i = 0; i < n; i += 1) out[i] &= part[i];
  }
  return out;
}

function count(mask: Uint8Array): number {
  let c = 0;
  for (let i = 0; i < mask.length; i += 1) c += mask[i];
  return c;
}

export function computeMasks(cols: DomainColumns, selection: SelectionSpec): MaskSet {
  let time: Uint8Array | null = null;
  if (selection.time) {
    const { start, stop } = selection.time;
    time = new Uint8Array(cols.n);
    for (let i = 0; i < cols.n; i += 1) if (inUtcInterval(cols.dates[i], start, stop)) time[i] = 1;
  }
  const sky = selection.sky ? skyPredicateMask(cols, selection.sky) : null;
  const predicate = selection.feature[cols.domain];
  const feature = predicate ? featurePredicateMask(cols, predicate) : null;

  const all = and(cols.n, [time, sky, feature]);
  const exceptTime = and(cols.n, [sky, feature]);
  const exceptSky = and(cols.n, [time, feature]);
  const exceptFeature = and(cols.n, [time, sky]);
  return {
    time,
    sky,
    feature,
    all,
    exceptTime,
    exceptSky,
    exceptFeature,
    featureActive: Boolean(predicate && feature),
    counts: {
      total: cols.n,
      all: count(all),
      exceptTime: count(exceptTime),
      exceptSky: count(exceptSky),
      exceptFeature: count(exceptFeature)
    }
  };
}

export function selectionIsEmpty(selection: SelectionSpec): boolean {
  return !selection.time && !selection.sky && Object.keys(selection.feature).length === 0;
}
