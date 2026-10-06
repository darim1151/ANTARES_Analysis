/**
 * HEALPix NESTED-scheme geometry (Górski et al. 2005), ported from the
 * reference healpix_base algorithms so pixel indices match healpy/HATS.
 *
 * Pure module: no runtime imports, so it runs unchanged in the browser, in
 * the bundle generator and under `node --test`.
 *
 * Orders are limited to 0..13 so every index fits in 31-bit integer math.
 */

export const MAX_ORDER = 13;
const HALF_PI = Math.PI / 2;
const TWO_THIRDS = 2 / 3;
const FULL_SKY_DEG2 = 129600 / Math.PI;
const JRLL = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

function assertOrder(order: number) {
  if (!Number.isInteger(order) || order < 0 || order > MAX_ORDER) {
    throw new RangeError(`HEALPix order must be an integer in [0, ${MAX_ORDER}]`);
  }
}

export function nside(order: number): number {
  assertOrder(order);
  return 2 ** order;
}

export function npix(order: number): number {
  assertOrder(order);
  return 12 * 4 ** order;
}

/** Equal area of every pixel at this order, in square degrees. */
export function pixelAreaDeg2(order: number): number {
  return FULL_SKY_DEG2 / npix(order);
}

function spread(value: number, order: number): number {
  let result = 0;
  for (let bit = 0; bit < order; bit += 1) {
    if ((value >>> bit) & 1) result += 2 ** (2 * bit);
  }
  return result;
}

function compress(value: number, order: number, offset: 0 | 1): number {
  let result = 0;
  for (let bit = 0; bit < order; bit += 1) {
    if (Math.floor(value / 2 ** (2 * bit + offset)) % 2 === 1) result += 2 ** bit;
  }
  return result;
}

function xyf2nest(ix: number, iy: number, face: number, order: number): number {
  return face * 4 ** order + spread(ix, order) + 2 * spread(iy, order);
}

export function nest2xyf(order: number, ipix: number): { ix: number; iy: number; face: number } {
  const nPerFace = 4 ** order;
  const face = Math.floor(ipix / nPerFace);
  const local = ipix - face * nPerFace;
  return { ix: compress(local, order, 0), iy: compress(local, order, 1), face };
}

/** Pixel containing colatitude theta and longitude phi (radians). */
export function ang2pixNest(order: number, theta: number, phi: number): number {
  const ns = nside(order);
  const z = Math.cos(theta);
  const za = Math.abs(z);
  let tt = (phi / HALF_PI) % 4;
  if (tt < 0) tt += 4;

  if (za <= TWO_THIRDS) {
    const temp1 = ns * (0.5 + tt);
    const temp2 = ns * z * 0.75;
    const jp = Math.floor(temp1 - temp2);
    const jm = Math.floor(temp1 + temp2);
    const ifp = Math.floor(jp / ns);
    const ifm = Math.floor(jm / ns);
    // `ifp | 4` (not `+ 4`): ifp may be 4 when phi wraps towards 2*pi.
    const face = ifp === ifm ? ifp | 4 : ifp < ifm ? ifp : ifm + 8;
    const ix = jm % ns;
    const iy = ns - (jp % ns) - 1;
    return xyf2nest(ix, iy, face, order);
  }

  const ntt = Math.min(3, Math.floor(tt));
  const tp = tt - ntt;
  const tmp = ns * Math.sqrt(3 * (1 - za));
  const jp = Math.min(ns - 1, Math.floor(tp * tmp));
  const jm = Math.min(ns - 1, Math.floor((1 - tp) * tmp));
  return z >= 0 ? xyf2nest(ns - jm - 1, ns - jp - 1, ntt, order) : xyf2nest(jp, jm, ntt + 8, order);
}

/** Pixel containing an ICRS position in degrees. */
export function radecToPix(order: number, raDeg: number, decDeg: number): number {
  const theta = ((90 - decDeg) * Math.PI) / 180;
  const phi = (raDeg * Math.PI) / 180;
  return ang2pixNest(order, theta, phi);
}

/** Continuous face coordinates (x, y in [0, 1]) to (z = cos theta, phi). */
function xyf2loc(x: number, y: number, face: number): { z: number; phi: number } {
  const jr = JRLL[face] - x - y;
  let nr: number;
  let z: number;
  if (jr < 1) {
    nr = jr;
    z = 1 - (nr * nr) / 3;
  } else if (jr > 3) {
    nr = 4 - jr;
    z = (nr * nr) / 3 - 1;
  } else {
    nr = 1;
    z = ((2 - jr) * 2) / 3;
  }
  let tmp = JPLL[face] * nr + x - y;
  if (tmp < 0) tmp += 8;
  if (tmp >= 8) tmp -= 8;
  const phi = nr < 1e-15 ? 0 : (0.5 * HALF_PI * tmp) / nr;
  return { z, phi };
}

function locToRaDec(z: number, phi: number): [number, number] {
  const dec = (Math.asin(Math.max(-1, Math.min(1, z))) * 180) / Math.PI;
  let ra = (phi * 180) / Math.PI;
  ra %= 360;
  if (ra < 0) ra += 360;
  return [ra, dec];
}

/** Pixel center as [ra, dec] in degrees. */
export function pixToRaDec(order: number, ipix: number): [number, number] {
  const ns = nside(order);
  const { ix, iy, face } = nest2xyf(order, ipix);
  const { z, phi } = xyf2loc((ix + 0.5) / ns, (iy + 0.5) / ns, face);
  return locToRaDec(z, phi);
}

/**
 * Pixel boundary as 4*step [ra, dec] vertices (degrees), counter-clockwise
 * from the north-east corner, following the reference `boundaries` routine.
 */
export function pixBoundaryRaDec(order: number, ipix: number, step = 1): Array<[number, number]> {
  const ns = nside(order);
  const { ix, iy, face } = nest2xyf(order, ipix);
  const dc = 0.5 / ns;
  const xc = (ix + 0.5) / ns;
  const yc = (iy + 0.5) / ns;
  const d = 1 / (step * ns);
  const out: Array<[number, number]> = new Array(4 * step);
  for (let i = 0; i < step; i += 1) {
    const a = xyf2loc(xc + dc - i * d, yc + dc, face);
    const b = xyf2loc(xc - dc, yc + dc - i * d, face);
    const c = xyf2loc(xc - dc + i * d, yc - dc, face);
    const e = xyf2loc(xc + dc, yc - dc + i * d, face);
    out[i] = locToRaDec(a.z, a.phi);
    out[i + step] = locToRaDec(b.z, b.phi);
    out[i + 2 * step] = locToRaDec(c.z, c.phi);
    out[i + 3 * step] = locToRaDec(e.z, e.phi);
  }
  return out;
}

/** Parent index of a NESTED pixel at a coarser order. */
export function degradePix(ipix: number, fromOrder: number, toOrder: number): number {
  if (toOrder > fromOrder) throw new RangeError("degradePix cannot refine");
  return Math.floor(ipix / 4 ** (fromOrder - toOrder));
}

/** Aggregate a sparse NESTED count map to a coarser order. */
export function degradeCounts(
  pixels: readonly number[],
  values: readonly number[],
  fromOrder: number,
  toOrder: number
): Map<number, number> {
  const out = new Map<number, number>();
  for (let i = 0; i < pixels.length; i += 1) {
    const parent = degradePix(pixels[i], fromOrder, toOrder);
    out.set(parent, (out.get(parent) ?? 0) + values[i]);
  }
  return out;
}

/** Collapse a single-order pixel set into a normalized multi-order MOC. */
export function normalizeMoc(pixels: Iterable<number>, maxOrder: number): Record<string, number[]> {
  let current = new Set(pixels);
  const out: Record<string, number[]> = {};
  for (let order = maxOrder; order > 0; order -= 1) {
    const parents = new Map<number, number>();
    for (const p of current) {
      const parent = Math.floor(p / 4);
      parents.set(parent, (parents.get(parent) ?? 0) + 1);
    }
    const keep: number[] = [];
    const next = new Set<number>();
    for (const p of current) {
      if (parents.get(Math.floor(p / 4)) === 4) next.add(Math.floor(p / 4));
      else keep.push(p);
    }
    if (keep.length) out[String(order)] = keep.sort((a, b) => a - b);
    current = next;
  }
  if (current.size) out["0"] = [...current].sort((a, b) => a - b);
  return out;
}

/**
 * Cells at `order` that intersect the MOC: coarser MOC cells expand to all
 * their descendants, finer MOC cells mark their ancestor at `order`.
 */
export function mocCellsAtOrder(moc: Record<string, number[]>, order: number): Set<number> {
  const out = new Set<number>();
  for (const [key, cells] of Object.entries(moc)) {
    const o = Number(key);
    for (const cell of cells) {
      if (o >= order) {
        out.add(degradePix(cell, o, order));
      } else {
        const span = 4 ** (order - o);
        for (let child = cell * span; child < (cell + 1) * span; child += 1) out.add(child);
      }
    }
  }
  return out;
}
