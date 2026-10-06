/**
 * Projected sky geometry for the Mollweide lens: HEALPix cell polygons with
 * seam handling, sampled curves split at the RA seam, and small circles.
 * Coordinates are in projection units (see kernel/projection.ts).
 */

import { pixBoundaryRaDec, pixToRaDec } from "./kernel/healpix.ts";
import { projectMollweide, wrapDeg180, MOLLWEIDE_HALF_HEIGHT, MOLLWEIDE_HALF_WIDTH } from "./kernel/projection.ts";

/** Map center RA in degrees; RA increases to the left (east left). */
export const CENTER_RA = 0;

export type Ring = Array<[number, number]>;

function boundaryStep(order: number): number {
  if (order <= 2) return 8;
  if (order === 3) return 4;
  if (order === 4) return 2;
  return 1;
}

/** One ring, or two when the cell straddles the RA seam (clip to the ellipse when filling). */
export function cellRings(order: number, pixel: number): Ring[] {
  const [raC] = pixToRaDec(order, pixel);
  const lonC = wrapDeg180(raC - CENTER_RA);
  const vertices = pixBoundaryRaDec(order, pixel, boundaryStep(order));
  const lons = vertices.map(([ra]) => lonC + wrapDeg180(ra - raC));
  const ring: Ring = vertices.map(([ra, dec], i) => {
    const p = projectMollweide(ra, dec, CENTER_RA, lons[i]);
    return [p.x, p.y];
  });
  const finite = lons.filter((_, i) => Math.abs(vertices[i][1]) < 89.999);
  const rings: Ring[] = [ring];
  const shifted = (delta: number): Ring =>
    vertices.map(([ra, dec], i) => {
      const p = projectMollweide(ra, dec, CENTER_RA, lons[i] + delta);
      return [p.x, p.y];
    });
  if (Math.max(...finite) > 180) rings.push(shifted(-360));
  if (Math.min(...finite) < -180) rings.push(shifted(360));
  return rings;
}

const pathCache = new Map<number, Map<number, Path2D>>();

/** Cached canvas path for a cell in projection units. */
export function cellPath(order: number, pixel: number): Path2D {
  let byPixel = pathCache.get(order);
  if (!byPixel) {
    byPixel = new Map();
    pathCache.set(order, byPixel);
  }
  let path = byPixel.get(pixel);
  if (!path) {
    path = new Path2D();
    for (const ring of cellRings(order, pixel)) {
      ring.forEach(([x, y], i) => (i === 0 ? path!.moveTo(x, y) : path!.lineTo(x, y)));
      path.closePath();
    }
    byPixel.set(pixel, path);
  }
  return path;
}

/** Project a sampled sky curve, splitting segments where it crosses the seam. */
export function projectCurve(points: Array<[number, number]>): Ring[] {
  const segments: Ring[] = [];
  let current: Ring = [];
  let prevLon: number | null = null;
  for (const [ra, dec] of points) {
    const lon = wrapDeg180(ra - CENTER_RA);
    if (prevLon !== null && Math.abs(lon - prevLon) > 180) {
      if (current.length > 1) segments.push(current);
      current = [];
    }
    const p = projectMollweide(ra, dec, CENTER_RA);
    current.push([p.x, p.y]);
    prevLon = lon;
  }
  if (current.length > 1) segments.push(current);
  return segments;
}

/** Small circle of angular radius r (deg) around (ra, dec), sampled. */
export function smallCircle(ra: number, dec: number, radius: number, samples = 160): Array<[number, number]> {
  const d = (Math.PI / 180) * radius;
  const p1 = (Math.PI / 180) * dec;
  const l1 = (Math.PI / 180) * ra;
  return Array.from({ length: samples + 1 }, (_, i) => {
    const bearing = (2 * Math.PI * i) / samples;
    const p2 = Math.asin(Math.sin(p1) * Math.cos(d) + Math.cos(p1) * Math.sin(d) * Math.cos(bearing));
    const l2 = l1 + Math.atan2(Math.sin(bearing) * Math.sin(d) * Math.cos(p1), Math.cos(d) - Math.sin(p1) * Math.sin(p2));
    return [((((l2 * 180) / Math.PI) % 360) + 360) % 360, (p2 * 180) / Math.PI];
  });
}

export function meridian(ra: number, samples = 91): Array<[number, number]> {
  return Array.from({ length: samples }, (_, i) => [ra, -90 + (180 * i) / (samples - 1)]);
}

export function parallel(dec: number, samples = 181): Array<[number, number]> {
  // Sample from just east of the seam to just west so the curve never wraps.
  return Array.from({ length: samples }, (_, i) => [((CENTER_RA + 180 - 1e-6 - (360 - 2e-6) * (i / (samples - 1))) % 360 + 360) % 360, dec]);
}

export function ellipsePath(steps = 180): Ring {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / steps;
    return [MOLLWEIDE_HALF_WIDTH * Math.cos(t), MOLLWEIDE_HALF_HEIGHT * Math.sin(t)];
  });
}

export type Frame = { cx: number; cy: number; scale: number; width: number; height: number };

/** Fit the projection extent into a pixel box, leaving room for labels. */
export function fitFrame(width: number, height: number, padX: number, padY: number): Frame {
  const usableW = Math.max(40, width - 2 * padX);
  const usableH = Math.max(20, height - 2 * padY);
  const scale = Math.min(usableW / (2 * MOLLWEIDE_HALF_WIDTH), usableH / (2 * MOLLWEIDE_HALF_HEIGHT));
  return { cx: width / 2, cy: height / 2, scale, width, height };
}

export function toPx(frame: Frame, x: number, y: number): [number, number] {
  return [frame.cx + x * frame.scale, frame.cy - y * frame.scale];
}

export function ringToSvg(frame: Frame, ring: Ring, close = false): string {
  return ring.map(([x, y], i) => `${i === 0 ? "M" : "L"}${(frame.cx + x * frame.scale).toFixed(1)},${(frame.cy - y * frame.scale).toFixed(1)}`).join("") + (close ? "Z" : "");
}
