/**
 * Mollweide equal-area projection in astronomical orientation (east left).
 *
 * Projected coordinates use the canonical extent x in [-2*sqrt(2), 2*sqrt(2)],
 * y in [-sqrt(2), sqrt(2)]; y grows towards the north. Pure module.
 */

const SQRT2 = Math.SQRT2;
const DEG = Math.PI / 180;

export const MOLLWEIDE_HALF_WIDTH = 2 * SQRT2;
export const MOLLWEIDE_HALF_HEIGHT = SQRT2;

export type ProjectedPoint = { x: number; y: number };

/** Auxiliary angle theta solving 2*theta + sin(2*theta) = pi * sin(lat). */
function auxiliaryTheta(lat: number): number {
  if (Math.abs(Math.abs(lat) - Math.PI / 2) < 1e-12) return Math.sign(lat) * (Math.PI / 2);
  const target = Math.PI * Math.sin(lat);
  // Newton on 2t + sin 2t; starting at lat converges everywhere away from the poles.
  let theta = lat;
  for (let i = 0; i < 60; i += 1) {
    const f = 2 * theta + Math.sin(2 * theta) - target;
    const df = 2 + 2 * Math.cos(2 * theta);
    if (df < 1e-14) break;
    const delta = f / df;
    theta -= delta;
    if (Math.abs(delta) < 1e-12) break;
  }
  return theta;
}

/** Wrap an angle in degrees to (-180, 180]. */
export function wrapDeg180(value: number): number {
  let v = ((value + 180) % 360 + 360) % 360 - 180;
  if (v === -180) v = 180;
  return v;
}

/**
 * Project (ra, dec) in degrees. `centerRa` is the RA at the map center; RA
 * increases to the left. `lonOffsetDeg` lets callers draw seam-crossing
 * polygons continuously (values outside +/-180 project outside the ellipse).
 */
export function projectMollweide(
  raDeg: number,
  decDeg: number,
  centerRa = 0,
  unwrappedLonDeg?: number
): ProjectedPoint {
  const lonDeg = unwrappedLonDeg ?? wrapDeg180(raDeg - centerRa);
  const lambda = -lonDeg * DEG;
  const theta = auxiliaryTheta(decDeg * DEG);
  return {
    x: ((2 * SQRT2) / Math.PI) * lambda * Math.cos(theta),
    y: SQRT2 * Math.sin(theta)
  };
}

/** Inverse projection; returns null outside the ellipse. */
export function unprojectMollweide(
  x: number,
  y: number,
  centerRa = 0
): { ra: number; dec: number } | null {
  if (Math.abs(y) > SQRT2) return null;
  const theta = Math.asin(y / SQRT2);
  const cosTheta = Math.cos(theta);
  let lambda = 0;
  if (cosTheta > 1e-12) {
    lambda = (Math.PI * x) / (2 * SQRT2 * cosTheta);
    if (Math.abs(lambda) > Math.PI + 1e-12) return null;
  } else if (Math.abs(x) > 1e-9) {
    return null;
  }
  const sinLat = (2 * theta + Math.sin(2 * theta)) / Math.PI;
  const dec = Math.asin(Math.max(-1, Math.min(1, sinLat))) / DEG;
  let ra = centerRa - lambda / DEG;
  ra = ((ra % 360) + 360) % 360;
  return { ra, dec };
}

/** True when a projected point lies inside the Mollweide ellipse. */
export function insideMollweide(x: number, y: number): boolean {
  return (x / MOLLWEIDE_HALF_WIDTH) ** 2 + (y / MOLLWEIDE_HALF_HEIGHT) ** 2 <= 1 + 1e-12;
}
