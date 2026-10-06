/**
 * Coordinate kernel: ICRS <-> Galactic, ICRS -> mean ecliptic of J2000, and
 * great-circle distance. Pure module.
 *
 * Galactic: the ICRS-to-Galactic rotation used by Astropy/Hipparcos.
 * Ecliptic: mean obliquity of J2000.0 (23.4392911 deg); the ICRS frame bias
 * (~23 mas) is ignored, which is far below any display or binning scale here.
 */

const DEG = Math.PI / 180;

/** Rows are the Galactic x, y, z axes expressed in ICRS. */
const ICRS_TO_GALACTIC = [
  [-0.0548755604162154, -0.873437090234885, -0.4838350155487132],
  [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
  [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]
] as const;

const OBLIQUITY_J2000 = 23.4392911 * DEG;

function toVector(lonDeg: number, latDeg: number): [number, number, number] {
  const lon = lonDeg * DEG;
  const lat = latDeg * DEG;
  return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
}

function fromVector(v: readonly number[]): [number, number] {
  const lat = Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG;
  let lon = Math.atan2(v[1], v[0]) / DEG;
  if (lon < 0) lon += 360;
  return [lon, lat];
}

/** ICRS (ra, dec) degrees -> Galactic (l, b) degrees. */
export function icrsToGalactic(raDeg: number, decDeg: number): [number, number] {
  const v = toVector(raDeg, decDeg);
  const m = ICRS_TO_GALACTIC;
  return fromVector([
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
  ]);
}

/** Galactic (l, b) degrees -> ICRS (ra, dec) degrees (transpose rotation). */
export function galacticToIcrs(lDeg: number, bDeg: number): [number, number] {
  const v = toVector(lDeg, bDeg);
  const m = ICRS_TO_GALACTIC;
  return fromVector([
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]
  ]);
}

/** ICRS (ra, dec) degrees -> ecliptic (lambda, beta) degrees, mean J2000. */
export function icrsToEcliptic(raDeg: number, decDeg: number): [number, number] {
  const [x, y, z] = toVector(raDeg, decDeg);
  const c = Math.cos(OBLIQUITY_J2000);
  const s = Math.sin(OBLIQUITY_J2000);
  return fromVector([x, c * y + s * z, -s * y + c * z]);
}

/** Ecliptic (lambda, beta) degrees -> ICRS (ra, dec) degrees, mean J2000. */
export function eclipticToIcrs(lambdaDeg: number, betaDeg: number): [number, number] {
  const [x, y, z] = toVector(lambdaDeg, betaDeg);
  const c = Math.cos(OBLIQUITY_J2000);
  const s = Math.sin(OBLIQUITY_J2000);
  return fromVector([x, c * y - s * z, s * y + c * z]);
}

/** Great-circle separation in degrees (haversine, stable at small angles). */
export function angularSeparationDeg(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const p1 = dec1 * DEG;
  const p2 = dec2 * DEG;
  const dp = p2 - p1;
  const dl = (ra2 - ra1) * DEG;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return (2 * Math.asin(Math.min(1, Math.sqrt(h)))) / DEG;
}

/** ICRS position of the Galactic center (l = 0, b = 0). */
export const GALACTIC_CENTER_ICRS: [number, number] = galacticToIcrs(0, 0);

/** Sampled great circle (or small circle) in ICRS, for overlays. */
export function galacticLatitudeCurve(bDeg: number, samples = 361): Array<[number, number]> {
  return Array.from({ length: samples }, (_, i) => galacticToIcrs((360 * i) / (samples - 1), bDeg));
}

export function eclipticCurve(samples = 361): Array<[number, number]> {
  return Array.from({ length: samples }, (_, i) => eclipticToIcrs((360 * i) / (samples - 1), 0));
}
