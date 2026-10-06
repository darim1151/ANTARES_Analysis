/**
 * Population binning for the Feature/Population Lab. Pure module.
 *
 * Values are Float64Array columns where NaN marks an undefined value. Log
 * axes bin in log10 space and treat non-positive values as undefined; the
 * Lab reports such exclusions in its coverage readout rather than hiding them.
 */

export type AxisScale = "linear" | "log";

export type BinAxis = {
  scale: AxisScale;
  min: number;
  max: number;
  bins: number;
};

function quantileSorted(sorted: Float64Array, q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function usable(value: number, scale: AxisScale): boolean {
  return Number.isFinite(value) && (scale === "linear" || value > 0);
}

/**
 * Robust axis extent from the full population (not the filtered subset), so
 * axes stay fixed while the selection changes.
 */
export function robustExtent(
  values: Float64Array,
  scale: AxisScale,
  fixed: [number, number] | null
): [number, number] {
  if (fixed) return fixed;
  const finite = values.filter((v) => usable(v, scale));
  if (finite.length === 0) return scale === "log" ? [1, 10] : [0, 1];
  const sorted = Float64Array.from(finite).sort();
  let lo = quantileSorted(sorted, 0.005);
  let hi = quantileSorted(sorted, 0.995);
  if (scale === "log") {
    let a = Math.log10(lo);
    let b = Math.log10(hi);
    if (b - a < 1e-9) {
      a -= 0.5;
      b += 0.5;
    }
    const pad = (b - a) * 0.04;
    return [10 ** (a - pad), 10 ** (b + pad)];
  }
  if (hi - lo < 1e-12) {
    lo -= 0.5;
    hi += 0.5;
  }
  const pad = (hi - lo) * 0.04;
  return [lo - pad, hi + pad];
}

/** Fractional position in [0, 1] along the axis, NaN if unusable. */
export function axisFraction(value: number, axis: BinAxis): number {
  if (!usable(value, axis.scale)) return NaN;
  if (axis.scale === "log") {
    const a = Math.log10(axis.min);
    const b = Math.log10(axis.max);
    return (Math.log10(value) - a) / (b - a);
  }
  return (value - axis.min) / (axis.max - axis.min);
}

/** Inverse of axisFraction. */
export function axisValue(fraction: number, axis: BinAxis): number {
  if (axis.scale === "log") {
    const a = Math.log10(axis.min);
    const b = Math.log10(axis.max);
    return 10 ** (a + fraction * (b - a));
  }
  return axis.min + fraction * (axis.max - axis.min);
}

/** Bin index, or -1 when outside the axis or undefined. */
export function binOf(value: number, axis: BinAxis): number {
  const f = axisFraction(value, axis);
  if (!(f >= 0 && f <= 1)) return -1;
  return Math.min(axis.bins - 1, Math.floor(f * axis.bins));
}

export function binEdges(axis: BinAxis): number[] {
  return Array.from({ length: axis.bins + 1 }, (_, i) => axisValue(i / axis.bins, axis));
}

export type Histogram1D = { counts: Float64Array; inRange: number; usable: number };

export function histogram1d(values: Float64Array, mask: Uint8Array | null, axis: BinAxis): Histogram1D {
  const counts = new Float64Array(axis.bins);
  let inRange = 0;
  let usableCount = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (mask && !mask[i]) continue;
    if (!usable(values[i], axis.scale)) continue;
    usableCount += 1;
    const b = binOf(values[i], axis);
    if (b < 0) continue;
    counts[b] += 1;
    inRange += 1;
  }
  return { counts, inRange, usable: usableCount };
}

export type Statistic = "count" | "median" | "mean";

export type Histogram2D = {
  nx: number;
  ny: number;
  /** Row-major [iy * nx + ix]. */
  counts: Float64Array;
  /** Per-bin statistic of z (NaN where undefined); equals counts for "count". */
  stat: Float64Array;
  considered: number;
  usableXY: number;
  inRange: number;
};

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function histogram2d(
  xs: Float64Array,
  ys: Float64Array,
  mask: Uint8Array | null,
  ax: BinAxis,
  ay: BinAxis,
  statistic: Statistic = "count",
  zs: Float64Array | null = null
): Histogram2D {
  const nx = ax.bins;
  const ny = ay.bins;
  const counts = new Float64Array(nx * ny);
  const buckets: number[][] | null = statistic === "count" || !zs ? null : Array.from({ length: nx * ny }, () => []);
  let considered = 0;
  let usableXY = 0;
  let inRange = 0;
  for (let i = 0; i < xs.length; i += 1) {
    if (mask && !mask[i]) continue;
    considered += 1;
    if (!usable(xs[i], ax.scale) || !usable(ys[i], ay.scale)) continue;
    usableXY += 1;
    const bx = binOf(xs[i], ax);
    const by = binOf(ys[i], ay);
    if (bx < 0 || by < 0) continue;
    inRange += 1;
    const k = by * nx + bx;
    counts[k] += 1;
    if (buckets && zs && Number.isFinite(zs[i])) buckets[k].push(zs[i]);
  }
  let stat: Float64Array;
  if (!buckets) {
    stat = counts;
  } else {
    stat = new Float64Array(nx * ny).fill(NaN);
    for (let k = 0; k < buckets.length; k += 1) {
      const b = buckets[k];
      if (b.length === 0) continue;
      stat[k] = statistic === "median" ? median(b) : b.reduce((s, v) => s + v, 0) / b.length;
    }
  }
  return { nx, ny, counts, stat, considered, usableXY, inRange };
}
