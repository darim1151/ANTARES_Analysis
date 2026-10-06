/** Axis ticks and pixel mapping for Lab plots (pure). */

import { axisFraction, axisValue, type BinAxis } from "./kernel/population.ts";

function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  const nice = unit >= 7.5 ? 10 : unit >= 3.5 ? 5 : unit >= 1.5 ? 2 : 1;
  return nice * power;
}

export function linearTicks(min: number, max: number, count: number): number[] {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, count);
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

export function logTicks(min: number, max: number): { major: number[]; minor: number[] } {
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const major: number[] = [];
  const minor: number[] = [];
  const decades = hi - lo;
  for (let e = lo; e <= hi; e += 1) {
    for (const m of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
      const v = m * 10 ** e;
      if (v < min * (1 - 1e-9) || v > max * (1 + 1e-9)) continue;
      if (m === 1 || (decades <= 2 && (m === 2 || m === 5))) major.push(v);
      else minor.push(v);
    }
  }
  if (major.length < 2) return { major: linearTicks(min, max, 4).filter((v) => v > 0), minor: [] };
  return { major, minor };
}

export type PlotBox = { left: number; top: number; width: number; height: number };

export type AxisMap = {
  axis: BinAxis;
  reversed: boolean;
  /** Value -> pixel (NaN when unusable). */
  toPx: (value: number) => number;
  /** Pixel -> value (clamped to the axis). */
  fromPx: (px: number) => number;
  /** Bin index -> [px0, px1] (ordered low to high pixel). */
  binPx: (bin: number) => [number, number];
  ticks: { major: number[]; minor: number[] };
};

export function xAxisMap(axis: BinAxis, reversed: boolean, box: PlotBox, tickCount: number): AxisMap {
  const toFrac = (f: number) => (reversed ? 1 - f : f);
  return {
    axis,
    reversed,
    toPx: (v) => box.left + toFrac(axisFraction(v, axis)) * box.width,
    fromPx: (px) => axisValue(toFrac(Math.max(0, Math.min(1, (px - box.left) / box.width))), axis),
    binPx: (b) => {
      const a = box.left + toFrac(b / axis.bins) * box.width;
      const c = box.left + toFrac((b + 1) / axis.bins) * box.width;
      return a < c ? [a, c] : [c, a];
    },
    ticks: axis.scale === "log" ? logTicks(axis.min, axis.max) : { major: linearTicks(axis.min, axis.max, tickCount), minor: [] }
  };
}

export function yAxisMap(axis: BinAxis, reversed: boolean, box: PlotBox, tickCount: number): AxisMap {
  const toFrac = (f: number) => (reversed ? f : 1 - f);
  return {
    axis,
    reversed,
    toPx: (v) => box.top + toFrac(axisFraction(v, axis)) * box.height,
    fromPx: (px) => axisValue(toFrac(Math.max(0, Math.min(1, (px - box.top) / box.height))), axis),
    binPx: (b) => {
      const a = box.top + toFrac(b / axis.bins) * box.height;
      const c = box.top + toFrac((b + 1) / axis.bins) * box.height;
      return a < c ? [a, c] : [c, a];
    },
    ticks: axis.scale === "log" ? logTicks(axis.min, axis.max) : { major: linearTicks(axis.min, axis.max, tickCount), minor: [] }
  };
}
