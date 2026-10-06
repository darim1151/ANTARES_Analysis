"use client";

import { useEffect, useRef, useState } from "react";
import type { DiaSourceRow, EvidenceClass, LightcurvePoint, TimeScale } from "@/types/observatory";
import { linearTicks } from "@/lib/observatory/axes";
import { fmtNum } from "@/lib/observatory/format";
import { BAND_COLOR, TOKENS } from "@/lib/observatory/theme";

const H = 176;
const M = { left: 48, right: 12, top: 22, bottom: 34 };

/** Compact tick text so labels never outgrow the left margin (fluxes reach 10^5 nJy). */
function tickText(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${fmtNum(v / 1e6, 3)}M`;
  if (a >= 1e4) return `${fmtNum(v / 1e3, 3)}k`;
  return fmtNum(v, 4);
}

type Point = { t: number; v: number; e: number | null; band: string };

function Plot({ points, xTitle, yTitle, reversedY, watermark }: { points: Point[]; xTitle: string; yTitle: string; reversedY: boolean; watermark?: string }) {
  // Drawn at the measured pixel width so text keeps its true size at any panel width.
  const ref = useRef<HTMLDivElement | null>(null);
  const [W, setW] = useState(360);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setW(Math.max(240, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const tMin = Math.min(...points.map((p) => p.t));
  const tMax = Math.max(...points.map((p) => p.t));
  const lo = Math.min(...points.map((p) => p.v - (p.e ?? 0)), reversedY ? Infinity : 0);
  const hi = Math.max(...points.map((p) => p.v + (p.e ?? 0)), reversedY ? -Infinity : 0);
  const padT = Math.max(0.5, (tMax - tMin) * 0.06);
  const padV = Math.max(1e-6, (hi - lo) * 0.08);
  const x0 = tMin - padT;
  const x1 = tMax + padT;
  const y0 = lo - padV;
  const y1 = hi + padV;
  const pw = W - M.left - M.right;
  const ph = H - M.top - M.bottom;
  const px = (t: number) => M.left + ((t - x0) / (x1 - x0)) * pw;
  const py = (v: number) => (reversedY ? M.top + ((v - y0) / (y1 - y0)) * ph : M.top + (1 - (v - y0) / (y1 - y0)) * ph);
  const bands = [...new Set(points.map((p) => p.band))].sort((a, b) => "ugrizy".indexOf(a) - "ugrizy".indexOf(b));
  // Keep only ticks whose centred label (~36 px) stays inside the plot.
  const xt = linearTicks(x0, x1, Math.max(3, Math.floor(pw / 90))).filter((t) => {
    const px0 = M.left + ((t - x0) / (x1 - x0)) * pw;
    return px0 >= M.left + 18 && px0 <= M.left + pw - 18;
  });
  const yt = linearTicks(y0, y1, 4);
  return (
    <div ref={ref} className="uso-lc-wrap">
    <svg width={W} height={H} className="uso-lc" role="img" aria-label={`${yTitle} versus ${xTitle}`}>
      <rect x={M.left} y={M.top} width={pw} height={ph} className="uso-labframe" />
      {watermark && (
        <text x={M.left + pw / 2} y={M.top + ph / 2 + 6} textAnchor="middle" className="uso-watermark">
          {watermark}
        </text>
      )}
      {!reversedY && y0 < 0 && y1 > 0 && <line x1={M.left} x2={M.left + pw} y1={py(0)} y2={py(0)} stroke={TOKENS.axis} />}
      <g className="uso-labaxis">
        {xt.map((t) => (
          <g key={t} transform={`translate(${px(t)},${M.top + ph})`}>
            <line y2={4} />
            <text y={14} textAnchor="middle">
              {fmtNum(t, 7)}
            </text>
          </g>
        ))}
        {yt.map((v) => (
          <g key={v} transform={`translate(${M.left},${py(v)})`}>
            <line x2={-4} />
            <text x={-6} y={3.5} textAnchor="end">
              {tickText(v)}
            </text>
          </g>
        ))}
        <text className="uso-axistitle" x={M.left + pw / 2} y={H - 3} textAnchor="middle">
          {xTitle}
        </text>
        <text className="uso-axistitle" transform={`translate(11,${M.top + ph / 2}) rotate(-90)`} textAnchor="middle">
          {yTitle}
        </text>
      </g>
      {points.map((p, i) => (
        <g key={i}>
          {p.e !== null && <line x1={px(p.t)} x2={px(p.t)} y1={py(p.v - p.e)} y2={py(p.v + p.e)} stroke={BAND_COLOR[p.band] ?? TOKENS.secondary} strokeOpacity={0.6} />}
          <circle cx={px(p.t)} cy={py(p.v)} r={3.2} fill={BAND_COLOR[p.band] ?? TOKENS.secondary} stroke="#131312" strokeWidth={1.5} />
        </g>
      ))}
      <g className="uso-lc-legend">
        {bands.map((b, i) => (
          <g key={b} transform={`translate(${M.left + 6 + i * 26},${M.top - 11})`}>
            <circle r={3.2} fill={BAND_COLOR[b] ?? TOKENS.secondary} />
            <text x={6} y={3.5}>
              {b}
            </text>
          </g>
        ))}
      </g>
    </svg>
    </div>
  );
}

export function DiaSourceFluxPlot({ sources }: { sources: DiaSourceRow[] }) {
  return (
    <Plot
      points={sources.map((s) => ({ t: s.midpointMjdTai, v: s.psfFlux, e: s.psfFluxErr, band: s.band }))}
      xTitle="midpointMjdTai [MJD, TAI]"
      yTitle="psfFlux [nJy] (difference)"
      reversedY={false}
    />
  );
}

/** Magnitude light curve; the watermark and time axis follow the payload's own evidence and scale. */
export function MagnitudePlot({ points, evidence, timeScale }: { points: LightcurvePoint[]; evidence: EvidenceClass; timeScale: TimeScale }) {
  const synthetic = evidence === "SYNTHETIC_DEMO" || evidence === "SYNTHETIC_FIXTURE";
  return (
    <Plot
      points={points.map((p) => ({ t: p.mjd, v: p.magnitude, e: null, band: p.band }))}
      xTitle={`MJD [${timeScale}]`}
      yTitle="magnitude"
      reversedY
      watermark={synthetic ? "SYNTHETIC · NOT PHOTOMETRY" : undefined}
    />
  );
}
