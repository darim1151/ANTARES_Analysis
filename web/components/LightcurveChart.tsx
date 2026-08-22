"use client";

import { line, scaleLinear } from "d3";
import type { LightcurveSample } from "@/types/skypulse";

type Props = {
  samples: LightcurveSample[];
  label: string;
};

const filterColor: Record<string, string> = {
  g: "#31D9FF",
  r: "#FFB84D",
  i: "#8F5BFF"
};

export default function LightcurveChart({ samples, label }: Props) {
  if (!samples.length) {
    return (
      <div className="lightcurve-empty">
        <strong>{label}</strong>
        <p>No public lightcurve sample is available for this object.</p>
      </div>
    );
  }

  const width = 420;
  const height = 220;
  const pad = 26;
  const mjds = samples.map((sample) => sample.mjd);
  const mags = samples.map((sample) => sample.magnitude);
  const x = scaleLinear()
    .domain([Math.min(...mjds), Math.max(...mjds)])
    .range([pad, width - pad]);
  const y = scaleLinear()
    .domain([Math.max(...mags) + 0.18, Math.min(...mags) - 0.18])
    .range([height - pad, pad]);
  const path = line<LightcurveSample>()
    .x((sample) => x(sample.mjd))
    .y((sample) => y(sample.magnitude))(samples);

  return (
    <section className="lightcurve-panel" aria-label="Brightness over time">
      <div className="panel-split-title">
        <strong>{label}</strong>
        <span>lower magnitude is brighter</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Lightcurve">
        <path className="lc-grid" d={`M ${pad} ${pad} H ${width - pad} M ${pad} ${height / 2} H ${width - pad} M ${pad} ${height - pad} H ${width - pad}`} />
        {path && <path className="lc-line" d={path} />}
        {samples.map((sample, index) => (
          <circle
            key={`${sample.mjd}-${index}`}
            cx={x(sample.mjd)}
            cy={y(sample.magnitude)}
            r="3.8"
            fill={filterColor[sample.filter] ?? "#F4F7FB"}
          />
        ))}
        <text x={pad} y={height - 6}>MJD {Math.min(...mjds).toFixed(1)}</text>
        <text x={width - pad - 72} y={height - 6}>MJD {Math.max(...mjds).toFixed(1)}</text>
      </svg>
    </section>
  );
}
