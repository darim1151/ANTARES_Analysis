"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight } from "lucide-react";
import type { FeatureDimension, FeatureFamily, LabStatistic } from "@/types/observatory";
import { histogram1d, histogram2d, robustExtent, type BinAxis } from "@/lib/observatory/kernel/population";
import { xAxisMap, yAxisMap, type PlotBox } from "@/lib/observatory/axes";
import { fmtInt, fmtLogTick, fmtNum, fmtPercent } from "@/lib/observatory/format";
import type { DomainModel } from "@/lib/observatory/model";
import { DOMAIN_COLOR, DOMAIN_RAMP, ENTITY_NOUN, rampColor, TOKENS, withAlpha } from "@/lib/observatory/theme";
import { useObservatory } from "./ObservatoryContext";
import { CapabilityMark, DomainBadge, EvidenceChip, Segmented, SelectField, type SelectOption } from "./ui";

const FAMILY_LABEL: Record<FeatureFamily, string> = {
  position: "Position",
  time: "Time",
  multiplicity: "Detection multiplicity",
  time_baseline: "Time baseline",
  photometry: "Photometry",
  colour: "Colour",
  variability: "Variability · lc_features",
  model_output: "Model outputs · scores",
  crossmatch: "Crossmatch"
};

function dimOptions(dims: FeatureDimension[]): SelectOption[] {
  return dims.map((d) => ({
    value: d.id,
    group: FAMILY_LABEL[d.family],
    disabled: d.state === "UNAVAILABLE",
    // Lead with the short symbol so a truncated select still identifies the axis.
    label: `${d.short} · ${d.label}${d.unit ? ` [${d.unit}]` : ""}${d.state === "PARTIALLY_QUALIFIED" ? "  ◐" : ""}${d.state === "UNAVAILABLE" ? "  — unavailable" : ""}`
  }));
}

/** Long axis title when it fits the available length (~6 px per character), else the short symbol. */
function axisTitle(d: FeatureDimension, compact: boolean, room: number, direction: string) {
  const suffix = `${d.unit ? ` [${d.unit}]` : ""}${d.scale === "log" ? " · log" : ""}${d.reversed ? direction : ""}`;
  const long = `${d.label}${suffix}`;
  return !compact && long.length * 6 <= room ? long : `${d.short}${suffix}`;
}

export function DimensionDefinition({ dim, role }: { dim: FeatureDimension; role: string }) {
  return (
    <div className="uso-def">
      <div className="uso-def-head">
        <b>{role}</b>
        <span className="uso-def-label">{dim.label}</span>
        <CapabilityMark state={dim.state} />
        {dim.evidence && <EvidenceChip evidence={dim.evidence} compact />}
      </div>
      <p>{dim.definition}</p>
      {dim.snapshot && <p className="uso-def-snap">Snapshot · {dim.snapshot}</p>}
      {dim.calibrated === false && <p className="uso-def-snap">Calibrated probability · no</p>}
      {dim.qualifications.map((q) => (
        <p key={q} className="uso-def-q">
          ◐ {q}
        </p>
      ))}
    </div>
  );
}

export default function LabPlot({
  domain,
  compact,
  definitions,
  sharedExtents
}: {
  domain: DomainModel;
  compact: boolean;
  /** "side": always-open definitions column; "collapsed": a disclosure under the plot. */
  definitions: "side" | "collapsed";
  sharedExtents: { x: [number, number] | null; y: [number, number] | null };
}) {
  const { state, dispatch, masks } = useObservatory();
  const d = domain.id;
  const config = state.lens.lab[d];
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [size, setSize] = useState({ width: 480, height: 320 });
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const dragRef = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const [hoverBin, setHoverBin] = useState<{ ix: number; iy: number; px: number; py: number } | null>(null);

  const dimX = domain.dims.get(config.x)!;
  const dimY = domain.dims.get(config.y)!;
  const dimZ = config.z ? domain.dims.get(config.z) ?? null : null;
  const xs = domain.cols.features.get(config.x)!;
  const ys = domain.cols.features.get(config.y)!;
  const zs = config.z ? domain.cols.features.get(config.z) ?? null : null;
  const mask = masks[d];
  const contextActive = Boolean(state.selection.time || state.selection.sky);
  const predicate = state.selection.feature[d];
  const brush = predicate && predicate.x.dimension === config.x && predicate.y?.dimension === config.y ? predicate : null;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setSize({ width: Math.max(220, Math.floor(entry.contentRect.width)), height: Math.max(150, Math.floor(entry.contentRect.height)) }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const marg = compact ? 22 : 34;
  const box: PlotBox = useMemo(() => {
    const left = compact ? 44 : 54;
    const bottom = compact ? 30 : 38;
    const top = marg + 6;
    return { left, top, width: Math.max(60, size.width - left - marg - 10), height: Math.max(60, size.height - top - bottom) };
  }, [compact, marg, size]);
  const nx = Math.max(12, Math.min(56, Math.round(box.width / 13)));
  const ny = Math.max(10, Math.min(44, Math.round(box.height / 13)));
  const ax: BinAxis = useMemo(() => {
    const [min, max] = sharedExtents.x ?? robustExtent(xs, dimX.scale, dimX.extent);
    return { scale: dimX.scale, min, max, bins: nx };
  }, [dimX, nx, sharedExtents.x, xs]);
  const ay: BinAxis = useMemo(() => {
    const [min, max] = sharedExtents.y ?? robustExtent(ys, dimY.scale, dimY.extent);
    return { scale: dimY.scale, min, max, bins: ny };
  }, [dimY, ny, sharedExtents.y, ys]);
  const mx = useMemo(() => xAxisMap(ax, dimX.reversed, box, compact ? 4 : 7), [ax, box, compact, dimX.reversed]);
  const my = useMemo(() => yAxisMap(ay, dimY.reversed, box, compact ? 4 : 6), [ay, box, compact, dimY.reversed]);

  const subsetMask = contextActive ? mask.exceptFeature : null;
  const full = useMemo(() => histogram2d(xs, ys, null, ax, ay, "count"), [ax, ay, xs, ys]);
  const h2 = useMemo(() => histogram2d(xs, ys, subsetMask, ax, ay, config.statistic, zs), [ax, ay, config.statistic, subsetMask, xs, ys, zs]);
  const margX = useMemo(() => ({ full: histogram1d(xs, null, ax), sub: histogram1d(xs, subsetMask, ax), brushed: brush ? histogram1d(xs, mask.all, ax) : null }), [ax, brush, mask.all, subsetMask, xs]);
  const margY = useMemo(() => ({ full: histogram1d(ys, null, ay), sub: histogram1d(ys, subsetMask, ay), brushed: brush ? histogram1d(ys, mask.all, ay) : null }), [ay, brush, mask.all, subsetMask, ys]);
  const zExtent = useMemo(() => (zs && dimZ ? robustExtent(zs, dimZ.scale, dimZ.extent) : null), [dimZ, zs]);
  const maxCount = Math.max(1, ...full.counts);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.width * dpr);
    canvas.height = Math.floor(size.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.fillStyle = "#0f0f0e";
    ctx.fillRect(box.left, box.top, box.width, box.height);
    const ramp = DOMAIN_RAMP[d];
    for (let iy = 0; iy < ny; iy += 1) {
      const [y0, y1] = my.binPx(iy);
      for (let ix = 0; ix < nx; ix += 1) {
        const k = iy * nx + ix;
        const [x0, x1] = mx.binPx(ix);
        if (contextActive && full.counts[k] > 0) {
          ctx.fillStyle = "rgba(242, 241, 236, 0.08)";
          ctx.fillRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
        }
        if (h2.counts[k] <= 0) continue;
        let t: number;
        if (config.statistic === "count") t = 0.14 + 0.86 * (Math.log(1 + h2.counts[k]) / Math.log(1 + maxCount));
        else {
          const v = h2.stat[k];
          if (!Number.isFinite(v) || !zExtent || !dimZ) {
            ctx.fillStyle = "rgba(195, 194, 183, 0.25)";
            ctx.fillRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
            continue;
          }
          const f =
            dimZ.scale === "log"
              ? (Math.log10(Math.max(v, zExtent[0])) - Math.log10(zExtent[0])) / (Math.log10(zExtent[1]) - Math.log10(zExtent[0]))
              : (v - zExtent[0]) / (zExtent[1] - zExtent[0]);
          t = 0.14 + 0.86 * Math.max(0, Math.min(1, f));
        }
        ctx.fillStyle = rampColor(ramp, t);
        ctx.fillRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1);
      }
    }
    if (brush) {
      // Recede everything outside the brushed region.
      const bx0 = mx.toPx(brush.x.min);
      const bx1 = mx.toPx(brush.x.max);
      const by0 = my.toPx(brush.y!.min);
      const by1 = my.toPx(brush.y!.max);
      const path = new Path2D();
      path.rect(box.left, box.top, box.width, box.height);
      path.rect(Math.min(bx0, bx1), Math.min(by0, by1), Math.abs(bx1 - bx0), Math.abs(by1 - by0));
      ctx.fillStyle = "rgba(11, 11, 10, 0.55)";
      ctx.fill(path, "evenodd");
    }
  }, [box, brush, config.statistic, contextActive, d, dimZ, full, h2, maxCount, mx, my, nx, ny, size, zExtent]);

  function binAt(px: number, py: number) {
    if (px < box.left || px > box.left + box.width || py < box.top || py > box.top + box.height) return null;
    for (let ix = 0; ix < nx; ix += 1) {
      const [a, b] = mx.binPx(ix);
      if (px >= a && px < b) {
        for (let iy = 0; iy < ny; iy += 1) {
          const [c, e] = my.binPx(iy);
          if (py >= c && py < e) return { ix, iy };
        }
      }
    }
    return null;
  }

  function local(e: React.PointerEvent) {
    const r = wrapRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  function commitBrush(x0: number, y0: number, x1: number, y1: number) {
    const a = mx.fromPx(x0);
    const b = mx.fromPx(x1);
    const c = my.fromPx(y0);
    const e = my.fromPx(y1);
    dispatch({
      type: "feature",
      domain: d,
      predicate: {
        x: { dimension: config.x, min: Math.min(a, b), max: Math.max(a, b) },
        y: { dimension: config.y, min: Math.min(c, e), max: Math.max(c, e) }
      }
    });
  }

  const options = dimOptions(domain.bundle.features.dimensions);
  const zOptions = options.filter((o) => !o.disabled);
  const coverage = h2.considered ? h2.usableXY / h2.considered : NaN;
  const noun = ENTITY_NOUN[domain.bundle.entities.entity_kind];
  const hk = hoverBin ? hoverBin.iy * nx + hoverBin.ix : null;
  const color = DOMAIN_COLOR[d];

  const marginalBars = (counts: Float64Array, along: "x" | "y", fill: string, peak: number) =>
    Array.from(counts, (c, i) => {
      if (c <= 0) return null;
      const len = (c / peak) * (marg - 4);
      if (along === "x") {
        const [a, b] = mx.binPx(i);
        return <rect key={i} x={a + 0.5} width={Math.max(0.5, b - a - 1)} y={box.top - 3 - len} height={len} fill={fill} />;
      }
      const [a, b] = my.binPx(i);
      return <rect key={i} y={a + 0.5} height={Math.max(0.5, b - a - 1)} x={box.left + box.width + 3} width={len} fill={fill} />;
    });

  return (
    <div className={`uso-lab-plot${compact ? " is-compact" : ""}`}>
      <div className="uso-lab-controls">
        <DomainBadge domain={d} />
        <SelectField label="X" value={config.x} options={options} onChange={(x) => dispatch({ type: "lab", domain: d, patch: { x } })} wide />
        <button
          type="button"
          className="uso-iconbtn"
          aria-label="Swap X and Y"
          title="Swap X and Y"
          onClick={() => dispatch({ type: "lab", domain: d, patch: { x: config.y, y: config.x } })}
        >
          <ArrowLeftRight aria-hidden="true" />
        </button>
        <SelectField label="Y" value={config.y} options={options} onChange={(y) => dispatch({ type: "lab", domain: d, patch: { y } })} wide />
        <Segmented<LabStatistic>
          label="Statistic"
          value={config.statistic}
          options={[
            { value: "count", label: "Count" },
            { value: "median", label: "Median" },
            { value: "mean", label: "Mean" }
          ]}
          onChange={(statistic) =>
            dispatch({
              type: "lab",
              domain: d,
              patch: statistic === "count" ? { statistic, z: null } : { statistic, z: config.z ?? zOptions.find((o) => o.value !== config.x && o.value !== config.y)?.value ?? null }
            })
          }
        />
        {config.statistic !== "count" && (
          <SelectField label="Z" value={config.z ?? ""} options={zOptions} onChange={(z) => dispatch({ type: "lab", domain: d, patch: { z } })} wide />
        )}
      </div>
      <div
        className="uso-lab-canvas"
        ref={wrapRef}
        onPointerDown={(e) => {
          const p = local(e);
          if (!binAt(p.x, p.y)) return;
          (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
          dragRef.current = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
          setDrag(dragRef.current);
        }}
        onPointerMove={(e) => {
          const p = local(e);
          const b = binAt(p.x, p.y);
          setHoverBin(b ? { ...b, px: p.x, py: p.y } : null);
          if (dragRef.current) {
            dragRef.current = { ...dragRef.current, x1: Math.max(box.left, Math.min(box.left + box.width, p.x)), y1: Math.max(box.top, Math.min(box.top + box.height, p.y)) };
            setDrag(dragRef.current);
          }
        }}
        onPointerLeave={() => setHoverBin(null)}
        onPointerUp={() => {
          const drag = dragRef.current;
          dragRef.current = null;
          if (drag) {
            if (Math.abs(drag.x1 - drag.x0) > 4 && Math.abs(drag.y1 - drag.y0) > 4) commitBrush(drag.x0, drag.y0, drag.x1, drag.y1);
            else {
              const b = binAt(drag.x0, drag.y0);
              if (b) {
                const [xa, xb] = mx.binPx(b.ix);
                const [ya, yb] = my.binPx(b.iy);
                commitBrush(xa, ya, xb, yb);
              }
            }
          }
          setDrag(null);
        }}
        onDoubleClick={() => dispatch({ type: "feature", domain: d, predicate: null })}
        role="img"
        aria-label={`${d.toUpperCase()} population: ${dimY.label} versus ${dimX.label}. Drag to brush a region; double-click to clear.`}
      >
        <canvas ref={canvasRef} style={{ width: size.width, height: size.height }} />
        <svg width={size.width} height={size.height} aria-hidden="true">
          <g className="uso-marginal">
            {marginalBars(margX.full.counts, "x", "rgba(195, 194, 183, 0.18)", Math.max(1, ...margX.full.counts))}
            {contextActive && marginalBars(margX.sub.counts, "x", withAlpha(color, 0.75), Math.max(1, ...margX.full.counts))}
            {margX.brushed && marginalBars(margX.brushed.counts, "x", TOKENS.primary, Math.max(1, ...margX.full.counts))}
            {marginalBars(margY.full.counts, "y", "rgba(195, 194, 183, 0.18)", Math.max(1, ...margY.full.counts))}
            {contextActive && marginalBars(margY.sub.counts, "y", withAlpha(color, 0.75), Math.max(1, ...margY.full.counts))}
            {margY.brushed && marginalBars(margY.brushed.counts, "y", TOKENS.primary, Math.max(1, ...margY.full.counts))}
          </g>
          <g className="uso-labaxis">
            <rect x={box.left} y={box.top} width={box.width} height={box.height} className="uso-labframe" />
            {mx.ticks.major.map((v) => {
              const px = mx.toPx(v);
              return (
                <g key={`x${v}`} transform={`translate(${px},${box.top + box.height})`}>
                  <line y2={4} />
                  <text y={15} textAnchor="middle">
                    {ax.scale === "log" ? fmtLogTick(v) : fmtNum(v, 3)}
                  </text>
                </g>
              );
            })}
            {mx.ticks.minor.map((v) => (
              <line key={`xm${v}`} x1={mx.toPx(v)} x2={mx.toPx(v)} y1={box.top + box.height} y2={box.top + box.height + 2} />
            ))}
            {my.ticks.major.map((v) => {
              const py = my.toPx(v);
              return (
                <g key={`y${v}`} transform={`translate(${box.left},${py})`}>
                  <line x2={-4} />
                  <text x={-7} y={3.5} textAnchor="end">
                    {ay.scale === "log" ? fmtLogTick(v) : fmtNum(v, 3)}
                  </text>
                </g>
              );
            })}
            {my.ticks.minor.map((v) => (
              <line key={`ym${v}`} x1={box.left} x2={box.left - 2} y1={my.toPx(v)} y2={my.toPx(v)} />
            ))}
            <text className="uso-axistitle" x={box.left + box.width / 2} y={size.height - 3} textAnchor="middle">
              {axisTitle(dimX, compact, box.width, " · brighter →")}
            </text>
            <text className="uso-axistitle" transform={`translate(${compact ? 10 : 12},${box.top + box.height / 2}) rotate(-90)`} textAnchor="middle">
              {axisTitle(dimY, compact, box.height, " · brighter ↑")}
            </text>
          </g>
          {brush && brush.y && (
            <rect
              className="uso-brush"
              x={Math.min(mx.toPx(brush.x.min), mx.toPx(brush.x.max))}
              y={Math.min(my.toPx(brush.y.min), my.toPx(brush.y.max))}
              width={Math.abs(mx.toPx(brush.x.max) - mx.toPx(brush.x.min))}
              height={Math.abs(my.toPx(brush.y.max) - my.toPx(brush.y.min))}
            />
          )}
          {drag && <rect className="uso-brush is-preview" x={Math.min(drag.x0, drag.x1)} y={Math.min(drag.y0, drag.y1)} width={Math.abs(drag.x1 - drag.x0)} height={Math.abs(drag.y1 - drag.y0)} />}
          {hoverBin && !drag && (
            <rect className="uso-binhover" x={mx.binPx(hoverBin.ix)[0]} y={my.binPx(hoverBin.iy)[0]} width={mx.binPx(hoverBin.ix)[1] - mx.binPx(hoverBin.ix)[0]} height={my.binPx(hoverBin.iy)[1] - my.binPx(hoverBin.iy)[0]} />
          )}
          {state.focus?.domain === d &&
            (() => {
              const i = domain.indexById.get(state.focus.id);
              if (i === undefined) return null;
              const fx = mx.toPx(xs[i]);
              const fy = my.toPx(ys[i]);
              if (!Number.isFinite(fx) || !Number.isFinite(fy)) return null;
              return (
                <g className="uso-labfocus">
                  <line x1={box.left} x2={box.left + box.width} y1={fy} y2={fy} />
                  <line y1={box.top} y2={box.top + box.height} x1={fx} x2={fx} />
                  <circle cx={fx} cy={fy} r={5} style={{ stroke: color }} />
                </g>
              );
            })()}
        </svg>
        {hoverBin && hk !== null && !drag && (
          <div className="uso-tip" style={{ left: Math.min(size.width - 230, hoverBin.px + 14), top: Math.max(0, hoverBin.py - 70) }} role="tooltip">
            <strong>
              {dimX.short} {fmtNum(Math.min(...mx.binPx(hoverBin.ix).map(mx.fromPx)), 3)}–{fmtNum(Math.max(...mx.binPx(hoverBin.ix).map(mx.fromPx)), 3)} · {dimY.short}{" "}
              {fmtNum(Math.min(...my.binPx(hoverBin.iy).map(my.fromPx)), 3)}–{fmtNum(Math.max(...my.binPx(hoverBin.iy).map(my.fromPx)), 3)}
            </strong>
            <span>
              {fmtInt(h2.counts[hk])} {noun.many}
              {contextActive ? ` in cross-filter (of ${fmtInt(full.counts[hk])})` : ""}
            </span>
            {config.statistic !== "count" && dimZ && (
              <span>
                {config.statistic} {dimZ.short}: {fmtNum(h2.stat[hk], 3)}
              </span>
            )}
            <small>Click to brush this bin · drag for a region</small>
          </div>
        )}
      </div>
      <div className="uso-lab-readout" aria-live="polite">
        <span>
          <b>N</b> {fmtInt(h2.considered)}
          {contextActive ? ` of ${fmtInt(mask.counts.total)} in cross-filter` : ` ${noun.many}`}
        </span>
        <span title="Entities with a defined value on both axes (log axes exclude non-positive values)">
          <b>Coverage</b> {fmtInt(h2.usableXY)} ({fmtPercent(coverage)})
        </span>
        <span>
          <b>On axes</b> {fmtInt(h2.inRange)}
        </span>
        {brush ? (
          <span>
            <b>Brushed</b> {fmtInt(mask.counts.all)}
            <button type="button" className="uso-linkbtn" onClick={() => dispatch({ type: "feature", domain: d, predicate: null })}>
              clear
            </button>
          </span>
        ) : (
          <span className="uso-hint">Drag to brush · click a bin</span>
        )}
        {config.statistic === "count" ? (
          <span className="uso-lab-scale">
            <i style={{ background: `linear-gradient(90deg, ${rampColor(DOMAIN_RAMP[d], 0.14)}, ${rampColor(DOMAIN_RAMP[d], 1)})` }} aria-hidden="true" /> 1–{fmtInt(maxCount)} per bin · log
          </span>
        ) : (
          dimZ &&
          zExtent && (
            <span className="uso-lab-scale">
              <i style={{ background: `linear-gradient(90deg, ${rampColor(DOMAIN_RAMP[d], 0.14)}, ${rampColor(DOMAIN_RAMP[d], 1)})` }} aria-hidden="true" /> {config.statistic} {dimZ.short} {fmtNum(zExtent[0], 3)}–{fmtNum(zExtent[1], 3)}
            </span>
          )
        )}
      </div>
      {definitions === "collapsed" ? (
        <details className="uso-lab-defs">
          <summary>Definitions &amp; qualifications</summary>
          <DimensionDefinition dim={dimX} role="X" />
          <DimensionDefinition dim={dimY} role="Y" />
          {dimZ && config.statistic !== "count" && <DimensionDefinition dim={dimZ} role="Z" />}
        </details>
      ) : (
        <div className="uso-lab-defs is-open">
          <DimensionDefinition dim={dimX} role="X" />
          <DimensionDefinition dim={dimY} role="Y" />
          {dimZ && config.statistic !== "count" && <DimensionDefinition dim={dimZ} role="Z" />}
        </div>
      )}
    </div>
  );
}
