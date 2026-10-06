"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { DomainId } from "@/types/observatory";
import { angularSeparationDeg, eclipticCurve, galacticLatitudeCurve, GALACTIC_CENTER_ICRS, icrsToEcliptic, icrsToGalactic } from "@/lib/observatory/kernel/astro";
import { degradePix, pixelAreaDeg2, radecToPix } from "@/lib/observatory/kernel/healpix";
import { projectMollweide, unprojectMollweide } from "@/lib/observatory/kernel/projection";
import { ENTITY_INDEX_ORDER } from "@/lib/observatory/kernel/selection";
import { fmtFixed, fmtInt, fmtNum } from "@/lib/observatory/format";
import { capability, entityRef, isUsable, type DomainModel } from "@/lib/observatory/model";
import {
  CENTER_RA,
  cellPath,
  cellRings,
  ellipsePath,
  fitFrame,
  meridian,
  parallel,
  projectCurve,
  ringToSvg,
  smallCircle,
  toPx,
  type Frame
} from "@/lib/observatory/skyGeometry";
import { DOMAIN_COLOR, DOMAIN_RAMP, ENTITY_NOUN, rampColor, TOKENS } from "@/lib/observatory/theme";
import { useObservatory, useSkyHover } from "./ObservatoryContext";

type Pointer = { x: number; y: number; ra: number; dec: number } | null;

const GALACTIC_PLANE = galacticLatitudeCurve(0, 721);
const ECLIPTIC = eclipticCurve(721);
const HALO = "uso-halo";
const MERIDIANS = [0, 30, 60, 90, 120, 150, 210, 240, 270, 300, 330].map((ra) => meridian(ra));
const PARALLELS = [-60, -30, 0, 30, 60].map((dec) => ({ dec, curve: parallel(dec) }));
// 12h sits on the seam at both limb edges, where it would collide with Dec labels.
const RA_LABELS = [0, 60, 120, 240, 300];

/** Full-population maximum per order, so filtering dims rather than renormalizes. */
export function densityScale(domain: DomainModel, order: number) {
  const full = domain.densityAt(order);
  const max = Math.max(1, ...full.values());
  const t = (count: number) => 0.12 + 0.88 * (Math.log(1 + count) / Math.log(1 + max));
  return { full, max, t };
}

export default function SkyMap({
  domain,
  compact,
  unadmitted = null
}: {
  domain: DomainModel;
  compact: boolean;
  /** Why the selected dates have no counts for this domain; null when they do. */
  unadmitted?: string | null;
}) {
  const { model, state, dispatch, masks } = useObservatory();
  const { hover, setHover } = useSkyHover();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const clipId = useId().replace(/:/g, "");
  const [size, setSize] = useState({ width: 800, height: 400 });
  const [pointer, setPointer] = useState<Pointer>(null);
  const [down, setDown] = useState<Pointer>(null);
  // The gesture origin lives in a ref: pointerup can fire before a re-render.
  const downRef = useRef<Pointer>(null);
  const coneRef = useRef<number | null>(null);
  const [cone, setCone] = useState<number | null>(null);
  const [nearest, setNearest] = useState<number | null>(null);

  const d: DomainId = domain.id;
  const order = state.presentation.skyOrder;
  const layer = state.presentation.skyLayer;
  const overlays = state.presentation.overlays;
  const mask = masks[d];
  const filteredCap = capability(model, d, "sky.filtered_density");
  const filterActive = Boolean(state.selection.time || mask.featureActive) && isUsable(filteredCap.state);
  const frame: Frame = useMemo(() => fitFrame(size.width, size.height, compact ? 26 : 38, compact ? 6 : 10), [compact, size]);
  const scale = useMemo(() => densityScale(domain, order), [domain, order]);
  const coverage = domain.coverageAt(order);

  const filtered = useMemo(() => {
    if (!filterActive) return null;
    const out = new Map<number, number>();
    const m = mask.exceptSky;
    for (let i = 0; i < domain.cols.n; i += 1) {
      if (!m[i]) continue;
      const p = degradePix(domain.cols.hpx[i], ENTITY_INDEX_ORDER, order);
      out.set(p, (out.get(p) ?? 0) + 1);
    }
    return out;
  }, [domain, filterActive, mask.exceptSky, order]);

  const projected = useMemo(() => {
    const xs = new Float32Array(domain.cols.n);
    const ys = new Float32Array(domain.cols.n);
    for (let i = 0; i < domain.cols.n; i += 1) {
      const p = projectMollweide(domain.cols.ra[i], domain.cols.dec[i], CENTER_RA);
      xs[i] = p.x;
      ys[i] = p.y;
    }
    return { xs, ys };
  }, [domain]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      setSize({ width: Math.max(200, Math.floor(entry.contentRect.width)), height: Math.max(110, Math.floor(entry.contentRect.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.width * dpr);
    canvas.height = Math.floor(size.height * dpr);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Projection units -> device pixels (y up).
    ctx.setTransform(dpr * frame.scale, 0, 0, -dpr * frame.scale, dpr * frame.cx, dpr * frame.cy);
    const ellipse = new Path2D();
    ellipsePath().forEach(([x, y], i) => (i ? ellipse.lineTo(x, y) : ellipse.moveTo(x, y)));
    ellipse.closePath();
    ctx.fillStyle = "#0d0d0c";
    ctx.fill(ellipse);
    ctx.save();
    ctx.clip(ellipse);
    if (coverage) {
      ctx.fillStyle = TOKENS.coverage;
      for (const p of coverage) ctx.fill(cellPath(order, p));
    }
    // Dates that are not admitted have no counts; draw no density rather than an empty (zero) map.
    if (layer === "density" && !unadmitted) {
      const ramp = DOMAIN_RAMP[d];
      if (filtered) {
        ctx.fillStyle = "rgba(242, 241, 236, 0.09)";
        for (const p of scale.full.keys()) ctx.fill(cellPath(order, p));
        for (const [p, c] of filtered) {
          ctx.fillStyle = rampColor(ramp, scale.t(c));
          ctx.fill(cellPath(order, p));
        }
      } else {
        for (const [p, c] of scale.full) {
          ctx.fillStyle = rampColor(ramp, scale.t(c));
          ctx.fill(cellPath(order, p));
        }
      }
    }
    ctx.restore();
    if (layer === "entities" && !unadmitted) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const all = mask.all;
      const subset = Boolean(state.selection.time || state.selection.sky || mask.featureActive);
      for (const pass of [0, 1]) {
        for (let i = 0; i < domain.cols.n; i += 1) {
          const on = !subset || all[i] === 1;
          if ((pass === 0) === on) continue;
          const [px, py] = toPx(frame, projected.xs[i], projected.ys[i]);
          ctx.beginPath();
          ctx.arc(px, py, on ? (compact ? 1.5 : 2.1) : 1.2, 0, Math.PI * 2);
          ctx.fillStyle = on ? DOMAIN_COLOR[d] : "rgba(195, 194, 183, 0.22)";
          ctx.fill();
        }
      }
    }
  }, [compact, coverage, d, domain, filtered, frame, layer, mask, order, projected, scale, size, state.selection, unadmitted]);

  function locate(event: React.PointerEvent): Pointer {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const sky = unprojectMollweide((x - frame.cx) / frame.scale, (frame.cy - y) / frame.scale, CENTER_RA);
    return sky ? { x, y, ra: sky.ra, dec: sky.dec } : { x, y, ra: NaN, dec: NaN };
  }

  function nearestEntity(x: number, y: number): number | null {
    let best = -1;
    let bestDist = 10;
    for (let i = 0; i < domain.cols.n; i += 1) {
      const [px, py] = toPx(frame, projected.xs[i], projected.ys[i]);
      const dist = Math.hypot(px - x, py - y);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    }
    return best >= 0 ? best : null;
  }

  const onSky = pointer && Number.isFinite(pointer.ra);
  const hoverPixel = onSky && layer === "density" ? radecToPix(order, pointer!.ra, pointer!.dec) : null;
  const hoverCount = hoverPixel !== null ? (filtered ?? scale.full).get(hoverPixel) ?? 0 : 0;
  const hoverCovered = hoverPixel !== null && coverage ? coverage.has(hoverPixel) : null;
  const area = pixelAreaDeg2(order);
  const sky = state.selection.sky;
  const focus = state.focus?.domain === d ? domain.indexById.get(state.focus.id) ?? null : null;
  const noun = ENTITY_NOUN[domain.bundle.entities.entity_kind];
  const coverageSynthetic = Boolean(domain.bundle.sky.coverage?.evidence.some((e) => e === "SYNTHETIC_FIXTURE" || e === "SYNTHETIC_DEMO"));

  const curveSvg = (points: Array<[number, number]>) => projectCurve(points).map((seg) => ringToSvg(frame, seg)).join("");
  const gc = toPx(frame, projectMollweide(GALACTIC_CENTER_ICRS[0], GALACTIC_CENTER_ICRS[1], CENTER_RA).x, projectMollweide(GALACTIC_CENTER_ICRS[0], GALACTIC_CENTER_ICRS[1], CENTER_RA).y);

  return (
    <div
      className="uso-skymap"
      ref={wrapRef}
      onPointerMove={(e) => {
        const p = locate(e);
        setPointer(p);
        if (p && Number.isFinite(p.ra)) setHover({ ra: p.ra, dec: p.dec, source: d });
        else setHover(null);
        if (layer === "entities" && p) setNearest(nearestEntity(p.x, p.y));
        const origin = downRef.current;
        if (origin && p && Number.isFinite(p.ra) && Math.hypot(p.x - origin.x, p.y - origin.y) > 6) {
          coneRef.current = angularSeparationDeg(origin.ra, origin.dec, p.ra, p.dec);
          setCone(coneRef.current);
        }
      }}
      onPointerLeave={() => {
        setPointer(null);
        setHover(null);
        setNearest(null);
      }}
      onPointerDown={(e) => {
        const p = locate(e);
        if (p && Number.isFinite(p.ra)) {
          (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
          downRef.current = p;
          coneRef.current = null;
          setDown(p);
          setCone(null);
        }
      }}
      onPointerUp={(e) => {
        const p = locate(e);
        const origin = downRef.current;
        const radius = coneRef.current;
        downRef.current = null;
        coneRef.current = null;
        if (origin && radius !== null && radius >= 0.25) {
          dispatch({ type: "sky", sky: { kind: "cone", ra: Number(origin.ra.toFixed(4)), dec: Number(origin.dec.toFixed(4)), radius_deg: Number(radius.toFixed(3)) } });
        } else if (origin && p && Number.isFinite(p.ra)) {
          const idx = layer === "entities" ? nearestEntity(p.x, p.y) : null;
          if (idx !== null) dispatch({ type: "focus", focus: entityRef(domain, idx) });
          else dispatch({ type: "skyCell", order, pixel: radecToPix(order, p.ra, p.dec), additive: e.shiftKey || e.metaKey || e.ctrlKey });
        }
        setDown(null);
        setCone(null);
      }}
      role="img"
      aria-label={`${domain.id.toUpperCase()} sky in Mollweide equal-area projection, HEALPix order ${order}. Click a cell to select it, shift-click to add, drag for a cone.`}
    >
      <canvas ref={canvasRef} style={{ width: size.width, height: size.height }} />
      {unadmitted && <div className="uso-unadmitted-veil">{`${domain.id.toUpperCase()}: ${unadmitted}. No counts exist, which is not zero.`}</div>}
      <svg width={size.width} height={size.height} className="uso-skysvg" aria-hidden="true">
        <defs>
          <clipPath id={`clip-${clipId}`}>
            <path d={ringToSvg(frame, ellipsePath(), true)} />
          </clipPath>
        </defs>
        <g clipPath={`url(#clip-${clipId})`}>
          {overlays.graticule && (
            <g className="uso-graticule">
              {MERIDIANS.map((m, i) => (
                <path key={`m${i}`} d={curveSvg(m)} />
              ))}
              {PARALLELS.map((p) => (
                <path key={`p${p.dec}`} d={curveSvg(p.curve)} className={p.dec === 0 ? "is-equator" : undefined} />
              ))}
            </g>
          )}
          {/* Dark halos keep the reference curves legible over any density ramp. */}
          {overlays.ecliptic && <path className={HALO} d={curveSvg(ECLIPTIC)} />}
          {overlays.galacticPlane && <path className={HALO} d={curveSvg(GALACTIC_PLANE)} />}
          {overlays.ecliptic && <path className="uso-ecliptic" d={curveSvg(ECLIPTIC)} />}
          {overlays.galacticPlane && <path className="uso-galactic" d={curveSvg(GALACTIC_PLANE)} />}
          {sky?.kind === "healpix" &&
            sky.pixels.map((p) => (
              <path key={`s${p}`} className="uso-skysel" d={cellRings(sky.order, p).map((r) => ringToSvg(frame, r, true)).join("")} />
            ))}
          {sky?.kind === "cone" && <path className="uso-skysel is-cone" d={curveSvg(smallCircle(sky.ra, sky.dec, sky.radius_deg))} />}
          {down && cone !== null && <path className="uso-skysel is-preview" d={curveSvg(smallCircle(down.ra, down.dec, cone))} />}
          {hoverPixel !== null && !down && <path className="uso-skyhover" d={cellRings(order, hoverPixel).map((r) => ringToSvg(frame, r, true)).join("")} />}
        </g>
        <path className="uso-ellipse" d={ringToSvg(frame, ellipsePath(), true)} />
        {overlays.graticule && !compact && (
          <g className="uso-skylabels">
            {RA_LABELS.map((ra) => {
              const p = projectMollweide(ra, 0, CENTER_RA);
              const [px, py] = toPx(frame, p.x, p.y);
              return (
                <text key={ra} x={px} y={py - 4} textAnchor="middle">
                  {ra / 15}h
                </text>
              );
            })}
            {PARALLELS.filter((p) => p.dec !== 0).map(({ dec }) => {
              const p = projectMollweide(CENTER_RA + 179.999, dec, CENTER_RA);
              const [px, py] = toPx(frame, p.x, p.y);
              return (
                <text key={dec} x={px - 5} y={py + 3} textAnchor="end">
                  {dec > 0 ? `+${dec}°` : `${dec}°`}
                </text>
              );
            })}
          </g>
        )}
        {overlays.galacticPlane && (
          <g className="uso-gc">
            <circle cx={gc[0]} cy={gc[1]} r={compact ? 3 : 4.5} />
            {!compact && (
              <text x={gc[0] + 7} y={gc[1] + 4}>
                GC
              </text>
            )}
          </g>
        )}
        {focus !== null && (
          <g className="uso-focusmark">
            {(() => {
              const [px, py] = toPx(frame, projected.xs[focus], projected.ys[focus]);
              return (
                <>
                  <circle cx={px} cy={py} r={9} className="ring-outer" />
                  <circle cx={px} cy={py} r={9} style={{ stroke: DOMAIN_COLOR[d] }} className="ring-inner" />
                  <line x1={px - 15} x2={px - 10} y1={py} y2={py} />
                  <line x1={px + 10} x2={px + 15} y1={py} y2={py} />
                  <line x1={px} x2={px} y1={py - 15} y2={py - 10} />
                  <line x1={px} x2={px} y1={py + 10} y2={py + 15} />
                </>
              );
            })()}
          </g>
        )}
        {hover && hover.source !== d && (
          <g className="uso-synchover">
            {(() => {
              const p = projectMollweide(hover.ra, hover.dec, CENTER_RA);
              const [px, py] = toPx(frame, p.x, p.y);
              return (
                <>
                  <circle cx={px} cy={py} r={6} />
                  <line x1={px - 11} x2={px + 11} y1={py} y2={py} />
                  <line x1={px} x2={px} y1={py - 11} y2={py + 11} />
                </>
              );
            })()}
          </g>
        )}
        {layer === "entities" && nearest !== null && (
          <circle className="uso-nearest" cx={toPx(frame, projected.xs[nearest], projected.ys[nearest])[0]} cy={toPx(frame, projected.xs[nearest], projected.ys[nearest])[1]} r={5} />
        )}
      </svg>
      {pointer && onSky && !down && (
        <div className="uso-tip uso-skytip" style={{ left: Math.min(size.width - 250, pointer.x + 14), top: Math.max(4, Math.min(size.height - 120, pointer.y + 12)) }} role="tooltip">
          {layer === "entities" && nearest !== null ? (
            <>
              <strong className="is-mono">{domain.records[nearest].id}</strong>
              <span>
                {noun.one} · RA {fmtFixed(domain.records[nearest].ra, 4)}° Dec {fmtFixed(domain.records[nearest].dec, 4)}°
              </span>
              <small>Click to focus in the Inspector</small>
            </>
          ) : (
            <>
              <strong>
                HEALPix {order}/{hoverPixel} · {fmtNum(area, 3)} deg²
              </strong>
              <span>
                RA {fmtFixed(pointer.ra, 2)}° Dec {fmtFixed(pointer.dec, 2)}° · b {fmtFixed(icrsToGalactic(pointer.ra, pointer.dec)[1], 1)}° · β {fmtFixed(icrsToEcliptic(pointer.ra, pointer.dec)[1], 1)}°
              </span>
              {layer === "density" && !unadmitted && (
                <span>
                  {fmtInt(hoverCount)} {hoverCount === 1 ? noun.one : noun.many}
                  {filtered ? " in cross-filter" : ""} · {fmtNum(hoverCount / area, 3)} / deg²
                </span>
              )}
              <small>
                {hoverCovered === null
                  ? "Coverage unavailable for this domain: an empty cell is not zero coverage."
                  : hoverCovered
                    ? `Inside the declared footprint${coverageSynthetic ? " (synthetic fixture, not Rubin coverage)" : ""}.`
                    : "Outside the declared footprint."}
              </small>
            </>
          )}
        </div>
      )}
    </div>
  );
}
