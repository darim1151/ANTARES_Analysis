"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { clamp } from "@/lib/format";
import { projectSky, tileCenter, visibleByMode } from "@/lib/sky";
import type { DensityTile, SkyPoint } from "@/types/skypulse";

type Mode = "blend" | "last" | "history";

type Props = {
  points: SkyPoint[];
  tiles: DensityTile[];
  mode: Mode;
  heatmap: boolean;
  blend: number;
  timeline: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
};

type HoverState = {
  point: SkyPoint;
  x: number;
  y: number;
} | null;

export default function SkyCanvas({
  points,
  tiles,
  mode,
  heatmap,
  blend,
  timeline,
  selectedId,
  onSelect
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const positionsRef = useRef<Array<{ point: SkyPoint; x: number; y: number; visible: boolean }>>([]);
  const [size, setSize] = useState({ width: 1200, height: 720 });
  const [hover, setHover] = useState<HoverState>(null);

  const mjdBounds = useMemo(() => {
    const values = points.map((point) => point.mjd);
    return { min: Math.min(...values), max: Math.max(...values) };
  }, [points]);

  const densityWeights = useMemo(() => {
    const threshold = mjdBounds.min + (mjdBounds.max - mjdBounds.min) * (timeline / 100);
    const weights = new Map<string, number>(tiles.map((tile) => [tile.id, 0]));
    for (const point of points) {
      const isVisible =
        visibleByMode(point, mode) &&
        (point.is_last_night || point.mjd <= threshold) &&
        (point.is_last_night || blend > 0);
      if (!isVisible) continue;
      const tile = tiles.find(
        (candidate) =>
          point.ra >= candidate.ra_min &&
          point.ra < candidate.ra_max &&
          point.dec >= candidate.dec_min &&
          (point.dec < candidate.dec_max ||
            (candidate.dec_max === 90 && point.dec === 90))
      );
      if (!tile) continue;
      const pointWeight = point.is_last_night ? 1 : blend / 100;
      weights.set(tile.id, (weights.get(tile.id) ?? 0) + pointWeight);
    }
    return weights;
  }, [blend, mjdBounds, mode, points, tiles, timeline]);

  useEffect(() => {
    if (!wrapRef.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const rect = entry.contentRect;
      setSize({
        width: Math.max(320, Math.floor(rect.width)),
        height: Math.max(420, Math.floor(rect.height))
      });
    });
    observer.observe(wrapRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(size.width * dpr);
    canvas.height = Math.floor(size.height * dpr);
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let start = performance.now();

    const draw = (time: number) => {
      const elapsed = reduceMotion ? 900 : time - start;
      drawSky(ctx, {
        width: size.width,
        height: size.height,
        points,
        tiles,
        densityWeights,
        mode,
        heatmap,
        blend,
        timeline,
        mjdBounds,
        selectedId,
        elapsed,
        positionsRef
      });
      if (!reduceMotion) raf = requestAnimationFrame(draw);
    };

    draw(start);
    return () => cancelAnimationFrame(raf);
  }, [blend, densityWeights, heatmap, mjdBounds, mode, points, selectedId, size, tiles, timeline]);

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let nearest: HoverState = null;
    let best = 9999;
    for (const item of positionsRef.current) {
      if (!item.visible) continue;
      const distance = Math.hypot(item.x - x, item.y - y);
      if (distance < best && distance < 18) {
        best = distance;
        nearest = { point: item.point, x: item.x, y: item.y };
      }
    }
    setHover(nearest);
  }

  return (
    <div className="sky-canvas-wrap" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        aria-label="Interactive all-sky map of LSST-associated ANTARES loci"
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHover(null)}
        onClick={() => hover && onSelect(hover.point.id)}
      />
      {hover && (
        <div className="sky-tooltip" style={{ left: hover.x, top: hover.y }}>
          <strong>{hover.point.id}</strong>
          <span>{hover.point.reason}</span>
          <small>
            RA {hover.point.ra.toFixed(2)} deg / Dec {hover.point.dec.toFixed(2)} deg
          </small>
        </div>
      )}
    </div>
  );
}

function drawSky(
  ctx: CanvasRenderingContext2D,
  args: {
    width: number;
    height: number;
    points: SkyPoint[];
    tiles: DensityTile[];
    densityWeights: Map<string, number>;
    mode: Mode;
    heatmap: boolean;
    blend: number;
    timeline: number;
    mjdBounds: { min: number; max: number };
    selectedId: string | null;
    elapsed: number;
    positionsRef: React.MutableRefObject<Array<{ point: SkyPoint; x: number; y: number; visible: boolean }>>;
  }
) {
  const {
    width,
    height,
    points,
    tiles,
    densityWeights,
    mode,
    heatmap,
    blend,
    timeline,
    mjdBounds,
    selectedId,
    elapsed,
    positionsRef
  } = args;

  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createRadialGradient(width * 0.5, height * 0.48, 20, width * 0.5, height * 0.48, width * 0.78);
  gradient.addColorStop(0, "#0a2142");
  gradient.addColorStop(0.54, "#071426");
  gradient.addColorStop(1, "#030814");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const marginX = width * 0.055;
  const marginY = height * 0.09;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;

  ctx.save();
  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, plotW / 2, plotH / 2, 0, 0, Math.PI * 2);
  ctx.clip();

  drawGrid(ctx, width, height, marginX, marginY, plotW, plotH);

  if (heatmap) {
    const maxCount = Math.max(1, ...densityWeights.values());
    for (const tile of tiles) {
      const visibleWeight = densityWeights.get(tile.id) ?? 0;
      if (visibleWeight <= 0) continue;
      const center = tileCenter(tile);
      const p = projectSky(center.ra, center.dec, width, height);
      const heat = Math.sqrt(visibleWeight / maxCount);
      const alpha = clamp(heat * 0.42, 0.04, 0.42);
      const radius = 14 + heat * 42;
      const glow = ctx.createRadialGradient(p.x, p.y, 1, p.x, p.y, radius);
      glow.addColorStop(0, `rgba(143, 91, 255, ${alpha})`);
      glow.addColorStop(0.55, `rgba(49, 217, 255, ${alpha * 0.38})`);
      glow.addColorStop(1, "rgba(143, 91, 255, 0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const threshold = mjdBounds.min + (mjdBounds.max - mjdBounds.min) * (timeline / 100);
  const positions: Array<{ point: SkyPoint; x: number; y: number; visible: boolean }> = [];
  for (const point of points) {
    const visible =
      visibleByMode(point, mode) &&
      (point.is_last_night || point.mjd <= threshold) &&
      (point.is_last_night || blend > 0);
    const projected = projectSky(point.ra, point.dec, width, height);
    positions.push({ point, x: projected.x, y: projected.y, visible });
    if (!visible) continue;

    const brightness = clamp((23.5 - point.brightness_mag) / 8, 0.18, 1);
    const obs = clamp(Math.log10(point.obs_count + 1) / 2.6, 0.16, 1);
    const pulse = 0.82 + Math.sin(elapsed / 660 + point.ra) * 0.18;
    const baseRadius = 1.4 + brightness * 2.5 + obs * 1.8;
    const radius = point.is_last_night ? baseRadius * pulse : baseRadius * 0.82;
    const color = point.is_last_night
      ? point.seen_before
        ? "49, 217, 255"
        : "255, 184, 77"
      : "143, 91, 255";
    const alpha = point.is_last_night ? 0.86 : clamp((blend / 100) * 0.55, 0, 0.55);

    ctx.fillStyle = `rgba(${color}, ${alpha})`;
    ctx.shadowColor = `rgba(${color}, ${point.is_last_night ? 0.72 : 0.32})`;
    ctx.shadowBlur = point.is_last_night ? 18 : 8;
    ctx.beginPath();
    ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    if (point.id === selectedId) {
      ctx.strokeStyle = `rgba(${color}, 0.92)`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, radius + 9 + Math.sin(elapsed / 330) * 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  positionsRef.current = positions;

  ctx.restore();

  ctx.strokeStyle = "rgba(244, 247, 251, 0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(width / 2, height / 2, plotW / 2, plotH / 2, 0, 0, Math.PI * 2);
  ctx.stroke();
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  marginX: number,
  marginY: number,
  plotW: number,
  plotH: number
) {
  ctx.strokeStyle = "rgba(244, 247, 251, 0.105)";
  ctx.lineWidth = 1;
  for (let ra = 0; ra <= 360; ra += 30) {
    const x = marginX + (ra / 360) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, marginY);
    ctx.lineTo(x, height - marginY);
    ctx.stroke();
  }
  for (let dec = -60; dec <= 60; dec += 30) {
    const y = marginY + ((90 - dec) / 180) * plotH;
    ctx.beginPath();
    ctx.moveTo(marginX, y);
    ctx.lineTo(width - marginX, y);
    ctx.stroke();
  }
}
