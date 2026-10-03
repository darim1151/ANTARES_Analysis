import type { DensityTile, SkyPoint } from "@/types/skypulse";

export type ScreenPoint = {
  x: number;
  y: number;
};

export function projectSky(ra: number, dec: number, width: number, height: number): ScreenPoint {
  const marginX = width * 0.055;
  const marginY = height * 0.09;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;
  const x = marginX + ((360 - ra) / 360) * plotW;
  const y = marginY + ((90 - dec) / 180) * plotH;
  return { x, y };
}

export function tileCenter(tile: DensityTile) {
  return {
    ra: (tile.ra_min + tile.ra_max) / 2,
    dec: (tile.dec_min + tile.dec_max) / 2
  };
}

export function visibleByMode(point: SkyPoint, mode: "blend" | "last" | "history") {
  if (mode === "last") return point.is_last_night;
  if (mode === "history") return !point.is_last_night;
  return true;
}
