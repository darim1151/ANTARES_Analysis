/**
 * Compact, versioned URL encoding of ScientificState.
 *
 * Decoding never trusts the URL: every reference is checked against the
 * loaded basis through `DecodeContext`, and anything unknown is dropped with
 * a warning rather than silently reinterpreted. Pure module.
 */

import type {
  DomainId,
  EntityKind,
  FeaturePredicate,
  FeatureRange,
  LabConfig,
  LabStatistic,
  NativeEntityRef,
  ScientificState,
  SkyPredicate,
  SourceMode
} from "@/types/observatory";
import { isUtcDate } from "./dates.ts";
import { MAX_ORDER, npix } from "./healpix.ts";

export const STATE_CODEC_VERSION = "1";

const DOMAINS: DomainId[] = ["antares", "fink"];
const MODES: SourceMode[] = ["antares", "fink", "compare"];
const KIND_DOMAIN: Record<EntityKind, DomainId> = { "antares.locus": "antares", "fink.diaObject": "fink" };
const STATISTICS: LabStatistic[] = ["count", "median", "mean"];
const DOMAIN_KEY: Record<DomainId, string> = { antares: "a", fink: "f" };

export type DecodeContext = {
  basisId: string;
  isSelectableDimension: (domain: DomainId, dimension: string) => boolean;
  hasEntity: (ref: NativeEntityRef) => boolean;
  skyOrders: number[];
};

function num(value: number, digits: number): string {
  return Number(value.toFixed(digits)).toString();
}

function encodeRange(r: FeatureRange): string {
  // Significant digits, not decimals: small-scale dimensions must survive the URL.
  return `${r.dimension}:${Number(r.min.toPrecision(10))}:${Number(r.max.toPrecision(10))}`;
}

function encodeLab(lab: LabConfig): string {
  return [lab.x, lab.y, lab.statistic, lab.z ?? ""].join("~");
}

export function encodeState(state: ScientificState): string {
  const p = new URLSearchParams();
  p.set("v", STATE_CODEC_VERSION);
  p.set("b", state.basis_id);
  p.set("m", state.mode);
  const sel = state.selection;
  if (sel.time) p.set("t", `${sel.time.start}..${sel.time.stop}`);
  if (sel.sky) {
    p.set(
      "s",
      sel.sky.kind === "healpix"
        ? `h${sel.sky.order}:${sel.sky.pixels.join(",")}`
        : `c:${num(sel.sky.ra, 4)},${num(sel.sky.dec, 4)},${num(sel.sky.radius_deg, 4)}`
    );
  }
  for (const domain of DOMAINS) {
    const f = sel.feature[domain];
    if (f) p.set(`f${DOMAIN_KEY[domain]}`, f.y ? `${encodeRange(f.x)}~${encodeRange(f.y)}` : encodeRange(f.x));
  }
  if (state.focus) p.set("focus", `${state.focus.kind}:${state.focus.id}`);
  p.set("lens", state.lens.primary);
  for (const domain of DOMAINS) p.set(`lab${DOMAIN_KEY[domain]}`, encodeLab(state.lens.lab[domain]));
  p.set("o", String(state.presentation.skyOrder));
  p.set("layer", state.presentation.skyLayer);
  const ov = state.presentation.overlays;
  p.set("ov", `${ov.graticule ? "g" : ""}${ov.galacticPlane ? "p" : ""}${ov.ecliptic ? "e" : ""}` || "-");
  return p.toString();
}

function decodeRange(text: string, domain: DomainId, ctx: DecodeContext): FeatureRange | null {
  const parts = text.split(":");
  if (parts.length !== 3) return null;
  const [dimension, a, b] = parts;
  const min = Number(a);
  const max = Number(b);
  if (!ctx.isSelectableDimension(domain, dimension) || !Number.isFinite(min) || !Number.isFinite(max) || min > max) {
    return null;
  }
  return { dimension, min, max };
}

function decodeSky(text: string, ctx: DecodeContext): SkyPredicate | null {
  const healpix = /^h(\d{1,2}):(\d+(?:,\d+)*)$/.exec(text);
  if (healpix) {
    const order = Number(healpix[1]);
    if (order > MAX_ORDER || !ctx.skyOrders.includes(order)) return null;
    const max = npix(order);
    const pixels = [...new Set(healpix[2].split(",").map(Number))].sort((x, y) => x - y);
    if (pixels.some((px) => !Number.isInteger(px) || px < 0 || px >= max) || pixels.length > 4096) return null;
    return { kind: "healpix", order, pixels };
  }
  const cone = /^c:([-\d.]+),([-\d.]+),([\d.]+)$/.exec(text);
  if (cone) {
    const ra = Number(cone[1]);
    const dec = Number(cone[2]);
    const radius = Number(cone[3]);
    if (ra >= 0 && ra < 360 && dec >= -90 && dec <= 90 && radius > 0 && radius <= 180) {
      return { kind: "cone", ra, dec, radius_deg: radius };
    }
  }
  return null;
}

function decodeLab(text: string | null, domain: DomainId, fallback: LabConfig, ctx: DecodeContext): LabConfig {
  if (!text) return fallback;
  const [x, y, statistic, z] = text.split("~");
  const ok = (dim: string | undefined) => Boolean(dim) && ctx.isSelectableDimension(domain, dim as string);
  if (!ok(x) || !ok(y) || !STATISTICS.includes(statistic as LabStatistic)) return fallback;
  const stat = statistic as LabStatistic;
  const zDim = z && ok(z) ? z : null;
  if (stat !== "count" && !zDim) return { x, y, statistic: "count", z: null };
  return { x, y, statistic: stat, z: stat === "count" ? null : zDim };
}

/** Decode onto `defaults`; returns the state and human-readable warnings. */
export function decodeState(
  query: string,
  defaults: ScientificState,
  ctx: DecodeContext
): { state: ScientificState; warnings: string[] } {
  const p = new URLSearchParams(query);
  const warnings: string[] = [];
  if (!p.has("v")) return { state: defaults, warnings };
  if (p.get("v") !== STATE_CODEC_VERSION) {
    return { state: defaults, warnings: [`Unsupported view encoding v${p.get("v")}; default view restored.`] };
  }
  if (p.get("b") !== ctx.basisId) {
    return {
      state: defaults,
      warnings: [`View was recorded against basis ${p.get("b") ?? "unknown"}; this workspace pins ${ctx.basisId}. Default view restored.`]
    };
  }
  const state: ScientificState = structuredClone(defaults);
  const mode = p.get("m");
  if (mode && MODES.includes(mode as SourceMode)) state.mode = mode as SourceMode;

  const t = p.get("t");
  if (t) {
    const [start, stop] = t.split("..");
    if (isUtcDate(start) && isUtcDate(stop) && start < stop) state.selection.time = { kind: "utc_dates", start, stop };
    else warnings.push("Ignored an invalid time selection.");
  }
  const s = p.get("s");
  if (s) {
    const sky = decodeSky(s, ctx);
    if (sky) state.selection.sky = sky;
    else warnings.push("Ignored an invalid sky selection.");
  }
  for (const domain of DOMAINS) {
    const raw = p.get(`f${DOMAIN_KEY[domain]}`);
    if (!raw) continue;
    const [xText, yText] = raw.split("~");
    const x = decodeRange(xText, domain, ctx);
    const y = yText ? decodeRange(yText, domain, ctx) : null;
    if (x && (!yText || y)) {
      const predicate: FeaturePredicate = { x, y };
      state.selection.feature[domain] = predicate;
    } else {
      warnings.push(`Ignored a ${domain.toUpperCase()} feature selection on a dimension this basis does not support.`);
    }
  }
  const focus = p.get("focus");
  if (focus) {
    const idx = focus.indexOf(":");
    const kind = focus.slice(0, idx) as EntityKind;
    const id = focus.slice(idx + 1);
    const domain = KIND_DOMAIN[kind];
    const ref: NativeEntityRef | null = domain ? { domain, kind, id } : null;
    if (ref && ctx.hasEntity(ref)) state.focus = ref;
    else warnings.push("Ignored a focus entity that is not present in this basis.");
  }
  const lens = p.get("lens");
  if (lens === "sky" || lens === "population") state.lens.primary = lens;
  for (const domain of DOMAINS) {
    state.lens.lab[domain] = decodeLab(p.get(`lab${DOMAIN_KEY[domain]}`), domain, defaults.lens.lab[domain], ctx);
  }
  // A brush is defined on the Lab axes it was drawn on; one that does not match
  // the decoded axes could never be seen, so it is not silently kept.
  for (const domain of DOMAINS) {
    const f = state.selection.feature[domain];
    const lab = state.lens.lab[domain];
    if (f && (!f.y || f.x.dimension !== lab.x || f.y.dimension !== lab.y)) {
      delete state.selection.feature[domain];
      warnings.push(`Ignored a ${domain.toUpperCase()} feature selection that does not match its Lab axes.`);
    }
  }
  const order = Number(p.get("o"));
  if (ctx.skyOrders.includes(order)) state.presentation.skyOrder = order;
  const layer = p.get("layer");
  if (layer === "density" || layer === "entities") state.presentation.skyLayer = layer;
  const ov = p.get("ov");
  if (ov) {
    state.presentation.overlays = { graticule: ov.includes("g"), galacticPlane: ov.includes("p"), ecliptic: ov.includes("e") };
  }
  return { state, warnings };
}
