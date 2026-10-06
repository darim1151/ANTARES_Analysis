import type { DomainId, EntityKind, EvidenceClass, NightState } from "@/types/observatory";

/**
 * Canvas-side mirror of the CSS tokens in app/observatory/observatory.css.
 * Domain identity hues are the first two validated categorical slots (dark
 * surface); each domain's magnitude ramp is a one-hue sequential ramp built at
 * the same OKLCH lightness steps, so the two ramps read with equal weight.
 */
export const TOKENS = {
  plane: "#0b0b0a",
  surface: "#131312",
  raised: "#1a1a18",
  hairline: "#272724",
  axis: "#3b3a36",
  muted: "#8a8880",
  secondary: "#c3c2b7",
  primary: "#f2f1ec",
  select: "#ffffff",
  warning: "#fab219",
  critical: "#d03b3b",
  galactic: "#c2a878",
  ecliptic: "#86a4bd",
  coverage: "rgba(242, 241, 236, 0.075)"
} as const;

export const DOMAIN_COLOR: Record<DomainId, string> = {
  antares: "#3987e5",
  fink: "#d95926"
};

export const DOMAIN_LABEL: Record<DomainId, string> = {
  antares: "ANTARES",
  fink: "FINK"
};

/** Dark-mode sequential ramps: index 0 recedes into the surface, last is brightest. */
export const DOMAIN_RAMP: Record<DomainId, string[]> = {
  antares: ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
  fink: ["#621e01", "#762500", "#8a2d01", "#9f3602", "#b2410f", "#c84b14", "#d85a29", "#de734d", "#e58766", "#ea9c81", "#f1b099", "#f4c4b3", "#fad7cb"]
};

/** LSST band identities inside a single-domain light curve (never domain colors). */
export const BAND_COLOR: Record<string, string> = {
  u: "#9085e9",
  g: "#199e70",
  r: "#e66767",
  i: "#c98500",
  z: "#d55181",
  y: "#8ea33a"
};

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** Sample a ramp at t in [0, 1] with linear RGB-space interpolation between steps. */
export function rampColor(ramp: string[], t: number, alpha = 1): string {
  const x = Math.max(0, Math.min(1, t)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - i;
  const a = hexToRgb(ramp[i]);
  const b = hexToRgb(ramp[i + 1]);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${alpha})`;
}

export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export const EVIDENCE_LABEL: Record<EvidenceClass, string> = {
  ACCEPTED_SCIENCE: "Accepted science",
  VALIDATED_TRANSPORT_EVIDENCE: "Transport evidence",
  COMMITTED_OPERATIONAL_RECORD: "Operational record",
  LEGACY_SAMPLE: "Legacy sample",
  SYNTHETIC_DEMO: "Synthetic demo",
  SYNTHETIC_FIXTURE: "Synthetic fixture"
};

export const EVIDENCE_TONE: Record<EvidenceClass, "real" | "legacy" | "synthetic"> = {
  ACCEPTED_SCIENCE: "real",
  VALIDATED_TRANSPORT_EVIDENCE: "real",
  COMMITTED_OPERATIONAL_RECORD: "real",
  LEGACY_SAMPLE: "legacy",
  SYNTHETIC_DEMO: "synthetic",
  SYNTHETIC_FIXTURE: "synthetic"
};

export const NIGHT_STATE_LABEL: Record<NightState, string> = {
  AVAILABLE: "Available",
  ZERO: "Zero records",
  UNQUALIFIED: "Unqualified",
  UNAVAILABLE: "Unavailable · not admitted",
  MISSING: "Missing",
  OUTSIDE_COVERAGE: "Outside coverage"
};

export const NIGHT_STATE_HELP: Record<NightState, string> = {
  AVAILABLE: "Admitted to this basis.",
  ZERO: "Available, and the source delivered zero records.",
  UNQUALIFIED: "Transport-validated but not characterized or admitted; never read as a rate.",
  UNAVAILABLE: "In process upstream; not delivery-validated and not admitted.",
  MISSING: "Genuinely absent data or evidence.",
  OUTSIDE_COVERAGE: "Outside what this domain build describes."
};

/** Native terminology per entity kind (never a shared "object" word). */
export const ENTITY_NOUN: Record<EntityKind, { one: string; many: string }> = {
  "antares.locus": { one: "locus", many: "loci" },
  "fink.diaObject": { one: "DiaObject", many: "DiaObjects" }
};
