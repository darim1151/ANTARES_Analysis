/**
 * ScientificState = Basis + Selection + Focus + Lens + Presentation.
 * Pure reducer; React wiring lives in components/observatory/ObservatoryContext.
 */

import {
  DOMAIN_IDS,
  type DomainId,
  type EvidenceClass,
  type FeaturePredicate,
  type LabConfig,
  type NativeEntityRef,
  type PresentationState,
  type PrimaryLens,
  type ScientificState,
  type SkyPredicate,
  type SourceMode,
  type TimePredicate,
  type ViewManifest
} from "../../types/observatory.ts";
import { decodeState, encodeState, type DecodeContext } from "./kernel/stateCodec.ts";
import { evidenceUnion, hasEntity, type WorkspaceModel } from "./model.ts";

export type Action =
  | { type: "mode"; mode: SourceMode }
  | { type: "time"; time: TimePredicate | null }
  | { type: "sky"; sky: SkyPredicate | null }
  | { type: "skyCell"; order: number; pixel: number; additive: boolean }
  | { type: "feature"; domain: DomainId; predicate: FeaturePredicate | null }
  | { type: "clearSelection" }
  | { type: "focus"; focus: NativeEntityRef | null }
  | { type: "lens"; primary: PrimaryLens }
  | { type: "lab"; domain: DomainId; patch: Partial<LabConfig> }
  | { type: "presentation"; patch: Partial<PresentationState> }
  | { type: "overlay"; key: keyof PresentationState["overlays"]; value: boolean }
  | { type: "replace"; state: ScientificState };

/**
 * Finest display order at which every domain's occupied cells hold, on
 * average, at least MIN_PER_CELL entities, so density reads as density rather
 * than as single-entity noise. Deterministic for a given basis.
 */
const MIN_PER_CELL = 3;
export function adaptiveSkyOrder(model: WorkspaceModel): number {
  for (const order of [...model.skyOrders].sort((a, b) => b - a)) {
    const ok = DOMAIN_IDS.every((d) => {
      const cells = model.domains[d].densityAt(order);
      let total = 0;
      for (const v of cells.values()) total += v;
      return cells.size > 0 && total / cells.size >= MIN_PER_CELL;
    });
    if (ok) return order;
  }
  return Math.min(...model.skyOrders);
}

export function defaultState(model: WorkspaceModel): ScientificState {
  const lab = {} as Record<DomainId, LabConfig>;
  for (const d of DOMAIN_IDS) {
    const defaults = model.domains[d].bundle.features.lab_defaults;
    lab[d] = { x: defaults.x, y: defaults.y, statistic: "count", z: null };
  }
  return {
    basis_id: model.bundle.basis.basis_id,
    mode: "compare",
    selection: { version: 1, time: null, sky: null, feature: {} },
    focus: null,
    lens: { primary: "sky", lab },
    presentation: { skyOrder: adaptiveSkyOrder(model), skyLayer: "density", overlays: { graticule: true, galacticPlane: true, ecliptic: true } }
  };
}

export function reduce(state: ScientificState, action: Action): ScientificState {
  switch (action.type) {
    case "mode":
      // Source switching preserves every compatible part of the state: time and
      // sky predicates are geometric; feature predicates are kept per domain.
      return { ...state, mode: action.mode };
    case "time":
      return { ...state, selection: { ...state.selection, time: action.time } };
    case "sky":
      return { ...state, selection: { ...state.selection, sky: action.sky } };
    case "skyCell": {
      const current = state.selection.sky;
      let pixels: number[];
      if (current?.kind === "healpix" && current.order === action.order) {
        const set = new Set(current.pixels);
        if (action.additive) {
          if (set.has(action.pixel)) set.delete(action.pixel);
          else set.add(action.pixel);
          pixels = [...set];
        } else {
          pixels = set.size === 1 && set.has(action.pixel) ? [] : [action.pixel];
        }
      } else {
        pixels = [action.pixel];
      }
      const sky: SkyPredicate | null = pixels.length ? { kind: "healpix", order: action.order, pixels: pixels.sort((a, b) => a - b) } : null;
      return { ...state, selection: { ...state.selection, sky } };
    }
    case "feature": {
      const feature = { ...state.selection.feature };
      if (action.predicate) feature[action.domain] = action.predicate;
      else delete feature[action.domain];
      return { ...state, selection: { ...state.selection, feature } };
    }
    case "clearSelection":
      return { ...state, selection: { version: 1, time: null, sky: null, feature: {} } };
    case "focus":
      return { ...state, focus: action.focus };
    case "lens":
      return { ...state, lens: { ...state.lens, primary: action.primary } };
    case "lab": {
      const next = { ...state.lens.lab[action.domain], ...action.patch };
      const lab = { ...state.lens.lab, [action.domain]: next };
      // A feature brush is defined on the axes it was drawn on; changing an axis
      // retires it rather than silently reinterpreting its ranges.
      const brush = state.selection.feature[action.domain];
      let feature = state.selection.feature;
      if (brush && (brush.x.dimension !== next.x || (brush.y && brush.y.dimension !== next.y))) {
        feature = { ...feature };
        delete feature[action.domain];
      }
      return { ...state, lens: { ...state.lens, lab }, selection: { ...state.selection, feature } };
    }
    case "presentation":
      return { ...state, presentation: { ...state.presentation, ...action.patch } };
    case "overlay":
      return {
        ...state,
        presentation: { ...state.presentation, overlays: { ...state.presentation.overlays, [action.key]: action.value } }
      };
    case "replace":
      return action.state;
  }
}

export function decodeContext(model: WorkspaceModel): DecodeContext {
  return {
    basisId: model.bundle.basis.basis_id,
    isSelectableDimension: (domain, dimension) => model.domains[domain]?.selectable.some((d) => d.id === dimension) ?? false,
    hasEntity: (ref) => hasEntity(model, ref),
    skyOrders: model.skyOrders
  };
}

export function stateFromUrl(model: WorkspaceModel, search: string) {
  return decodeState(search.startsWith("?") ? search.slice(1) : search, defaultState(model), decodeContext(model));
}

export function stateToUrl(state: ScientificState): string {
  return `?${encodeState(state)}`;
}

export function displayedDomains(mode: SourceMode): DomainId[] {
  return mode === "compare" ? [...DOMAIN_IDS] : [mode];
}

export function buildViewManifest(model: WorkspaceModel, state: ScientificState): ViewManifest {
  const { bundle } = model;
  const shown = displayedDomains(state.mode);
  // Evidence of what is actually drawn: both time lanes (one is context in a
  // single-domain view), each shown domain's sky layers and Lab axes, and the
  // focused record's fields.
  const byLens: Record<string, EvidenceClass[]> = {};
  for (const d of DOMAIN_IDS) {
    const t = model.domains[d].bundle.time;
    byLens[`time.${d}`] = evidenceUnion([t.counts?.evidence ?? [], ...t.windows.map((w) => w.evidence)]);
  }
  for (const d of shown) {
    const dm = model.domains[d];
    byLens[`sky.${d}.density`] = dm.bundle.sky.density.evidence;
    if (dm.bundle.sky.coverage) byLens[`sky.${d}.coverage`] = dm.bundle.sky.coverage.evidence;
    const lab = state.lens.lab[d];
    for (const [axis, id] of [["x", lab.x], ["y", lab.y], ["z", lab.z]] as const) {
      const ev = id ? dm.dims.get(id)?.evidence : null;
      if (ev) byLens[`lab.${d}.${axis}`] = [ev];
    }
  }
  if (state.focus) {
    byLens[`inspector.${state.focus.domain}`] = evidenceUnion([model.domains[state.focus.domain].bundle.entities.fields.map((f) => f.evidence)]);
  }
  const domains = Object.fromEntries(
    DOMAIN_IDS.map((d) => [d, { build_id: bundle.basis.domains[d].build_id, evidence: bundle.basis.domains[d].evidence }])
  ) as ViewManifest["basis"]["domains"];
  return {
    kind: "uso.view-manifest",
    version: 1,
    basis: {
      basis_id: bundle.basis.basis_id,
      status: bundle.basis.status,
      science_ready: bundle.basis.science_ready,
      domains,
      relation: bundle.basis.relation,
      semantic_contract: bundle.basis.semantic_contract,
      feature_registry: bundle.basis.feature_registry,
      analysis_kernel: bundle.basis.analysis_kernel
    },
    bundle: { bundle_id: bundle.manifest.bundle_id, manifest_sha256: bundle.manifestSha256, integrity: bundle.integrity.method },
    state,
    evidence_in_view: evidenceUnion(Object.values(byLens)),
    evidence_by_lens: byLens,
    caveats: [
      bundle.basis.science_ready ? "" : "This view is drawn from a FIRST_LIGHT_FIXTURE basis and is not science-ready.",
      ...bundle.basis.invariants
    ].filter(Boolean)
  };
}
