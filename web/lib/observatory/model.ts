/**
 * In-memory workspace model derived from a verified bundle. Presentation
 * components read capabilities and dimensions from here rather than branching
 * on the domain name.
 */

import {
  DOMAIN_IDS,
  type Capability,
  type CapabilityState,
  type DomainId,
  type DomainSkyPayload,
  type EntitySummary,
  type EvidenceClass,
  type FeatureDimension,
  type NativeEntityRef,
  type ObservatoryBundle
} from "../../types/observatory.ts";
import { ENTITY_INDEX_ORDER, type DomainColumns } from "./kernel/selection.ts";
import { degradeCounts, mocCellsAtOrder, radecToPix } from "./kernel/healpix.ts";
import { addDays, utcDateRange } from "./kernel/dates.ts";

export const SKY_ORDERS = [3, 4, 5, 6];

export type DomainModel = {
  id: DomainId;
  bundle: ObservatoryBundle["domains"][DomainId];
  pin: ObservatoryBundle["basis"]["domains"][DomainId];
  cols: DomainColumns;
  records: EntitySummary[];
  indexById: Map<string, number>;
  dims: Map<string, FeatureDimension>;
  /** Dimensions offered by the Lab (AVAILABLE or PARTIALLY_QUALIFIED). */
  selectable: FeatureDimension[];
  /** Full-population density by display order (adapter map, degraded). */
  densityAt: (order: number) => Map<number, number>;
  /** Coverage cells by display order, or null when coverage is unavailable. */
  coverageAt: (order: number) => Set<number> | null;
};

export type WorkspaceModel = {
  bundle: ObservatoryBundle;
  domains: Record<DomainId, DomainModel>;
  caps: Map<string, Capability>;
  /** Union of all lanes' UTC date ranges. */
  dateAxis: string[];
};

function memoByOrder<T>(build: (order: number) => T): (order: number) => T {
  const cache = new Map<number, T>();
  return (order) => {
    if (!cache.has(order)) cache.set(order, build(order));
    return cache.get(order) as T;
  };
}

function densityBuilder(sky: DomainSkyPayload) {
  return memoByOrder((order) => degradeCounts(sky.density.pixels, sky.density.values, sky.density.order, Math.min(order, sky.density.order)));
}

export function buildWorkspaceModel(bundle: ObservatoryBundle): WorkspaceModel {
  const domains = {} as Record<DomainId, DomainModel>;
  for (const id of DOMAIN_IDS) {
    const payloads = bundle.domains[id];
    const records = payloads.entities.records;
    const n = records.length;
    const features = new Map<string, Float64Array>();
    for (const [key, column] of Object.entries(payloads.features.columns)) {
      features.set(key, Float64Array.from(column, (v) => (v === null ? NaN : v)));
    }
    const cols: DomainColumns = {
      domain: id,
      n,
      ra: Float64Array.from(records, (r) => r.ra),
      dec: Float64Array.from(records, (r) => r.dec),
      dates: records.map((r) => r.entity_date),
      hpx: Int32Array.from(records, (r) => radecToPix(ENTITY_INDEX_ORDER, r.ra, r.dec)),
      features
    };
    const dims = new Map(payloads.features.dimensions.map((d) => [d.id, d]));
    const coverage = payloads.sky.coverage;
    domains[id] = {
      id,
      bundle: payloads,
      pin: bundle.basis.domains[id],
      cols,
      records,
      indexById: new Map(records.map((r, i) => [r.id, i])),
      dims,
      selectable: payloads.features.dimensions.filter((d) => d.state !== "UNAVAILABLE"),
      densityAt: densityBuilder(payloads.sky),
      coverageAt: coverage ? memoByOrder((order) => mocCellsAtOrder(coverage.moc, order)) : () => null
    };
  }
  const starts = DOMAIN_IDS.map((d) => bundle.domains[d].time.range.start).sort();
  const stops = DOMAIN_IDS.map((d) => bundle.domains[d].time.range.stop).sort();
  return {
    bundle,
    domains,
    caps: new Map(bundle.capabilities.capabilities.map((c) => [c.id, c])),
    dateAxis: utcDateRange(starts[0], stops[stops.length - 1])
  };
}

const UNKNOWN_CAPABILITY: Capability = {
  id: "unknown",
  scope: "workspace",
  area: "compare",
  name: "unknown",
  state: "UNAVAILABLE",
  evidence: [],
  summary: "Not declared",
  reason: "This capability is not declared by the pinned basis.",
  codes: ["NOT_DECLARED"],
  qualifications: []
};

/** Capability lookup; undeclared capabilities are UNAVAILABLE by construction. */
export function capability(model: WorkspaceModel, scope: DomainId | "relation" | "workspace", id: string): Capability {
  return model.caps.get(`${scope}:${id}`) ?? { ...UNKNOWN_CAPABILITY, id: `${scope}:${id}` };
}

export function isUsable(state: CapabilityState): boolean {
  return state !== "UNAVAILABLE";
}

export function hasEntity(model: WorkspaceModel, ref: NativeEntityRef): boolean {
  const d = model.domains[ref.domain];
  return Boolean(d && d.bundle.entities.entity_kind === ref.kind && d.indexById.has(ref.id));
}

export function entityRef(domain: DomainModel, index: number): NativeEntityRef {
  return { domain: domain.id, kind: domain.bundle.entities.entity_kind, id: domain.records[index].id };
}

/** Weakest-first sorted, de-duplicated evidence classes. */
export function evidenceUnion(lists: EvidenceClass[][]): EvidenceClass[] {
  const order: EvidenceClass[] = [
    "SYNTHETIC_FIXTURE",
    "SYNTHETIC_DEMO",
    "LEGACY_SAMPLE",
    "COMMITTED_OPERATIONAL_RECORD",
    "VALIDATED_TRANSPORT_EVIDENCE",
    "ACCEPTED_SCIENCE"
  ];
  const set = new Set(lists.flat());
  return order.filter((e) => set.has(e));
}

/** Last date inside a half-open [start, stop) interval. */
export function lastDate(stop: string): string {
  return addDays(stop, -1);
}
