"use client";

import { Maximize2 } from "lucide-react";
import { DOMAIN_IDS, type DomainId } from "@/types/observatory";
import { robustExtent } from "@/lib/observatory/kernel/population";
import { capability } from "@/lib/observatory/model";
import { displayedDomains } from "@/lib/observatory/state";
import { useObservatory } from "./ObservatoryContext";
import LabPlot from "./LabPlot";
import { CapabilityMark } from "./ui";

/**
 * Axes may be shared across domains only when both dimensions carry the same
 * definition_id (kernel-derived quantities); otherwise each panel keeps its own.
 */
function useSharedExtents(shown: DomainId[]) {
  const { model, state } = useObservatory();
  const none = { x: null, y: null } as { x: [number, number] | null; y: [number, number] | null };
  const out = Object.fromEntries(DOMAIN_IDS.map((d) => [d, { ...none }])) as Record<DomainId, typeof none>;
  if (shown.length < 2) return { extents: out, shared: { x: false, y: false } };
  const shared = { x: false, y: false };
  for (const axis of ["x", "y"] as const) {
    const dims = shown.map((d) => model.domains[d].dims.get(state.lens.lab[d][axis])!);
    if (dims.every((dm) => dm.definition_id === dims[0].definition_id)) {
      shared[axis] = true;
      const ranges = shown.map((d, i) => robustExtent(model.domains[d].cols.features.get(dims[i].id)!, dims[i].scale, dims[i].extent));
      const union: [number, number] = [Math.min(...ranges.map((r) => r[0])), Math.max(...ranges.map((r) => r[1]))];
      for (const d of shown) out[d][axis] = union;
    }
  }
  return { extents: out, shared };
}

export default function PopulationLab({ placement }: { placement: "primary" | "secondary" | "dock" }) {
  const { model, state, dispatch } = useObservatory();
  const shown = displayedDomains(state.mode);
  const compact = placement !== "primary";
  const { extents, shared } = useSharedExtents(shown);
  const sharedCap = capability(model, "workspace", "compare.shared_axes");

  return (
    <section className={`uso-panel uso-lab is-${placement}`} aria-label="Feature and population lab">
      <header className="uso-panel-head">
        <h2>Population Lab</h2>
        <span className="uso-sub">
          {compact ? "Cross-filtered by Time and Sky" : "Native parameter space per domain · cross-filtered by Time and Sky · brushes feed the shared selection"}
        </span>
        <div className="uso-tools">
          {shown.length > 1 && (
            <span className="uso-sharedaxes" title={sharedCap.reason}>
              <CapabilityMark state={sharedCap.state} label={false} />
              {shared.x || shared.y ? `Shared ${[shared.x && "X", shared.y && "Y"].filter(Boolean).join(" & ")} (identical definition)` : "Independent axes"}
            </span>
          )}
          {placement !== "primary" && (
            <button type="button" className="uso-btn uso-btn-quiet" onClick={() => dispatch({ type: "lens", primary: "population" })} title="Make the Lab the primary lens">
              <Maximize2 aria-hidden="true" /> Expand
            </button>
          )}
        </div>
      </header>
      <div className={`uso-lab-body is-n${shown.length}`}>
        {shown.map((d) => (
          <LabPlot
            key={d}
            domain={model.domains[d]}
            compact={compact}
            definitions={!compact && shown.length === 1 ? "side" : "collapsed"}
            sharedExtents={extents[d]}
          />
        ))}
      </div>
    </section>
  );
}
