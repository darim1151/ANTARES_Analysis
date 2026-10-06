"use client";

import { Maximize2 } from "lucide-react";
import type { DomainId, SkyLayer } from "@/types/observatory";
import { pixelAreaDeg2 } from "@/lib/observatory/kernel/healpix";
import { fmtNum } from "@/lib/observatory/format";
import { capability, SKY_ORDERS, type DomainModel } from "@/lib/observatory/model";
import { displayedDomains } from "@/lib/observatory/state";
import { DOMAIN_RAMP, ENTITY_NOUN, rampColor } from "@/lib/observatory/theme";
import { useObservatory } from "./ObservatoryContext";
import SkyMap, { densityScale } from "./SkyMap";
import { CapabilityMark, CapabilityNote, DomainBadge, EvidenceChips, Segmented, SelectField, Toggle } from "./ui";

function SkyLegend({ domain }: { domain: DomainModel }) {
  const { model, state } = useObservatory();
  const order = state.presentation.skyOrder;
  const area = pixelAreaDeg2(order);
  const { max } = densityScale(domain, order);
  const coverageCap = capability(model, domain.id, "sky.coverage");
  const coverage = domain.bundle.sky.coverage;
  const ramp = DOMAIN_RAMP[domain.id];
  const stops = Array.from({ length: 9 }, (_, i) => rampColor(ramp, 0.12 + (0.88 * i) / 8)).join(", ");
  const noun = ENTITY_NOUN[domain.bundle.entities.entity_kind];
  return (
    <div className="uso-skylegend">
      {state.presentation.skyLayer === "density" ? (
        <div className="uso-ramp" title="Log-scaled; normalized to the full basis population at this order, so filtering dims rather than re-normalizes">
          <span className="uso-ramp-label">{noun.many} / deg²</span>
          <span className="uso-ramp-bar" style={{ background: `linear-gradient(90deg, ${stops})` }} />
          <span className="uso-ramp-ticks">
            <span>{fmtNum(1 / area, 2)}</span>
            <span>{fmtNum(max / area, 2)}</span>
          </span>
        </div>
      ) : (
        <span className="uso-legend-item">
          <i className={`uso-dot uso-dot-${domain.id}`} aria-hidden="true" /> {noun.many} in selection
          <i className="uso-dot uso-dot-context" aria-hidden="true" /> outside selection
        </span>
      )}
      {coverage ? (
        <span className="uso-legend-item" title={coverage.meaning}>
          <i className="uso-swatch uso-swatch-coverage" aria-hidden="true" />
          Footprint <EvidenceChips list={coverage.evidence} compact /> <em>not Rubin coverage</em>
        </span>
      ) : (
        <span className="uso-legend-item" title={coverageCap.reason}>
          <CapabilityMark state={coverageCap.state} label={false} /> Coverage unavailable · density ≠ coverage
        </span>
      )}
      {state.presentation.overlays.galacticPlane && (
        <span className="uso-legend-item">
          <i className="uso-line uso-line-galactic" aria-hidden="true" /> Galactic plane · GC
        </span>
      )}
      {state.presentation.overlays.ecliptic && (
        <span className="uso-legend-item">
          <i className="uso-line uso-line-ecliptic" aria-hidden="true" /> Ecliptic
        </span>
      )}
    </div>
  );
}

export default function SkyLens({ placement }: { placement: "primary" | "secondary" | "dock" }) {
  const { model, state, dispatch, masks } = useObservatory();
  const shown = displayedDomains(state.mode);
  const compact = placement !== "primary";
  const order = state.presentation.skyOrder;
  const diff = capability(model, "workspace", "compare.difference_map");
  const relation = capability(model, "relation", "relation.cross_broker_association");

  return (
    <section className={`uso-panel uso-sky is-${placement}`} aria-label="Sky lens">
      <header className="uso-panel-head">
        <h2>Sky</h2>
        <span className="uso-sub">
          Mollweide equal-area · ICRS · RA increases left · HEALPix NESTED order {order} ({fmtNum(pixelAreaDeg2(order), 3)} deg² cells)
        </span>
        <div className="uso-tools">
          {placement !== "dock" && (
            <>
              <SelectField
                label="Order"
                value={String(order)}
                options={SKY_ORDERS.map((o) => ({ value: String(o), label: `${o} · ${fmtNum(pixelAreaDeg2(o), 3)} deg²` }))}
                onChange={(v) => dispatch({ type: "presentation", patch: { skyOrder: Number(v) } })}
              />
              <Segmented<SkyLayer>
                label="Sky layer"
                value={state.presentation.skyLayer}
                options={[
                  { value: "density", label: "Density", title: "Equal-area counts per HEALPix cell" },
                  { value: "entities", label: "Entities", title: "Native entities; click one to focus it" }
                ]}
                onChange={(skyLayer) => dispatch({ type: "presentation", patch: { skyLayer } })}
              />
              <Toggle on={state.presentation.overlays.graticule} onChange={(v) => dispatch({ type: "overlay", key: "graticule", value: v })}>
                Grid
              </Toggle>
              <Toggle on={state.presentation.overlays.galacticPlane} onChange={(v) => dispatch({ type: "overlay", key: "galacticPlane", value: v })}>
                Galactic
              </Toggle>
              <Toggle on={state.presentation.overlays.ecliptic} onChange={(v) => dispatch({ type: "overlay", key: "ecliptic", value: v })}>
                Ecliptic
              </Toggle>
            </>
          )}
          {placement !== "primary" && (
            <button type="button" className="uso-btn uso-btn-quiet" onClick={() => dispatch({ type: "lens", primary: "sky" })} title="Make the Sky the primary lens">
              <Maximize2 aria-hidden="true" /> Expand
            </button>
          )}
        </div>
      </header>
      <div className={`uso-sky-body is-n${shown.length}`}>
        {shown.map((d: DomainId) => {
          const domain = model.domains[d];
          const m = masks[d];
          return (
            <div key={d} className="uso-skycol">
              <div className="uso-skycol-head">
                <DomainBadge domain={d} />
                <span className="uso-skycol-label">{domain.bundle.entities.native_label}</span>
                <EvidenceChips list={domain.bundle.sky.density.evidence} compact />
                <span className="uso-skycol-n">
                  {m.counts.exceptSky === m.counts.total ? `N ${m.counts.total.toLocaleString("en-US")}` : `${m.counts.exceptSky.toLocaleString("en-US")} of ${m.counts.total.toLocaleString("en-US")} in cross-filter`}
                </span>
              </div>
              <SkyMap domain={domain} compact={placement === "dock"} />
              {placement !== "dock" && <SkyLegend domain={domain} />}
            </div>
          );
        })}
      </div>
      {state.mode === "compare" && placement === "primary" && (
        <footer className="uso-panel-foot">
          <span>Synchronized geometry, independent normalizations: each map is scaled to its own population.</span>
          <span title={relation.reason}>
            <CapabilityMark state={relation.state} label={false} /> No cross-broker association: a cell shared by both maps is not a matched object.
          </span>
          <span title={diff.reason}>
            <CapabilityMark state={diff.state} label={false} /> No difference map: {diff.reason.replace(/^Raw /, "raw ")}
          </span>
        </footer>
      )}
      {state.mode !== "compare" && !compact && shown.length === 1 && (
        <CapabilityNote capability={capability(model, shown[0], "sky.coverage")} />
      )}
    </section>
  );
}
