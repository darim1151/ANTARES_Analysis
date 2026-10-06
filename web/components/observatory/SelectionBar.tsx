"use client";

import { X } from "lucide-react";
import { DOMAIN_IDS, type DomainId, type FeatureRange } from "@/types/observatory";
import { fmtFixed, fmtInt, fmtNum } from "@/lib/observatory/format";
import { lastDate } from "@/lib/observatory/model";
import { ENTITY_NOUN } from "@/lib/observatory/theme";
import { pixelAreaDeg2 } from "@/lib/observatory/kernel/healpix";
import { displayedDomains } from "@/lib/observatory/state";
import { useObservatory } from "./ObservatoryContext";
import { admissionLabel, admissionSummary, DomainBadge } from "./ui";

function rangeText(range: FeatureRange, short: string, unit: string | null) {
  return `${short} ${fmtNum(range.min, 3)} – ${fmtNum(range.max, 3)}${unit ? ` ${unit}` : ""}`;
}

export default function SelectionBar() {
  const { model, state, dispatch, masks, admission, notices, dismissNotice } = useObservatory();
  const { time, sky, feature } = state.selection;
  const shown = displayedDomains(state.mode);
  const active = Boolean(time || sky || Object.keys(feature).length);

  return (
    <section className="uso-selectionbar" aria-label="Shared selection">
      <span className="uso-eyebrow">Selection</span>
      <div className="uso-predicates">
        {!active && <span className="uso-hint">None · choose UTC dates, sky cells or a cone, or brush the Lab</span>}
        {time && (
          <span
            className="uso-pred"
            title={shown.map((d) => `${d.toUpperCase()}: ${model.domains[d].bundle.time.semantics.entity_date_rule}`).join("\n")}
          >
            <b>Time</b>
            {time.start === lastDate(time.stop) ? time.start : `${time.start} → ${lastDate(time.stop)}`} UTC dates
            <button type="button" aria-label="Clear time selection" onClick={() => dispatch({ type: "time", time: null })}>
              <X aria-hidden="true" />
            </button>
          </span>
        )}
        {sky && (
          <span className="uso-pred">
            <b>Sky</b>
            {sky.kind === "cone"
              ? `cone r = ${fmtFixed(sky.radius_deg, 2)}° at (${fmtFixed(sky.ra, 2)}°, ${fmtFixed(sky.dec, 2)}°)`
              : `${sky.pixels.length} HEALPix cell${sky.pixels.length > 1 ? "s" : ""} · order ${sky.order} · ${fmtNum(sky.pixels.length * pixelAreaDeg2(sky.order), 3)} deg²`}
            <button type="button" aria-label="Clear sky selection" onClick={() => dispatch({ type: "sky", sky: null })}>
              <X aria-hidden="true" />
            </button>
          </span>
        )}
        {DOMAIN_IDS.map((d: DomainId) => {
          const p = feature[d];
          if (!p) return null;
          const dx = model.domains[d].dims.get(p.x.dimension);
          const dy = p.y ? model.domains[d].dims.get(p.y.dimension) : null;
          const suspended = !shown.includes(d);
          return (
            <span key={d} className={`uso-pred${suspended ? " is-suspended" : ""}`} title={suspended ? `Retained; applies only to ${d.toUpperCase()} entities, which are not displayed in this mode.` : undefined}>
              <b>Feature</b>
              <DomainBadge domain={d} quiet />
              {dx && rangeText(p.x, dx.short, dx.unit)}
              {dy && p.y && <> × {rangeText(p.y, dy.short, dy.unit)}</>}
              {suspended && <em>retained · not shown</em>}
              <button type="button" aria-label={`Clear ${d} feature selection`} onClick={() => dispatch({ type: "feature", domain: d, predicate: null })}>
                <X aria-hidden="true" />
              </button>
            </span>
          );
        })}
      </div>
      <div className="uso-counts" aria-live="polite">
        {shown.map((d) => {
          const a = admission[d];
          const noun = ENTITY_NOUN[model.domains[d].bundle.entities.entity_kind].many;
          const sample = model.domains[d].complete ? "" : " (sample)";
          if (a?.status === "NONE") {
            return (
              <span key={d} className="uso-count is-unadmitted" title={admissionSummary(a)}>
                <DomainBadge domain={d} quiet />
                <b>—</b>
                <span>{admissionLabel(a)}</span>
              </span>
            );
          }
          return (
            <span key={d} className="uso-count" title={a?.status === "PARTIAL" ? `${a.admitted}/${a.total} dates admitted; ${admissionSummary(a)}` : undefined}>
              <DomainBadge domain={d} quiet />
              <b>{fmtInt(masks[d].counts.all)}</b>
              <span>
                / {fmtInt(masks[d].counts.total)} {noun}
                {sample}
                {a?.status === "PARTIAL" ? ` · ${a.admitted}/${a.total} dates admitted` : ""}
              </span>
            </span>
          );
        })}
        {active && (
          <button type="button" className="uso-btn uso-btn-quiet" onClick={() => dispatch({ type: "clearSelection" })}>
            Clear all
          </button>
        )}
      </div>
      {notices.map((n, i) => (
        <p key={n} className="uso-notice" role="status">
          {n}
          <button type="button" aria-label="Dismiss notice" onClick={() => dismissNotice(i)}>
            <X aria-hidden="true" />
          </button>
        </p>
      ))}
    </section>
  );
}
