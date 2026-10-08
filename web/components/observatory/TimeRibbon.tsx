"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { DOMAIN_IDS, type DomainId, type NightRecord, type NightState, type TimeWindow } from "@/types/observatory";
import { fmtInt, shortDate } from "@/lib/observatory/format";
import { capability, lastDate } from "@/lib/observatory/model";
import { addDays } from "@/lib/observatory/kernel/dates";
import { displayedDomains } from "@/lib/observatory/state";
import { DOMAIN_COLOR, ENTITY_NOUN, NIGHT_STATE_HELP, NIGHT_STATE_LABEL, TOKENS, withAlpha } from "@/lib/observatory/theme";
import { useObservatory } from "./ObservatoryContext";
import { CapabilityMark, DomainBadge, EvidenceChips } from "./ui";

const WIN_H = 16;
const BAR_H = 30;
const STATE_H = 7;
const LANE_GAP = 10;
const LANE_H = WIN_H + BAR_H + STATE_H + 3;
const AXIS_H = 22;

const CODE_WORDS: Record<string, string> = {
  DELIVERY_VALIDATED: "delivery-validated",
  NOT_DELIVERY_VALIDATED: "not delivery-validated",
  ADMITTED: "admitted",
  NOT_ADMITTED: "not admitted",
  CHARACTERIZED: "characterized",
  UNCHARACTERIZED: "uncharacterized",
  LEGACY_DEMO_EXPORT: "",
  SYNTHETIC_DATE_ASSIGNMENT: "synthetic date assignment"
};

function windowText(w: TimeWindow): string {
  const head = w.label.split(" · ")[0];
  const words = w.status_codes.map((c) => CODE_WORDS[c] ?? c.toLowerCase()).filter(Boolean);
  if (w.state === "UNAVAILABLE") words.unshift(w.source_state.toLowerCase().replace(/_/g, " "));
  if (w.rate_comparison === "PROHIBITED" && w.state !== "UNAVAILABLE") words.push("no rate comparison");
  return [head, ...words].join(" · ");
}

const STATES_IN_LEGEND: NightState[] = ["AVAILABLE", "ZERO", "UNQUALIFIED", "UNAVAILABLE", "MISSING", "OUTSIDE_COVERAGE"];

function StateSwatch({ state }: { state: NightState }) {
  return <i className={`uso-night uso-night-${state.toLowerCase()}`} aria-hidden="true" />;
}

export default function TimeRibbon() {
  const { model, state, dispatch, masks } = useObservatory();
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(900);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ from: number; to: number } | null>(null);
  const dragRef = useRef<{ from: number; to: number } | null>(null);
  const axis = model.dateAxis;
  const shown = displayedDomains(state.mode);

  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(320, Math.floor(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const dayW = width / axis.length;
  const x = (i: number) => i * dayW;
  const indexOf = useMemo(() => new Map(axis.map((d, i) => [d, i])), [axis]);
  const lanes = useMemo(
    () =>
      DOMAIN_IDS.map((d) => {
        const dm = model.domains[d];
        const time = dm.bundle.time;
        const nights = new Map<string, NightRecord>(time.nights.map((n) => [n.date, n]));
        const base = time.counts?.values ?? {};
        // A sampled entity layer is never drawn against population counts.
        const subsetActive = dm.complete && Boolean(state.selection.sky || masks[d].featureActive);
        const selected: Record<string, number> = {};
        if (subsetActive) {
          const m = masks[d].exceptTime;
          for (let i = 0; i < dm.cols.n; i += 1) if (m[i]) selected[dm.cols.dates[i]] = (selected[dm.cols.dates[i]] ?? 0) + 1;
        }
        const max = Math.max(1, ...Object.values(base));
        // Synthetic count series are hatched so they never read as delivered counts.
        const synthetic = Boolean(time.counts?.evidence.some((e) => e === "SYNTHETIC_DEMO" || e === "SYNTHETIC_FIXTURE"));
        return { d, time, nights, base, selected, subsetActive, max, synthetic, countsCap: capability(model, d, "time.date_counts") };
      }),
    [masks, model, state.selection.sky]
  );

  const countUnit = (d: DomainId) => model.domains[d].bundle.time.counts?.unit ?? ENTITY_NOUN[model.domains[d].bundle.entities.entity_kind].many;

  const sel = state.selection.time;
  const selFrom = sel ? indexOf.get(sel.start) ?? 0 : null;
  const selTo = sel ? (indexOf.get(lastDate(sel.stop)) ?? axis.length - 1) : null;
  const svgH = lanes.length * LANE_H + (lanes.length - 1) * LANE_GAP + AXIS_H;

  function pointerIndex(clientX: number) {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect) return 0;
    return Math.max(0, Math.min(axis.length - 1, Math.floor((clientX - rect.left) / dayW)));
  }

  function commit(from: number, to: number) {
    const a = Math.min(from, to);
    const b = Math.max(from, to);
    const start = axis[a];
    const stop = addDays(axis[b], 1);
    if (sel && sel.start === start && sel.stop === stop) dispatch({ type: "time", time: null });
    else dispatch({ type: "time", time: { kind: "utc_dates", start, stop } });
  }

  function onKeyDown(event: React.KeyboardEvent) {
    const current = hoverIndex ?? selFrom ?? 0;
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      const next = Math.max(0, Math.min(axis.length - 1, current + (event.key === "ArrowRight" ? 1 : -1)));
      setHoverIndex(next);
      if (event.shiftKey && selFrom !== null) commit(selFrom, next);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      commit(current, current);
    }
  }

  // Month starts always get a tick; day ticks are thinned and never crowd a month label.
  // The first date carries a month label too, so it anchors the thinning like a month start.
  const monthStarts = [0, ...axis.map((date, i) => (date.endsWith("-01") && i > 0 ? i : -1)).filter((i) => i >= 0)];
  const clearOfMonth = (i: number) => monthStarts.every((m) => Math.abs(m - i) * dayW >= 34);
  const ticks = axis
    .map((date, i) => ({ date, i }))
    .filter(({ date, i }) => i === 0 || date.endsWith("-01") || (clearOfMonth(i) && (dayW >= 20 || (dayW >= 9 ? i % 7 === 0 : i % 14 === 0))));

  const hovered = hoverIndex !== null ? axis[hoverIndex] : null;

  return (
    <section className="uso-panel uso-ribbon" aria-label="Time: broker-aware UTC-date lanes">
      <header className="uso-panel-head">
        <h2>Time</h2>
        <span className="uso-sub">UTC-date bins, half-open [00:00, 24:00) UTC · each lane maps its own time scale · lanes never share a denominator</span>
        <ul className="uso-legend uso-legend-states" aria-label="Date states">
          {STATES_IN_LEGEND.map((s) => (
            <li key={s} title={NIGHT_STATE_HELP[s]}>
              <StateSwatch state={s} />
              {NIGHT_STATE_LABEL[s]}
            </li>
          ))}
          <li title="Count series whose evidence is synthetic (demo or fixture): never delivered counts">
            <i className="uso-night uso-night-synthetic" aria-hidden="true" />
            Synthetic counts
          </li>
          <li title="Shared UTC-date selection">
            <i className="uso-night uso-night-selected" aria-hidden="true" />
            Selected
          </li>
        </ul>
      </header>
      <div className="uso-ribbon-body">
        <div className="uso-ribbon-gutter">
          {lanes.map((lane, li) => (
            <div key={lane.d} className={`uso-lanelabel${shown.includes(lane.d) ? "" : " is-context"}`} style={{ height: LANE_H, marginBottom: li < lanes.length - 1 ? LANE_GAP : 0 }}>
              <div className="uso-lanelabel-top">
                <DomainBadge domain={lane.d} />
                <EvidenceChips list={lane.countsCap.evidence} compact />
              </div>
              <span className="uso-lanelabel-q">
                {countUnit(lane.d)} / date · max {fmtInt(lane.max)}
              </span>
              <span className="uso-lanelabel-t" title={`${lane.time.semantics.scale_basis} ${lane.time.semantics.date_binning}`}>
                {lane.time.semantics.stored_field} · MJD {lane.time.semantics.scale_label}
              </span>
            </div>
          ))}
        </div>
        <div
          className="uso-ribbon-plot"
          ref={plotRef}
          tabIndex={0}
          role="group"
          aria-label="UTC-date selection. Drag or use arrow keys and Enter to select dates."
          onKeyDown={onKeyDown}
          onPointerMove={(e) => {
            const i = pointerIndex(e.clientX);
            setHoverIndex(i);
            if (dragRef.current) {
              dragRef.current = { ...dragRef.current, to: i };
              setDrag(dragRef.current);
            }
          }}
          onPointerLeave={() => setHoverIndex(null)}
          onPointerDown={(e) => {
            (e.target as Element).setPointerCapture?.(e.pointerId);
            const i = pointerIndex(e.clientX);
            dragRef.current = { from: i, to: i };
            setDrag(dragRef.current);
          }}
          onPointerUp={() => {
            const gesture = dragRef.current;
            dragRef.current = null;
            if (gesture) commit(gesture.from, gesture.to);
            setDrag(null);
          }}
        >
          <svg width={width} height={svgH} role="img" aria-label="Per-domain date lanes with counts and evidence states">
            <defs>
              {DOMAIN_IDS.map((d) => (
                <pattern key={d} id={`uso-hatch-${d}`} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="4" height="4" fill={withAlpha(DOMAIN_COLOR[d], 0.55)} />
                  <line x1="0" y1="0" x2="0" y2="4" stroke={DOMAIN_COLOR[d]} strokeWidth="2.2" />
                </pattern>
              ))}
            </defs>
            {lanes.map((lane, li) => {
              const top = li * (LANE_H + LANE_GAP);
              const barTop = top + WIN_H;
              const stateTop = barTop + BAR_H + 2;
              const color = DOMAIN_COLOR[lane.d];
              const context = !shown.includes(lane.d);
              return (
                <g key={lane.d} opacity={context ? 0.42 : 1}>
                  <line x1={0} x2={width} y1={stateTop + STATE_H / 2} y2={stateTop + STATE_H / 2} stroke={TOKENS.hairline} />
                  {lane.time.windows.map((w, wi, all) => {
                    const a = indexOf.get(w.start);
                    const b = indexOf.get(lastDate(w.stop));
                    if (a === undefined || b === undefined) return null;
                    const x0 = x(a) + 1;
                    const x1 = x(b + 1) - 1;
                    const text = windowText(w);
                    // A label may run into free space up to the next window, never across it.
                    const nextStart = all[wi + 1] ? indexOf.get(all[wi + 1].start) : undefined;
                    const limit = nextStart !== undefined ? x(nextStart) - 1 : width;
                    const room = (limit - x0 - 8) / 5.4;
                    const label = text.length <= room ? text : w.label.split(" · ")[0].length <= room ? w.label.split(" · ")[0] : "";
                    return (
                      <g key={w.id} className={`uso-window uso-window-${w.state.toLowerCase()}`}>
                        <path d={`M${x0},${top + WIN_H - 2}V${top + WIN_H - 7}H${x1}V${top + WIN_H - 2}`} fill="none" />
                        {label && (
                          <text x={x0 + 4} y={top + WIN_H - 9}>
                            {label}
                          </text>
                        )}
                        <title>{`${w.label}\n${w.caveat}`}</title>
                      </g>
                    );
                  })}
                  {axis.map((date, i) => {
                    const night = lane.nights.get(date);
                    const st: NightState = night?.state ?? "OUTSIDE_COVERAGE";
                    const n = lane.base[date];
                    const bw = Math.max(1, Math.min(24, dayW - 2));
                    const bx = x(i) + (dayW - bw) / 2;
                    const bars = [];
                    if (n !== undefined && n > 0) {
                      const h = Math.max(1.5, (n / lane.max) * BAR_H);
                      bars.push(
                        <rect
                          key="b"
                          x={bx}
                          y={barTop + BAR_H - h}
                          width={bw}
                          height={h}
                          rx={Math.min(2, bw / 3)}
                          fill={lane.subsetActive ? withAlpha(color, 0.28) : lane.synthetic ? `url(#uso-hatch-${lane.d})` : color}
                        />
                      );
                      if (lane.subsetActive) {
                        const s = lane.selected[date] ?? 0;
                        if (s > 0) {
                          const hs = Math.max(1.5, (s / lane.max) * BAR_H);
                          bars.push(
                            <rect key="s" x={bx} y={barTop + BAR_H - hs} width={bw} height={hs} rx={Math.min(2, bw / 3)} fill={lane.synthetic ? `url(#uso-hatch-${lane.d})` : color} />
                          );
                        }
                      }
                    }
                    return (
                      <g key={date}>
                        {bars}
                        {st !== "OUTSIDE_COVERAGE" && (
                          <rect
                            className={`uso-night-cell uso-night-${st.toLowerCase()}`}
                            x={x(i) + 0.5}
                            y={stateTop}
                            width={Math.max(0.5, dayW - 1)}
                            height={STATE_H}
                          />
                        )}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {sel && selFrom !== null && selTo !== null && (
              <g className="uso-ribbon-sel">
                {/* Recede dates outside the selection; brighten the selected band. */}
                <rect x={0} y={0} width={x(selFrom)} height={svgH - AXIS_H} fill="rgba(11,11,10,0.5)" />
                <rect x={x(selTo + 1)} y={0} width={Math.max(0, width - x(selTo + 1))} height={svgH - AXIS_H} fill="rgba(11,11,10,0.5)" />
                <rect x={x(selFrom)} y={0} width={x(selTo + 1) - x(selFrom)} height={svgH - AXIS_H} fill="rgba(255,255,255,0.11)" />
                <line x1={x(selFrom)} x2={x(selFrom)} y1={0} y2={svgH - AXIS_H} stroke={TOKENS.select} strokeOpacity={0.7} />
                <line x1={x(selTo + 1)} x2={x(selTo + 1)} y1={0} y2={svgH - AXIS_H} stroke={TOKENS.select} strokeOpacity={0.7} />
              </g>
            )}
            {drag && drag.from !== drag.to && (
              <rect
                x={x(Math.min(drag.from, drag.to))}
                y={0}
                width={x(Math.max(drag.from, drag.to) + 1) - x(Math.min(drag.from, drag.to))}
                height={svgH - AXIS_H}
                fill="rgba(255,255,255,0.06)"
                stroke={TOKENS.select}
                strokeOpacity={0.6}
              />
            )}
            {hoverIndex !== null && (
              <line x1={x(hoverIndex) + dayW / 2} x2={x(hoverIndex) + dayW / 2} y1={0} y2={svgH - AXIS_H} stroke={TOKENS.secondary} strokeOpacity={0.45} />
            )}
            <g className="uso-axis" transform={`translate(0, ${svgH - AXIS_H})`}>
              <line x1={0} x2={width} y1={0.5} y2={0.5} stroke={TOKENS.axis} />
              {ticks.map(({ date, i }) => (
                <g key={date} transform={`translate(${x(i) + 0.5}, 0)`}>
                  <line y1={0} y2={date.endsWith("-01") ? 6 : 3} stroke={TOKENS.axis} />
                  <text y={16} className={date.endsWith("-01") ? "is-month" : undefined}>
                    {date.endsWith("-01") || i === 0 ? shortDate(date) : Number(date.slice(8))}
                  </text>
                </g>
              ))}
            </g>
          </svg>
          {hovered && hoverIndex !== null && (
            <div className="uso-tip" style={{ left: Math.min(width - 260, Math.max(0, x(hoverIndex) + dayW + 8)), top: 4 }} role="tooltip">
              <strong>{hovered} UTC date</strong>
              {lanes.map((lane) => {
                const night = lane.nights.get(hovered);
                const st: NightState = night?.state ?? "OUTSIDE_COVERAGE";
                const n = lane.base[hovered];
                const win = night?.window_id ? lane.time.windows.find((w) => w.id === night.window_id) : null;
                return (
                  <div key={lane.d} className="uso-tip-row">
                    <DomainBadge domain={lane.d} quiet />
                    <span className="uso-tip-state">
                      <StateSwatch state={st} /> {NIGHT_STATE_LABEL[st]}
                    </span>
                    {n !== undefined && (
                      <span>
                        {fmtInt(n)} {countUnit(lane.d)}
                        {lane.subsetActive && <> · {fmtInt(lane.selected[hovered] ?? 0)} in selection</>}
                      </span>
                    )}
                    {win && <small>{windowText(win)}</small>}
                  </div>
                );
              })}
              {lanes.some((l) => l.countsCap.state !== "AVAILABLE") && (
                <small className="uso-tip-foot">
                  <CapabilityMark state="PARTIALLY_QUALIFIED" label={false} />{" "}
                  {lanes.filter((l) => l.countsCap.state !== "AVAILABLE").map((l) => l.d.toUpperCase()).join(" and ")} counts are sample or fixture allocations,
                  never rates.
                </small>
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
