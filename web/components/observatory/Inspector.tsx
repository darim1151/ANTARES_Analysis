"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Info, X } from "lucide-react";
import type {
  AntaresLocusDetail,
  AntaresLocusSummary,
  DomainId,
  EntityDetail,
  FileIntegrity,
  FinkDiaObjectDetail,
  FinkDiaObjectSummary,
  NativeEntityRef
} from "@/types/observatory";
import { icrsToEcliptic, icrsToGalactic } from "@/lib/observatory/kernel/astro";
import { radecToPix } from "@/lib/observatory/kernel/healpix";
import { decDms, fmtFixed, fmtInt, fmtNum, raHms, shortHash } from "@/lib/observatory/format";
import { capability, entityRef, evidenceUnion, type DomainModel } from "@/lib/observatory/model";
import { displayedDomains } from "@/lib/observatory/state";
import { ENTITY_NOUN } from "@/lib/observatory/theme";
import { useObservatory } from "./ObservatoryContext";
import { DiaSourceFluxPlot, MagnitudePlot } from "./LightcurvePlot";
import { admissionLabel, AdmissionNote, CapabilityMark, CapabilityNote, DomainBadge, EvidenceChip, EvidenceChips, KeyValue } from "./ui";

type DetailState =
  | { status: "idle" }
  | { status: "loading"; key: string }
  | { status: "ready"; key: string; detail: EntityDetail | null; integrity: FileIntegrity }
  | { status: "error"; key: string; message: string };

const keyOf = (ref: NativeEntityRef) => `${ref.kind}:${ref.id}`;

function useEntityDetail(ref: NativeEntityRef | null): DetailState {
  const { model, reader } = useObservatory();
  const [state, setState] = useState<DetailState>({ status: "idle" });
  useEffect(() => {
    if (!ref) {
      setState({ status: "idle" });
      return;
    }
    let live = true;
    const key = keyOf(ref);
    setState({ status: "loading", key });
    reader
      .loadEntityDetail(model.bundle, ref)
      .then(({ detail, integrity }) => live && setState({ status: "ready", key, detail, integrity }))
      .catch((error: Error) => live && setState({ status: "error", key, message: error.message }));
    return () => {
      live = false;
    };
  }, [model.bundle, reader, ref]);
  return state;
}

function Section({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="uso-insp-section">
      <h3>
        {title}
        {aside}
      </h3>
      {children}
    </section>
  );
}

function PositionSection({ ra, dec }: { ra: number; dec: number }) {
  const [l, b] = icrsToGalactic(ra, dec);
  const [, beta] = icrsToEcliptic(ra, dec);
  return (
    <Section title="Position · ICRS">
      <dl className="uso-kvs">
        <KeyValue k="RA" v={`${fmtFixed(ra, 5)}°`} mono note={raHms(ra)} />
        <KeyValue k="Dec" v={`${fmtFixed(dec, 5)}°`} mono note={decDms(dec)} />
        <KeyValue k="Galactic l, b" v={`${fmtFixed(l, 3)}°, ${fmtFixed(b, 3)}°`} mono />
        <KeyValue k="Ecliptic β" v={`${fmtFixed(beta, 3)}°`} mono note="mean J2000" />
        <KeyValue k="HEALPix" v={`order 6 · ${radecToPix(6, ra, dec)}`} mono note="NESTED" />
      </dl>
    </Section>
  );
}

function AntaresRecord({ domain, summary, detail }: { domain: DomainModel; summary: AntaresLocusSummary; detail: AntaresLocusDetail | null }) {
  const { model } = useObservatory();
  const fields = new Map(domain.bundle.entities.fields.map((f) => [f.key, f]));
  const lcCap = capability(model, "antares", "entity.lightcurve");
  const tagsCap = capability(model, "antares", "entity.broker_inference");
  return (
    <>
      <PositionSection ra={summary.ra} dec={summary.dec} />
      <Section title="Native locus fields">
        <dl className="uso-kvs">
          <KeyValue k={<>Newest alert <EvidenceChip evidence={fields.get("newest_alert_observation_time")!.evidence} compact /></>} v={fmtFixed(summary.newest_alert_observation_time, 6)} mono note={`MJD, UTC-treated · UTC date ${summary.entity_date}`} />
          <KeyValue
            k={<>Brightest alert mag <EvidenceChip evidence={fields.get("brightest_alert_magnitude")!.evidence} compact /></>}
            v={summary.brightest_alert_magnitude === null ? "—" : fmtFixed(summary.brightest_alert_magnitude, 3)}
            mono
            note={summary.brightest_alert_magnitude === null ? "clipped by the legacy demo exporter; true value unknown here" : "band/survey not carried in this basis"}
          />
          <KeyValue k={<>Magnitude values <EvidenceChip evidence={fields.get("num_mag_values")!.evidence} compact /></>} v={fmtInt(summary.num_mag_values)} mono />
        </dl>
      </Section>
      <Section title="ANTARES tags" aside={<CapabilityMark state={tagsCap.state} />}>
        <p className="uso-insp-note">Filter-pipeline memberships. Not astrophysical classes, not scores.</p>
        <div className="uso-tags">
          {summary.tags.map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
      </Section>
      <Section title="Brightness history" aside={<CapabilityMark state={lcCap.state} />}>
        {detail?.lightcurve ? (
          <>
            <MagnitudePlot points={detail.lightcurve.points} evidence={detail.lightcurve.evidence} timeScale={detail.lightcurve.time_scale} />
            <p className="uso-insp-note">
              <EvidenceChip evidence={detail.lightcurve.evidence} /> {detail.lightcurve.label}
            </p>
          </>
        ) : (
          <p className="uso-empty">{detail?.lightcurve_unavailable_reason ?? "Loading…"}</p>
        )}
      </Section>
    </>
  );
}

function FinkRecord({ domain, summary, detail }: { domain: DomainModel; summary: FinkDiaObjectSummary; detail: FinkDiaObjectDetail | null }) {
  const { model } = useObservatory();
  const evidence = domain.bundle.entities.fields[0].evidence;
  const inference = capability(model, "fink", "entity.broker_inference");
  const snaps = detail?.snapshots ?? [];
  const rows: Array<{ label: string; get: (i: number) => string }> = [
    { label: "midpointMjdTai", get: (i) => fmtFixed(snaps[i].midpointMjdTai, 5) },
    { label: "pred.is_first", get: (i) => String(snaps[i].pred.is_first) },
    { label: "pred.is_cataloged", get: (i) => String(snaps[i].pred.is_cataloged) },
    { label: "clf.snnSnVsOthers_score", get: (i) => fmtNum(snaps[i].clf.snnSnVsOthers_score, 3) },
    { label: "clf.cats_score", get: (i) => fmtNum(snaps[i].clf.cats_score, 3) },
    { label: "clf.cats_class", get: (i) => (snaps[i].clf.cats_class === null || snaps[i].clf.cats_class === undefined ? "—" : `code ${snaps[i].clf.cats_class}`) },
    { label: "clf.earlySNIa_score", get: (i) => fmtNum(snaps[i].clf.earlySNIa_score, 3) },
    { label: "xm.simbad_otype", get: (i) => String(snaps[i].xm.simbad_otype ?? "—") },
    { label: "lc_features[r].chi2", get: (i) => fmtNum(snaps[i].lc_features.r?.chi2 ?? null, 3) },
    { label: "lc_features[r].stetson_K", get: (i) => fmtNum(snaps[i].lc_features.r?.stetson_K ?? null, 3) },
    { label: "lc_features bands", get: (i) => Object.keys(snaps[i].lc_features).join(" ") || "—" }
  ];
  return (
    <>
      <PositionSection ra={summary.ra} dec={summary.dec} />
      <Section title="Delivered DiaSources" aside={<EvidenceChip evidence={evidence} compact />}>
        <dl className="uso-kvs">
          <KeyValue k="Delivered rows" v={fmtInt(summary.n_dia_sources)} mono note="in this basis; Light Static has no complete history" />
          <KeyValue k="First midpointMjdTai" v={fmtFixed(summary.first_midpoint_mjd_tai, 6)} mono note={`TAI · first UTC date ${summary.entity_date}`} />
          <KeyValue k="Last midpointMjdTai" v={fmtFixed(summary.last_midpoint_mjd_tai, 6)} mono note={`TAI · Δt ${fmtFixed(summary.last_midpoint_mjd_tai - summary.first_midpoint_mjd_tai, 2)} d`} />
          <KeyValue k="Bands" v={summary.bands.join(" ")} mono />
        </dl>
        {detail ? <DiaSourceFluxPlot sources={detail.sources} /> : <p className="uso-empty">Loading delivered DiaSources…</p>}
        <p className="uso-insp-note">Difference-image psfFlux with 1σ psfFluxErr. No forced photometry or upper limits in Light Static.</p>
      </Section>
      <Section title="Broker inference · source-time snapshots" aside={<CapabilityMark state={inference.state} />}>
        <p className="uso-insp-note">
          Each column is the Fink output delivered with one DiaSource. Classifier outputs change as alerts arrive; they are not a timeless classification. Scores are not calibrated probabilities.
        </p>
        {detail ? (
          <div className="uso-snaptable-wrap">
            <table className="uso-snaptable">
              <thead>
                <tr>
                  <th>field</th>
                  {snaps.map((s, i) => (
                    <th key={s.diaSourceId} title={`diaSourceId ${s.diaSourceId}`}>
                      {i === 0 ? "first" : i === snaps.length - 1 ? "latest" : "prior"}
                      <small>…{s.diaSourceId.slice(-6)}</small>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.label}>
                    <th>{r.label}</th>
                    {snaps.map((s, i) => (
                      <td key={s.diaSourceId}>{r.get(i)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="uso-insp-note">{detail.snapshot_policy}</p>
          </div>
        ) : (
          <p className="uso-empty">Loading snapshots…</p>
        )}
      </Section>
    </>
  );
}

function FocusView({ focus }: { focus: NativeEntityRef }) {
  const { model, state, dispatch } = useObservatory();
  const domain = model.domains[focus.domain];
  const index = domain.indexById.get(focus.id);
  const detailState = useEntityDetail(focus);
  const shown = displayedDomains(state.mode);
  const relation = capability(model, "relation", "relation.cross_broker_association");
  if (index === undefined) return <p className="uso-empty">This entity is not part of the pinned basis.</p>;
  const summary = domain.records[index];
  const ready = detailState.status === "ready" && detailState.key === keyOf(focus) ? detailState : null;
  return (
    <div className="uso-insp-focus" key={keyOf(focus)}>
      <div className="uso-insp-id">
        <DomainBadge domain={focus.domain} />
        <span className="uso-insp-kind">{domain.bundle.entities.native_label}</span>
        <button type="button" className="uso-iconbtn" aria-label="Clear focus" onClick={() => dispatch({ type: "focus", focus: null })}>
          <X aria-hidden="true" />
        </button>
      </div>
      <p className="uso-insp-native">
        <span>{domain.bundle.entities.id_field}</span>
        <code>{summary.id}</code>
      </p>
      {/* The record's own evidence, not the basis pin (which also carries domain-level transport evidence). */}
      <EvidenceChips list={evidenceUnion([domain.bundle.entities.fields.map((f) => f.evidence)])} />
      {!shown.includes(focus.domain) && (
        <p className="uso-banner" role="note">
          <Info aria-hidden="true" />
          This {focus.domain.toUpperCase()} {ENTITY_NOUN[focus.kind].one} is retained from an earlier view. {relation.reason}
        </p>
      )}
      {summary.kind === "antares.locus" ? (
        <AntaresRecord domain={domain} summary={summary} detail={ready?.detail?.kind === "antares.locus" ? ready.detail : null} />
      ) : (
        <FinkRecord domain={domain} summary={summary} detail={ready?.detail?.kind === "fink.diaObject" ? ready.detail : null} />
      )}
      {detailState.status === "error" && <p className="uso-banner is-error">{detailState.message}</p>}
      <Section title="Record provenance">
        <dl className="uso-kvs">
          <KeyValue k="Build" v={domain.pin.build_id} mono />
          <KeyValue k="Ontology" v={domain.pin.native_ontology} />
          {ready && <KeyValue k="Detail shard" v={ready.integrity.path} mono note={`sha256 ${shortHash(ready.integrity.actual ?? ready.integrity.expected)} · ${ready.integrity.status.toLowerCase()}`} />}
        </dl>
      </Section>
    </div>
  );
}

function memberKey(domain: DomainModel): string | null {
  return domain.selectable.find((d) => d.family === "multiplicity")?.id ?? null;
}

function MembersList({ d }: { d: DomainId }) {
  const { model, masks, state, dispatch, admission } = useObservatory();
  const domain = model.domains[d];
  const mask = masks[d].all;
  const key = memberKey(domain);
  const yDim = domain.dims.get(state.lens.lab[d].y)!;
  const yCol = domain.cols.features.get(yDim.id)!;
  const members = useMemo(() => {
    const idx: number[] = [];
    for (let i = 0; i < domain.cols.n; i += 1) if (mask[i]) idx.push(i);
    const col = key ? domain.cols.features.get(key) : null;
    if (col) idx.sort((a, b) => (col[b] || 0) - (col[a] || 0));
    return idx.slice(0, 40);
  }, [domain, key, mask]);
  const keyDim = key ? domain.dims.get(key) : null;
  const keyCol = key ? domain.cols.features.get(key) : null;
  const noun = ENTITY_NOUN[domain.bundle.entities.entity_kind];
  return (
    <div className="uso-members">
      <div className="uso-members-head">
        <DomainBadge domain={d} />
        <span>
          {admission[d]?.status === "NONE" ? admissionLabel(admission[d]!) : `${fmtInt(masks[d].counts.all)} ${noun.many} · top ${members.length} by ${keyDim?.short ?? "id"}`}
        </span>
      </div>
      <ol>
        {members.map((i) => (
          <li key={i}>
            <button type="button" onClick={() => dispatch({ type: "focus", focus: entityRef(domain, i) })}>
              <code>{domain.records[i].id}</code>
              <span>{domain.records[i].entity_date}</span>
              <span>
                {keyDim?.short} {fmtNum(keyCol ? keyCol[i] : NaN, 3)}
              </span>
              <span>
                {yDim.short} {fmtNum(yCol[i], 3)}
              </span>
            </button>
          </li>
        ))}
      </ol>
      <AdmissionNote domain={d} admission={admission[d]} block />
      {members.length === 0 && admission[d]?.status !== "NONE" && <p className="uso-empty">No {noun.many} satisfy every active predicate.</p>}
    </div>
  );
}

export default function Inspector() {
  const { model, state, dispatch } = useObservatory();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // A newly opened record (or the return to the selection) starts at the top.
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [state.focus]);
  const shown = displayedDomains(state.mode);
  const active = Boolean(state.selection.time || state.selection.sky || Object.keys(state.selection.feature).length);
  return (
    <section className="uso-panel uso-inspector" aria-label="Inspector">
      <header className="uso-panel-head">
        <h2>Inspector</h2>
        <span className="uso-sub">{state.focus ? "Native record" : "Selection"}</span>
        {state.focus && (
          <div className="uso-tools">
            <button type="button" className="uso-btn uso-btn-quiet" onClick={() => dispatch({ type: "focus", focus: null })}>
              <ArrowLeft aria-hidden="true" /> Selection
            </button>
          </div>
        )}
      </header>
      <div className="uso-panel-scroll" ref={scrollRef}>
        {state.focus ? (
          <FocusView focus={state.focus} />
        ) : (
          <div className="uso-insp-selection">
            {!active && (
              <p className="uso-insp-intro">
                Select UTC dates in Time, cells or a cone on the Sky, or brush the Lab. Members of the shared selection appear here; open one to see its native record. A date selection applies each domain&apos;s own date rule: ANTARES assigns a locus to its newest-alert date, Fink a DiaObject to its first delivered DiaSource.
              </p>
            )}
            {shown.map((d) => (
              <MembersList key={d} d={d} />
            ))}
            {shown.map((d) => (
              <CapabilityNote key={`cap-${d}`} capability={capability(model, d, "entity.lightcurve")} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
