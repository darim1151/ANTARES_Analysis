"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, Link2, X } from "lucide-react";
import { DOMAIN_IDS } from "@/types/observatory";
import { fmtBytes, fmtInt, shortHash } from "@/lib/observatory/format";
import { buildViewManifest, stateToUrl } from "@/lib/observatory/state";
import { useObservatory } from "./ObservatoryContext";
import { CapabilityMark, DomainBadge, EvidenceChip, EvidenceChips, KeyValue } from "./ui";

type Tab = "basis" | "evidence" | "capabilities" | "integrity" | "view";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "basis", label: "Basis" },
  { id: "evidence", label: "Evidence" },
  { id: "capabilities", label: "Capabilities" },
  { id: "integrity", label: "Integrity" },
  { id: "view", label: "View manifest" }
];

function useCopy() {
  const [copied, setCopied] = useState<string | null>(null);
  const copy = async (key: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(key);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      setCopied(null);
    }
  };
  return { copied, copy };
}

export default function ProvenanceDrawer() {
  const { model, state, provenanceOpen, setProvenanceOpen } = useObservatory();
  const [tab, setTab] = useState<Tab>("basis");
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const { copied, copy } = useCopy();
  const { basis, provenance, capabilities, manifest, integrity, manifestSha256 } = model.bundle;
  const view = useMemo(() => buildViewManifest(model, state), [model, state]);
  const viewText = useMemo(() => JSON.stringify(view, null, 2), [view]);

  const drawerRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!provenanceOpen) return;
    // Modal behaviour: focus moves in, Tab cycles inside, and focus returns to the trigger.
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Tab" || !drawerRef.current) return;
      const focusable = [...drawerRef.current.querySelectorAll<HTMLElement>("button, [href], select, [tabindex]:not([tabindex='-1'])")].filter((el) => !el.hasAttribute("disabled"));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      // Deferred: the workspace stays inert until its own effect runs in this commit.
      window.setTimeout(() => opener?.focus?.(), 0);
    };
  }, [provenanceOpen]);

  if (!provenanceOpen) return null;

  return (
    <div className="uso-drawer-layer" onClick={() => setProvenanceOpen(false)}>
      <aside ref={drawerRef} className="uso-drawer" role="dialog" aria-modal="true" aria-label="Basis provenance" onClick={(e) => e.stopPropagation()}>
        <header className="uso-drawer-head">
          <div>
            <p className="uso-eyebrow">Provenance</p>
            <h2>{basis.label}</h2>
            <code>{basis.basis_id}</code>
          </div>
          <button ref={closeRef} type="button" className="uso-iconbtn" aria-label="Close provenance" onClick={() => setProvenanceOpen(false)}>
            <X aria-hidden="true" />
          </button>
        </header>
        <nav className="uso-tabs" role="tablist" aria-label="Provenance sections">
          {TABS.map((t) => (
            <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={tab === t.id ? "is-on" : ""} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="uso-drawer-body" role="tabpanel">
          {tab === "basis" && (
            <>
              <p className="uso-banner">
                <strong>{basis.status}</strong> · science_ready = {String(basis.science_ready)}. {manifest.evidence_policy}
              </p>
              <h3>Independently versioned pins</h3>
              <dl className="uso-kvs">
                {DOMAIN_IDS.map((d) => (
                  <KeyValue
                    key={d}
                    k={<DomainBadge domain={d} />}
                    v={basis.domains[d].build_id}
                    mono
                    note={
                      <>
                        {basis.domains[d].build_kind} · {basis.domains[d].source.repository}
                        {basis.domains[d].source.revision ? ` @ ${shortHash(basis.domains[d].source.revision, 12)}` : ""}
                      </>
                    }
                  />
                ))}
                <KeyValue
                  k="Relation"
                  v={basis.relation ? `${basis.relation.relation_id} ${basis.relation.version}` : "none"}
                  mono
                  note={basis.relation ? basis.relation.method : model.caps.get("relation:relation.cross_broker_association")?.reason}
                />
                <KeyValue k="Semantic contract" v={`${basis.semantic_contract.id} ${basis.semantic_contract.version}`} mono />
                <KeyValue k="Feature registry" v={`${basis.feature_registry.id} ${basis.feature_registry.version}`} mono />
                <KeyValue k="Analysis kernel" v={`${basis.analysis_kernel.id} ${basis.analysis_kernel.version}`} mono />
                <KeyValue k="Bundle" v={`${manifest.bundle_id} · contract ${manifest.contract_version}`} mono note={`as of ${manifest.as_of_utc}`} />
              </dl>
              <h3>Time semantics</h3>
              <dl className="uso-kvs">
                {DOMAIN_IDS.map((d) => (
                  <KeyValue key={d} k={<DomainBadge domain={d} />} v={`${basis.domains[d].time.stored_field} · MJD ${basis.domains[d].time.scale}`} mono note={`${basis.domains[d].time.scale_basis} ${basis.domains[d].time.entity_date_rule}`} />
                ))}
              </dl>
              <h3>Scientific invariants</h3>
              <ul className="uso-invariants">
                {basis.invariants.map((i) => (
                  <li key={i}>{i}</li>
                ))}
              </ul>
            </>
          )}
          {tab === "evidence" && (
            <>
              <h3>Acquisitions at the pinned revisions</h3>
              <div className="uso-tablewrap">
                <table className="uso-table">
                  <thead>
                    <tr>
                      <th>Window (UTC dates, half-open)</th>
                      <th>Upstream state</th>
                      <th>Delivery · admission · science</th>
                      <th>Transport facts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {provenance.acquisitions.map((a) => (
                      <tr key={a.acquisition_id}>
                        <td>
                          <DomainBadge domain={a.domain} quiet /> <b>{a.label}</b>
                          <small>
                            {a.window.start} → {a.window.stop}
                          </small>
                        </td>
                        <td>
                          <code>{a.state}</code>
                        </td>
                        <td>
                          {a.delivery_validation.replace(/_/g, " ").toLowerCase()} · {a.admission === "ADMITTED" ? `admitted (${a.cohort})` : "not admitted"} ·{" "}
                          {a.scientific_status.replace(/_/g, " ").toLowerCase()}
                        </td>
                        <td>
                          {/* Deliberately prose, not an aligned numeric column: these totals are not rates. */}
                          {a.delivery
                            ? `${fmtInt(a.delivery.readable_rows)} readable rows reconciled${a.delivery.terminal_lag === null ? "" : ` (lag ${a.delivery.terminal_lag})`}`
                            : "no validated delivery"}
                          <small>rate comparison prohibited</small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="uso-insp-note">
                Delivered rows are transport reconciliation of each topic (expected = committed = readable, lag 0). They are not Rubin scientific completeness and are never
                compared across windows as alert rates.
              </p>
              <h3>Time windows and their facts</h3>
              {DOMAIN_IDS.map((d) =>
                model.domains[d].bundle.time.windows.map((w) => (
                  <div key={w.id} className="uso-window-facts">
                    <div className="uso-def-head">
                      <DomainBadge domain={d} quiet />
                      <b>{w.label}</b>
                      <code>{w.state}</code>
                      <span className="uso-hint">{w.status_codes.join(" · ")} · rate comparison {w.rate_comparison.toLowerCase()}</span>
                    </div>
                    <p>{w.caveat}</p>
                    <dl className="uso-kvs">
                      {w.facts.map((f) => (
                        <KeyValue
                          key={f.label}
                          k={
                            <>
                              {f.label} <EvidenceChip evidence={f.evidence} compact />
                            </>
                          }
                          v={typeof f.value === "number" ? `${fmtInt(f.value)}${f.unit ? ` ${f.unit}` : ""}` : String(f.value)}
                          mono
                        />
                      ))}
                    </dl>
                  </div>
                ))
              )}
              <h3>Field-level evidence</h3>
              {DOMAIN_IDS.map((d) => (
                <div key={d} className="uso-fieldev">
                  <DomainBadge domain={d} />
                  <ul>
                    {provenance.field_evidence[d].map((f) => (
                      <li key={f.field}>
                        <code>{f.field}</code> <EvidenceChip evidence={f.evidence} compact /> <span>{f.note}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              <h3>Derivations</h3>
              <ul className="uso-invariants">
                {provenance.derivations.map((d) => (
                  <li key={d.id}>
                    <code>{d.id}</code> {d.description}
                  </li>
                ))}
              </ul>
              <h3>Sources</h3>
              <div className="uso-tablewrap">
                <table className="uso-table">
                  <thead>
                    <tr>
                      <th>Path</th>
                      <th>Repository @ revision</th>
                      <th>Evidence</th>
                      <th>sha256</th>
                    </tr>
                  </thead>
                  <tbody>
                    {provenance.sources.map((s) => (
                      <tr key={s.id}>
                        <td>
                          <code>{s.path}</code>
                        </td>
                        <td>
                          {s.repository}
                          {s.revision ? ` @ ${shortHash(s.revision, 7)}` : ""}
                        </td>
                        <td>
                          <EvidenceChip evidence={s.evidence} compact />
                        </td>
                        <td>
                          <code>{shortHash(s.sha256, 12)}</code>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {tab === "capabilities" && (
            <div className="uso-tablewrap">
              <table className="uso-table">
                <thead>
                  <tr>
                    <th>Capability</th>
                    <th>State</th>
                    <th>Evidence</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {capabilities.capabilities.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <code>{c.id}</code>
                        <small>{c.summary}</small>
                      </td>
                      <td>
                        <CapabilityMark state={c.state} />
                        {c.codes.length > 0 && <small className="is-mono">{c.codes.join(" · ")}</small>}
                      </td>
                      <td>{c.evidence.length ? <EvidenceChips list={c.evidence} compact /> : "—"}</td>
                      <td>
                        {c.reason}
                        {c.qualifications.map((q) => (
                          <small key={q}>◐ {q}</small>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {tab === "integrity" && (
            <>
              <p className="uso-banner">
                Method: <strong>{integrity.method}</strong> · manifest sha256 <code>{shortHash(manifestSha256, 16)}</code>. Detail shards are verified when first opened.
              </p>
              <div className="uso-tablewrap">
                <table className="uso-table">
                  <thead>
                    <tr>
                      <th>Payload</th>
                      <th>Status</th>
                      <th>Bytes</th>
                      <th>sha256</th>
                    </tr>
                  </thead>
                  <tbody>
                    {integrity.files.map((f) => (
                      <tr key={f.path}>
                        <td>
                          <code>{f.path}</code>
                        </td>
                        <td>
                          {f.status === "VERIFIED" ? (
                            <span className="uso-ok">
                              <Check aria-hidden="true" /> verified
                            </span>
                          ) : (
                            f.status.toLowerCase()
                          )}
                        </td>
                        <td className="is-num">{fmtBytes(manifest.files.find((m) => m.path === f.path)?.bytes ?? 0)}</td>
                        <td>
                          <code>{shortHash(f.expected, 16)}</code>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
          {tab === "view" && (
            <>
              <p className="uso-insp-note">
                A reproducible description of this view: the pinned basis and every part of the scientific state. Evidence in view:{" "}
                <EvidenceChips list={view.evidence_in_view} compact />
              </p>
              <div className="uso-drawer-actions">
                <button type="button" className="uso-btn" onClick={() => copy("json", viewText)}>
                  {copied === "json" ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />} Copy JSON
                </button>
                <button type="button" className="uso-btn uso-btn-quiet" onClick={() => copy("url", `${window.location.origin}${window.location.pathname}${stateToUrl(state)}`)}>
                  {copied === "url" ? <Check aria-hidden="true" /> : <Link2 aria-hidden="true" />} Copy view link
                </button>
              </div>
              <pre className="uso-json">{viewText}</pre>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
