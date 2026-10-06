"use client";

import Link from "next/link";
import { AlertTriangle, FileJson, ShieldAlert, ShieldCheck } from "lucide-react";
import { DOMAIN_IDS, type SourceMode } from "@/types/observatory";
import { useObservatory } from "./ObservatoryContext";
import { DomainBadge, EvidenceChips, Segmented } from "./ui";

const MODE_OPTIONS: Array<{ value: SourceMode; label: React.ReactNode; title: string }> = [
  { value: "antares", label: <DomainBadge domain="antares" />, title: "ANTARES-native view" },
  { value: "fink", label: <DomainBadge domain="fink" />, title: "Fink-native view" },
  {
    value: "compare",
    label: <span className="uso-compare-label">Compare</span>,
    title: "Side-by-side, independently normalized; no cross-broker association"
  }
];

export default function BasisBar() {
  const { model, state, dispatch, setProvenanceOpen } = useObservatory();
  const { basis, manifest, integrity } = model.bundle;
  const verified = integrity.files.filter((f) => f.status === "VERIFIED").length;
  const allVerified = integrity.method === "sha256" && verified === integrity.files.length;

  return (
    <header className="uso-basisbar">
      <div className="uso-wordmark">
        <span>Unified Scientific Observatory</span>
        <small>First Light</small>
      </div>

      <Segmented label="Source domain" value={state.mode} options={MODE_OPTIONS} onChange={(mode) => dispatch({ type: "mode", mode })} />

      <button type="button" className="uso-basis" onClick={() => setProvenanceOpen(true)} title="Open basis provenance">
        <span className="uso-basis-id">
          <small>Basis</small>
          {basis.basis_id}
        </span>
        {DOMAIN_IDS.map((d) => (
          <span key={d} className="uso-basis-pin">
            <DomainBadge domain={d} quiet />
            {/* Compact display; the full build id is in the tooltip and the provenance drawer. */}
            <code title={basis.domains[d].build_id}>
              {basis.domains[d].build_id
                .split(".")
                .slice(1)
                .join(".")
                .replace(/T\d{2}:\d{2}:\d{2}Z/, "")}
            </code>
            <EvidenceChips list={basis.domains[d].evidence} compact />
          </span>
        ))}
        <span className="uso-basis-pin">
          <span className="uso-domain is-quiet">
            <i aria-hidden="true" className={basis.relation ? undefined : "is-none"} />
            Relation
          </span>
          <code>{basis.relation ? `${basis.relation.relation_id} ${basis.relation.version}` : "none"}</code>
        </span>
      </button>

      <div className="uso-basisbar-end">
        {basis.science_ready ? (
          <span className="uso-status" title={manifest.evidence_policy}>
            <ShieldCheck aria-hidden="true" />
            {basis.status.replace(/_/g, " ").toLowerCase()} basis
          </span>
        ) : (
          <span className="uso-status uso-status-warning" title={manifest.evidence_policy}>
            <AlertTriangle aria-hidden="true" />
            {basis.status === "FIRST_LIGHT_FIXTURE" ? "Fixture basis" : `${basis.status.replace(/_/g, " ").toLowerCase()} basis`} · not science-ready
          </span>
        )}
        <button
          type="button"
          className={`uso-integrity${allVerified ? "" : " is-warn"}`}
          onClick={() => setProvenanceOpen(true)}
          title={allVerified ? "Every loaded payload matched its manifest sha256" : "Some payloads could not be verified"}
        >
          {allVerified ? <ShieldCheck aria-hidden="true" /> : <ShieldAlert aria-hidden="true" />}
          {integrity.method === "sha256" ? `${verified}/${integrity.files.length} sha256 ✓` : "integrity unverified"}
        </button>
        <button type="button" className="uso-btn uso-btn-quiet" onClick={() => setProvenanceOpen(true)}>
          <FileJson aria-hidden="true" />
          Provenance
        </button>
        <Link className="uso-link" href="/">
          SkyPulse
        </Link>
      </div>
    </header>
  );
}
