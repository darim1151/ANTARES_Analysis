"use client";

import type { ReactNode } from "react";
import type { Capability, CapabilityState, DomainId, EvidenceClass, NightState } from "@/types/observatory";
import type { TimeAdmission } from "@/lib/observatory/model";
import { DOMAIN_LABEL, EVIDENCE_LABEL, EVIDENCE_TONE, NIGHT_STATE_LABEL } from "@/lib/observatory/theme";

export function DomainBadge({ domain, quiet = false }: { domain: DomainId; quiet?: boolean }) {
  return (
    <span className={`uso-domain uso-domain-${domain}${quiet ? " is-quiet" : ""}`}>
      <i aria-hidden="true" />
      {DOMAIN_LABEL[domain]}
    </span>
  );
}

export function EvidenceChip({ evidence, compact = false }: { evidence: EvidenceClass; compact?: boolean }) {
  const tone = EVIDENCE_TONE[evidence];
  return (
    <span className={`uso-ev uso-ev-${tone}`} title={EVIDENCE_LABEL[evidence]}>
      <i aria-hidden="true" />
      {compact ? EVIDENCE_LABEL[evidence].split(" ")[0] : EVIDENCE_LABEL[evidence]}
    </span>
  );
}

export function EvidenceChips({ list, compact = false }: { list: EvidenceClass[]; compact?: boolean }) {
  return (
    <span className="uso-evlist">
      {list.map((e) => (
        <EvidenceChip key={e} evidence={e} compact={compact} />
      ))}
    </span>
  );
}

const STATE_GLYPH: Record<CapabilityState, string> = { AVAILABLE: "●", PARTIALLY_QUALIFIED: "◐", UNAVAILABLE: "○" };
const STATE_TEXT: Record<CapabilityState, string> = {
  AVAILABLE: "Available",
  PARTIALLY_QUALIFIED: "Partially qualified",
  UNAVAILABLE: "Unavailable"
};

export function CapabilityMark({ state, label = true }: { state: CapabilityState; label?: boolean }) {
  return (
    <span className={`uso-cap uso-cap-${state.toLowerCase()}`}>
      <b aria-hidden="true">{STATE_GLYPH[state]}</b>
      {label ? STATE_TEXT[state] : <span className="uso-sr">{STATE_TEXT[state]}</span>}
    </span>
  );
}

/** Inline statement of a capability that is not fully available, with its reason. */
export function CapabilityNote({ capability }: { capability: Capability }) {
  if (capability.state === "AVAILABLE") return null;
  return (
    <p className={`uso-capnote uso-capnote-${capability.state.toLowerCase()}`}>
      <CapabilityMark state={capability.state} />
      <span>
        <strong>{capability.summary}.</strong> {capability.reason}
      </span>
    </p>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label
}: {
  value: T;
  options: Array<{ value: T; label: ReactNode; title?: string }>;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="uso-seg" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          className={value === o.value ? "is-on" : ""}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export type SelectOption = { value: string; label: string; disabled?: boolean; group?: string };

export function SelectField({
  label,
  value,
  options,
  onChange,
  wide = false
}: {
  label: string;
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  wide?: boolean;
}) {
  const groups = [...new Set(options.map((o) => o.group ?? ""))];
  return (
    <label className={`uso-field${wide ? " is-wide" : ""}`}>
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {groups.map((g) =>
          g ? (
            <optgroup key={g} label={g}>
              {options
                .filter((o) => o.group === g)
                .map((o) => (
                  <option key={o.value} value={o.value} disabled={o.disabled}>
                    {o.label}
                  </option>
                ))}
            </optgroup>
          ) : (
            options
              .filter((o) => !o.group)
              .map((o) => (
                <option key={o.value} value={o.value} disabled={o.disabled}>
                  {o.label}
                </option>
              ))
          )
        )}
      </select>
    </label>
  );
}

export function Toggle({ on, onChange, children, title }: { on: boolean; onChange: (on: boolean) => void; children: ReactNode; title?: string }) {
  return (
    <button type="button" className={`uso-toggle${on ? " is-on" : ""}`} aria-pressed={on} title={title} onClick={() => onChange(!on)}>
      {children}
    </button>
  );
}

export function KeyValue({ k, v, mono = false, note }: { k: ReactNode; v: ReactNode; mono?: boolean; note?: ReactNode }) {
  return (
    <div className="uso-kv">
      <dt>{k}</dt>
      <dd className={mono ? "is-mono" : undefined}>
        {v}
        {note && <small>{note}</small>}
      </dd>
    </div>
  );
}

/** "30 Unavailable · not admitted, 2 Outside coverage" for the non-admitted part of a time selection. */
export function admissionSummary(admission: TimeAdmission): string {
  return (Object.entries(admission.byState) as Array<[NightState, number]>)
    .filter(([state]) => state !== "AVAILABLE" && state !== "ZERO")
    .map(([state, n]) => `${n} ${NIGHT_STATE_LABEL[state].toLowerCase()}`)
    .join(", ");
}

/**
 * Statement shown wherever a count would otherwise appear for dates that are
 * not admitted: those dates have no counts, which is different from zero.
 */
export function AdmissionNote({ domain, admission, block = false }: { domain: DomainId; admission: TimeAdmission | null; block?: boolean }) {
  if (!admission || admission.status === "FULL") return null;
  const text =
    admission.status === "NONE"
      ? `${DOMAIN_LABEL[domain]}: ${admissionSummary(admission)} in the selection. No count exists for these dates, which is not zero.`
      : `${admission.admitted} of ${admission.total} selected dates admitted for ${DOMAIN_LABEL[domain]}; ${admissionSummary(admission)} carry no counts.`;
  return <p className={`uso-admission${block ? " is-block" : ""}${admission.status === "NONE" ? " is-none" : ""}`}>{text}</p>;
}

/** One phrase for why a time selection has no counts in a domain. */
export function admissionLabel(admission: TimeAdmission): string {
  const states = (Object.keys(admission.byState) as NightState[]).filter((st) => st !== "AVAILABLE" && st !== "ZERO");
  if (states.length === 1) {
    if (states[0] === "OUTSIDE_COVERAGE") return "outside coverage on these dates";
    if (states[0] === "UNAVAILABLE") return "unavailable · not admitted";
    if (states[0] === "UNQUALIFIED") return "unqualified · not admitted";
    if (states[0] === "MISSING") return "missing on these dates";
  }
  return "not admitted on these dates";
}
