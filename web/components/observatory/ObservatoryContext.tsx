"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { Dispatch, ReactNode } from "react";
import { DOMAIN_IDS, type DomainId, type ScientificState } from "@/types/observatory";
import { computeMasks, type MaskSet } from "@/lib/observatory/kernel/selection";
import { timeAdmission, type TimeAdmission, type WorkspaceModel } from "@/lib/observatory/model";
import type { BundleReader } from "@/lib/observatory/reader";
import { reduce, stateFromUrl, stateToUrl, type Action } from "@/lib/observatory/state";

type ObservatoryValue = {
  model: WorkspaceModel;
  reader: BundleReader;
  state: ScientificState;
  dispatch: Dispatch<Action>;
  masks: Record<DomainId, MaskSet>;
  /** What the shared time selection means for each domain (null: no time selection). */
  admission: Record<DomainId, TimeAdmission | null>;
  notices: string[];
  dismissNotice: (index: number) => void;
  provenanceOpen: boolean;
  setProvenanceOpen: (open: boolean) => void;
};

/** Ephemeral pointer position shared by synchronized sky maps. */
export type SkyHover = { ra: number; dec: number; source: DomainId } | null;
type HoverValue = { hover: SkyHover; setHover: (hover: SkyHover) => void };

const ObservatoryContext = createContext<ObservatoryValue | null>(null);
const HoverContext = createContext<HoverValue | null>(null);

export function ObservatoryProvider({
  model,
  reader,
  children
}: {
  model: WorkspaceModel;
  reader: BundleReader;
  children: ReactNode;
}) {
  const initial = useMemo(() => stateFromUrl(model, typeof window === "undefined" ? "" : window.location.search), [model]);
  const [state, dispatch] = useReducer(reduce, initial.state);
  const [notices, setNotices] = useState<string[]>(initial.warnings);
  const [provenanceOpen, setProvenanceOpen] = useState(false);
  const [hover, setHover] = useState<SkyHover>(null);
  const urlTimer = useRef<number | null>(null);

  useEffect(() => {
    if (urlTimer.current !== null) window.clearTimeout(urlTimer.current);
    urlTimer.current = window.setTimeout(() => {
      window.history.replaceState(window.history.state, "", stateToUrl(state));
    }, 120);
    return () => {
      if (urlTimer.current !== null) window.clearTimeout(urlTimer.current);
    };
  }, [state]);

  const masks = useMemo(() => {
    const out = {} as Record<DomainId, MaskSet>;
    for (const d of DOMAIN_IDS) out[d] = computeMasks(model.domains[d].cols, state.selection);
    return out;
  }, [model, state.selection]);

  const admission = useMemo(() => {
    const out = {} as Record<DomainId, TimeAdmission | null>;
    for (const d of DOMAIN_IDS) out[d] = timeAdmission(model.domains[d], state.selection.time);
    return out;
  }, [model, state.selection.time]);

  const dismissNotice = useCallback((index: number) => setNotices((list) => list.filter((_, i) => i !== index)), []);

  const value = useMemo(
    () => ({ model, reader, state, dispatch, masks, admission, notices, dismissNotice, provenanceOpen, setProvenanceOpen }),
    [model, reader, state, masks, admission, notices, dismissNotice, provenanceOpen]
  );
  const hoverValue = useMemo(() => ({ hover, setHover }), [hover]);

  return (
    <ObservatoryContext.Provider value={value}>
      <HoverContext.Provider value={hoverValue}>{children}</HoverContext.Provider>
    </ObservatoryContext.Provider>
  );
}

export function useObservatory(): ObservatoryValue {
  const value = useContext(ObservatoryContext);
  if (!value) throw new Error("useObservatory must be used inside ObservatoryProvider");
  return value;
}

export function useSkyHover(): HoverValue {
  const value = useContext(HoverContext);
  if (!value) throw new Error("useSkyHover must be used inside ObservatoryProvider");
  return value;
}
