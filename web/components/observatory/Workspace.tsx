"use client";

import { useEffect, useRef } from "react";
import BasisBar from "./BasisBar";
import Inspector from "./Inspector";
import PopulationLab from "./PopulationLab";
import ProvenanceDrawer from "./ProvenanceDrawer";
import SelectionBar from "./SelectionBar";
import SkyLens from "./SkyLens";
import TimeRibbon from "./TimeRibbon";
import { useObservatory } from "./ObservatoryContext";

export default function Workspace() {
  const { state, dispatch, provenanceOpen, setProvenanceOpen } = useObservatory();
  const primary = state.lens.primary;
  const shellRef = useRef<HTMLDivElement | null>(null);
  // While the provenance dialog is open the workspace behind it is inert.
  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;
    if (provenanceOpen) shell.setAttribute("inert", "");
    else shell.removeAttribute("inert");
  }, [provenanceOpen]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (provenanceOpen) setProvenanceOpen(false);
      else if (state.focus) dispatch({ type: "focus", focus: null });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dispatch, provenanceOpen, setProvenanceOpen, state.focus]);

  return (
    <main className="uso" data-mode={state.mode}>
      <div className="uso-shell" ref={shellRef}>
        <BasisBar />
        <SelectionBar />
        <TimeRibbon />
        {state.mode === "compare" ? (
          // Compare: both lenses span the main column with domains side by side;
          // the Inspector takes the full side column.
          <div className={`uso-stage is-compare is-${primary}-primary`}>
            <div className="uso-main">
              {primary === "sky" ? (
                <>
                  <SkyLens placement="primary" />
                  <PopulationLab placement="secondary" />
                </>
              ) : (
                <>
                  <PopulationLab placement="primary" />
                  <SkyLens placement="secondary" />
                </>
              )}
            </div>
            <div className="uso-side is-single">
              <Inspector />
            </div>
          </div>
        ) : (
          <div className={`uso-stage is-${primary}-primary`}>
            <div className="uso-primary">{primary === "sky" ? <SkyLens placement="primary" /> : <PopulationLab placement="primary" />}</div>
            <div className={`uso-side${state.focus ? " has-focus" : ""}`}>
              <Inspector />
              {primary === "sky" ? <PopulationLab placement="dock" /> : <SkyLens placement="dock" />}
            </div>
          </div>
        )}
      </div>
      <ProvenanceDrawer />
    </main>
  );
}
