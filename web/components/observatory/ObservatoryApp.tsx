"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertOctagon, RotateCcw } from "lucide-react";
import type { ObservatoryBundle } from "@/types/observatory";
import { buildWorkspaceModel } from "@/lib/observatory/model";
import { BundleError, StaticBundleReader } from "@/lib/observatory/reader";
import { ObservatoryProvider } from "./ObservatoryContext";
import Workspace from "./Workspace";

type LoadState =
  | { status: "loading" }
  | { status: "ready"; bundle: ObservatoryBundle }
  | { status: "error"; message: string; path: string | null };

export default function ObservatoryApp({ base }: { base: string }) {
  const reader = useMemo(() => new StaticBundleReader(base), [base]);
  const [load, setLoad] = useState<LoadState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoad({ status: "loading" });
    reader
      .loadBundle(controller.signal)
      .then((bundle) => setLoad({ status: "ready", bundle }))
      .catch((error: Error) => {
        if (error.name === "AbortError") return;
        setLoad({
          status: "error",
          message: error.message,
          path: error instanceof BundleError ? error.path : null
        });
      });
    return () => controller.abort();
  }, [reader, attempt]);

  const model = useMemo(() => (load.status === "ready" ? buildWorkspaceModel(load.bundle) : null), [load]);

  if (load.status === "error") {
    return (
      <main className="uso uso-fullstate" aria-live="assertive">
        <div className="uso-errorcard">
          <AlertOctagon aria-hidden="true" />
          <p className="uso-eyebrow">Observatory basis could not be opened</p>
          <h1>The workspace refuses to render an unverifiable basis.</h1>
          <p className="uso-errormsg">{load.message}</p>
          {load.path && <p className="uso-errorpath">Payload: {load.path}</p>}
          <button type="button" className="uso-btn" onClick={() => setAttempt((n) => n + 1)}>
            <RotateCcw aria-hidden="true" /> Retry
          </button>
        </div>
      </main>
    );
  }

  if (!model) {
    return (
      <main className="uso uso-loading" aria-busy="true" aria-live="polite">
        <div className="uso-shell">
          <header className="uso-basisbar">
            <div className="uso-wordmark">
              <span>Unified Scientific Observatory</span>
              <small>First Light</small>
            </div>
            <span className="uso-loading-note">Verifying basis payloads (sha256)…</span>
          </header>
          <div className="uso-selectionbar" />
          <section className="uso-panel uso-ribbon uso-skeleton" />
          <div className="uso-stage">
            <section className="uso-panel uso-skeleton" />
            <div className="uso-side">
              <section className="uso-panel uso-skeleton" />
              <section className="uso-panel uso-skeleton" />
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <ObservatoryProvider model={model} reader={reader}>
      <Workspace />
    </ObservatoryProvider>
  );
}
