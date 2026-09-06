"use client";

import Link from "next/link";
import { BookOpen, Flame, Info, Layers3, Moon, Telescope, Waves } from "lucide-react";
import { useMemo, useState } from "react";
import ObjectDrawer from "@/components/ObjectDrawer";
import SkyCanvas from "@/components/SkyCanvas";
import { formatNumber, shortDate } from "@/lib/format";
import { sourceIdentity } from "@/lib/terminology";
import type {
  DensityTilesFile,
  LightcurveSamplesFile,
  PublicManifest,
  PublicSummary,
  SkyPoint,
  SkyPointsFile,
  TopCandidatesFile
} from "@/types/skypulse";

type Mode = "blend" | "last" | "history";

type Props = {
  manifest: PublicManifest;
  summary: PublicSummary;
  pointsFile: SkyPointsFile;
  densityFile: DensityTilesFile;
  candidatesFile: TopCandidatesFile;
  lightcurveFile: LightcurveSamplesFile;
};

const modeLabels: Record<Mode, string> = {
  blend: "Sky memory",
  last: "Last night",
  history: "History"
};

export default function SkyPulseExperience({
  manifest,
  summary,
  pointsFile,
  densityFile,
  candidatesFile,
  lightcurveFile
}: Props) {
  const [mode, setMode] = useState<Mode>("blend");
  const [heatmap, setHeatmap] = useState(true);
  const [blend, setBlend] = useState(68);
  const [timeline, setTimeline] = useState(100);
  const [selectedId, setSelectedId] = useState<string | null>(
    candidatesFile.candidates[0]?.id ?? pointsFile.points[0]?.id ?? null
  );

  const selectedPoint = useMemo(
    () => pointsFile.points.find((point) => point.id === selectedId) ?? null,
    [pointsFile.points, selectedId]
  );

  const selectedCandidate = useMemo(
    () => candidatesFile.candidates.find((candidate) => candidate.id === selectedId) ?? null,
    [candidatesFile.candidates, selectedId]
  );

  const lightcurve = selectedId ? lightcurveFile.lightcurves[selectedId] ?? [] : [];
  const exportLabel = manifest.export_mode === "rsp_parquet" ? "RSP parquet export" : "Eye-Candy Demo";
  const lightcurveLabel =
    lightcurveFile.sample_source === "alerts_parquet"
      ? "brightness history from ANTARES alert records"
      : lightcurveFile.public_label ?? "Brightness over time";

  return (
    <main className="experience-shell">
      <nav className="site-nav floating-nav">
        <Link href="/" className="brand-link" aria-label="SkyPulse home">
          <Telescope aria-hidden="true" />
          SkyPulse
        </Link>
        <div className="nav-actions">
          <Link href="/methodology">
            <BookOpen aria-hidden="true" />
            Methodology
          </Link>
          <Link href="/glossary">
            <Info aria-hidden="true" />
            Glossary
          </Link>
        </div>
      </nav>

      <section className="observatory" aria-label="SkyPulse living sky explorer">
        <SkyCanvas
          points={pointsFile.points}
          tiles={densityFile.tiles}
          mode={mode}
          heatmap={heatmap}
          blend={blend}
          timeline={timeline}
          selectedId={selectedId}
          onSelect={setSelectedId}
        />

        <section className="hero-panel glass-panel">
          <p className="eyebrow">A living map of the changing night sky</p>
          <h1>SkyPulse</h1>
          <p className="hero-copy">
            Every night, the sky changes. Explore LSST-associated ANTARES loci
            processed on RSP as a cinematic sky memory, then open the science
            layer when a point catches your eye.
          </p>
          <div className="metric-row" aria-label="Dataset summary">
            {summary.metrics.map((metric) => (
              <div className="metric" key={metric.label}>
                <strong>{metric.value}</strong>
                <span>{metric.label}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="control-panel glass-panel" aria-label="Sky controls">
          <div className="segmented" role="group" aria-label="Sky layer">
            {(Object.keys(modeLabels) as Mode[]).map((value) => (
              <button
                className={mode === value ? "active" : ""}
                key={value}
                type="button"
                onClick={() => setMode(value)}
              >
                {value === "blend" && <Layers3 aria-hidden="true" />}
                {value === "last" && <Flame aria-hidden="true" />}
                {value === "history" && <Moon aria-hidden="true" />}
                {modeLabels[value]}
              </button>
            ))}
          </div>

          <label className="range-control">
            <span>History opacity</span>
            <input
              aria-label="History opacity"
              type="range"
              min="0"
              max="100"
              value={blend}
              onChange={(event) => setBlend(Number(event.target.value))}
            />
          </label>

          <label className="range-control">
            <span>Processed timeline</span>
            <input
              aria-label="Processed timeline"
              type="range"
              min="1"
              max="100"
              value={timeline}
              onChange={(event) => setTimeline(Number(event.target.value))}
            />
          </label>

          <button
            className={heatmap ? "icon-toggle active" : "icon-toggle"}
            type="button"
            onClick={() => setHeatmap((value) => !value)}
            aria-pressed={heatmap}
          >
            <Waves aria-hidden="true" />
            Glow density
          </button>
        </section>

        <aside className="candidate-rail glass-panel" aria-label="Top interesting objects">
          <p className="panel-title">Objects worth opening</p>
          {candidatesFile.candidates.slice(0, 5).map((candidate) => (
            <button
              className={selectedId === candidate.id ? "candidate active" : "candidate"}
              key={candidate.id}
              type="button"
              onClick={() => setSelectedId(candidate.id)}
            >
              <span>{candidate.id}</span>
              <small>{candidate.reason}</small>
            </button>
          ))}
        </aside>

        <aside className="explain-panel glass-panel">
          <p className="panel-title">What am I looking at?</p>
          <p>
            Cyan and amber points are the latest processed night from{" "}
            {shortDate(manifest.source_data_range.latest_night_utc)}. Violet
            is the saved sky memory from earlier processed nights.
          </p>
          <p>{exportLabel}: {summary.public_data_note ?? "Static public JSON export."}</p>
          <p className="source-note">{sourceIdentity}.</p>
        </aside>

        <ObjectDrawer
          point={selectedPoint}
          candidate={selectedCandidate}
          lightcurve={lightcurve}
          lightcurveLabel={lightcurveLabel}
          caveats={manifest.scientific_caveats}
          onClose={() => setSelectedId(null)}
        />
      </section>

      <section className="below-fold">
        <div className="method-strip">
          <div>
            <p className="eyebrow">{exportLabel}</p>
            <h2>Static data first. Scientific caveats always visible.</h2>
          </div>
          <p>
            This build uses {formatNumber(manifest.counts.sky_points)} public
            sky points, {formatNumber(manifest.counts.density_tiles)} density
            cells, and {formatNumber(manifest.counts.lightcurve_objects)} sample
            brightness stories. The frontend never queries ANTARES, Rubin
            Butler, TAP, or RSP.
          </p>
        </div>
      </section>
    </main>
  );
}
