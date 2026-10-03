"use client";

import { ChevronDown, Telescope, X } from "lucide-react";
import LightcurveChart from "@/components/LightcurveChart";
import { formatMagnitude } from "@/lib/format";
import type { LightcurveSample, SkyPoint, TopCandidate } from "@/types/skypulse";

type Props = {
  point: SkyPoint | null;
  candidate: TopCandidate | null;
  lightcurve: LightcurveSample[];
  lightcurveLabel: string;
  caveats: string[];
  onClose: () => void;
};

export default function ObjectDrawer({ point, candidate, lightcurve, lightcurveLabel, caveats, onClose }: Props) {
  if (!point) return null;

  return (
    <aside className="object-drawer" aria-label={`Details for ${point.id}`}>
      <div className="drawer-head">
        <div>
          <p className="eyebrow">Telescope panel</p>
          <h2>{point.id}</h2>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close object panel">
          <X aria-hidden="true" />
        </button>
      </div>

      <p className="object-story">
        {candidate?.public_summary ??
          "This sky object is included because its latest processed alert-analysis record stands out in the public sample."}
      </p>

      <div className="object-stats">
        <div>
          <span>Brightness</span>
          <strong>{formatMagnitude(point.brightness_mag)}</strong>
        </div>
        <div>
          <span>Observations</span>
          <strong>{point.obs_count}</strong>
        </div>
        <div>
          <span>Sky memory</span>
          <strong>{point.seen_before ? "Seen before" : "New here"}</strong>
        </div>
      </div>

      <LightcurveChart samples={lightcurve} label={lightcurveLabel} />

      <div className="tag-list" aria-label="ANTARES tags">
        {point.tags.slice(0, 4).map((tag) => (
          <span key={tag}>{tag.replaceAll("_", " ")}</span>
        ))}
      </div>

      <details className="science-details">
        <summary>
          Scientific details
          <ChevronDown aria-hidden="true" />
        </summary>
        <dl>
          <div>
            <dt>Right ascension</dt>
            <dd>{point.ra.toFixed(5)} deg</dd>
          </div>
          <div>
            <dt>Declination</dt>
            <dd>{point.dec.toFixed(5)} deg</dd>
          </div>
          <div>
            <dt>Astronomer&apos;s time stamp</dt>
            <dd>MJD {point.mjd.toFixed(5)}</dd>
          </div>
          <div>
            <dt>Reason label</dt>
            <dd>{point.reason}</dd>
          </div>
        </dl>
      </details>

      <div className="drawer-caveat">
        <Telescope aria-hidden="true" />
        <p>{caveats[0]}</p>
      </div>
    </aside>
  );
}
