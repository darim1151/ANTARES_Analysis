import Link from "next/link";
import { Activity, Database, ShieldCheck, Telescope } from "lucide-react";
import manifest from "../../public/data/public_manifest.json";
import { sourceIdentity } from "@/lib/terminology";
import type { PublicManifest } from "@/types/skypulse";

const typedManifest = manifest as PublicManifest;

const steps = [
  {
    title: "ANTARES source",
    body: "The research workflow queries ANTARES loci with Rubin/LSST survey identifiers. This is broker-indexed alert-analysis data, not a direct Rubin Butler or TAP query.",
    icon: Telescope
  },
  {
    title: "RSP processing",
    body: "Rubin Science Platform is used for compute and persistent storage. The notebooks write nightly parquet partitions and manifests outside Git.",
    icon: Database
  },
  {
    title: "Strict comparison",
    body: "The selected night is compared against cumulative saved history strictly before that night, so the sky does not compare against itself.",
    icon: Activity
  },
  {
    title: "Public export",
    body: "SkyPulse consumes small static JSON files with coordinates, density bins, candidate summaries, and lightcurve samples.",
    icon: ShieldCheck
  }
];

export default function MethodologyPage() {
  return (
    <main className="min-h-screen bg-void text-star">
      <nav className="site-nav">
        <Link href="/" className="brand-link">SkyPulse</Link>
        <div className="nav-actions">
          <Link href="/glossary">Glossary</Link>
          <Link href="/">Explorer</Link>
        </div>
      </nav>

      <section className="page-shell">
        <p className="eyebrow">Methodology</p>
        <h1>What SkyPulse is allowed to say about the sky.</h1>
        <p className="lead">
          SkyPulse is a public interpretation layer over {sourceIdentity}. It
          is designed to be beautiful, but the beauty must never outrun the
          provenance.
        </p>

        <div className="method-grid">
          {steps.map((step) => {
            const Icon = step.icon;
            return (
              <article className="method-panel" key={step.title}>
                <Icon aria-hidden="true" />
                <h2>{step.title}</h2>
                <p>{step.body}</p>
              </article>
            );
          })}
        </div>

        <section className="method-band">
          <h2>Current data contract</h2>
          <p>
            Dataset: {typedManifest.dataset_name}. Generated at{" "}
            {typedManifest.generated_at_utc}. Latest processed night:{" "}
            {typedManifest.source_data_range.latest_night_utc}.
          </p>
          <ul>
            {typedManifest.scientific_caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </section>
      </section>
    </main>
  );
}
