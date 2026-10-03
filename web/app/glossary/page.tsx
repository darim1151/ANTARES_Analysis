import Link from "next/link";
import { glossary } from "@/lib/terminology";

export default function GlossaryPage() {
  return (
    <main className="min-h-screen bg-void text-star">
      <nav className="site-nav">
        <Link href="/" className="brand-link">SkyPulse</Link>
        <div className="nav-actions">
          <Link href="/methodology">Methodology</Link>
          <Link href="/">Explorer</Link>
        </div>
      </nav>

      <section className="page-shell">
        <p className="eyebrow">Glossary</p>
        <h1>Plain language for an alert-analysis sky.</h1>
        <p className="lead">
          The public layer names what people can understand first, then lets
          scientific details expand on demand.
        </p>
        <div className="glossary-list">
          {glossary.map((item) => (
            <article className="glossary-row" key={item.term}>
              <div>
                <p className="term-public">{item.publicTerm}</p>
                <h2>{item.term}</h2>
              </div>
              <p>{item.expanded}</p>
              <p className="avoid">{item.avoid}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
