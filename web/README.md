# SkyPulse web

SkyPulse is a static-facing Next.js application. It reads a complete public-data
bundle from `public/data`; it must never query ANTARES, Rubin services, or private
analysis storage at runtime.

## Reproducible local check

Use exact Node.js 22.22.0 and the pnpm version declared in `package.json`:

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm run validate:data:test
pnpm run validate:data
pnpm run typecheck
pnpm run lint
pnpm run build
```

The repository includes a sanitized, explicitly labelled demo bundle so a clean
checkout can be checked and built without credentials or access to scientific
infrastructure. Generated directories such as `.next`, `out`, and `node_modules`
are local artifacts and must not be committed.

## Public-data gate

`pnpm run validate:data` validates the six required JSON files as one coherent
bundle. It checks their schema/mode/timestamps, declared counts, object references,
coordinate bounds, validation flags, and absence of private filesystem paths or
secret-bearing fields.

The bundled demo is valid for development and CI. A real release pipeline must
instead run:

```bash
pnpm run validate:data:production
```

That stricter command rejects `export_mode: "demo"`. Passing either command is a
build-integrity check, not scientific approval of a newly exported dataset.

This is a Phase 1 demo/CI surface, not a public production deployment. The
current Next.js 14 line must be migrated to a supported release and pass a
dedicated UI/accessibility regression phase before public hosting. CI still
builds and starts the current application so its declared `start` command and
all static routes cannot silently regress while that migration remains pending.
