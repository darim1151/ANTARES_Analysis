# ADR-0003: Single ANTARES Shire Project Namespace

Status: accepted as an administrative governance correction. Phase 5 remains
paused pending human review of the corrected namespace.

## Context

The original migration placed the production dataset and its audit evidence at
two ANTARES-specific top-level paths beneath `/astro/store/shire`:

```text
/astro/store/shire/ANTARES_Analysis_Data
/astro/store/shire/ANTARES_Analysis_Data_migration_audits
```

Those names remain valid historical provenance in accepted migration records.
They are not the current runtime contract and must not be rewritten inside
historical manifests or audit outputs.

## Decision

ANTARES owns exactly one top-level Shire project namespace:

```text
/astro/store/shire/ANTARES
```

Its governed children are:

```text
data/                 authoritative production science
migration_audits/     preserved migration evidence
work/                 non-authoritative operational workspace
work/canary/<RUN_ID>  disposable run-specific qualification material
cache/                reserved rebuildable cache; absent until authorized
```

The Middle Earth profile resolves production data to
`/astro/store/shire/ANTARES/data` and its separately configured future cache to
`/astro/store/shire/ANTARES/cache`, with private storage policy. No permission
broadening, shared-group behavior, cache creation, or science change follows
from this namespace decision.

## Enforcement

`src.cli_profiles.middle_earth_work_path` is a side-effect-free, fail-closed
boundary for proposed operational paths. It accepts only run-specific paths
beneath `/astro/store/shire/ANTARES/work`; top-level Shire siblings such as
`/astro/store/shire/ANTARES_*`, traversal outside `work/`, and the bare work root
are refused. Any future filesystem-writing canary or experiment must use this
validator and then apply its own no-follow, ownership, containment, and
authorization checks before mutation.

Current documentation and tests use the canonical paths. Historical migration
documents keep their original absolute paths so the evidence continues to say
where the migration actually landed at that time.
