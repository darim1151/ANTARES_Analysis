# SkyPulse Public Contract

SkyPulse is a static public interpretation layer over ANTARES alert-analysis
data processed on Rubin Science Platform storage. It must not claim to query
Rubin Butler, TAP, official Rubin production systems, or live ANTARES streams.

## Export Files

- `public_manifest.json`: dataset date range, counts, validation flags, and
  scientific caveats. Includes `export_mode` (`demo` or `rsp_parquet`),
  selected night, source files, validation summary, alerts availability, and
  record counts.
- `public_summary.json`: headline metrics and Last Night vs prior-history
  comparison counts.
- `sky_points.json`: sampled sky objects with `id`, `ra`, `dec`, `date_utc`,
  `mjd`, `brightness_mag`, `obs_count`, `tags`, `is_last_night`,
  `seen_before`, `interest_score`, and `reason`. RSP exports also include
  `locus_id`, `label`, `group`, `newest_alert_observation_time`,
  `brightest_alert_magnitude`, `num_mag_values`, `has_lightcurve`,
  `is_highlighted`, and `public_description`.
- `density_tiles.json`: RA/Dec bin counts for the glow-density layer.
- `top_candidates.json`: transparent ranking labels for objects worth opening.
- `lightcurve_samples.json`: public brightness-over-time samples keyed by
  object id. In RSP mode these are real alert-record samples when compatible
  alert columns are present; unavailable samples are reported instead of
  synthesized.

Every file includes `schema_version`, `generated_at_utc`, `source_data_range`,
and `scientific_caveats`.

## Guardrails

- Say “LSST-associated ANTARES loci” or “Rubin/LSST-associated objects indexed
  through ANTARES.”
- Say “ANTARES alert-analysis data processed on RSP.”
- Do not say “Rubin live feed,” “official Rubin result,” “direct Rubin query,”
  “real-time LSST stream,” or “classified transient.”
- Historical comparison must use data strictly before the selected night.
- The public app consumes static files only.

## CLI Usage

Demo/local fallback:

```bash
python scripts/export_skypulse_public_data.py --demo
```

RSP-backed latest usable night:

```bash
python scripts/export_skypulse_public_data.py \
  --data-root /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data \
  --out web/public/data \
  --latest
```

RSP-backed specific UTC night:

```bash
python scripts/export_skypulse_public_data.py \
  --data-root /home/ivezic/AntaresAlerts/ANTARES_Analysis_Data \
  --out web/public/data \
  --date 2026-05-30
```

If `--data-root` is supplied without `--latest` or `--date`, the exporter uses
latest-night selection. If no data root is supplied, it stays in demo mode so
the frontend remains usable outside RSP.

## Required RSP Inputs

The exporter expects the existing research layout:

```text
data/lsst_only/nightly/YYYY/MM/DD/loci.parquet
data/lsst_only/nightly/YYYY/MM/DD/alerts.parquet
data/lsst_only/nightly/YYYY/MM/DD/manifest.json
data/lsst_only/cumulative/loci_index.parquet
data/lsst_only/cumulative/nightly_summary.parquet
```

Required for RSP export: selected nightly `manifest.json`, selected nightly
`loci.parquet`, and cumulative `loci_index.parquet`. The selected
`alerts.parquet` and cumulative `nightly_summary.parquet` are read when
available; missing alert rows produce warnings and unavailable lightcurve
entries rather than synthetic samples.

## RSP Selection Rules

- `--latest` discovers nightly manifests under
  `data/lsst_only/nightly/YYYY/MM/DD/`.
- It prefers nights with manifest status `complete` or `under_target`.
- `saturated_unresolved` can be exported only when no preferred latest night is
  available, or when explicitly selected with `--date`; the public manifest
  discloses the warning.
- Historical comparison uses cumulative rows strictly before the selected UTC
  night and selected MJD boundary.

## Column Assumptions

Locus exports prefer these columns when present: `locus_id`, `ra`, `dec`,
`tags`, `newest_alert_observation_time`, `brightest_alert_magnitude`,
`num_mag_values`, `night_date_utc`, `night_mjd_min`, and `night_mjd_max`.

Alert-record lightcurves require `locus_id`, one magnitude column such as
`ztf_magpsf`, and one time column such as `ant_mjd`, `mjd`, `obs_mjd`,
`ztf_mjd`, or `alert_mjd`. Filter labels use `ztf_fid` when available. The
public label is “brightness history from ANTARES alert records,” never “LSST
lightcurve” or “official Rubin lightcurve.”
