"""Read-only extract of the accepted Fink five-window analytical catalog.

Runs ON THE DATA HOST, streamed over SSH stdin; it writes nothing there and
prints one JSON document to stdout:

    ssh arnor 'cd /tmp && PYTHONPATH=<fink release>/src \
        <fink analytics python> -I -B - <catalog run id>' \
        < web/scripts/observatory/extract-fink-catalog.py > extract.json

(`-I` ignores PYTHONPATH, so the release `src` is added explicitly below.)
The local wrapper `fetch-fink-catalog.mjs` runs exactly this and records the
script digest; see its header.

Read-only guarantees:
  - the catalog is opened with Fink's own `open_catalog`, which opens DuckDB
    read_only=True and re-verifies the contract, qualification status and every
    raw input's stat fingerprint;
  - the catalog file's SHA256 is recomputed and must equal its build manifest;
  - DuckDB spilling is disabled (empty temp_directory), so no temp files are
    written next to the catalog;
  - only SELECT statements are executed; no Fink acquisition state is touched.

What is extracted (all Fink-native; no cross-broker matching):
  - catalog/cohort/acquisition identities and per-acquisition QC (complete);
  - per-UTC-date delivered alert rows by population (complete);
  - DiaObject density on HEALPix NESTED order 6 (complete, every derived DIA
    group; position = unit-vector mean of its delivered DiaSources);
  - a simple random sample of DiaObjects (lowest md5 of the decimal id) with
    every delivered DiaSource and first/last-two source-time broker snapshots.
"""

import hashlib
import json
import math
import os
import sys
from datetime import date, timedelta

DATA_ROOT = "/astro/store/shire/FINK"
RELEASES = os.path.expanduser("~/opt/fink-lsst-analysis/releases")
SAMPLE_SIZE = 5000
DENSITY_ORDER = 6
POSITION_DECIMALS = 5
# Curated lc_features keys kept in sampled snapshots (all keys are in the
# pinned Light Static schema); the full map is too large for a static bundle.
LC_KEYS = [
    "mean", "standard_deviation", "amplitude", "chi2", "skew", "kurtosis",
    "stetson_K", "linear_fit_slope", "linear_fit_reduced_chi2", "median_absolute_deviation",
]
CLF_KEYS = ["snnSnVsOthers_score", "cats_class", "cats_score", "earlySNIa_score",
            "elephant_kstest_science", "elephant_kstest_template"]


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


run_id = sys.argv[1] if len(sys.argv) > 1 else "final-five-window-analytics-20261006-v1"
manifest_path = f"{DATA_ROOT}/manifests/{run_id}/manifest.json"
catalog_path = f"{DATA_ROOT}/data/processed/{run_id}/analytics.duckdb"
with open(manifest_path, "rb") as fh:
    manifest_bytes = fh.read()
manifest = json.loads(manifest_bytes)
code_sha = manifest["code_commit_sha"]
release = f"{RELEASES}/{code_sha}"
sys.path.insert(0, f"{release}/src")

import duckdb  # noqa: E402
import erfa  # noqa: E402
import numpy as np  # noqa: E402
from fink_lsst.analytics.catalog import open_catalog  # noqa: E402

catalog_sha = sha256_file(catalog_path)
if manifest["artifact_sha256"].get(catalog_path) != catalog_sha:
    raise SystemExit("catalog SHA256 differs from its build manifest")
log("catalog sha256 verified")

con = open_catalog(catalog_path, DATA_ROOT)  # read_only=True + fingerprint re-verification
log("native open_catalog verified raw input fingerprints")
con.execute("SET threads=16")
con.execute("SET memory_limit='32GB'")
con.execute("SET temp_directory=''")
con.execute("SET enable_progress_bar=false")


def rows(sql, *params):
    cur = con.execute(sql, list(params))
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


metadata = {r["key"]: json.loads(r["value"]) for r in rows("SELECT key, value FROM catalog_metadata") if r["key"] != "inputs"}

# --- time scale: TAI - UTC over the requested window from the ERFA table ---
start = date.fromisoformat(manifest["cohort"]["requested_start"])
stop = date.fromisoformat(manifest["cohort"]["requested_stop"])
dates = [start + timedelta(days=i) for i in range((stop - start).days)]
tai_minus_utc = sorted({float(erfa.dat(d.year, d.month, d.day, 0.0)) for d in dates + [stop]})
if tai_minus_utc != [37.0]:
    raise SystemExit(f"TAI-UTC is not constant over the window: {tai_minus_utc}")
TAI_UTC_DAYS = 37.0 / 86400.0

# --- acquisitions and QC (complete) ---
qc = {r["acquisition_id"]: r for r in rows("SELECT * FROM source_qc")}
acquisitions = []
for a in manifest["acquisitions"]:
    q = qc[a["acquisition_id"]]
    acquisitions.append({
        "acquisition_id": a["acquisition_id"],
        "topic": a["topic"],
        "start": a["start"],
        "stop": a["stop"],
        "state": a["state"],
        "profile": a["profile"],
        "packet": a["packet"],
        "expected_rows": a["expected_rows"],
        "validated_rows": a["validated_rows"],
        "parquet_files": sum(g["files"] for g in a["schema_groups"].values()),
        "schema_groups": sorted(a["schema_groups"]),
        "delivery_receipt_canonical_sha256": a["delivery_receipt_canonical_sha256"],
        "raw_stat_fingerprint": a["raw_stat_fingerprint"],
        "raw_total_bytes": next(r["after"]["total_bytes"] for r in manifest["raw_read_only"] if r["acquisition_id"] == a["acquisition_id"]),
        "characterization": {k: a["characterization_linkage"][k] for k in ("run_id", "summary_sha256", "code_sha")},
        "qc": {k: q[k] for k in ("total_rows", "distinct_source_ids", "dia_rows", "sso_rows", "ambiguous_rows", "distinct_dia_objects", "invalid_times")},
    })
cohort_ids = [a["acquisition_id"] for a in manifest["cohort"]["acquisitions"]]

# --- per-UTC-date delivered rows by population (complete) ---
# UTC date bins are the catalog's own requested_date_boundaries (TAI MJD of UTC midnights).
daily = rows(
    """
    WITH b AS (SELECT acquisition_id, utc_date, start_mjd_tai, stop_mjd_tai FROM requested_date_boundaries),
    s AS (
      SELECT b.utc_date, s.population, count(*) AS n
      FROM sources s JOIN b ON s.observation_mjd_tai >= b.start_mjd_tai AND s.observation_mjd_tai < b.stop_mjd_tai
      GROUP BY 1, 2
    )
    SELECT strftime(d.utc_date, '%Y-%m-%d') AS date, d.acquisition_id, d.delivered_rows, d.status,
           coalesce(sum(n) FILTER (WHERE population = 'DIA'), 0) AS dia_rows,
           coalesce(sum(n) FILTER (WHERE population = 'SSO'), 0) AS sso_rows,
           coalesce(sum(n) FILTER (WHERE population = 'AMBIGUOUS'), 0) AS ambiguous_rows
    FROM daily_coverage d LEFT JOIN s ON s.utc_date = d.utc_date
    GROUP BY ALL ORDER BY 1
    """
)
for d in daily:
    if d["dia_rows"] + d["sso_rows"] + d["ambiguous_rows"] != d["delivered_rows"]:
        raise SystemExit(f"recomputed rows for {d['date']} disagree with daily_coverage")
outside = rows(
    """
    SELECT count(*) AS n FROM sources s
    WHERE NOT EXISTS (SELECT 1 FROM requested_date_boundaries b
                      WHERE s.observation_mjd_tai >= b.start_mjd_tai AND s.observation_mjd_tai < b.stop_mjd_tai)
    """
)[0]["n"]
log("daily coverage recomputed")

# --- DiaObject positions (complete) -> HEALPix NESTED density ---
obj = con.execute(
    """
    SELECT dia_object_id,
           sum(cos(radians(dec_deg)) * cos(radians(ra_deg))) AS x,
           sum(cos(radians(dec_deg)) * sin(radians(ra_deg))) AS y,
           sum(sin(radians(dec_deg))) AS z
    FROM sources WHERE population = 'DIA' GROUP BY 1
    """
).fetchnumpy()
ra = np.mod(np.degrees(np.arctan2(obj["y"], obj["x"])), 360.0)
dec = np.degrees(np.arctan2(obj["z"], np.hypot(obj["x"], obj["y"])))
ra = np.round(ra, POSITION_DECIMALS)
ra[ra >= 360.0] = 0.0
dec = np.round(dec, POSITION_DECIMALS)


def spread(v, order):
    out = np.zeros_like(v)
    for bit in range(order):
        out += ((v >> bit) & 1) << (2 * bit)
    return out


def ang2pix_nest(order, ra_deg, dec_deg):
    """Vectorized port of web/lib/observatory/kernel/healpix.ts ang2pixNest."""
    ns = 1 << order
    theta = (90.0 - dec_deg) * math.pi / 180.0
    phi = ra_deg * math.pi / 180.0
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi / (math.pi / 2), 4.0)
    # equatorial belt
    temp1 = ns * (0.5 + tt)
    temp2 = ns * z * 0.75
    jp = np.floor(temp1 - temp2).astype(np.int64)
    jm = np.floor(temp1 + temp2).astype(np.int64)
    ifp = jp // ns
    ifm = jm // ns
    face_eq = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix_eq = jm % ns
    iy_eq = ns - (jp % ns) - 1
    # polar caps
    ntt = np.minimum(3, np.floor(tt)).astype(np.int64)
    tp = tt - ntt
    tmp = ns * np.sqrt(3 * (1 - za))
    jp2 = np.minimum(ns - 1, np.floor(tp * tmp)).astype(np.int64)
    jm2 = np.minimum(ns - 1, np.floor((1 - tp) * tmp)).astype(np.int64)
    north = z >= 0
    ix_p = np.where(north, ns - jm2 - 1, jp2)
    iy_p = np.where(north, ns - jp2 - 1, jm2)
    face_p = np.where(north, ntt, ntt + 8)
    eq = za <= 2.0 / 3.0
    ix = np.where(eq, ix_eq, ix_p)
    iy = np.where(eq, iy_eq, iy_p)
    face = np.where(eq, face_eq, face_p)
    return face * (4 ** order) + spread(ix, order) + 2 * spread(iy, order)


pix = ang2pix_nest(DENSITY_ORDER, ra, dec)
counts = np.bincount(pix, minlength=12 * 4 ** DENSITY_ORDER)
occupied = np.nonzero(counts)[0]
density = {"order": DENSITY_ORDER, "pixels": occupied.tolist(), "values": counts[occupied].tolist(), "objects": int(len(pix))}
log(f"density over {len(pix)} DiaObjects in {len(occupied)} cells")

# --- multiplicity (complete) ---
multiplicity = rows("SELECT alerts_per_object AS n_sources, objects FROM dia_object_multiplicity ORDER BY 1")

# --- population-level clf sentinel census (complete; DIA rows) ---
clf_census = {}
for key in ["snnSnVsOthers_score", "cats_score", "earlySNIa_score"]:
    clf_census[key] = rows(
        f"""
        WITH v AS (SELECT TRY_CAST(json_extract(clf_json, '$.{key}') AS DOUBLE) AS v FROM dia_source_snapshots)
        SELECT count(*) AS rows, count(*) FILTER (WHERE v IS NULL) AS null_rows,
               count(*) FILTER (WHERE v = -1) AS minus_one_rows,
               count(*) FILTER (WHERE v >= 0 AND v <= 1) AS unit_interval_rows,
               count(*) FILTER (WHERE v IS NOT NULL AND v <> -1 AND (v < 0 OR v > 1 OR isnan(v))) AS other_rows
        FROM v
        """
    )[0]
log("clf census done")

# --- simple random sample of DiaObjects ---
index = {int(i): k for k, i in enumerate(obj["dia_object_id"])}
sample_ids = [r[0] for r in con.execute(
    f"""
    SELECT dia_object_id FROM (SELECT DISTINCT dia_object_id FROM sources WHERE population = 'DIA')
    ORDER BY md5(CAST(dia_object_id AS VARCHAR)), dia_object_id LIMIT {SAMPLE_SIZE}
    """
).fetchall()]
src = rows(
    """
    SELECT dia_object_id, source_id, observation_mjd_tai, band, psf_flux, psf_flux_err, snr,
           reliability, acquisition_id, raw_file, pred_json, clf_json, lc_features_json, xm_json, misc_json
    FROM dia_source_snapshots
    WHERE dia_object_id IN (SELECT unnest(?::BIGINT[]))
    ORDER BY dia_object_id, observation_mjd_tai, source_id
    """,
    sample_ids,
)
log(f"sample rows: {len(src)}")

by_obj = {}
for r in src:
    by_obj.setdefault(r["dia_object_id"], []).append(r)
if sorted(by_obj) != sorted(sample_ids):
    raise SystemExit("sample sources do not cover every sampled DiaObject")

# Snapshot policy: first and last two delivered source-time snapshots (as G4A).
snap_rows = []
for oid, srcs in by_obj.items():
    for k in sorted({0, len(srcs) - 2, len(srcs) - 1} - {-1}):
        snap_rows.append(srcs[k])
# fink_science_version is not projected by the catalog views: read it from the
# exact raw files those snapshot rows came from.
files = sorted({r["raw_file"] for r in snap_rows})
versions = {}
for i in range(0, len(files), 2000):
    chunk = files[i:i + 2000]
    for sid, ver, bver in con.execute(
        "SELECT CAST(diaSourceId AS BIGINT), fink_science_version, fink_broker_version FROM read_parquet(?)",
        [chunk],
    ).fetchall():
        versions[sid] = (ver, bver)
log(f"read versions from {len(files)} raw files")


def finite(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def sig(v, n=6):
    v = finite(v)
    return None if v is None else float(f"{v:.{n}g}")


objects = []
for oid in sample_ids:
    srcs = by_obj[oid]
    k = index[oid]
    sources = []
    for r in srcs:
        sources.append({
            "diaSourceId": str(r["source_id"]),
            "midpointMjdTai": r["observation_mjd_tai"],
            "band": r["band"],
            "psfFlux": sig(r["psf_flux"]),
            "psfFluxErr": sig(r["psf_flux_err"]),
            "snr": sig(r["snr"], 5),
            "reliability": sig(r["reliability"], 4),
            "acquisition_id": r["acquisition_id"],
        })
    snaps = []
    for idx in sorted({0, len(srcs) - 2, len(srcs) - 1} - {-1}):
        r = srcs[idx]
        pred = json.loads(r["pred_json"])
        clf = json.loads(r["clf_json"]) or {}
        lc = json.loads(r["lc_features_json"]) or {}
        xm = json.loads(r["xm_json"]) or {}
        misc = json.loads(r["misc_json"]) or {}
        ver = versions.get(r["source_id"])
        if ver is None:
            raise SystemExit(f"no raw row for snapshot source {r['source_id']}")
        snaps.append({
            "diaSourceId": str(r["source_id"]),
            "midpointMjdTai": r["observation_mjd_tai"],
            "fink_science_version": ver[0],
            "fink_broker_version": ver[1],
            "pred": pred,
            "clf": {key: (sig(clf.get(key)) if key != "cats_class" else clf.get(key)) for key in CLF_KEYS},
            "xm": {key: (sig(v) if isinstance(v, float) else v) for key, v in xm.items() if v is not None},
            "lc_features": {band: {key: sig(feats.get(key), 4) for key in LC_KEYS} for band, feats in sorted(lc.items()) if feats},
            "misc": {key: finite(v) if isinstance(v, float) else v for key, v in misc.items()},
        })
    objects.append({
        "id": str(oid),
        "ra": float(ra[k]),
        "dec": float(dec[k]),
        "pix": int(pix[k]),
        "sources": sources,
        "snapshots": snaps,
    })

extract = {
    "kind": "uso.fink-catalog-extract",
    "version": 1,
    "statement": "Read-only extract of the accepted Fink analytical catalog. Fink-native DiaSource/DiaObject semantics; no cross-broker matching.",
    "catalog": {
        "run_id": manifest["run_id"],
        "contract_id": manifest["contract_id"],
        "completion_status": manifest["completion_status"],
        "qualification_status": metadata.get("qualification_status"),
        "finished_utc": manifest["finished_utc"],
        "code_commit_sha": code_sha,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "catalog_sha256": catalog_sha,
        "catalog_relative_path": f"data/processed/{run_id}/analytics.duckdb",
        "manifest_relative_path": f"manifests/{run_id}/manifest.json",
        "cohort_name": manifest["cohort"]["cohort_name"],
        "cohort_sha256": manifest["cohort_sha256"],
        "cohort_config": "configs/analysis_cohorts/final_20260225_20260714.json",
        "cohort_acquisitions": cohort_ids,
        "requested_start": manifest["cohort"]["requested_start"],
        "requested_stop": manifest["cohort"]["requested_stop"],
        "package_versions": manifest["package_versions"],
    },
    "reader": {
        "native_open": "fink_lsst.analytics.catalog.open_catalog",
        "release": code_sha,
        "read_only": True,
        "raw_fingerprints_reverified": True,
        "duckdb": duckdb.__version__,
        "numpy": np.__version__,
        "erfa": erfa.__version__,
        "python": sys.version.split()[0],
    },
    "time_scale": {"tai_minus_utc_s": 37.0, "source": "erfa.dat over every UTC date of the requested window"},
    "acquisitions": acquisitions,
    "daily": daily,
    "rows_outside_requested_dates": outside,
    "multiplicity": multiplicity,
    "clf_census": clf_census,
    "density": density,
    "sample": {
        "method": "Simple random sample without replacement: the DiaObjects with the lowest md5(decimal diaObjectId).",
        "size": len(objects),
        "population": int(len(pix)),
        "lc_feature_keys": LC_KEYS,
        "objects": objects,
    },
}
con.close()
json.dump(extract, sys.stdout, separators=(",", ":"), sort_keys=True, allow_nan=False)
sys.stdout.write("\n")
log("done")
