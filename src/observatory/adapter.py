"""Build bounded native products from saved, locked ANTARES evidence only.

No publisher, live provider, repair, or cache writer is invoked. Status reads
can describe unresolved authority; science reads additionally require the
existing authoritative_read generation guard.
"""

import hashlib
import json
import re
from collections import Counter
from contextlib import ExitStack
from datetime import date, datetime, timedelta
from numbers import Integral, Real
from pathlib import Path

from .features import feature_registry, numeric_value, row_features
from .model import (
    AntaresCapabilityManifest, AntaresLocusRecord, AntaresLocusSnapshot,
    AntaresNightRecord, AntaresObservatoryBasis, AntaresObservatoryProduct,
    AntaresProvenanceRecord, AntaresTemporalProduct, ArtifactIdentity,
    Capability, NativeIdentifier, NightQualification,
)
from .serialization import canonical_bytes
from .spatial import build_sky


class NativeReadRefused(ValueError):
    """Saved evidence cannot safely support the requested native product."""


def _date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise NativeReadRefused("Dates must be canonical YYYY-MM-DD strings")
    return date.fromisoformat(value)


def _dates(start, end):
    first, last = _date(start), _date(end)
    if last < first or (last - first).days > 3660:
        raise NativeReadRefused("Night interval must be ordered and at most 3661 nights")
    return tuple((first + timedelta(days=index)).isoformat()
                 for index in range((last - first).days + 1))


def _code_identity(code_sha, code_state):
    if not isinstance(code_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise NativeReadRefused("A full lowercase code SHA is required")
    if code_state not in {"CLEAN", "DIRTY", "UNVERIFIED"}:
        raise NativeReadRefused("Invalid code state")


def _regular_bytes(path):
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)) or not path.is_file():
        raise NativeReadRefused("UNSAFE_OR_MISSING_INPUT")
    return path.read_bytes()


def _artifact(path, identity):
    payload = _regular_bytes(path)
    return ArtifactIdentity(identity, hashlib.sha256(payload).hexdigest(), len(payload))


def _evidence(identity, payload, refs=(), limitations=()):
    return AntaresProvenanceRecord(identity, tuple(sorted(refs)),
                                   canonical_bytes(payload).decode().strip(), limitations)


def _code_artifacts():
    root = Path(__file__).resolve().parents[2]
    names = (
        "pyproject.toml", "src/authority.py", "src/history.py", "src/query.py", "src/feature_analysis.py",
        "src/lightcurves.py", "src/operations/publication.py",
        "src/operations/science.py", "scripts/export_skypulse_public_data.py",
        "scripts/export_observatory_bundle.py",
        *(f"src/observatory/{name}" for name in
          ("__init__.py", "model.py", "adapter.py", "features.py", "spatial.py", "serialization.py")),
    )
    # Installed wheels need not contain the old script. Never fabricate its hash.
    return tuple(_artifact(root / name, "code:" + name)
                 for name in names if (root / name).is_file())


def identifier_json(value):
    """Lossless string leaves; reject identifiers already rounded in floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [identifier_json(item) for item in value]
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise NativeReadRefused("MALFORMED_IDENTIFIER")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        number, missing = numeric_value(value)
        if missing == "NONFINITE" and str(value).lower() == "nan":
            return None
        if (number is not None and number.is_integer() and abs(number) <= 2**53 - 1
                and value == int(number)):
            return str(int(number))
    raise NativeReadRefused("LOSSY_OR_MALFORMED_IDENTIFIER")


def _locus_id(value):
    result = identifier_json(value)
    if not isinstance(result, str) or not result.strip():
        raise NativeReadRefused("MISSING_OR_CONTAINER_LOCUS_IDENTIFIER")
    return result


def _identifiers(row):
    # Each source representation is retained; aliases are not merged into one ID.
    found = []
    fields = (
        ("lsst.dia_object_id", "dia_object_id"),
        ("lsst.dia_object_id", "lsst_dia_object_id"),
        ("lsst.dia_object_id", "survey.lsst.dia_object_id"),
        ("lsst.ss_object_id", "ss_object_id"),
        ("lsst.ss_object_id", "lsst_ss_object_id"),
        ("lsst.ss_object_id", "survey.lsst.ss_object_id"),
        ("ztf.object_id", "ztf_object_id"),
    )
    def has_content(value):
        if isinstance(value, list):
            return any(has_content(item) for item in value)
        return isinstance(value, str) and bool(value.strip())
    def add(namespace, field, value):
        encoded = identifier_json(value)
        found.append(NativeIdentifier(namespace, field, canonical_bytes(encoded).decode().strip(),
                                      "PRESENT" if has_content(encoded) else "MISSING"))
    for namespace, field in fields:
        if field in row:
            add(namespace, field, row[field])
    survey = row.get("survey")
    if isinstance(survey, dict):
        for broker, field, namespace in (
            ("lsst", "dia_object_id", "lsst.dia_object_id"),
            ("lsst", "ss_object_id", "lsst.ss_object_id"),
            ("ztf", "id", "ztf.object_id"),
        ):
            nested = survey.get(broker)
            if isinstance(nested, dict) and field in nested:
                add(namespace, f"survey.{broker}.{field} (nested)", nested[field])
    return tuple(sorted(found, key=lambda item: (item.namespace, item.source_field)))


def _tags(value):
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return (), "MISSING"
    if isinstance(value, str):
        result = tuple(sorted(set(item.strip() for item in value.split(",") if item.strip())))
        return result, "PRESENT" if result else "KNOWN_EMPTY"
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value):
        result = tuple(sorted(set(value)))
        return result, "PRESENT" if result else "KNOWN_EMPTY"
    if isinstance(value, Real) and str(value).lower() == "nan":
        return (), "MISSING"
    raise NativeReadRefused("MALFORMED_TAGS")


def _snapshots(loci, alerts, night, provenance_ref, lower, upper, upper_bound):
    rows = loci.to_dict(orient="records")
    ids = [_locus_id(row.get("locus_id")) for row in rows]
    if len(set(ids)) != len(ids):
        raise NativeReadRefused("DUPLICATE_LOCUS_IDENTIFIER")
    alert_ids = [_locus_id(value) for value in alerts["locus_id"].tolist()]
    if set(alert_ids) - set(ids):
        raise NativeReadRefused("UNLINKED_ALERT_SOURCE_ROW")
    counts = Counter(alert_ids)
    snapshots = []
    for row, locus_id in zip(rows, ids):
        if "night_date_utc" in row and row["night_date_utc"] != night:
            raise NativeReadRefused("MEMBERSHIP_NIGHT_MISMATCH")
        for field, expected in (("night_mjd_min", lower), ("night_mjd_max", upper)):
            if field in row:
                actual, missing = numeric_value(row[field])
                if missing or actual != expected:
                    raise NativeReadRefused("MEMBERSHIP_BOUND_MISMATCH")
        ra, ra_missing = numeric_value(row.get("ra"), present="ra" in row)
        dec, dec_missing = numeric_value(row.get("dec"), present="dec" in row)
        coordinate_state = "VALID"
        if ra_missing or dec_missing:
            coordinate_state = "INVALID_" + (ra_missing or dec_missing)
        elif not (0 <= ra < 360 and -90 <= dec <= 90):
            coordinate_state = "INVALID_RANGE"
        mjd, time_missing = numeric_value(row.get("newest_alert_observation_time"),
                                          present="newest_alert_observation_time" in row)
        if mjd is not None and (mjd < lower or mjd > upper or (upper_bound == "exclusive" and mjd == upper)):
            raise NativeReadRefused("OBSERVATION_OUTSIDE_QUERY_BOUNDS")
        tags, tag_state = _tags(row.get("tags"))
        snapshots.append((locus_id, AntaresLocusSnapshot(
            night, ra, dec, coordinate_state, _identifiers(row), tags,
            row_features(row, counts[locus_id]), counts[locus_id], mjd, time_missing,
            provenance_ref, tag_state,
        )))
    return tuple(sorted(snapshots, key=lambda pair: pair[0]))


def _qualification(state, availability, content, view, reason, ref, manifest=None, legacy=False):
    qualification = "UNAVAILABLE" if availability != "AVAILABLE" else (
        "PARTIALLY_QUALIFIED" if legacy else "QUALIFIED_SAVED_CONTENT"
    )
    return NightQualification(state, availability, content, view, qualification,
                               (manifest or {}).get("status"), reason, ref)


def _utc_timestamp(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise NativeReadRefused("INVALID_RECORDED_UTC_TIMESTAMP")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise NativeReadRefused("INVALID_RECORDED_UTC_TIMESTAMP")
    return value


def _night(root, journal_root, night, input_kind, global_gate, artifacts):
    from src import history
    from src.operations.publication import classify_night_authority
    import pandas as pd

    ref = "night:" + night
    authority = classify_night_authority(root, journal_root, night)
    state = authority["state"]
    paths = history.nightly_paths(root, night)
    refs = []
    for name in ("manifest", "loci", "alerts"):
        path = paths[name]
        if path.exists() or path.is_symlink():
            artifact = _artifact(path, f"night:{night}:{path.name}")
            artifacts.append(artifact)
            refs.append(artifact.identity)
    def unavailable(content, view, reason, manifest=None):
        qualification = _qualification(state, "UNAVAILABLE", content, view, reason, ref, manifest)
        return AntaresNightRecord(night, qualification), (), _evidence(ref, authority, refs)
    if state != "COMPLETE":
        content = "MISSING" if not authority["manifest_present"] else "UNKNOWN"
        view = "MISSING" if state == "NOT_COMMITTED" and not authority["pending_pre_gate"] else state
        return unavailable(content, view, authority.get("reason") or state)
    if global_gate:
        return unavailable("UNKNOWN", "UNAVAILABLE", "GLOBAL_PUBLICATION_GATE")
    if not all(paths[name].is_file() for name in ("manifest", "loci", "alerts")):
        return unavailable("MISSING", "UNAVAILABLE", "MISSING_REQUIRED_PRODUCT")
    manifest = json.loads(_regular_bytes(paths["manifest"]))
    synthetic = (manifest.get("synthetic") is True or manifest.get("provider") == "synthetic"
                 or str(manifest.get("schema_version", "")).startswith("phase5.synthetic"))
    if synthetic and input_kind != "SYNTHETIC_AUTHORITY_FIXTURE":
        raise NativeReadRefused("Synthetic source cannot be labeled saved science")
    try:
        lower, lower_missing = numeric_value(manifest.get("mjd_min"))
        upper, upper_missing = numeric_value(manifest.get("mjd_max"))
        if lower_missing or upper_missing or lower >= upper or manifest.get("date_utc") != night:
            raise NativeReadRefused("INVALID_MANIFEST_NIGHT_BOUNDS")
        if history.mjd_to_utc_date(lower) != night:
            raise NativeReadRefused("MJD_CALENDAR_NIGHT_MISMATCH")
        if manifest.get("status") not in {"complete", "under_target"}:
            raise NativeReadRefused("UNQUALIFIED_ACQUISITION_STATE")
        if history.recorded_query_fetch_errors(manifest):
            raise NativeReadRefused("RECORDED_QUERY_FETCH_ERRORS")
        combined = manifest.get("query_fetch_evidence")
        if combined is not None and (combined.get("query_completed") is not True
                                     or combined.get("fetch_completed") is not True):
            raise NativeReadRefused("INCOMPLETE_QUERY_FETCH_EVIDENCE")
        for stage in ("query_evidence", "fetch_evidence"):
            evidence = manifest.get(stage)
            if evidence is not None and (evidence.get("completed") is not True or evidence.get("partial") is True):
                raise NativeReadRefused("INCOMPLETE_QUERY_FETCH_EVIDENCE")
        expected = manifest.get("authority", {}).get("candidate", {}).get("artifacts", {})
        expected = expected or manifest.get("artifacts", {})
        for name in ("loci", "alerts"):
            identity = next(item for item in artifacts if item.identity == f"night:{night}:{paths[name].name}")
            binding = expected.get(paths[name].name)
            if binding and (binding.get("sha256") != identity.sha256 or binding.get("bytes") != identity.bytes):
                raise NativeReadRefused("ARTIFACT_HASH_MISMATCH")
        if "authority" in manifest and not all(name in expected for name in ("loci.parquet", "alerts.parquet")):
            raise NativeReadRefused("V3_ARTIFACT_BINDING_MISSING")
        with history.authoritative_read(root):
            loci, alerts = pd.read_parquet(paths["loci"]), pd.read_parquet(paths["alerts"])
            for key, count in (("actual_loci", len(loci)), ("alert_rows", len(alerts))):
                declared = manifest.get(key)
                if isinstance(declared, bool) or not isinstance(declared, int) or declared != count:
                    raise NativeReadRefused("MANIFEST_ROW_COUNT_MISMATCH")
            if "locus_id" not in loci or "locus_id" not in alerts:
                raise NativeReadRefused("LOCUS_LINK_COLUMN_MISSING")
            if loci.empty:
                # Existing conservative empty-night proof; discard its generated timestamp.
                history.revalidate_zero_row_night(root, night)
            upper_bound = manifest.get("validation", {}).get("mjd_upper_bound", "UNKNOWN")
            if upper_bound not in {"inclusive", "exclusive", "UNKNOWN"}:
                raise NativeReadRefused("INVALID_QUERY_INTERVAL_SEMANTICS")
            snapshots = _snapshots(loci, alerts, night, ref, lower, upper, upper_bound)
        content = "ZERO_ROW_NIGHT" if loci.empty else "COMPLETE_WITH_ROWS"
        chronology = manifest.get("authority", {}).get("chronology", {})
        qualification = _qualification(state, "AVAILABLE", content,
            "PUBLISHED_ZERO_ROW_NIGHT" if loci.empty else "PUBLISHED_WITH_ROWS",
            "LEGACY_AUTHORITY_ACCEPTED_BY_NATIVE_CLASSIFIER" if authority["legacy"] else
            "NATIVE_AUTHORITY_AND_SAVED_CONTENT_VERIFIED", ref, manifest, authority["legacy"])
        temporal = AntaresNightRecord(night, qualification, lower, upper, len(loci), len(alerts),
            acquisition_started_at_utc=_utc_timestamp(manifest.get("started_at_utc")),
            fetch_completed_at_utc=_utc_timestamp(manifest.get("fetch_evidence", {}).get("details", {}).get("request_completed_at_utc")),
            publication_started_at_utc=_utc_timestamp(chronology.get("publication_transaction_started_at_utc")),
            acquisition_completed_at_utc=_utc_timestamp(manifest.get("ingested_at_utc")),
            build_finished_at_utc=_utc_timestamp(manifest.get("finished_at_utc")),
            mjd_upper_bound=upper_bound)
        authority = dict(authority, acquisition_state=manifest.get("status"),
                         synthetic_source=synthetic, chronology=chronology,
                         mjd_upper_bound=manifest.get("validation", {}).get("mjd_upper_bound", "UNKNOWN"))
        return temporal, snapshots, _evidence(ref, authority, refs,
            ("Saved alerts may contain historical observations acquired later.",
             "Observation MJD time scale is not established by the saved-column contract."))
    except NativeReadRefused as exc:
        return unavailable("UNKNOWN", "UNAVAILABLE", str(exc), manifest)
    except (ValueError, KeyError, TypeError, OSError):
        # No partial rows survive a rejected partition. Stable reason, no host-specific path.
        return unavailable("UNKNOWN", "UNAVAILABLE", "SAVED_CONTENT_REJECTED", manifest)


def _assemble(code_sha, code_state, release, input_kind, start, end, artifacts,
              nights, snapshots, provenance, generation=None, sentinel_ref=None, max_loci=10000):
    by_locus = {}
    for locus_id, snapshot in snapshots:
        by_locus.setdefault(locus_id, []).append(snapshot)
    if len(by_locus) > max_loci:
        raise NativeReadRefused("Native entity bound exceeded; select a smaller basis interval")
    loci = tuple(AntaresLocusRecord(locus_id, tuple(row.membership_night for row in rows), tuple(rows))
                 for locus_id, rows in sorted(by_locus.items()))
    night_refs = tuple(item.identity for item in provenance if item.identity.startswith("night:"))
    feature_refs = ("semantics:features",)
    definitions = feature_registry((row for locus in loci for row in locus.snapshots), feature_refs)
    available = any(night.qualification.availability == "AVAILABLE" for night in nights)
    partial = "PARTIALLY_QUALIFIED" if loci else "UNAVAILABLE"
    capabilities = (
        Capability("night_authority", "AVAILABLE" if nights else "UNAVAILABLE", "Native classifier evidence; science availability is independent", night_refs),
        Capability("night_saved_row_counts", "AVAILABLE" if available else "UNAVAILABLE", "Qualified saved partition counts only", night_refs),
        Capability("locus_snapshots", "AVAILABLE" if loci else "UNAVAILABLE", "Saved membership snapshots; no universal Rubin identity", night_refs),
        Capability("sky_locus_density", partial, "Native equal-solid-angle aggregation; celestial frame unestablished", ("semantics:sky",)),
        Capability("broker_features", "PARTIALLY_QUALIFIED" if any(item.finite_rows for item in definitions if item.namespace == "antares.broker") else "UNAVAILABLE", "No generating broker estimator is committed here", feature_refs),
        Capability("tags", partial, "Broker labels; not astrophysical truth or calibrated classification", night_refs),
        Capability("photometric_summary", "UNAVAILABLE", "No qualified cross-survey photometric estimator in this gate", feature_refs),
        Capability("survey_footprint", "UNAVAILABLE", "Source density does not establish survey coverage", ("semantics:sky",)),
        Capability("color", "UNAVAILABLE", "Separate band values do not establish a scientific color", feature_refs),
    )
    basis = AntaresObservatoryBasis(code_sha, code_state, release, input_kind, start, end,
        tuple(sorted(artifacts, key=lambda item: item.identity)), night_refs, generation, sentinel_ref)
    return AntaresObservatoryProduct(basis, AntaresCapabilityManifest(capabilities, definitions),
        AntaresTemporalProduct(tuple(nights), night_refs), build_sky(loci, ("semantics:sky", *night_refs)),
        loci, tuple(sorted(provenance, key=lambda item: item.identity)),
        "METADATA / CONTRACT FIXTURE" if input_kind == "METADATA_CONTRACT_FIXTURE" else
        "SYNTHETIC AUTHORITY FIXTURE" if input_kind == "SYNTHETIC_AUTHORITY_FIXTURE" else "SAVED NATIVE READ PRODUCT")


def _semantic_provenance(artifacts):
    refs = tuple(item.identity for item in artifacts if item.identity.startswith("code:"))
    return [
        _evidence("semantics:features", {"source": "src/feature_analysis.py", "estimator_implementation": "NOT_PRESENT_IN_REPOSITORY", "colors": "UNAVAILABLE"}, refs),
        _evidence("semantics:sky", {"scheme": "uniform-ra-sin-dec.v1", "frame": "UNESTABLISHED", "footprint": "UNAVAILABLE"}, refs),
        _evidence("semantics:time", {"night_conversion": "history uses astropy UTC MJD; legacy exporter uses UTC epoch plus timedelta", "observation_scale": "UNESTABLISHED", "legacy_exporter_limitation": "UTC calendar convention does not qualify broker observation time scale or leap-second handling"}, refs),
    ]


def build_metadata_fixture(*, code_sha, code_state="UNVERIFIED", release=None):
    """Zero science rows; committed implementation evidence only."""
    _code_identity(code_sha, code_state)
    artifacts = list(_code_artifacts())
    return _assemble(code_sha, code_state, release, "METADATA_CONTRACT_FIXTURE", None, None,
                     artifacts, [], [], _semantic_provenance(artifacts))


def build_native_product(data_root, journal_root, *, start, end, code_sha,
                         input_kind, code_state="UNVERIFIED", release=None,
                         ribbon_start=None, ribbon_end=None, sentinel_path=None, max_loci=10000):
    """Read explicit offline roots; never discover or contact production.

    Input kind is a caller attestation, not publication authority. Known
    synthetic manifests cannot be relabeled as saved science. Gate-held states
    emit status evidence only, and never provide source rows.
    """
    from src import history
    _code_identity(code_sha, code_state)
    if input_kind not in {"SYNTHETIC_AUTHORITY_FIXTURE", "SAVED_SCIENCE_SNAPSHOT"}:
        raise NativeReadRefused("Explicit saved-science or synthetic-fixture input kind required")
    if isinstance(max_loci, bool) or not isinstance(max_loci, int) or not 1 <= max_loci <= 100000:
        raise NativeReadRefused("max_loci must be an integer in [1, 100000]")
    covered = set(_dates(start, end))
    ribbon = _dates(ribbon_start or start, ribbon_end or end)
    if not covered.issubset(ribbon):
        raise NativeReadRefused("Ribbon interval must contain the complete basis interval")
    root, journals = Path(data_root), Path(journal_root)
    if any(part.is_symlink() for path in (root, journals) for part in (path, *path.parents)):
        raise NativeReadRefused("Symlinked input roots are forbidden")
    artifacts = list(_code_artifacts())
    provenance = _semantic_provenance(artifacts)
    nights, snapshots = [], []
    with history.authority_read_lock(root), ExitStack() as reads:
        gate = history.publication_gate_path(root)
        global_gate = gate.exists() or gate.is_symlink()
        if not global_gate:
            reads.enter_context(history.authoritative_read(root))
        if global_gate:
            artifacts.append(_artifact(gate, "authority:publication-gate"))
        if journals.exists():
            for path in sorted(journals.iterdir()):
                if not path.name.startswith("."):
                    artifacts.append(_artifact(path, "authority:journal:" + path.name))
        for key, path in history.cumulative_paths(root).items():
            if key != "dir" and path.exists():
                artifacts.append(_artifact(path, "generation:" + key))
        sentinel_ref = None
        if sentinel_path is not None:
            if Path(sentinel_path).is_symlink():
                raise NativeReadRefused("Symlinked Sentinel input is forbidden")
            sentinel_ref = "authority:supplied-sentinel"
            artifacts.append(_artifact(Path(sentinel_path).resolve(strict=True), sentinel_ref))
            provenance.append(_evidence("sentinel:supplied", {"status": "SUPPLIED_IDENTITY_ONLY; NOT_REQUALIFIED"}, (sentinel_ref,)))
        for night in ribbon:
            if night not in covered:
                ref = "night:" + night
                qualification = _qualification("UNKNOWN", "OUTSIDE_BASIS", "UNKNOWN", "OUTSIDE_BASIS_COVERAGE", "OUTSIDE_BASIS_COVERAGE", ref)
                nights.append(AntaresNightRecord(night, qualification))
                provenance.append(_evidence(ref, {"basis_membership": "OUTSIDE"}))
                continue
            record, rows, evidence = _night(root, journals, night, input_kind, global_gate, artifacts)
            nights.append(record)
            snapshots.extend(rows)
            provenance.append(evidence)
        generation_artifacts = [item for item in artifacts if item.identity.startswith(("generation:", "authority:", "night:"))]
        generation = hashlib.sha256(canonical_bytes(generation_artifacts)).hexdigest()
        result = _assemble(code_sha, code_state, release, input_kind, start, end, artifacts,
                           nights, snapshots, provenance, generation, sentinel_ref, max_loci)
        canonical_bytes(result)  # Fail before returning malformed/nonfinite JSON.
        return result
