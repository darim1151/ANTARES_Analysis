"""Execute or resume the single Control-approved June 27 publication canary.

This module is intentionally not a general production CLI.  Every production
path, accepted baseline, candidate identity, and resulting cumulative identity
is fixed below.  Initial execution reads one ephemeral 256-bit Control token
from standard input; only its SHA-256 digest is persisted.  ``--resume`` is
accepted only when the exact durable gate and journal already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import stat
import sys
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .commissioning import capture_production_sentinel
from .publication import (
    AuthorityState,
    NightPublisher,
    ProductionPublicationBinding,
    authorize_publication,
    authoritative_nights,
    issue_production_publication_capability,
    load_authorization,
    load_offline_recovery_candidate,
    plan_cumulative_extension,
    production_authority_lock_identity,
    production_binding_from_sentinel,
    read_publication_gate,
    summary_source_manifest,
    _read_json,
    _read_regular,
    _write_json_new,
)
from .storage import (
    PRODUCTION_AUTHORITY_ROOT,
    PRODUCTION_CONTROL_ROOT,
    PRODUCTION_DATA_ROOT,
    PRODUCTION_EVIDENCE_ROOT,
    PRODUCTION_STAGE_ROOT,
)
from .writer import EXPECTED_ARTIFACTS, _ensure_private_tree


NIGHT = "2026-06-27"
PREDECESSOR = "2026-06-26"
CANDIDATE_RELEASE = "4378bce9a78e250dc897a9252b21dc69d41dcd0a"
CANDIDATE_ROOT = Path(
    "/astro/store/shire/ANTARES/work/canary/"
    "phase6f-recovery-0.4.3-4378bce-20260627-20260928T185444Z"
)
CACHE_ROOT = Path("/astro/store/shire/ANTARES/cache")
SEGMENT_CACHE_ROOT = Path(
    "/astro/store/shire/ANTARES/work/cache/fetch-segments-v1"
)
EVIDENCE_DIR = PRODUCTION_EVIDENCE_ROOT / "g4-20260627"
EXPECTED_SENTINEL = "52d9d30f0e004622485ba819af3bb56c81b704c22e7618b086a4ac398a76bf63"
EXPECTED_MANIFEST_COUNT = 90
EXPECTED_DURABLE_FILES = 324
EXPECTED_DURABLE_BYTES = 1141241743
EXPECTED_BASELINE = {
    "loci_index": "f75196d18690e610ab6e79231b244c3fddca396a68eea08dd2d0408e91d8b587",
    "nightly_summary": "85c5fac9c242fa2e7993155036ada649336b0affe8ffc8843d2c5733ea765114",
}
EXPECTED_CANDIDATE = {
    "loci.parquet": "86974614dc66349b5f0ad575905e81b8affdf6761205e4dbf7b59efa583f6f0c",
    "alerts.parquet": "2942e07c190d8bed1931c49c4f1f46149b4e39bcc2a440d36860c972d9cd5198",
    "manifest.json": "e5a9b2d3803df2bfec8f4bfd2caf230b12eda51f713eda2280bf3f22526647ad",
}
EXPECTED_RESULT = {
    "loci_index": "1cc0bad54423070cd8326db56409e618acb23123041070daab0b35314f1ca91f",
    "nightly_summary": "20a8230af18c3b5876ac778dc1747ffb91fe447caeea387da0d569fa1202d9f2",
}


class ProductionCanaryRefused(RuntimeError):
    """The exact G4 execution barrier did not pass."""


def _is_sha(value: str, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _verify_release(release_sha: str, wheel_sha256: str) -> Dict[str, str]:
    if not _is_sha(release_sha, 40) or not _is_sha(wheel_sha256, 64):
        raise ProductionCanaryRefused("Release or wheel identity is malformed.")
    if metadata.version("antares-analysis") != "0.4.7":
        raise ProductionCanaryRefused("Installed package version is not 0.4.7.")
    release_root = Path(sys.prefix).resolve().parent
    if release_root.name != release_sha:
        raise ProductionCanaryRefused("Interpreter is not in the exact release directory.")
    release_marker = release_root / "RELEASE_SHA"
    wheel_marker = release_root / "WHEEL_SHA256"
    if (
        release_marker.is_symlink()
        or wheel_marker.is_symlink()
        or release_marker.read_text(encoding="ascii").strip() != release_sha
        or wheel_marker.read_text(encoding="ascii").strip() != wheel_sha256
    ):
        raise ProductionCanaryRefused("Immutable release markers differ.")
    module_root = Path(__file__).resolve()
    if Path(sys.prefix).resolve() not in module_root.parents:
        raise ProductionCanaryRefused("Production module was not loaded from the release venv.")
    if sys.version_info[:3] != (3, 11, 16):
        raise ProductionCanaryRefused("Installed interpreter is not Python 3.11.16.")
    return {
        "release_sha": release_sha,
        "wheel_sha256": wheel_sha256,
        "release_root": str(release_root),
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "package_version": "0.4.7",
        "module": str(module_root),
    }


def _read_control_token() -> str:
    token = sys.stdin.read().strip()
    if not _is_sha(token, 64):
        raise ProductionCanaryRefused(
            "Initial execution requires exactly one 256-bit lowercase hex token on stdin."
        )
    return token


def _verify_lock() -> Dict[str, Any]:
    identity = production_authority_lock_identity()
    if (
        identity["mode"] != "0600"
        or identity["uid"] != os.geteuid()
        or identity["size"] != 0
    ):
        raise ProductionCanaryRefused("Canonical authority-lock identity differs.")
    lock_path = Path(identity["path"])
    for directory in (lock_path.parent.parent, lock_path.parent):
        observed = directory.lstat()
        if (
            directory.is_symlink()
            or not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or observed.st_uid != os.geteuid()
        ):
            raise ProductionCanaryRefused(
                f"Canonical authority-lock directory differs: {directory}."
            )
    return identity


def _verify_candidate():
    candidate = load_offline_recovery_candidate(CANDIDATE_ROOT)
    if (
        candidate.date_utc != NIGHT
        or candidate.release_sha != CANDIDATE_RELEASE
        or candidate.loci != 331786
        or candidate.alerts != 509431
        or candidate.artifact_sha256() != EXPECTED_CANDIDATE
        or candidate.provenance.get("binding_sha256") is None
    ):
        raise ProductionCanaryRefused("Accepted June 27 candidate identity differs.")
    return candidate


def _verify_baseline(sentinel: Mapping[str, Any]) -> Dict[str, Any]:
    state = sentinel["durable_state"]
    production = production_binding_from_sentinel(sentinel)
    predicates = production["predicates"]
    if (
        socket.getfqdn() != "arnor.astro.washington.edu"
        or production["canonical_root"] != str(PRODUCTION_DATA_ROOT)
        or production["durable_fingerprint_sha256"] != EXPECTED_SENTINEL
        or production["manifest_count"] != EXPECTED_MANIFEST_COUNT
        or state["durable_file_count"] != EXPECTED_DURABLE_FILES
        or state["durable_bytes"] != EXPECTED_DURABLE_BYTES
        or production["cumulative_sha256"] != EXPECTED_BASELINE
        or predicates["target_absent"] is not True
        or predicates["cache_absent"] is not True
        or predicates["transaction_artifacts"] != []
        or tuple(authoritative_nights(PRODUCTION_DATA_ROOT))[-1:] != (PREDECESSOR,)
        or read_publication_gate(PRODUCTION_DATA_ROOT) is not None
    ):
        raise ProductionCanaryRefused("Fresh production baseline differs from G4 Control.")
    return production


def _plan(candidate, sentinel: Mapping[str, Any]):
    payloads = {
        name: _read_regular(candidate.candidate_dir / name)
        for name in EXPECTED_ARTIFACTS
    }
    manifest = summary_source_manifest(
        json.loads(payloads["manifest.json"].decode("utf-8")), candidate
    )
    plan = plan_cumulative_extension(
        PRODUCTION_DATA_ROOT, manifest, payloads["loci.parquet"]
    )
    production = production_binding_from_sentinel(sentinel)
    if (
        dict(plan.baseline_sha256) != EXPECTED_BASELINE
        or dict(plan.baseline_sha256) != production["cumulative_sha256"]
        or dict(plan.expected_sha256) != EXPECTED_RESULT
    ):
        raise ProductionCanaryRefused("Accepted cumulative plan differs.")
    return plan


def _prepare_infrastructure() -> None:
    for directory in (
        PRODUCTION_STAGE_ROOT,
        PRODUCTION_CONTROL_ROOT / "journals",
        PRODUCTION_EVIDENCE_ROOT,
    ):
        _ensure_private_tree(directory, PRODUCTION_AUTHORITY_ROOT)


def _initial(
    release: Mapping[str, str], token: str, candidate, lock: Mapping[str, Any]
):
    if EVIDENCE_DIR.exists() or EVIDENCE_DIR.is_symlink():
        raise ProductionCanaryRefused("G4 pre-publication evidence already exists.")
    _prepare_infrastructure()
    sentinel = capture_production_sentinel(PRODUCTION_DATA_ROOT, CACHE_ROOT, NIGHT)
    production = _verify_baseline(sentinel)
    plan = _plan(candidate, sentinel)
    now = _utc_now()
    authorization = authorize_publication(
        candidate,
        production=production,
        predecessor_date_utc=PREDECESSOR,
        publisher_release_sha=release["release_sha"],
        expected_cumulative_sha256=EXPECTED_RESULT,
        authorized_by="ANTARES-Control-G4",
        authorized_at_utc=now.isoformat(),
        expires_at_utc=(now + timedelta(hours=2)).isoformat(),
        nonce=os.urandom(16).hex(),
    )
    token_sha256 = hashlib.sha256(token.encode("ascii")).hexdigest()
    binding = ProductionPublicationBinding(
        hostname=socket.getfqdn(),
        service_uid=os.geteuid(),
        production_root=str(PRODUCTION_DATA_ROOT),
        stage_root=str(PRODUCTION_STAGE_ROOT),
        control_root=str(PRODUCTION_CONTROL_ROOT),
        evidence_root=str(PRODUCTION_EVIDENCE_ROOT),
        sentinel_cache_path=str(CACHE_ROOT),
        segment_cache_root=str(SEGMENT_CACHE_ROOT),
        mount_binding=production["mount_binding"],
        sentinel_fingerprint_sha256=production["durable_fingerprint_sha256"],
        manifest_count=production["manifest_count"],
        authority_lock=lock,
        night_utc=NIGHT,
        predecessor_night_utc=PREDECESSOR,
        candidate_root=str(CANDIDATE_ROOT),
        candidate_record_sha256=candidate.record_sha256,
        candidate_binding_sha256=str(candidate.provenance["binding_sha256"]),
        candidate_provenance_sha256=candidate.provenance_sha256,
        artifact_sha256=candidate.artifact_sha256(),
        publisher_release_sha=release["release_sha"],
        candidate_release_sha=candidate.release_sha,
        cumulative_baseline_sha256=EXPECTED_BASELINE,
        expected_cumulative_sha256=EXPECTED_RESULT,
        expected_cumulative_schema_sha256=plan.schema_sha256,
        authorization_sha256=authorization.digest,
        control_token_sha256=token_sha256,
        nonce=authorization.nonce,
        expires_at_utc=authorization.expires_at_utc,
    )
    capability = issue_production_publication_capability(
        binding,
        authorization,
        candidate,
        control_token=token,
        sentinel=sentinel,
        now=now,
        authority_lock=lock,
    )
    _ensure_private_tree(EVIDENCE_DIR, PRODUCTION_AUTHORITY_ROOT)
    _write_json_new(EVIDENCE_DIR / "production-sentinel-before.json", sentinel)
    _write_json_new(EVIDENCE_DIR / "authorization.json", authorization.as_dict())
    _write_json_new(EVIDENCE_DIR / "binding.json", binding.as_dict())
    _write_json_new(
        EVIDENCE_DIR / "pre-publication.json",
        {
            "schema_version": "antares.v3-g4.pre-publication.v1",
            "release": dict(release),
            "candidate_record_sha256": candidate.record_sha256,
            "candidate_provenance_sha256": candidate.provenance_sha256,
            "candidate_artifact_sha256": candidate.artifact_sha256(),
            "sentinel_fingerprint_sha256": production["durable_fingerprint_sha256"],
            "manifest_count": production["manifest_count"],
            "authority_tail": PREDECESSOR,
            "cumulative_baseline_sha256": EXPECTED_BASELINE,
            "expected_cumulative_sha256": EXPECTED_RESULT,
            "expected_cumulative_schema_sha256": plan.schema_sha256,
            "authority_lock": dict(lock),
            "authorization_sha256": authorization.digest,
            "nonce": authorization.nonce,
            "expires_at_utc": authorization.expires_at_utc,
            "maximum_successful_uses": 1,
            "target_absent": True,
            "gate_absent": True,
            "control_token_sha256": token_sha256,
        },
    )
    return capability, authorization


def _resume(release: Mapping[str, str], candidate, lock: Mapping[str, Any]):
    pre = _read_json(EVIDENCE_DIR / "pre-publication.json")
    authorization = load_authorization(EVIDENCE_DIR / "authorization.json")
    binding = ProductionPublicationBinding(
        **_read_json(EVIDENCE_DIR / "binding.json")
    )
    sentinel = _read_json(EVIDENCE_DIR / "production-sentinel-before.json")
    if (
        pre.get("release") != dict(release)
        or pre.get("authorization_sha256") != authorization.digest
        or binding.authorization_sha256 != authorization.digest
        or binding.authority_lock != dict(lock)
    ):
        raise ProductionCanaryRefused("Persisted G4 resume evidence differs.")
    capability = issue_production_publication_capability(
        binding,
        authorization,
        candidate,
        control_token=None,
        sentinel=sentinel,
        authority_lock=lock,
    )
    return capability, authorization


def execute(release_sha: str, wheel_sha256: str, *, resume: bool) -> Dict[str, Any]:
    release = _verify_release(release_sha, wheel_sha256)
    candidate = _verify_candidate()
    lock = _verify_lock()
    if resume:
        capability, authorization = _resume(release, candidate, lock)
    else:
        capability, authorization = _initial(
            release, _read_control_token(), candidate, lock
        )
    publisher = NightPublisher(
        capability,
        publisher_release_sha=release_sha,
        cache_root=CACHE_ROOT,
        lock_wait_seconds=60.0,
    )
    outcome = publisher.publish(candidate, authorization)
    return {
        "success": outcome.success,
        "status": outcome.status,
        "authority_state": outcome.authority_state,
        "transaction_id": outcome.record.get("transaction_id"),
        "authorization_sha256": authorization.digest,
        "binding_sha256": capability.binding_sha256,
        "control_token_sha256": capability.control_token_sha256,
        "record_path": str(outcome.record_path) if outcome.record_path else None,
        "retryable": outcome.retryable,
        "failure_category": outcome.record.get("failure_category"),
        "message": outcome.record.get("message"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute(args.release_sha, args.wheel_sha256, resume=args.resume)
    except Exception as exc:
        result = {
            "success": False,
            "status": "REFUSED",
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }
        print(json.dumps(result, sort_keys=True))
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0 if result["success"] and result["authority_state"] == AuthorityState.COMPLETE.value else 5


if __name__ == "__main__":
    raise SystemExit(main())
