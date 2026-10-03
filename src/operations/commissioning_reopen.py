"""Fresh-process forensic reopening for a retained Phase 6 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .journal import TransactionJournal
from .science import ARTIFACT_NAMES, reopen_and_validate_artifacts


def _read_regular_nofollow(path: Path) -> bytes:
    descriptor = os.open(
        str(path),
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"Candidate artifact is not a regular file: {path}.")
        chunks = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            observed.st_dev,
            observed.st_ino,
            observed.st_size,
            observed.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"Candidate artifact changed while reopening: {path}.")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _schema(frame: Any) -> Mapping[str, str]:
    return {str(name): str(dtype) for name, dtype in frame.dtypes.items()}


def _json_regular(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(_read_regular_nofollow(path).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Forensic JSON is invalid: {path}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Forensic JSON is not an object: {path}.")
    return value


def reopen_candidate(
    stage: Path,
    *,
    run_root: Path,
    journal: Path,
    query_evidence: Path,
    fetch_evidence: Path,
    production_target: Path,
    expected_release_sha: str,
) -> Dict[str, Any]:
    stage = Path(stage)
    run_root = Path(run_root)
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("Commissioning run root is missing or unsafe.")
    canonical_run = run_root.resolve(strict=True)
    if stage.is_symlink() or not stage.is_dir():
        raise ValueError("Candidate stage is missing or unsafe.")
    canonical_stage = stage.resolve(strict=True)
    try:
        canonical_stage.relative_to(canonical_run)
    except ValueError as exc:
        raise ValueError("Candidate stage escaped the exact commissioning run.") from exc
    if production_target.exists() or production_target.is_symlink():
        raise ValueError("Authoritative production target is not absent.")

    snapshot = TransactionJournal.load(Path(journal)).snapshot
    journal_checks = {
        "state_validated": snapshot.state.value == "validated",
        "release_matches": snapshot.descriptor.release_sha == expected_release_sha,
        "stage_matches": Path(snapshot.descriptor.stage_path).resolve(strict=True)
        == canonical_stage,
        "publication_unavailable": snapshot.publication.get("available") is False,
        "publication_unattempted": snapshot.publication.get("attempted") is False,
    }
    if not all(journal_checks.values()):
        failed = ",".join(
            name for name, passed in journal_checks.items() if not passed
        )
        raise ValueError(f"Journal candidate checks failed: {failed}.")
    lock_path = Path(snapshot.descriptor.lock_path)
    if lock_path.is_symlink() or not lock_path.is_dir():
        raise ValueError("Writer lock was not held during the forensic reopen.")

    query_document = _json_regular(Path(query_evidence))
    fetch_document = _json_regular(Path(fetch_evidence))
    payloads = {
        name: _read_regular_nofollow(stage / name)
        for name in ARTIFACT_NAMES
    }
    reopened = reopen_and_validate_artifacts(payloads)
    if (
        query_document.get("evidence") != reopened.manifest.get("query_evidence")
        or fetch_document.get("evidence") != reopened.manifest.get("fetch_evidence")
        or query_document.get("provider") != "live-antares"
        or fetch_document.get("provider") != "live-antares"
    ):
        raise ValueError("External query/fetch evidence disagrees with the candidate.")
    artifacts = {}
    for name in ARTIFACT_NAMES:
        observed = (stage / name).stat()
        artifacts[name] = {
            "device": observed.st_dev,
            "inode": observed.st_ino,
            "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
            "bytes": len(payloads[name]),
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
        }
    return {
        "schema_version": "phase6.independent-reopen.v1",
        "passed": True,
        "process_id": os.getpid(),
        "run_root": str(canonical_run),
        "stage": str(canonical_stage),
        "release_sha": expected_release_sha,
        "date_utc": reopened.manifest.get("date_utc"),
        "loci_rows": len(reopened.loci),
        "alert_rows": len(reopened.alerts),
        "loci_schema": _schema(reopened.loci),
        "alerts_schema": _schema(reopened.alerts),
        "validation": reopened.manifest.get("validation"),
        "query_evidence": reopened.manifest.get("query_evidence"),
        "fetch_evidence": reopened.manifest.get("fetch_evidence"),
        "artifacts": artifacts,
        "journal": {
            "path": str(Path(journal).resolve(strict=True)),
            "state": snapshot.state.value,
            "outcome": snapshot.outcome.value,
            "release_sha": snapshot.descriptor.release_sha,
            "stage_path": snapshot.descriptor.stage_path,
            "publication_available": snapshot.publication.get("available"),
            "publication_attempted": snapshot.publication.get("attempted"),
            "writer_lock_observed": True,
        },
        "query_evidence_path": str(Path(query_evidence).resolve(strict=True)),
        "fetch_evidence_path": str(Path(fetch_evidence).resolve(strict=True)),
        "production_target": str(production_target),
        "production_target_absent": True,
        "publication_observed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.operations.commissioning_reopen"
    )
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--query-evidence", type=Path, required=True)
    parser.add_argument("--fetch-evidence", type=Path, required=True)
    parser.add_argument("--production-target", type=Path, required=True)
    parser.add_argument("--expected-release-sha", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(
            json.dumps(
                reopen_candidate(
                    args.stage,
                    run_root=args.run_root,
                    journal=args.journal,
                    query_evidence=args.query_evidence,
                    fetch_evidence=args.fetch_evidence,
                    production_target=args.production_target,
                    expected_release_sha=args.expected_release_sha,
                ),
                sort_keys=True,
            )
        )
    except Exception as exc:
        # Exception messages name local paths/schema failures only.  Provider
        # transport exceptions and credentials never reach this process.
        print(
            json.dumps(
                {
                    "schema_version": "phase6.independent-reopen.v1",
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["reopen_candidate"]
