"""Durable, non-authoritative Phase 6 query-result checkpoints.

The live provider intentionally separates exhaustive query enumeration from
per-locus history fetching.  This module gives that boundary a run-local,
manifest-bound representation which can be reopened without invoking a search.

The checkpoint format is deliberately not Parquet.  ANTARES locus properties
contain nested mappings for which Arrow may conflate a missing struct member
with a present member whose value is null.  The canonical JSON-lines codec
below represents mappings as sorted key/value pairs, so those two scientific
states remain distinct.  A terminal commit marker is written only after every
data file and the manifest have been fsynced.  A directory without that marker
is incomplete and is never reusable.

Checkpoints are evidence beneath one run root.  They are explicitly
non-authoritative and cannot be used as publication artifacts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .science import (
    NightQueryResult,
    NightScienceRequest,
    ProviderOutcome,
    QueryStageEvidence,
)


CHECKPOINT_SCHEMA_VERSION = "phase6.query-result-checkpoint.v1"
COMMIT_SCHEMA_VERSION = "phase6.query-result-checkpoint-commit.v1"
RECORD_CODEC_VERSION = "phase6.canonical-jsonl.v1"
CHECKPOINT_PARENT_NAME = "checkpoints"
DEFAULT_CHECKPOINT_NAME = "query-result"
MANIFEST_NAME = "manifest.json"
QUERY_EVIDENCE_NAME = "query-evidence.json"
COMMIT_MARKER_NAME = "COMMITTED.json"
NON_AUTHORITATIVE_CLASSIFICATION = "run-local-non-authoritative-query-result"
DEFAULT_ROWS_PER_SHARD = 10_000

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TAG = "$phase6"
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class QueryCheckpointError(RuntimeError):
    """A query checkpoint is incomplete, unsafe, corrupt, or mismatched."""


@dataclass(frozen=True)
class QueryResultCheckpointBindings:
    """Stable identities a later process must independently know.

    The potentially large query evidence is intentionally not a caller-supplied
    load expectation.  It is copied from the sealed ``NightQueryResult``, hashed
    into the manifest and commit marker, and verified internally when reopened.
    """

    run_id: str
    release_sha: str
    configuration_hash: str
    target_date_utc: str
    provider_name: str
    provider_scenario: str
    query_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        _safe_component(self.run_id, "run_id")
        if not _HEX_40.fullmatch(str(self.release_sha)):
            raise ValueError("release_sha must be one lowercase 40-character SHA.")
        if not _HEX_64.fullmatch(str(self.configuration_hash)):
            raise ValueError(
                "configuration_hash must be one lowercase 64-character SHA-256."
            )
        try:
            parsed = date.fromisoformat(str(self.target_date_utc))
        except (TypeError, ValueError) as exc:
            raise ValueError("target_date_utc must use canonical YYYY-MM-DD form.") from exc
        if parsed.isoformat() != self.target_date_utc:
            raise ValueError("target_date_utc must use canonical YYYY-MM-DD form.")
        for field_name, value in (
            ("provider_name", self.provider_name),
            ("provider_scenario", self.provider_scenario),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if not isinstance(self.query_policy, Mapping):
            raise ValueError("query_policy must be a mapping.")
        normalized_policy = json.loads(_canonical_json_bytes(dict(self.query_policy)))
        if set(normalized_policy) != {"scientific_contract", "execution_policy"}:
            raise ValueError(
                "query_policy must contain exact scientific_contract and execution_policy mappings."
            )
        scientific_contract = normalized_policy["scientific_contract"]
        execution_policy = normalized_policy["execution_policy"]
        if not isinstance(scientific_contract, dict) or not isinstance(
            execution_policy, dict
        ):
            raise ValueError("Query contract and execution policy must be mappings.")
        interval = scientific_contract.get("interval")
        if (
            scientific_contract.get("target_date_utc") != self.target_date_utc
            or scientific_contract.get("query_tag") is not None
            or scientific_contract.get("lsst_only") is not True
            or not isinstance(interval, dict)
            or isinstance(interval.get("mjd_min"), bool)
            or not isinstance(interval.get("mjd_min"), (int, float))
            or isinstance(interval.get("mjd_max"), bool)
            or not isinstance(interval.get("mjd_max"), (int, float))
            or not math.isfinite(float(interval["mjd_min"]))
            or not math.isfinite(float(interval["mjd_max"]))
            or float(interval["mjd_min"]) >= float(interval["mjd_max"])
        ):
            raise ValueError("Scientific query contract is incomplete or contradictory.")


@dataclass(frozen=True)
class LoadedQueryResultCheckpoint:
    """A fully verified checkpoint and reconstructed provider fetch input."""

    path: Path
    query_result: NightQueryResult
    manifest: Mapping[str, Any]
    query_evidence: Mapping[str, Any]
    query_evidence_document: Mapping[str, Any]
    query_evidence_sha256: str
    integrity_sha256: str


def _safe_component(value: object, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or not _SAFE_COMPONENT.fullmatch(text)
    ):
        raise ValueError(f"{label} must be one safe path component.")
    return text


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint metadata is not canonical JSON.") from exc
    return (text + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    """Match the live provider's canonical JSON digest (without a trailing LF)."""
    payload = _canonical_json_bytes(value)
    return _sha256(payload[:-1])


def _binding_payload(bindings: QueryResultCheckpointBindings) -> Dict[str, Any]:
    policy = json.loads(_canonical_json_bytes(dict(bindings.query_policy)))
    payload = {
        "run_id": bindings.run_id,
        "release_sha": bindings.release_sha,
        "configuration_hash": bindings.configuration_hash,
        "target_date_utc": bindings.target_date_utc,
        "provider_name": bindings.provider_name,
        "provider_scenario": bindings.provider_scenario,
        "query_policy": policy,
    }
    payload["query_policy_sha256"] = _sha256(_canonical_json_bytes(policy))
    return payload


def _request_binding(request: NightScienceRequest) -> Dict[str, Any]:
    """Return the scientific/fetch identity, excluding operational ingestion time."""
    if not isinstance(request, NightScienceRequest):
        raise QueryCheckpointError("Checkpoint request is not a NightScienceRequest.")
    prior_ids = list(request.prior_locus_ids)
    return {
        "date_utc": request.date_utc,
        "mjd_min": float(request.mjd_min),
        "mjd_max": float(request.mjd_max),
        "query_tag": request.query_tag,
        "target_loci": request.target_loci,
        "lsst_only": request.lsst_only,
        "range_label": request.range_label,
        "prior_locus_count": len(prior_ids),
        "prior_locus_identity_sha256": _identifier_hash(prior_ids),
        "operational_fields_excluded": ["ingested_at_utc"],
    }


def _canonical_run_root(run_root: Path, run_id: str) -> Path:
    lexical = Path(run_root).expanduser()
    try:
        observed = lexical.lstat()
    except OSError as exc:
        raise QueryCheckpointError("The checkpoint run root is inaccessible.") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise QueryCheckpointError("The checkpoint run root must be a real directory.")
    try:
        canonical = lexical.resolve(strict=True)
    except OSError as exc:
        raise QueryCheckpointError("The checkpoint run root cannot be resolved.") from exc
    if canonical.name != run_id:
        raise QueryCheckpointError("The checkpoint run root does not match its run id.")
    return canonical


def _open_directory_at(parent_fd: int, name: str, label: str) -> int:
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise QueryCheckpointError(f"{label} is missing or unsafe.") from exc
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise QueryCheckpointError(f"{label} is not a directory.")
    return descriptor


def _ensure_parent_directory(run_fd: int) -> int:
    try:
        os.mkdir(CHECKPOINT_PARENT_NAME, 0o700, dir_fd=run_fd)
        os.fsync(run_fd)
    except FileExistsError:
        pass
    except OSError as exc:
        raise QueryCheckpointError("Could not create the checkpoint parent.") from exc
    descriptor = _open_directory_at(
        run_fd, CHECKPOINT_PARENT_NAME, "Checkpoint parent"
    )
    observed = os.fstat(descriptor)
    if stat.S_IMODE(observed.st_mode) != 0o700:
        os.close(descriptor)
        raise QueryCheckpointError("Checkpoint parent mode must be exactly 0700.")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("Short checkpoint write.")
        offset += written


def _write_new_file_at(directory_fd: int, name: str, payload: bytes) -> None:
    try:
        descriptor = os.open(name, _WRITE_FLAGS, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise QueryCheckpointError(f"Could not create checkpoint file {name}.") from exc
    try:
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    except OSError as exc:
        raise QueryCheckpointError(f"Could not durably write checkpoint file {name}.") from exc
    finally:
        os.close(descriptor)


def _entry_snapshot(directory_fd: int, name: str) -> os.stat_result:
    try:
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise QueryCheckpointError(f"Checkpoint entry {name} is inaccessible.") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise QueryCheckpointError(f"Checkpoint entry {name} is not a regular file.")
    if stat.S_IMODE(observed.st_mode) != 0o600:
        raise QueryCheckpointError(f"Checkpoint entry {name} mode must be exactly 0600.")
    return observed


def _read_regular_at(directory_fd: int, name: str) -> bytes:
    initial = _entry_snapshot(directory_fd, name)
    try:
        descriptor = os.open(name, _READ_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise QueryCheckpointError(f"Checkpoint entry {name} is unsafe.") from exc
    try:
        pinned = os.fstat(descriptor)
        if (
            (pinned.st_dev, pinned.st_ino, pinned.st_size)
            != (initial.st_dev, initial.st_ino, initial.st_size)
            or not stat.S_ISREG(pinned.st_mode)
        ):
            raise QueryCheckpointError(f"Checkpoint entry {name} changed before read.")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    ending = _entry_snapshot(directory_fd, name)
    if (
        (ending.st_dev, ending.st_ino, ending.st_size)
        != (initial.st_dev, initial.st_ino, initial.st_size)
        or len(payload) != initial.st_size
    ):
        raise QueryCheckpointError(f"Checkpoint entry {name} changed during read.")
    return payload


def _parse_canonical_object(payload: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueryCheckpointError(f"{label} is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise QueryCheckpointError(f"{label} is not a JSON object.")
    try:
        canonical = _canonical_json_bytes(value)
    except ValueError as exc:
        raise QueryCheckpointError(f"{label} is not canonical JSON.") from exc
    if canonical != payload:
        raise QueryCheckpointError(f"{label} does not use canonical encoding.")
    return value


def _encode_value(value: Any) -> Any:
    """Encode one value without conflating nested missing keys with nulls."""
    if value is None:
        return None
    if value is pd.NA:
        return {_TAG: "pandas-na"}
    if value is pd.NaT:
        return {_TAG: "pandas-nat"}
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_TAG: "float", "value": "nan"}
        if math.isinf(value):
            return {_TAG: "float", "value": "+inf" if value > 0 else "-inf"}
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return {_TAG: "float", "value": "-0x0.0p+0"}
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {_TAG: "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, pd.Timestamp):
        return {_TAG: "timestamp", "value": value.isoformat()}
    if isinstance(value, datetime):
        return {_TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {_TAG: "date", "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {
            _TAG: "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise QueryCheckpointError(
                "Nested checkpoint mappings require string keys."
            )
        items = []
        for key in sorted(value):
            items.append([key, _encode_value(value[key])])
        return {_TAG: "mapping", "items": items}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    raise QueryCheckpointError(
        f"Unsupported normalized query value type: {type(value).__module__}.{type(value).__name__}."
    )


def _decode_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(_TAG)
    if tag == "pandas-na" and set(value) == {_TAG}:
        return pd.NA
    if tag == "pandas-nat" and set(value) == {_TAG}:
        return pd.NaT
    if tag == "float" and set(value) == {_TAG, "value"}:
        return {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
            "-0x0.0p+0": -0.0,
        }.get(value["value"], _INVALID)
    if tag == "bytes" and set(value) == {_TAG, "value"}:
        try:
            return base64.b64decode(value["value"], validate=True)
        except (TypeError, ValueError) as exc:
            raise QueryCheckpointError("Invalid encoded checkpoint bytes.") from exc
    if tag in {"timestamp", "datetime", "date"} and set(value) == {
        _TAG,
        "value",
    }:
        try:
            if tag == "timestamp":
                return pd.Timestamp(value["value"])
            if tag == "datetime":
                return datetime.fromisoformat(value["value"])
            return date.fromisoformat(value["value"])
        except (TypeError, ValueError) as exc:
            raise QueryCheckpointError(f"Invalid encoded checkpoint {tag}.") from exc
    if tag == "timedelta" and set(value) == {
        _TAG,
        "days",
        "seconds",
        "microseconds",
    }:
        try:
            return timedelta(
                days=int(value["days"]),
                seconds=int(value["seconds"]),
                microseconds=int(value["microseconds"]),
            )
        except (TypeError, ValueError) as exc:
            raise QueryCheckpointError("Invalid encoded checkpoint timedelta.") from exc
    if tag == "mapping" and set(value) == {_TAG, "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise QueryCheckpointError("Invalid encoded checkpoint mapping.")
        decoded: Dict[str, Any] = {}
        previous: Optional[str] = None
        for item in items:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or (previous is not None and item[0] <= previous)
            ):
                raise QueryCheckpointError("Invalid encoded checkpoint mapping order.")
            decoded[item[0]] = _decode_value(item[1])
            previous = item[0]
        return decoded
    if tag == "tuple" and set(value) == {_TAG, "items"}:
        if not isinstance(value["items"], list):
            raise QueryCheckpointError("Invalid encoded checkpoint tuple.")
        return tuple(_decode_value(item) for item in value["items"])
    raise QueryCheckpointError("Unknown or malformed checkpoint value tag.")


class _InvalidValue:
    pass


_INVALID = _InvalidValue()


def _decode_checked(value: Any) -> Any:
    decoded = _decode_value(value)
    if decoded is _INVALID:
        raise QueryCheckpointError("Invalid encoded checkpoint float.")
    return decoded


def _frame_contract(frame: pd.DataFrame) -> Tuple[Sequence[str], Sequence[str]]:
    if not isinstance(frame, pd.DataFrame):
        raise QueryCheckpointError("A query checkpoint requires a DataFrame.")
    if not isinstance(frame.index, pd.RangeIndex) or not frame.index.equals(
        pd.RangeIndex(len(frame))
    ):
        raise QueryCheckpointError(
            "The normalized query frame must use the canonical zero-based RangeIndex."
        )
    columns = list(frame.columns)
    if (
        any(not isinstance(value, str) or not value for value in columns)
        or len(columns) != len(set(columns))
    ):
        raise QueryCheckpointError(
            "Normalized query columns must be unique non-empty strings."
        )
    dtypes = [str(frame[column].dtype) for column in columns]
    for column, dtype in zip(columns, dtypes):
        try:
            pd.Series([], dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise QueryCheckpointError(
                f"Checkpoint column {column!r} has an unsupported dtype {dtype!r}."
            ) from exc
    return columns, dtypes


def _locus_ids(frame: pd.DataFrame) -> Sequence[str]:
    if frame.empty:
        if "locus_id" not in frame.columns:
            return []
        values = frame["locus_id"].tolist()
    else:
        if "locus_id" not in frame.columns:
            raise QueryCheckpointError("A non-empty query frame lacks locus_id.")
        values = frame["locus_id"].tolist()
    if any(not isinstance(value, str) or not value for value in values):
        raise QueryCheckpointError("Checkpoint locus identities must be non-empty strings.")
    if len(values) != len(set(values)):
        raise QueryCheckpointError("Checkpoint locus identities must be unique.")
    return values


def _identifier_hash(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _query_evidence_payload(query_result: NightQueryResult) -> Dict[str, Any]:
    if not isinstance(query_result, NightQueryResult):
        raise QueryCheckpointError("Checkpoint sealing requires a NightQueryResult.")
    if not query_result.clean or query_result.loci is None:
        raise QueryCheckpointError("Only a clean completed query result can be sealed.")
    evidence = query_result.evidence.as_dict()
    try:
        return json.loads(_canonical_json_bytes(evidence))
    except ValueError as exc:
        raise QueryCheckpointError("Query evidence is not canonical JSON.") from exc


def _validate_result_and_bindings(
    query_result: NightQueryResult,
    bindings: QueryResultCheckpointBindings,
) -> Tuple[pd.DataFrame, Dict[str, Any], Sequence[str], Sequence[str], Sequence[str]]:
    evidence = _query_evidence_payload(query_result)
    frame = query_result.loci
    assert frame is not None
    columns, dtypes = _frame_contract(frame)
    locus_ids = _locus_ids(frame)
    if (
        query_result.request.date_utc != bindings.target_date_utc
        or query_result.provider_name != bindings.provider_name
        or query_result.scenario != bindings.provider_scenario
    ):
        raise QueryCheckpointError("Query result disagrees with checkpoint bindings.")
    expected_outcome = (
        ProviderOutcome.SUCCESS_ZERO.value if frame.empty else ProviderOutcome.SUCCESS.value
    )
    details = evidence.get("details")
    policy = json.loads(_canonical_json_bytes(dict(bindings.query_policy)))
    if (
        query_result.outcome.value != expected_outcome
        or evidence.get("completed") is not True
        or evidence.get("partial") is not False
        or evidence.get("errors") != []
        or evidence.get("returned_loci") != len(frame)
        or not isinstance(details, dict)
        or details.get("target_date_utc") != bindings.target_date_utc
        or details.get("returned_loci") != len(frame)
        or details.get("locus_order_sha256") != _identifier_hash(locus_ids)
        or details.get("query_contract_sha256")
        != _canonical_json_sha256(policy["scientific_contract"])
        or details.get("execution_policy") != policy["execution_policy"]
    ):
        raise QueryCheckpointError(
            "Query completion evidence does not bind the complete ordered frame."
        )
    return frame, evidence, columns, dtypes, locus_ids


def _record_line(ordinal: int, values: Sequence[Any]) -> bytes:
    return _canonical_json_bytes([ordinal, [_encode_value(value) for value in values]])


def _semantic_header(columns: Sequence[str], dtypes: Sequence[str]) -> bytes:
    return _canonical_json_bytes(
        {"codec": RECORD_CODEC_VERSION, "columns": list(columns), "dtypes": list(dtypes)}
    )


def _content_integrity_payload(
    *,
    bindings_sha256: str,
    request_binding_sha256: str,
    query_evidence_sha256: str,
    semantic_sha256: str,
    records_sha256: str,
    shards: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "bindings_sha256": bindings_sha256,
        "request_binding_sha256": request_binding_sha256,
        "query_evidence_sha256": query_evidence_sha256,
        "semantic_sha256": semantic_sha256,
        "records_sha256": records_sha256,
        "shards": list(shards),
    }


def seal_query_result_checkpoint(
    run_root: Path,
    query_result: NightQueryResult,
    bindings: QueryResultCheckpointBindings,
    *,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    rows_per_shard: int = DEFAULT_ROWS_PER_SHARD,
) -> Path:
    """Durably seal one complete ordered query result beneath its run root.

    No existing checkpoint is overwritten.  Data and manifest are created in a
    new private final directory and ``COMMITTED.json`` is atomically created
    last.  Exclusive directory creation prevents replacement of prior evidence.
    A crash before the last step leaves evidence which the loader always refuses.
    """
    if not isinstance(bindings, QueryResultCheckpointBindings):
        raise TypeError("bindings must be QueryResultCheckpointBindings.")
    name = _safe_component(checkpoint_name, "checkpoint_name")
    if isinstance(rows_per_shard, bool) or not isinstance(rows_per_shard, int):
        raise ValueError("rows_per_shard must be a positive integer.")
    if rows_per_shard <= 0 or rows_per_shard > 100_000:
        raise ValueError("rows_per_shard must be between 1 and 100000.")
    canonical_root = _canonical_run_root(run_root, bindings.run_id)
    frame, evidence, columns, dtypes, locus_ids = _validate_result_and_bindings(
        query_result, bindings
    )
    binding_payload = _binding_payload(bindings)
    binding_bytes = _canonical_json_bytes(binding_payload)
    bindings_sha256 = _sha256(binding_bytes)
    request_payload = _request_binding(query_result.request)
    request_binding_sha256 = _sha256(_canonical_json_bytes(request_payload))
    evidence_document = {
        "provider": query_result.provider_name,
        "outcome": query_result.outcome.value,
        "evidence": evidence,
    }
    evidence_bytes = _canonical_json_bytes(evidence_document)
    evidence_sha256 = _sha256(evidence_bytes)

    run_fd = os.open(str(canonical_root), _DIRECTORY_FLAGS)
    parent_fd: Optional[int] = None
    temporary_fd: Optional[int] = None
    final_fd: Optional[int] = None
    try:
        parent_fd = _ensure_parent_directory(run_fd)
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise QueryCheckpointError("The query checkpoint already exists.") from exc
        except OSError as exc:
            raise QueryCheckpointError("Could not create the checkpoint transaction.") from exc
        os.fsync(parent_fd)
        temporary_fd = _open_directory_at(
            parent_fd, name, "Checkpoint transaction"
        )
        if stat.S_IMODE(os.fstat(temporary_fd).st_mode) != 0o700:
            raise QueryCheckpointError("Checkpoint transaction mode must be 0700.")

        shards = []
        records_digest = hashlib.sha256()
        semantic_digest = hashlib.sha256()
        semantic_digest.update(_semantic_header(columns, dtypes))
        row_iterator = iter(frame.itertuples(index=False, name=None))
        for shard_index, start in enumerate(range(0, len(frame), rows_per_shard)):
            end = min(start + rows_per_shard, len(frame))
            shard_name = f"records-{shard_index:06d}.jsonl"
            try:
                descriptor = os.open(
                    shard_name, _WRITE_FLAGS, 0o600, dir_fd=temporary_fd
                )
            except OSError as exc:
                raise QueryCheckpointError(
                    f"Could not create checkpoint shard {shard_name}."
                ) from exc
            shard_digest = hashlib.sha256()
            shard_bytes = 0
            try:
                os.fchmod(descriptor, 0o600)
                for ordinal, values in enumerate(
                    islice(row_iterator, end - start), start=start
                ):
                    line = _record_line(ordinal, values)
                    # Prove the codec itself preserves every value before commit.
                    encoded = json.loads(line)
                    decoded = [_decode_checked(value) for value in encoded[1]]
                    if [_encode_value(value) for value in decoded] != encoded[1]:
                        raise QueryCheckpointError(
                            "Checkpoint value codec failed semantic round-trip."
                        )
                    _write_all(descriptor, line)
                    shard_digest.update(line)
                    records_digest.update(line)
                    semantic_digest.update(line)
                    shard_bytes += len(line)
                os.fsync(descriptor)
            except OSError as exc:
                raise QueryCheckpointError(
                    f"Could not durably write checkpoint shard {shard_name}."
                ) from exc
            finally:
                os.close(descriptor)
            shards.append(
                {
                    "path": shard_name,
                    "first_ordinal": start,
                    "last_ordinal_exclusive": end,
                    "rows": end - start,
                    "bytes": shard_bytes,
                    "sha256": shard_digest.hexdigest(),
                }
            )
        if not shards:
            shard_name = "records-000000.jsonl"
            _write_new_file_at(temporary_fd, shard_name, b"")
            shards.append(
                {
                    "path": shard_name,
                    "first_ordinal": 0,
                    "last_ordinal_exclusive": 0,
                    "rows": 0,
                    "bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                }
            )

        _write_new_file_at(temporary_fd, QUERY_EVIDENCE_NAME, evidence_bytes)
        records_sha256 = records_digest.hexdigest()
        semantic_sha256 = semantic_digest.hexdigest()
        content_payload = _content_integrity_payload(
            bindings_sha256=bindings_sha256,
            request_binding_sha256=request_binding_sha256,
            query_evidence_sha256=evidence_sha256,
            semantic_sha256=semantic_sha256,
            records_sha256=records_sha256,
            shards=shards,
        )
        content_integrity_sha256 = _sha256(_canonical_json_bytes(content_payload))
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": "phase6-query-result",
            "classification": NON_AUTHORITATIVE_CLASSIFICATION,
            "authoritative": False,
            "publication_eligible": False,
            "run_local": True,
            "bindings": binding_payload,
            "bindings_sha256": bindings_sha256,
            "scientific_request": request_payload,
            "scientific_request_sha256": request_binding_sha256,
            "provider_outcome": query_result.outcome.value,
            "query_evidence": {
                "path": QUERY_EVIDENCE_NAME,
                "bytes": len(evidence_bytes),
                "sha256": evidence_sha256,
            },
            "frame": {
                "codec": RECORD_CODEC_VERSION,
                "arrow_used": False,
                "nested_null_semantics": (
                    "canonical-mapping-distinguishes-missing-key-from-present-null"
                ),
                "row_count": len(frame),
                "columns": list(columns),
                "dtypes": list(dtypes),
                "locus_id_column": "locus_id",
                "locus_order_sha256": _identifier_hash(locus_ids),
                "records_sha256": records_sha256,
                "semantic_sha256": semantic_sha256,
                "shards": shards,
            },
            "semantic_round_trip": {
                "verified_before_commit": True,
                "column_order_preserved": True,
                "row_order_preserved_by_ordinal": True,
                "nested_missing_vs_null_preserved": True,
            },
            "content_integrity_sha256": content_integrity_sha256,
            "commit_protocol": (
                "exclusive-directory-files-fsync-manifest-fsync-commit-marker-last"
            ),
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        _write_new_file_at(temporary_fd, MANIFEST_NAME, manifest_bytes)
        os.fsync(temporary_fd)
        final_fd = temporary_fd
        temporary_fd = None
        commit = {
            "schema_version": COMMIT_SCHEMA_VERSION,
            "committed": True,
            "classification": NON_AUTHORITATIVE_CLASSIFICATION,
            "authoritative": False,
            "manifest_path": MANIFEST_NAME,
            "manifest_sha256": _sha256(manifest_bytes),
            "query_evidence_sha256": evidence_sha256,
            "content_integrity_sha256": content_integrity_sha256,
        }
        commit_bytes = _canonical_json_bytes(commit)
        marker_temporary = f".{COMMIT_MARKER_NAME}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        _write_new_file_at(final_fd, marker_temporary, commit_bytes)
        try:
            os.rename(
                marker_temporary,
                COMMIT_MARKER_NAME,
                src_dir_fd=final_fd,
                dst_dir_fd=final_fd,
            )
        except OSError as exc:
            raise QueryCheckpointError("Could not atomically commit the checkpoint.") from exc
        os.fsync(final_fd)
    finally:
        if final_fd is not None:
            os.close(final_fd)
        if temporary_fd is not None:
            os.close(temporary_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        os.close(run_fd)

    checkpoint_path = canonical_root / CHECKPOINT_PARENT_NAME / name
    # Freshly reopen from durable bytes before returning success.
    reopened = load_query_result_checkpoint(
        canonical_root,
        query_result.request,
        bindings,
        checkpoint_name=name,
    )
    if reopened.manifest.get("content_integrity_sha256") != content_integrity_sha256:
        raise QueryCheckpointError("Committed checkpoint failed final integrity reproof.")
    return checkpoint_path


def _validate_binding_manifest(
    manifest: Mapping[str, Any], expected: QueryResultCheckpointBindings
) -> Tuple[Mapping[str, Any], str]:
    expected_payload = _binding_payload(expected)
    expected_bytes = _canonical_json_bytes(expected_payload)
    expected_sha = _sha256(expected_bytes)
    observed = manifest.get("bindings")
    if observed != expected_payload or manifest.get("bindings_sha256") != expected_sha:
        raise QueryCheckpointError("Checkpoint stable bindings do not match expectations.")
    return observed, expected_sha


def _construct_frame(
    *,
    directory_fd: int,
    frame_manifest: Mapping[str, Any],
) -> Tuple[pd.DataFrame, str, str]:
    columns = frame_manifest.get("columns")
    dtypes = frame_manifest.get("dtypes")
    row_count = frame_manifest.get("row_count")
    shards = frame_manifest.get("shards")
    if (
        frame_manifest.get("codec") != RECORD_CODEC_VERSION
        or frame_manifest.get("arrow_used") is not False
        or frame_manifest.get("nested_null_semantics")
        != "canonical-mapping-distinguishes-missing-key-from-present-null"
        or not isinstance(columns, list)
        or any(not isinstance(value, str) or not value for value in columns)
        or len(columns) != len(set(columns))
        or not isinstance(dtypes, list)
        or len(dtypes) != len(columns)
        or any(not isinstance(value, str) or not value for value in dtypes)
        or isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count < 0
        or not isinstance(shards, list)
        or not shards
    ):
        raise QueryCheckpointError("Checkpoint frame manifest is invalid.")

    column_values = [[] for _column in columns]
    expected_ordinal = 0
    records_digest = hashlib.sha256()
    semantic_digest = hashlib.sha256()
    semantic_digest.update(_semantic_header(columns, dtypes))
    for shard_index, shard in enumerate(shards):
        expected_name = f"records-{shard_index:06d}.jsonl"
        if not isinstance(shard, dict) or shard.get("path") != expected_name:
            raise QueryCheckpointError("Checkpoint shard ordering is invalid.")
        initial = _entry_snapshot(directory_fd, expected_name)
        try:
            descriptor = os.open(expected_name, _READ_FLAGS, dir_fd=directory_fd)
        except OSError as exc:
            raise QueryCheckpointError(f"Checkpoint shard {expected_name} is unsafe.") from exc
        shard_digest = hashlib.sha256()
        shard_bytes = 0
        shard_rows = 0
        try:
            pinned = os.fstat(descriptor)
            if (pinned.st_dev, pinned.st_ino, pinned.st_size) != (
                initial.st_dev,
                initial.st_ino,
                initial.st_size,
            ):
                raise QueryCheckpointError(
                    f"Checkpoint shard {expected_name} changed before read."
                )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for line in handle:
                    if not line.endswith(b"\n"):
                        raise QueryCheckpointError(
                            f"Checkpoint shard {expected_name} has a partial record."
                        )
                    try:
                        value = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise QueryCheckpointError(
                            f"Checkpoint shard {expected_name} has invalid JSON."
                        ) from exc
                    try:
                        canonical_line = _canonical_json_bytes(value)
                    except ValueError as exc:
                        raise QueryCheckpointError(
                            f"Checkpoint shard {expected_name} is not canonical."
                        ) from exc
                    if canonical_line != line:
                        raise QueryCheckpointError(
                            f"Checkpoint shard {expected_name} is not canonical."
                        )
                    if (
                        not isinstance(value, list)
                        or len(value) != 2
                        or value[0] != expected_ordinal
                        or not isinstance(value[1], list)
                        or len(value[1]) != len(columns)
                    ):
                        raise QueryCheckpointError(
                            "Checkpoint row order or column cardinality is invalid."
                        )
                    decoded = [_decode_checked(item) for item in value[1]]
                    if [_encode_value(item) for item in decoded] != value[1]:
                        raise QueryCheckpointError(
                            "Checkpoint value failed semantic round-trip on reopen."
                        )
                    for index, item in enumerate(decoded):
                        column_values[index].append(item)
                    expected_ordinal += 1
                    shard_rows += 1
                    shard_bytes += len(line)
                    shard_digest.update(line)
                    records_digest.update(line)
                    semantic_digest.update(line)
        finally:
            os.close(descriptor)
        ending = _entry_snapshot(directory_fd, expected_name)
        if (ending.st_dev, ending.st_ino, ending.st_size) != (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
        ):
            raise QueryCheckpointError(
                f"Checkpoint shard {expected_name} changed during read."
            )
        if shard != {
            "path": expected_name,
            "first_ordinal": expected_ordinal - shard_rows,
            "last_ordinal_exclusive": expected_ordinal,
            "rows": shard_rows,
            "bytes": shard_bytes,
            "sha256": shard_digest.hexdigest(),
        }:
            raise QueryCheckpointError(f"Checkpoint shard {expected_name} identity differs.")
    if expected_ordinal != row_count:
        raise QueryCheckpointError("Checkpoint row count is incomplete.")
    if records_digest.hexdigest() != frame_manifest.get("records_sha256"):
        raise QueryCheckpointError("Checkpoint record integrity hash differs.")
    if semantic_digest.hexdigest() != frame_manifest.get("semantic_sha256"):
        raise QueryCheckpointError("Checkpoint semantic integrity hash differs.")

    data: Dict[str, pd.Series] = {}
    for column, dtype, values in zip(columns, dtypes, column_values):
        try:
            data[column] = pd.Series(values, dtype=dtype)
        except (TypeError, ValueError, OverflowError) as exc:
            raise QueryCheckpointError(
                f"Checkpoint column {column!r} cannot restore dtype {dtype!r}."
            ) from exc
    frame = pd.DataFrame(data, columns=columns).reset_index(drop=True)
    if [str(frame[column].dtype) for column in columns] != dtypes:
        raise QueryCheckpointError("Checkpoint dtypes changed during reconstruction.")
    return frame, records_digest.hexdigest(), semantic_digest.hexdigest()


def load_query_result_checkpoint(
    run_root: Path,
    request: NightScienceRequest,
    expected_bindings: QueryResultCheckpointBindings,
    *,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
) -> LoadedQueryResultCheckpoint:
    """Reopen a committed checkpoint and reconstruct ``NightQueryResult``.

    This function imports or calls no ANTARES search function.  The returned
    ``query_result`` is suitable for ``provider.fetch(request, query_result)``.
    """
    if not isinstance(expected_bindings, QueryResultCheckpointBindings):
        raise TypeError("expected_bindings must be QueryResultCheckpointBindings.")
    if not isinstance(request, NightScienceRequest):
        raise TypeError("request must be NightScienceRequest.")
    if request.date_utc != expected_bindings.target_date_utc:
        raise QueryCheckpointError("Resume request night does not match expectations.")
    name = _safe_component(checkpoint_name, "checkpoint_name")
    canonical_root = _canonical_run_root(run_root, expected_bindings.run_id)
    run_fd = os.open(str(canonical_root), _DIRECTORY_FLAGS)
    parent_fd: Optional[int] = None
    checkpoint_fd: Optional[int] = None
    try:
        parent_fd = _open_directory_at(
            run_fd, CHECKPOINT_PARENT_NAME, "Checkpoint parent"
        )
        if stat.S_IMODE(os.fstat(parent_fd).st_mode) != 0o700:
            raise QueryCheckpointError("Checkpoint parent mode must be exactly 0700.")
        checkpoint_fd = _open_directory_at(parent_fd, name, "Query checkpoint")
        if stat.S_IMODE(os.fstat(checkpoint_fd).st_mode) != 0o700:
            raise QueryCheckpointError("Query checkpoint mode must be exactly 0700.")

        try:
            marker_bytes = _read_regular_at(checkpoint_fd, COMMIT_MARKER_NAME)
        except QueryCheckpointError as exc:
            raise QueryCheckpointError(
                "Query checkpoint has no valid terminal commit marker."
            ) from exc
        marker = _parse_canonical_object(marker_bytes, "Checkpoint commit marker")
        if (
            marker.get("schema_version") != COMMIT_SCHEMA_VERSION
            or marker.get("committed") is not True
            or marker.get("classification") != NON_AUTHORITATIVE_CLASSIFICATION
            or marker.get("authoritative") is not False
            or marker.get("manifest_path") != MANIFEST_NAME
        ):
            raise QueryCheckpointError("Checkpoint commit marker is invalid.")

        manifest_bytes = _read_regular_at(checkpoint_fd, MANIFEST_NAME)
        if _sha256(manifest_bytes) != marker.get("manifest_sha256"):
            raise QueryCheckpointError("Checkpoint manifest integrity hash differs.")
        manifest = _parse_canonical_object(manifest_bytes, "Checkpoint manifest")
        if (
            manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("checkpoint_kind") != "phase6-query-result"
            or manifest.get("classification") != NON_AUTHORITATIVE_CLASSIFICATION
            or manifest.get("authoritative") is not False
            or manifest.get("publication_eligible") is not False
            or manifest.get("run_local") is not True
            or manifest.get("commit_protocol")
            != "exclusive-directory-files-fsync-manifest-fsync-commit-marker-last"
        ):
            raise QueryCheckpointError("Checkpoint authority or schema boundary is invalid.")
        _observed_bindings, bindings_sha256 = _validate_binding_manifest(
            manifest, expected_bindings
        )
        request_payload = _request_binding(request)
        request_binding_sha256 = _sha256(_canonical_json_bytes(request_payload))
        if (
            manifest.get("scientific_request") != request_payload
            or manifest.get("scientific_request_sha256")
            != request_binding_sha256
        ):
            raise QueryCheckpointError(
                "Resume scientific request does not match the sealed query request."
            )

        evidence_descriptor = manifest.get("query_evidence")
        if (
            not isinstance(evidence_descriptor, dict)
            or evidence_descriptor.get("path") != QUERY_EVIDENCE_NAME
        ):
            raise QueryCheckpointError("Checkpoint query-evidence descriptor is invalid.")
        evidence_bytes = _read_regular_at(checkpoint_fd, QUERY_EVIDENCE_NAME)
        evidence_sha256 = _sha256(evidence_bytes)
        if evidence_descriptor != {
            "path": QUERY_EVIDENCE_NAME,
            "bytes": len(evidence_bytes),
            "sha256": evidence_sha256,
        }:
            raise QueryCheckpointError("Checkpoint query-evidence identity differs.")
        if marker.get("query_evidence_sha256") != evidence_sha256:
            raise QueryCheckpointError("Commit marker query-evidence hash differs.")
        evidence_document = _parse_canonical_object(
            evidence_bytes, "Checkpoint query evidence"
        )
        stored_outcome = manifest.get("provider_outcome")
        if (
            set(evidence_document) != {"provider", "outcome", "evidence"}
            or evidence_document.get("provider") != expected_bindings.provider_name
            or evidence_document.get("outcome") != stored_outcome
            or not isinstance(evidence_document.get("evidence"), dict)
        ):
            raise QueryCheckpointError(
                "Checkpoint query-evidence document identity is invalid."
            )
        evidence = evidence_document["evidence"]

        frame_manifest = manifest.get("frame")
        if not isinstance(frame_manifest, dict):
            raise QueryCheckpointError("Checkpoint frame descriptor is missing.")
        frame, records_sha256, semantic_sha256 = _construct_frame(
            directory_fd=checkpoint_fd,
            frame_manifest=frame_manifest,
        )
        locus_ids = _locus_ids(frame)
        details = evidence.get("details")
        policy = json.loads(
            _canonical_json_bytes(dict(expected_bindings.query_policy))
        )
        expected_outcome = (
            ProviderOutcome.SUCCESS_ZERO.value if frame.empty else ProviderOutcome.SUCCESS.value
        )
        if (
            stored_outcome != expected_outcome
            or evidence.get("completed") is not True
            or evidence.get("partial") is not False
            or evidence.get("errors") != []
            or evidence.get("returned_loci") != len(frame)
            or not isinstance(details, dict)
            or details.get("target_date_utc") != expected_bindings.target_date_utc
            or details.get("returned_loci") != len(frame)
            or details.get("locus_order_sha256") != _identifier_hash(locus_ids)
            or details.get("query_contract_sha256")
            != _canonical_json_sha256(policy["scientific_contract"])
            or details.get("execution_policy") != policy["execution_policy"]
            or frame_manifest.get("row_count") != len(frame)
            or frame_manifest.get("locus_order_sha256") != _identifier_hash(locus_ids)
            or frame_manifest.get("records_sha256") != records_sha256
            or frame_manifest.get("semantic_sha256") != semantic_sha256
        ):
            raise QueryCheckpointError(
                "Checkpoint evidence does not prove the reopened ordered query result."
            )
        semantic_proof = manifest.get("semantic_round_trip")
        if semantic_proof != {
            "verified_before_commit": True,
            "column_order_preserved": True,
            "row_order_preserved_by_ordinal": True,
            "nested_missing_vs_null_preserved": True,
        }:
            raise QueryCheckpointError("Checkpoint semantic round-trip proof is invalid.")
        shards = frame_manifest.get("shards")
        content_payload = _content_integrity_payload(
            bindings_sha256=bindings_sha256,
            request_binding_sha256=request_binding_sha256,
            query_evidence_sha256=evidence_sha256,
            semantic_sha256=semantic_sha256,
            records_sha256=records_sha256,
            shards=shards,
        )
        integrity_sha256 = _sha256(_canonical_json_bytes(content_payload))
        if (
            manifest.get("content_integrity_sha256") != integrity_sha256
            or marker.get("content_integrity_sha256") != integrity_sha256
        ):
            raise QueryCheckpointError("Checkpoint aggregate integrity hash differs.")

        expected_entries = {
            COMMIT_MARKER_NAME,
            MANIFEST_NAME,
            QUERY_EVIDENCE_NAME,
            *(shard["path"] for shard in shards),
        }
        observed_entries = set(os.listdir(checkpoint_fd))
        if observed_entries != expected_entries:
            raise QueryCheckpointError("Checkpoint contains unexpected or missing entries.")

        stage_evidence = QueryStageEvidence(
            completed=True,
            partial=False,
            returned_loci=len(frame),
            errors=(),
            details=details,
        )
        query_result = NightQueryResult(
            request=request,
            provider_name=expected_bindings.provider_name,
            scenario=expected_bindings.provider_scenario,
            outcome=ProviderOutcome(stored_outcome),
            loci=frame,
            evidence=stage_evidence,
        )
        if not query_result.clean:
            raise QueryCheckpointError("Reconstructed query result is not clean.")
    finally:
        if checkpoint_fd is not None:
            os.close(checkpoint_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        os.close(run_fd)

    return LoadedQueryResultCheckpoint(
        path=canonical_root / CHECKPOINT_PARENT_NAME / name,
        query_result=query_result,
        manifest=manifest,
        query_evidence=evidence,
        query_evidence_document=evidence_document,
        query_evidence_sha256=evidence_sha256,
        integrity_sha256=integrity_sha256,
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "COMMIT_MARKER_NAME",
    "LoadedQueryResultCheckpoint",
    "NON_AUTHORITATIVE_CLASSIFICATION",
    "QueryCheckpointError",
    "QueryResultCheckpointBindings",
    "load_query_result_checkpoint",
    "seal_query_result_checkpoint",
]
