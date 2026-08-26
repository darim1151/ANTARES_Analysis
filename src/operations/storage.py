"""Storage boundaries shared by planning and guarded local transactions."""

from __future__ import annotations

import json
import os
import socket
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from src.cli_profiles import MIDDLE_EARTH_CANARY_ROOT

from .context import OperationContext


APPENDABLE_STATUSES = frozenset({"complete", "under_target", "saturated_unresolved"})
ACCEPTED_ZERO_ROW_NIGHTS = frozenset({"2026-03-05", "2026-03-11"})
OPERATIONS_DIRECTORY = ".antares-operations"
_DEVELOPMENT_CAPABILITY_TOKEN = object()
_SYNTHETIC_CAPABILITY_TOKEN = object()
ARNOR_CANARY_ROOT = MIDDLE_EARTH_CANARY_ROOT


class StorageContractError(ValueError):
    pass


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _relative_to(path: Path, root: Path) -> Optional[Path]:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def validate_root_separation(data_root: Path, cache_root: Path) -> Tuple[Path, Path]:
    """Reject equivalent, nested, or symlink-equivalent durable/cache roots."""
    durable = _resolved(data_root)
    cache = _resolved(cache_root)
    if durable == cache:
        raise StorageContractError("Data and cache roots resolve to the same path.")
    if _relative_to(cache, durable) is not None:
        raise StorageContractError("Cache root must not be inside durable data.")
    if _relative_to(durable, cache) is not None:
        raise StorageContractError("Durable data must not be inside the cache root.")
    return durable, cache


def contained_path(root: Path, relative: Path) -> Path:
    """Resolve a root-relative path and reject traversal or symlink escape."""
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or relative == Path("."):
        raise StorageContractError("Storage targets must be non-empty relative paths.")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise StorageContractError(f"Unsafe storage target: {relative}.")

    root_path = Path(root).expanduser()
    resolved_root = root_path.resolve(strict=False)
    cursor = root_path
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            resolved_link = cursor.resolve(strict=False)
            if _relative_to(resolved_link, resolved_root) is None:
                raise StorageContractError(
                    f"Storage target crosses a symlink outside the root: {relative}."
                )
    candidate = root_path / relative
    resolved_candidate = candidate.resolve(strict=False)
    if _relative_to(resolved_candidate, resolved_root) is None:
        raise StorageContractError(
            f"Storage target escapes configured root: {relative}."
        )
    return resolved_candidate


@dataclass(frozen=True)
class NightLocation:
    date_utc: str
    relative_directory: Path
    directory: Path
    loci: Path
    alerts: Path
    manifest: Path


@dataclass(frozen=True)
class NightInspection:
    state: str
    reason: str
    manifest_present: bool
    loci_present: bool
    alerts_present: bool
    append_ready: Optional[bool]
    status: Optional[str]
    actual_loci: Optional[int]
    alert_rows: Optional[int]
    zero_row: bool
    zero_row_evidence_valid: Optional[bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "reason": self.reason,
            "manifest_present": self.manifest_present,
            "loci_present": self.loci_present,
            "alerts_present": self.alerts_present,
            "append_ready": self.append_ready,
            "status": self.status,
            "actual_loci": self.actual_loci,
            "alert_rows": self.alert_rows,
            "zero_row": self.zero_row,
            "zero_row_evidence_valid": self.zero_row_evidence_valid,
        }


def _meaningful_error(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def recorded_query_fetch_errors(value: object, prefix: str = "") -> dict[str, object]:
    """Find explicit query/fetch/lightcurve error evidence recursively."""
    findings: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            error_field = lowered == "error" or (
                "error" in lowered
                and any(token in lowered for token in ("query", "fetch", "lightcurve"))
            )
            if error_field and _meaningful_error(child):
                findings[path] = child
            if isinstance(child, (Mapping, list, tuple)):
                findings.update(recorded_query_fetch_errors(child, path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.update(recorded_query_fetch_errors(child, f"{prefix}[{index}]"))
    return findings


def valid_zero_row_evidence(manifest: Mapping[str, Any]) -> bool:
    """Require completed, clean query evidence for an accepted empty night."""
    date_utc = manifest.get("date_utc")
    validation = manifest.get("validation")
    validation = validation if isinstance(validation, Mapping) else {}
    chunk_count = manifest.get("chunk_count")
    return bool(
        date_utc in ACCEPTED_ZERO_ROW_NIGHTS
        and manifest.get("status") == "complete"
        and manifest.get("actual_loci") == 0
        and manifest.get("alert_rows") == 0
        and type(chunk_count) is int
        and chunk_count > 0
        and manifest.get("finished_at_utc")
        and validation.get("append_ready") is True
        and validation.get("query_completed_pass") is True
        and validation.get("query_fetch_clean") is True
        and validation.get("zero_row_schema_pass") is True
        and not recorded_query_fetch_errors(manifest)
    )


class StorageLayout:
    """Configured durable/cache layout with contained target constructors."""

    def __init__(self, data_root: Path, cache_root: Path) -> None:
        self.data_root, self.cache_root = validate_root_separation(
            data_root, cache_root
        )

    @classmethod
    def from_context(cls, context: OperationContext) -> "StorageLayout":
        return cls(context.data_root, context.cache_root)

    def night(self, date_utc: str) -> NightLocation:
        year, month, day = date_utc.split("-")
        relative = Path("data") / "lsst_only" / "nightly" / year / month / day
        directory = contained_path(self.data_root, relative)
        return NightLocation(
            date_utc=date_utc,
            relative_directory=relative,
            directory=directory,
            loci=contained_path(self.data_root, relative / "loci.parquet"),
            alerts=contained_path(self.data_root, relative / "alerts.parquet"),
            manifest=contained_path(self.data_root, relative / "manifest.json"),
        )

    def lock_resource(self, target: NightLocation) -> Path:
        """Return the legacy temporary-fixture lock path, never production config."""
        relative = (
            Path(OPERATIONS_DIRECTORY)
            / "locks"
            / ("night-" + target.date_utc + ".lock")
        )
        return contained_path(self.data_root, relative)

    def staging_parent(self) -> Path:
        """Return the legacy temporary-fixture stage path, never production config."""
        return contained_path(
            self.data_root, Path(OPERATIONS_DIRECTORY) / "staging"
        )

    def manifest_science_paths(self, manifest_path: Path) -> tuple[Path, Path]:
        """Resolve current science only from manifest siblings."""
        relative_manifest = _relative_to(_resolved(manifest_path), self.data_root)
        if relative_manifest is None:
            raise StorageContractError("Manifest is outside the configured data root.")
        if relative_manifest.name != "manifest.json":
            raise StorageContractError("Expected a nightly manifest.json path.")
        parent = relative_manifest.parent
        return (
            contained_path(self.data_root, parent / "loci.parquet"),
            contained_path(self.data_root, parent / "alerts.parquet"),
        )

    def inspect_night(self, target: NightLocation) -> NightInspection:
        if target.directory.is_symlink():
            return NightInspection(
                "conflicting", "partition_directory_is_symlink", False, False,
                False, None, None, None, None, False, None,
            )
        if target.directory.exists() and not target.directory.is_dir():
            return NightInspection(
                "conflicting", "partition_path_is_not_directory", False, False,
                False, None, None, None, None, False, None,
            )
        if not target.directory.exists():
            return NightInspection(
                "missing", "partition_absent", False, False, False, None, None,
                None, None, False, None,
            )
        if target.manifest.is_symlink() or target.loci.is_symlink() or target.alerts.is_symlink():
            return NightInspection(
                "conflicting", "partition_contains_symlink", target.manifest.exists(),
                target.loci.exists(), target.alerts.exists(), None, None, None,
                None, False, None,
            )

        present = {
            "manifest": target.manifest.is_file(),
            "loci": target.loci.is_file(),
            "alerts": target.alerts.is_file(),
        }
        if not present["manifest"]:
            return NightInspection(
                "incomplete", "manifest_missing", False, present["loci"],
                present["alerts"], None, None, None, None, False, None,
            )
        try:
            manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return NightInspection(
                "conflicting", "manifest_unreadable", True, present["loci"],
                present["alerts"], None, None, None, None, False, None,
            )
        if not isinstance(manifest, dict) or manifest.get("date_utc") != target.date_utc:
            return NightInspection(
                "conflicting", "manifest_date_mismatch", True, present["loci"],
                present["alerts"], None, None, None, None, False, None,
            )
        validation = manifest.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        append_ready = validation.get("append_ready")
        append_ready = append_ready if isinstance(append_ready, bool) else None
        status = manifest.get("status")
        status = status if isinstance(status, str) else None
        actual_loci = manifest.get("actual_loci")
        actual_loci = actual_loci if type(actual_loci) is int else None
        alert_rows = manifest.get("alert_rows")
        alert_rows = alert_rows if type(alert_rows) is int else None
        zero_row = actual_loci == 0 and alert_rows == 0
        zero_valid = valid_zero_row_evidence(manifest) if zero_row else None
        complete = bool(
            all(present.values())
            and status in APPENDABLE_STATUSES
            and append_ready is True
            and actual_loci is not None
            and actual_loci >= 0
            and alert_rows is not None
            and alert_rows >= 0
            and (not zero_row or zero_valid)
        )
        return NightInspection(
            "complete" if complete else "incomplete",
            "append_ready_partition" if complete else "partition_not_append_ready",
            True,
            present["loci"],
            present["alerts"],
            append_ready,
            status,
            actual_loci,
            alert_rows,
            zero_row,
            zero_valid,
        )


@dataclass(frozen=True)
class DevelopmentWriteCapability:
    """Unforgeable-by-accident guard limiting writes to local temp roots."""

    root: Path
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _DEVELOPMENT_CAPABILITY_TOKEN:
            raise StorageContractError(
                "Development write capabilities must be issued by the "
                "temporary-root factory."
            )

    @property
    def published_root(self) -> Path:
        """Return the legacy fixture publication root."""
        return self.root

    @property
    def staging_root(self) -> Path:
        return contained_path(
            self.root, Path(OPERATIONS_DIRECTORY) / "staging"
        )

    @property
    def lock_root(self) -> Path:
        return contained_path(self.root, Path(OPERATIONS_DIRECTORY) / "locks")

    @property
    def journal_root(self) -> Path:
        return contained_path(
            self.root, Path(OPERATIONS_DIRECTORY) / "journals"
        )

    @property
    def evidence_root(self) -> Path:
        return contained_path(
            self.root, Path(OPERATIONS_DIRECTORY) / "evidence"
        )

    @classmethod
    def for_temporary_root(cls, root: Path) -> "DevelopmentWriteCapability":
        resolved = _resolved(root)
        temporary = _resolved(Path(tempfile.gettempdir()))
        if resolved == temporary or _relative_to(resolved, temporary) is None:
            raise StorageContractError(
                "Development write capability is restricted to a child of the "
                "local temporary directory."
            )
        if not resolved.is_dir() or resolved.is_symlink():
            raise StorageContractError(
                "Development write capability requires an existing real directory."
            )
        return cls(resolved, _DEVELOPMENT_CAPABILITY_TOKEN)


def _validated_run_id(value: object) -> str:
    run_id = str(value).strip()
    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or len(Path(run_id).parts) != 1
    ):
        raise StorageContractError(
            "Synthetic write capability requires one safe run-id path component."
        )
    return run_id


def _existing_real_directory(root: Path, *, label: str) -> Path:
    lexical = Path(root).expanduser()
    if lexical.is_symlink() or not lexical.is_dir():
        raise StorageContractError(f"{label} requires an existing real directory.")
    return lexical.resolve(strict=True)


@dataclass(frozen=True)
class SyntheticWriteCapability:
    """Sealed authority for one synthetic run; never production science.

    The capability only describes already-created, run-scoped roots. Factories
    perform no filesystem mutation. There is deliberately no production-data
    factory.
    """

    root: Path
    run_id: str
    environment: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _SYNTHETIC_CAPABILITY_TOKEN:
            raise StorageContractError(
                "Synthetic write capabilities must be issued by a sealed "
                "synthetic-run factory."
            )
        if self.environment not in {"local-temporary", "arnor-canary"}:
            raise StorageContractError("Unknown synthetic capability environment.")
        _validated_run_id(self.run_id)

    @property
    def published_root(self) -> Path:
        return contained_path(self.root, Path("published"))

    @property
    def staging_root(self) -> Path:
        return contained_path(self.root, Path("staging"))

    @property
    def lock_root(self) -> Path:
        return contained_path(self.root, Path("control") / "locks")

    @property
    def journal_root(self) -> Path:
        return contained_path(self.root, Path("control") / "journals")

    @property
    def evidence_root(self) -> Path:
        return contained_path(self.root, Path("evidence"))

    @classmethod
    def for_local_run_root(
        cls,
        root: Path,
        run_id: Optional[str] = None,
    ) -> "SyntheticWriteCapability":
        """Issue authority for one existing run root below the local temp root."""
        resolved = _existing_real_directory(root, label="Local synthetic run root")
        temporary = _resolved(Path(tempfile.gettempdir()))
        if resolved == temporary or _relative_to(resolved, temporary) is None:
            raise StorageContractError(
                "Local synthetic write capability is restricted to a child of "
                "the local temporary directory."
            )
        identity = _validated_run_id(run_id if run_id is not None else resolved.name)
        if run_id is not None and resolved.name != identity:
            raise StorageContractError(
                "Local synthetic run root must be named for its run id."
            )
        return cls(
            resolved,
            identity,
            "local-temporary",
            _SYNTHETIC_CAPABILITY_TOKEN,
        )

    @classmethod
    def for_temporary_run_root(
        cls,
        root: Path,
        run_id: Optional[str] = None,
    ) -> "SyntheticWriteCapability":
        """Compatibility alias for :meth:`for_local_run_root`."""
        return cls.for_local_run_root(root, run_id)

    @classmethod
    def for_arnor_canary_root(
        cls,
        root: Path,
        run_id: Optional[str] = None,
        *,
        hostname: Optional[str] = None,
    ) -> "SyntheticWriteCapability":
        """Issue authority for one exact existing Arnor canary run directory.

        ``hostname`` exists solely so tests can prove the host gate without
        pretending to run on Arnor.
        """
        observed_hostname = hostname or socket.gethostname()
        if observed_hostname.strip().lower().split(".", 1)[0] != "arnor":
            raise StorageContractError(
                "Arnor canary capability may only be issued on host arnor."
            )

        proposed = Path(root).expanduser()
        identity = _validated_run_id(
            run_id if run_id is not None else proposed.name
        )
        expected = ARNOR_CANARY_ROOT / identity
        lexical = Path(os.path.abspath(os.fspath(proposed)))
        if lexical != expected or lexical.parent != ARNOR_CANARY_ROOT:
            raise StorageContractError(
                "Arnor canary capability requires the exact direct child "
                f"{expected}."
            )
        resolved = _existing_real_directory(
            lexical, label="Arnor canary run root"
        )
        try:
            canonical_canary = ARNOR_CANARY_ROOT.resolve(strict=True)
        except OSError as exc:
            raise StorageContractError(
                "Canonical Arnor canary parent is missing or inaccessible."
            ) from exc
        if canonical_canary != ARNOR_CANARY_ROOT or resolved != expected:
            raise StorageContractError(
                "Arnor canary root must not use symlink or path aliases."
            )
        return cls(
            resolved,
            identity,
            "arnor-canary",
            _SYNTHETIC_CAPABILITY_TOKEN,
        )
