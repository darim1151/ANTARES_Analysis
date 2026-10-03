"""Stable machine- and human-readable operation reports.

The report model is intentionally standard-library-only so planning and
refusal paths remain available before the scientific Python stack is loaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


REPORT_SCHEMA_VERSION = "1.0"


class ExitCode(IntEnum):
    """Small, documented exit-code classes for automation."""

    SUCCESS = 0
    VALIDATION_FAILURE = 1
    INVALID_REQUEST = 2
    OPERATIONAL_FAILURE = 3
    REFUSED = 4
    INTERNAL_ERROR = 5


@dataclass(frozen=True)
class Issue:
    """One warning, error, or refusal with a stable code."""

    code: str
    message: str
    detail: Optional[str] = None

    def as_dict(self) -> dict[str, Optional[str]]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Artifact:
    """A durable or operational artifact referenced by an operation."""

    name: str
    kind: str
    path: str
    state: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    """One validation/preflight observation."""

    code: str
    status: str
    summary: str
    details: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status,
            "summary": self.summary,
            "details": _json_value(self.details),
        }


def _json_value(value: Any) -> Any:
    """Convert supported values without falling back to object repr strings."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(
        "Operation reports cannot serialize values of type "
        f"{type(value).__name__}."
    )


@dataclass(frozen=True)
class OperationReport:
    """Versioned result shared by Python callers and the CLI."""

    operation: str
    success: bool
    status: str
    started_at_utc: str
    finished_at_utc: str
    elapsed_seconds: float
    exit_code: ExitCode = ExitCode.SUCCESS
    run_id: Optional[str] = None
    warnings: Tuple[Issue, ...] = ()
    errors: Tuple[Issue, ...] = ()
    refusal_reasons: Tuple[Issue, ...] = ()
    counts: Mapping[str, Optional[int]] = field(default_factory=dict)
    artifacts: Tuple[Artifact, ...] = ()
    evidence: Tuple[Evidence, ...] = ()
    next_actions: Tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = REPORT_SCHEMA_VERSION
    read_only: bool = True

    def __post_init__(self) -> None:
        if self.success and self.exit_code != ExitCode.SUCCESS:
            raise ValueError("A successful report must use the success exit code.")
        if not self.success and self.exit_code == ExitCode.SUCCESS:
            raise ValueError("A failed report must use a non-success exit code.")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative.")

    def as_dict(self) -> dict[str, Any]:
        """Return a fixed-field deterministic JSON-compatible mapping."""
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "success": self.success,
            "status": self.status,
            "read_only": self.read_only,
            "run_id": self.run_id,
            "timing": {
                "started_at_utc": self.started_at_utc,
                "finished_at_utc": self.finished_at_utc,
                "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            },
            "exit_code": int(self.exit_code),
            "warnings": [item.as_dict() for item in self.warnings],
            "errors": [item.as_dict() for item in self.errors],
            "refusal_reasons": [item.as_dict() for item in self.refusal_reasons],
            "counts": _json_value(self.counts),
            "artifacts": [item.as_dict() for item in self.artifacts],
            "evidence": [item.as_dict() for item in self.evidence],
            "next_actions": list(self.next_actions),
            "details": _json_value(self.details),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize with sorted keys and no implementation-specific reprs."""
        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    def render_human(self) -> str:
        """Render a concise diagnostic view without hiding refusals."""
        lines = [
            f"Operation: {self.operation}",
            f"Status:    {self.status}",
            f"Result:    {'PASS' if self.success else 'FAIL'} (read-only={self.read_only})",
        ]
        for label, items in (
            ("Warning", self.warnings),
            ("Error", self.errors),
            ("Refusal", self.refusal_reasons),
        ):
            for item in items:
                lines.append(f"{label}:  {item.code}: {item.message}")
        if self.counts:
            rendered = ", ".join(
                f"{key}={self.counts[key]}" for key in sorted(self.counts)
            )
            lines.append(f"Counts:    {rendered}")
        if self.next_actions:
            lines.append("Next:      " + "; ".join(self.next_actions))
        return "\n".join(lines)


def issues(values: Sequence[tuple[str, str]]) -> Tuple[Issue, ...]:
    """Create an immutable issue tuple from code/message pairs."""
    return tuple(Issue(code, message) for code, message in values)
