"""Explicit immutable dependencies for one operation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol, Tuple

from src.cli_profiles import StorageProfile, resolve_profile


class OperationLogger(Protocol):
    """Minimal logging hook accepted by operations."""

    def info(self, message: str, **fields: object) -> None:
        ...

    def warning(self, message: str, **fields: object) -> None:
        ...

    def error(self, message: str, **fields: object) -> None:
        ...


class NullLogger:
    """Default logger that intentionally performs no I/O."""

    def info(self, message: str, **fields: object) -> None:
        del message, fields

    def warning(self, message: str, **fields: object) -> None:
        del message, fields

    def error(self, message: str, **fields: object) -> None:
        del message, fields


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class OperationContext:
    """Resolved operation dependencies; construction never creates paths."""

    profile_name: str
    profile_source: str
    data_root: Path
    cache_root: Path
    storage_policy: str
    shared_group: Optional[str]
    run_id: str
    execution_metadata: Tuple[Tuple[str, str], ...]
    clock: Callable[[], datetime]
    logger: OperationLogger

    def now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("Operation clocks must return timezone-aware datetimes.")
        return value.astimezone(timezone.utc)

    def configuration(self, *, include_paths: bool = True) -> dict[str, object]:
        values: dict[str, object] = {
            "profile": self.profile_name,
            "profile_source": self.profile_source,
            "storage_policy": self.storage_policy,
            "shared_group": self.shared_group,
            "execution_metadata": dict(self.execution_metadata),
        }
        if include_paths:
            values.update(
                {
                    "data_root": str(self.data_root),
                    "cache_root": str(self.cache_root),
                }
            )
        return values


def context_from_profile(
    profile: StorageProfile,
    *,
    run_id: Optional[str] = None,
    execution_metadata: Optional[Mapping[str, object]] = None,
    clock: Callable[[], datetime] = utc_now,
    logger: Optional[OperationLogger] = None,
) -> OperationContext:
    """Capture a resolved profile and dependencies exactly once."""
    metadata = tuple(
        sorted(
            (str(key), str(value))
            for key, value in (execution_metadata or {}).items()
        )
    )
    return OperationContext(
        profile_name=profile.name,
        profile_source=profile.source,
        data_root=Path(profile.data_root).expanduser(),
        cache_root=Path(profile.cache_root).expanduser(),
        storage_policy=profile.storage_policy,
        shared_group=profile.shared_group,
        run_id=run_id or f"run-{uuid.uuid4().hex}",
        execution_metadata=metadata,
        clock=clock,
        logger=logger or NullLogger(),
    )


def context_from_environment(
    profile: str = "auto",
    *,
    environ: Optional[Mapping[str, str]] = None,
    data_root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
    storage_policy: Optional[str] = None,
    shared_group: Optional[str] = None,
    run_id: Optional[str] = None,
    execution_metadata: Optional[Mapping[str, object]] = None,
    clock: Callable[[], datetime] = utc_now,
    logger: Optional[OperationLogger] = None,
) -> OperationContext:
    """Resolve environment/profile configuration at one explicit boundary."""
    resolved = resolve_profile(
        profile,
        environ=environ,
        data_root=data_root,
        cache_root=cache_root,
        storage_policy=storage_policy,
        shared_group=shared_group,
    )
    return context_from_profile(
        resolved,
        run_id=run_id,
        execution_metadata=execution_metadata,
        clock=clock,
        logger=logger,
    )
