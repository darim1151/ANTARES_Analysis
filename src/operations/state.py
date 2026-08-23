"""Explicit state transitions for future writer execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Tuple


class ExecutionState(str, Enum):
    PLANNED = "planned"
    PRECHECKED = "prechecked"
    LOCKED = "locked"
    QUERYING = "querying"
    FETCHING = "fetching"
    STAGED = "staged"
    VALIDATED = "validated"
    PUBLISHED = "published"
    RECONCILING = "reconciling"
    COMPLETE = "complete"
    FAILED = "failed"


LEGAL_TRANSITIONS = {
    ExecutionState.PLANNED: {ExecutionState.PRECHECKED, ExecutionState.FAILED},
    ExecutionState.PRECHECKED: {ExecutionState.LOCKED, ExecutionState.FAILED},
    ExecutionState.LOCKED: {ExecutionState.QUERYING, ExecutionState.FAILED},
    ExecutionState.QUERYING: {ExecutionState.FETCHING, ExecutionState.FAILED},
    ExecutionState.FETCHING: {ExecutionState.STAGED, ExecutionState.FAILED},
    ExecutionState.STAGED: {ExecutionState.VALIDATED, ExecutionState.FAILED},
    ExecutionState.VALIDATED: {ExecutionState.PUBLISHED, ExecutionState.FAILED},
    ExecutionState.PUBLISHED: {ExecutionState.RECONCILING, ExecutionState.FAILED},
    ExecutionState.RECONCILING: {ExecutionState.COMPLETE, ExecutionState.FAILED},
    ExecutionState.COMPLETE: set(),
    ExecutionState.FAILED: set(),
}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class Transition:
    previous: ExecutionState
    current: ExecutionState
    at_utc: str
    reason: Optional[str] = None


@dataclass(frozen=True)
class StateSnapshot:
    state: ExecutionState
    published: bool
    reconciliation_required: bool
    failure_reason: Optional[str]
    transitions: Tuple[Transition, ...]


def _timestamp(value: Optional[datetime]) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("State transition timestamps must be timezone-aware.")
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()


class ExecutionStateMachine:
    """Fail-closed state machine with publication-aware failure semantics."""

    def __init__(self) -> None:
        self._state = ExecutionState.PLANNED
        self._published = False
        self._reconciliation_required = False
        self._failure_reason: Optional[str] = None
        self._transitions: list[Transition] = []

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def published(self) -> bool:
        return self._published

    @property
    def reconciliation_required(self) -> bool:
        return self._reconciliation_required

    def transition(
        self,
        target: ExecutionState,
        *,
        at: Optional[datetime] = None,
        reason: Optional[str] = None,
    ) -> StateSnapshot:
        if target not in LEGAL_TRANSITIONS[self._state]:
            raise IllegalTransition(
                f"Illegal writer transition: {self._state.value} -> {target.value}."
            )
        previous = self._state
        self._state = target
        if target == ExecutionState.PUBLISHED:
            self._published = True
        if target == ExecutionState.FAILED:
            self._failure_reason = reason or "unspecified_failure"
            if self._published:
                self._reconciliation_required = True
        if target == ExecutionState.COMPLETE:
            self._reconciliation_required = False
        self._transitions.append(
            Transition(previous, target, _timestamp(at), reason)
        )
        return self.snapshot()

    def fail(
        self,
        reason: str,
        *,
        at: Optional[datetime] = None,
    ) -> StateSnapshot:
        return self.transition(ExecutionState.FAILED, at=at, reason=reason)

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            state=self._state,
            published=self._published,
            reconciliation_required=self._reconciliation_required,
            failure_reason=self._failure_reason,
            transitions=tuple(self._transitions),
        )
