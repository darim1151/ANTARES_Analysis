"""Pure science-provider contracts for the guarded nightly writer.

This module deliberately contains no live ANTARES implementation.  The only
provider is deterministic and synthetic: it gives the writer a realistic,
in-memory way to qualify the QUERYING and FETCHING lifecycle boundaries
without network or filesystem access.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple, runtime_checkable

import pandas as pd

from .transaction import QueryFetchEvidence


ARTIFACT_NAMES = ("loci.parquet", "alerts.parquet", "manifest.json")
ZERO_ROW_PROOF = "completed_successful_query"


class ProviderStage(str, Enum):
    """The provider lifecycle boundary at which an outcome was observed."""

    QUERY = "query"
    FETCH = "fetch"
    VALIDATION = "validation"
    ARTIFACT = "artifact"
    CONTRACT = "contract"


class ProviderOutcome(str, Enum):
    """Stable, machine-readable provider outcomes."""

    SUCCESS = "success"
    SUCCESS_ZERO = "success_zero"
    QUERY_FAILURE = "query_failure"
    QUERY_INTERRUPTION = "query_interruption"
    FETCH_FAILURE = "fetch_failure"
    PARTIAL_FETCH = "partial_fetch"
    MALFORMED_RESULT = "malformed_result"
    VALIDATION_FAILURE = "validation_failure"


class SyntheticScenario(str, Enum):
    """Supported deterministic synthetic-provider scenarios."""

    SUCCESS_NONZERO = "success_nonzero"
    SUCCESS_ZERO = "success_zero"
    QUERY_FAILURE = "query_failure"
    QUERY_INTERRUPTION = "query_interruption"
    FETCH_FAILURE = "fetch_failure"
    PARTIAL_FETCH = "partial_fetch"
    MALFORMED_RESULT = "malformed_result"
    VALIDATION_FAILURE = "validation_failure"


@dataclass(frozen=True)
class ProviderIssue:
    """Serializable evidence for one provider failure or refusal."""

    code: str
    stage: ProviderStage
    outcome: ProviderOutcome
    message: str
    retryable: bool = False
    partial: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "message": self.message,
            "retryable": self.retryable,
            "partial": self.partial,
        }


class ProviderError(RuntimeError):
    """Base exception carrying a structured provider issue and result."""

    def __init__(
        self,
        issue: ProviderIssue,
        *,
        result: Optional[object] = None,
    ) -> None:
        self.issue = issue
        self.result = result
        super().__init__(f"{issue.code}: {issue.message}")

    @property
    def code(self) -> str:
        return self.issue.code

    @property
    def stage(self) -> ProviderStage:
        return self.issue.stage

    @property
    def outcome(self) -> ProviderOutcome:
        return self.issue.outcome


class ProviderContractError(ProviderError):
    """The caller violated the provider contract."""


class QueryProviderError(ProviderError):
    """A query completed unsuccessfully."""


class QueryInterruptedError(QueryProviderError):
    """A query was interrupted and may carry partial rows."""


class FetchProviderError(ProviderError):
    """Fetching science products failed."""


class PartialFetchError(FetchProviderError):
    """Fetching returned an explicitly incomplete science result."""


class MalformedResultError(FetchProviderError):
    """The provider returned data that does not satisfy its schema."""


class ScienceValidationError(ProviderError):
    """Completed query/fetch data failed the existing science validation."""


class ArtifactValidationError(ProviderError):
    """In-memory artifact construction or reopening failed validation."""


def _canonical_utc(value: str, field_name: str) -> str:
    text = str(value)
    parsed_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(parsed_text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class NightScienceRequest:
    """An explicit, LSST-only request for one UTC nightly science window."""

    date_utc: str
    mjd_min: float
    mjd_max: float
    ingested_at_utc: Optional[str] = None
    query_tag: Optional[str] = None
    target_loci: int = 2
    range_label: Optional[str] = None
    lsst_only: bool = True
    prior_locus_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            parsed_date = date.fromisoformat(self.date_utc)
        except (TypeError, ValueError) as exc:
            raise ValueError("date_utc must use canonical YYYY-MM-DD form.") from exc
        if parsed_date.isoformat() != self.date_utc:
            raise ValueError("date_utc must use canonical YYYY-MM-DD form.")

        if isinstance(self.mjd_min, bool) or isinstance(self.mjd_max, bool):
            raise ValueError("MJD bounds must be finite numbers.")
        try:
            lower = float(self.mjd_min)
            upper = float(self.mjd_max)
        except (TypeError, ValueError) as exc:
            raise ValueError("MJD bounds must be finite numbers.") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("mjd_min must be finite and strictly below mjd_max.")
        object.__setattr__(self, "mjd_min", lower)
        object.__setattr__(self, "mjd_max", upper)

        if isinstance(self.target_loci, bool) or not isinstance(self.target_loci, int):
            raise ValueError("target_loci must be a positive integer.")
        if self.target_loci <= 0:
            raise ValueError("target_loci must be a positive integer.")
        if self.lsst_only is not True:
            raise ValueError("The nightly science-provider contract is LSST-only.")

        ingested = self.ingested_at_utc or f"{self.date_utc}T00:00:00+00:00"
        object.__setattr__(
            self,
            "ingested_at_utc",
            _canonical_utc(ingested, "ingested_at_utc"),
        )
        label = self.range_label or f"Synthetic LSST night {self.date_utc}"
        if not str(label).strip():
            raise ValueError("range_label must not be blank.")
        object.__setattr__(self, "range_label", str(label))
        object.__setattr__(
            self,
            "prior_locus_ids",
            tuple(str(value) for value in self.prior_locus_ids),
        )


@dataclass(frozen=True)
class QueryStageEvidence:
    """Evidence produced at the end of the query boundary."""

    completed: bool
    partial: bool
    returned_loci: int
    errors: Tuple[ProviderIssue, ...] = ()

    @property
    def clean(self) -> bool:
        return bool(
            self.completed
            and not self.partial
            and not self.errors
            and self.returned_loci >= 0
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "completed": self.completed,
            "partial": self.partial,
            "returned_loci": self.returned_loci,
            "errors": [issue.as_dict() for issue in self.errors],
        }


@dataclass(frozen=True)
class FetchStageEvidence:
    """Evidence produced at the end of the fetch boundary."""

    completed: bool
    partial: bool
    loci_rows: int
    alert_rows: int
    errors: Tuple[ProviderIssue, ...] = ()

    @property
    def clean(self) -> bool:
        return bool(
            self.completed
            and not self.partial
            and not self.errors
            and self.loci_rows >= 0
            and self.alert_rows >= 0
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "completed": self.completed,
            "partial": self.partial,
            "loci_rows": self.loci_rows,
            "alert_rows": self.alert_rows,
            "errors": [issue.as_dict() for issue in self.errors],
        }


@dataclass(frozen=True, eq=False)
class NightQueryResult:
    """The query result and its independent completion evidence."""

    request: NightScienceRequest
    provider_name: str
    scenario: str
    outcome: ProviderOutcome
    loci: Optional[pd.DataFrame]
    evidence: QueryStageEvidence

    @property
    def query_completed(self) -> bool:
        return self.evidence.completed

    @property
    def query_partial(self) -> bool:
        return self.evidence.partial

    @property
    def query_errors(self) -> Tuple[ProviderIssue, ...]:
        return self.evidence.errors

    @property
    def clean(self) -> bool:
        return self.evidence.clean

    def require_completed(self) -> "NightQueryResult":
        if self.clean:
            return self
        issue = self.evidence.errors[0] if self.evidence.errors else ProviderIssue(
            "query_not_complete",
            ProviderStage.QUERY,
            self.outcome,
            "The query did not produce clean completion evidence.",
        )
        if self.outcome == ProviderOutcome.QUERY_INTERRUPTION:
            raise QueryInterruptedError(issue, result=self)
        raise QueryProviderError(issue, result=self)


@dataclass(frozen=True, eq=False)
class NightScienceResult:
    """Fetched nightly frames with query, fetch, and validation evidence."""

    request: NightScienceRequest
    provider_name: str
    scenario: str
    outcome: ProviderOutcome
    query_result: NightQueryResult
    loci: Optional[pd.DataFrame]
    alerts: Optional[pd.DataFrame]
    fetch_evidence: FetchStageEvidence
    validation: Mapping[str, Any]
    validation_errors: Tuple[ProviderIssue, ...]
    evidence: QueryFetchEvidence

    @property
    def query_evidence(self) -> QueryStageEvidence:
        return self.query_result.evidence

    @property
    def query_completed(self) -> bool:
        return self.query_evidence.completed

    @property
    def query_partial(self) -> bool:
        return self.query_evidence.partial

    @property
    def query_errors(self) -> Tuple[ProviderIssue, ...]:
        return self.query_evidence.errors

    @property
    def fetch_completed(self) -> bool:
        return self.fetch_evidence.completed

    @property
    def fetch_partial(self) -> bool:
        return self.fetch_evidence.partial

    @property
    def fetch_errors(self) -> Tuple[ProviderIssue, ...]:
        return self.fetch_evidence.errors

    @property
    def publishable(self) -> bool:
        return bool(
            self.outcome in {ProviderOutcome.SUCCESS, ProviderOutcome.SUCCESS_ZERO}
            and self.evidence.clean
            and not self.validation_errors
            and self.validation.get("append_ready") is True
            and self.loci is not None
            and self.alerts is not None
        )

    def require_publishable(self) -> "NightScienceResult":
        if self.publishable:
            return self
        issues = (
            self.query_result.evidence.errors
            + self.fetch_evidence.errors
            + self.validation_errors
        )
        issue = issues[0] if issues else ProviderIssue(
            "science_not_publishable",
            ProviderStage.VALIDATION,
            self.outcome,
            "Nightly science does not carry complete publication evidence.",
        )
        error_type = {
            ProviderOutcome.QUERY_FAILURE: QueryProviderError,
            ProviderOutcome.QUERY_INTERRUPTION: QueryInterruptedError,
            ProviderOutcome.FETCH_FAILURE: FetchProviderError,
            ProviderOutcome.PARTIAL_FETCH: PartialFetchError,
            ProviderOutcome.MALFORMED_RESULT: MalformedResultError,
            ProviderOutcome.VALIDATION_FAILURE: ScienceValidationError,
        }.get(self.outcome, ProviderError)
        raise error_type(issue, result=self)


@runtime_checkable
class ScienceProvider(Protocol):
    """Provider boundary used by a future guarded writer."""

    @property
    def provider_name(self) -> str:
        ...

    def query(self, request: NightScienceRequest) -> NightQueryResult:
        ...

    def fetch(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
    ) -> NightScienceResult:
        ...

    def fetch_night(self, request: NightScienceRequest) -> NightScienceResult:
        ...


def _issue_text(issues: Tuple[ProviderIssue, ...]) -> Tuple[str, ...]:
    return tuple(f"{issue.code}: {issue.message}" for issue in issues)


def _query_issue(
    code: str,
    outcome: ProviderOutcome,
    message: str,
    *,
    retryable: bool,
    partial: bool = False,
) -> ProviderIssue:
    return ProviderIssue(
        code,
        ProviderStage.QUERY,
        outcome,
        message,
        retryable=retryable,
        partial=partial,
    )


def _fetch_issue(
    code: str,
    outcome: ProviderOutcome,
    message: str,
    *,
    retryable: bool,
    partial: bool = False,
) -> ProviderIssue:
    return ProviderIssue(
        code,
        ProviderStage.FETCH,
        outcome,
        message,
        retryable=retryable,
        partial=partial,
    )


def _result_evidence(
    query_evidence: QueryStageEvidence,
    fetch_evidence: FetchStageEvidence,
) -> QueryFetchEvidence:
    zero_proof = None
    if (
        query_evidence.clean
        and fetch_evidence.clean
        and fetch_evidence.loci_rows == 0
        and fetch_evidence.alert_rows == 0
    ):
        zero_proof = ZERO_ROW_PROOF
    return QueryFetchEvidence(
        query_completed=query_evidence.completed,
        fetch_completed=fetch_evidence.completed,
        loci_rows=fetch_evidence.loci_rows,
        alert_rows=fetch_evidence.alert_rows,
        query_errors=_issue_text(query_evidence.errors),
        fetch_errors=_issue_text(fetch_evidence.errors),
        zero_row_proof=zero_proof,
    )


class SyntheticScienceProvider:
    """Deterministic provider for lifecycle, failure, and artifact tests.

    No method performs network or filesystem I/O.  The scenario is fixed when
    the provider is constructed, making repeated calls byte-for-byte stable
    for the same request and dependency versions.
    """

    provider_name = "synthetic"

    def __init__(
        self,
        scenario: str = SyntheticScenario.SUCCESS_NONZERO.value,
    ) -> None:
        try:
            self.scenario = SyntheticScenario(scenario)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in SyntheticScenario)
            raise ValueError(
                f"Unknown synthetic scenario {scenario!r}; choose from {allowed}."
            ) from exc

    @staticmethod
    def _loci(request: NightScienceRequest) -> pd.DataFrame:
        width = request.mjd_max - request.mjd_min
        return pd.DataFrame(
            {
                "locus_id": ["ANT-SYNTH-LSST-0001", "ANT-SYNTH-LSST-0002"],
                "ra": [12.345678, 287.654321],
                "dec": [-21.25, 44.5],
                "newest_alert_observation_time": [
                    request.mjd_min + width * 0.25,
                    request.mjd_min + width * 0.75,
                ],
                "tags": ["lsst, synthetic", "lsst, synthetic"],
                "dia_object_id": ["LSST-DIA-SYNTH-0001", ""],
                "ss_object_id": ["", "LSST-SS-SYNTH-0002"],
                "ztf_object_id": ["", ""],
                "brightest_alert_magnitude": [20.125, 21.375],
                "num_mag_values": [2, 2],
            }
        )

    @staticmethod
    def _alerts(request: NightScienceRequest, loci: pd.DataFrame) -> pd.DataFrame:
        width = request.mjd_max - request.mjd_min
        locus_ids = loci["locus_id"].astype(str).tolist()
        rows = []
        for locus_index, locus_id in enumerate(locus_ids):
            for sample_index in range(2):
                fraction = 0.2 + 0.2 * (locus_index * 2 + sample_index)
                rows.append(
                    {
                        "locus_id": locus_id,
                        "alert_id": (
                            f"SYNTH-ALERT-{locus_index + 1:02d}-{sample_index + 1:02d}"
                        ),
                        "mjd": request.mjd_min + width * fraction,
                        "ztf_magpsf": 20.0 + locus_index + sample_index * 0.1,
                        "ztf_sigmapsf": 0.05 + sample_index * 0.01,
                        "ztf_fid": sample_index + 1,
                        "lsst_band": "g" if sample_index == 0 else "r",
                    }
                )
        return pd.DataFrame(rows)

    def query(self, request: NightScienceRequest) -> NightQueryResult:
        if not isinstance(request, NightScienceRequest):
            issue = ProviderIssue(
                "invalid_request_type",
                ProviderStage.CONTRACT,
                ProviderOutcome.QUERY_FAILURE,
                "query() requires a NightScienceRequest.",
            )
            raise ProviderContractError(issue)

        scenario = self.scenario
        if scenario == SyntheticScenario.QUERY_FAILURE:
            issue = _query_issue(
                "synthetic_query_failure",
                ProviderOutcome.QUERY_FAILURE,
                "The synthetic query failed before returning rows.",
                retryable=True,
            )
            return NightQueryResult(
                request,
                self.provider_name,
                scenario.value,
                ProviderOutcome.QUERY_FAILURE,
                None,
                QueryStageEvidence(False, False, 0, (issue,)),
            )

        loci = self._loci(request)
        if scenario == SyntheticScenario.SUCCESS_ZERO:
            loci = pd.DataFrame()
        elif scenario == SyntheticScenario.QUERY_INTERRUPTION:
            loci = loci.head(1).copy()
            issue = _query_issue(
                "synthetic_query_interrupted",
                ProviderOutcome.QUERY_INTERRUPTION,
                "The synthetic query was interrupted after a partial result.",
                retryable=True,
                partial=True,
            )
            return NightQueryResult(
                request,
                self.provider_name,
                scenario.value,
                ProviderOutcome.QUERY_INTERRUPTION,
                loci,
                QueryStageEvidence(False, True, len(loci), (issue,)),
            )
        elif scenario == SyntheticScenario.MALFORMED_RESULT:
            loci = loci.drop(columns=["locus_id"])
        elif scenario == SyntheticScenario.VALIDATION_FAILURE:
            loci = loci.copy()
            loci.loc[loci.index[0], "ra"] = 361.0

        return NightQueryResult(
            request,
            self.provider_name,
            scenario.value,
            ProviderOutcome.SUCCESS,
            loci,
            QueryStageEvidence(True, False, len(loci), ()),
        )

    def _failed_or_skipped_fetch(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
    ) -> NightScienceResult:
        outcome = query_result.outcome
        issue = _fetch_issue(
            "fetch_not_attempted_after_query",
            outcome,
            "Fetch was not attempted because the query was not cleanly completed.",
            retryable=any(item.retryable for item in query_result.evidence.errors),
            partial=query_result.evidence.partial,
        )
        fetch_evidence = FetchStageEvidence(False, False, 0, 0, (issue,))
        return NightScienceResult(
            request,
            self.provider_name,
            self.scenario.value,
            outcome,
            query_result,
            None,
            None,
            fetch_evidence,
            {},
            (),
            _result_evidence(query_result.evidence, fetch_evidence),
        )

    def fetch(
        self,
        request: NightScienceRequest,
        query_result: NightQueryResult,
    ) -> NightScienceResult:
        if not isinstance(request, NightScienceRequest):
            issue = ProviderIssue(
                "invalid_request_type",
                ProviderStage.CONTRACT,
                ProviderOutcome.FETCH_FAILURE,
                "fetch() requires a NightScienceRequest.",
            )
            raise ProviderContractError(issue)
        if not isinstance(query_result, NightQueryResult):
            issue = ProviderIssue(
                "invalid_query_result_type",
                ProviderStage.CONTRACT,
                ProviderOutcome.FETCH_FAILURE,
                "fetch() requires a NightQueryResult.",
            )
            raise ProviderContractError(issue)
        if query_result.request != request:
            issue = ProviderIssue(
                "query_request_mismatch",
                ProviderStage.CONTRACT,
                ProviderOutcome.FETCH_FAILURE,
                "The query result belongs to a different nightly request.",
            )
            raise ProviderContractError(issue, result=query_result)
        if (
            query_result.provider_name != self.provider_name
            or query_result.scenario != self.scenario.value
        ):
            issue = ProviderIssue(
                "query_provider_mismatch",
                ProviderStage.CONTRACT,
                ProviderOutcome.FETCH_FAILURE,
                "The query result belongs to a different provider instance.",
            )
            raise ProviderContractError(issue, result=query_result)
        if not query_result.clean:
            return self._failed_or_skipped_fetch(request, query_result)

        # Import existing preparation/validation semantics only at the fetch
        # boundary.  Importing or invoking these functions performs no live
        # query; the synthetic provider never calls ingestion entry points.
        from ..history import prepare_alerts, prepare_loci, validation_summary

        raw_loci = query_result.loci
        if not isinstance(raw_loci, pd.DataFrame):
            missing = "query result is not a pandas DataFrame"
        elif not raw_loci.empty:
            required = {
                "locus_id",
                "ra",
                "dec",
                "newest_alert_observation_time",
            }
            absent = sorted(required - set(raw_loci.columns))
            missing = f"missing required columns: {', '.join(absent)}" if absent else ""
        else:
            missing = ""
        if missing:
            issue = _fetch_issue(
                "synthetic_malformed_query_result",
                ProviderOutcome.MALFORMED_RESULT,
                f"The synthetic query result is malformed ({missing}).",
                retryable=False,
            )
            fetch_evidence = FetchStageEvidence(False, False, 0, 0, (issue,))
            return NightScienceResult(
                request,
                self.provider_name,
                self.scenario.value,
                ProviderOutcome.MALFORMED_RESULT,
                query_result,
                None,
                None,
                fetch_evidence,
                {},
                (),
                _result_evidence(query_result.evidence, fetch_evidence),
            )

        if self.scenario == SyntheticScenario.FETCH_FAILURE:
            issue = _fetch_issue(
                "synthetic_fetch_failure",
                ProviderOutcome.FETCH_FAILURE,
                "The synthetic light-curve fetch failed before completion.",
                retryable=True,
            )
            fetch_evidence = FetchStageEvidence(False, False, 0, 0, (issue,))
            return NightScienceResult(
                request,
                self.provider_name,
                self.scenario.value,
                ProviderOutcome.FETCH_FAILURE,
                query_result,
                None,
                None,
                fetch_evidence,
                {},
                (),
                _result_evidence(query_result.evidence, fetch_evidence),
            )

        loci = prepare_loci(
            raw_loci,
            request.date_utc,
            request.mjd_min,
            request.mjd_max,
            request.ingested_at_utc,
        )
        raw_alerts = (
            pd.DataFrame()
            if loci.empty
            else self._alerts(request, raw_loci)
        )

        partial = self.scenario == SyntheticScenario.PARTIAL_FETCH
        fetch_issues: Tuple[ProviderIssue, ...] = ()
        if partial:
            first_locus = str(raw_loci.iloc[0]["locus_id"])
            raw_alerts = raw_alerts[raw_alerts["locus_id"] == first_locus].copy()
            issue = _fetch_issue(
                "synthetic_partial_fetch",
                ProviderOutcome.PARTIAL_FETCH,
                "Only one of the requested synthetic light curves was fetched.",
                retryable=True,
                partial=True,
            )
            fetch_issues = (issue,)

        alerts = prepare_alerts(raw_alerts, request.date_utc, request.range_label)
        fetch_evidence = FetchStageEvidence(
            completed=not partial,
            partial=partial,
            loci_rows=len(loci),
            alert_rows=len(alerts),
            errors=fetch_issues,
        )
        combined_evidence = _result_evidence(
            query_result.evidence,
            fetch_evidence,
        )
        validation = validation_summary(
            loci,
            alerts,
            mjd_min=request.mjd_min,
            mjd_max=request.mjd_max,
            prior_locus_ids=request.prior_locus_ids,
            lsst_only=True,
            query_completed=query_result.evidence.completed,
            query_fetch_clean=combined_evidence.clean,
        )

        if partial:
            return NightScienceResult(
                request,
                self.provider_name,
                self.scenario.value,
                ProviderOutcome.PARTIAL_FETCH,
                query_result,
                loci,
                alerts,
                fetch_evidence,
                validation,
                (),
                combined_evidence,
            )

        validation_errors: Tuple[ProviderIssue, ...] = ()
        outcome = (
            ProviderOutcome.SUCCESS_ZERO
            if loci.empty and alerts.empty
            else ProviderOutcome.SUCCESS
        )
        if validation.get("append_ready") is not True:
            outcome = ProviderOutcome.VALIDATION_FAILURE
            validation_errors = (
                ProviderIssue(
                    "science_validation_failed",
                    ProviderStage.VALIDATION,
                    outcome,
                    "Existing nightly validation did not grant append readiness.",
                    retryable=False,
                ),
            )

        return NightScienceResult(
            request,
            self.provider_name,
            self.scenario.value,
            outcome,
            query_result,
            loci,
            alerts,
            fetch_evidence,
            validation,
            validation_errors,
            combined_evidence,
        )

    def fetch_night(self, request: NightScienceRequest) -> NightScienceResult:
        return self.fetch(request, self.query(request))


def _artifact_issue(code: str, message: str) -> ProviderIssue:
    return ProviderIssue(
        code,
        ProviderStage.ARTIFACT,
        ProviderOutcome.VALIDATION_FAILURE,
        message,
        retryable=False,
    )


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize one frame deterministically to an in-memory Parquet file."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
        sink = pa.BufferOutputStream()
        pq.write_table(
            table,
            sink,
            version="2.6",
            data_page_version="1.0",
            compression="snappy",
            use_dictionary=False,
            write_statistics=True,
        )
        return sink.getvalue().to_pybytes()
    except Exception as exc:
        raise ArtifactValidationError(
            _artifact_issue(
                "parquet_construction_failed",
                f"Could not construct an in-memory Parquet artifact: {exc}",
            )
        ) from exc


def build_night_artifacts(result: NightScienceResult) -> Dict[str, bytes]:
    """Build the exact three manifest-last transaction payloads in memory."""

    if not isinstance(result, NightScienceResult):
        raise ProviderContractError(
            ProviderIssue(
                "invalid_science_result_type",
                ProviderStage.CONTRACT,
                ProviderOutcome.VALIDATION_FAILURE,
                "Artifact construction requires a NightScienceResult.",
            )
        )
    result.require_publishable()
    assert result.loci is not None
    assert result.alerts is not None

    loci_bytes = _parquet_bytes(result.loci)
    alerts_bytes = _parquet_bytes(result.alerts)
    request = result.request
    validation = dict(result.validation)
    manifest: Dict[str, Any] = {
        "schema_version": "phase5.synthetic-night.v1",
        "provider": result.provider_name,
        "provider_scenario": result.scenario,
        "synthetic": result.provider_name == "synthetic",
        "date_utc": request.date_utc,
        "mjd_min": request.mjd_min,
        "mjd_max": request.mjd_max,
        "query_tag": request.query_tag,
        "target_loci": request.target_loci,
        "actual_loci": len(result.loci),
        "alert_rows": len(result.alerts),
        "chunk_count": 1,
        "split_count": 0,
        "saturated_chunk_count": 0,
        "status": "complete",
        "survey_mode": "lsst",
        "lsst_filter_used": True,
        "parallel_shards": 1,
        "lsst_dia_count": int(validation.get("lsst_dia_count", 0)),
        "lsst_ss_count": int(validation.get("lsst_ss_count", 0)),
        "ztf_object_id_count": int(validation.get("ztf_object_id_count", 0)),
        "started_at_utc": request.ingested_at_utc,
        "finished_at_utc": request.ingested_at_utc,
        "runtime_seconds": 0.0,
        "validation": validation,
        "validation_inputs": {
            "prior_locus_ids": list(request.prior_locus_ids),
        },
        "query_evidence": result.query_evidence.as_dict(),
        "fetch_evidence": result.fetch_evidence.as_dict(),
        "query_fetch_evidence": {
            "query_completed": result.evidence.query_completed,
            "fetch_completed": result.evidence.fetch_completed,
            "loci_rows": result.evidence.loci_rows,
            "alert_rows": result.evidence.alert_rows,
            "query_errors": list(result.evidence.query_errors),
            "fetch_errors": list(result.evidence.fetch_errors),
            "zero_row_proof": result.evidence.zero_row_proof,
        },
        "paths": {
            "loci": "loci.parquet",
            "alerts": "alerts.parquet",
            "manifest": "manifest.json",
        },
        "artifacts": {
            "loci.parquet": {
                "bytes": len(loci_bytes),
                "sha256": hashlib.sha256(loci_bytes).hexdigest(),
            },
            "alerts.parquet": {
                "bytes": len(alerts_bytes),
                "sha256": hashlib.sha256(alerts_bytes).hexdigest(),
            },
        },
    }
    try:
        manifest_bytes = (
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            _artifact_issue(
                "manifest_construction_failed",
                f"Could not construct deterministic manifest JSON: {exc}",
            )
        ) from exc
    return {
        "loci.parquet": loci_bytes,
        "alerts.parquet": alerts_bytes,
        "manifest.json": manifest_bytes,
    }


@dataclass(frozen=True, eq=False)
class ReopenedNightArtifacts:
    """Independently reopened, in-memory nightly artifacts."""

    loci: pd.DataFrame
    alerts: pd.DataFrame
    manifest: Mapping[str, Any]


def _raise_artifact(code: str, message: str) -> None:
    raise ArtifactValidationError(_artifact_issue(code, message))


def _read_parquet_bytes(payload: bytes, name: str) -> pd.DataFrame:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pq.read_table(pa.BufferReader(payload)).to_pandas()
    except Exception as exc:
        raise ArtifactValidationError(
            _artifact_issue(
                "parquet_reopen_failed",
                f"Could not reopen {name} from memory: {exc}",
            )
        ) from exc


def reopen_and_validate_artifacts(
    artifacts: Mapping[str, bytes],
    expected: Optional[NightScienceResult] = None,
) -> ReopenedNightArtifacts:
    """Reopen and independently validate a complete in-memory artifact set."""

    if set(artifacts) != set(ARTIFACT_NAMES):
        _raise_artifact(
            "artifact_set_invalid",
            "The nightly artifact set must contain exactly loci, alerts, and manifest.",
        )
    for name in ARTIFACT_NAMES:
        if not isinstance(artifacts[name], bytes):
            _raise_artifact(
                "artifact_payload_invalid",
                f"Artifact {name} is not an exact bytes payload.",
            )

    try:
        manifest = json.loads(artifacts["manifest.json"].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            _artifact_issue(
                "manifest_reopen_failed",
                f"Could not reopen manifest.json: {exc}",
            )
        ) from exc
    if not isinstance(manifest, dict):
        _raise_artifact("manifest_type_invalid", "manifest.json is not a JSON object.")
    if manifest.get("schema_version") != "phase5.synthetic-night.v1":
        _raise_artifact("manifest_schema_invalid", "Unexpected manifest schema version.")
    if manifest.get("lsst_filter_used") is not True:
        _raise_artifact("manifest_lsst_filter_invalid", "Manifest is not LSST-only.")

    loci = _read_parquet_bytes(artifacts["loci.parquet"], "loci.parquet")
    alerts = _read_parquet_bytes(artifacts["alerts.parquet"], "alerts.parquet")
    if manifest.get("actual_loci") != len(loci):
        _raise_artifact("loci_count_mismatch", "Manifest and loci row counts differ.")
    if manifest.get("alert_rows") != len(alerts):
        _raise_artifact("alert_count_mismatch", "Manifest and alert row counts differ.")

    recorded_artifacts = manifest.get("artifacts")
    if not isinstance(recorded_artifacts, dict):
        _raise_artifact("artifact_digest_missing", "Manifest artifact digests are missing.")
    for name in ("loci.parquet", "alerts.parquet"):
        recorded = recorded_artifacts.get(name)
        if not isinstance(recorded, dict):
            _raise_artifact("artifact_digest_missing", f"Digest for {name} is missing.")
        payload = artifacts[name]
        if (
            recorded.get("bytes") != len(payload)
            or recorded.get("sha256") != hashlib.sha256(payload).hexdigest()
        ):
            _raise_artifact(
                "artifact_digest_mismatch",
                f"Size or SHA-256 mismatch for {name}.",
            )

    query_fetch = manifest.get("query_fetch_evidence")
    if not isinstance(query_fetch, dict):
        _raise_artifact("query_fetch_evidence_missing", "Query/fetch evidence is missing.")
    clean = bool(
        query_fetch.get("query_completed") is True
        and query_fetch.get("fetch_completed") is True
        and query_fetch.get("query_errors") == []
        and query_fetch.get("fetch_errors") == []
    )
    if not clean:
        _raise_artifact("query_fetch_evidence_invalid", "Artifacts record an unclean fetch.")
    if query_fetch.get("loci_rows") != len(loci):
        _raise_artifact("query_loci_count_mismatch", "Evidence and loci rows differ.")
    if query_fetch.get("alert_rows") != len(alerts):
        _raise_artifact("query_alert_count_mismatch", "Evidence and alert rows differ.")
    expected_zero_proof = ZERO_ROW_PROOF if loci.empty and alerts.empty else None
    if query_fetch.get("zero_row_proof") != expected_zero_proof:
        _raise_artifact("zero_row_proof_invalid", "Zero-row proof is missing or spurious.")

    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("append_ready") is not True:
        _raise_artifact("manifest_validation_invalid", "Manifest is not append-ready.")

    # Independently re-run the existing scientific checks against reopened
    # Parquet frames instead of trusting the serialized validation object.
    from ..history import validation_summary

    validation_inputs = manifest.get("validation_inputs", {})
    prior_locus_ids = (
        validation_inputs.get("prior_locus_ids", [])
        if isinstance(validation_inputs, dict)
        else []
    )
    try:
        regenerated = validation_summary(
            loci,
            alerts,
            mjd_min=manifest.get("mjd_min"),
            mjd_max=manifest.get("mjd_max"),
            prior_locus_ids=prior_locus_ids,
            lsst_only=True,
            query_completed=True,
            query_fetch_clean=True,
        )
    except Exception as exc:
        raise ArtifactValidationError(
            _artifact_issue(
                "artifact_science_revalidation_failed",
                f"Reopened science could not be revalidated: {exc}",
            )
        ) from exc
    if regenerated != validation or regenerated.get("append_ready") is not True:
        _raise_artifact(
            "artifact_science_validation_mismatch",
            "Reopened science does not reproduce manifest validation.",
        )

    if expected is not None:
        expected.require_publishable()
        assert expected.loci is not None
        assert expected.alerts is not None
        try:
            pd.testing.assert_frame_equal(loci, expected.loci, check_exact=True)
            pd.testing.assert_frame_equal(alerts, expected.alerts, check_exact=True)
        except AssertionError as exc:
            raise ArtifactValidationError(
                _artifact_issue(
                    "artifact_frame_mismatch",
                    f"Reopened frames differ from the provider result: {exc}",
                )
            ) from exc
        if expected.request.date_utc != manifest.get("date_utc"):
            _raise_artifact(
                "artifact_request_mismatch",
                "Reopened manifest belongs to a different nightly request.",
            )

    return ReopenedNightArtifacts(loci, alerts, manifest)


def validate_night_artifacts(
    artifacts: Mapping[str, bytes],
    expected: Optional[NightScienceResult] = None,
) -> bool:
    """Return true after complete in-memory reopen and validation."""

    reopen_and_validate_artifacts(artifacts, expected=expected)
    return True


__all__ = [
    "ARTIFACT_NAMES",
    "ArtifactValidationError",
    "FetchProviderError",
    "FetchStageEvidence",
    "MalformedResultError",
    "NightQueryResult",
    "NightScienceRequest",
    "NightScienceResult",
    "PartialFetchError",
    "ProviderContractError",
    "ProviderError",
    "ProviderIssue",
    "ProviderOutcome",
    "ProviderStage",
    "QueryInterruptedError",
    "QueryProviderError",
    "QueryStageEvidence",
    "ReopenedNightArtifacts",
    "ScienceProvider",
    "ScienceValidationError",
    "SyntheticScenario",
    "SyntheticScienceProvider",
    "ZERO_ROW_PROOF",
    "build_night_artifacts",
    "reopen_and_validate_artifacts",
    "validate_night_artifacts",
]
