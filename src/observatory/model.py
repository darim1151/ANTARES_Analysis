"""Immutable ANTARES-native read products; this is not Observatory Bundle V1."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


ADAPTER_VERSION = "1.0.0"
NATIVE_CONTRACT_VERSION = "antares.observatory.native.v1"
BASELINE_SHA = "812c545e14693cdce7ff7458f1d2b50b0804dcd8"
DERIVATION = "DERIVED / NON-AUTHORITATIVE"


@dataclass(frozen=True)
class ArtifactIdentity:
    identity: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class AntaresProvenanceRecord:
    identity: str
    artifact_refs: Tuple[str, ...]
    evidence_json: str
    limitations: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AntaresObservatoryBasis:
    code_sha: str
    code_state: str
    release: Optional[str]
    input_kind: str
    coverage_start: Optional[str]
    coverage_end: Optional[str]
    input_artifacts: Tuple[ArtifactIdentity, ...]
    authority_evidence_refs: Tuple[str, ...]
    generation_sha256: Optional[str]
    sentinel_artifact_ref: Optional[str]
    sentinel_qualification: str = "NOT_REQUALIFIED_BY_ADAPTER"
    baseline_sha: str = BASELINE_SHA
    adapter_version: str = ADAPTER_VERSION
    native_contract_version: str = NATIVE_CONTRACT_VERSION
    derivation: str = DERIVATION


@dataclass(frozen=True)
class NightQualification:
    authority_state: str
    availability: str
    content_state: str
    view_state: str
    scientific_qualification: str
    acquisition_state: Optional[str]
    reason: str
    provenance_ref: str


@dataclass(frozen=True)
class AntaresNightRecord:
    night_date: str
    qualification: NightQualification
    mjd_min: Optional[float] = None
    mjd_max: Optional[float] = None
    loci_rows: Optional[int] = None
    alert_source_rows: Optional[int] = None
    acquisition_started_at_utc: Optional[str] = None
    fetch_completed_at_utc: Optional[str] = None
    publication_started_at_utc: Optional[str] = None
    acquisition_completed_at_utc: Optional[str] = None
    build_finished_at_utc: Optional[str] = None
    mjd_upper_bound: str = "UNKNOWN"


@dataclass(frozen=True)
class AntaresTemporalProduct:
    nights: Tuple[AntaresNightRecord, ...]
    provenance_refs: Tuple[str, ...]
    night_convention: str = "repository UTC-calendar MJD query windows"
    observation_time_scale: str = "UNESTABLISHED"
    interval_semantics: str = "manifest bounds; upper inclusivity not assumed"
    loci_count_definition: str = "saved locus rows in a qualified nightly partition"
    alert_count_definition: str = "saved alert/history source rows; not alerts observed that night"
    derivation: str = DERIVATION


@dataclass(frozen=True)
class FeatureValue:
    feature_ref: str
    value: Optional[Union[float, int]]
    missingness: Optional[str]


@dataclass(frozen=True)
class NativeIdentifier:
    namespace: str
    source_field: str
    # Canonical JSON preserves container shape, order, nulls and multiplicity.
    # Every non-null leaf is a lossless string, including numeric identifiers.
    value_json: str
    availability: str


@dataclass(frozen=True)
class AntaresLocusSnapshot:
    membership_night: str
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    coordinate_state: str
    external_identifiers: Tuple[NativeIdentifier, ...]
    tags: Tuple[str, ...]
    features: Tuple[FeatureValue, ...]
    saved_alert_source_rows: int
    newest_alert_observation_mjd: Optional[float]
    observation_time_missingness: Optional[str]
    provenance_ref: str
    tag_state: str


@dataclass(frozen=True)
class AntaresLocusRecord:
    locus_id: str
    membership_nights: Tuple[str, ...]
    snapshots: Tuple[AntaresLocusSnapshot, ...]
    membership_definition: str = "membership preserved in included saved nightly partitions"
    derivation: str = DERIVATION


@dataclass(frozen=True)
class AntaresFeatureDefinition:
    name: str
    namespace: str
    entity_level: str
    dtype: str
    unit: Optional[str]
    band_label: Optional[str]
    passband_semantics: str
    definition: str
    estimator_semantics: str
    qualification_state: str
    source_rows: int
    column_present_rows: int
    finite_rows: int
    missing_rows: int
    missingness_counts: Tuple[Tuple[str, int], ...]
    laboratory_role: str
    evidence_refs: Tuple[str, ...]


@dataclass(frozen=True)
class Capability:
    name: str
    state: str
    reason: str
    evidence_refs: Tuple[str, ...]


@dataclass(frozen=True)
class AntaresCapabilityManifest:
    capabilities: Tuple[Capability, ...]
    features: Tuple[AntaresFeatureDefinition, ...]
    selection_axes: Tuple[str, ...] = (
        "membership_night_interval", "native_sky_cell_set", "feature_availability",
        "qualified_feature_value_range", "tag_presence", "identifier_namespace",
        "saved_alert_source_row_threshold",
    )
    derivation: str = DERIVATION


@dataclass(frozen=True)
class SkyCell:
    cell_id: str
    ra_min_deg: float
    ra_max_deg: float
    sin_dec_min: float
    sin_dec_max: float
    locus_count: int


@dataclass(frozen=True)
class AntaresSkyProduct:
    cells: Tuple[SkyCell, ...]
    ra_bins: int
    sin_dec_bins: int
    input_loci: int
    included_loci: int
    excluded_coordinates: Tuple[Tuple[str, int], ...]
    provenance_refs: Tuple[str, ...]
    scheme: str = "uniform-ra-sin-dec.v1"
    geometry: str = "equal solid angle longitude/sine-latitude cells"
    ordering: str = "sin-dec index ascending, then RA index ascending"
    coordinate_frame: str = "UNESTABLISHED; source ra/dec convention only"
    population: str = "one latest included saved snapshot per native ANTARES locus"
    interpretation: str = "locus density; survey coverage/footprint unavailable"
    derivation: str = DERIVATION


@dataclass(frozen=True)
class AntaresObservatoryProduct:
    basis: AntaresObservatoryBasis
    capabilities: AntaresCapabilityManifest
    time: AntaresTemporalProduct
    sky: AntaresSkyProduct
    loci: Tuple[AntaresLocusRecord, ...]
    provenance: Tuple[AntaresProvenanceRecord, ...]
    product_kind: str
    shared_contract_binding: str = "UNBOUND"
    derivation: str = DERIVATION
