/**
 * Unified Scientific Observatory — First-Light read contract.
 *
 * One scientific workspace, multiple broker-native scientific domains, and no
 * false ontology unification. Every type here keeps the domain explicit:
 * an ANTARES locus is never a Fink DiaObject, an ANTARES tag is never an
 * astrophysical class, and a Fink classifier output is a source-time snapshot,
 * not a timeless object label.
 *
 * The bundle is a static, GET-only, immutable read product. Adapters that
 * later read Parquet, Arrow, HEALPix/HATS products or an immutable read API
 * must produce exactly these payloads; the frontend does not change.
 */

export const OBSERVATORY_BUNDLE_CONTRACT = "uso.observatory-bundle" as const;
export const OBSERVATORY_BUNDLE_CONTRACT_VERSION = "1.0.0" as const;

/* -------------------------------------------------------------------------- */
/* Domains, evidence and capabilities                                          */
/* -------------------------------------------------------------------------- */

/** Broker-native scientific domains. There is no "unified" domain. */
export type DomainId = "antares" | "fink";
export const DOMAIN_IDS: readonly DomainId[] = ["antares", "fink"];

/** Presentation mode of the workspace; COMPARE never merges domains. */
export type SourceMode = DomainId | "compare";

/**
 * What kind of evidence backs a value. Ordered from strongest to weakest in
 * {@link EVIDENCE_RANK}. ACCEPTED_SCIENCE is reserved for a Control-adjudicated
 * future basis and is rejected in a FIRST_LIGHT_FIXTURE bundle.
 */
export type EvidenceClass =
  | "ACCEPTED_SCIENCE"
  | "VALIDATED_TRANSPORT_EVIDENCE"
  | "COMMITTED_OPERATIONAL_RECORD"
  | "LEGACY_SAMPLE"
  | "SYNTHETIC_DEMO"
  | "SYNTHETIC_FIXTURE";

export const EVIDENCE_RANK: Record<EvidenceClass, number> = {
  ACCEPTED_SCIENCE: 5,
  VALIDATED_TRANSPORT_EVIDENCE: 4,
  COMMITTED_OPERATIONAL_RECORD: 3,
  LEGACY_SAMPLE: 2,
  SYNTHETIC_DEMO: 1,
  SYNTHETIC_FIXTURE: 0
};

export const SYNTHETIC_EVIDENCE: readonly EvidenceClass[] = ["SYNTHETIC_DEMO", "SYNTHETIC_FIXTURE"];

export type CapabilityState = "AVAILABLE" | "PARTIALLY_QUALIFIED" | "UNAVAILABLE";

export type CapabilityArea = "time" | "sky" | "entity" | "features" | "relation" | "compare";

/**
 * One feature of the workspace as supported by the pinned basis. Presentation
 * components ask the manifest, never `if (domain === "fink")`.
 */
export type Capability = {
  /** `${scope}:${area}.${name}`, e.g. `fink:sky.coverage`. */
  id: string;
  scope: DomainId | "relation" | "workspace";
  area: CapabilityArea;
  name: string;
  state: CapabilityState;
  /** Evidence backing the payload that drives this capability (empty if UNAVAILABLE). */
  evidence: EvidenceClass[];
  summary: string;
  reason: string;
  /** Stable machine-readable status codes, e.g. NOT_DELIVERY_VALIDATED, NOT_ADMITTED. */
  codes: string[];
  qualifications: string[];
};

export type CapabilityManifest = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  contract_version: string;
  basis_id: string;
  capabilities: Capability[];
};

/* -------------------------------------------------------------------------- */
/* Basis                                                                       */
/* -------------------------------------------------------------------------- */

export type TimeScale = "UTC" | "TAI";

/**
 * How one domain stores time and bins it into UTC dates. The workspace never
 * converts between scales; each adapter declares and performs its own mapping.
 */
export type TimeSemantics = {
  /** Native field that carries the stored time. */
  stored_field: string;
  format: "MJD";
  scale: TimeScale;
  /** How the scale was established (e.g. "historical exporter treats MJD as UTC"). */
  scale_basis: string;
  /** UTC-date binning rule, always half-open [00:00, 24:00) UTC. */
  date_binning: string;
  /** Which single UTC date an entity is assigned to for time selection. */
  entity_date_rule: string;
};

export type DomainBuildKind =
  | "LEGACY_DEMO_EXPORT"
  | "SYNTHETIC_FIXTURE_WITH_COMMITTED_EVIDENCE"
  | "PUBLISHED_AUTHORITY_BUILD"
  | "QUALIFIED_COHORT_CATALOG";

/** An independently versioned source build pinned by a basis. */
export type DomainBuildPin = {
  domain: DomainId;
  build_id: string;
  build_kind: DomainBuildKind;
  label: string;
  evidence: EvidenceClass[];
  science_ready: boolean;
  /** Broker-native ontology of the entities this build exposes. */
  native_ontology: string;
  source: {
    repository: string;
    revision: string | null;
    paths: string[];
  };
  time: TimeSemantics;
};

/** Future versioned cross-broker relation product. Always null in this gate. */
export type RelationBuildPin = {
  relation_id: string;
  version: string;
  method: string;
  evidence: EvidenceClass[];
};

export type VersionPin = { id: string; version: string };

/**
 * The pinned scientific basis. Each source domain is versioned independently;
 * there is no synchronized "Observatory generation".
 */
export type ObservatoryBasis = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  contract_version: string;
  basis_id: string;
  label: string;
  status: "FIRST_LIGHT_FIXTURE" | "QUALIFIED";
  science_ready: boolean;
  domains: Record<DomainId, DomainBuildPin>;
  relation: RelationBuildPin | null;
  semantic_contract: VersionPin;
  feature_registry: VersionPin;
  analysis_kernel: VersionPin;
  invariants: string[];
};

/* -------------------------------------------------------------------------- */
/* Time                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Per-UTC-date state. These are never interchangeable:
 * - AVAILABLE: admitted to this basis with qualified evidence.
 * - ZERO: available, and the source delivered zero records for the date.
 * - UNQUALIFIED: data exist and are transport-validated, but are not
 *   scientifically characterized/admitted; never read as a rate.
 * - UNAVAILABLE: data are in process upstream (e.g. producer complete) but are
 *   not delivery-validated and not admitted to this basis.
 * - MISSING: reserved for genuinely absent data or evidence.
 * - OUTSIDE_COVERAGE: the date is outside what this domain build describes.
 */
export type NightState =
  | "AVAILABLE"
  | "ZERO"
  | "UNQUALIFIED"
  | "UNAVAILABLE"
  | "MISSING"
  | "OUTSIDE_COVERAGE";

export type NightRecord = {
  /** UTC date label YYYY-MM-DD, half-open [00:00, 24:00) UTC. */
  date: string;
  state: NightState;
  /** Stable reason code, e.g. `DELIVERY_VALIDATED_CHARACTERIZED`. */
  reason: string;
  window_id: string | null;
  evidence: EvidenceClass[];
};

export type EvidenceFact = {
  label: string;
  value: string | number | boolean | null;
  unit?: string;
  evidence: EvidenceClass;
};

export type DeliveryValidation = "DELIVERY_VALIDATED" | "NOT_DELIVERY_VALIDATED" | "NOT_APPLICABLE";
export type BasisAdmission = "ADMITTED" | "NOT_ADMITTED";

export type TimeWindow = {
  id: string;
  label: string;
  /** Half-open UTC date window [start, stop). */
  start: string;
  stop: string;
  state: NightState;
  /** Upstream acquisition/build state exactly as recorded (e.g. PRODUCER_COMPLETE). */
  source_state: string;
  delivery_validation: DeliveryValidation;
  /** Whether this window's data are admitted to the pinned scientific basis. */
  admission: BasisAdmission;
  /** Stable machine-readable status codes, e.g. NOT_DELIVERY_VALIDATED. */
  status_codes: string[];
  evidence: EvidenceClass[];
  facts: EvidenceFact[];
  /** Whether row totals may be compared with other windows as a rate. */
  rate_comparison: "PROHIBITED" | "PERMITTED";
  caveat: string;
};

export type CountSeries = {
  quantity: string;
  unit: string;
  evidence: EvidenceClass[];
  /** Count per UTC date; only dates whose state is AVAILABLE or ZERO. */
  values: Record<string, number>;
};

export type DomainTimePayload = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  domain: DomainId;
  build_id: string;
  semantics: TimeSemantics;
  /** Half-open UTC date range this lane describes; outside it is OUTSIDE_COVERAGE. */
  range: { start: string; stop: string };
  nights: NightRecord[];
  windows: TimeWindow[];
  counts: CountSeries | null;
};

/* -------------------------------------------------------------------------- */
/* Sky                                                                         */
/* -------------------------------------------------------------------------- */

export type HealpixOrdering = "NESTED";

/** Population density: counts of native entities per equal-area cell. */
export type HealpixCountMap = {
  order: number;
  ordering: HealpixOrdering;
  frame: "ICRS";
  quantity: string;
  unit: string;
  evidence: EvidenceClass[];
  pixels: number[];
  values: number[];
};

/**
 * Coverage/footprint as a normalized multi-order coverage map (MOC): where the
 * source could have produced records. Never density.
 */
export type CoverageMoc = {
  max_order: number;
  ordering: HealpixOrdering;
  frame: "ICRS";
  meaning: string;
  evidence: EvidenceClass[];
  /** NESTED cells keyed by order; no cell is duplicated by an ancestor and no
   *  complete set of four siblings remains uncollapsed. */
  moc: Record<string, number[]>;
};

export type DomainSkyPayload = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  domain: DomainId;
  build_id: string;
  density: HealpixCountMap;
  coverage: CoverageMoc | null;
  coverage_unavailable_reason: string | null;
};

/* -------------------------------------------------------------------------- */
/* Entities                                                                    */
/* -------------------------------------------------------------------------- */

/** Namespaced by domain on purpose: there is no shared entity kind. */
export type EntityKind = "antares.locus" | "fink.diaObject";

export type NativeEntityRef = {
  domain: DomainId;
  kind: EntityKind;
  /** Native identifier as a string; int64 identifiers are never JS numbers. */
  id: string;
};

export type FieldSpec = {
  key: string;
  label: string;
  unit: string | null;
  evidence: EvidenceClass;
  description: string;
};

export type EntitySummaryBase = {
  id: string;
  ra: number;
  dec: number;
  /** UTC date this entity is assigned to by the domain's entity_date_rule. */
  entity_date: string;
};

export type AntaresLocusSummary = EntitySummaryBase & {
  kind: "antares.locus";
  newest_alert_observation_time: number;
  /** Null where the legacy demo exporter clipped the value (true value unknown). */
  brightest_alert_magnitude: number | null;
  num_mag_values: number;
  tags: string[];
};

export type FinkDiaObjectSummary = EntitySummaryBase & {
  kind: "fink.diaObject";
  n_dia_sources: number;
  first_midpoint_mjd_tai: number;
  last_midpoint_mjd_tai: number;
  bands: string[];
};

export type EntitySummary = AntaresLocusSummary | FinkDiaObjectSummary;

export type PopulationDescriptor = {
  /** True when records are the complete population this basis describes. */
  complete: boolean;
  represented: number;
  total: number | null;
  sampling: string;
};

export type DomainEntitiesPayload = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  domain: DomainId;
  build_id: string;
  entity_kind: EntityKind;
  native_label: string;
  id_field: string;
  population: PopulationDescriptor;
  fields: FieldSpec[];
  records: EntitySummary[];
  detail: {
    shard_count: number;
    /** Relative path template, `{shard}` is a two-digit decimal index. */
    path_template: string;
    shard_rule: "fnv1a32(id) mod shard_count";
  };
};

export type LightcurvePoint = { mjd: number; magnitude: number; band: string };

export type AntaresLocusDetail = {
  kind: "antares.locus";
  id: string;
  tags: string[];
  lightcurve: {
    evidence: EvidenceClass;
    time_scale: TimeScale;
    label: string;
    points: LightcurvePoint[];
  } | null;
  lightcurve_unavailable_reason: string | null;
};

export type DiaSourceRow = {
  diaSourceId: string;
  midpointMjdTai: number;
  band: string;
  psfFlux: number;
  psfFluxErr: number;
  snr: number | null;
  reliability: number | null;
};

/** Fink broker fields as delivered with one DiaSource: a source-time snapshot. */
export type BrokerSnapshot = {
  diaSourceId: string;
  midpointMjdTai: number;
  fink_science_version: string;
  pred: { is_sso: boolean; is_first: boolean | null; is_cataloged: boolean | null };
  clf: Record<string, number | null>;
  xm: Record<string, string | number | null>;
  lc_features: Record<string, Record<string, number | null>>;
};

export type FinkDiaObjectDetail = {
  kind: "fink.diaObject";
  id: string;
  sources: DiaSourceRow[];
  snapshots: BrokerSnapshot[];
  snapshot_policy: string;
};

export type EntityDetail = AntaresLocusDetail | FinkDiaObjectDetail;

export type EntityDetailShard = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  domain: DomainId;
  build_id: string;
  shard: number;
  evidence: EvidenceClass[];
  records: Record<string, EntityDetail>;
};

/* -------------------------------------------------------------------------- */
/* Features / population                                                       */
/* -------------------------------------------------------------------------- */

export type FeatureFamily =
  | "position"
  | "time"
  | "multiplicity"
  | "time_baseline"
  | "photometry"
  | "colour"
  | "variability"
  | "model_output"
  | "crossmatch";

export type FeatureDimension = {
  /** Domain-qualified id, e.g. `fink.lc_features.r.chi2`. */
  id: string;
  domain: DomainId;
  family: FeatureFamily;
  label: string;
  short: string;
  unit: string | null;
  scale: "linear" | "log";
  /** Display reversed (magnitudes: brighter is up/right). */
  reversed: boolean;
  /** Fixed axis extent when the quantity has natural bounds. */
  extent: [number, number] | null;
  definition: string;
  /**
   * Semantic identity. Two dimensions may share axes across domains only if
   * their definition_id is identical (e.g. a kernel-derived Galactic latitude).
   */
  definition_id: string;
  state: CapabilityState;
  evidence: EvidenceClass | null;
  qualifications: string[];
  unavailable_reason: string | null;
  /** Snapshot semantics for broker inference outputs. */
  snapshot: string | null;
  /** Model outputs are scores; calibrated is false unless separately established. */
  calibrated: boolean | null;
};

export type DomainFeaturesPayload = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  domain: DomainId;
  build_id: string;
  registry: VersionPin;
  kernel: VersionPin;
  dimensions: FeatureDimension[];
  /** Initial Lab axes for this domain; both must be selectable dimensions. */
  lab_defaults: { x: string; y: string };
  /** Columns aligned with entities.records order; null where undefined. */
  columns: Record<string, Array<number | null>>;
};

/* -------------------------------------------------------------------------- */
/* Provenance and manifest                                                     */
/* -------------------------------------------------------------------------- */

export type ProvenanceSource = {
  id: string;
  label: string;
  repository: string;
  revision: string | null;
  path: string;
  sha256: string | null;
  role: string;
  evidence: EvidenceClass;
};

export type AcquisitionEvidence = {
  acquisition_id: string;
  label: string;
  window: { start: string; stop: string; semantics: string };
  science_profile: string;
  state: string;
  state_at_utc: string;
  delivery: {
    topic: string;
    readable_rows: number;
    parquet_files: number;
    total_bytes: number;
    reconciliation_passed: boolean;
    terminal_lag: number;
    meaning: string;
  } | null;
  characterization: { run_id: string; summary_sha256: string; code_sha: string } | null;
  delivery_validation: DeliveryValidation;
  admission: BasisAdmission;
  scientific_status: "CHARACTERIZED" | "UNCHARACTERIZED" | "NOT_DELIVERY_VALIDATED";
  /** Analytical cohort that admits this acquisition, if any. */
  cohort: string | null;
};

export type ProvenancePayload = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  basis_id: string;
  sources: ProvenanceSource[];
  acquisitions: AcquisitionEvidence[];
  derivations: Array<{ id: string; description: string; kernel: string }>;
  field_evidence: Record<DomainId, Array<{ field: string; evidence: EvidenceClass; note: string }>>;
};

export type BundleFileEntry = {
  path: string;
  bytes: number;
  sha256: string;
  role: string;
};

export type DomainPayloadRefs = {
  time: string;
  sky: string;
  entities: string;
  features: string;
};

export type BundleManifest = {
  contract: typeof OBSERVATORY_BUNDLE_CONTRACT;
  contract_version: string;
  bundle_id: string;
  bundle_class: "FIRST_LIGHT_FIXTURE" | "QUALIFIED";
  science_ready: boolean;
  as_of_utc: string;
  evidence_policy: string;
  generator: {
    name: string;
    version: string;
    deterministic: boolean;
    seed: string | null;
    inputs: Array<{ path: string; sha256: string; role: string }>;
  };
  basis: string;
  capabilities: string;
  provenance: string;
  domains: Record<DomainId, DomainPayloadRefs>;
  files: BundleFileEntry[];
};

/** Everything the workspace needs, loaded and integrity-checked by a reader. */
export type ObservatoryBundle = {
  base: string;
  manifest: BundleManifest;
  manifestSha256: string | null;
  basis: ObservatoryBasis;
  capabilities: CapabilityManifest;
  provenance: ProvenancePayload;
  domains: Record<
    DomainId,
    {
      time: DomainTimePayload;
      sky: DomainSkyPayload;
      entities: DomainEntitiesPayload;
      features: DomainFeaturesPayload;
    }
  >;
  integrity: BundleIntegrity;
};

export type FileIntegrity = {
  path: string;
  expected: string;
  actual: string | null;
  status: "VERIFIED" | "MISMATCH" | "UNVERIFIED";
};

export type BundleIntegrity = {
  method: "sha256" | "unavailable";
  files: FileIntegrity[];
};

/* -------------------------------------------------------------------------- */
/* Scientific state                                                            */
/* -------------------------------------------------------------------------- */

/** Half-open UTC date interval [start, stop). */
export type TimePredicate = { kind: "utc_dates"; start: string; stop: string };

/** Geometric sky predicates. Geometry is broker-independent; ontology is not. */
export type SkyPredicate =
  | { kind: "healpix"; order: number; pixels: number[] }
  | { kind: "cone"; ra: number; dec: number; radius_deg: number };

/** Closed numeric interval on one native dimension; null values never match. */
export type FeatureRange = { dimension: string; min: number; max: number };

export type FeaturePredicate = { x: FeatureRange; y: FeatureRange | null };

/**
 * Shared selection. Time and sky predicates are geometric and apply to every
 * domain through its own declared mapping; feature predicates are native to one
 * domain and are retained (suspended) when that domain is not displayed.
 */
export type SelectionSpec = {
  version: 1;
  time: TimePredicate | null;
  sky: SkyPredicate | null;
  feature: Partial<Record<DomainId, FeaturePredicate>>;
};

export type LabStatistic = "count" | "median" | "mean";

export type LabConfig = {
  x: string;
  y: string;
  statistic: LabStatistic;
  z: string | null;
};

export type PrimaryLens = "sky" | "population";

export type LensState = {
  primary: PrimaryLens;
  lab: Record<DomainId, LabConfig>;
};

export type SkyLayer = "density" | "entities";

export type PresentationState = {
  skyOrder: number;
  skyLayer: SkyLayer;
  overlays: { graticule: boolean; galacticPlane: boolean; ecliptic: boolean };
};

/** Basis + Selection + Focus + Lens + Presentation. */
export type ScientificState = {
  basis_id: string;
  mode: SourceMode;
  selection: SelectionSpec;
  focus: NativeEntityRef | null;
  lens: LensState;
  presentation: PresentationState;
};

/** Reproducible description of one view against one pinned basis. */
export type ViewManifest = {
  kind: "uso.view-manifest";
  version: 1;
  basis: {
    basis_id: string;
    status: ObservatoryBasis["status"];
    science_ready: boolean;
    domains: Record<DomainId, { build_id: string; evidence: EvidenceClass[] }>;
    relation: RelationBuildPin | null;
    semantic_contract: VersionPin;
    feature_registry: VersionPin;
    analysis_kernel: VersionPin;
  };
  bundle: { bundle_id: string; manifest_sha256: string | null; integrity: BundleIntegrity["method"] };
  state: ScientificState;
  evidence_in_view: EvidenceClass[];
  caveats: string[];
};
