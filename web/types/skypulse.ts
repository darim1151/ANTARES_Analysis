export type SourceDataRange = {
  latest_night_utc: string;
  latest_mjd_min: number;
  latest_mjd_max: number;
  historical_mjd_min: number | null;
  historical_mjd_max: number | null;
};

export type ExportMode = "demo" | "rsp_parquet";

export type PublicManifest = {
  schema_version: number;
  export_mode?: ExportMode;
  dataset_name: string;
  generated_at_utc: string;
  selected_night_date?: string;
  data_root_used?: string | null;
  alerts_available?: boolean;
  lightcurve_sample_source?: string;
  source_data_range: SourceDataRange;
  counts: {
    sky_points: number;
    last_night_points: number;
    historical_points: number;
    density_tiles: number;
    top_candidates: number;
    lightcurve_objects: number;
    alert_rows?: number;
    nightly_loci_rows?: number;
    historical_loci_rows_before_selected?: number;
  };
  validation: Record<string, boolean | number | string | null>;
  scientific_caveats: string[];
  source_caveats?: string[];
  source_files?: Record<string, string | null>;
};

export type SummaryMetric = {
  label: string;
  value: string;
  detail: string;
};

export type PublicSummary = {
  schema_version: number;
  export_mode?: ExportMode;
  generated_at_utc: string;
  selected_night_date?: string;
  public_data_note?: string;
  source_data_range: SourceDataRange;
  scientific_caveats: string[];
  promise: string;
  metrics: SummaryMetric[];
  comparison: {
    night_loci: number;
    historical_loci: number;
    new_loci: number;
    overlap_loci: number;
    overlap_fraction_of_night: number;
    alert_rows?: number;
    highlighted_objects?: number;
    density_tiles?: number;
  };
};

export type SkyPoint = {
  id: string;
  locus_id?: string;
  label?: string;
  group?: "last_night" | "historical";
  ra: number;
  dec: number;
  date_utc: string;
  mjd: number;
  newest_alert_observation_time?: number | null;
  brightness_mag: number;
  brightest_alert_magnitude?: number | null;
  obs_count: number;
  num_mag_values?: number | null;
  tags: string[];
  is_last_night: boolean;
  seen_before: boolean;
  has_lightcurve?: boolean;
  is_highlighted?: boolean;
  interest_score: number;
  reason: string;
  public_description?: string;
};

export type SkyPointsFile = {
  schema_version: number;
  export_mode?: ExportMode;
  generated_at_utc: string;
  selected_night_date?: string;
  source_data_range: SourceDataRange;
  scientific_caveats: string[];
  points: SkyPoint[];
};

export type DensityTile = {
  id: string;
  ra_min: number;
  ra_max: number;
  dec_min: number;
  dec_max: number;
  count: number;
  last_night_count: number;
  historical_count: number;
  difference_score?: number;
  difference_note?: string;
};

export type DensityTilesFile = {
  schema_version: number;
  export_mode?: ExportMode;
  generated_at_utc: string;
  selected_night_date?: string;
  source_data_range: SourceDataRange;
  scientific_caveats: string[];
  tiles: DensityTile[];
};

export type TopCandidate = {
  id: string;
  locus_id?: string;
  label?: string;
  rank: number;
  ra: number;
  dec: number;
  brightness_mag: number;
  brightest_alert_magnitude?: number | null;
  num_mag_values?: number | null;
  obs_count: number;
  score?: number;
  reason: string;
  public_summary: string;
  caveat?: string;
};

export type TopCandidatesFile = {
  schema_version: number;
  export_mode?: ExportMode;
  generated_at_utc: string;
  selected_night_date?: string;
  public_label?: string;
  ranking_note?: string;
  source_data_range: SourceDataRange;
  scientific_caveats: string[];
  candidates: TopCandidate[];
};

export type LightcurveSample = {
  mjd: number;
  magnitude: number;
  filter: "g" | "r" | "i" | string;
  source?: string;
};

export type LightcurveSamplesFile = {
  schema_version: number;
  export_mode?: ExportMode;
  generated_at_utc: string;
  selected_night_date?: string;
  public_label?: string;
  sample_source?: "synthetic_demo" | "alerts_parquet" | "unavailable" | string;
  source_columns?: Record<string, string | null>;
  source_data_range: SourceDataRange;
  scientific_caveats: string[];
  lightcurves: Record<string, LightcurveSample[]>;
  unavailable?: Record<string, string>;
};
