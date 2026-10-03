#!/usr/bin/env node

import { readdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const argumentsReceived = process.argv.slice(2);
let productionMode = false;
let dataDirectoryOverride = null;
const unknownArguments = [];

for (const argument of argumentsReceived) {
  if (argument === "--production") {
    productionMode = true;
  } else if (argument.startsWith("--data-dir=")) {
    const value = argument.slice("--data-dir=".length);
    if (!value || dataDirectoryOverride !== null) {
      unknownArguments.push(argument);
    } else {
      dataDirectoryOverride = value;
    }
  } else {
    unknownArguments.push(argument);
  }
}

if (unknownArguments.length > 0) {
  console.error(`Unknown or invalid argument(s): ${unknownArguments.join(", ")}`);
  process.exit(2);
}

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const dataDirectory = dataDirectoryOverride
  ? path.resolve(dataDirectoryOverride)
  : path.resolve(scriptDirectory, "..", "public", "data");
const requiredFiles = Object.freeze({
  manifest: "public_manifest.json",
  summary: "public_summary.json",
  skyPoints: "sky_points.json",
  densityTiles: "density_tiles.json",
  topCandidates: "top_candidates.json",
  lightcurveSamples: "lightcurve_samples.json"
});
const supportedSchemaVersion = 2;
const supportedExportModes = new Set(["demo", "rsp_parquet"]);
const errors = [];

function reportError(location, message) {
  errors.push(`${location}: ${message}`);
}

function assert(condition, location, message) {
  if (!condition) {
    reportError(location, message);
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requireRecord(value, location) {
  if (!isRecord(value)) {
    reportError(location, "must be a JSON object");
    return {};
  }
  return value;
}

function requireArray(value, location) {
  if (!Array.isArray(value)) {
    reportError(location, "must be an array");
    return [];
  }
  return value;
}

function requireNonEmptyString(value, location) {
  if (typeof value !== "string" || value.trim().length === 0) {
    reportError(location, "must be a non-empty string");
    return "";
  }
  return value;
}

function requireFiniteNumber(value, location, minimum, maximum) {
  if (!Number.isFinite(value)) {
    reportError(location, "must be a finite number");
    return 0;
  }
  if (minimum !== undefined && value < minimum) {
    reportError(location, `must be at least ${minimum}`);
  }
  if (maximum !== undefined && value > maximum) {
    reportError(location, `must be at most ${maximum}`);
  }
  return value;
}

function requireNonNegativeInteger(value, location) {
  if (!Number.isInteger(value) || value < 0) {
    reportError(location, "must be a non-negative integer");
    return 0;
  }
  return value;
}

function requireBoolean(value, location) {
  if (typeof value !== "boolean") {
    reportError(location, "must be a boolean");
    return false;
  }
  return value;
}

function requireDate(value, location) {
  const matches = typeof value === "string" && /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!matches) {
    reportError(location, "must be a real calendar date using YYYY-MM-DD");
    return;
  }

  const [, year, month, day] = matches;
  const parsed = new Date(Date.UTC(Number(year), Number(month) - 1, Number(day)));
  assert(
    Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value,
    location,
    "must be a real calendar date using YYYY-MM-DD"
  );
}

function mjdToUtcDate(mjd) {
  const milliseconds = (mjd - 40587) * 86_400_000;
  const parsed = new Date(milliseconds);
  return Number.isFinite(parsed.getTime()) ? parsed.toISOString().slice(0, 10) : null;
}

function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])])
    );
  }
  return value;
}

function sameJsonValue(left, right) {
  return JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right));
}

function scanForPrivateMaterial(value, location) {
  if (typeof value === "string") {
    // Reject every absolute POSIX path, including RSP/Arnor prefixes such as
    // /sdf, /project, and /scratch.  The negative lookahead prevents the two
    // slashes in https:// URLs from being mistaken for a filesystem path.
    const hasPrivateUnixPath = /(?:^|[\s("'=:\[])\/(?!\/)[A-Za-z0-9._-]+(?:\/[^\s"'<>\]]*)?/i.test(
      value
    );
    const hasWindowsPath = /(?:^|[\s("'=])[a-z]:[\\/]/i.test(value);
    const hasWindowsUncPath = /(?:^|[\s("'=])\\\\[^\\\s]+\\[^\s]+/i.test(value);
    const hasFileUri = /file:\/\//i.test(value);

    if (hasPrivateUnixPath || hasWindowsPath || hasWindowsUncPath || hasFileUri) {
      reportError(location, "contains a private or host-local filesystem path");
    }
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item, index) =>
      scanForPrivateMaterial(item, `${location}[${index}]`)
    );
    return;
  }

  if (!isRecord(value)) {
    return;
  }

  for (const [key, item] of Object.entries(value)) {
    if (
      /(?:^|_)(?:password|passwd|secret|api[_-]?key|access[_-]?token|token|authorization|cookie|credential|private[_-]?key|client[_-]?secret|session[_-]?id|ssh[_-]?key)(?:_|$)/i.test(
        key
      )
    ) {
      reportError(`${location}.${key}`, "uses a secret-bearing field name");
    }
    scanForPrivateMaterial(item, `${location}.${key}`);
  }
}

async function loadDocuments() {
  const documents = {};
  const expectedNames = new Set(Object.values(requiredFiles));
  try {
    const entries = await readdir(dataDirectory, { withFileTypes: true });
    for (const entry of entries) {
      if (!expectedNames.has(entry.name)) {
        reportError(entry.name, "unexpected entry in the public-data bundle");
      } else if (!entry.isFile()) {
        reportError(entry.name, "must be a regular file, not a symlink or directory");
      }
    }
  } catch (error) {
    reportError(dataDirectory, `cannot list public-data directory (${error.message})`);
  }
  for (const [name, fileName] of Object.entries(requiredFiles)) {
    const filePath = path.join(dataDirectory, fileName);
    try {
      documents[name] = JSON.parse(await readFile(filePath, "utf8"));
    } catch (error) {
      reportError(fileName, `cannot be read as JSON (${error.message})`);
    }
  }
  return documents;
}

function exitWithErrors() {
  if (errors.length === 0) {
    return;
  }
  console.error("SkyPulse public-data validation failed:");
  errors.forEach((error) => console.error(`- ${error}`));
  process.exit(1);
}

const documents = await loadDocuments();
exitWithErrors();

const manifest = requireRecord(documents.manifest, requiredFiles.manifest);
const summary = requireRecord(documents.summary, requiredFiles.summary);
const skyPointsDocument = requireRecord(documents.skyPoints, requiredFiles.skyPoints);
const densityTilesDocument = requireRecord(
  documents.densityTiles,
  requiredFiles.densityTiles
);
const topCandidatesDocument = requireRecord(
  documents.topCandidates,
  requiredFiles.topCandidates
);
const lightcurveSamplesDocument = requireRecord(
  documents.lightcurveSamples,
  requiredFiles.lightcurveSamples
);

const exportMode = manifest.export_mode;
const generatedAt = manifest.generated_at_utc;
const selectedNight = manifest.selected_night_date;
const manifestCaveats = requireArray(
  manifest.scientific_caveats,
  `${requiredFiles.manifest}.scientific_caveats`
);
const sourceDataRange = requireRecord(
  manifest.source_data_range,
  `${requiredFiles.manifest}.source_data_range`
);

assert(
  supportedExportModes.has(exportMode),
  `${requiredFiles.manifest}.export_mode`,
  `must be one of ${[...supportedExportModes].join(", ")}`
);
assert(
  !productionMode || exportMode === "rsp_parquet",
  `${requiredFiles.manifest}.export_mode`,
  "production data must use the rsp_parquet export mode; demo data cannot pass"
);
assert(
  typeof generatedAt === "string" &&
    Number.isFinite(Date.parse(generatedAt)) &&
    /(?:Z|[+-]\d{2}:\d{2})$/.test(generatedAt),
  `${requiredFiles.manifest}.generated_at_utc`,
  "must be a timezone-qualified ISO-8601 timestamp"
);
requireDate(selectedNight, `${requiredFiles.manifest}.selected_night_date`);
requireDate(
  sourceDataRange.latest_night_utc,
  `${requiredFiles.manifest}.source_data_range.latest_night_utc`
);
assert(
  sourceDataRange.latest_night_utc === selectedNight,
  `${requiredFiles.manifest}.source_data_range.latest_night_utc`,
  "must match selected_night_date"
);
const latestMjdMin = requireFiniteNumber(
  sourceDataRange.latest_mjd_min,
  `${requiredFiles.manifest}.source_data_range.latest_mjd_min`
);
const latestMjdMax = requireFiniteNumber(
  sourceDataRange.latest_mjd_max,
  `${requiredFiles.manifest}.source_data_range.latest_mjd_max`
);
assert(
  sourceDataRange.latest_night_utc === mjdToUtcDate(latestMjdMin),
  `${requiredFiles.manifest}.source_data_range.latest_night_utc`,
  "must match the UTC calendar date containing latest_mjd_min"
);
assert(
  latestMjdMin <= latestMjdMax,
  `${requiredFiles.manifest}.source_data_range`,
  "latest MJD minimum must not exceed maximum"
);
const historicalMjdMin = sourceDataRange.historical_mjd_min;
const historicalMjdMax = sourceDataRange.historical_mjd_max;
assert(
  (historicalMjdMin === null) === (historicalMjdMax === null),
  `${requiredFiles.manifest}.source_data_range`,
  "historical MJD bounds must either both be null or both be finite"
);
if (historicalMjdMin !== null && historicalMjdMax !== null) {
  requireFiniteNumber(
    historicalMjdMin,
    `${requiredFiles.manifest}.source_data_range.historical_mjd_min`
  );
  requireFiniteNumber(
    historicalMjdMax,
    `${requiredFiles.manifest}.source_data_range.historical_mjd_max`
  );
  assert(
    historicalMjdMin <= historicalMjdMax,
    `${requiredFiles.manifest}.source_data_range`,
    "historical MJD minimum must not exceed maximum"
  );
  assert(
    historicalMjdMax <= latestMjdMin,
    `${requiredFiles.manifest}.source_data_range`,
    "historical MJD maximum must not exceed the latest-night minimum"
  );
}

for (const [name, document] of Object.entries(documents)) {
  const fileName = requiredFiles[name];
  const record = requireRecord(document, fileName);
  assert(
    record.schema_version === supportedSchemaVersion,
    `${fileName}.schema_version`,
    `must equal ${supportedSchemaVersion}`
  );
  assert(record.export_mode === exportMode, `${fileName}.export_mode`, "must match manifest");
  assert(
    record.generated_at_utc === generatedAt,
    `${fileName}.generated_at_utc`,
    "must match manifest"
  );
  assert(
    record.selected_night_date === selectedNight,
    `${fileName}.selected_night_date`,
    "must match manifest"
  );
  assert(
    sameJsonValue(record.source_data_range, sourceDataRange),
    `${fileName}.source_data_range`,
    "must structurally match manifest"
  );
  const caveats = requireArray(record.scientific_caveats, `${fileName}.scientific_caveats`);
  assert(caveats.length > 0, `${fileName}.scientific_caveats`, "must not be empty");
  caveats.forEach((value, index) =>
    requireNonEmptyString(value, `${fileName}.scientific_caveats[${index}]`)
  );
  assert(
    sameJsonValue(caveats, manifestCaveats),
    `${fileName}.scientific_caveats`,
    "must structurally match the manifest caveats"
  );
  scanForPrivateMaterial(record, fileName);
}

const counts = requireRecord(manifest.counts, `${requiredFiles.manifest}.counts`);
const points = requireArray(skyPointsDocument.points, `${requiredFiles.skyPoints}.points`);
const tiles = requireArray(densityTilesDocument.tiles, `${requiredFiles.densityTiles}.tiles`);
const candidates = requireArray(
  topCandidatesDocument.candidates,
  `${requiredFiles.topCandidates}.candidates`
);
const lightcurves = requireRecord(
  lightcurveSamplesDocument.lightcurves,
  `${requiredFiles.lightcurveSamples}.lightcurves`
);
const unavailableLightcurves = requireRecord(
  lightcurveSamplesDocument.unavailable ?? {},
  `${requiredFiles.lightcurveSamples}.unavailable`
);

const expectedCounts = {
  sky_points: points.length,
  last_night_points: points.filter((point) => point?.is_last_night === true).length,
  historical_points: points.filter((point) => point?.is_last_night !== true).length,
  density_tiles: tiles.length,
  top_candidates: candidates.length,
  lightcurve_objects: Object.keys(lightcurves).length
};

for (const [name, expected] of Object.entries(expectedCounts)) {
  const actual = requireNonNegativeInteger(
    counts[name],
    `${requiredFiles.manifest}.counts.${name}`
  );
  assert(
    actual === expected,
    `${requiredFiles.manifest}.counts.${name}`,
    `declares ${actual}, but the data contains ${expected}`
  );
}

const pointIds = new Set();
const pointById = new Map();
for (const [index, rawPoint] of points.entries()) {
  const location = `${requiredFiles.skyPoints}.points[${index}]`;
  const point = requireRecord(rawPoint, location);
  const identifier = requireNonEmptyString(point.id, `${location}.id`);
  if (identifier) {
    assert(!pointIds.has(identifier), `${location}.id`, "must be unique");
    pointIds.add(identifier);
    pointById.set(identifier, point);
  }
  const ra = requireFiniteNumber(point.ra, `${location}.ra`, 0, 360);
  assert(ra < 360, `${location}.ra`, "must be less than 360 degrees");
  requireFiniteNumber(point.dec, `${location}.dec`, -90, 90);
  requireDate(point.date_utc, `${location}.date_utc`);
  const mjd = requireFiniteNumber(point.mjd, `${location}.mjd`);
  requireFiniteNumber(point.brightness_mag, `${location}.brightness_mag`);
  requireNonNegativeInteger(point.obs_count, `${location}.obs_count`);
  requireFiniteNumber(point.interest_score, `${location}.interest_score`);
  requireNonEmptyString(point.reason, `${location}.reason`);
  const isLastNight = requireBoolean(point.is_last_night, `${location}.is_last_night`);
  requireBoolean(point.seen_before, `${location}.seen_before`);
  const tags = requireArray(point.tags, `${location}.tags`);
  tags.forEach((tag, tagIndex) =>
    requireNonEmptyString(tag, `${location}.tags[${tagIndex}]`)
  );
  if (point.group !== undefined) {
    assert(
      point.group === "last_night" || point.group === "historical",
      `${location}.group`,
      "must be last_night or historical"
    );
    assert(
      (point.group === "last_night") === isLastNight,
      `${location}.group`,
      "must agree with is_last_night"
    );
  }
  if (isLastNight) {
    assert(
      point.date_utc === selectedNight,
      `${location}.date_utc`,
      "last-night points must use selected_night_date"
    );
    assert(
      mjd >= latestMjdMin && mjd <= latestMjdMax,
      `${location}.mjd`,
      "last-night points must fall inside the latest MJD range"
    );
  } else {
    assert(
      typeof point.date_utc === "string" && point.date_utc < selectedNight,
      `${location}.date_utc`,
      "historical points must predate selected_night_date"
    );
    assert(
      historicalMjdMin !== null &&
        historicalMjdMax !== null &&
        mjd >= historicalMjdMin &&
        mjd <= historicalMjdMax,
      `${location}.mjd`,
      "historical points must fall inside the historical MJD range"
    );
  }
}

const candidateIds = new Set();
const candidateRanks = new Set();
for (const [index, rawCandidate] of candidates.entries()) {
  const location = `${requiredFiles.topCandidates}.candidates[${index}]`;
  const candidate = requireRecord(rawCandidate, location);
  const identifier = requireNonEmptyString(candidate.id, `${location}.id`);
  if (identifier) {
    assert(!candidateIds.has(identifier), `${location}.id`, "must be unique");
    assert(pointIds.has(identifier), `${location}.id`, "must refer to a sky point");
    candidateIds.add(identifier);
  }
  const rank = requireNonNegativeInteger(candidate.rank, `${location}.rank`);
  assert(rank > 0, `${location}.rank`, "must be a positive integer");
  assert(!candidateRanks.has(rank), `${location}.rank`, "must be unique");
  candidateRanks.add(rank);
  const candidateRa = requireFiniteNumber(candidate.ra, `${location}.ra`, 0, 360);
  const candidateDec = requireFiniteNumber(candidate.dec, `${location}.dec`, -90, 90);
  const candidateBrightness = requireFiniteNumber(
    candidate.brightness_mag,
    `${location}.brightness_mag`
  );
  const candidateObservations = requireNonNegativeInteger(
    candidate.obs_count,
    `${location}.obs_count`
  );
  requireNonEmptyString(candidate.reason, `${location}.reason`);
  requireNonEmptyString(candidate.public_summary, `${location}.public_summary`);
  const sourcePoint = pointById.get(identifier);
  if (sourcePoint) {
    assert(candidateRa === sourcePoint.ra, `${location}.ra`, "must match the sky point");
    assert(candidateDec === sourcePoint.dec, `${location}.dec`, "must match the sky point");
    assert(
      candidateBrightness === sourcePoint.brightness_mag,
      `${location}.brightness_mag`,
      "must match the sky point"
    );
    assert(
      candidateObservations === sourcePoint.obs_count,
      `${location}.obs_count`,
      "must match the sky point"
    );
    assert(candidate.reason === sourcePoint.reason, `${location}.reason`, "must match the sky point");
  }
}
if (candidateRanks.size > 0) {
  assert(
    Math.min(...candidateRanks) === 1 && Math.max(...candidateRanks) === candidateRanks.size,
    `${requiredFiles.topCandidates}.candidates.rank`,
    "ranks must be contiguous from 1"
  );
}

for (const [identifier, rawSamples] of Object.entries(lightcurves)) {
  assert(
    candidateIds.has(identifier),
    `${requiredFiles.lightcurveSamples}.lightcurves.${identifier}`,
    "must refer to a top candidate"
  );
  const samples = requireArray(
    rawSamples,
    `${requiredFiles.lightcurveSamples}.lightcurves.${identifier}`
  );
  assert(
    samples.length > 0,
    `${requiredFiles.lightcurveSamples}.lightcurves.${identifier}`,
    "must contain at least one sample"
  );
  let previousMjd = -Infinity;
  samples.forEach((rawSample, sampleIndex) => {
    const location = `${requiredFiles.lightcurveSamples}.lightcurves.${identifier}[${sampleIndex}]`;
    const sample = requireRecord(rawSample, location);
    const mjd = requireFiniteNumber(sample.mjd, `${location}.mjd`);
    requireFiniteNumber(sample.magnitude, `${location}.magnitude`);
    requireNonEmptyString(sample.filter, `${location}.filter`);
    assert(
      mjd <= latestMjdMax,
      `${location}.mjd`,
      "must not be later than the exported latest MJD range"
    );
    assert(
      sample.source === lightcurveSamplesDocument.sample_source,
      `${location}.source`,
      "must match the document sample_source provenance"
    );
    assert(mjd >= previousMjd, `${location}.mjd`, "samples must be sorted by MJD");
    previousMjd = mjd;
  });
}
for (const [identifier, reason] of Object.entries(unavailableLightcurves)) {
  assert(
    candidateIds.has(identifier),
    `${requiredFiles.lightcurveSamples}.unavailable.${identifier}`,
    "must refer to a top candidate"
  );
  requireNonEmptyString(
    reason,
    `${requiredFiles.lightcurveSamples}.unavailable.${identifier}`
  );
  assert(
    !Object.hasOwn(lightcurves, identifier),
    `${requiredFiles.lightcurveSamples}.unavailable.${identifier}`,
    "cannot also have lightcurve samples"
  );
}
for (const identifier of candidateIds) {
  const hasSamples = Object.hasOwn(lightcurves, identifier);
  const isUnavailable = Object.hasOwn(unavailableLightcurves, identifier);
  assert(
    hasSamples !== isUnavailable,
    `${requiredFiles.lightcurveSamples}.${identifier}`,
    "every candidate must have exactly one lightcurve or unavailability reason"
  );
}

const tileIds = new Set();
const densityTotals = { all: 0, latest: 0, historical: 0 };
const normalizedTiles = [];
for (const [index, rawTile] of tiles.entries()) {
  const location = `${requiredFiles.densityTiles}.tiles[${index}]`;
  const tile = requireRecord(rawTile, location);
  const identifier = requireNonEmptyString(tile.id, `${location}.id`);
  if (identifier) {
    assert(!tileIds.has(identifier), `${location}.id`, "must be unique");
    tileIds.add(identifier);
  }
  const raMin = requireFiniteNumber(tile.ra_min, `${location}.ra_min`, 0, 360);
  const raMax = requireFiniteNumber(tile.ra_max, `${location}.ra_max`, 0, 360);
  const decMin = requireFiniteNumber(tile.dec_min, `${location}.dec_min`, -90, 90);
  const decMax = requireFiniteNumber(tile.dec_max, `${location}.dec_max`, -90, 90);
  assert(raMin < raMax, `${location}.ra_min`, "must be less than ra_max");
  assert(decMin < decMax, `${location}.dec_min`, "must be less than dec_max");
  const count = requireNonNegativeInteger(tile.count, `${location}.count`);
  const latest = requireNonNegativeInteger(
    tile.last_night_count,
    `${location}.last_night_count`
  );
  const historical = requireNonNegativeInteger(
    tile.historical_count,
    `${location}.historical_count`
  );
  assert(
    latest + historical === count,
    `${location}.count`,
    "must equal last_night_count plus historical_count"
  );
  densityTotals.all += count;
  densityTotals.latest += latest;
  densityTotals.historical += historical;
  normalizedTiles.push({
    identifier,
    location,
    raMin,
    raMax,
    decMin,
    decMax,
    count,
    latest,
    historical
  });
}

const recomputedTileCounts = new Map(
  normalizedTiles.map((tile) => [tile.identifier, { all: 0, latest: 0, historical: 0 }])
);
for (const [index, point] of points.entries()) {
  const matchingTiles = normalizedTiles.filter(
    (tile) =>
      point.ra >= tile.raMin &&
      point.ra < tile.raMax &&
      point.dec >= tile.decMin &&
      (point.dec < tile.decMax || (tile.decMax === 90 && point.dec === 90))
  );
  assert(
    matchingTiles.length === 1,
    `${requiredFiles.skyPoints}.points[${index}]`,
    `must belong to exactly one density tile (found ${matchingTiles.length})`
  );
  if (matchingTiles.length === 1) {
    const observed = recomputedTileCounts.get(matchingTiles[0].identifier);
    observed.all += 1;
    if (point.is_last_night === true) {
      observed.latest += 1;
    } else {
      observed.historical += 1;
    }
  }
}
for (const tile of normalizedTiles) {
  const observed = recomputedTileCounts.get(tile.identifier);
  assert(observed.all === tile.count, `${tile.location}.count`, "must match contained points");
  assert(
    observed.latest === tile.latest,
    `${tile.location}.last_night_count`,
    "must match contained last-night points"
  );
  assert(
    observed.historical === tile.historical,
    `${tile.location}.historical_count`,
    "must match contained historical points"
  );
}

assert(
  densityTotals.all === expectedCounts.sky_points,
  `${requiredFiles.densityTiles}.tiles`,
  "tile counts must sum to the number of sky points"
);
assert(
  densityTotals.latest === expectedCounts.last_night_points,
  `${requiredFiles.densityTiles}.tiles`,
  "last-night tile counts must sum to the number of last-night points"
);
assert(
  densityTotals.historical === expectedCounts.historical_points,
  `${requiredFiles.densityTiles}.tiles`,
  "historical tile counts must sum to the number of historical points"
);

requireNonEmptyString(summary.promise, `${requiredFiles.summary}.promise`);
const metrics = requireArray(summary.metrics, `${requiredFiles.summary}.metrics`);
metrics.forEach((rawMetric, index) => {
  const location = `${requiredFiles.summary}.metrics[${index}]`;
  const metric = requireRecord(rawMetric, location);
  requireNonEmptyString(metric.label, `${location}.label`);
  requireNonEmptyString(metric.value, `${location}.value`);
  requireNonEmptyString(metric.detail, `${location}.detail`);
});
const summaryComparison = requireRecord(
  summary.comparison,
  `${requiredFiles.summary}.comparison`
);
const nightLoci = requireNonNegativeInteger(
  summaryComparison.night_loci,
  `${requiredFiles.summary}.comparison.night_loci`
);
const historicalLoci = requireNonNegativeInteger(
  summaryComparison.historical_loci,
  `${requiredFiles.summary}.comparison.historical_loci`
);
const newLoci = requireNonNegativeInteger(
  summaryComparison.new_loci,
  `${requiredFiles.summary}.comparison.new_loci`
);
const overlapLoci = requireNonNegativeInteger(
  summaryComparison.overlap_loci,
  `${requiredFiles.summary}.comparison.overlap_loci`
);
const overlapFraction = requireFiniteNumber(
  summaryComparison.overlap_fraction_of_night,
  `${requiredFiles.summary}.comparison.overlap_fraction_of_night`,
  0,
  1
);
assert(
  nightLoci === expectedCounts.last_night_points,
  `${requiredFiles.summary}.comparison.night_loci`,
  "must match the last-night point count"
);
assert(
  historicalLoci === expectedCounts.historical_points,
  `${requiredFiles.summary}.comparison.historical_loci`,
  "must match the historical point count"
);
const observedOverlap = points.filter(
  (point) => point?.is_last_night === true && point?.seen_before === true
).length;
assert(
  newLoci + overlapLoci === nightLoci,
  `${requiredFiles.summary}.comparison`,
  "new_loci plus overlap_loci must equal night_loci"
);
assert(
  overlapLoci === observedOverlap,
  `${requiredFiles.summary}.comparison.overlap_loci`,
  "must match last-night points marked seen_before"
);
const expectedOverlapFraction = nightLoci === 0 ? 0 : overlapLoci / nightLoci;
assert(
  Math.abs(overlapFraction - expectedOverlapFraction) <= 1e-12,
  `${requiredFiles.summary}.comparison.overlap_fraction_of_night`,
  "must equal overlap_loci divided by night_loci"
);
const alertRows = requireNonNegativeInteger(
  counts.alert_rows,
  `${requiredFiles.manifest}.counts.alert_rows`
);
const comparisonAlertRows = requireNonNegativeInteger(
  summaryComparison.alert_rows,
  `${requiredFiles.summary}.comparison.alert_rows`
);
const highlightedObjects = requireNonNegativeInteger(
  summaryComparison.highlighted_objects,
  `${requiredFiles.summary}.comparison.highlighted_objects`
);
const comparisonDensityTiles = requireNonNegativeInteger(
  summaryComparison.density_tiles,
  `${requiredFiles.summary}.comparison.density_tiles`
);
assert(
  comparisonAlertRows === alertRows,
  `${requiredFiles.summary}.comparison.alert_rows`,
  "must match the manifest alert-row count"
);
assert(
  highlightedObjects === expectedCounts.top_candidates,
  `${requiredFiles.summary}.comparison.highlighted_objects`,
  "must match the candidate count"
);
assert(
  comparisonDensityTiles === expectedCounts.density_tiles,
  `${requiredFiles.summary}.comparison.density_tiles`,
  "must match the density-tile count"
);

const manifestValidation = requireRecord(
  manifest.validation,
  `${requiredFiles.manifest}.validation`
);
for (const name of [
  "json_serializable",
  "ra_dec_bounds_pass",
  "top_candidates_in_sky_points",
  "lightcurves_refer_to_candidates"
]) {
  assert(
    manifestValidation[name] === true,
    `${requiredFiles.manifest}.validation.${name}`,
    "must be true"
  );
}
assert(
  manifestValidation.bad_coordinate_count === 0,
  `${requiredFiles.manifest}.validation.bad_coordinate_count`,
  "must be zero"
);
assert(
  manifestValidation.duplicate_sky_point_id_count === 0,
  `${requiredFiles.manifest}.validation.duplicate_sky_point_id_count`,
  "must be zero"
);

const nightlyValidation = requireRecord(
  manifest.nightly_manifest_validation,
  `${requiredFiles.manifest}.nightly_manifest_validation`
);
for (const name of [
  "append_ready",
  "coordinate_pass",
  "mjd_pass",
  "lsst_only_pass",
  "history_start_pass",
  "alert_locus_link_pass"
]) {
  assert(
    nightlyValidation[name] === true,
    `${requiredFiles.manifest}.nightly_manifest_validation.${name}`,
    "must be true"
  );
}

const sourceCaveats = requireArray(
  manifest.source_caveats,
  `${requiredFiles.manifest}.source_caveats`
);
assert(
  sameJsonValue(sourceCaveats, manifestCaveats),
  `${requiredFiles.manifest}.source_caveats`,
  "must structurally match scientific_caveats"
);
const alertsAvailable = requireBoolean(
  manifest.alerts_available,
  `${requiredFiles.manifest}.alerts_available`
);
assert(
  alertsAvailable === (alertRows > 0),
  `${requiredFiles.manifest}.alerts_available`,
  "must agree with whether alert_rows is positive"
);
const manifestLightcurveSource = requireNonEmptyString(
  manifest.lightcurve_sample_source,
  `${requiredFiles.manifest}.lightcurve_sample_source`
);
const documentLightcurveSource = requireNonEmptyString(
  lightcurveSamplesDocument.sample_source,
  `${requiredFiles.lightcurveSamples}.sample_source`
);
assert(
  manifestLightcurveSource === documentLightcurveSource,
  `${requiredFiles.lightcurveSamples}.sample_source`,
  "must match the manifest lightcurve source"
);

if (exportMode === "demo") {
  assert(
    manifest.data_root_used === null,
    `${requiredFiles.manifest}.data_root_used`,
    "must be null for a repository demo"
  );
  assert(
    alertsAvailable === false,
    `${requiredFiles.manifest}.alerts_available`,
    "must be false for the bundled synthetic demo"
  );
  assert(
    manifestLightcurveSource === "synthetic_demo" &&
      documentLightcurveSource === "synthetic_demo",
    `${requiredFiles.lightcurveSamples}.sample_source`,
    "must identify synthetic demo lightcurves"
  );
}

if (exportMode === "rsp_parquet") {
  assert(
    manifest.dataset_name === "SkyPulse RSP Parquet Export",
    `${requiredFiles.manifest}.dataset_name`,
    "must identify the RSP parquet export"
  );
  requireNonEmptyString(
    manifest.data_root_used,
    `${requiredFiles.manifest}.data_root_used`
  );
  const sourceTypeSummary = requireNonEmptyString(
    manifest.source_type_summary,
    `${requiredFiles.manifest}.source_type_summary`
  );
  assert(
    /RSP parquet/i.test(sourceTypeSummary) && !/(?:demo|synthetic)/i.test(sourceTypeSummary),
    `${requiredFiles.manifest}.source_type_summary`,
    "must identify saved RSP parquet data without demo provenance"
  );
  const sourceFiles = requireRecord(
    manifest.source_files,
    `${requiredFiles.manifest}.source_files`
  );
  for (const name of [
    "nightly_manifest",
    "nightly_loci",
    "cumulative_loci_index",
    "cumulative_nightly_summary"
  ]) {
    const source = requireNonEmptyString(
      sourceFiles[name],
      `${requiredFiles.manifest}.source_files.${name}`
    );
    assert(
      !/(?:demo|synthetic):?\/\//i.test(source),
      `${requiredFiles.manifest}.source_files.${name}`,
      "must not use demo or synthetic provenance"
    );
  }
  if (alertsAvailable) {
    requireNonEmptyString(
      sourceFiles.nightly_alerts,
      `${requiredFiles.manifest}.source_files.nightly_alerts`
    );
  }

  const expectedLightcurveSource = expectedCounts.lightcurve_objects > 0
    ? "alerts_parquet"
    : "unavailable";
  assert(
    manifestLightcurveSource === expectedLightcurveSource,
    `${requiredFiles.manifest}.lightcurve_sample_source`,
    `must be ${expectedLightcurveSource} for this bundle`
  );
  if (manifestLightcurveSource === "alerts_parquet") {
    assert(
      alertsAvailable,
      `${requiredFiles.manifest}.lightcurve_sample_source`,
      "alert-record lightcurves require available alert rows"
    );
    const sourceColumns = requireRecord(
      lightcurveSamplesDocument.source_columns,
      `${requiredFiles.lightcurveSamples}.source_columns`
    );
    assert(
      sameJsonValue(sourceColumns, manifest.alert_source_columns),
      `${requiredFiles.lightcurveSamples}.source_columns`,
      "must match the manifest alert source columns"
    );
    for (const name of ["magnitude", "magnitude_error", "time", "filter"]) {
      assert(
        Object.hasOwn(sourceColumns, name),
        `${requiredFiles.lightcurveSamples}.source_columns.${name}`,
        "must be declared"
      );
      assert(
        sourceColumns[name] === null ||
          (typeof sourceColumns[name] === "string" && sourceColumns[name].trim().length > 0),
        `${requiredFiles.lightcurveSamples}.source_columns.${name}`,
        "must be a source-column name or null"
      );
    }
    requireNonEmptyString(
      sourceColumns.magnitude,
      `${requiredFiles.lightcurveSamples}.source_columns.magnitude`
    );
    requireNonEmptyString(
      sourceColumns.time,
      `${requiredFiles.lightcurveSamples}.source_columns.time`
    );
  }
  assert(
    !/(?:demo|synthetic)/i.test(
      [
        manifest.dataset_name,
        manifest.data_root_used,
        manifestLightcurveSource,
        documentLightcurveSource,
        ...sourceCaveats
      ].join(" ")
    ),
    `${requiredFiles.manifest}`,
    "production provenance must not contain demo or synthetic labels"
  );
}

exitWithErrors();
console.log(
  `Validated ${Object.keys(requiredFiles).length} SkyPulse public-data files ` +
    `(schema ${supportedSchemaVersion}, mode ${exportMode}, ` +
    `${expectedCounts.sky_points.toLocaleString("en-US")} points).`
);
