#!/usr/bin/env node

import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const validatorPath = path.join(scriptDirectory, "validate-public-data.mjs");
const sourceDataDirectory = path.resolve(scriptDirectory, "..", "public", "data");

function runValidator(dataDirectory, extraArguments = []) {
  return spawnSync(
    process.execPath,
    [validatorPath, `--data-dir=${dataDirectory}`, ...extraArguments],
    { encoding: "utf8" }
  );
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function readJson(dataDirectory, fileName) {
  return JSON.parse(await readFile(path.join(dataDirectory, fileName), "utf8"));
}

async function writeJson(dataDirectory, fileName, value) {
  await writeFile(
    path.join(dataDirectory, fileName),
    `${JSON.stringify(value, null, 2)}\n`,
    "utf8"
  );
}

async function withFixture(callback) {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "skypulse-validator-"));
  const dataDirectory = path.join(temporaryRoot, "data");
  try {
    await cp(sourceDataDirectory, dataDirectory, { recursive: true });
    await callback(dataDirectory);
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
}

await withFixture(async (dataDirectory) => {
  const completed = runValidator(dataDirectory);
  assert(completed.status === 0, completed.stderr || completed.stdout);

  const production = runValidator(dataDirectory, ["--production"]);
  assert(production.status === 1, "demo data unexpectedly passed production mode");
  assert(
    production.stderr.includes("demo data cannot pass"),
    "production rejection did not identify demo mode"
  );
});

await withFixture(async (dataDirectory) => {
  for (const fileName of [
    "public_manifest.json",
    "public_summary.json",
    "sky_points.json",
    "density_tiles.json",
    "top_candidates.json",
    "lightcurve_samples.json"
  ]) {
    const document = await readJson(dataDirectory, fileName);
    document.export_mode = "rsp_parquet";
    await writeJson(dataDirectory, fileName, document);
  }
  const completed = runValidator(dataDirectory, ["--production"]);
  assert(completed.status === 1, "relabelled demo data passed production mode");
  assert(
    completed.stderr.includes("must identify the RSP parquet export") &&
      completed.stderr.includes("production provenance must not contain demo or synthetic"),
    `relabelled demo rejection did not identify synthetic provenance\n${completed.stderr}`
  );
});

const invalidCases = [
  {
    name: "impossible calendar date",
    file: "public_manifest.json",
    expected: "must be a real calendar date using YYYY-MM-DD",
    mutate(document) {
      document.selected_night_date = "2026-99-99";
    }
  },
  {
    name: "calendar date and MJD drift",
    file: "public_manifest.json",
    expected: "must match the UTC calendar date containing latest_mjd_min",
    mutate(document) {
      document.selected_night_date = "2026-03-04";
      document.source_data_range.latest_night_utc = "2026-03-04";
    }
  },
  {
    name: "non-finite sky-point MJD",
    file: "sky_points.json",
    expected: ".mjd: must be a finite number",
    mutate(document) {
      document.points[0].mjd = "not-a-number";
    }
  },
  {
    name: "negative observation count",
    file: "sky_points.json",
    expected: ".obs_count: must be a non-negative integer",
    mutate(document) {
      document.points[0].obs_count = -1;
    }
  },
  {
    name: "invalid candidate rank",
    file: "top_candidates.json",
    expected: ".rank: must be a positive integer",
    mutate(document) {
      document.candidates[0].rank = 0;
    }
  },
  {
    name: "missing candidate summary",
    file: "top_candidates.json",
    expected: ".public_summary: must be a non-empty string",
    mutate(document) {
      delete document.candidates[0].public_summary;
    }
  },
  {
    name: "candidate coordinate drift",
    file: "top_candidates.json",
    expected: ".ra: must match the sky point",
    mutate(document) {
      document.candidates[0].ra += 0.01;
    }
  },
  {
    name: "invalid lightcurve magnitude",
    file: "lightcurve_samples.json",
    expected: ".magnitude: must be a finite number",
    mutate(document) {
      const identifier = Object.keys(document.lightcurves)[0];
      document.lightcurves[identifier][0].magnitude = null;
    }
  },
  {
    name: "future lightcurve sample",
    file: "lightcurve_samples.json",
    expected: "must not be later than the exported latest MJD range",
    mutate(document) {
      const identifier = Object.keys(document.lightcurves)[0];
      document.lightcurves[identifier].at(-1).mjd = 61104;
    }
  },
  {
    name: "lightcurve sample provenance drift",
    file: "lightcurve_samples.json",
    expected: "must match the document sample_source provenance",
    mutate(document) {
      const identifier = Object.keys(document.lightcurves)[0];
      document.lightcurves[identifier][0].source = "alerts_parquet";
    }
  },
  {
    name: "empty lightcurve filter",
    file: "lightcurve_samples.json",
    expected: ".filter: must be a non-empty string",
    mutate(document) {
      const identifier = Object.keys(document.lightcurves)[0];
      document.lightcurves[identifier][0].filter = "";
    }
  },
  {
    name: "missing candidate lightcurve disposition",
    file: "lightcurve_samples.json",
    expected: "every candidate must have exactly one lightcurve or unavailability reason",
    mutate(document) {
      delete document.lightcurves[Object.keys(document.lightcurves)[0]];
    }
  },
  {
    name: "reversed density bounds",
    file: "density_tiles.json",
    expected: ".ra_min: must be less than ra_max",
    mutate(document) {
      document.tiles[0].ra_min = document.tiles[0].ra_max;
    }
  },
  {
    name: "density membership drift",
    file: "sky_points.json",
    expected: "must match contained",
    mutate(document) {
      const point = document.points.at(-1);
      point.ra = (point.ra + 180) % 360;
    }
  },
  {
    name: "negative density count",
    file: "density_tiles.json",
    expected: ".count: must be a non-negative integer",
    mutate(document) {
      document.tiles[0].count = -1;
    }
  },
  {
    name: "incoherent summary arithmetic",
    file: "public_summary.json",
    expected: "new_loci plus overlap_loci must equal night_loci",
    mutate(document) {
      document.comparison.new_loci += 1;
    }
  },
  {
    name: "private filesystem material",
    file: "public_manifest.json",
    expected: "contains a private or host-local filesystem path",
    mutate(document) {
      document.dataset_name = "/home/private/science";
    }
  },
  {
    name: "RSP filesystem material",
    file: "public_manifest.json",
    expected: "contains a private or host-local filesystem path",
    mutate(document) {
      document.dataset_name = "/sdf/group/rubin/shared/catalog.parquet";
    }
  },
  {
    name: "secret-bearing extension field",
    file: "public_manifest.json",
    expected: "uses a secret-bearing field name",
    mutate(document) {
      document.client_secret = "must-never-ship";
    }
  }
];

for (const testCase of invalidCases) {
  await withFixture(async (dataDirectory) => {
    const document = await readJson(dataDirectory, testCase.file);
    testCase.mutate(document);
    await writeJson(dataDirectory, testCase.file, document);
    const completed = runValidator(dataDirectory);
    assert(completed.status === 1, `${testCase.name}: invalid fixture passed`);
    assert(
      completed.stderr.includes(testCase.expected),
      `${testCase.name}: expected ${testCase.expected}\n${completed.stderr}`
    );
  });
}

await withFixture(async (dataDirectory) => {
  const summary = await readJson(dataDirectory, "public_summary.json");
  summary.source_data_range = Object.fromEntries(
    Object.entries(summary.source_data_range).reverse()
  );
  await writeJson(dataDirectory, "public_summary.json", summary);
  const completed = runValidator(dataDirectory);
  assert(
    completed.status === 0,
    `key-order-only change should remain structurally equal\n${completed.stderr}`
  );
});

await withFixture(async (dataDirectory) => {
  await writeFile(path.join(dataDirectory, "unexpected.json"), "{}\n", "utf8");
  const completed = runValidator(dataDirectory);
  assert(completed.status === 1, "unexpected public-data file passed validation");
  assert(
    completed.stderr.includes("unexpected entry in the public-data bundle"),
    `unexpected-file rejection was not explicit\n${completed.stderr}`
  );
});

console.log(
  `Public-data validator tests passed: baseline, production/relabel rejection, ` +
    `${invalidCases.length} invalid mutations, key-order invariance, and extra-file rejection.`
);
