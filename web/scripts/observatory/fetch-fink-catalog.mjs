#!/usr/bin/env node
// Fetch the read-only extract of the accepted Fink catalog from the data host
// and write it as the generator input
// scripts/observatory/inputs/fink-catalog.<run id>.json.gz.
//
//   node scripts/observatory/fetch-fink-catalog.mjs [--host=arnor] [--run-id=<id>]
//
// The extractor (extract-fink-catalog.py) is streamed over SSH stdin and runs
// under the Fink analytics interpreter with the catalog's own release on its
// path. It opens the catalog with Fink's native read-only opener and prints
// JSON; nothing is written on the host. This script only adds the extractor's
// digest and compresses the result.

import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const PYTHON = "/astro/store/shiren/mdarim/envs/fink-lsst-analysis-g4b-py311/bin/python";
let host = "arnor";
let runId = "final-five-window-analytics-20261006-v1";
for (const argument of process.argv.slice(2)) {
  if (argument.startsWith("--host=")) host = argument.slice("--host=".length);
  else if (argument.startsWith("--run-id=")) runId = argument.slice("--run-id=".length);
  else {
    console.error(`Unknown argument: ${argument}`);
    process.exit(2);
  }
}
if (!/^[A-Za-z0-9._-]+$/.test(runId)) {
  console.error("run id must be a plain path component");
  process.exit(2);
}

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const extractorPath = path.join(scriptDirectory, "extract-fink-catalog.py");
const outputPath = path.join(scriptDirectory, "inputs", `fink-catalog.${runId}.json.gz`);
const extractor = await readFile(extractorPath);

const result = spawnSync("ssh", ["-o", "BatchMode=yes", host, `cd /tmp && ${PYTHON} -I -B - ${runId}`], {
  input: extractor,
  maxBuffer: 256 * 1024 * 1024,
  stdio: ["pipe", "pipe", "inherit"]
});
if (result.status !== 0) {
  console.error(`extractor exited ${result.status}`);
  process.exit(1);
}
const extract = JSON.parse(result.stdout.toString("utf8"));
if (extract.kind !== "uso.fink-catalog-extract" || extract.catalog?.run_id !== runId) {
  console.error("unexpected extractor output");
  process.exit(1);
}
extract.extractor = {
  path: "web/scripts/observatory/extract-fink-catalog.py",
  sha256: createHash("sha256").update(extractor).digest("hex")
};
const text = `${JSON.stringify(extract)}\n`;
await writeFile(outputPath, gzipSync(text, { level: 9 }));
console.log(
  `Wrote ${path.relative(process.cwd(), outputPath)}: ${extract.density.objects} DiaObjects in ${extract.density.pixels.length} cells, ` +
    `${extract.daily.length} dates, sample of ${extract.sample.size}.`
);
