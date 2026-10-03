import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_antares_data_root.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_antares_data_root", SCRIPT_PATH
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditAntaresDataRootTests(unittest.TestCase):
    def _write_parquet(self, path, columns, row_group_size=2):
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(columns)
        pq.write_table(table, path, row_group_size=row_group_size)
        return table.num_rows

    def _write_night(
        self,
        root,
        date_utc,
        loci_columns,
        alert_count,
        status="complete",
        manifest_loci_count=None,
        manifest_alert_count=None,
    ):
        year, month, day = date_utc.split("-")
        night = (
            root
            / "data"
            / "lsst_only"
            / "nightly"
            / year
            / month
            / day
        )
        loci_columns = dict(loci_columns)
        loci_row_count = len(next(iter(loci_columns.values()), []))
        loci_columns.setdefault(
            "night_date_utc", [date_utc] * loci_row_count
        )
        loci_count = self._write_parquet(
            night / "loci.parquet", loci_columns
        )
        self._write_parquet(
            night / "alerts.parquet",
            {"locus_id": [f"alert-{i}" for i in range(alert_count)]},
        )
        manifest = {
            "date_utc": date_utc,
            "mjd_min": 61_000.0,
            "mjd_max": 61_001.0,
            "query_tag": "fixture-query",
            "target_loci": loci_count,
            "status": status,
            "actual_loci": (
                loci_count
                if manifest_loci_count is None
                else manifest_loci_count
            ),
            "alert_rows": (
                alert_count
                if manifest_alert_count is None
                else manifest_alert_count
            ),
            "chunk_count": 1,
            "split_count": 0,
            "saturated_chunk_count": 0,
            "survey_mode": "lsst",
            "lsst_filter_used": True,
            "parallel_shards": 1,
            "lsst_dia_count": loci_count,
            "lsst_ss_count": 0,
            "ztf_object_id_count": 0,
            "validation": {
                "append_ready": True,
                "mjd_pass": True,
                "duplicate_locus_count": 0,
                "coordinate_pass": True,
                "overlap_count": 0,
                "alert_locus_link_pass": True,
                "lsst_only_pass": True,
                "history_start_pass": True,
            },
            # These intentionally point elsewhere.  The audit must use the
            # manifest's sibling files so a migrated root remains auditable.
            "paths": {
                "manifest": (
                    f"/stale/rsp/root/data/lsst_only/nightly/"
                    f"{year}/{month}/{day}/manifest.json"
                ),
                "loci": (
                    f"/stale/rsp/root/data/lsst_only/nightly/"
                    f"{year}/{month}/{day}/loci.parquet"
                ),
                "alerts": (
                    f"/stale/rsp/root/data/lsst_only/nightly/"
                    f"{year}/{month}/{day}/alerts.parquet"
                ),
            },
        }
        (night / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return night

    def _make_legacy_zero_row_root(self, base, recorded_error=None):
        root = Path(base) / "ANTARES_Analysis_Data"
        date_utc = "2026-03-05"
        night = (
            root
            / "data"
            / "lsst_only"
            / "nightly"
            / "2026"
            / "03"
            / "05"
        )
        self._write_parquet(
            night / "loci.parquet",
            {
                "locus_id": pa.array([], type=pa.string()),
                "ra": pa.array([], type=pa.float64()),
                "dec": pa.array([], type=pa.float64()),
                "newest_alert_observation_time": pa.array(
                    [], type=pa.float64()
                ),
                "night_date_utc": pa.array([], type=pa.string()),
                "night_mjd_min": pa.array([], type=pa.float64()),
                "night_mjd_max": pa.array([], type=pa.float64()),
                "ingested_at_utc": pa.array([], type=pa.string()),
                "source_query_mode": pa.array([], type=pa.string()),
            },
        )
        self._write_parquet(
            night / "alerts.parquet",
            {
                "locus_id": pa.array([], type=pa.string()),
                "night_date_utc": pa.array([], type=pa.string()),
                "range_label": pa.array([], type=pa.string()),
            },
        )
        manifest = {
            "date_utc": date_utc,
            "mjd_min": 61_104.0,
            "mjd_max": 61_105.0,
            "query_tag": "fixture-query",
            "target_loci": 100_000,
            "actual_loci": 0,
            "alert_rows": 0,
            "chunk_count": 6_912,
            "split_count": 0,
            "saturated_chunk_count": 0,
            "status": "complete",
            "survey_mode": "lsst_only",
            "lsst_filter_used": True,
            "parallel_shards": 1,
            "lsst_dia_count": 0,
            "lsst_ss_count": 0,
            "ztf_object_id_count": 0,
            "finished_at_utc": "2026-05-28T06:29:55+00:00",
            "validation": {
                "append_ready": False,
                "mjd_pass": True,
                "duplicate_locus_count": 0,
                "coordinate_pass": True,
                "overlap_count": 0,
                "alert_locus_link_pass": True,
                "lsst_only_pass": False,
                "history_start_pass": True,
            },
            "paths": {
                "manifest": (
                    "/stale/rsp/root/data/lsst_only/nightly/"
                    "2026/03/05/manifest.json"
                ),
                "loci": (
                    "/stale/rsp/root/data/lsst_only/nightly/"
                    "2026/03/05/loci.parquet"
                ),
                "alerts": (
                    "/stale/rsp/root/data/lsst_only/nightly/"
                    "2026/03/05/alerts.parquet"
                ),
            },
        }
        if recorded_error is not None:
            manifest["query_error_count"] = recorded_error
        (night / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        cumulative = root / "data" / "lsst_only" / "cumulative"
        self._write_parquet(
            cumulative / "loci_index.parquet",
            {
                "locus_id": pa.array([], type=pa.string()),
                "night_date_utc": pa.array([], type=pa.string()),
            },
        )
        expected = audit._manifest_summary_expected(manifest)
        self._write_parquet(
            cumulative / "nightly_summary.parquet",
            {
                field: [expected[field]]
                for field in audit.NIGHTLY_SUMMARY_CORE_FIELDS
            },
        )
        return root

    def _make_valid_root(self, base):
        root = Path(base) / "ANTARES_Analysis_Data"
        first_columns = {
            "locus_id": ["a", "b", "c"],
            "feature_chi2_magn_r": [1.0, np.nan, np.inf],
            "feature_standard_deviation_magn_r": [1.0, 2.0, 3.0],
            "feature_weighted_mean_magn_u": [10.0, 11.0, None],
            "feature_weighted_mean_magn_g": [9.0, 10.0, 11.0],
            "feature_weighted_mean_magn_r": [8.0, 9.0, 10.0],
            "feature_weighted_mean_magn_i": [7.0, None, 9.0],
            "feature_weighted_mean_magn_z": [6.0, 7.0, 8.0],
            "feature_weighted_mean_magn_y": [5.0, 6.0, 7.0],
        }
        second_columns = {
            "locus_id": ["d", "e"],
            "feature_chi2_magn_r": [4.0, 5.0],
            "feature_standard_deviation_magn_r": [4.0, np.nan],
            "feature_weighted_mean_magn_g": [1.0, 2.0],
            "feature_weighted_mean_magn_r": [2.0, 3.0],
            "feature_weighted_mean_magn_i": [3.0, 4.0],
        }
        self._write_night(root, "2026-02-25", first_columns, 4)
        self._write_night(
            root,
            "2026-02-26",
            second_columns,
            2,
            status="under_target",
        )
        cumulative = root / "data" / "lsst_only" / "cumulative"
        self._write_parquet(
            cumulative / "loci_index.parquet",
            {
                "locus_id": ["a", "b", "c", "d", "e"],
                "night_date_utc": [
                    "2026-02-25",
                    "2026-02-25",
                    "2026-02-25",
                    "2026-02-26",
                    "2026-02-26",
                ],
            },
        )
        self._write_parquet(
            cumulative / "nightly_summary.parquet",
            {
                field: [
                    audit._manifest_summary_expected(payload)[field]
                    for payload in (
                        json.loads(
                            (
                                root
                                / "data"
                                / "lsst_only"
                                / "nightly"
                                / date_utc.replace("-", "/")
                                / "manifest.json"
                            ).read_text(encoding="utf-8")
                        )
                        for date_utc in ("2026-02-25", "2026-02-26")
                    )
                ]
                for field in audit.NIGHTLY_SUMMARY_CORE_FIELDS
            },
        )
        cache = root / "cache" / "probe50_v1"
        cache.mkdir(parents=True)
        (cache / "cached.json").write_text("{}\n", encoding="utf-8")
        (cache / "opaque").write_bytes(b"cache")
        return root

    def test_full_audit_is_normalized_streaming_and_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            out = Path(tmp) / "audit"
            root_files_before = {
                path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }

            summary = audit.audit_data_root(root, out, batch_size=2)

            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )
            self.assertEqual(summary["audit_status"], "PASS")
            self.assertTrue(summary["audit_complete"])
            self.assertTrue(summary["integrity"]["ok"])
            self.assertEqual(summary["nightly_manifest_count"], 2)
            self.assertEqual(summary["first_date"], "2026-02-25")
            self.assertEqual(summary["last_date"], "2026-02-26")
            self.assertEqual(summary["complete_nights"], 2)
            self.assertEqual(summary["status_complete_nights"], 1)
            self.assertEqual(summary["physically_complete_nights"], 2)
            self.assertEqual(summary["total_actual_loci"], 5)
            self.assertEqual(summary["total_alert_rows"], 6)
            self.assertEqual(summary["total_loci_parquet_rows"], 5)
            self.assertEqual(summary["total_alerts_parquet_rows"], 6)
            self.assertTrue(summary["cache_present"])
            self.assertEqual(summary["cache_file_count"], 2)

            with (out / "feature_coverage.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                coverage_rows = {
                    (row["kind"], row["name"]): row
                    for row in csv.DictReader(handle)
                }
            expected_counts = {
                ("feature", "feature_chi2_magn_r"): 3,
                (
                    "feature",
                    "feature_standard_deviation_magn_r",
                ): 4,
                ("feature", "feature_weighted_mean_magn_u"): 2,
                ("feature", "feature_weighted_mean_magn_g"): 5,
                ("feature", "feature_weighted_mean_magn_r"): 5,
                ("feature", "feature_weighted_mean_magn_i"): 4,
                ("feature", "feature_weighted_mean_magn_z"): 3,
                ("feature", "feature_weighted_mean_magn_y"): 3,
                ("pairwise", "variability_r"): 2,
                ("pairwise", "weighted_mean_g_i"): 4,
                ("pairwise", "weighted_mean_u_g_r"): 2,
                ("pairwise", "weighted_mean_r_i_g"): 4,
            }
            self.assertEqual(
                {
                    key: int(row["finite_count"])
                    for key, row in coverage_rows.items()
                },
                expected_counts,
            )
            self.assertEqual(
                coverage_rows[
                    ("feature", "feature_weighted_mean_magn_u")
                ]["files_missing_any_column"],
                "1",
            )

            nightly_checksums = (
                out / "nightly_parquet_sha256.txt"
            ).read_text(encoding="utf-8")
            self.assertNotIn(str(root), nightly_checksums)
            self.assertIn(
                "data/lsst_only/nightly/2026/02/25/loci.parquet",
                nightly_checksums,
            )
            all_science = (
                out / "all_science_files_sha256.txt"
            ).read_text(encoding="utf-8")
            self.assertIn(
                "data/lsst_only/nightly/2026/02/25/manifest.json",
                all_science,
            )
            self.assertIn(
                "data/lsst_only/cumulative/loci_index.parquet",
                all_science,
            )
            self.assertNotIn("cache/", all_science)

            manifest_rows = list(
                csv.DictReader(
                    (out / "nightly_manifest_table.csv").read_text(
                        encoding="utf-8"
                    ).splitlines()
                )
            )
            self.assertTrue(
                manifest_rows[0]["declared_loci_path"].startswith(
                    "/stale/rsp/root/"
                )
            )
            self.assertEqual(
                manifest_rows[0]["loci_path"],
                "data/lsst_only/nightly/2026/02/25/loci.parquet",
            )

            inventory = json.loads(
                (out / "file_counts.json").read_text(encoding="utf-8")
            )
            self.assertEqual(inventory["cache"]["file_count"], 2)
            self.assertEqual(
                inventory["cache"]["file_counts_by_extension"],
                {"[no extension]": 1, ".json": 1},
            )
            self.assertGreater(
                inventory["root"]["file_count"],
                inventory["cache"]["file_count"],
            )

            root_files_after = {
                path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(root_files_before, root_files_after)

            with self.assertRaises(audit.AuditPreflightError):
                audit.audit_data_root(root, out, batch_size=2)

    def test_valid_legacy_zero_row_night_is_accepted_by_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_legacy_zero_row_root(tmp)
            out = Path(tmp) / "zero-row-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            self.assertEqual(summary["audit_status"], "PASS")
            self.assertEqual(summary["complete_nights"], 1)
            self.assertEqual(summary["zero_row_policy_accepted_nights"], 1)
            with (out / "nightly_manifest_table.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["append_ready"], "False")
            self.assertEqual(row["effective_append_ready"], "True")
            self.assertEqual(row["zero_row_policy_accepted"], "True")
            self.assertEqual(row["integrity_issue_codes"], "")

    def test_zero_row_policy_rejects_recorded_query_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_legacy_zero_row_root(
                tmp, recorded_error=1
            )
            out = Path(tmp) / "zero-row-error-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertEqual(summary["complete_nights"], 0)
            codes = {
                item["code"] for item in summary["integrity"]["issues"]
            }
            self.assertIn("manifest_recorded_query_fetch_error", codes)
            self.assertIn("manifest_not_append_ready", codes)

    def test_cli_writes_all_reports_and_returns_one_for_integrity_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bad-root"
            columns = {
                "locus_id": ["a", "b"],
                "feature_chi2_magn_r": [1.0, 2.0],
            }
            night = self._write_night(
                root,
                "2026-03-01",
                columns,
                1,
                manifest_loci_count=99,
            )
            (night / "alerts.parquet").unlink()
            orphan_night = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "03"
                / "02"
            )
            self._write_parquet(
                orphan_night / "loci.parquet",
                {
                    "locus_id": ["orphan"],
                    "feature_chi2_magn_r": [3.0],
                },
            )
            cumulative = root / "data" / "lsst_only" / "cumulative"
            self._write_parquet(
                cumulative / "loci_index.parquet", {"locus_id": ["a", "b"]}
            )
            out = Path(tmp) / "bad-audit"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root),
                    "--out",
                    str(out),
                    "--batch-size",
                    "1",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, audit.EXIT_INTEGRITY_ISSUES)
            self.assertIn(f"ANTARES data root: {root.resolve()}", result.stdout)
            self.assertIn("AUDIT FAIL", result.stdout)
            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )
            summary = json.loads(
                (out / "summary.json").read_text(encoding="utf-8")
            )
            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertIn("loci_row_count_mismatch", codes)
            self.assertIn("missing_alerts_parquet", codes)
            self.assertIn("missing_manifest_json", codes)
            self.assertEqual(summary["nightly_manifest_count"], 1)
            self.assertEqual(summary["nightly_partition_count"], 2)
            self.assertIn(
                "data/lsst_only/nightly/2026/03/02/loci.parquet",
                (out / "nightly_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_cli_returns_two_and_does_not_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            out = Path(tmp) / "existing-audit"
            out.mkdir()
            sentinel = out / "keep.txt"
            sentinel.write_text("do not overwrite\n", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root),
                    "--out",
                    str(out),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, audit.EXIT_ERROR)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "do not overwrite\n"
            )
            self.assertEqual(list(out.iterdir()), [sentinel])

    def test_identical_copies_have_identical_normalized_checksum_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_valid_root(Path(tmp) / "source-parent")
            destination_parent = Path(tmp) / "destination-parent"
            destination_parent.mkdir()
            destination = destination_parent / source.name
            shutil.copytree(source, destination)
            source_out = Path(tmp) / "source-audit"
            destination_out = Path(tmp) / "destination-audit"

            audit.audit_data_root(source, source_out, batch_size=2)
            audit.audit_data_root(destination, destination_out, batch_size=2)

            for filename in (
                "nightly_parquet_sha256.txt",
                "cumulative_parquet_sha256.txt",
                "all_science_files_sha256.txt",
            ):
                self.assertEqual(
                    (source_out / filename).read_text(encoding="utf-8"),
                    (destination_out / filename).read_text(encoding="utf-8"),
                )

    def test_nested_science_directory_symlink_is_flagged_and_not_traversed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            original_night = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
            )
            external_night = Path(tmp) / "external-night"
            shutil.copytree(original_night, external_night)
            linked_parent = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "03"
            )
            linked_parent.mkdir()
            linked_night = linked_parent / "03"
            linked_night.symlink_to(external_night, target_is_directory=True)
            out = Path(tmp) / "symlink-directory-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertIn("science_path_symlink", codes)
            self.assertIn("science_path_outside_root", codes)
            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )
            for filename in (
                "nightly_parquet_sha256.txt",
                "all_science_files_sha256.txt",
            ):
                self.assertNotIn(
                    "nightly/2026/03/03",
                    (out / filename).read_text(encoding="utf-8"),
                )

    def test_science_file_symlink_is_flagged_and_never_checksummed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            loci_path = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
                / "loci.parquet"
            )
            external_loci = Path(tmp) / "external-loci.parquet"
            shutil.copy2(loci_path, external_loci)
            loci_path.unlink()
            loci_path.symlink_to(external_loci)
            out = Path(tmp) / "symlink-file-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertIn("science_path_symlink", codes)
            self.assertIn("science_path_outside_root", codes)
            normalized_path = (
                "data/lsst_only/nightly/2026/02/25/loci.parquet"
            )
            self.assertNotIn(
                normalized_path,
                (out / "nightly_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertNotIn(
                normalized_path,
                (out / "all_science_files_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_symlinked_cumulative_root_is_flagged_and_not_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            cumulative = root / "data" / "lsst_only" / "cumulative"
            external_cumulative = Path(tmp) / "external-cumulative"
            shutil.copytree(cumulative, external_cumulative)
            shutil.rmtree(cumulative)
            cumulative.symlink_to(
                external_cumulative, target_is_directory=True
            )
            out = Path(tmp) / "symlink-cumulative-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertIn("science_path_symlink", codes)
            self.assertIn("science_path_outside_root", codes)
            self.assertEqual(
                (out / "cumulative_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
                "",
            )
            self.assertNotIn(
                "data/lsst_only/cumulative/",
                (out / "all_science_files_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_resolved_protected_output_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            cache = root / "cache"
            external_cache = Path(tmp) / "external-cache"
            external_cache.mkdir()
            shutil.rmtree(cache)
            cache.symlink_to(external_cache, target_is_directory=True)
            out = external_cache / "audit"

            with self.assertRaisesRegex(
                audit.AuditPreflightError,
                "cannot be placed inside cache",
            ):
                audit.audit_data_root(root, out, batch_size=2)
            self.assertFalse(out.exists())

    def test_source_replacement_after_feature_scan_is_blocking_and_unhashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            loci_path = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
                / "loci.parquet"
            )
            normalized_path = (
                "data/lsst_only/nightly/2026/02/25/loci.parquet"
            )
            original_scan = audit._scan_feature_coverage

            def scan_then_replace(*args, **kwargs):
                result = original_scan(*args, **kwargs)
                replacement = Path(tmp) / "replacement-loci.parquet"
                shutil.copy2(loci_path, replacement)
                os.replace(replacement, loci_path)
                return result

            out = Path(tmp) / "identity-change-audit"
            with mock.patch.object(
                audit,
                "_scan_feature_coverage",
                side_effect=scan_then_replace,
            ):
                summary = audit.audit_data_root(root, out, batch_size=2)

            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertIn("source_identity_changed", codes)
            self.assertNotIn(
                normalized_path,
                (out / "nightly_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertNotIn(
                normalized_path,
                (out / "all_science_files_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_cli_preserves_dangling_out_symlink_and_rejects_inside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            dangling_target = Path(tmp) / "missing-audit-target"
            dangling_out = Path(tmp) / "dangling-audit-link"
            dangling_out.symlink_to(
                dangling_target, target_is_directory=True
            )

            dangling_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root),
                    "--out",
                    str(dangling_out),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(dangling_result.returncode, audit.EXIT_ERROR)
            self.assertIn("refusing to overwrite", dangling_result.stderr)
            self.assertTrue(dangling_out.is_symlink())
            self.assertFalse(dangling_target.exists())

            inside_out = root / "migration_audit" / "run"
            inside_result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root),
                    "--out",
                    str(inside_out),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(inside_result.returncode, audit.EXIT_ERROR)
            self.assertIn(
                "must be entirely outside the data root",
                inside_result.stderr,
            )
            self.assertFalse(inside_out.exists())

    def test_output_inside_data_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            out = root / "migration_audit" / "audit"

            with self.assertRaisesRegex(
                audit.AuditPreflightError,
                "must be entirely outside the data root",
            ):
                audit.audit_data_root(root, out, batch_size=2)
            self.assertFalse(out.exists())

    def test_summary_is_not_published_when_final_persistence_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            out = Path(tmp) / "failed-persistence-audit"
            real_open = Path.open

            def injected_open(path, mode="r", *args, **kwargs):
                if path.name == ".summary.json.tmp" and mode == "x":
                    raise OSError("injected summary persistence failure")
                return real_open(path, mode, *args, **kwargs)

            with mock.patch.object(Path, "open", new=injected_open):
                with self.assertRaisesRegex(
                    audit.AuditPreflightError,
                    "Could not publish complete audit summary",
                ):
                    audit.audit_data_root(root, out, batch_size=2)

            self.assertTrue(out.is_dir())
            self.assertFalse((out / "summary.json").exists())
            self.assertFalse((out / ".summary.json.tmp").exists())
            for name in audit.OUTPUT_NAMES:
                if name != "summary.json":
                    self.assertTrue((out / name).is_file(), name)

    def test_preserved_mtime_in_place_mutation_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            loci_path = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
                / "loci.parquet"
            )
            normalized_path = (
                "data/lsst_only/nightly/2026/02/25/loci.parquet"
            )
            original_scan = audit._scan_feature_coverage
            observed = {}

            def scan_then_mutate(*args, **kwargs):
                result = original_scan(*args, **kwargs)
                before = loci_path.stat()
                with loci_path.open("r+b") as handle:
                    handle.seek(64)
                    original = handle.read(1)
                    handle.seek(64)
                    handle.write(bytes([original[0] ^ 0x01]))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.utime(
                    loci_path,
                    ns=(before.st_atime_ns, before.st_mtime_ns),
                )
                after = loci_path.stat()
                observed.update(
                    {
                        "same_inode": before.st_ino == after.st_ino,
                        "same_size": before.st_size == after.st_size,
                        "same_mtime": before.st_mtime_ns
                        == after.st_mtime_ns,
                        "ctime_changed": before.st_ctime_ns
                        != after.st_ctime_ns,
                    }
                )
                return result

            out = Path(tmp) / "in-place-mutation-audit"
            with mock.patch.object(
                audit,
                "_scan_feature_coverage",
                side_effect=scan_then_mutate,
            ):
                summary = audit.audit_data_root(root, out, batch_size=2)

            self.assertEqual(
                observed,
                {
                    "same_inode": True,
                    "same_size": True,
                    "same_mtime": True,
                    "ctime_changed": True,
                },
            )
            self.assertFalse(summary["audit_complete"])
            self.assertIn(
                "source_identity_changed",
                {
                    issue["code"]
                    for issue in summary["integrity"]["issues"]
                },
            )
            self.assertNotIn(
                normalized_path,
                (out / "nightly_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_canonical_cumulative_products_are_required_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_root = self._make_valid_root(Path(tmp) / "missing")
            (
                missing_root
                / "data"
                / "lsst_only"
                / "cumulative"
                / "nightly_summary.parquet"
            ).unlink()
            missing_out = Path(tmp) / "missing-cumulative-audit"

            missing_summary = audit.audit_data_root(
                missing_root, missing_out, batch_size=2
            )

            self.assertFalse(missing_summary["audit_complete"])
            self.assertIn(
                "missing_canonical_cumulative_product",
                {
                    issue["code"]
                    for issue in missing_summary["integrity"]["issues"]
                },
            )

            stale_root = self._make_valid_root(Path(tmp) / "stale")
            stale_index = (
                stale_root
                / "data"
                / "lsst_only"
                / "cumulative"
                / "loci_index.parquet"
            )
            self._write_parquet(
                stale_index,
                {
                    "locus_id": ["a"],
                    "night_date_utc": ["2026-02-25"],
                },
            )
            stale_out = Path(tmp) / "stale-cumulative-audit"

            stale_summary = audit.audit_data_root(
                stale_root, stale_out, batch_size=2
            )

            stale_codes = {
                issue["code"]
                for issue in stale_summary["integrity"]["issues"]
            }
            self.assertEqual(stale_summary["audit_status"], "FAIL")
            self.assertTrue(stale_summary["audit_complete"])
            self.assertTrue(stale_summary["report_set_complete"])
            self.assertIn("stale_cumulative_row_count", stale_codes)
            self.assertIn("stale_cumulative_date_coverage", stale_codes)

    def test_cumulative_locus_id_multiset_mismatch_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            index_path = (
                root
                / "data"
                / "lsst_only"
                / "cumulative"
                / "loci_index.parquet"
            )
            self._write_parquet(
                index_path,
                {
                    "locus_id": ["wrong-id", "b", "c", "d", "e"],
                    "night_date_utc": [
                        "2026-02-25",
                        "2026-02-25",
                        "2026-02-25",
                        "2026-02-26",
                        "2026-02-26",
                    ],
                },
            )
            out = Path(tmp) / "wrong-cumulative-id-audit"

            summary = audit.audit_data_root(root, out, batch_size=1)

            codes = {
                issue["code"]
                for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertTrue(summary["audit_complete"])
            self.assertIn(
                "cumulative_loci_id_multiset_mismatch", codes
            )
            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )

    def test_cumulative_summary_core_field_mismatch_is_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            summary_path = (
                root
                / "data"
                / "lsst_only"
                / "cumulative"
                / "nightly_summary.parquet"
            )
            summary_columns = pq.read_table(summary_path).to_pydict()
            summary_columns["actual_loci"][0] += 1
            self._write_parquet(summary_path, summary_columns)
            out = Path(tmp) / "wrong-cumulative-summary-audit"

            summary = audit.audit_data_root(root, out, batch_size=1)

            matching_issues = [
                issue
                for issue in summary["integrity"]["issues"]
                if issue["code"]
                == "cumulative_nightly_summary_semantic_mismatch"
            ]
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertTrue(summary["audit_complete"])
            self.assertEqual(len(matching_issues), 1)
            self.assertIn("actual_loci", matching_issues[0]["message"])
            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )

    def test_final_snapshot_detects_add_remove_and_post_hash_change(self):
        actions = ("add", "remove", "change")
        for action in actions:
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                root = self._make_valid_root(tmp)
                original_checksum = audit._checksum_records
                calls = {"count": 0}

                def checksum_then_change(*args, **kwargs):
                    records = original_checksum(*args, **kwargs)
                    calls["count"] += 1
                    if calls["count"] != 3:
                        return records
                    if action == "add":
                        analysis_root = root / "analysis"
                        analysis_root.mkdir()
                        (analysis_root / "late.txt").write_text(
                            "late\n", encoding="utf-8"
                        )
                    elif action == "remove":
                        (
                            root
                            / "data"
                            / "lsst_only"
                            / "nightly"
                            / "2026"
                            / "02"
                            / "25"
                            / "alerts.parquet"
                        ).unlink()
                    else:
                        manifest = (
                            root
                            / "data"
                            / "lsst_only"
                            / "nightly"
                            / "2026"
                            / "02"
                            / "25"
                            / "manifest.json"
                        )
                        with manifest.open("a", encoding="utf-8") as handle:
                            handle.write(" ")
                    return records

                out = Path(tmp) / f"final-snapshot-{action}"
                with mock.patch.object(
                    audit,
                    "_checksum_records",
                    side_effect=checksum_then_change,
                ):
                    summary = audit.audit_data_root(
                        root, out, batch_size=2
                    )

                codes = {
                    issue["code"]
                    for issue in summary["integrity"]["issues"]
                }
                self.assertFalse(summary["audit_complete"])
                if action in {"add", "remove"}:
                    self.assertIn("durable_path_set_changed", codes)
                else:
                    self.assertIn("source_identity_changed", codes)

    def test_full_page_decode_detects_corrupt_parquet_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            alerts_path = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
                / "alerts.parquet"
            )
            pq.write_table(
                pa.table({"locus_id": ["a", "b", "c", "d"]}),
                alerts_path,
                compression=None,
                data_page_version="2.0",
                write_page_checksum=True,
            )
            payload = bytearray(alerts_path.read_bytes())
            payload[50] ^= 0xFF
            alerts_path.write_bytes(payload)
            out = Path(tmp) / "corrupt-page-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            codes = {
                issue["code"] for issue in summary["integrity"]["issues"]
            }
            self.assertEqual(summary["audit_status"], "FAIL")
            self.assertFalse(summary["audit_complete"])
            self.assertIn("parquet_data_decode_error", codes)
            self.assertEqual(
                {path.name for path in out.iterdir()},
                set(audit.OUTPUT_NAMES),
            )

    def test_optional_analysis_files_are_in_all_science_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._make_valid_root(tmp)
            top_analysis = root / "analysis"
            survey_analysis = (
                root / "data" / "lsst_only" / "analysis"
            )
            top_analysis.mkdir()
            survey_analysis.mkdir()
            (top_analysis / "report.txt").write_text(
                "report\n", encoding="utf-8"
            )
            (survey_analysis / "figure.png").write_bytes(b"png")
            out = Path(tmp) / "analysis-artifact-audit"

            summary = audit.audit_data_root(root, out, batch_size=2)

            self.assertEqual(summary["audit_status"], "PASS")
            checksum_text = (
                out / "all_science_files_sha256.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("analysis/report.txt", checksum_text)
            self.assertIn(
                "data/lsst_only/analysis/figure.png", checksum_text
            )
            self.assertNotIn(
                "analysis/report.txt",
                (out / "nightly_parquet_sha256.txt").read_text(
                    encoding="utf-8"
                ),
            )

    def test_invalid_counts_and_duplicate_dates_clear_complete_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            invalid_root = self._make_valid_root(Path(tmp) / "invalid")
            invalid_manifest = (
                invalid_root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "25"
                / "manifest.json"
            )
            invalid_payload = json.loads(
                invalid_manifest.read_text(encoding="utf-8")
            )
            invalid_payload["actual_loci"] = "not-an-integer"
            invalid_manifest.write_text(
                json.dumps(invalid_payload) + "\n", encoding="utf-8"
            )
            invalid_out = Path(tmp) / "invalid-count-audit"

            invalid_summary = audit.audit_data_root(
                invalid_root, invalid_out, batch_size=2
            )

            self.assertEqual(invalid_summary["complete_nights"], 1)
            invalid_rows = list(
                csv.DictReader(
                    (invalid_out / "nightly_manifest_table.csv").read_text(
                        encoding="utf-8"
                    ).splitlines()
                )
            )
            invalid_row = next(
                row
                for row in invalid_rows
                if row["folder_date_utc"] == "2026-02-25"
            )
            self.assertEqual(invalid_row["science_files_complete"], "False")
            self.assertEqual(invalid_row["operationally_complete"], "False")

            duplicate_root = self._make_valid_root(Path(tmp) / "duplicate")
            duplicate_manifest = (
                duplicate_root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "02"
                / "26"
                / "manifest.json"
            )
            duplicate_payload = json.loads(
                duplicate_manifest.read_text(encoding="utf-8")
            )
            duplicate_payload["date_utc"] = "2026-02-25"
            duplicate_manifest.write_text(
                json.dumps(duplicate_payload) + "\n", encoding="utf-8"
            )
            duplicate_out = Path(tmp) / "duplicate-date-audit"

            duplicate_summary = audit.audit_data_root(
                duplicate_root, duplicate_out, batch_size=2
            )

            self.assertEqual(duplicate_summary["complete_nights"], 0)
            self.assertEqual(duplicate_summary["physically_complete_nights"], 0)
            duplicate_rows = list(
                csv.DictReader(
                    (
                        duplicate_out / "nightly_manifest_table.csv"
                    ).read_text(encoding="utf-8").splitlines()
                )
            )
            self.assertTrue(
                all(
                    row["science_files_complete"] == "False"
                    and row["operationally_complete"] == "False"
                    for row in duplicate_rows
                )
            )


if __name__ == "__main__":
    unittest.main()
