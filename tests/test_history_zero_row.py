import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import history


REPO_ROOT = Path(__file__).resolve().parents[1]
REPAIR_SCRIPT = REPO_ROOT / "scripts" / "repair_zero_row_nights.py"
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "repair_zero_row_nights", REPAIR_SCRIPT
)
repair = importlib.util.module_from_spec(REPAIR_SPEC)
REPAIR_SPEC.loader.exec_module(repair)


class HistoryZeroRowTests(unittest.TestCase):
    def _write_zero_night(self, root, date_utc="2026-03-05", **updates):
        paths = history.nightly_paths(root, date_utc)
        paths["dir"].mkdir(parents=True)
        loci = history.prepare_loci(
            pd.DataFrame(),
            date_utc,
            61_104.0,
            61_105.0,
            "2026-05-28T06:00:00+00:00",
        )
        alerts = history.prepare_alerts(pd.DataFrame(), date_utc, "fixture")
        loci.to_parquet(paths["loci"], index=False)
        alerts.to_parquet(paths["alerts"], index=False)
        manifest = {
            "date_utc": date_utc,
            "mjd_min": 61_104.0,
            "mjd_max": 61_105.0,
            "query_tag": "fixture",
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
                "alert_locus_link_pass": True,
                "lsst_only_pass": False,
                "history_start_pass": True,
            },
            "paths": {
                "manifest": "/stale/root/manifest.json",
                "loci": "/stale/root/loci.parquet",
                "alerts": "/stale/root/alerts.parquet",
            },
        }
        manifest.update(updates)
        paths["manifest"].write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths

    def test_zero_row_validation_requires_expected_schemas(self):
        loci = history.prepare_loci(
            pd.DataFrame(),
            "2026-03-05",
            61_104.0,
            61_105.0,
            "2026-05-28T06:00:00+00:00",
        )
        alerts = history.prepare_alerts(
            pd.DataFrame(), "2026-03-05", "fixture"
        )
        valid = history.validation_summary(
            loci, alerts, 61_104.0, 61_105.0
        )
        self.assertTrue(valid["zero_row_night"])
        self.assertTrue(valid["zero_row_schema_pass"])
        self.assertTrue(valid["lsst_only_pass"])
        self.assertTrue(valid["append_ready"])

        invalid = history.validation_summary(
            pd.DataFrame(), pd.DataFrame(), 61_104.0, 61_105.0
        )
        self.assertFalse(invalid["zero_row_schema_pass"])
        self.assertFalse(invalid["lsst_only_pass"])
        self.assertFalse(invalid["append_ready"])

    def test_revalidate_zero_row_night_and_stage_cumulative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            paths = self._write_zero_night(root)
            before = paths["manifest"].read_bytes()

            regenerated = history.revalidate_zero_row_night(
                root, "2026-03-05"
            )

            self.assertEqual(paths["manifest"].read_bytes(), before)
            self.assertTrue(regenerated["validation"]["append_ready"])
            self.assertTrue(regenerated["validation"]["lsst_only_pass"])
            self.assertEqual(
                regenerated["revalidation_policy"],
                history.ZERO_ROW_REVALIDATION_POLICY,
            )
            self.assertEqual(
                regenerated["paths"]["loci"], str(paths["loci"])
            )

            stage = Path(tmp) / "staged-cumulative"
            loci_index, summary = history.update_cumulative_indexes(
                root,
                output_dir=stage,
                manifest_overrides={"2026-03-05": regenerated},
            )
            self.assertEqual(len(loci_index), 0)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.loc[0, "date_utc"], "2026-03-05")
            self.assertTrue(bool(summary.loc[0, "append_ready"]))
            self.assertTrue((stage / "loci_index.parquet").is_file())
            self.assertTrue((stage / "nightly_summary.parquet").is_file())

    def test_revalidate_rejects_recorded_query_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            self._write_zero_night(root, query_error_count=1)
            with self.assertRaisesRegex(ValueError, "recorded query/fetch"):
                history.revalidate_zero_row_night(root, "2026-03-05")

    def test_revalidate_rejects_missing_empty_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            paths = self._write_zero_night(root)
            pd.DataFrame().to_parquet(paths["alerts"], index=False)
            with self.assertRaisesRegex(ValueError, "did not pass"):
                history.revalidate_zero_row_night(root, "2026-03-05")

    def test_repair_applies_staged_files_and_retains_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            paths = self._write_zero_night(root)
            history.update_cumulative_indexes(
                root, require_append_ready=False
            )
            backup = Path(tmp) / "repair-evidence"

            with mock.patch.object(
                repair,
                "_quota_headroom_bytes",
                return_value=("test", 2 * 1024**3),
            ):
                report = repair.repair_zero_row_nights(
                    root,
                    ["2026-03-05"],
                    backup,
                    apply=True,
                )

            self.assertTrue(report["applied"])
            updated = json.loads(
                paths["manifest"].read_text(encoding="utf-8")
            )
            self.assertTrue(updated["validation"]["append_ready"])
            original = json.loads(
                (
                    backup
                    / "original"
                    / paths["manifest"].relative_to(root)
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(original["validation"]["append_ready"])
            summary = pd.read_parquet(
                history.cumulative_paths(root)["nightly_summary"]
            )
            self.assertEqual(len(summary), 1)
            self.assertTrue(bool(summary.loc[0, "append_ready"]))
            self.assertTrue((backup / "repair_report.json").is_file())
            self.assertEqual(
                report["promotion_mode"],
                "same_filesystem_atomic_replace_no_rollback",
            )

    def test_repair_refuses_apply_below_quota_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            paths = self._write_zero_night(root)
            history.update_cumulative_indexes(
                root, require_append_ready=False
            )
            before = paths["manifest"].read_bytes()
            backup = Path(tmp) / "repair-evidence"

            with mock.patch.object(
                repair,
                "_quota_headroom_bytes",
                return_value=("user-quota", 100 * 1024**2),
            ):
                with self.assertRaisesRegex(
                    repair.RepairError, "Insufficient write headroom"
                ):
                    repair.repair_zero_row_nights(
                        root,
                        ["2026-03-05"],
                        backup,
                        apply=True,
                    )

            self.assertEqual(paths["manifest"].read_bytes(), before)
            self.assertFalse(backup.exists())

    def test_atomic_promotion_failure_does_not_truncate_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ANTARES_Analysis_Data"
            paths = self._write_zero_night(root)
            history.update_cumulative_indexes(
                root, require_append_ready=False
            )
            targets = repair._source_targets(root, ["2026-03-05"])
            before = {
                label: path.read_bytes() for label, path in targets.items()
            }
            backup = Path(tmp) / "repair-evidence"

            with (
                mock.patch.object(
                    repair,
                    "_quota_headroom_bytes",
                    return_value=("test", 2 * 1024**3),
                ),
                mock.patch.object(
                    repair.os,
                    "replace",
                    side_effect=OSError("simulated atomic replace failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    repair.RepairError, "No rollback was attempted"
                ):
                    repair.repair_zero_row_nights(
                        root,
                        ["2026-03-05"],
                        backup,
                        apply=True,
                    )

            for label, path in targets.items():
                self.assertEqual(path.read_bytes(), before[label])
            self.assertFalse(
                list(root.rglob("*.zero-row-repair-*.tmp"))
            )


if __name__ == "__main__":
    unittest.main()
