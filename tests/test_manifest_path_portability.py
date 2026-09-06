import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")

from src import feature_analysis, history


class ManifestPathPortabilityTests(unittest.TestCase):
    DATE_UTC = "2026-03-01"
    MJD_MIN = 61100.0

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _loci_frame(self, locus_ids):
        size = len(locus_ids)
        return pd.DataFrame(
            {
                "locus_id": list(locus_ids),
                "newest_alert_observation_time": [
                    self.MJD_MIN + index / 1000 for index in range(size)
                ],
                "night_date_utc": [self.DATE_UTC] * size,
                "night_mjd_min": [self.MJD_MIN] * size,
                "night_mjd_max": [self.MJD_MIN + 1] * size,
                "tags": ["science"] * size,
                "feature_chi2_magn_r": [float(index + 1) for index in range(size)],
                "feature_standard_deviation_magn_r": [0.1] * size,
                "feature_weighted_mean_magn_g": [20.0] * size,
                "feature_weighted_mean_magn_r": [19.0] * size,
                "feature_weighted_mean_magn_i": [18.0] * size,
            }
        )

    def _write_night(
        self,
        root,
        locus_ids,
        alert_locus_ids=None,
        declared_root=None,
        append_ready=True,
    ):
        root = Path(root)
        alert_locus_ids = (
            list(locus_ids) if alert_locus_ids is None else list(alert_locus_ids)
        )
        paths = history.nightly_paths(root, self.DATE_UTC)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        self._loci_frame(locus_ids).to_parquet(paths["loci"], index=False)
        pd.DataFrame(
            {
                "locus_id": alert_locus_ids,
                "night_date_utc": [self.DATE_UTC] * len(alert_locus_ids),
                "range_label": ["fixture"] * len(alert_locus_ids),
            }
        ).to_parquet(paths["alerts"], index=False)

        declared_paths = history.nightly_paths(
            Path(declared_root) if declared_root is not None else root,
            self.DATE_UTC,
        )
        manifest = {
            "date_utc": self.DATE_UTC,
            "mjd_min": self.MJD_MIN,
            "mjd_max": self.MJD_MIN + 1,
            "actual_loci": len(locus_ids),
            "alert_rows": len(alert_locus_ids),
            "status": "complete",
            "validation": {"append_ready": append_ready},
            "paths": {
                "loci": str(declared_paths["loci"]),
                "alerts": str(declared_paths["alerts"]),
                "manifest": str(declared_paths["manifest"]),
            },
        }
        paths["manifest"].write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return paths

    @staticmethod
    def _update_manifest(paths, **updates):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest.update(updates)
        paths["manifest"].write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _snapshot_bytes(root):
        paths = feature_analysis._snapshot_paths(root)
        return paths, paths["parquet"].read_bytes(), paths["manifest"].read_bytes()

    def test_copied_siblings_win_over_existing_conflicting_declared_paths(self):
        original = self.tmp_path / "original"
        copied = self.tmp_path / "copied"
        original_paths = self._write_night(
            original, ["copied-locus"], ["copied-alert"]
        )
        original_inventory = feature_analysis._source_inventory(original)

        shutil.copytree(original, copied, copy_function=shutil.copy2)
        copied_loci = history.nightly_paths(copied, self.DATE_UTC)["loci"]
        copied_stat = copied_loci.stat()
        os.utime(
            copied_loci,
            ns=(copied_stat.st_atime_ns, copied_stat.st_mtime_ns + 1_000_000),
        )
        copied_inventory = feature_analysis._source_inventory(copied)
        self.assertEqual(original_inventory, copied_inventory)
        self.assertEqual(
            feature_analysis._inventory_hash(original_inventory),
            feature_analysis._inventory_hash(copied_inventory),
        )
        self.assertEqual(
            copied_inventory[0]["path"],
            "data/lsst_only/nightly/2026/03/01/loci.parquet",
        )
        self.assertFalse(Path(copied_inventory[0]["path"]).is_absolute())

        # The copied manifest still declares the original paths. Make those
        # files conflict so following the provenance is immediately visible.
        self._loci_frame(["wrong-1", "wrong-2"]).to_parquet(
            original_paths["loci"], index=False
        )
        pd.DataFrame(
            {
                "locus_id": ["wrong-alert-1", "wrong-alert-2"],
                "night_date_utc": [self.DATE_UTC, self.DATE_UTC],
                "range_label": ["wrong", "wrong"],
            }
        ).to_parquet(original_paths["alerts"], index=False)

        loci_index, _ = history.update_cumulative_indexes(
            copied, output_dir=self.tmp_path / "staged-cumulative"
        )
        alerts = history.load_cumulative_alerts(copied)
        snapshots, coverage, manifest = (
            feature_analysis.build_or_load_feature_snapshots(copied)
        )

        self.assertEqual(set(loci_index["locus_id"]), {"copied-locus"})
        self.assertEqual(set(alerts["locus_id"]), {"copied-alert"})
        self.assertEqual(set(snapshots["locus_id"]), {"copied-locus"})
        self.assertEqual(
            set(coverage["source_path"]), {copied_inventory[0]["path"]}
        )
        self.assertEqual(manifest["source_inventory"], copied_inventory)

    def test_nonexistent_embedded_paths_do_not_break_valid_siblings(self):
        root = self.tmp_path / "relocated"
        self._write_night(
            root,
            ["local-locus"],
            ["local-alert"],
            declared_root=self.tmp_path / "does-not-exist",
        )

        loci_index, _ = history.update_cumulative_indexes(
            root, output_dir=self.tmp_path / "cumulative"
        )
        alerts = history.load_cumulative_alerts(root)
        snapshots, _, _ = feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(set(loci_index["locus_id"]), {"local-locus"})
        self.assertEqual(set(alerts["locus_id"]), {"local-alert"})
        self.assertEqual(set(snapshots["locus_id"]), {"local-locus"})

    def test_missing_append_ready_siblings_fail_closed(self):
        loci_root = self.tmp_path / "missing-loci"
        loci_paths = self._write_night(loci_root, ["one"])
        loci_paths["loci"].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sibling loci.parquet"):
            history.update_cumulative_indexes(
                loci_root, output_dir=self.tmp_path / "loci-output"
            )
        with self.assertRaisesRegex(FileNotFoundError, "sibling loci.parquet"):
            feature_analysis.build_or_load_feature_snapshots(loci_root)

        alerts_root = self.tmp_path / "missing-alerts"
        alerts_paths = self._write_night(alerts_root, ["one"])
        alerts_paths["alerts"].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "sibling alerts.parquet"):
            history.load_cumulative_alerts(alerts_root)

    def test_unreadable_append_ready_alerts_fail_closed(self):
        root = self.tmp_path / "unreadable-alerts"
        paths = self._write_night(root, ["one"])
        paths["alerts"].write_bytes(b"not a parquet file")
        with self.assertRaisesRegex(ValueError, "Could not read sibling alerts.parquet"):
            history.load_cumulative_alerts(root)

    def test_cumulative_summary_and_tri_state_append_gate_are_preserved(self):
        rejected_root = self.tmp_path / "explicitly-rejected"
        rejected = self._write_night(
            rejected_root, ["rejected"], append_ready=False
        )
        rejected["loci"].unlink()
        rejected["alerts"].unlink()
        loci, summary = history.update_cumulative_indexes(
            rejected_root, output_dir=self.tmp_path / "rejected-output"
        )
        self.assertTrue(loci.empty)
        self.assertEqual(len(summary), 1)
        self.assertFalse(bool(summary.loc[0, "append_ready"]))
        self.assertTrue(history.load_cumulative_alerts(rejected_root).empty)

        legacy_root = self.tmp_path / "legacy-missing-gate"
        legacy = self._write_night(legacy_root, ["legacy"])
        self._update_manifest(legacy, validation={})
        loci, summary = history.update_cumulative_indexes(
            legacy_root, output_dir=self.tmp_path / "legacy-output"
        )
        alerts = history.load_cumulative_alerts(legacy_root)
        snapshots, _, _ = feature_analysis.build_or_load_feature_snapshots(
            legacy_root
        )
        self.assertEqual(set(loci["locus_id"]), {"legacy"})
        self.assertEqual(set(alerts["locus_id"]), {"legacy"})
        self.assertEqual(set(snapshots["locus_id"]), {"legacy"})
        self.assertEqual(len(summary), 1)

        failed_root = self.tmp_path / "failed-summary"
        failed = self._write_night(failed_root, ["failed"])
        self._update_manifest(failed, status="failed")
        failed["loci"].unlink()
        _, summary = history.update_cumulative_indexes(
            failed_root, output_dir=self.tmp_path / "failed-output"
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "status"], "failed")

    def test_source_failure_preserves_existing_snapshot_bytes(self):
        root = self.tmp_path / "preserve"
        source_paths = self._write_night(root, ["valid"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)

        source_paths["loci"].write_bytes(b"not a parquet file")
        with self.assertRaisesRegex(ValueError, "Could not read sibling loci.parquet"):
            # This is intentionally not forced: even a cache-current call must
            # validate its append-ready source inventory first.
            feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)
        self.assertEqual(
            list(snapshot_paths["root"].glob(".feature-snapshot-stage-*")), []
        )

    def test_cache_current_call_still_decodes_every_source_column(self):
        root = self.tmp_path / "cache-current-read-check"
        source_paths = self._write_night(root, ["valid"])
        source = pd.read_parquet(source_paths["loci"])
        source["unrequested_payload"] = ["must-also-decode"]
        source.to_parquet(source_paths["loci"], index=False)
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)
        real_parquet_file = feature_analysis.pq.ParquetFile

        class FooterReadableButDataUnreadable:
            def __init__(self, path):
                self._wrapped = real_parquet_file(path)
                self.schema_arrow = self._wrapped.schema_arrow
                self.metadata = self._wrapped.metadata

            def iter_batches(self, *args, **kwargs):
                raise OSError("simulated unrequested-column data-page failure")

        with mock.patch.object(
            feature_analysis.pq,
            "ParquetFile",
            side_effect=FooterReadableButDataUnreadable,
        ):
            with self.assertRaisesRegex(ValueError, "Could not decode all columns"):
                feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)

    def test_source_change_between_inventory_and_build_preserves_snapshot(self):
        root = self.tmp_path / "source-changed-during-build"
        source_paths = self._write_night(root, ["old"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)
        real_inventory = feature_analysis._source_inventory

        def inventory_then_replace(data_root):
            inventory = real_inventory(data_root)
            self._loci_frame(["new"]).to_parquet(
                source_paths["loci"], index=False
            )
            return inventory

        with mock.patch.object(
            feature_analysis,
            "_source_inventory",
            side_effect=inventory_then_replace,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed while"):
                feature_analysis.build_or_load_feature_snapshots(
                    root, force=True
                )

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)

    def test_cache_current_pair_is_fully_validated_before_return(self):
        root = self.tmp_path / "validate-current-cache"
        self._write_night(root, ["valid"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths = feature_analysis._snapshot_paths(root)

        pd.DataFrame({"junk": ["readable-but-invalid"]}).to_parquet(
            snapshot_paths["parquet"], index=False
        )
        snapshots, _, manifest = feature_analysis.build_or_load_feature_snapshots(
            root
        )
        self.assertEqual(list(snapshots.columns), feature_analysis.REQUESTED_COLUMNS)
        self.assertEqual(set(snapshots["locus_id"]), {"valid"})
        self.assertEqual(manifest["snapshot_rows"], 1)

        tampered = json.loads(
            snapshot_paths["manifest"].read_text(encoding="utf-8")
        )
        tampered["source_inventory"][0]["sha256"] = "0" * 64
        snapshot_paths["manifest"].write_text(
            json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8"
        )
        _, _, repaired = feature_analysis.build_or_load_feature_snapshots(root)
        self.assertNotEqual(
            repaired["source_inventory"][0]["sha256"], "0" * 64
        )
        self.assertEqual(
            repaired["source_inventory_hash"],
            feature_analysis._inventory_hash(repaired["source_inventory"]),
        )

    def test_empty_inventory_cannot_replace_existing_nonempty_snapshot(self):
        root = self.tmp_path / "inventory-disappeared"
        source_paths = self._write_night(root, ["valid"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)

        source_paths["manifest"].unlink()
        with self.assertRaisesRegex(RuntimeError, "No append-ready feature sources"):
            feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)

    def test_first_empty_inventory_builds_and_reloads_only_valid_empty_cache(self):
        root = self.tmp_path / "legitimate-empty-root"
        snapshots, coverage, manifest = (
            feature_analysis.build_or_load_feature_snapshots(root)
        )
        self.assertTrue(snapshots.empty)
        self.assertTrue(coverage.empty)
        self.assertEqual(manifest["source_inventory"], [])
        self.assertEqual(manifest["snapshot_rows"], 0)

        loaded, _, loaded_manifest = (
            feature_analysis.build_or_load_feature_snapshots(root)
        )
        self.assertTrue(loaded.empty)
        self.assertEqual(loaded_manifest["source_inventory"], [])

    def test_count_mismatch_preserves_existing_snapshot_bytes(self):
        root = self.tmp_path / "count-mismatch"
        source_paths = self._write_night(root, ["valid"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)

        self._update_manifest(source_paths, actual_loci=2)
        with self.assertRaisesRegex(ValueError, "loci row mismatch"):
            feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)

    def test_successful_rebuild_validates_then_atomically_replaces_each_file(self):
        root = self.tmp_path / "atomic"
        source_paths = self._write_night(root, ["first"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths = feature_analysis._snapshot_paths(root)

        replacement = self._loci_frame(["first", "second"])
        replacement.to_parquet(source_paths["loci"], index=False)
        self._update_manifest(source_paths, actual_loci=2)

        real_replace = os.replace
        replacements = []

        def tracked_replace(source, destination):
            source = Path(source)
            destination = Path(destination)
            replacements.append((source, destination))
            self.assertEqual(source.parent.parent, snapshot_paths["root"])
            self.assertTrue(source.parent.name.startswith(".feature-snapshot-stage-"))
            return real_replace(source, destination)

        with mock.patch.object(
            feature_analysis.os, "replace", side_effect=tracked_replace
        ):
            snapshots, _, manifest = (
                feature_analysis.build_or_load_feature_snapshots(root)
            )

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(
            [destination for _, destination in replacements],
            [snapshot_paths["parquet"], snapshot_paths["manifest"]],
        )
        parquet_file = pq.ParquetFile(snapshot_paths["parquet"])
        self.assertEqual(parquet_file.schema_arrow.names, feature_analysis.REQUESTED_COLUMNS)
        self.assertEqual(parquet_file.metadata.num_rows, 2)
        written_manifest = json.loads(
            snapshot_paths["manifest"].read_text(encoding="utf-8")
        )
        self.assertEqual(written_manifest["snapshot_rows"], 2)
        self.assertEqual(
            written_manifest["source_inventory_hash"],
            feature_analysis._inventory_hash(written_manifest["source_inventory"]),
        )
        self.assertEqual(manifest["snapshot_rows"], 2)
        self.assertEqual(
            list(snapshot_paths["root"].glob(".feature-snapshot-stage-*")), []
        )

    def test_second_promotion_failure_rolls_back_existing_pair(self):
        root = self.tmp_path / "promotion-rollback"
        source_paths = self._write_night(root, ["first"])
        feature_analysis.build_or_load_feature_snapshots(root)
        snapshot_paths, parquet_before, manifest_before = self._snapshot_bytes(root)

        replacement = self._loci_frame(["first", "second"])
        replacement.to_parquet(source_paths["loci"], index=False)
        self._update_manifest(source_paths, actual_loci=2)
        real_replace = os.replace

        def fail_manifest_promotion(source, destination):
            source = Path(source)
            destination = Path(destination)
            if (
                destination == snapshot_paths["manifest"]
                and source.name == snapshot_paths["manifest"].name
            ):
                raise OSError("simulated manifest promotion failure")
            return real_replace(source, destination)

        with mock.patch.object(
            feature_analysis.os,
            "replace",
            side_effect=fail_manifest_promotion,
        ):
            with self.assertRaisesRegex(OSError, "manifest promotion failure"):
                feature_analysis.build_or_load_feature_snapshots(root)

        self.assertEqual(snapshot_paths["parquet"].read_bytes(), parquet_before)
        self.assertEqual(snapshot_paths["manifest"].read_bytes(), manifest_before)
        self.assertEqual(
            list(snapshot_paths["root"].glob(".feature-snapshot-stage-*")), []
        )


if __name__ == "__main__":
    unittest.main()
