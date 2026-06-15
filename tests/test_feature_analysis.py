import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src import feature_analysis


class FeatureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_night(self, date_utc, mjd, frame, append_ready=True):
        path = (
            self.root
            / "data"
            / "lsst_only"
            / "nightly"
            / date_utc.replace("-", "/")
        )
        path.mkdir(parents=True)
        loci_path = path / "loci.parquet"
        alerts_path = path / "alerts.parquet"
        manifest_path = path / "manifest.json"
        frame.to_parquet(loci_path, index=False)
        pd.DataFrame({"locus_id": frame["locus_id"]}).to_parquet(
            alerts_path, index=False
        )
        manifest = {
            "date_utc": date_utc,
            "mjd_min": mjd,
            "mjd_max": mjd + 1,
            "status": "complete",
            "validation": {"append_ready": append_ready},
            "paths": {
                "loci": str(loci_path),
                "alerts": str(alerts_path),
                "manifest": str(manifest_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest))
        return loci_path

    def _frame(self, ids, mjd, offset=0.0, include_u=True):
        n = len(ids)
        frame = pd.DataFrame(
            {
                "locus_id": ids,
                "newest_alert_observation_time": mjd + np.arange(n) / 1000,
                "night_date_utc": "2026-03-01",
                "night_mjd_min": mjd,
                "night_mjd_max": mjd + 1,
                "tags": ["science, high_snr"] * n,
                "feature_chi2_magn_r": np.arange(1, n + 1) + offset,
                "feature_standard_deviation_magn_r": np.linspace(0.1, 1, n),
                "feature_weighted_mean_magn_g": 20 + offset + np.arange(n) / 10,
                "feature_weighted_mean_magn_r": 19 + offset + np.arange(n) / 10,
                "feature_weighted_mean_magn_i": 18 + offset + np.arange(n) / 10,
            }
        )
        if include_u:
            frame["feature_weighted_mean_magn_u"] = (
                21 + offset + np.arange(n) / 10
            )
        return frame

    def test_schema_audit_build_and_cache_invalidation(self):
        first = self._write_night(
            "2026-03-01", 61100.0, self._frame(["a", "b"], 61100.0)
        )
        snapshots, coverage, manifest = (
            feature_analysis.build_or_load_feature_snapshots(self.root)
        )
        self.assertEqual(len(snapshots), 2)
        self.assertTrue(
            coverage.loc[
                coverage["feature"] == "feature_weighted_mean_magn_u",
                "column_present",
            ].all()
        )
        self.assertEqual(
            coverage.loc[
                coverage["feature"] == "feature_weighted_mean_magn_u",
                "finite_count",
            ].sum(),
            2,
        )
        old_hash = manifest["source_inventory_hash"]

        changed = self._frame(["a", "b", "c"], 61100.0)
        changed.to_parquet(first, index=False)
        snapshots2, _, manifest2 = feature_analysis.build_or_load_feature_snapshots(
            self.root
        )
        self.assertEqual(len(snapshots2), 3)
        self.assertNotEqual(old_hash, manifest2["source_inventory_hash"])

    def test_missing_band_does_not_block_other_features(self):
        self._write_night(
            "2026-03-01",
            61100.0,
            self._frame(["a", "b"], 61100.0, include_u=False),
        )
        snapshots, coverage, _ = feature_analysis.build_or_load_feature_snapshots(
            self.root
        )
        self.assertTrue(snapshots["feature_weighted_mean_magn_u"].isna().all())
        u_coverage = coverage[
            coverage["feature"] == "feature_weighted_mean_magn_u"
        ]
        self.assertFalse(u_coverage["column_present"].any())

    def test_latest_historical_snapshot_and_current_exclusion(self):
        self._write_night(
            "2026-03-01", 61100.0, self._frame(["repeat", "old"], 61100.0)
        )
        newer = self._frame(["repeat", "new"], 61101.0, offset=10)
        newer["night_date_utc"] = "2026-03-02"
        self._write_night("2026-03-02", 61101.0, newer)
        current_frame = self._frame(["repeat", "current"], 61102.0, offset=20)
        current_frame["night_date_utc"] = "2026-03-03"
        self._write_night("2026-03-03", 61102.0, current_frame)
        snapshots, _, _ = feature_analysis.build_or_load_feature_snapshots(self.root)

        historical, current = feature_analysis.select_comparison_cohorts(
            snapshots, current_frame, 61102.0
        )
        self.assertEqual(historical["locus_id"].nunique(), len(historical))
        repeat = historical[historical["locus_id"] == "repeat"].iloc[0]
        self.assertEqual(repeat["night_mjd_min"], 61101.0)
        self.assertTrue((historical["night_mjd_min"] < 61102.0).all())
        self.assertEqual(set(current["locus_id"]), {"repeat", "current"})

    def test_color_definitions_and_tag_parsing(self):
        frame = feature_analysis.add_color_columns(
            self._frame(["a"], 61100.0)
        )
        self.assertEqual(frame.loc[0, "color_g_minus_i"], 2)
        self.assertEqual(frame.loc[0, "color_u_minus_g"], 1)
        self.assertEqual(frame.loc[0, "color_g_minus_r"], 1)
        self.assertEqual(frame.loc[0, "color_r_minus_i"], 1)
        selected, table = feature_analysis.rank_tag_subsets(
            pd.concat([frame] * 500, ignore_index=True),
            pd.concat([frame] * 30, ignore_index=True),
        )
        self.assertEqual(selected, ["science"])
        self.assertNotIn("high_snr", set(table.get("tag", [])))
        self.assertIn("variability_r", table.loc[0, "eligible_planes"])

    def test_statistics_are_deterministic_and_handle_invalid_values(self):
        historical = feature_analysis.add_color_columns(
            self._frame([f"h{i}" for i in range(80)], 61100.0)
        )
        current = feature_analysis.add_color_columns(
            self._frame([f"c{i}" for i in range(60)], 61101.0, offset=0.4)
        )
        current.loc[0, "feature_chi2_magn_r"] = -1
        kwargs = dict(
            seed=7,
            permutations=20,
            sample_cap=50,
            min_tag_historical=10,
            min_tag_current=10,
        )
        first = feature_analysis.compute_feature_diagnostics(
            historical, current, **kwargs
        )
        second = feature_analysis.compute_feature_diagnostics(
            historical, current, **kwargs
        )
        pd.testing.assert_frame_equal(
            first["feature_statistics"], second["feature_statistics"]
        )
        variability = first["feature_statistics"].query(
            "plane == 'variability_r'"
        ).iloc[0]
        self.assertEqual(variability["current_complete"], len(current) - 1)
        self.assertTrue(
            0 <= variability["js_permutation_p_value"] <= 1
        )
        ks_stat, ks_p = feature_analysis._ks_2samp([1, 2, 3], [1, 2, 3])
        self.assertEqual(ks_stat, 0)
        self.assertEqual(ks_p, 1)

    def test_figures_and_saved_products(self):
        historical = feature_analysis.add_color_columns(
            self._frame([f"h{i}" for i in range(40)], 61100.0)
        )
        current = feature_analysis.add_color_columns(
            self._frame([f"c{i}" for i in range(20)], 61101.0, offset=0.2)
        )
        results = feature_analysis.compute_feature_diagnostics(
            historical,
            current,
            permutations=5,
            min_tag_historical=10,
            min_tag_current=10,
        )
        output = self.root / "products"
        metadata = feature_analysis.save_feature_products(
            results,
            output,
            coverage=pd.DataFrame([{"feature": "x"}]),
            analysis_context={"date_utc": "2026-03-02"},
        )
        expected = [
            "feature_coverage.csv",
            "cohort_summary.csv",
            "feature_statistics.csv",
            "tag_statistics.csv",
            "analysis_metadata.json",
            "variability_plane.png",
            "color_diagnostics.png",
        ]
        for filename in expected:
            self.assertTrue((output / filename).exists(), filename)
        color_figure = feature_analysis.plot_color_diagnostics(historical, current)
        self.assertTrue(color_figure.axes[0].yaxis_inverted())
        self.assertEqual(metadata["analysis_context"]["date_utc"], "2026-03-02")

    def test_coverage_only_products(self):
        output = self.root / "coverage-only"
        metadata = feature_analysis.save_feature_coverage_audit(
            pd.DataFrame([{"feature": "missing", "column_present": False}]),
            pd.DataFrame([{"cohort": "historical", "loci": 0}]),
            pd.DataFrame(
                [{"plane": "variability_r", "available_in_both_cohorts": False}]
            ),
            output,
        )
        self.assertEqual(
            metadata["status"], "coverage_only_no_usable_feature_plane"
        )
        self.assertTrue((output / "analysis_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
