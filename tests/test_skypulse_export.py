import unittest

from scripts.export_skypulse_public_data import (
    ExportError,
    comparison_payload,
    make_density,
    make_demo_points,
    mjd_to_utc_date,
)

import pandas as pd


class SkyPulseExportTests(unittest.TestCase):
    def test_mjd_calendar_conversion_and_demo_manifest_agree(self):
        self.assertEqual(mjd_to_utc_date(61102.0), "2026-03-03")
        frame = pd.DataFrame(
            [
                {
                    "locus_id": "ANT-DEMO",
                    "ra": 10.0,
                    "dec": -5.0,
                    "brightness_mag": 19.0,
                    "obs_count": 2,
                    "tags": [],
                }
            ]
        )
        with self.assertRaisesRegex(ExportError, "does not match mjd_min"):
            make_demo_points(
                frame,
                {"mjd_min": 61102.0, "mjd_max": 61103.0, "date_utc": "2026-04-28"},
            )

    def test_density_bins_cover_coordinate_boundaries_once(self):
        points = [
            {"ra": 0.0, "dec": -90.0, "is_last_night": True},
            {"ra": 359.999, "dec": 90.0, "is_last_night": False},
            {"ra": 15.0, "dec": 0.0, "is_last_night": True},
        ]

        tiles = make_density(points, 15.0, 10.0)

        self.assertEqual(sum(tile["count"] for tile in tiles), len(points))
        self.assertTrue(all(0 <= tile["ra_min"] < tile["ra_max"] <= 360 for tile in tiles))
        self.assertTrue(all(-90 <= tile["dec_min"] < tile["dec_max"] <= 90 for tile in tiles))
        for point in points:
            matches = [
                tile
                for tile in tiles
                if tile["ra_min"] <= point["ra"] < tile["ra_max"]
                and tile["dec_min"] <= point["dec"]
                and (
                    point["dec"] < tile["dec_max"]
                    or (point["dec"] == 90 and tile["dec_max"] == 90)
                )
            ]
            self.assertEqual(len(matches), 1)

    def test_density_rejects_nonpositive_or_nonfinite_bins(self):
        with self.assertRaises(ExportError):
            make_density([], 0.0, 10.0)
        with self.assertRaises(ExportError):
            make_density([], 15.0, float("nan"))

    def test_comparison_describes_shipped_sample_not_uncapped_sources(self):
        last_points = [{"seen_before": True}, {"seen_before": False}]
        historical_points = [{"seen_before": False}]

        comparison = comparison_payload(
            last_points,
            historical_points,
            alert_rows=250_000,
            candidates=[{}, {}],
            tiles=[{}],
        )

        self.assertEqual(comparison["night_loci"], 2)
        self.assertEqual(comparison["historical_loci"], 1)
        self.assertEqual(comparison["new_loci"], 1)
        self.assertEqual(comparison["overlap_loci"], 1)
        self.assertEqual(comparison["overlap_fraction_of_night"], 0.5)
        self.assertEqual(comparison["alert_rows"], 250_000)
        self.assertEqual(comparison["highlighted_objects"], 2)
        self.assertEqual(comparison["density_tiles"], 1)


if __name__ == "__main__":
    unittest.main()
