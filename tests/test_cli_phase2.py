import contextlib
import hashlib
import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import cli
from src import cli_diagnostics
from src.cli_profiles import (
    MIDDLE_EARTH_CANARY_ROOT,
    MIDDLE_EARTH_WORK_ROOT,
    middle_earth_work_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _PythonVersion(tuple):
    major = property(lambda value: value[0])
    minor = property(lambda value: value[1])
    micro = property(lambda value: value[2])


class CliPhase2Tests(unittest.TestCase):
    def _run(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def _write_manifest(
        self,
        root,
        date_utc,
        *,
        loci,
        alerts,
        status="complete",
        append_ready=True,
    ):
        year, month, day = date_utc.split("-")
        directory = root / "data" / "lsst_only" / "nightly" / year / month / day
        directory.mkdir(parents=True)
        payload = {
            "date_utc": date_utc,
            "status": status,
            "actual_loci": loci,
            "alert_rows": alerts,
            "validation": {"append_ready": append_ready},
        }
        (directory / "manifest.json").write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _dataset(self, base):
        root = Path(base) / "ANTARES_Analysis_Data"
        cumulative = root / "data" / "lsst_only" / "cumulative"
        cumulative.mkdir(parents=True)
        (cumulative / "loci_index.parquet").write_bytes(b"PAR1-loci-fixture")
        (cumulative / "nightly_summary.parquet").write_bytes(b"PAR1-summary-fixture")
        self._write_manifest(root, "2026-03-03", loci=12, alerts=40)
        self._write_manifest(
            root,
            "2026-03-05",
            loci=0,
            alerts=0,
            append_ready=False,
        )
        os.chmod(root, 0o700)
        return root

    def _snapshot(self, root):
        snapshot = []
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                snapshot.append((relative, "file", digest))
            elif path.is_dir():
                snapshot.append((relative, "directory", None))
        return snapshot

    def _notebook_checkout(self, base):
        root = Path(base) / "checkout with $ characters"
        notebooks = root / "notebooks"
        notebooks.mkdir(parents=True)
        (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
        for spec in cli.NOTEBOOKS:
            (notebooks / spec.filename).write_text("{}\n", encoding="utf-8")
        return root

    def test_help_exposes_navigable_phase2_commands(self):
        status, stdout, stderr = self._run([])

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        for command in ("profile", "doctor", "data", "jupyter"):
            self.assertIn(command, stdout)
        self.assertIn("read-only", stdout.lower())

    def test_profile_list_json_contains_middle_earth_contract(self):
        status, stdout, stderr = self._run(["profile", "list", "--json"])

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        profiles = {profile["name"]: profile for profile in payload["profiles"]}
        middle_earth = profiles["middle-earth"]
        self.assertEqual(
            middle_earth["data_root"],
            "/astro/store/shire/ANTARES/data",
        )
        self.assertEqual(
            middle_earth["cache_root"],
            "/astro/store/shire/ANTARES/cache",
        )
        self.assertEqual(middle_earth["storage_policy"], "private")
        self.assertTrue(payload["read_only"])

    def test_profile_export_is_copy_paste_safe_and_private(self):
        status, stdout, stderr = self._run(
            ["profile", "export", "--profile", "middle-earth"]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn(
            "export ANTARES_ANALYSIS_DATA_ROOT=/astro/store/shire/ANTARES/data",
            stdout,
        )
        self.assertIn(
            "export ANTARES_ANALYSIS_CACHE_ROOT=/astro/store/shire/ANTARES/cache",
            stdout,
        )
        self.assertIn("export ANTARES_STORAGE_POLICY=private", stdout)
        self.assertIn("unset ANTARES_SHARED_GROUP", stdout)

    def test_middle_earth_operational_paths_are_confined_to_work(self):
        self.assertEqual(
            middle_earth_work_path(Path("canary") / "run-20260825T000000Z"),
            MIDDLE_EARTH_CANARY_ROOT / "run-20260825T000000Z",
        )
        self.assertEqual(
            middle_earth_work_path(
                MIDDLE_EARTH_WORK_ROOT / "qualification" / "run-1"
            ),
            MIDDLE_EARTH_WORK_ROOT / "qualification" / "run-1",
        )
        refused = (
            Path("/astro/store/shire/ANTARES_canary_run-1"),
            Path("/astro/store/shire/ANTARES_Analysis_infra_canary_run-1"),
            Path("../cache"),
            MIDDLE_EARTH_WORK_ROOT,
        )
        for path in refused:
            with self.subTest(path=path), self.assertRaises(ValueError):
                middle_earth_work_path(path)

    def test_environment_profile_uses_explicit_process_configuration(self):
        values = {
            "ANTARES_ANALYSIS_PROFILE": "environment",
            "ANTARES_ANALYSIS_DATA_ROOT": "/data/custom root",
            "ANTARES_ANALYSIS_CACHE_ROOT": "/cache/custom root",
            "ANTARES_STORAGE_POLICY": "private",
        }
        with mock.patch.dict(os.environ, values, clear=True):
            status, stdout, stderr = self._run(
                ["profile", "show", "--profile", "auto", "--json"]
            )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["name"], "environment")
        self.assertEqual(payload["data_root"], "/data/custom root")
        self.assertEqual(payload["cache_root"], "/cache/custom root")

    def test_profile_policy_overrides_fail_closed_on_ambiguous_group(self):
        status, stdout, stderr = self._run(
            [
                "profile",
                "show",
                "--profile",
                "middle-earth",
                "--storage-policy",
                "shared-group",
                "--json",
            ]
        )

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("pass --shared-group explicitly", stderr)

        status, stdout, stderr = self._run(
            [
                "profile",
                "show",
                "--profile",
                "middle-earth",
                "--shared-group",
                "unexpected-group",
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("cannot be used with the private", stderr)

    def test_data_status_summarizes_manifests_without_modifying_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            before = self._snapshot(root)
            status, stdout, stderr = self._run(
                [
                    "data",
                    "status",
                    "--profile",
                    "middle-earth",
                    "--data-root",
                    str(root),
                    "--cache-root",
                    str(Path(temporary) / "external-cache"),
                    "--json",
                ]
            )
            after = self._snapshot(root)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(after, before)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["summary"]["manifest_count"], 2)
        self.assertEqual(payload["summary"]["append_ready_nights"], 1)
        self.assertEqual(payload["summary"]["first_date"], "2026-03-03")
        self.assertEqual(payload["summary"]["last_date"], "2026-03-05")
        self.assertEqual(payload["summary"]["total_loci"], 12)
        self.assertEqual(payload["summary"]["total_alerts"], 40)
        self.assertEqual(payload["summary"]["zero_row_nights"], ["2026-03-05"])

    def test_data_status_fails_closed_for_missing_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            status, stdout, stderr = self._run(
                ["data", "status", "--data-root", str(missing), "--json"]
            )

        self.assertEqual(status, 1)
        self.assertEqual(stderr, "")
        self.assertFalse(json.loads(stdout)["ok"])

    def test_data_status_rejects_manifest_date_directory_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            manifest = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "03"
                / "03"
                / "manifest.json"
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["date_utc"] = "2026-03-04"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            status, stdout, _stderr = self._run(
                ["data", "status", "--data-root", str(root), "--json"]
            )

        self.assertEqual(status, 1)
        payload = json.loads(stdout)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("does not match" in error for error in payload["errors"]))

    def test_data_status_rejects_manifest_symlink_without_following_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            manifest = (
                root
                / "data"
                / "lsst_only"
                / "nightly"
                / "2026"
                / "03"
                / "03"
                / "manifest.json"
            )
            outside = Path(temporary) / "outside-manifest.json"
            manifest.replace(outside)
            manifest.symlink_to(outside)
            status, stdout, _stderr = self._run(
                ["data", "status", "--data-root", str(root), "--json"]
            )

        self.assertEqual(status, 1)
        payload = json.loads(stdout)
        self.assertTrue(any("must not be a symlink" in error for error in payload["errors"]))
        self.assertEqual(payload["summary"]["manifest_count"], 1)

    def test_data_status_rejects_empty_cumulative_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            (root / "data" / "lsst_only" / "cumulative" / "loci_index.parquet").write_bytes(
                b""
            )
            status, stdout, _stderr = self._run(
                ["data", "status", "--data-root", str(root), "--json"]
            )

        self.assertEqual(status, 1)
        self.assertFalse(json.loads(stdout)["ok"])

    def test_doctor_passes_control_plane_checks_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            before = self._snapshot(root)
            with mock.patch.object(
                cli_diagnostics.sys,
                "version_info",
                _PythonVersion((3, 11, 9)),
            ):
                status, stdout, stderr = self._run(
                    [
                        "doctor",
                        "--profile",
                        "middle-earth",
                        "--data-root",
                        str(root),
                        "--cache-root",
                        str(Path(temporary) / "external-cache"),
                        "--repo-root",
                        str(REPO_ROOT),
                        "--no-dependencies",
                        "--no-jupyter",
                        "--json",
                    ]
                )
            after = self._snapshot(root)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(after, before)
        payload = json.loads(stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["counts"]["fail"], 0)
        self.assertTrue(payload["read_only"])
        codes = {check["code"] for check in payload["checks"]}
        self.assertIn("cache-separation", codes)
        self.assertIn("dataset-layout", codes)
        self.assertIn("notebooks", codes)
        self.assertIn("private-owner", codes)

    def test_doctor_installed_runtime_without_checkout_is_informational(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            before = self._snapshot(root)
            with mock.patch.object(
                cli_diagnostics.sys,
                "version_info",
                _PythonVersion((3, 11, 9)),
            ), mock.patch.object(
                cli_diagnostics,
                "discover_repo_root",
                side_effect=ValueError("no checkout beside the installed wheel"),
            ) as discover:
                status, stdout, stderr = self._run(
                    [
                        "doctor",
                        "--data-root",
                        str(root),
                        "--cache-root",
                        str(Path(temporary) / "external-cache"),
                        "--no-dependencies",
                        "--no-jupyter",
                        "--json",
                    ]
                )
            after = self._snapshot(root)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(after, before)
        discover.assert_called_once_with(None)
        payload = json.loads(stdout)
        repository = next(
            check for check in payload["checks"] if check["code"] == "repository"
        )
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["counts"]["fail"], 0)
        self.assertEqual(repository["status"], "info")
        self.assertIn("notebook checks were skipped", repository["summary"])
        self.assertIn("--repo-root", repository["detail"])

    def test_doctor_explicit_missing_checkout_remains_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            missing_checkout = Path(temporary) / "missing-checkout"
            with mock.patch.object(
                cli_diagnostics.sys,
                "version_info",
                _PythonVersion((3, 11, 9)),
            ):
                status, stdout, stderr = self._run(
                    [
                        "doctor",
                        "--data-root",
                        str(root),
                        "--cache-root",
                        str(Path(temporary) / "external-cache"),
                        "--repo-root",
                        str(missing_checkout),
                        "--no-dependencies",
                        "--no-jupyter",
                        "--json",
                    ]
                )

        self.assertEqual(status, 1)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        repository = next(
            check for check in payload["checks"] if check["code"] == "repository"
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(repository["status"], "fail")
        self.assertIn("Requested source checkout", repository["summary"])

    def test_doctor_explicit_checkout_still_validates_notebooks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            checkout = self._notebook_checkout(temporary)
            missing_name = cli.NOTEBOOKS[-1].filename
            (checkout / "notebooks" / missing_name).unlink()
            with mock.patch.object(
                cli_diagnostics.sys,
                "version_info",
                _PythonVersion((3, 11, 9)),
            ):
                status, stdout, stderr = self._run(
                    [
                        "doctor",
                        "--data-root",
                        str(root),
                        "--cache-root",
                        str(Path(temporary) / "external-cache"),
                        "--repo-root",
                        str(checkout),
                        "--no-dependencies",
                        "--no-jupyter",
                        "--json",
                    ]
                )

        self.assertEqual(status, 1)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        notebooks = next(
            check for check in payload["checks"] if check["code"] == "notebooks"
        )
        self.assertFalse(payload["ok"])
        self.assertEqual(notebooks["status"], "fail")
        self.assertIn(missing_name, notebooks["detail"])

    def test_doctor_rejects_regular_file_as_cache_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._dataset(temporary)
            cache_file = Path(temporary) / "cache-file"
            cache_file.write_text("not a directory", encoding="utf-8")
            with mock.patch.object(
                cli_diagnostics.sys,
                "version_info",
                _PythonVersion((3, 11, 9)),
            ):
                status, stdout, _stderr = self._run(
                    [
                        "doctor",
                        "--data-root",
                        str(root),
                        "--cache-root",
                        str(cache_file),
                        "--repo-root",
                        str(REPO_ROOT),
                        "--no-dependencies",
                        "--no-jupyter",
                        "--json",
                    ]
                )

        self.assertEqual(status, 1)
        payload = json.loads(stdout)
        cache_check = next(
            check for check in payload["checks"] if check["code"] == "cache-root"
        )
        self.assertEqual(cache_check["status"], "fail")

    def test_jupyter_list_reports_all_supported_notebooks(self):
        status, stdout, stderr = self._run(
            ["jupyter", "list", "--repo-root", str(REPO_ROOT), "--json"]
        )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        aliases = {item["alias"] for item in json.loads(stdout)["notebooks"]}
        self.assertEqual(aliases, {"setup", "historical-backfill", "last-night"})

    def test_jupyter_command_quotes_paths_and_never_executes(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = self._notebook_checkout(temporary)
            data_root = Path(temporary) / "data with $ characters"
            cache_root = Path(temporary) / "cache with $ characters"
            status, stdout, stderr = self._run(
                [
                    "jupyter",
                    "command",
                    "setup",
                    "--repo-root",
                    str(checkout),
                    "--data-root",
                    str(data_root),
                    "--cache-root",
                    str(cache_root),
                    "--json",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        tokens = shlex.split(payload["command"])
        self.assertIn(f"ANTARES_ANALYSIS_DATA_ROOT={data_root}", tokens)
        self.assertIn(f"ANTARES_ANALYSIS_CACHE_ROOT={cache_root}", tokens)
        self.assertEqual(tokens[-3:-1], ["jupyter", "lab"])
        self.assertEqual(
            tokens[-1],
            str((checkout / "notebooks" / "rsp_setup.ipynb").resolve()),
        )
        self.assertFalse(payload["executed"])
        self.assertFalse(data_root.exists())
        self.assertFalse(cache_root.exists())

    def test_unknown_notebook_returns_usage_error_without_traceback(self):
        status, stdout, stderr = self._run(
            [
                "jupyter",
                "command",
                "unknown",
                "--repo-root",
                str(REPO_ROOT),
            ]
        )

        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("Unknown notebook", stderr)
        self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
