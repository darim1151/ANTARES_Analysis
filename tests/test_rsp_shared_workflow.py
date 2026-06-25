import grp
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat

from src import config, rsp_permissions


REPO_ROOT = Path(__file__).resolve().parents[1]


class RspSharedWorkflowTests(unittest.TestCase):
    def _primary_group(self):
        return grp.getgrgid(os.getgid()).gr_name

    def _setgid_supported(self, directory):
        probe = Path(directory) / "setgid_probe"
        probe.mkdir()
        probe.chmod(probe.stat().st_mode | stat.S_ISGID)
        return bool(probe.stat().st_mode & stat.S_ISGID)

    def test_config_defaults_and_env_override(self):
        self.assertEqual(
            config.DEFAULT_DATA_ROOT,
            Path("/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["ANTARES_ANALYSIS_DATA_ROOT"] = str(Path(tmp) / "custom-root")
            env.pop("ANTARES_DATA_ROOT", None)
            out = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from src import config; print(config.DATA_ROOT)",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
            ).strip()
            self.assertEqual(out, str(Path(tmp) / "custom-root"))

    def test_permission_helpers_create_group_shared_tree(self):
        expected_group = self._primary_group()
        with tempfile.TemporaryDirectory() as tmp:
            setgid_supported = self._setgid_supported(tmp)
            data_root = Path(tmp) / "ANTARES_Analysis_Data"
            created = rsp_permissions.ensure_project_directories(
                data_root,
                expected_group=expected_group,
            )
            self.assertIn(data_root / "cache", created)
            for path in created:
                summary = rsp_permissions.path_permission_summary(path)
                self.assertTrue(summary["exists"], path)
                self.assertTrue(summary["group_writable"], path)
                if setgid_supported:
                    self.assertTrue(summary["setgid"], path)
                self.assertFalse(path.stat().st_mode & stat.S_IWOTH, path)

            test_file = data_root / "cache" / "permission_test_unit.txt"
            test_file.write_text("ok\n", encoding="utf-8")
            rsp_permissions.mark_file_group_writable(test_file)
            file_summary = rsp_permissions.path_permission_summary(test_file)
            self.assertTrue(file_summary["group_writable"])
            self.assertFalse(test_file.stat().st_mode & stat.S_IWOTH)

    def test_preflight_cli_accepts_temp_root_with_local_group(self):
        expected_group = self._primary_group()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "ANTARES_Analysis_Data"
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_rsp_shared_root.py",
                    "--data-root",
                    str(data_root),
                    "--expected-group",
                    expected_group,
                    "--set-umask",
                    "--allow-missing-setgid",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Preflight passed", result.stdout)

    def test_notebooks_are_rsp_native_and_output_clean(self):
        notebook_paths = [
            REPO_ROOT / "notebooks" / "rsp_setup.ipynb",
            REPO_ROOT / "notebooks" / "historical_backfill.ipynb",
            REPO_ROOT / "notebooks" / "alerts_time_comparison.ipynb",
        ]
        forbidden = [
            'os.environ["ANTARES_DATA_ROOT"]',
            'DATA_ROOT = Path("/home/mdarim/AntaresAlerts/ANTARES_Analysis_Data")',
            "google.colab",
            "/content",
            "drive.mount",
        ]
        for path in notebook_paths:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                self.assertNotIn(pattern, text, f"{path} contains {pattern}")
            self.assertIn("require_shared_data_root", text, path)
            self.assertIn("os.umask(0o002)", text, path)

            nb = nbformat.read(path, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    self.assertEqual(cell.get("outputs", []), [], path)
                    self.assertIsNone(cell.get("execution_count"), path)


if __name__ == "__main__":
    unittest.main()
