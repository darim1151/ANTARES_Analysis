import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PackageInitializationTests(unittest.TestCase):
    def test_version_is_independent_of_storage_configuration(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["ANTARES_STORAGE_POLICY"] = "intentionally-invalid"
        completed = subprocess.run(
            [sys.executable, "-m", "src.cli", "--version"],
            cwd=tempfile.gettempdir(),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertRegex(completed.stdout.strip(), r"^antares-analysis \d+\.\d+\.\d+$")

    def test_public_modules_remain_lazily_compatible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(REPO_ROOT)
            environment["ANTARES_ANALYSIS_DATA_ROOT"] = str(root / "data")
            environment["ANTARES_ANALYSIS_CACHE_ROOT"] = str(root / "cache")
            environment["ANTARES_STORAGE_POLICY"] = "private"
            code = (
                "import src; "
                "assert 'config' not in vars(src); "
                "from src import config; "
                "assert 'config' in vars(src); "
                "assert config.STORAGE_POLICY == 'private'"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tempfile.gettempdir(),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((root / "data").exists())
            self.assertFalse((root / "cache").exists())

    def test_middle_earth_data_override_uses_canonical_sibling_cache_without_io(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT)
        environment["ANTARES_ANALYSIS_DATA_ROOT"] = "/astro/store/shire/ANTARES/data"
        environment.pop("ANTARES_ANALYSIS_CACHE_ROOT", None)
        environment["ANTARES_STORAGE_POLICY"] = "private"
        code = (
            "from src import config; "
            "assert str(config.DATA_ROOT) == '/astro/store/shire/ANTARES/data'; "
            "assert str(config.CACHE_ROOT) == '/astro/store/shire/ANTARES/cache'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tempfile.gettempdir(),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
