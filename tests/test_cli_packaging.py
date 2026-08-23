import contextlib
import importlib.util
import io
import re
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "src" / "cli.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
ENVIRONMENT_PATH = REPO_ROOT / "environment.yml"

RUNTIME_PINS = {
    "antares-client": "1.14.0",
    "astropy": "6.0.1",
    "matplotlib": "3.9.4",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
}

CLI_SPEC = importlib.util.spec_from_file_location("antares_analysis_cli", CLI_PATH)
cli = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(cli)


class CliPackagingTests(unittest.TestCase):
    def test_no_argument_cli_prints_help_and_succeeds(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main([])

        self.assertEqual(status, 0)
        self.assertIn("usage: antares-analysis", output.getvalue())
        self.assertIn("--version", output.getvalue())

    def test_version_uses_installed_distribution_metadata(self):
        output = io.StringIO()
        with (
            mock.patch.object(cli.metadata, "version", return_value="9.8.7"),
            contextlib.redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main(["--version"])

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), "antares-analysis 9.8.7")

    def test_source_version_matches_project_metadata(self):
        metadata_text = PYPROJECT_PATH.read_text(encoding="utf-8")
        match = re.search(
            r'^version\s*=\s*"([^"]+)"\s*$',
            metadata_text,
            flags=re.MULTILINE,
        )
        self.assertIsNotNone(match)
        self.assertEqual(cli.SOURCE_VERSION, match.group(1))

    def test_project_declares_supported_python_range_and_console_entry_point(self):
        metadata_text = PYPROJECT_PATH.read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.9,<3.12"', metadata_text)
        self.assertIn(
            'antares-analysis = "src.cli:main"',
            metadata_text,
        )
        self.assertIn('packages = ["src", "src.operations"]', metadata_text)

    def test_runtime_versions_are_explicitly_pinned(self):
        metadata_text = PYPROJECT_PATH.read_text(encoding="utf-8")
        for name, version in RUNTIME_PINS.items():
            with self.subTest(requirement=name):
                self.assertIn(f'"{name}=={version}"', metadata_text)

    def test_environment_input_matches_runtime_pins(self):
        environment_text = ENVIRONMENT_PATH.read_text(encoding="utf-8")
        self.assertIn("  - python=3.11", environment_text)
        for name, version in RUNTIME_PINS.items():
            with self.subTest(requirement=name):
                expected = f"      - {name}=={version}" if name == "antares-client" else f"  - {name}={version}"
                self.assertIn(expected, environment_text)


if __name__ == "__main__":
    unittest.main()
