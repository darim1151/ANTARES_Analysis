import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_release_artifacts", SCRIPT_PATH)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseArtifactTests(unittest.TestCase):
    def test_lock_parser_and_layer_accept_exact_production_superset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            production_path = root / "production.lock"
            test_path = root / "test.lock"
            production_path.write_text(
                "--index-url https://pypi.org/simple\n"
                "Example_Pkg==1.2.3 \\\n"
                f"    --hash=sha256:{'a' * 64}\n",
                encoding="utf-8",
            )
            test_path.write_text(
                "example-pkg==1.2.3 \\\n"
                f"    --hash=sha256:{'a' * 64}\n"
                "nbformat==5.10.4 \\\n"
                f"    --hash=sha256:{'b' * 64}\n",
                encoding="utf-8",
            )

            production = release.locked_versions(production_path)
            test = release.locked_versions(test_path)
            release.verify_lock_layer(production, test)

        self.assertEqual(production, {"example-pkg": "1.2.3"})

    def test_lock_parser_rejects_unhashed_exact_pin(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "unhashed.lock"
            lock_path.write_text("example==1.2.3\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "has no SHA-256 hash"):
                release.locked_versions(lock_path)

    def test_lock_layer_rejects_transitive_version_drift(self):
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            release.verify_lock_layer({"example": "1.0"}, {"example": "2.0"})

    def test_wheelhouses_require_identical_shared_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            production_directory = root / "production"
            test_directory = root / "test"
            production_directory.mkdir()
            test_directory.mkdir()
            filename = "example_pkg-1.2.3-py3-none-any.whl"
            (production_directory / filename).write_bytes(b"same wheel")
            (test_directory / filename).write_bytes(b"same wheel")
            (test_directory / "nbformat-5.10.4-py3-none-any.whl").write_bytes(
                b"test wheel"
            )

            production = release.wheel_inventory(production_directory)
            test = release.wheel_inventory(test_directory)
            release.verify_wheelhouse(
                {"example-pkg": "1.2.3"}, production, "production"
            )
            release.verify_wheelhouse(
                {"example-pkg": "1.2.3", "nbformat": "5.10.4"},
                test,
                "test",
            )
            release.verify_shared_wheels(production, test)

            (test_directory / filename).write_bytes(b"different wheel")
            changed_test = release.wheel_inventory(test_directory)
            with self.assertRaisesRegex(ValueError, "disagree"):
                release.verify_shared_wheels(production, changed_test)


if __name__ == "__main__":
    unittest.main()
