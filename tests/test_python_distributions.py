import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_python_distributions.py"
SPEC = importlib.util.spec_from_file_location("verify_python_distributions", SCRIPT_PATH)
distributions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distributions)


class PythonDistributionTests(unittest.TestCase):
    def make_source(self, root):
        (root / "src").mkdir(parents=True)
        (root / "src" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (root / "README.md").write_text("# Example\n", encoding="utf-8")

    def make_sdist(self, path, source_root, *, mtime, metadata="Metadata-Version: 2.4\n"):
        files = {
            "README.md": (source_root / "README.md").read_bytes(),
            "pyproject.toml": (source_root / "pyproject.toml").read_bytes(),
            "src/__init__.py": (source_root / "src" / "__init__.py").read_bytes(),
            "PKG-INFO": metadata.encode("utf-8"),
        }
        with tarfile.open(path, "w:gz") as archive:
            for relative, content in files.items():
                info = tarfile.TarInfo(f"example-1.0/{relative}")
                info.mode = 0o644
                info.mtime = mtime
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    def test_timestamp_differences_do_not_break_sdist_equivalence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            self.make_source(source_root)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first_sdist = first_dir / "example-1.0.tar.gz"
            second_sdist = second_dir / "example-1.0.tar.gz"
            self.make_sdist(first_sdist, source_root, mtime=1)
            self.make_sdist(second_sdist, source_root, mtime=9_999)
            first_wheel = first_dir / "example-1.0-py3-none-any.whl"
            second_wheel = second_dir / first_wheel.name
            first_wheel.write_bytes(b"deterministic wheel")
            second_wheel.write_bytes(b"deterministic wheel")

            distributions.verify_distributions(
                first_wheel,
                second_wheel,
                first_sdist,
                second_sdist,
                source_root,
            )

    def test_sdist_content_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_root = root / "source"
            self.make_source(source_root)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first_sdist = first_dir / "example-1.0.tar.gz"
            second_sdist = second_dir / "example-1.0.tar.gz"
            self.make_sdist(first_sdist, source_root, mtime=1)
            self.make_sdist(
                second_sdist,
                source_root,
                mtime=2,
                metadata="Metadata-Version: 2.4\nName: changed\n",
            )
            first_wheel = first_dir / "example-1.0-py3-none-any.whl"
            second_wheel = second_dir / first_wheel.name
            first_wheel.write_bytes(b"deterministic wheel")
            second_wheel.write_bytes(b"deterministic wheel")

            with self.assertRaisesRegex(ValueError, "not content-equivalent"):
                distributions.verify_distributions(
                    first_wheel,
                    second_wheel,
                    first_sdist,
                    second_sdist,
                    source_root,
                )


if __name__ == "__main__":
    unittest.main()
