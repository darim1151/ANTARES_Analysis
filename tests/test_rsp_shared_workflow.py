import grp
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nbformat
import pandas as pd

from src import cache, config, rsp_permissions


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ENV_KEYS = [
    "ANTARES_ANALYSIS_DATA_ROOT",
    "ANTARES_DATA_ROOT",
    "ANTARES_ANALYSIS_CACHE_ROOT",
    "ANTARES_STORAGE_POLICY",
    "ANTARES_SHARED_GROUP",
]


class StoragePortabilityTests(unittest.TestCase):
    def _primary_group(self):
        return grp.getgrgid(os.getgid()).gr_name

    def _setgid_supported(self, directory):
        probe = Path(directory) / "setgid_probe"
        probe.mkdir()
        probe.chmod(probe.stat().st_mode | stat.S_ISGID)
        return bool(probe.stat().st_mode & stat.S_ISGID)

    def _config_subprocess(self, updates=None, code=None):
        env = os.environ.copy()
        for key in STORAGE_ENV_KEYS:
            env.pop(key, None)
        env.update(updates or {})
        program = code or (
            "import json\n"
            "from src import config\n"
            "print(json.dumps({\n"
            "  'data': str(config.DATA_ROOT),\n"
            "  'cache': str(config.CACHE_ROOT),\n"
            "  'policy': config.STORAGE_POLICY,\n"
            "  'group': config.SHARED_GROUP,\n"
            "}))\n"
        )
        return subprocess.run(
            [sys.executable, "-c", program],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_config_defaults_remain_compatible_and_private_safe(self):
        result = self._config_subprocess()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(
            payload["data"],
            "/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data",
        )
        self.assertEqual(payload["cache"], payload["data"] + "/cache")
        self.assertEqual(payload["policy"], "private")
        self.assertIsNone(payload["group"])

    def test_config_independent_roots_and_canonical_data_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            durable = base / "durable"
            legacy = base / "legacy"
            external_cache = base / "external-cache"
            result = self._config_subprocess(
                {
                    "ANTARES_ANALYSIS_DATA_ROOT": str(durable),
                    "ANTARES_DATA_ROOT": str(legacy),
                    "ANTARES_ANALYSIS_CACHE_ROOT": str(external_cache),
                    "ANTARES_STORAGE_POLICY": "private",
                    "ANTARES_SHARED_GROUP": "must-be-ignored",
                }
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["data"], str(durable))
            self.assertEqual(payload["cache"], str(external_cache))
            self.assertEqual(payload["policy"], "private")
            self.assertIsNone(payload["group"])
            self.assertFalse(durable.exists())
            self.assertFalse(external_cache.exists())
            self.assertFalse((durable / "cache").exists())

    def test_config_cache_fallback_and_legacy_data_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy-root"
            result = self._config_subprocess(
                {"ANTARES_DATA_ROOT": str(legacy)}
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["data"], str(legacy))
            self.assertEqual(payload["cache"], str(legacy / "cache"))
            self.assertFalse(legacy.exists())

    def test_config_shared_group_is_explicit_and_policy_is_validated(self):
        result = self._config_subprocess(
            {
                "ANTARES_STORAGE_POLICY": "shared-group",
                "ANTARES_SHARED_GROUP": "fixture_group",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["policy"], "shared-group")
        self.assertEqual(payload["group"], "fixture_group")

        invalid = self._config_subprocess(
            {"ANTARES_STORAGE_POLICY": "hostname-magic"}
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("Invalid ANTARES_STORAGE_POLICY", invalid.stderr)

    def test_importing_config_creates_no_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            durable = Path(tmp) / "new" / "durable"
            external_cache = Path(tmp) / "new" / "cache"
            code = (
                "import json\n"
                "from src import config\n"
                "print(json.dumps([config.DATA_ROOT.exists(), "
                "config.CACHE_ROOT.exists(), (config.DATA_ROOT / 'cache').exists()]))\n"
            )
            result = self._config_subprocess(
                {
                    "ANTARES_ANALYSIS_DATA_ROOT": str(durable),
                    "ANTARES_ANALYSIS_CACHE_ROOT": str(external_cache),
                    "ANTARES_STORAGE_POLICY": "private",
                },
                code=code,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [False, False, False])
            self.assertFalse((durable / "cache").exists())

    def test_private_helpers_never_enter_group_widening_logic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-root"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            nested = root / "new" / "nested"
            with (
                mock.patch.object(
                    rsp_permissions, "ensure_group_shared_path"
                ) as shared_helper,
                mock.patch.object(rsp_permissions.grp, "getgrnam") as getgrnam,
                mock.patch.object(rsp_permissions.os, "chown") as chown,
            ):
                rsp_permissions.ensure_storage_path(
                    nested,
                    policy="private",
                    expected_group="does-not-exist",
                )
                shared_helper.assert_not_called()
                getgrnam.assert_not_called()
                chown.assert_not_called()

            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "new").stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(nested.stat().st_mode), 0o700)
            self.assertFalse(nested.stat().st_mode & stat.S_ISGID)

            output = nested / "result.parquet"
            output.write_text("fixture\n", encoding="utf-8")
            output.chmod(0o666)
            with mock.patch.object(
                rsp_permissions, "mark_file_group_writable"
            ) as group_marker:
                rsp_permissions.mark_file_for_storage(output, policy="private")
                group_marker.assert_not_called()
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)

    def test_project_directory_helper_defaults_to_configured_private_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-project"
            with mock.patch.object(
                rsp_permissions, "ensure_group_shared_path"
            ) as shared_helper:
                created = rsp_permissions.ensure_project_directories(root)
                shared_helper.assert_not_called()
            self.assertNotIn(root / "cache", created)
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)

    def test_private_helper_does_not_normalize_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "existing"
            root.mkdir(mode=0o750)
            root.chmod(0o750)
            before = stat.S_IMODE(root.stat().st_mode)
            rsp_permissions.ensure_storage_path(root, policy="private")
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), before)

            report = rsp_permissions.check_storage_root(
                root, policy="private", write_test=False
            )
            self.assertFalse(report["ok"])
            self.assertTrue(any("group/world" in item for item in report["failures"]))
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), before)

    def test_noncreating_preflight_does_not_create_data_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            durable = Path(tmp) / "durable"
            external_cache = Path(tmp) / "external-cache"
            report = rsp_permissions.check_storage_root(
                durable,
                cache_root=external_cache,
                policy="private",
                write_test=False,
            )
            self.assertFalse(report["ok"])
            self.assertFalse(durable.exists())
            self.assertFalse(external_cache.exists())
            self.assertFalse((durable / "cache").exists())

            durable.mkdir(mode=0o700)
            durable.chmod(0o700)
            report = rsp_permissions.check_storage_root(
                durable,
                cache_root=external_cache,
                policy="private",
                write_test=False,
            )
            self.assertTrue(report["ok"], report["failures"])
            self.assertFalse(external_cache.exists())

    def test_preflight_rejects_a_regular_file_as_a_storage_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "not-a-directory"
            data_root.write_text("fixture\n", encoding="utf-8")
            data_root.chmod(0o700)

            report = rsp_permissions.check_storage_root(
                data_root,
                policy="private",
                write_test=True,
            )

            self.assertFalse(report["ok"])
            self.assertFalse(report["paths"][0]["is_directory"])
            self.assertTrue(
                any("not a directory" in item for item in report["failures"])
            )
            self.assertEqual(report["test_file"], {})

    def test_shared_preflight_checks_every_existing_managed_directory(self):
        expected_group = self._primary_group()
        with tempfile.TemporaryDirectory() as tmp:
            setgid_supported = self._setgid_supported(tmp)
            data_root = Path(tmp) / "shared-root"
            cache_root = Path(tmp) / "shared-cache"
            rsp_permissions.ensure_project_directories(
                data_root,
                cache_root=cache_root,
                policy="shared-group",
                expected_group=expected_group,
            )
            nightly_root = data_root / "data" / "lsst_only" / "nightly"
            nightly_root.chmod(0o700)
            cache_root.chmod(0o700)

            report = rsp_permissions.check_storage_root(
                data_root,
                cache_root=cache_root,
                policy="shared-group",
                expected_group=expected_group,
                write_test=False,
                require_setgid=setgid_supported,
            )
            self.assertFalse(report["ok"])
            self.assertTrue(
                any(str(nightly_root) in item for item in report["failures"])
            )
            self.assertTrue(
                any(str(cache_root) in item for item in report["failures"])
            )

            rsp_permissions.ensure_storage_path(
                nightly_root,
                policy="shared-group",
                expected_group=expected_group,
            )
            rsp_permissions.ensure_storage_path(
                cache_root,
                policy="shared-group",
                expected_group=expected_group,
            )
            repaired = rsp_permissions.check_storage_root(
                data_root,
                cache_root=cache_root,
                policy="shared-group",
                expected_group=expected_group,
                write_test=False,
                require_setgid=setgid_supported,
            )
            self.assertTrue(repaired["ok"], repaired["failures"])

    def test_shared_group_helpers_preserve_rsp_behavior(self):
        expected_group = self._primary_group()
        with tempfile.TemporaryDirectory() as tmp:
            setgid_supported = self._setgid_supported(tmp)
            data_root = Path(tmp) / "ANTARES_Analysis_Data"
            created = rsp_permissions.ensure_project_directories(
                data_root,
                expected_group=expected_group,
                cache_root=data_root / "cache",
                policy="shared-group",
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
            rsp_permissions.mark_file_for_storage(
                test_file,
                policy="shared-group",
                expected_group=expected_group,
            )
            self.assertTrue(test_file.stat().st_mode & stat.S_IWGRP)
            self.assertFalse(test_file.stat().st_mode & stat.S_IWOTH)

    def test_external_cache_writes_never_create_in_tree_cache_or_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "durable"
            cache_root = Path(tmp) / "external-cache"
            data_root.mkdir(mode=0o700)
            data_root.chmod(0o700)
            paths = cache.cache_paths(
                cache_root, 1.0, 2.0, 3.0, 4.0, 5
            )
            frames = [
                pd.DataFrame({"locus_id": [name]})
                for name in ("one", "two", "three", "four")
            ]
            cache.save_cache(paths, *frames)
            for path in paths.values():
                resolved = Path(path)
                self.assertTrue(resolved.is_file())
                self.assertEqual(resolved.parent, cache_root)
            self.assertFalse((data_root / "cache").exists())
            self.assertFalse(cache_root.is_symlink())

    def test_policy_umask_is_explicit_and_restored_by_test(self):
        original = os.umask(0)
        os.umask(original)
        try:
            private = rsp_permissions.configure_process_umask("private")
            self.assertEqual(private["current"], "077")
            shared = rsp_permissions.configure_process_umask("shared-group")
            self.assertEqual(shared["current"], "002")
        finally:
            os.umask(original)

    def test_rsp_compatibility_cli_checks_existing_temp_root(self):
        expected_group = self._primary_group()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "ANTARES_Analysis_Data"
            rsp_permissions.ensure_storage_path(
                data_root,
                policy="shared-group",
                expected_group=expected_group,
            )
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
            self.assertFalse((data_root / "cache").exists())

    def test_notebooks_are_policy_portable_and_output_clean(self):
        notebook_paths = [
            REPO_ROOT / "notebooks" / "rsp_setup.ipynb",
            REPO_ROOT / "notebooks" / "historical_backfill.ipynb",
            REPO_ROOT / "notebooks" / "alerts_time_comparison.ipynb",
        ]
        forbidden_literals = [
            'os.environ["ANTARES_DATA_ROOT"]',
            "google.colab",
            "/content",
            "drive.mount",
            "require_shared_data_root",
            "ensure_group_shared_path",
            "mark_file_group_writable",
            "os.umask(0o002)",
            "/home/ivezic/AntaresAlerts/ANTARES_Analysis_Data",
            "/astro/store/shire/ANTARES_Analysis_Data",
            "/astro/store/shire/ANTARES/data",
        ]
        direct_cache = re.compile(r"DATA_ROOT\s*/\s*['\"]cache['\"]")
        manifest_authority = re.compile(
            r"Path\([^\n]*manifest\s*\[['\"]paths['\"]\]"
        )

        code_by_name = {}
        for path in notebook_paths:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            code_text = "\n".join(
                cell.source for cell in nb.cells if cell.cell_type == "code"
            )
            code_by_name[path.name] = code_text
            for pattern in forbidden_literals:
                self.assertNotIn(pattern, code_text, f"{path} contains {pattern}")
            self.assertIsNone(direct_cache.search(code_text), path)
            self.assertIsNone(manifest_authority.search(code_text), path)
            self.assertIn("STORAGE_POLICY", code_text, path)
            self.assertIn("require_storage_root", code_text, path)
            self.assertIn("configure_process_umask", code_text, path)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    self.assertEqual(cell.get("outputs", []), [], path)
                    self.assertIsNone(cell.get("execution_count"), path)

        setup = code_by_name["rsp_setup.ipynb"]
        self.assertIn("CREATE_CACHE_ROOT = False", setup)
        self.assertIn("SCRATCH_DIR", setup)
        self.assertNotIn('Path("/scratch', setup)

        historical = code_by_name["historical_backfill.ipynb"]
        comparison = code_by_name["alerts_time_comparison.ipynb"]
        self.assertIn("CACHE_ROOT / CACHE_VERSION", historical)
        self.assertIn("CACHE_ROOT / CACHE_VERSION", comparison)
        self.assertIn('CACHE_VERSION = "probe50_v1"', historical)
        self.assertIn(
            'CACHE_VERSION = "probe50_time_ra_dec_v1"', comparison
        )
        self.assertIn('current_partition_dir / "loci.parquet"', comparison)
        self.assertIn('current_partition_dir / "alerts.parquet"', comparison)

    def test_permission_module_contains_no_acl_mutation(self):
        source = (REPO_ROOT / "src" / "rsp_permissions.py").read_text(
            encoding="utf-8"
        )
        for pattern in ("setfacl", "chmod +a", "acl_set", "setxattr"):
            self.assertNotIn(pattern, source)


if __name__ == "__main__":
    unittest.main()
