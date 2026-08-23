import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src import cli


class CliPhase3Tests(unittest.TestCase):
    def _run(self, arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def _snapshot(root):
        snapshot = []
        for path in sorted(Path(root).rglob("*")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            snapshot.append((path.relative_to(root).as_posix(), digest))
        return snapshot

    def test_help_adds_planning_without_writer_commands(self):
        status, stdout, stderr = self._run([])

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertIn("night", stdout)
        self.assertIn("backfill", stdout)
        self.assertNotIn("ingest", stdout)
        self.assertNotIn("backfill run", stdout)

    def test_night_plan_json_is_read_only_and_versioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            before = self._snapshot(root)
            status, stdout, stderr = self._run(
                [
                    "night",
                    "plan",
                    "2026-06-27",
                    "--data-root",
                    str(root),
                    "--cache-root",
                    str(cache),
                    "--json",
                ]
            )
            after = self._snapshot(root)

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(before, after)
        payload = json.loads(stdout)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(payload["operation"], "night.plan")
        self.assertTrue(payload["read_only"])
        self.assertFalse(payload["details"]["future_execution"]["authorized"])
        refusal_codes = {item["code"] for item in payload["refusal_reasons"]}
        self.assertIn("execution-authorization", refusal_codes)

    def test_invalid_date_uses_invalid_request_exit_code_without_traceback(self):
        status, stdout, stderr = self._run(
            [
                "night",
                "plan",
                "not-a-date",
                "--data-root",
                "/tmp/data",
                "--cache-root",
                "/tmp/cache",
                "--json",
            ]
        )

        self.assertEqual(status, 2)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout)["status"], "invalid_request")
        self.assertNotIn("Traceback", stdout)

    def test_backfill_plan_is_inclusive_and_sequential(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            cache = Path(temporary) / "cache"
            root.mkdir()
            status, stdout, stderr = self._run(
                [
                    "backfill",
                    "plan",
                    "2026-06-27",
                    "2026-06-29",
                    "--data-root",
                    str(root),
                    "--cache-root",
                    str(cache),
                    "--json",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(stderr, "")
        payload = json.loads(stdout)
        self.assertEqual(payload["counts"]["requested_nights"], 3)
        self.assertEqual(payload["details"]["ordering"], "sequential_ascending")
        self.assertTrue(payload["details"]["stop_on_first_anomaly"])
        self.assertFalse(payload["details"]["writer_enabled"])


if __name__ == "__main__":
    unittest.main()
