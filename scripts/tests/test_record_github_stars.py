"""Exercise star snapshots against a local HTTP server."""

import datetime as dt
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import record_github_stars as stars

ROOT = Path(__file__).resolve().parents[2]


class TestGitHubStars(unittest.TestCase):
    def setUp(self):
        self.payloads = {
            "/repos/onnx/onnx": {"stargazers_count": 10},
            "/repos/xadupre/onnx-light": {"stargazers_count": 0},
        }
        case = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in case.payloads:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(case.payloads[self.path]).encode())

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.original_api = stars.API
        stars.API = f"http://127.0.0.1:{self.server.server_port}"
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "github-stars.json"

    def tearDown(self):
        stars.API = self.original_api
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temporary.cleanup()

    def record(self, day=12, repositories=None, now=None):
        return stars.record_stars(
            self.temporary.name,
            repositories=(
                ["onnx/onnx", "xadupre/onnx-light"]
                if repositories is None else repositories
            ),
            now=now or dt.datetime(2026, 11, day, 6, tzinfo=dt.UTC),
        )

    def test_daily_snapshots_replace_preserve_and_sort(self):
        first = self.record()
        self.assertEqual(first["history"], [{
            "date": "2026-11-12",
            "stars": {"onnx/onnx": 10, "xadupre/onnx-light": 0},
        }])
        self.payloads["/repos/onnx/onnx"]["stargazers_count"] = 9
        same_day = self.record(repositories=["onnx/onnx", "onnx/onnx"])
        self.assertEqual(len(same_day["history"]), 1)
        self.assertEqual(same_day["history"][0]["stars"],
                         {"onnx/onnx": 9, "xadupre/onnx-light": 0})
        self.record(day=14)
        data = self.record(day=13)
        self.assertEqual([row["date"] for row in data["history"]],
                         ["2026-11-12", "2026-11-13", "2026-11-14"])
        self.assertEqual(data["history"][0], same_day["history"][0])
        self.assertEqual(json.loads(self.path.read_text()), data)

    def test_utc_date_and_naive_time(self):
        data = self.record(now=dt.datetime(
            2026, 11, 12, 1, tzinfo=dt.timezone(dt.timedelta(hours=3))
        ))
        self.assertEqual(data["history"][0]["date"], "2026-11-11")
        self.assertEqual(data["updated_at"], "2026-11-11T22:00:00Z")
        data = self.record(now=dt.datetime(2026, 11, 12, 1))
        self.assertEqual(data["updated_at"], "2026-11-12T01:00:00Z")

    def test_failures_preserve_cache(self):
        self.record()
        previous = self.path.read_bytes()
        with self.assertRaises(HTTPError):
            self.record(day=13, repositories=["onnx/onnx", "z/missing"])
        self.assertEqual(self.path.read_bytes(), previous)
        for invalid in [None, -1, True, "12", 1.5]:
            with self.subTest(count=invalid):
                self.payloads["/repos/xadupre/onnx-light"] = {
                    "stargazers_count": invalid
                }
                with self.assertRaises(ValueError):
                    self.record(day=13)
                self.assertEqual(self.path.read_bytes(), previous)

    def test_invalid_selection(self):
        for repositories in [[], ["bad"], ["owner/name?token"], ["../owner/name"]]:
            with self.subTest(repositories=repositories):
                with self.assertRaises(ValueError):
                    self.record(repositories=repositories)
        self.assertFalse(self.path.exists())

    def test_cli_explicit_repositories(self):
        self.assertEqual(stars.main([
            "--cache-dir", self.temporary.name,
            "--repo", "onnx/onnx",
            "--repo", "xadupre/onnx-light",
        ]), 0)
        data = json.loads(self.path.read_text())
        self.assertEqual(data["history"][0]["stars"],
                         {"onnx/onnx": 10, "xadupre/onnx-light": 0})

    def test_workflow_and_dashboard_wiring(self):
        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/record_github_stars.yml").read_text()
        )
        triggers = workflow.get("on", workflow.get(True))
        self.assertIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertTrue(workflow["name"].startswith("DATA "))
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        steps = workflow["jobs"]["record"]["steps"]
        self.assertTrue(any(step.get("run") ==
                            "python -u scripts/record_github_stars.py"
                            for step in steps))
        self.assertTrue(any(step.get("run") ==
                            'bash scripts/commit_cache_data.sh cache_data "Update GitHub stars"'
                            for step in steps))
        page = (ROOT / "dashboard/github-stars.html").read_text()
        self.assertIn("../assets/github-stars.js", page)
        self.assertIn("../cache_data/github-stars.json", page)
        self.assertIn("dashboard/github-stars.html", (ROOT / "index.html").read_text())

    def test_browser_logic(self):
        subprocess.run(
            ["node", "--test", str(ROOT / "scripts/tests/github_stars.test.cjs")],
            check=True, capture_output=True, text=True,
        )


if __name__ == "__main__":
    unittest.main()
