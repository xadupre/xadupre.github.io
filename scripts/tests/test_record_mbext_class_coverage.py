"""Tests for ``scripts.record_mbext_class_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_mbext_class_coverage as rmcc  # noqa: E402


def _file(path: str, download_url: str) -> dict[str, str]:
    return {"path": path, "download_url": download_url}


class TestRecordMbextClassCoverage(unittest.TestCase):
    def test_extract_class_names_top_level_and_nested(self):
        source = (
            "class A:\n"
            "    pass\n"
            "\n"
            "def f():\n"
            "    class Inner:\n"
            "        pass\n"
            "    return Inner\n"
            "\n"
            "class B(A):\n"
            "    pass\n"
        )
        self.assertEqual(
            sorted(rmcc.extract_class_names(source)),
            ["A", "B", "Inner"],
        )

    def test_extract_class_names_ignores_syntax_errors(self):
        # Invalid Python should not crash the script.
        self.assertEqual(rmcc.extract_class_names("def !!"), [])

    def test_build_payload_merges_projects(self):
        sources = {
            "https://example/a/m.py": "class Shared:\n    pass\nclass Only1:\n    pass\n",
            "https://example/a/extra.py": "class Shared:\n    pass\n",
            "https://example/b/n.py": "class Shared:\n    pass\nclass Only2:\n    pass\n",
        }
        listings = {
            ("o1", "r1", "p1", "main"): [
                _file("p1/m.py", "https://example/a/m.py"),
                _file("p1/extra.py", "https://example/a/extra.py"),
            ],
            ("o2", "r2", "p2", "main"): [
                _file("p2/n.py", "https://example/b/n.py"),
            ],
        }

        def fake_list(owner, repo, path, ref):
            return listings[(owner, repo, path, ref)]

        def fake_request(url):
            return sources[url].encode("utf-8")

        def fake_commit(owner, repo, path, ref):
            return f"sha-{owner}"

        projects = (
            {
                "key": "alpha",
                "owner": "o1",
                "repo": "r1",
                "path": "p1",
                "ref": "main",
            },
            {
                "key": "beta",
                "owner": "o2",
                "repo": "r2",
                "path": "p2",
                "ref": "main",
            },
        )

        originals = (rmcc.list_python_files, rmcc._request, rmcc.fetch_latest_commit)
        rmcc.list_python_files = fake_list
        rmcc._request = fake_request
        rmcc.fetch_latest_commit = fake_commit
        try:
            payload = rmcc.build_payload(projects=projects)
        finally:
            (
                rmcc.list_python_files,
                rmcc._request,
                rmcc.fetch_latest_commit,
            ) = originals

        self.assertEqual(payload["totals"], {"alpha": 2, "beta": 2})
        self.assertEqual(payload["projects"]["alpha"]["commit"], "sha-o1")
        names = [row["name"] for row in payload["classes"]]
        # Sorted alphabetically and de-duplicated across files.
        self.assertEqual(names, ["Only1", "Only2", "Shared"])
        by_name = {row["name"]: row for row in payload["classes"]}
        self.assertTrue(by_name["Only1"]["in_alpha"])
        self.assertFalse(by_name["Only1"]["in_beta"])
        self.assertFalse(by_name["Only2"]["in_alpha"])
        self.assertTrue(by_name["Only2"]["in_beta"])
        self.assertTrue(by_name["Shared"]["in_alpha"])
        self.assertTrue(by_name["Shared"]["in_beta"])
        # ``Shared`` is defined in two alpha files; files are sorted/unique.
        self.assertEqual(by_name["Shared"]["alpha_files"], ["p1/extra.py", "p1/m.py"])
        self.assertNotIn("alpha_files", by_name["Only2"])

    def test_write_payload_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "mbext", "class_coverage.json")
            payload = {"date": "2024-01-01T00:00:00Z", "classes": []}
            rmcc.write_payload(json_path, payload)
            with open(json_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), payload)

    def test_main_writes_cache_file(self):
        original_build = rmcc.build_payload

        def fake_build():
            return {
                "date": "2024-01-01T00:00:00Z",
                "projects": {},
                "totals": {"mbext": 0, "ort_genai": 0},
                "classes": [{"name": "X", "in_mbext": True, "in_ort_genai": False}],
            }

        rmcc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rmcc.main(["--cache-dir", tmp])
                self.assertEqual(code, 0)
                with open(
                    os.path.join(tmp, "mbext", "class_coverage.json"),
                    encoding="utf-8",
                ) as fh:
                    payload = json.load(fh)
                self.assertEqual(payload["classes"][0]["name"], "X")
        finally:
            rmcc.build_payload = original_build

    def test_main_returns_one_on_network_failure(self):
        import urllib.error

        original_build = rmcc.build_payload

        def fake_build():
            raise urllib.error.URLError("boom")

        rmcc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rmcc.main(["--cache-dir", tmp])
                self.assertEqual(code, 1)
                self.assertFalse(
                    os.path.exists(os.path.join(tmp, "mbext", "class_coverage.json"))
                )
        finally:
            rmcc.build_payload = original_build


if __name__ == "__main__":
    unittest.main()
