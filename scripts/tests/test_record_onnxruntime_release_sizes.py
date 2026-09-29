"""Tests for the GitHub release size collector."""

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import record_onnxruntime_release_sizes as sizes  # noqa: E402


def release(version="1.20.0", **kwargs):
    return {
        "id": 123,
        "tag_name": f"v{version}",
        "published_at": "2024-10-30T12:00:00Z",
        "draft": False,
        "prerelease": False,
        **kwargs,
    }


def asset(name="onnxruntime-linux-x64-1.20.0.tgz", size=123456, **kwargs):
    return {"name": name, "size": size, "state": "uploaded", **kwargs}


class TestReleaseSizes(unittest.TestCase):
    def test_binary_sizes_and_variant_names(self):
        names = [
            "onnxruntime-linux-x64-1.20.0.tgz",
            "onnxruntime-linux-x64-gpu-1.20.0.tgz",
            "onnxruntime-win-arm64-1.20.0.zip",
            "Microsoft.ML.OnnxRuntime.1.20.0.nupkg",
        ]
        rows = sizes.build_rows(release(), [asset(name) for name in names])
        self.assertEqual([row["name"] for row in rows], names)
        self.assertEqual(len({row["package"] for row in rows}), 4)
        for row in rows:
            self.assertEqual(row["size"], 123456)
            self.assertEqual(row["version"], "v1.20.0")
            self.assertEqual(row["date"], "2024-10-30T12:00:00Z")
            self.assertIn("{version}", row["package"])
        newer = sizes.build_rows(
            release("1.21.0"), [asset(names[0].replace("1.20.0", "1.21.0"))]
        )
        self.assertEqual(rows[0]["package"], newer[0]["package"])

    def test_skip_previews_drafts_and_non_binaries(self):
        for flag in ("draft", "prerelease"):
            self.assertEqual(sizes.build_rows(release(**{flag: True}), [asset()]), [])
        self.assertEqual(sizes.build_rows(release(), [
            asset("SHA256SUMS.txt"), asset("onnxruntime.zip.sha256"),
            asset(state="starter"),
        ]), [])

    def test_paginated_requests(self):
        requests = []

        def open_page(request, timeout):
            requests.append(request)
            self.assertEqual(timeout, 60)
            data = [{"id": n} for n in range(100)] if len(requests) == 1 else []
            return io.BytesIO(json.dumps(data).encode())

        original = sizes.urllib.request.urlopen
        sizes.urllib.request.urlopen = open_page
        try:
            rows = sizes.fetch_pages(sizes.RELEASES_URL, "test-token")
        finally:
            sizes.urllib.request.urlopen = original
        self.assertEqual(len(rows), 100)
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[1].full_url.endswith("per_page=100&page=2"))
        self.assertEqual(
            requests[0].get_header("Authorization"), "Bearer " + "test-token"
        )

    def test_anonymous_request_rejects_invalid_collection(self):
        def open_page(request, timeout):
            self.assertIsNone(request.get_header("Authorization"))
            return io.BytesIO(b'{"message": "not a collection"}')

        original = sizes.urllib.request.urlopen
        sizes.urllib.request.urlopen = open_page
        try:
            with self.assertRaisesRegex(ValueError, "collection"):
                sizes.fetch_pages(sizes.RELEASES_URL)
        finally:
            sizes.urllib.request.urlopen = original

    def test_refresh_is_sorted_idempotent_and_updates_sizes(self):
        calls = []
        binaries = [asset()]

        def fetch(url, token):
            calls.append(url)
            if url == sizes.RELEASES_URL:
                return [release(), release(prerelease=True), release()]
            return binaries

        original = sizes.fetch_pages
        sizes.fetch_pages = fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "onnxruntime" / "release_sizes.json"
                rows = sizes.record_sizes(tmp)
                first = path.read_bytes()
                sizes.record_sizes(tmp)
                self.assertEqual(first, path.read_bytes())
                self.assertEqual(len(rows), 1)
                self.assertEqual(calls.count(sizes.RELEASES_URL + "/123/assets"), 4)
                binaries[0]["size"] = 456
                binaries.append(asset("onnxruntime-win-x64-1.20.0.zip"))
                rows = sizes.record_sizes(tmp)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]["size"], 456)
                self.assertEqual(json.loads(path.read_text()), rows)
                self.assertEqual(rows, sorted(rows, key=lambda row: (
                    row["date"], row["version"], row["name"]
                )))
        finally:
            sizes.fetch_pages = original

    def test_failure_does_not_overwrite_history(self):
        for fail_assets in (False, True):
            def fetch(url, token):
                if not fail_assets:
                    return []
                if url == sizes.RELEASES_URL:
                    return [release()]
                raise RuntimeError("API failure")

            original = sizes.fetch_pages
            sizes.fetch_pages = fetch
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "onnxruntime" / "release_sizes.json"
                    path.parent.mkdir()
                    path.write_text("existing history", encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        sizes.record_sizes(tmp)
                    self.assertEqual(path.read_text(), "existing history")
            finally:
                sizes.fetch_pages = original

    def test_main_backfills_in_publication_order(self):
        def fetch(url, token):
            if url == sizes.RELEASES_URL:
                return [
                    release("1.21.0", id=124, published_at="2025-01-01T00:00:00Z"),
                    release(),
                ]
            version = "1.21.0" if "/124/" in url else "1.20.0"
            return [asset(f"onnxruntime-linux-x64-{version}.tgz")]

        original = sizes.fetch_pages
        sizes.fetch_pages = fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(sizes.main(["--cache-dir", tmp]), 0)
                path = Path(tmp) / "onnxruntime" / "release_sizes.json"
                rows = json.loads(path.read_text())
                self.assertEqual(
                    [row["version"] for row in rows], ["v1.20.0", "v1.21.0"]
                )
        finally:
            sizes.fetch_pages = original


class TestReleaseSizeIntegration(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    def test_workflow_collects_and_publishes_metadata(self):
        path = self.ROOT / ".github" / "workflows" / "record_size_onnxruntime.yml"
        workflow = yaml.safe_load(path.read_text())
        self.assertTrue(workflow["name"].startswith("DATA "))
        triggers = workflow.get("on", workflow.get(True))
        self.assertIn("schedule", triggers)
        self.assertIn("workflow_dispatch", triggers)
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        job = workflow["jobs"]["record"]
        self.assertIn("github.actor == 'github-actions[bot]'", job["if"])
        steps = job["steps"]
        checkout = next(s for s in steps if s["name"] == "Checkout cache data")
        self.assertEqual(checkout["with"]["repository"], "xadupre/cache_data")
        self.assertEqual(checkout["with"]["path"], "cache_data")
        self.assertEqual(
            checkout["with"]["ssh-key"], "${{ secrets.CACHE_DATA_SSH_KEY }}"
        )
        record = next(s for s in steps if "release binary sizes" in s["name"])
        self.assertIn("scripts/record_onnxruntime_release_sizes.py", record["run"])
        self.assertEqual(record["env"]["GITHUB_TOKEN"], "${{ secrets.GITHUB_TOKEN }}")
        self.assertIn("bash scripts/commit_cache_data.sh cache_data", steps[-1]["run"])

    def test_dashboard_and_home_link_to_release_sizes(self):
        page = (self.ROOT / "dashboard" / "onnxruntime" / "package-size.html").read_text()
        self.assertIn("../../cache_data/onnxruntime/release_sizes.json", page)
        self.assertIn("loadChartJs()", page)
        self.assertIn("row.package === select.value", page)
        self.assertIn("row.size / 1048576", page)
        self.assertIn("encodeURIComponent(row.version)", page)
        self.assertNotIn("innerHTML", page)
        home = (self.ROOT / "index.html").read_text()
        self.assertIn('href="dashboard/onnxruntime/package-size.html"', home)
        self.assertIn("record_size_onnxruntime.yml/badge.svg", home)


if __name__ == "__main__":
    unittest.main()
