"""Tests that Chart.js is loaded through the resilient chart-loader helper.

The package-size dashboards create charts as soon as their CSV data loads. If
Chart.js is fetched from a single CDN and that CDN is unreachable, the pages
fail with "Chart is not defined". These tests ensure the pages rely on
assets/chart-loader.js (which tries several CDNs) and defer chart creation until
the loader resolves.
"""

from __future__ import annotations

import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# (page path relative to repo root, depth from repo root to the page's dir)
CHART_LOADER_PAGES = [
    ("dashboard/pypi-downloads.html", 1),
    ("dashboard/onnx/stats.html", 2),
    ("dashboard/onnx/build-durations.html", 2),
    ("dashboard/onnx/lost-compute.html", 2),
    ("dashboard/onnx-light/package-size.html", 2),
    ("dashboard/onnx-light/pr-stats.html", 2),
    ("dashboard/onnx-light/build-durations.html", 2),
    ("dashboard/onnx-light/lost-compute.html", 2),
    ("dashboard/yet-another-onnx-builder/package-size.html", 2),
    ("dashboard/yet-another-onnx-builder/pr-stats.html", 2),
    ("dashboard/yet-another-onnx-builder/sklearn-coverage.html", 2),
    ("dashboard/yet-another-onnx-builder/build-durations.html", 2),
    ("dashboard/yet-another-onnxruntime-extensions/pr-stats.html", 2),
]

LOADER_SCRIPT_RE = re.compile(r'<script\s+src="((?:\.\./)*)assets/chart-loader\.js"')
# A direct <script src> that pulls chart.js straight from a CDN.
DIRECT_CHART_CDN_RE = re.compile(r'<script\s+src="https?://[^"]*/chart\.js@')


class TestChartLoader(unittest.TestCase):
    def test_helper_script_exists(self):
        path = os.path.join(REPO_ROOT, "assets", "chart-loader.js")
        self.assertTrue(os.path.isfile(path), f"missing helper: {path}")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        # The helper must expose loadChartJs and try more than one CDN.
        self.assertIn("window.loadChartJs", text)
        self.assertIn("cdn.jsdelivr.net", text)
        self.assertIn("unpkg.com", text)

    def test_pages_use_loader(self):
        for rel, depth in CHART_LOADER_PAGES:
            with self.subTest(page=rel):
                full = os.path.join(REPO_ROOT, rel)
                self.assertTrue(os.path.isfile(full), f"missing page: {rel}")
                with open(full, encoding="utf-8") as fh:
                    text = fh.read()

                m = LOADER_SCRIPT_RE.search(text)
                self.assertIsNotNone(m, f"{rel}: missing chart-loader.js include")
                self.assertEqual(
                    m.group(1),
                    "../" * depth,
                    f"{rel}: wrong chart-loader.js path prefix",
                )

                self.assertIsNone(
                    DIRECT_CHART_CDN_RE.search(text),
                    f"{rel}: chart.js must be loaded via chart-loader.js, "
                    "not a direct CDN <script> tag",
                )

                # Chart rendering must be gated on the loader promise.
                self.assertIn("loadChartJs()", text)


if __name__ == "__main__":
    unittest.main()
