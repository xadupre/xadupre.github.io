"""Tests that every HTML page exposes a "Data last updated" footer."""

from __future__ import annotations

import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# (page path relative to repo root, depth from repo root to the page's dir)
PAGES = [
    ("index.html", 0),
    ("dashboard/pypi-downloads.html", 1),
    ("dashboard/mbext/class-coverage.html", 2),
    ("dashboard/onnx/stats.html", 2),
    ("dashboard/onnx/backend-test-coverage.html", 2),
    ("dashboard/onnx/build-durations.html", 2),
    ("dashboard/onnx-light/backend-test-coverage.html", 2),
    ("dashboard/onnx-light/build-durations.html", 2),
    ("dashboard/onnx-light/package-size.html", 2),
    ("dashboard/onnx-light/pr-stats.html", 2),
    ("dashboard/onnx-light/schema-comparison.html", 2),
    ("dashboard/onnx-light/shape-inference-coverage.html", 2),
    ("dashboard/yet-another-onnx-builder/build-durations.html", 2),
    ("dashboard/yet-another-onnx-builder/model-validate.html", 2),
    ("dashboard/yet-another-onnx-builder/package-size.html", 2),
    ("dashboard/yet-another-onnx-builder/pr-stats.html", 2),
    ("dashboard/yet-another-onnx-builder/sklearn-coverage.html", 2),
    ("dashboard/yet-another-onnx-builder/torch-coverage.html", 2),
    ("dashboard/yet-another-onnxruntime-extensions/pr-stats.html", 2),
]

FOOTER_RE = re.compile(
    r'<footer\b[^>]*\bclass="data-updated"[^>]*\bdata-source="([^"]+)"'
)
SCRIPT_RE = re.compile(r'<script\s+src="((?:\.\./)*)assets/last-updated\.js"')


class TestLastUpdatedFooter(unittest.TestCase):
    def test_helper_script_exists(self):
        path = os.path.join(REPO_ROOT, "assets", "last-updated.js")
        self.assertTrue(os.path.isfile(path), f"missing helper: {path}")

    def test_pages_have_footer_and_script(self):
        for rel, depth in PAGES:
            with self.subTest(page=rel):
                full = os.path.join(REPO_ROOT, rel)
                self.assertTrue(os.path.isfile(full), f"missing page: {rel}")
                with open(full, encoding="utf-8") as fh:
                    text = fh.read()

                m = FOOTER_RE.search(text)
                self.assertIsNotNone(
                    m, f"{rel}: missing data-updated footer"
                )
                sources = [s.strip() for s in m.group(1).split(",") if s.strip()]
                self.assertTrue(sources, f"{rel}: empty data-source")

                page_dir = os.path.dirname(full)
                for src in sources:
                    # Skip non-existing path checks for build-durations
                    # which uses generated jobs/index.json that may not be
                    # in the repo if no run has been recorded yet.
                    resolved = os.path.normpath(os.path.join(page_dir, src))
                    parent = os.path.dirname(resolved)
                    self.assertTrue(
                        os.path.isdir(parent),
                        f"{rel}: data-source parent missing: {src}",
                    )

                s = SCRIPT_RE.search(text)
                self.assertIsNotNone(
                    s, f"{rel}: missing last-updated.js include"
                )
                expected_prefix = "../" * depth
                self.assertEqual(
                    s.group(1),
                    expected_prefix,
                    f"{rel}: wrong script path prefix",
                )


if __name__ == "__main__":
    unittest.main()
