"""Tests that every HTML page exposes a "Data last updated" footer."""

from __future__ import annotations

import os
import re
import unittest
from html.parser import HTMLParser

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
    ("dashboard/onnx-light/inplace-reuse-coverage.html", 2),
    ("dashboard/onnx-light/release-after-coverage.html", 2),
    ("dashboard/onnx-light/constant-info-coverage.html", 2),
    ("dashboard/onnx-light/cgen-comparison.html", 2),
    ("dashboard/onnx-light/lost-compute.html", 2),
    ("dashboard/onnx-light-cpu/benchmark.html", 2),
    ("dashboard/onnx-light-cpu/examples-benchmark.html", 2),
    ("dashboard/onnx-light-cpu/package-size.html", 2),
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


class _DocLinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.doc_links = []
        self._current_link = None
        self._in_doc_label = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("class") == "doc-link":
            self._current_link = {"data_word": attrs.get("data-word"), "label": ""}
        elif (
            tag == "span"
            and self._current_link is not None
            and attrs.get("class") == "doc-label"
        ):
            self._in_doc_label = True

    def handle_data(self, data):
        if self._in_doc_label and self._current_link is not None:
            self._current_link["label"] += data

    def handle_endtag(self, tag):
        if tag == "span" and self._in_doc_label:
            self._in_doc_label = False
        elif tag == "a" and self._current_link is not None:
            self._current_link["label"] = self._current_link["label"].strip()
            self.doc_links.append(self._current_link)
            self._current_link = None


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
                self.assertIsNotNone(m, f"{rel}: missing data-updated footer")
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
                self.assertIsNotNone(s, f"{rel}: missing last-updated.js include")
                expected_prefix = "../" * depth
                self.assertEqual(
                    s.group(1),
                    expected_prefix,
                    f"{rel}: wrong script path prefix",
                )

    def test_homepage_doc_links_have_icon_words(self):
        full = os.path.join(REPO_ROOT, "index.html")
        with open(full, encoding="utf-8") as fh:
            text = fh.read()

        parser = _DocLinkParser()
        parser.feed(text)

        self.assertTrue(parser.doc_links)
        for link in parser.doc_links:
            self.assertTrue(link["data_word"])
            self.assertTrue(link["label"])
            self.assertEqual(link["data_word"], link["label"].upper())

    def test_homepage_icon_word_style(self):
        full = os.path.join(REPO_ROOT, "index.html")
        with open(full, encoding="utf-8") as fh:
            text = fh.read()

        match = re.search(r"\.doc-label\s*\{([^}]*)\}", text, re.DOTALL)
        self.assertIsNotNone(match)
        rule = match.group(1)

        self.assertIn("text-transform: uppercase;", rule)
        self.assertIn("color: #8b949e;", rule)
        self.assertIn("font-weight: 200;", rule)


if __name__ == "__main__":
    unittest.main()
