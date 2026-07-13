"""Tests for the onnx-light backend test coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(
    REPO_ROOT, "dashboard", "onnx-light", "backend-test-coverage.html"
)


class TestOnnxLightBackendTestCoverageDashboard(unittest.TestCase):
    def test_big_examples_shortcuts_are_rendered(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("function renderBigExamples(rows) {", text)
        self.assertIn("Big examples (largest models)", text)
        self.assertIn("renderBigExamples(state.rows);", text)

    def test_big_examples_uses_svg_length_as_size_metric(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("row.graph && row.graph.svg ? row.graph.svg.length : 0", text)


if __name__ == "__main__":
    unittest.main()
