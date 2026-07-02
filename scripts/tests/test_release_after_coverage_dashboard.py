"""Tests for the onnx-light release-after coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(
    REPO_ROOT, "dashboard", "onnx-light", "release-after-coverage.html"
)


class TestReleaseAfterCoverageDashboard(unittest.TestCase):
    def test_unfolded_rows_prefer_svg_graph_rendering(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("function hasValidSvgGraph(row) {", text)
        self.assertIn("if (hasValidSvgGraph(row)) {", text)
        self.assertIn(
            'const doc = new DOMParser().parseFromString(row.graph.svg, "image/svg+xml");',
            text,
        )
        self.assertIn("targetDiv.className = \"onnx-svg\";", text)
        self.assertIn("if (row.mermaid) {", text)


if __name__ == "__main__":
    unittest.main()
