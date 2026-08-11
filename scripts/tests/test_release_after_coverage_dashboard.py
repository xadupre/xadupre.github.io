"""Tests for the onnx-light release-after coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light", "release-after-coverage.html")


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
        self.assertIn('targetDiv.className = "onnx-svg";', text)
        self.assertIn("if (row.mermaid) {", text)

    def test_detail_rows_render_input_output_info(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("function formatNodeIo(node) {", text)
        self.assertIn("return { inputs, outputs };", text)
        self.assertIn("function formatNodeOp(node) {", text)
        self.assertIn(
            'const inputSig = opType + "(" + io.inputs.join(", ") + ")";', text
        )
        self.assertIn(
            'return outputText ? (inputSig + " -> " + outputText) : inputSig;', text
        )
        self.assertIn("const opText = formatNodeOp(node);", text)

    def test_values_section_rendered_in_detail(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("row.values", text)
        self.assertIn(
            'filter(val => val.kind !== "input" && val.kind !== "initializer")', text
        )
        self.assertIn("Outputs", text)

    def test_summary_is_bounded(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("const MAX_SUMMARY_MISMATCHES = 3;", text)
        self.assertIn("mismatches.length >= MAX_SUMMARY_MISMATCHES", text)
        self.assertIn("expand for details", text)


if __name__ == "__main__":
    unittest.main()
