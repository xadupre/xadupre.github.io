"""Tests for the onnx-light inplace-reuse coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light", "inplace-reuse-coverage.html")


class TestInplaceReuseCoverageDashboard(unittest.TestCase):
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

    def test_detail_renders_graph_inputs_section(self):
        """Dashboard renders a separate section for graph-level input metadata."""
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("const graphInputs = Array.isArray(row.inputs) ? row.inputs : [];", text)
        self.assertIn("graph inputs", text)
        self.assertIn("for (const inp of graphInputs) {", text)


if __name__ == "__main__":
    unittest.main()
