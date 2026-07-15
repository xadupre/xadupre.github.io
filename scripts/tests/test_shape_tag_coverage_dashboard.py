"""Tests for the onnx-light shape-tag coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light", "shape-tag-coverage.html")


class TestShapeTagCoverageDashboard(unittest.TestCase):
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
        self.assertIn('const inputSig = opType + "(" + io.inputs.join(", ") + ")";', text)
        self.assertIn('return outputText ? (inputSig + " -> " + outputText) : inputSig;', text)
        self.assertIn("const opText = formatNodeOp(node);", text)

    def test_values_section_rendered_in_detail(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("row.values", text)
        self.assertIn("Inputs / outputs / initializers / results", text)

    def test_summary_row_reports_value_mismatches(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        # summarizeRow must iterate over row.values (not only row.nodes)
        self.assertIn("row.values", text)
        # val.success is checked to skip matched values
        self.assertIn("val.success", text)
        # val.kind and val.name are used in the mismatch description
        self.assertIn("val.kind", text)
        self.assertIn("val.name", text)

    def test_values_ratio_card_in_render_ratios(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("value tags matched", text)
        self.assertIn("totals.values", text)
        self.assertIn('cls: "values"', text)

    def test_values_column_in_table_and_rows(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn('data-key="values"', text)
        self.assertIn("matched_values", text)
        self.assertIn("total_values", text)

    def test_summarize_row_reports_missing_metadata(self):
        """summarizeRow must return a specific message when row.missing_metadata is true."""
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        # The summarizeRow function must check missing_metadata before falling through
        self.assertIn("row.missing_metadata", text)
        self.assertIn("No shape tag metadata found", text)

    def test_detail_view_shows_missing_metadata_indicator(self):
        """renderDetailFor must show a specific message when row.missing_metadata is true."""
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn("row.missing_metadata", text)
        self.assertIn("No shape tag metadata found", text)

if __name__ == "__main__":
    unittest.main()
