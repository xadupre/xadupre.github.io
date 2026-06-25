"""Tests for the onnx-light shape inference coverage dashboard page."""

from __future__ import annotations

import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PAGE = os.path.join(
    REPO_ROOT, "dashboard", "onnx-light", "shape-inference-coverage.html"
)


class TestShapeInferenceCoverageDashboard(unittest.TestCase):
    def test_cc_shape_inference_names_are_formatted_for_readability(self):
        with open(PAGE, encoding="utf-8") as f:
            text = f.read()

        self.assertIn('const prefix = "test_cc_shape_inference_";', text)
        self.assertIn(
            'return "test_cc_shape_inference " + name.slice(prefix.length);',
            text,
        )
        self.assertIn(
            "tdName.appendChild(document.createTextNode(formatTestName(r.name)));",
            text,
        )


if __name__ == "__main__":
    unittest.main()
