"""Tests for the plot_onnx_time history dashboard."""

import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PAGE = os.path.join(ROOT, "dashboard", "onnx-light", "onnx-time.html")


class TestOnnxTimeDashboard(unittest.TestCase):
    def test_page_loads_history_and_links_example(self):
        with open(PAGE, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("../../cache_data/onnx-light/onnx_time.csv", text)
        self.assertIn("plot_onnx_time.html", text)
        self.assertIn('id="scenario"', text)
        self.assertIn('id="metric"', text)
        self.assertIn("loadChartJs()", text)


if __name__ == "__main__":
    unittest.main()
