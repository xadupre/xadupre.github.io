"""Tests for the plot_onnx_time history dashboard."""

import csv
import datetime
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PAGE = os.path.join(ROOT, "dashboard", "onnx-light", "onnx-time.html")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "record_onnx_time.yml")
DATA = os.path.join(ROOT, "cache_data", "onnx-light", "onnx_time.csv")


class TestOnnxTimeDashboard(unittest.TestCase):
    def test_page_loads_history_and_links_example(self):
        with open(PAGE, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("../../cache_data/onnx-light/onnx_time.csv", text)
        self.assertIn("plot_onnx_time.html", text)
        self.assertIn('id="scenario"', text)
        self.assertIn('id="metric"', text)
        self.assertIn('<option value="All">All</option>', text)
        for chart_id in ("loadChart", "saveChart", "cppChart"):
            self.assertIn(f'id="{chart_id}"', text)
        self.assertIn('kind === "cpp"', text)
        self.assertIn('/-cpp(?:-|$)/', text)
        self.assertIn('row.name.startsWith(kind + "/")', text)
        self.assertIn("x: Date.parse(row.date)", text)
        self.assertIn("loadChartJs()", text)

        with open(DATA, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        for row in rows:
            datetime.datetime.fromisoformat(row["date"].replace("Z", "+00:00"))

    def test_dedicated_workflow_records_history(self):
        with open(WORKFLOW, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("name: DATA onnx-light load/save timings", text)
        self.assertIn("schedule:", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("docs/examples/proto/plot_onnx_time.py", text)
        self.assertIn("python scripts/record_onnx_time.py", text)
        self.assertIn("--output cache_data/onnx-light/onnx_time.csv", text)
        self.assertIn('git -C onnx-light rev-parse HEAD', text)
        self.assertIn('git add cache_data/onnx-light/onnx_time.csv', text)


if __name__ == "__main__":
    unittest.main()
