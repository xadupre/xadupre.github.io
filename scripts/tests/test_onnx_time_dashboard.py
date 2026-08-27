"""Tests for the plot_onnx_time history dashboard."""

import csv
import datetime
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PAGE = os.path.join(ROOT, "dashboard", "onnx-light", "onnx-time.html")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "record_onnx_time.yml")
DOC_WORKFLOW = os.path.join(ROOT, ".github", "workflows", "build_onnx_light_docs.yml")
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
        for chart_id in (
            "load1FileChart",
            "load1FileSpeedupChart",
            "load2FileChart",
            "load2FileSpeedupChart",
            "save1FileChart",
            "save1FileSpeedupChart",
            "save2FileChart",
            "save2FileSpeedupChart",
        ):
            self.assertIn(f'id="{chart_id}"', text)
        for prefix in ("load/1file", "load/2file", "save/1file", "save/2file"):
            self.assertIn(f'prefix: "{prefix}"', text)
        self.assertIn('row.name.startsWith(prefix + "x")', text)
        self.assertIn("x: Date.parse(row.date)", text)
        self.assertIn("loadChartJs()", text)
        for baseline in (
            "load/1filex1/onnx",
            "load/2filex1/onnx",
            "save/1filex1/onnx",
            "save/2filex1/onnx",
        ):
            self.assertIn(f'baseline: "{baseline}"', text)
        self.assertIn("baselineByRun.get(row.run_id) / Number(row[metric])", text)
        self.assertIn("CHARTS.forEach(renderSpeedupChart)", text)
        self.assertEqual(text.count('type:"logarithmic"'), 2)
        self.assertEqual(text.count('"Machine: " + item.raw.machine'), 2)

        with open(DATA, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(rows)
        names = {row["name"] for row in rows}
        self.assertIn("load/1filex1/onnx", names)
        self.assertIn("save/1filex1/onnx", names)
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
        self.assertIn("--machine", text)
        self.assertIn("git -C onnx-light rev-parse HEAD", text)
        self.assertIn("git add cache_data/onnx-light/onnx_time.csv", text)

    def test_documentation_workflow_records_machine(self):
        with open(DOC_WORKFLOW, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("python site/scripts/record_onnx_time.py", text)
        self.assertIn('machine="${{ runner.os }} ${{ runner.arch }} / $(lscpu', text)
        self.assertIn('--machine "${machine}"', text)


if __name__ == "__main__":
    unittest.main()
