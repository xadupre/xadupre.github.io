"""Tests for the plot_onnx_time history dashboard."""

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PAGE = os.path.join(ROOT, "dashboard", "onnx-light", "onnx-time.html")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "record_onnx_time.yml")
DOC_WORKFLOW = os.path.join(ROOT, ".github", "workflows", "build_onnx_light_docs.yml")
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
        self.assertEqual(text.count('time:{unit:"day"}'), 2)
        self.assertEqual(text.count('"Machine: " + item.raw.machine'), 2)

    def test_timing_and_speedup_charts_zoom_in_both_axes(self):
        with open(PAGE, encoding="utf-8") as stream:
            text = stream.read()
        self.assertEqual(
            text.count(
                'zoom: { pan:{enabled:true,mode:"xy"}, '
                'zoom:{wheel:{enabled:true},pinch:{enabled:true},mode:"xy"} }'
            ),
            2,
        )
        self.assertIn(
            'addEventListener("dblclick", () => charts[key].resetZoom())', text
        )
        self.assertIn(
            'addEventListener("dblclick", () => charts[key + "Speedup"].resetZoom())',
            text,
        )

    def test_timing_and_speedup_charts_use_library_line_styles(self):
        with open(PAGE, encoding="utf-8") as stream:
            text = stream.read()
        for function_name in ("renderChart", "renderSpeedupChart"):
            with self.subTest(function=function_name):
                body = text.split(f"function {function_name}(", 1)[1].split(
                    "\nfunction ", 1
                )[0]
                self.assertIn(
                    r"borderDash: /\/(?:onnx|ort|ir-py)(?:-|$)/.test(name) ? [2, 3] : []",
                    body,
                )
                pattern = body.split("borderDash: /", 1)[1].split("/.test(name)", 1)[0]
                for operation in ("load", "save"):
                    for files in (1, 2):
                        for threads in (1, 4):
                            for library, dotted in (
                                ("onnx", True),
                                ("onnx-cpp", True),
                                ("ort", True),
                                ("ir-py", True),
                                ("onnxlight", False),
                                ("onnxlight-cpp", False),
                                ("onnxlight-cpp-nocopy", False),
                                ("onnxlight-nocopy", False),
                                ("onnxlight-ifstream", False),
                                ("onnxlight-mmap", False),
                                ("reference", False),
                            ):
                                name = f"{operation}/{files}filex{threads}/{library}"
                                with self.subTest(name=name):
                                    self.assertEqual(
                                        re.search(pattern, name) is not None, dotted
                                    )

    def test_dedicated_workflow_records_history(self):
        with open(WORKFLOW, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("name: DATA onnx-light load/save timings", text)
        self.assertIn("schedule:", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("docs/examples/proto/plot_onnx_time.py", text)
        self.assertIn("examples/load_onnx_light_time/build.sh", text)
        self.assertIn("examples/save_onnx_light_time/build.sh", text)
        self.assertIn("examples/load_onnx_time/build.sh", text)
        self.assertIn('CICPP: "1"', text)
        self.assertIn("python scripts/record_onnx_time.py", text)
        self.assertIn("--output cache_data/onnx-light/onnx_time.csv", text)
        self.assertIn("--machine", text)
        self.assertIn("git -C onnx-light rev-parse HEAD", text)
        self.assertIn(
            'bash scripts/commit_cache_data.sh cache_data '
            '"Update onnx-light timing cache"',
            text,
        )

    def test_documentation_workflow_records_machine(self):
        with open(DOC_WORKFLOW, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("python site/scripts/record_onnx_time.py", text)
        self.assertIn('machine="${{ runner.os }} ${{ runner.arch }} / $(lscpu', text)
        self.assertIn('--machine "${machine}"', text)


if __name__ == "__main__":
    unittest.main()
