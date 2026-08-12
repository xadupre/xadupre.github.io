"""Tests for ``scripts.record_onnx_light_cpu_dtype_benchmark`` and its wiring."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_cpu_dtype_benchmark as rcd  # noqa: E402


def _shape(label="single-tile", m=64, n=64, k=64, has_builtin=True):
    return {"label": label, "M": m, "N": n, "K": k, "has_builtin": has_builtin}


class TestRowFromTimes(unittest.TestCase):
    def test_all_series_and_speedup(self):
        row = rcd._row_from_times(
            _shape(),
            {
                "onnx_light_cpu_float32": 0.025,
                "onnx_light_cpu_float16": 0.03,
                "onnx_light_cpu_bfloat16": 0.04,
                "onnxruntime_float32": 0.05,
                "onnxruntime_float16": 0.06,
                "onnx_light_float32": 0.2,
            },
        )
        self.assertEqual(row["shape"], "single-tile")
        self.assertEqual(row["M"], 64)
        self.assertEqual(row["N"], 64)
        self.assertEqual(row["K"], 64)
        self.assertEqual(row["onnx_light_cpu_float32_ms"], 0.025)
        self.assertEqual(row["onnxruntime_float16_ms"], 0.06)
        self.assertEqual(row["onnx_light_float32_ms"], 0.2)
        # speedup_cpu = onnxruntime_float32 / onnx_light_cpu_float32 = 0.05 / 0.025
        self.assertEqual(row["speedup_cpu"], 2.0)

    def test_missing_series_is_omitted(self):
        row = rcd._row_from_times(
            _shape(),
            {"onnx_light_cpu_float32": 1.0, "onnxruntime_float32": 2.0},
        )
        self.assertNotIn("onnx_light_cpu_bfloat16_ms", row)
        self.assertNotIn("onnx_light_float32_ms", row)
        self.assertEqual(row["speedup_cpu"], 2.0)

    def test_no_speedup_without_both_sides(self):
        row = rcd._row_from_times(_shape(), {"onnxruntime_float32": 2.0})
        self.assertNotIn("speedup_cpu", row)

    def test_no_speedup_when_cpu_zero(self):
        row = rcd._row_from_times(
            _shape(), {"onnx_light_cpu_float32": 0.0, "onnxruntime_float32": 2.0}
        )
        self.assertNotIn("speedup_cpu", row)

    def test_dims_are_int(self):
        row = rcd._row_from_times(_shape(m=4, n=4096, k=128), {})
        for key in ("M", "N", "K"):
            self.assertIsInstance(row[key], int)


class TestSummarize(unittest.TestCase):
    def test_summary_stats(self):
        rows = [
            rcd._row_from_times(
                _shape(label="a"),
                {"onnx_light_cpu_float32": 1.0, "onnxruntime_float32": 2.0},
            ),
            rcd._row_from_times(
                _shape(label="b"),
                {"onnx_light_cpu_float32": 4.0, "onnxruntime_float32": 2.0},
            ),
        ]
        summary = rcd._summarize(rows)
        self.assertEqual(summary["shapes"], 2)
        self.assertEqual(summary["cpu_succeeded"], 2)
        self.assertEqual(summary["max_speedup_cpu"], 2.0)
        self.assertEqual(summary["min_speedup_cpu"], 0.5)
        self.assertEqual(summary["avg_speedup_cpu"], round((2.0 + 0.5) / 2, 4))

    def test_summary_without_speedups(self):
        rows = [rcd._row_from_times(_shape(), {"onnx_light_cpu_float16": 1.0})]
        summary = rcd._summarize(rows)
        self.assertEqual(summary["shapes"], 1)
        self.assertEqual(summary["cpu_succeeded"], 0)
        self.assertNotIn("avg_speedup_cpu", summary)


class TestDefaultShapes(unittest.TestCase):
    def test_shapes_shape(self):
        shapes = rcd.default_shapes()
        labels = [s["label"] for s in shapes]
        self.assertEqual(
            labels, ["single-tile", "K-chunked", "multi-panel", "skinny-M/wide-N"]
        )
        for shape in shapes:
            for key in ("label", "M", "N", "K", "has_builtin"):
                self.assertIn(key, shape, f"{shape['label']} missing {key}")

    def test_builtin_only_on_lighter_shapes(self):
        shapes = rcd.default_shapes()
        by_label = {s["label"]: s for s in shapes}
        self.assertTrue(by_label["single-tile"]["has_builtin"])
        self.assertTrue(by_label["K-chunked"]["has_builtin"])
        self.assertFalse(by_label["multi-panel"]["has_builtin"])
        self.assertFalse(by_label["skinny-M/wide-N"]["has_builtin"])

    def test_max_size_filters_heavy_shapes(self):
        shapes = rcd.default_shapes(max_size=64)
        labels = [s["label"] for s in shapes]
        self.assertEqual(labels, ["single-tile"])

    def test_max_size_never_empty(self):
        shapes = rcd.default_shapes(max_size=1)
        self.assertTrue(shapes)


class TestRepeatFor(unittest.TestCase):
    def test_repeat_bounds(self):
        # Tiny shape hits the upper cap; huge shape hits the floor of 7.
        self.assertEqual(rcd.repeat_for(1, 1, 1), 50)
        self.assertEqual(rcd.repeat_for(512, 512, 512), 7)


class TestMeasure(unittest.TestCase):
    def test_median_of_calls(self):
        counter = {"n": 0}

        def func():
            counter["n"] += 1

        value = rcd.measure(func, repeat=5, warmup=2)
        self.assertGreaterEqual(value, 0.0)
        self.assertEqual(counter["n"], 7)


class TestBuildPayload(unittest.TestCase):
    def _fake_run(self, shapes, n_warmup, n_measure):
        rows = [
            rcd._row_from_times(
                _shape(),
                {"onnx_light_cpu_float32": 1.0, "onnxruntime_float32": 2.0},
            )
        ]
        meta = {"simd_level": 3, "simd_name": "AVX2"}
        return rows, meta

    def test_payload_structure(self):
        payload = rcd.build_payload(
            run=self._fake_run,
            versions=lambda: {"numpy": "1.0"},
            now=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(payload["date"], "2024-01-02T03:04:05Z")
        self.assertEqual(payload["n_warmup"], rcd.N_WARMUP)
        self.assertEqual(payload["n_measure"], rcd.N_MEASURE)
        self.assertEqual(payload["versions"], {"numpy": "1.0"})
        self.assertEqual(payload["op"], "Gemm")
        self.assertEqual(payload["source"], "plot_gemm_dtype_benchmark.py")
        self.assertEqual(payload["simd_level"], 3)
        self.assertEqual(payload["simd_name"], "AVX2")
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(
            [s["key"] for s in payload["series"]],
            [key for key, _ in rcd.SERIES],
        )
        self.assertEqual(payload["summary"]["shapes"], 1)

    def test_shapes_are_passed_through(self):
        captured = {}

        def fake_run(shapes, n_warmup, n_measure):
            captured["shapes"] = shapes
            captured["n_warmup"] = n_warmup
            captured["n_measure"] = n_measure
            return [], {}

        sentinel = [_shape(label="custom")]
        rcd.build_payload(
            shapes=sentinel,
            n_warmup=1,
            n_measure=2,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertIs(captured["shapes"], sentinel)
        self.assertEqual(captured["n_warmup"], 1)
        self.assertEqual(captured["n_measure"], 2)


class TestWritePayload(unittest.TestCase):
    def test_roundtrip(self):
        payload = {"date": "2024-01-01T00:00:00Z", "rows": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnx-light-cpu", "dtype_benchmark.json")
            rcd.write_payload(path, payload)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded, payload)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = rcd.parse_args([])
        self.assertEqual(args.cache_dir, "cache_data")
        self.assertEqual(args.n_warmup, rcd.N_WARMUP)
        self.assertEqual(args.n_measure, rcd.N_MEASURE)
        self.assertIsNone(args.max_size)

    def test_overrides(self):
        args = rcd.parse_args(
            ["--n-warmup", "1", "--n-measure", "2", "--max-size", "128"]
        )
        self.assertEqual(args.n_warmup, 1)
        self.assertEqual(args.n_measure, 2)
        self.assertEqual(args.max_size, 128)


class TestMain(unittest.TestCase):
    def test_main_writes_file(self):
        rows = [
            rcd._row_from_times(
                _shape(),
                {"onnx_light_cpu_float32": 1.0, "onnxruntime_float32": 2.0},
            )
        ]

        def fake_build(**kwargs):
            return {
                "date": "2024-01-01T00:00:00Z",
                "n_warmup": 3,
                "n_measure": 10,
                "versions": {},
                "rows": rows,
                "summary": {},
            }

        original = rcd.build_payload
        rcd.build_payload = fake_build  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc = rcd.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(tmp, "onnx-light-cpu", "dtype_benchmark.json")
                self.assertTrue(os.path.exists(path))
        finally:
            rcd.build_payload = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dashboard / index / workflow wiring
# ---------------------------------------------------------------------------

PAGE = os.path.join(REPO_ROOT, "dashboard", "onnx-light-cpu", "dtype-benchmark.html")
INDEX = os.path.join(REPO_ROOT, "index.html")
WORKFLOW = os.path.join(
    REPO_ROOT, ".github", "workflows", "record_onnx_light_cpu_dtype_benchmark.yml"
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class TestDashboard(unittest.TestCase):
    def test_page_exists(self):
        self.assertTrue(os.path.isfile(PAGE), f"missing page: {PAGE}")

    def test_page_loads_the_expected_json(self):
        text = _read(PAGE)
        self.assertIn(
            'const JSON_URL = "../../cache_data/onnx-light-cpu/dtype_benchmark.json";',
            text,
        )

    def test_page_renders_rows_and_speedup(self):
        text = _read(PAGE)
        self.assertIn("function renderBenchmark(payload)", text)
        self.assertIn("payload.rows", text)
        self.assertIn("speedup_cpu", text)
        for label in (
            "onnx-light-cpu float32",
            "onnx-light-cpu bfloat16",
            "onnxruntime float16",
            "onnx-light (built-in) float32",
        ):
            self.assertIn(label, text)

    def test_page_has_footer_pointing_at_cache(self):
        text = _read(PAGE)
        self.assertIn(
            'data-source="../../cache_data/onnx-light-cpu/dtype_benchmark.json"',
            text,
        )
        self.assertIn('<script src="../../assets/last-updated.js">', text)


class TestIndexWiring(unittest.TestCase):
    def test_index_links_dashboard(self):
        text = _read(INDEX)
        self.assertIn('href="dashboard/onnx-light-cpu/dtype-benchmark.html"', text)

    def test_index_has_workflow_badge(self):
        text = _read(INDEX)
        self.assertIn("record_onnx_light_cpu_dtype_benchmark.yml", text)


class TestWorkflow(unittest.TestCase):
    def test_workflow_exists(self):
        self.assertTrue(os.path.isfile(WORKFLOW), f"missing workflow: {WORKFLOW}")

    def test_workflow_builds_from_source_and_runs_script(self):
        text = _read(WORKFLOW)
        self.assertIn("name: DATA onnx-light-cpu dtype benchmark", text)
        self.assertIn("repository: xadupre/onnx-light", text)
        self.assertIn("repository: xadupre/onnx-light-cpu", text)
        self.assertIn("ONNX_LIGHT_CPU_WITH_ONNX_LIGHT=ON", text)
        self.assertIn(
            "python -u scripts/record_onnx_light_cpu_dtype_benchmark.py", text
        )
        self.assertIn("cache_data/onnx-light-cpu/dtype_benchmark.json", text)


class TestScriptCLI(unittest.TestCase):
    def test_script_is_executable_module(self):
        import importlib

        module = importlib.import_module("record_onnx_light_cpu_dtype_benchmark")
        for name in ("build_payload", "write_payload", "main", "parse_args"):
            self.assertTrue(hasattr(module, name), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
