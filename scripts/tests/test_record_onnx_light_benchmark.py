"""Tests for ``scripts.record_onnx_light_benchmark``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_benchmark as rlb  # noqa: E402

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # noqa: BLE001
    _HAS_NUMPY = False


class TestStringifyError(unittest.TestCase):
    def test_none(self):
        self.assertEqual(rlb._stringify_error(None), "")

    def test_short(self):
        self.assertEqual(rlb._stringify_error("boom"), "boom")

    def test_multiline_keeps_first_line(self):
        self.assertEqual(rlb._stringify_error("boom\nrest"), "boom")

    def test_long_single_line_is_truncated_in_middle(self):
        long = "x" * 500
        out = rlb._stringify_error(long)
        self.assertEqual(len(out), 300)
        self.assertIn(" ... ", out)
        self.assertTrue(out.startswith("x"))
        self.assertTrue(out.endswith("x"))


class TestRowFromResults(unittest.TestCase):
    def _make_results(self, ort_ok=True, light_ok=True, ort_avg=1.0, light_avg=0.5):
        results = {}
        if ort_ok:
            results["onnxruntime"] = {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": ort_avg,
                "min_ms": ort_avg * 0.9,
                "max_ms": ort_avg * 1.1,
                "n_warmup": 3,
                "n_measure": 10,
            }
        else:
            results["onnxruntime"] = {
                "success": False,
                "error": "ort load error",
                "error_step": "load",
            }
        if light_ok:
            results["onnx_light"] = {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": light_avg,
                "min_ms": light_avg * 0.9,
                "max_ms": light_avg * 1.1,
                "n_warmup": 3,
                "n_measure": 10,
            }
        else:
            results["onnx_light"] = {
                "success": False,
                "error": "light load error",
                "error_step": "load",
            }
        return results

    def test_speedup_computed_when_both_succeed(self):
        results = self._make_results(ort_ok=True, light_ok=True, ort_avg=2.0, light_avg=1.0)
        row = rlb._row_from_results("test_relu", results)
        self.assertEqual(row["name"], "test_relu")
        self.assertTrue(row["onnxruntime_success"])
        self.assertTrue(row["onnx_light_success"])
        self.assertAlmostEqual(row["speedup"], 2.0)
        self.assertAlmostEqual(row["onnxruntime_avg_ms"], 2.0)
        self.assertAlmostEqual(row["onnx_light_avg_ms"], 1.0)

    def test_no_speedup_when_ort_fails(self):
        results = self._make_results(ort_ok=False, light_ok=True)
        row = rlb._row_from_results("test_relu", results)
        self.assertFalse(row["onnxruntime_success"])
        self.assertTrue(row["onnx_light_success"])
        self.assertNotIn("speedup", row)
        self.assertIn("onnxruntime_error", row)

    def test_no_speedup_when_light_fails(self):
        results = self._make_results(ort_ok=True, light_ok=False)
        row = rlb._row_from_results("test_relu", results)
        self.assertTrue(row["onnxruntime_success"])
        self.assertFalse(row["onnx_light_success"])
        self.assertNotIn("speedup", row)
        self.assertIn("onnx_light_error", row)

    def test_no_speedup_when_both_fail(self):
        results = self._make_results(ort_ok=False, light_ok=False)
        row = rlb._row_from_results("test_relu", results)
        self.assertNotIn("speedup", row)

    def test_speedup_gt_one_when_onnx_light_is_faster(self):
        # ort_avg > light_avg → speedup > 1 → onnx-light is faster
        results = self._make_results(ort_ok=True, light_ok=True, ort_avg=4.0, light_avg=1.0)
        row = rlb._row_from_results("test_abs", results)
        self.assertGreater(row["speedup"], 1.0)

    def test_speedup_lt_one_when_onnx_light_is_slower(self):
        # ort_avg < light_avg → speedup < 1 → onnx-light is slower
        results = self._make_results(ort_ok=True, light_ok=True, ort_avg=1.0, light_avg=4.0)
        row = rlb._row_from_results("test_abs", results)
        self.assertLess(row["speedup"], 1.0)

    def test_tag_included_when_provided(self):
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, tag="training")
        self.assertEqual(row["tag"], "training")

    def test_tag_absent_when_empty(self):
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, tag="")
        self.assertNotIn("tag", row)

    def test_graph_included_when_provided(self):
        graph = {"svg": "<svg></svg>"}
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, graph=graph)
        self.assertEqual(row["graph"], graph)

    def test_graph_absent_when_none(self):
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, graph=None)
        self.assertNotIn("graph", row)

    def test_error_step_absent_when_success(self):
        results = self._make_results(ort_ok=True, light_ok=True)
        row = rlb._row_from_results("test_relu", results)
        self.assertNotIn("onnxruntime_error_step", row)
        self.assertNotIn("onnx_light_error_step", row)


class TestRunBenchmark(unittest.TestCase):
    def test_unknown_backend_returns_failure(self):
        result = rlb.run_benchmark(None, [], "totally-unknown-backend")
        self.assertFalse(result["success"])
        self.assertIn("unknown backend", result["error"])
        self.assertEqual(result["error_step"], "load")

    def test_empty_data_sets_returns_failure(self):
        # Even with a valid backend name, empty data_sets must fail cleanly.
        result = rlb.run_benchmark(None, [], "onnxruntime")
        self.assertFalse(result["success"])
        self.assertIn("no test_data_set", result["error"])
        self.assertEqual(result["error_step"], "load")

    @unittest.skipUnless(_HAS_NUMPY, "numpy required")
    def test_runner_factory_success(self):
        """run_benchmark succeeds when injecting a dummy runner via monkey-patching."""
        import numpy as np

        call_log = []

        def _dummy_factory(model):
            def _run(inputs):
                call_log.append(inputs)
                return [np.zeros((2,), dtype=np.float32)]

            return _run

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _dummy_factory
            dummy_model = object()
            inputs = [np.ones((2,), dtype=np.float32)]
            data_sets = [(inputs, [np.zeros((2,), dtype=np.float32)])]
            result = rlb.run_benchmark(
                dummy_model, data_sets, "onnxruntime", n_warmup=2, n_measure=5
            )
        finally:
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertTrue(result["success"])
        self.assertEqual(result["n_warmup"], 2)
        self.assertEqual(result["n_measure"], 5)
        self.assertIn("avg_ms", result)
        self.assertIn("min_ms", result)
        self.assertIn("max_ms", result)
        # warmup (2) + measure (5) = 7 calls
        self.assertEqual(len(call_log), 7)

    @unittest.skipUnless(_HAS_NUMPY, "numpy required")
    def test_runner_factory_load_failure(self):
        def _bad_factory(model):
            raise RuntimeError("cannot load model")

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _bad_factory
            result = rlb.run_benchmark(
                object(), [([],)], "onnxruntime", n_warmup=1, n_measure=3
            )
        finally:
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "load")
        self.assertIn("cannot load model", result["error"])

    @unittest.skipUnless(_HAS_NUMPY, "numpy required")
    def test_runner_factory_measure_failure(self):
        import numpy as np

        n_calls = [0]

        def _failing_factory(model):
            def _run(inputs):
                n_calls[0] += 1
                if n_calls[0] > 2:  # warmup passes; first timed call fails
                    raise RuntimeError("run error")
                return [np.zeros((1,))]

            return _run

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _failing_factory
            inputs = [np.ones((1,))]
            data_sets = [(inputs, [np.zeros((1,))])]
            result = rlb.run_benchmark(
                object(), data_sets, "onnxruntime", n_warmup=2, n_measure=5
            )
        finally:
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "measure")
        self.assertIn("run error", result["error"])


class TestBuildPayload(unittest.TestCase):
    @unittest.skipUnless(_HAS_NUMPY, "numpy required")
    def test_build_payload_with_stub_discover_and_run(self):
        """build_payload wires together discovery and benchmarking correctly."""
        import numpy as np

        def _discover(_kind):
            return [
                {
                    "name": "test_abs",
                    "model": object(),
                    "data_sets": [
                        ([np.array([1.0])], [np.array([1.0])])
                    ],
                    "tag": "",
                },
                {
                    "name": "test_relu",
                    "model": object(),
                    "data_sets": [
                        ([np.array([-1.0, 2.0])], [np.array([0.0, 2.0])])
                    ],
                    "tag": "mygroup",
                },
            ]

        call_log = []

        def _run(model, data_sets, backend, n_warmup, n_measure):
            call_log.append((backend, n_warmup, n_measure))
            return {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": 1.5,
                "min_ms": 1.0,
                "max_ms": 2.0,
                "n_warmup": n_warmup,
                "n_measure": n_measure,
            }

        payload = rlb.build_payload(
            kind="node",
            n_warmup=2,
            n_measure=7,
            discover=_discover,
            run=_run,
            versions=lambda: {"onnxruntime": "1.0", "onnx_light": "0.1"},
        )

        self.assertIn("date", payload)
        self.assertIn("kind", payload)
        self.assertEqual(payload["n_warmup"], 2)
        self.assertEqual(payload["n_measure"], 7)
        self.assertIn("summary", payload)
        self.assertIn("tests", payload)
        self.assertEqual(len(payload["tests"]), 2)

        # Each test should have been run against both BENCHMARK_BACKENDS.
        for backend in rlb.BENCHMARK_BACKENDS:
            self.assertGreater(
                sum(1 for b, _, _ in call_log if b == backend), 0,
                msg=f"{backend} was never called",
            )

        # Check that both warm-up and measure counts are forwarded.
        for backend, nw, nm in call_log:
            self.assertEqual(nw, 2)
            self.assertEqual(nm, 7)

        # test_abs and test_relu both succeeded on both backends → speedup set.
        for row in payload["tests"]:
            self.assertIn("speedup", row, msg=row["name"])

        # Summary stats should include avg_speedup.
        summary = payload["summary"]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["both_succeeded"], 2)
        self.assertIn("avg_speedup", summary)

    def test_build_payload_limit(self):
        """The ``limit`` parameter caps the number of tests."""

        def _discover(_kind):
            return [
                {"name": f"test_{i}", "model": object(), "data_sets": [([],)], "tag": ""}
                for i in range(20)
            ]

        def _run(model, data_sets, backend, n_warmup, n_measure):
            return {
                "success": False,
                "error": "no data",
                "error_step": "load",
            }

        payload = rlb.build_payload(
            kind="node",
            limit=5,
            discover=_discover,
            run=_run,
            versions=lambda: {},
        )
        self.assertEqual(len(payload["tests"]), 5)


class TestWriteAndLoadPayload(unittest.TestCase):
    def test_write_and_reload(self):
        payload = {
            "date": "2026-01-01T00:00:00Z",
            "kind": "node",
            "n_warmup": 3,
            "n_measure": 10,
            "summary": {"total": 1, "both_succeeded": 1, "avg_speedup": 2.0},
            "tests": [
                {
                    "name": "test_abs",
                    "onnxruntime_success": True,
                    "onnxruntime_avg_ms": 1.0,
                    "onnx_light_success": True,
                    "onnx_light_avg_ms": 0.5,
                    "speedup": 2.0,
                }
            ],
            "versions": {"onnxruntime": "1.27.0", "onnx_light": "0.1.5"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "benchmark.json")
            rlb.write_payload(path, payload)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
        self.assertEqual(loaded["date"], payload["date"])
        self.assertEqual(len(loaded["tests"]), 1)
        self.assertAlmostEqual(loaded["tests"][0]["speedup"], 2.0)

    def test_load_missing_file_returns_empty_dict(self):
        result = rlb.load_previous_payload("/nonexistent/path/benchmark.json")
        self.assertEqual(result, {})

    def test_load_malformed_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json {{")
            path = f.name
        try:
            result = rlb.load_previous_payload(path)
            self.assertEqual(result, {})
        finally:
            os.unlink(path)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = rlb.parse_args([])
        self.assertEqual(args.kind, rlb.DEFAULT_KIND)
        self.assertEqual(args.n_warmup, rlb.N_WARMUP)
        self.assertEqual(args.n_measure, rlb.N_MEASURE)
        self.assertIsNone(args.limit)

    def test_override(self):
        args = rlb.parse_args(["--n-warmup", "5", "--n-measure", "20", "--limit", "10"])
        self.assertEqual(args.n_warmup, 5)
        self.assertEqual(args.n_measure, 20)
        self.assertEqual(args.limit, 10)


class TestBenchmarkBackends(unittest.TestCase):
    def test_benchmark_backends_contains_ort_and_onnx_light(self):
        self.assertIn("onnxruntime", rlb.BENCHMARK_BACKENDS)
        self.assertIn("onnx_light", rlb.BENCHMARK_BACKENDS)
        self.assertNotIn("reference", rlb.BENCHMARK_BACKENDS)

    def test_runner_factories_match_benchmark_backends(self):
        for b in rlb.BENCHMARK_BACKENDS:
            self.assertIn(b, rlb._RUNNER_FACTORIES, msg=f"no factory for {b}")


if __name__ == "__main__":
    unittest.main()
