"""Tests for ``scripts.record_onnx_light_benchmark``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_benchmark as rlb  # noqa: E402


class TestOnnxruntimeRunner(unittest.TestCase):
    """``_make_onnxruntime_runner`` builds a plain CPU session and never edits the model.

    Models come as-is from the packages the benchmark measures, so the runner
    must pass the serialized model through unchanged and let any session-build
    error propagate.
    """

    class _FakeModel:
        def __init__(self, serialized=b"model-bytes"):
            self._serialized = serialized

        def SerializeToString(self):
            return self._serialized

    def _with_modules(self, modules, fn):
        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            return fn()
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_session_built_from_model_bytes_as_is(self):
        import types

        created = []

        class _SessionOptions:
            def __init__(self):
                self.entries = {}

            def add_session_config_entry(self, key, value):
                self.entries[key] = value

        class _Session:
            def __init__(self, serialized, sess_options=None, providers=None):
                created.append((serialized, sess_options, providers))

            def get_inputs(self):
                return []

            def run(self, output_names, feeds):
                return ["output"]

        onnxruntime = types.ModuleType("onnxruntime")
        onnxruntime.SessionOptions = _SessionOptions
        onnxruntime.InferenceSession = _Session
        model = self._FakeModel(serialized=b"corpus-model")
        runner = self._with_modules(
            {"onnxruntime": onnxruntime},
            lambda: rlb._make_onnxruntime_runner(model),
        )
        # The serialized model is passed through unchanged on the CPU provider.
        self.assertEqual(created[0][0], b"corpus-model")
        self.assertEqual(created[0][2], ["CPUExecutionProvider"])
        self.assertIsNone(created[0][1])
        self.assertEqual(runner([]), ["output"])

    def test_session_error_propagates(self):
        import types

        class _SessionOptions:
            def add_session_config_entry(self, key, value):
                pass

        class _BrokenSession:
            def __init__(self, serialized, sess_options=None, providers=None):
                raise RuntimeError("Unsupported model IR version: 14")

        onnxruntime = types.ModuleType("onnxruntime")
        onnxruntime.SessionOptions = _SessionOptions
        onnxruntime.InferenceSession = _BrokenSession
        model = self._FakeModel()
        with self.assertRaises(RuntimeError) as ctx:
            self._with_modules(
                {"onnxruntime": onnxruntime},
                lambda: rlb._make_onnxruntime_runner(model),
            )
        self.assertIn("Unsupported model IR version", str(ctx.exception))


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
    def _make_results(
        self,
        ort_ok=True,
        light_ok=True,
        ort_avg=1.0,
        light_avg=0.5,
        cpu_ok=True,
        cpu_avg=0.25,
    ):
        results = {}
        if ort_ok:
            results["onnxruntime"] = {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": ort_avg,
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
                "n_warmup": 3,
                "n_measure": 10,
            }
        else:
            results["onnx_light"] = {
                "success": False,
                "error": "light load error",
                "error_step": "load",
            }
        if cpu_ok:
            results["onnx_light_cpu"] = {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": cpu_avg,
                "n_warmup": 3,
                "n_measure": 10,
            }
        else:
            results["onnx_light_cpu"] = {
                "success": False,
                "error": "cpu load error",
                "error_step": "load",
            }
        return results

    def test_speedup_computed_when_both_succeed(self):
        results = self._make_results(
            ort_ok=True, light_ok=True, ort_avg=2.0, light_avg=1.0
        )
        row = rlb._row_from_results("test_relu", results)
        self.assertEqual(row["name"], "test_relu")
        self.assertTrue(row["onnxruntime_success"])
        self.assertTrue(row["onnx_light_success"])
        self.assertAlmostEqual(row["speedup"], 2.0)
        self.assertAlmostEqual(row["onnxruntime_avg_ms"], 2.0)
        self.assertAlmostEqual(row["onnx_light_avg_ms"], 1.0)

    def test_min_max_ms_surfaced_when_present(self):
        results = self._make_results(
            ort_ok=True, light_ok=True, ort_avg=2.0, light_avg=1.0
        )
        results["onnxruntime"]["min_ms"] = 1.5
        results["onnxruntime"]["max_ms"] = 3.0
        row = rlb._row_from_results("test_relu", results)
        self.assertAlmostEqual(row["onnxruntime_min_ms"], 1.5)
        self.assertAlmostEqual(row["onnxruntime_max_ms"], 3.0)

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
        results = self._make_results(
            ort_ok=True, light_ok=True, ort_avg=4.0, light_avg=1.0
        )
        row = rlb._row_from_results("test_abs", results)
        self.assertGreater(row["speedup"], 1.0)

    def test_speedup_lt_one_when_onnx_light_is_slower(self):
        # ort_avg < light_avg → speedup < 1 → onnx-light is slower
        results = self._make_results(
            ort_ok=True, light_ok=True, ort_avg=1.0, light_avg=4.0
        )
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

    def test_input_type_included_when_provided(self):
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, input_type="float32")
        self.assertEqual(row["input_type"], "float32")

    def test_input_type_absent_when_empty(self):
        results = self._make_results()
        row = rlb._row_from_results("test_relu", results, input_type="")
        self.assertNotIn("input_type", row)

    def test_error_step_absent_when_success(self):
        results = self._make_results(ort_ok=True, light_ok=True)
        row = rlb._row_from_results("test_relu", results)
        self.assertNotIn("onnxruntime_error_step", row)
        self.assertNotIn("onnx_light_error_step", row)

    def test_speedup_cpu_computed_when_ort_and_cpu_succeed(self):
        results = self._make_results(
            ort_ok=True, light_ok=True, ort_avg=2.0, light_avg=1.0, cpu_avg=0.5
        )
        row = rlb._row_from_results("test_abs", results)
        self.assertTrue(row["onnx_light_cpu_success"])
        self.assertAlmostEqual(row["onnx_light_cpu_avg_ms"], 0.5)
        # speedup_cpu = ort_avg / cpu_avg = 2.0 / 0.5 = 4.0
        self.assertAlmostEqual(row["speedup_cpu"], 4.0)

    def test_no_speedup_cpu_when_cpu_fails(self):
        results = self._make_results(ort_ok=True, light_ok=True, cpu_ok=False)
        row = rlb._row_from_results("test_abs", results)
        self.assertFalse(row["onnx_light_cpu_success"])
        self.assertNotIn("speedup_cpu", row)
        self.assertIn("onnx_light_cpu_error", row)

    def test_no_speedup_cpu_when_ort_fails(self):
        results = self._make_results(ort_ok=False, light_ok=True, cpu_ok=True)
        row = rlb._row_from_results("test_abs", results)
        self.assertNotIn("speedup_cpu", row)


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
        self.assertLessEqual(result["min_ms"], result["avg_ms"])
        self.assertLessEqual(result["avg_ms"], result["max_ms"])
        # warmup (2) + measure (5) = 7 calls
        self.assertEqual(len(call_log), 7)

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

    def test_runner_factory_measure_failure(self):
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

    def test_avg_ms_is_trimmed_mean_excluding_min_and_max(self):
        """avg_ms averages the sorted samples with the fastest/slowest dropped."""
        # Deterministic per-iteration durations (ms): one slow outlier that
        # would skew a plain mean, and one very fast iteration.
        durations_ms = [1.0, 1.0, 1.0, 0.0, 100.0]
        ticks = []
        for d in durations_ms:
            start = len(ticks)
            ticks.append(float(start))
            ticks.append(float(start) + d / 1_000.0)

        tick_iter = iter(ticks)

        def _dummy_factory(model):
            def _run(inputs):
                return [np.zeros((1,))]

            return _run

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        saved_pc = rlb.time.perf_counter
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _dummy_factory
            rlb.time.perf_counter = lambda: next(tick_iter)
            data_sets = [([np.ones((1,))], [np.zeros((1,))])]
            result = rlb.run_benchmark(
                object(), data_sets, "onnxruntime", n_warmup=0, n_measure=5
            )
        finally:
            rlb.time.perf_counter = saved_pc
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertTrue(result["success"])
        # min/max preserved as the raw extremes.
        self.assertAlmostEqual(result["min_ms"], 0.0, places=6)
        self.assertAlmostEqual(result["max_ms"], 100.0, places=6)
        # avg drops 0.0 and 100.0, leaving [1.0, 1.0, 1.0] -> 1.0.
        self.assertAlmostEqual(result["avg_ms"], 1.0, places=6)

    def test_measurement_stops_after_cumulative_duration(self):
        ticks = iter((0.0, 1.1, 2.0, 3.1, 4.0, 5.1))

        def _dummy_factory(model):
            return lambda inputs: [np.zeros((1,))]

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        saved_pc = rlb.time.perf_counter
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _dummy_factory
            rlb.time.perf_counter = lambda: next(ticks)
            result = rlb.run_benchmark(
                object(),
                [([np.ones((1,))], [np.zeros((1,))])],
                "onnxruntime",
                n_warmup=0,
                n_measure=5,
                max_repeat_time_s=1.0,
            )
        finally:
            rlb.time.perf_counter = saved_pc
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertTrue(result["success"])
        self.assertEqual(result["n_warmup"], 0)
        self.assertEqual(result["n_measure"], 3)
        self.assertEqual(result["avg_ms"], 1100.0)

    def test_warmup_stops_after_cumulative_duration(self):
        ticks = iter((0.0, 1.1, 2.0, 2.1))

        def _dummy_factory(model):
            return lambda inputs: [np.zeros((1,))]

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        saved_pc = rlb.time.perf_counter
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _dummy_factory
            rlb.time.perf_counter = lambda: next(ticks)
            result = rlb.run_benchmark(
                object(),
                [([np.ones((1,))], [np.zeros((1,))])],
                "onnxruntime",
                n_warmup=5,
                n_measure=1,
                max_repeat_time_s=1.0,
            )
        finally:
            rlb.time.perf_counter = saved_pc
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertTrue(result["success"])
        self.assertEqual(result["n_warmup"], 1)
        self.assertEqual(result["n_measure"], 1)

    def test_warmup_uses_independent_time_limit(self):
        ticks = iter((0.0, 0.06, 1.0, 1.01))

        def _dummy_factory(model):
            return lambda inputs: [np.zeros((1,))]

        saved = rlb._RUNNER_FACTORIES.get("onnxruntime")
        saved_pc = rlb.time.perf_counter
        try:
            rlb._RUNNER_FACTORIES["onnxruntime"] = _dummy_factory
            rlb.time.perf_counter = lambda: next(ticks)
            result = rlb.run_benchmark(
                object(),
                [([np.ones((1,))], [np.zeros((1,))])],
                "onnxruntime",
                n_warmup=2,
                n_measure=1,
                max_warmup_time_s=0.05,
                max_repeat_time_s=0.2,
            )
        finally:
            rlb.time.perf_counter = saved_pc
            if saved is None:
                del rlb._RUNNER_FACTORIES["onnxruntime"]
            else:
                rlb._RUNNER_FACTORIES["onnxruntime"] = saved

        self.assertTrue(result["success"])
        self.assertEqual(result["n_warmup"], 1)
        self.assertEqual(result["n_measure"], 1)


class TestSymbolicCost(unittest.TestCase):
    def test_none_when_no_data_sets(self):
        self.assertIsNone(rlb._symbolic_cost([]))

    def test_sums_input_and_output_elements(self):
        inputs = [np.zeros((2, 3)), np.zeros((4,))]
        outputs = [np.zeros((5, 5))]
        data_sets = [(inputs, outputs)]
        # 2*3 + 4 + 5*5 = 6 + 4 + 25 = 35
        self.assertEqual(rlb._symbolic_cost(data_sets), 35)

    def test_matrix_multiplication_and_attention_are_quadratic(self):
        data_sets = [([np.zeros((2, 3)), np.zeros((3, 4))], [np.zeros((2, 4))])]
        size = 6 + 12 + 8
        for operator in ("MatMul", "Gemm", "Attention", "QLinearMatMul"):
            with self.subTest(operator=operator):
                self.assertEqual(rlb._symbolic_cost(data_sets, operator), size**2)

    def test_unary_and_binary_operators_are_linear(self):
        data_sets = [([np.zeros((8,)), np.zeros((8,))], [np.zeros((8,))])]
        self.assertEqual(rlb._symbolic_cost(data_sets, "Abs"), 24)
        self.assertEqual(rlb._symbolic_cost(data_sets, "Add"), 24)

    def test_only_uses_first_data_set(self):
        small = ([np.zeros((1,))], [np.zeros((1,))])
        big = ([np.zeros((100,))], [np.zeros((100,))])
        self.assertEqual(rlb._symbolic_cost([small, big]), 2)

    def _model_with_output_shapes(self, shapes):
        """Return a minimal model exposing ``graph.output`` shapes.

        ``None`` dimensions stand for dynamic (symbolic) dimensions.
        """

        from types import SimpleNamespace

        outputs = []
        for shape in shapes:
            dims = [SimpleNamespace(dim_value=0 if d is None else d) for d in shape]
            outputs.append(
                SimpleNamespace(
                    type=SimpleNamespace(
                        tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=dims))
                    )
                )
            )
        return SimpleNamespace(graph=SimpleNamespace(output=outputs))

    def test_declared_graph_outputs_are_used_when_data_set_has_none(self):
        # AffineGrid-like test: tiny inputs (1x2x3 + 4 = 10 elements) but a much
        # larger declared output (1x1448x1448x2 = 4194304 elements).
        model = self._model_with_output_shapes([[1, 1448, 1448, 2]])
        data_sets = [([np.zeros((1, 2, 3)), np.zeros((4,), dtype=np.int64)], [])]
        self.assertEqual(
            rlb._symbolic_cost(data_sets, "AffineGrid", model), 10 + 1448 * 1448 * 2
        )
        # Without the model, only the inputs are counted.
        self.assertEqual(rlb._symbolic_cost(data_sets, "AffineGrid"), 10)

    def test_declared_graph_outputs_ignore_dynamic_dimensions(self):
        model = self._model_with_output_shapes([[None, 8]])
        data_sets = [([np.zeros((4,))], [])]
        self.assertEqual(rlb._symbolic_cost(data_sets, "Abs", model), 4)

    def test_data_set_outputs_win_over_declared_shapes(self):
        model = self._model_with_output_shapes([[1000]])
        data_sets = [([np.zeros((4,))], [np.zeros((4,))])]
        self.assertEqual(rlb._symbolic_cost(data_sets, "Abs", model), 8)


class TestFirstInputType(unittest.TestCase):
    def test_empty_when_no_data_sets(self):
        self.assertEqual(rlb._first_input_type([]), "")

    def test_empty_when_no_inputs(self):
        self.assertEqual(rlb._first_input_type([([], [])]), "")

    def test_dtype_name_of_first_input(self):
        data_sets = [([np.zeros((2, 3), dtype=np.float32), np.zeros(4)], [])]
        self.assertEqual(rlb._first_input_type(data_sets), "float32")

    def test_uses_first_input_and_first_data_set(self):
        first = ([np.zeros((1,), dtype=np.int64)], [])
        second = ([np.zeros((1,), dtype=np.float32)], [])
        self.assertEqual(rlb._first_input_type([first, second]), "int64")

    def test_descends_into_sequence_input(self):
        data_sets = [([[np.zeros((1,), dtype=np.float64)]], [])]
        self.assertEqual(rlb._first_input_type(data_sets), "float64")

    def test_empty_when_no_dtype(self):
        data_sets = [([object()], [])]
        self.assertEqual(rlb._first_input_type(data_sets), "")


class TestOperatorName(unittest.TestCase):
    def _model(self, op_types):
        class Node:
            def __init__(self, op_type):
                self.op_type = op_type

        class Graph:
            def __init__(self, nodes):
                self.node = nodes

        class Model:
            def __init__(self, nodes):
                self.graph = Graph(nodes)

        return Model([Node(op_type) for op_type in op_types])

    def test_single_node(self):
        self.assertEqual(rlb._operator_name(self._model(["Add"])), "Add")

    def test_multiple_nodes_joined_and_deduplicated(self):
        model = self._model(["Cast", "Add", "Cast"])
        self.assertEqual(rlb._operator_name(model), "Cast+Add")

    def test_empty_without_graph(self):
        self.assertEqual(rlb._operator_name(object()), "")


class TestOperatorWeights(unittest.TestCase):
    def test_aggregates_weight_and_count_per_operator(self):
        rows = [
            {"operator": "Add", "cost_n": 10},
            {"operator": "Add", "cost_n": 5},
            {"operator": "Gemm", "cost_n": 1000},
        ]
        weights = rlb._operator_weights(rows)
        self.assertEqual(
            weights,
            [
                {"operator": "Gemm", "weight": 1000, "tests": 1},
                {"operator": "Add", "weight": 15, "tests": 2},
            ],
        )

    def test_skips_rows_missing_operator_or_weight(self):
        rows = [
            {"operator": "", "cost_n": 10},
            {"cost_n": 10},
            {"operator": "Add", "cost_n": None},
            {"operator": "Add", "cost_n": 0},
        ]
        self.assertEqual(rlb._operator_weights(rows), [])


class TestWeightedAvgSpeedup(unittest.TestCase):
    def test_none_when_no_rows(self):
        self.assertIsNone(rlb._weighted_avg_speedup([], "speedup"))

    def test_weighted_average_favors_expensive_tests(self):
        # A cheap O(1)-like test (tiny tensors, small cost_n) where onnx-light
        # is much slower, and an expensive O(n^2)-like test (large tensors,
        # large cost_n) where onnx-light is much faster. The unweighted mean
        # of the two speedups treats both tests equally, but the weighted
        # average should track the costly test's speedup much more closely
        # since it processes almost all of the aggregate data.
        rows = [
            {"speedup": 0.1, "cost_n": 4},  # cheap, slower
            {"speedup": 10.0, "cost_n": 10_000},  # costly, faster
        ]
        unweighted = sum(r["speedup"] for r in rows) / len(rows)
        weighted = rlb._weighted_avg_speedup(rows, "speedup")
        self.assertGreater(weighted, 1.0)
        # The costly test dominates, so the weighted average is much closer
        # to its 10x speedup than the unweighted mean of ~5.05x.
        self.assertGreater(weighted, unweighted)
        expected = (0.1 * 4 + 10.0 * 10_000) / (4 + 10_000)
        self.assertAlmostEqual(weighted, expected, places=4)

    def test_skips_rows_missing_or_zero_values(self):
        rows = [
            {"speedup": 1.0, "cost_n": 0},
            {"speedup": 2.0},
            {"speedup": 4.0, "cost_n": 2},
        ]
        weighted = rlb._weighted_avg_speedup(rows, "speedup")
        self.assertAlmostEqual(weighted, 4.0, places=4)


class TestSumLatencySpeedup(unittest.TestCase):
    def test_ratio_uses_sums_of_latencies(self):
        rows = [
            {"onnxruntime_avg_ms": 2.0, "onnx_light_avg_ms": 1.0},
            {"onnxruntime_avg_ms": 8.0, "onnx_light_avg_ms": 4.0},
        ]
        self.assertEqual(rlb._sum_latency_speedup(rows, "onnx_light_avg_ms"), 2.0)

    def test_skips_missing_and_invalid_latencies(self):
        rows = [
            {"onnxruntime_avg_ms": 2.0, "onnx_light_avg_ms": 0.0},
            {"onnxruntime_avg_ms": 3.0},
            {"onnxruntime_avg_ms": 8.0, "onnx_light_avg_ms": 4.0},
        ]
        self.assertEqual(rlb._sum_latency_speedup(rows, "onnx_light_avg_ms"), 2.0)


def _stub_model(op_type):
    """Return a minimal object exposing ``model.graph.node[i].op_type``."""

    class Node:
        def __init__(self, op_type):
            self.op_type = op_type

    class Graph:
        def __init__(self, op_type):
            self.node = [Node(op_type)]

    class Model:
        def __init__(self, op_type):
            self.graph = Graph(op_type)

    return Model(op_type)


class TestBuildPayload(unittest.TestCase):
    def test_build_payload_with_stub_discover_and_run(self):
        """build_payload wires together discovery and benchmarking correctly."""

        def _discover(_kind):
            return [
                {
                    "name": "test_abs",
                    "model": _stub_model("Abs"),
                    "data_sets": [([np.array([1.0])], [np.array([1.0])])],
                    "tag": "",
                },
                {
                    "name": "test_relu",
                    "model": _stub_model("Relu"),
                    "data_sets": [([np.array([-1.0, 2.0])], [np.array([0.0, 2.0])])],
                    "tag": "mygroup",
                },
            ]

        call_log = []

        def _run(
            model,
            data_sets,
            backend,
            n_warmup,
            n_measure,
            max_repeat_time_s,
        ):
            call_log.append((backend, n_warmup, n_measure, max_repeat_time_s))
            return {
                "success": True,
                "error": "",
                "error_step": "",
                "avg_ms": 1.5,
                "n_warmup": n_warmup,
                "n_measure": n_measure,
            }

        payload = rlb.build_payload(
            kind="node",
            n_warmup=2,
            n_measure=7,
            max_repeat_time_s=0.5,
            discover=_discover,
            run=_run,
            versions=lambda: {"onnxruntime": "1.0", "onnx_light": "0.1"},
        )

        self.assertIn("date", payload)
        self.assertIn("kind", payload)
        self.assertEqual(payload["n_warmup"], 2)
        self.assertEqual(payload["n_measure"], 7)
        self.assertEqual(payload["max_repeat_time_s"], 0.5)
        self.assertIn("summary", payload)
        self.assertIn("tests", payload)
        self.assertEqual(len(payload["tests"]), 2)

        # Each test should have been run against both BENCHMARK_BACKENDS.
        for backend in rlb.BENCHMARK_BACKENDS:
            self.assertGreater(
                sum(1 for b, _, _, _ in call_log if b == backend),
                0,
                msg=f"{backend} was never called",
            )

        # Check that both warm-up and measure counts are forwarded.
        for backend, nw, nm, max_repeat_time_s in call_log:
            self.assertEqual(nw, 2)
            self.assertEqual(nm, 7)
            self.assertEqual(max_repeat_time_s, 0.5)
        self.assertEqual(
            [backend for backend, _, _, _ in call_log],
            [backend for backend in rlb.BENCHMARK_EXECUTION_ORDER for _ in range(2)],
        )

        # test_abs and test_relu both succeeded on both backends → speedup set.
        for row in payload["tests"]:
            self.assertIn("speedup", row, msg=row["name"])

        # Summary stats should include avg_speedup.
        summary = payload["summary"]
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["both_succeeded"], 2)
        self.assertIn("avg_speedup", summary)
        # A cost-weighted average speedup is reported alongside the
        # unweighted mean of per-test ratios.
        self.assertIn("avg_speedup_weighted", summary)
        self.assertIn("speedup_sum_latency", summary)
        # The onnx-light-cpu backend is timed as well, so its summary and
        # per-row speedup are present too.
        self.assertEqual(summary["cpu_succeeded"], 2)
        self.assertIn("avg_speedup_cpu", summary)
        self.assertIn("avg_speedup_weighted_cpu", summary)
        self.assertIn("speedup_sum_latency_cpu", summary)
        for row in payload["tests"]:
            self.assertIn("speedup_cpu", row, msg=row["name"])
            self.assertIn("cost_n", row, msg=row["name"])
            self.assertEqual(row["cost_complexity"], "linear")

        # Each row's operator is derived from its model's node op_type, and
        # the summary exposes the symbolic weight attributed to each
        # operator that feeds into avg_speedup_weighted.
        operators = {row["name"]: row.get("operator") for row in payload["tests"]}
        self.assertEqual(operators, {"test_abs": "Abs", "test_relu": "Relu"})

        # Each row records the type of its first input.
        input_types = {row["name"]: row.get("input_type") for row in payload["tests"]}
        self.assertEqual(
            input_types, {"test_abs": "float64", "test_relu": "float64"}
        )
        self.assertIn("operator_weights", summary)
        weighted_operators = {w["operator"] for w in summary["operator_weights"]}
        self.assertEqual(weighted_operators, {"Abs", "Relu"})
        for entry in summary["operator_weights"]:
            self.assertEqual(entry["tests"], 1)
            self.assertGreater(entry["weight"], 0)

    def test_build_payload_limit(self):
        """The ``limit`` parameter caps the number of tests."""

        def _discover(_kind):
            return [
                {
                    "name": f"test_{i}",
                    "model": object(),
                    "data_sets": [([],)],
                    "tag": "",
                }
                for i in range(20)
            ]

        def _run(
            model,
            data_sets,
            backend,
            n_warmup,
            n_measure,
            max_repeat_time_s,
        ):
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
        self.assertEqual(args.max_repeat_time, rlb.MAX_REPEAT_TIME_S)
        self.assertEqual(rlb.N_WARMUP, 2 * (os.cpu_count() or 1))
        self.assertEqual(rlb.N_MEASURE, 10 * (os.cpu_count() or 1))
        self.assertIsNone(args.limit)

    def test_override(self):
        args = rlb.parse_args(
            [
                "--n-warmup",
                "5",
                "--n-measure",
                "20",
                "--max-repeat-time",
                "0.25",
                "--limit",
                "10",
            ]
        )
        self.assertEqual(args.n_warmup, 5)
        self.assertEqual(args.n_measure, 20)
        self.assertEqual(args.max_repeat_time, 0.25)
        self.assertEqual(args.limit, 10)


class TestOnnxLightReferenceRunner(unittest.TestCase):
    """The onnx-light runner uses the public NumPy-based evaluator API."""

    def _install_fake_onnx_light(self, out_value, input_kind="tensor", output_kind="tensor"):
        """Register minimal fake ``onnx_light`` modules exposing the runtime API.

        Returns ``(model, telemetry)`` where ``telemetry`` records how many
        ``ExecutionPlan`` / ``RuntimeSession`` objects were built and how many
        times the session was run, so the test can assert the plan/session are
        built once and reused. ``input_kind`` / ``output_kind`` select the graph
        boundary type (``"tensor"``, ``"sequence"`` or ``"map"``) so the runner's
        non-tensor stores can be exercised.
        """
        import types

        import onnx

        telemetry = {
            "plans": 0,
            "sessions": 0,
            "runs": 0,
            "registered": 0,
            "registered_kernels": 0,
        }

        def _type_for(kind):
            return None if kind == "tensor" else _FakeTypeProto(kind)

        class _FakeModelProto:
            def SerializeToString(self):
                return b"model"

        class _Named:
            def __init__(self, name, type_proto=None):
                self.name = name
                self.type = type_proto

        class _Opset:
            def __init__(self, domain, version):
                self.domain = domain
                self.version = version

        class _Graph:
            def __init__(self):
                self.input = [_Named("x", _type_for(input_kind))]
                self.output = [_Named("y", _type_for(output_kind))]
                self.initializer = []

        model = _FakeModelProto()
        model.opset_import = [_Opset("", 18)]
        model.graph = _Graph()

        class _OutTensor:
            data_type = int(onnx.TensorProto.FLOAT)
            shape = (2,)

        out_tensor = _OutTensor()

        class _RuntimeContext:
            def __init__(self, kctx):
                self._store = {}
                self._sequences = {}
                self._maps = {}
                self._custom_kernels = {}

            def set(self, name, tensor, kind=None):
                self._store[name] = tensor

            def has(self, name):
                return name in self._store

            def get(self, name):
                return self._store[name]

            def put(self, name, tensor, kind=None):
                self._store[name] = tensor

            def register_custom_kernel(self, domain, op_type, fn):
                telemetry["registered_kernels"] += 1
                self._custom_kernels[(domain, op_type)] = fn

            def put_sequence(self, name, values):
                self._sequences[name] = list(values)

            def get_sequence(self, name):
                return self._sequences[name]

            def put_map(self, name, mapping):
                self._maps[name] = dict(mapping)

            def get_map(self, name):
                return self._maps[name]

        class _RuntimeSession:
            def __init__(self, plan):
                telemetry["sessions"] += 1
                self._plan = plan

            def run(self, ctx):
                telemetry["runs"] += 1
                if output_kind == "sequence":
                    ctx.put_sequence("y", [out_tensor])
                elif output_kind == "map":
                    ctx.put_map("y", {1: 2})
                else:
                    ctx.set("y", out_tensor)

        class _ExecutionPlan:
            def __init__(self, graph):
                telemetry["plans"] += 1
                self.graph = graph

        runtime = types.ModuleType("onnx_light.onnx_py._onnxpykernels.runtime")
        runtime.RuntimeContext = _RuntimeContext
        runtime.KernelContext = lambda opset: ("kctx", opset)
        runtime.default_opset = lambda v: v
        runtime.ExecutionPlan = _ExecutionPlan
        runtime.RuntimeSession = _RuntimeSession
        runtime.tensor_from_proto = lambda tp: ("tensor", tp)
        runtime.tensor_from_numpy = lambda name, dtype, shape, raw, copy=True: (
            "tensor",
            raw.view(rlb._cc_numpy_dtype_for(dtype)).reshape(shape),
        )

        def _register(model_, ctx):
            telemetry["registered"] += 1

        runtime.register_model_functions = _register
        # Mirrors the real ``runtime.tensor_to_numpy`` which returns a 1-D
        # uint8 byte view; ``_runtime_tensor_to_numpy`` reinterprets it as the
        # tensor's dtype (float32 here) and reshapes it.
        runtime.tensor_to_numpy = lambda t: np.asarray(out_value, dtype=np.float32).view(np.uint8)
        runtime.tensor_to_proto = lambda t: t

        numpy_helper = types.ModuleType("onnx_light.onnx_lib.numpy_helper")
        numpy_helper.from_array = lambda arr, name=None: {"name": name, "arr": arr}
        numpy_helper.to_array = lambda tp: np.asarray(out_value, dtype=np.float32)

        onnx_lib = types.ModuleType("onnx_light.onnx_lib")
        onnx_lib.ModelProto = _FakeModelProto
        onnx_lib.numpy_helper = numpy_helper

        onnx_light = types.ModuleType("onnx_light")
        onnx_pkg = types.ModuleType("onnx_light.onnx")
        reference = types.ModuleType("onnx_light.onnx.reference")

        class _ReferenceEvaluator:
            def __init__(self, model_bytes, cpu_execution=None):
                self.input_names = ["x"]
                telemetry["plans"] += 1
                telemetry["sessions"] += 1
                telemetry["cpu_execution"] = cpu_execution

            def run(self, output_names, feeds):
                telemetry["runs"] += 1
                telemetry.setdefault("feeds", []).append(feeds)
                if output_kind == "sequence":
                    return [[np.asarray(out_value)]]
                if output_kind == "map":
                    return [{1: 2}]
                return [np.asarray(out_value)]

            def used_kernels(self):
                return []

        reference.ReferenceEvaluator = _ReferenceEvaluator
        onnx_py = types.ModuleType("onnx_light.onnx_py")
        pyk = types.ModuleType("onnx_light.onnx_py._onnxpykernels")
        pyk.runtime = runtime
        onnx_py._onnxpykernels = pyk

        modules = {
            "onnx_light": onnx_light,
            "onnx_light.onnx": onnx_pkg,
            "onnx_light.onnx.reference": reference,
            "onnx_light.onnx_lib": onnx_lib,
            "onnx_light.onnx_lib.numpy_helper": numpy_helper,
            "onnx_light.onnx_py": onnx_py,
            "onnx_light.onnx_py._onnxpykernels": pyk,
            "onnx_light.onnx_py._onnxpykernels.runtime": runtime,
        }
        return model, telemetry, modules

    def test_runtime_session_is_built_once_and_reused(self):
        out_value = np.array([1.5, 2.5], dtype=np.float32)
        model, telemetry, modules = self._install_fake_onnx_light(out_value)

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_reference_runner(model)
            first_input = np.array([1.0, 2.0], dtype=np.float32)
            first = runner([first_input])
            second = runner([np.array([3.0, 4.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        np.testing.assert_allclose(first[0], out_value)
        np.testing.assert_allclose(second[0], out_value)
        # ExecutionPlan and RuntimeSession are built exactly once, then reused
        # across both runs.
        self.assertEqual(telemetry["plans"], 1)
        self.assertEqual(telemetry["sessions"], 1)
        self.assertEqual(telemetry["runs"], 2)
        self.assertIs(telemetry["feeds"][0]["x"], first_input)
        self.assertIsNone(telemetry["cpu_execution"])

    def test_make_onnx_light_runner_uses_runtime_session(self):
        """The onnx_light backend runs through the ``RuntimeSession`` path, the
        same way as the other backends (no fallback runner)."""
        session_calls = []
        saved_session = rlb._make_onnx_light_reference_runner

        def _session(model):
            def _run(inputs):
                session_calls.append(inputs)
                return ["session"]

            return _run

        try:
            rlb._make_onnx_light_reference_runner = _session
            runner = rlb._make_onnx_light_runner(object())
            self.assertEqual(runner(["in"]), ["session"])
            self.assertEqual(session_calls, [["in"]])
        finally:
            rlb._make_onnx_light_reference_runner = saved_session

    def test_make_onnx_light_runner_propagates_run_time_error(self):
        """A ``RuntimeSession`` that raises while a kernel runs surfaces the
        error instead of falling back to another runner."""
        saved_session = rlb._make_onnx_light_reference_runner

        def _session(model):
            def _run(inputs):
                raise RuntimeError("kernel cannot run")

            return _run

        try:
            rlb._make_onnx_light_reference_runner = _session
            runner = rlb._make_onnx_light_runner(object())
            with self.assertRaises(RuntimeError):
                runner(["a"])
        finally:
            rlb._make_onnx_light_reference_runner = saved_session

    def test_make_onnx_light_runner_propagates_load_error(self):
        """A failure while building the ``RuntimeSession`` propagates (no
        fallback runner is built)."""
        saved_session = rlb._make_onnx_light_reference_runner

        def _raise(model):
            raise ImportError("no runtime bindings")

        try:
            rlb._make_onnx_light_reference_runner = _raise
            with self.assertRaises(ImportError):
                rlb._make_onnx_light_runner(object())
        finally:
            rlb._make_onnx_light_reference_runner = saved_session

    def test_make_onnx_light_cpu_runner_registers_kernels_on_session(self):
        """The cpu runner installs the SIMD kernels **on the session** (not
        globally): it passes ``onnx_light_cpu.register_kernels`` to the session
        runner's ``register`` hook, which calls it with the freshly created
        ``RuntimeSession``. The model then runs through the same
        ``RuntimeSession`` execution path as the plain onnx-light backend.
        """
        import types

        saved_session = rlb._make_onnx_light_reference_runner
        calls = {"register_fn": None, "model": None}

        def _fake(model, register=None):
            calls["model"] = model
            calls["register_fn"] = register
            return "runner"

        registered = {"session": None}

        def _register_kernels():
            registered["calls"] = registered.get("calls", 0) + 1

        cpu_module = types.ModuleType("onnx_light_cpu")
        cpu_module.register_kernels = _register_kernels

        saved_cpu = sys.modules.get("onnx_light_cpu")
        model = object()
        try:
            rlb._make_onnx_light_reference_runner = _fake
            sys.modules["onnx_light_cpu"] = cpu_module
            result = rlb._make_onnx_light_cpu_runner(model)
        finally:
            rlb._make_onnx_light_reference_runner = saved_session
            if saved_cpu is None:
                sys.modules.pop("onnx_light_cpu", None)
            else:
                sys.modules["onnx_light_cpu"] = saved_cpu

        self.assertEqual(result, "runner")
        # Registration is wired into evaluator construction and runs globally
        # before the evaluator is created.
        self.assertIs(calls["model"], model)
        self.assertTrue(callable(calls["register_fn"]))
        calls["register_fn"]()
        self.assertEqual(registered["calls"], 1)

    def test_make_onnx_light_cpu_runner_raises_when_package_missing(self):
        """An unavailable onnx-light-cpu surfaces as an ImportError (no fallback)."""
        saved = sys.modules.get("onnx_light_cpu")
        try:
            # Ensure the package is not importable so the import raises.
            sys.modules["onnx_light_cpu"] = None
            with self.assertRaises((ImportError, AttributeError)):
                rlb._make_onnx_light_cpu_runner(object())
        finally:
            if saved is None:
                sys.modules.pop("onnx_light_cpu", None)
            else:
                sys.modules["onnx_light_cpu"] = saved


class TestOnnxLightCpuRunner(unittest.TestCase):
    """The onnx-light-cpu backend evaluates the model via the same
    ``RuntimeSession`` execution path as the plain onnx-light backend, after
    ``onnx_light_cpu.register_kernels(session)`` installs the SIMD-accelerated
    kernels **on that session** (not process-wide)."""

    def _install_fakes(self, session_used_kernels=None):
        import types

        events = {
            "sessions": 0,
            "runs": 0,
            "registered": 0,
            "tensor_copies": [],
            "usage_recording": [],
        }

        class _FakeModelProto:
            def ParseFromString(self, data):
                pass

            def SerializeToString(self):
                return b"model"

        class _Named:
            def __init__(self, name):
                self.name = name
                self.type = None

        class _Opset:
            def __init__(self, domain, version):
                self.domain = domain
                self.version = version

        class _Graph:
            def __init__(self):
                self.input = [_Named("x")]
                self.output = [_Named("y")]
                self.initializer = []

        model = _FakeModelProto()
        model.opset_import = [_Opset("", 18)]
        model.graph = _Graph()

        class _RuntimeContext:
            def __init__(self, kctx):
                self._store = {}

            def set(self, name, tensor, kind=None):
                self._store[name] = tensor

            def has(self, name):
                return name in self._store

            def get(self, name):
                return self._store[name]

            def put(self, name, tensor, kind=None):
                self._store[name] = tensor

        class _RuntimeSession:
            def __init__(self, plan):
                events["sessions"] += 1

            def run(self, ctx):
                events["runs"] += 1
                # Mirror the real engine feeding the graph output as a tensor
                # carrying data_type/shape used by _runtime_tensor_to_numpy.
                ctx.set("y", _OutTensor(np.abs(ctx.get("x"))))

        # onnx-light#4391 exposes ``RuntimeSession.used_kernels()``; only attach
        # it when the test asks for it so the "old build" path is exercised too.
        if session_used_kernels is not None:
            _RuntimeSession.used_kernels = lambda self: list(session_used_kernels)

        class _ExecutionPlan:
            def __init__(self, graph):
                self.graph = graph

        runtime = types.ModuleType("onnx_light.onnx_py._onnxpykernels.runtime")
        runtime.RuntimeContext = _RuntimeContext
        runtime.KernelContext = lambda opset: ("kctx", opset)
        runtime.default_opset = lambda v: v
        runtime.ExecutionPlan = _ExecutionPlan
        runtime.RuntimeSession = _RuntimeSession
        runtime.tensor_from_proto = lambda tp: tp["arr"] if isinstance(tp, dict) else tp
        runtime.tensor_from_numpy = (
            lambda name, dtype, shape, raw, copy=True: (
                events["tensor_copies"].append(copy)
                or raw.view(np.float32).reshape(shape)
            )
        )
        runtime.register_model_functions = lambda m, ctx: None
        runtime.tensor_to_numpy = lambda t: np.asarray(t, dtype=np.float32).view(np.uint8)
        runtime.tensor_to_proto = lambda t: t

        import onnx as _onnx

        class _OutTensor:
            def __init__(self, arr):
                self._arr = np.asarray(arr, dtype=np.float32)
                self.data_type = int(_onnx.TensorProto.FLOAT)
                self.shape = self._arr.shape

        runtime.tensor_to_numpy = lambda t: t._arr.view(np.uint8)

        numpy_helper = types.ModuleType("onnx_light.onnx_lib.numpy_helper")
        numpy_helper.from_array = lambda arr, name=None: {"name": name, "arr": arr}
        numpy_helper.to_array = lambda tp: np.asarray(tp, dtype=np.float32)

        onnx_lib = types.ModuleType("onnx_light.onnx_lib")
        onnx_lib.ModelProto = _FakeModelProto
        onnx_lib.numpy_helper = numpy_helper

        onnx_light = types.ModuleType("onnx_light")
        onnx_pkg = types.ModuleType("onnx_light.onnx")
        reference = types.ModuleType("onnx_light.onnx.reference")

        class _ReferenceEvaluator:
            def __init__(self, model_bytes, cpu_execution=None):
                self.input_names = ["x"]

            def run(self, output_names, feeds):
                events["runs"] += 1
                events.setdefault("feeds", []).append(feeds)
                return [np.abs(feeds["x"])]

            def used_kernels(self):
                return list(session_used_kernels or [])

        reference.ReferenceEvaluator = _ReferenceEvaluator
        onnx_py = types.ModuleType("onnx_light.onnx_py")
        pyk = types.ModuleType("onnx_light.onnx_py._onnxpykernels")
        pyk.runtime = runtime
        onnx_py._onnxpykernels = pyk

        # Fake onnx-light-cpu exposing only the global registration helper, as
        # the real package does. The benchmark passes the session to it so the
        # kernels are registered on that session.
        def _register_kernels(sess=None):
            events["registered"] += 1
            return sess

        cpu = types.ModuleType("onnx_light_cpu")
        cpu.register_kernels = _register_kernels
        cpu_py = types.ModuleType("onnx_light_cpu.onnx_py")
        cpu_register = types.ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        cpu_register.set_kernel_usage_recording = (
            lambda enabled: events["usage_recording"].append(enabled)
        )

        modules = {
            "onnx_light": onnx_light,
            "onnx_light.onnx": onnx_pkg,
            "onnx_light.onnx.reference": reference,
            "onnx_light.onnx_lib": onnx_lib,
            "onnx_light.onnx_lib.numpy_helper": numpy_helper,
            "onnx_light.onnx_py": onnx_py,
            "onnx_light.onnx_py._onnxpykernels": pyk,
            "onnx_light.onnx_py._onnxpykernels.runtime": runtime,
            "onnx_light_cpu": cpu,
            "onnx_light_cpu.onnx_py": cpu_py,
            "onnx_light_cpu.onnx_py._cpuregister": cpu_register,
        }
        return model, modules, events

    def test_cpu_runner_registers_kernels_and_runs(self):
        model, modules, events = self._install_fakes()
        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            out = runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        # The SIMD kernels were installed on the session exactly once, and the
        # model ran through a single RuntimeSession (the same execution path as
        # the plain onnx-light backend).
        self.assertEqual(events["registered"], 1)
        self.assertEqual(events["runs"], 1)
        self.assertIsInstance(events["feeds"][0]["x"], np.ndarray)
        np.testing.assert_allclose(out[0], np.array([1.0, 2.0], dtype=np.float32))

    def test_cpu_runner_checks_used_kernel_names(self):
        """When the build exposes the kernel-name helpers, the runner clears the
        used-kernel record before the first run and accepts the run when at
        least one onnx-light-cpu kernel actually ran."""
        model, modules, events = self._install_fakes()
        used = {"cleared": 0, "names": ["onnx_light_cpu::Abs"]}
        cpu = modules["onnx_light_cpu"]

        def _clear():
            used["cleared"] += 1

        cpu.clear_used_kernel_names = _clear
        cpu.used_kernel_names = lambda: list(used["names"])
        cpu.registered_kernel_names = lambda: {"Abs": "onnx_light_cpu::Abs"}

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            out = runner([np.array([-1.0, 2.0], dtype=np.float32)])
            # A second call does not re-clear or re-check.
            runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(used["cleared"], 1)
        self.assertEqual(events["runs"], 2)
        self.assertEqual(events["usage_recording"], [True, False])
        np.testing.assert_allclose(out[0], np.array([1.0, 2.0], dtype=np.float32))

    def test_cpu_runner_registers_globally_before_evaluator(self):
        """The public API registers CPU kernels globally before construction."""
        model, modules, events = self._install_fakes()
        seen = {"args": None}

        def _register_kernels(*args):
            events["registered"] += 1
            seen["args"] = args

        modules["onnx_light_cpu"].register_kernels = _register_kernels

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            rlb._make_onnx_light_cpu_runner(model)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(events["registered"], 1)
        self.assertEqual(seen["args"], ())

    def test_cpu_runner_raises_when_non_cpu_kernel_used(self):
        """Explicitly checks the used kernel names come from onnx-light-cpu: if a
        name that is not an onnx-light-cpu kernel (e.g. a built-in kernel) is
        recorded, the runner raises so a contaminated measurement is not
        reported as an onnx-light-cpu result."""
        model, modules, events = self._install_fakes()
        cpu = modules["onnx_light_cpu"]
        cpu.clear_used_kernel_names = lambda: None
        # A built-in onnx-light kernel ran, not the onnx-light-cpu one.
        cpu.used_kernel_names = lambda: ["onnx_light::Abs"]
        cpu.registered_kernel_names = lambda: {"Abs": "onnx_light_cpu::Abs"}

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            with self.assertRaises(RuntimeError) as ctx:
                runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertIn("not all from onnx-light-cpu", str(ctx.exception))
        self.assertIn("onnx_light::Abs", str(ctx.exception))

    def test_cpu_runner_raises_when_no_cpu_kernel_used(self):
        """If no onnx-light-cpu kernel ran (the model uses none of the
        overridden operators), the runner raises so the backend records an
        error instead of reporting built-in-kernel timings as cpu results."""
        model, modules, events = self._install_fakes()
        cpu = modules["onnx_light_cpu"]
        cpu.clear_used_kernel_names = lambda: None
        cpu.used_kernel_names = lambda: []
        cpu.registered_kernel_names = lambda: {"Abs": "onnx_light_cpu::Abs"}

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            with self.assertRaises(RuntimeError) as ctx:
                runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertIn("no onnx-light-cpu kernel ran", str(ctx.exception))
        self.assertIn("Abs", str(ctx.exception))

    def test_cpu_runner_accepts_when_session_kernels_all_cpu(self):
        """When the build exposes ``RuntimeSession.used_kernels()``
        (onnx-light#4391), the runner cross-checks the operators the session
        executed against the onnx-light-cpu kernels that ran: if every
        overridable operator was served by onnx-light-cpu, the run is accepted."""
        model, modules, events = self._install_fakes(
            session_used_kernels=["ai.onnx:Abs"]
        )
        cpu = modules["onnx_light_cpu"]
        cpu.clear_used_kernel_names = lambda: None
        cpu.used_kernel_names = lambda: ["onnx_light_cpu::Abs"]
        cpu.registered_kernel_names = lambda: {"Abs": "onnx_light_cpu::Abs"}

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            out = runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        np.testing.assert_allclose(out[0], np.array([1.0, 2.0], dtype=np.float32))

    def test_cpu_runner_raises_when_overridable_op_falls_back(self):
        """The session executed two operators onnx-light-cpu overrides but only
        one onnx-light-cpu kernel ran: the other fell back to a built-in kernel,
        so onnx-light-cpu was *not* really used and the runner raises."""
        model, modules, events = self._install_fakes(
            session_used_kernels=["ai.onnx:Abs", "ai.onnx:Exp"]
        )
        cpu = modules["onnx_light_cpu"]
        cpu.clear_used_kernel_names = lambda: None
        # Only Abs ran through onnx-light-cpu; Exp silently used the built-in.
        cpu.used_kernel_names = lambda: ["onnx_light_cpu::Abs"]
        cpu.registered_kernel_names = lambda: {
            "Abs": "onnx_light_cpu::Abs",
            "Exp": "onnx_light_cpu::Exp",
        }

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_cpu_runner(model)
            with self.assertRaises(RuntimeError) as ctx:
                runner([np.array([-1.0, 2.0], dtype=np.float32)])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertIn("built-in kernels", str(ctx.exception))
        self.assertIn("Exp", str(ctx.exception))


class _FakeTypeProto:
    """Minimal ``TypeProto`` stand-in exposing a protobuf-like ``HasField``."""

    def __init__(self, kind):
        self._kind = kind  # "tensor", "sequence" or "map"

    def HasField(self, field):
        if field == "sequence_type":
            return self._kind == "sequence"
        if field == "map_type":
            return self._kind == "map"
        return False


class TestTypeProtoKindDetection(unittest.TestCase):
    def test_tensor_type_is_tensor(self):
        self.assertEqual(rlb._type_proto_kind(_FakeTypeProto("tensor")), "tensor")

    def test_sequence_type_is_sequence(self):
        self.assertEqual(rlb._type_proto_kind(_FakeTypeProto("sequence")), "sequence")

    def test_map_type_is_map(self):
        self.assertEqual(rlb._type_proto_kind(_FakeTypeProto("map")), "map")

    def test_none_type_is_tensor(self):
        self.assertEqual(rlb._type_proto_kind(None), "tensor")

    def test_has_method_fallback_detects_sequence(self):
        class _Type:
            @staticmethod
            def has_sequence_type():
                return True

            @staticmethod
            def has_map_type():
                return False

        self.assertEqual(rlb._type_proto_kind(_Type()), "sequence")

    def test_has_method_fallback_detects_map(self):
        class _Type:
            @staticmethod
            def has_sequence_type():
                return False

            @staticmethod
            def has_map_type():
                return True

        self.assertEqual(rlb._type_proto_kind(_Type()), "map")


class TestReferenceEvaluatorSequenceIO(unittest.TestCase):
    def _run_with_kinds(self, out_value, input_kind, output_kind, inputs):
        helper = TestOnnxLightReferenceRunner()
        model, telemetry, modules = helper._install_fake_onnx_light(
            out_value, input_kind=input_kind, output_kind=output_kind
        )

        saved = {name: sys.modules.get(name) for name in modules}
        try:
            sys.modules.update(modules)
            runner = rlb._make_onnx_light_reference_runner(model)
            result = runner(inputs)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod
        return result, telemetry

    def test_sequence_output_is_returned_as_list(self):
        out_value = np.array([1.5, 2.5], dtype=np.float32)
        result, telemetry = self._run_with_kinds(
            out_value, "tensor", "sequence", [np.array([1.0, 2.0], dtype=np.float32)]
        )
        # The single graph output is a sequence, so it is materialised as a
        # Python list of numpy arrays rather than rejected.
        self.assertIsInstance(result[0], list)
        np.testing.assert_allclose(result[0][0], out_value)
        self.assertEqual(telemetry["plans"], 1)
        self.assertEqual(telemetry["sessions"], 1)

    def test_sequence_input_is_fed_through_put_sequence(self):
        out_value = np.array([1.5, 2.5], dtype=np.float32)
        seq_input = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32),
        ]
        result, telemetry = self._run_with_kinds(
            out_value, "sequence", "sequence", [seq_input]
        )
        self.assertIsInstance(result[0], list)
        np.testing.assert_allclose(result[0][0], out_value)

    def test_map_output_is_returned(self):
        out_value = np.array([1.5, 2.5], dtype=np.float32)
        result, _ = self._run_with_kinds(
            out_value, "tensor", "map", [np.array([1.0, 2.0], dtype=np.float32)]
        )
        self.assertEqual(result[0], {1: 2})


class TestBenchmarkBackends(unittest.TestCase):
    def test_benchmark_backends_contains_ort_and_onnx_light(self):
        self.assertIn("onnxruntime", rlb.BENCHMARK_BACKENDS)
        self.assertIn("onnx_light", rlb.BENCHMARK_BACKENDS)
        self.assertIn("onnx_light_cpu", rlb.BENCHMARK_BACKENDS)
        self.assertNotIn("reference", rlb.BENCHMARK_BACKENDS)

    def test_runner_factories_match_benchmark_backends(self):
        for b in rlb.BENCHMARK_BACKENDS:
            self.assertIn(b, rlb._RUNNER_FACTORIES, msg=f"no factory for {b}")


class TestNormalizeKinds(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(rlb._normalize_kinds(None), ())

    def test_empty_string_returns_empty(self):
        self.assertEqual(rlb._normalize_kinds(""), ())

    def test_single_kind(self):
        self.assertEqual(rlb._normalize_kinds("node"), ("node",))

    def test_comma_separated(self):
        self.assertEqual(rlb._normalize_kinds("node,model"), ("node", "model"))

    def test_deduplicates(self):
        self.assertEqual(rlb._normalize_kinds("node,node"), ("node",))

    def test_strips_whitespace(self):
        self.assertEqual(rlb._normalize_kinds("node , model"), ("node", "model"))


class TestDiscoverNodeTests(unittest.TestCase):
    """Verify discover_node_tests calls collect_test_case with include_big=True."""

    def test_benchmark_mode_is_preferred_when_available(self):
        import types

        expected_inputs = np.array([1.0, 2.0], dtype=np.float32)

        class _FakeInputType:
            @staticmethod
            def has_map_type():
                return False

        class _FakeInput:
            def __init__(self, name):
                self.name = name
                self.type = _FakeInputType()

        class _FakeGraph:
            def __init__(self):
                self.input = [_FakeInput("x")]

        class _FakeModel:
            def __init__(self):
                self.graph = _FakeGraph()

        class _FakeTensor:
            def __init__(self, name, array):
                self.name = name
                self.data_type = 1
                self.shape = array.shape
                self._raw = array.astype(np.float32).tobytes()

            def raw_data(self):
                return self._raw

        class _FakeDataSet:
            def __init__(self, array):
                self.inputs = [_FakeTensor("x", array)]
                self.maps = []

        class _FakeCase:
            def __init__(self):
                self.name = "test_cc_benchmark"
                self.kind = "node"
                self.model = _FakeModel()
                self.data_sets = [_FakeDataSet(expected_inputs)]
                self.tag = "bench"

        call_kwargs = {}

        class _FakeTestMode:
            BENCHMARK = "BENCHMARK"

        def _fake_collect_test_cases(**kwargs):
            call_kwargs.update(kwargs)
            return [_FakeCase()]

        fake_backend = types.ModuleType("onnx_light.onnx.backend")
        fake_backend.TestMode = _FakeTestMode
        fake_backend.collect_test_cases = _fake_collect_test_cases

        saved_light = sys.modules.get("onnx_light")
        saved_onnx_pkg = sys.modules.get("onnx_light.onnx")
        saved_backend = sys.modules.get("onnx_light.onnx.backend")
        orig_to_onnx = rlb._onnx_light_model_to_onnx
        orig_cc_tensor_to_numpy = rlb._cc_tensor_to_numpy
        rlb._onnx_light_model_to_onnx = lambda m: m
        rlb._cc_tensor_to_numpy = (
            lambda t: np.frombuffer(t.raw_data(), dtype=np.float32).reshape(t.shape)
        )
        try:
            sys.modules["onnx_light"] = types.ModuleType("onnx_light")
            sys.modules["onnx_light.onnx"] = types.ModuleType("onnx_light.onnx")
            sys.modules["onnx_light.onnx.backend"] = fake_backend
            discovered = rlb.discover_node_tests("node")
        finally:
            if saved_light is None:
                sys.modules.pop("onnx_light", None)
            else:
                sys.modules["onnx_light"] = saved_light
            if saved_onnx_pkg is None:
                sys.modules.pop("onnx_light.onnx", None)
            else:
                sys.modules["onnx_light.onnx"] = saved_onnx_pkg
            if saved_backend is None:
                sys.modules.pop("onnx_light.onnx.backend", None)
            else:
                sys.modules["onnx_light.onnx.backend"] = saved_backend
            rlb._onnx_light_model_to_onnx = orig_to_onnx
            rlb._cc_tensor_to_numpy = orig_cc_tensor_to_numpy

        self.assertEqual(call_kwargs["include_big"], True)
        self.assertEqual(call_kwargs["mode"], _FakeTestMode.BENCHMARK)
        self.assertEqual([d["name"] for d in discovered], ["test_cc_benchmark"])
        self.assertEqual(discovered[0]["tag"], "bench")
        data_sets = discovered[0]["data_sets"]
        self.assertEqual(len(data_sets), 1)
        first_inputs, _ = data_sets[0]
        self.assertEqual(len(first_inputs), 1)
        np.testing.assert_allclose(first_inputs[0], expected_inputs)

    def test_benchmark_mode_filters_out_non_benchmark_cases(self):
        import types

        expected_inputs = np.array([1.0, 2.0], dtype=np.float32)

        class _FakeInputType:
            @staticmethod
            def has_map_type():
                return False

        class _FakeInput:
            def __init__(self, name):
                self.name = name
                self.type = _FakeInputType()

        class _FakeGraph:
            def __init__(self):
                self.input = [_FakeInput("x")]

        class _FakeModel:
            def __init__(self):
                self.graph = _FakeGraph()

        class _FakeTensor:
            def __init__(self, name, array):
                self.name = name
                self.data_type = 1
                self.shape = array.shape
                self._raw = array.astype(np.float32).tobytes()

            def raw_data(self):
                return self._raw

        class _FakeDataSet:
            def __init__(self, array):
                self.inputs = [_FakeTensor("x", array)]
                self.maps = []

        class _FakeCase:
            def __init__(self, name):
                self.name = name
                self.kind = "node"
                self.model = _FakeModel()
                self.data_sets = [_FakeDataSet(expected_inputs)]
                self.tag = ""

        class _FakeTestMode:
            BENCHMARK = "BENCHMARK"

        def _fake_collect_test_cases(**kwargs):
            # ``TestMode.BENCHMARK`` returns a genuine benchmark model plus a
            # leftover correctness case whose name lacks the benchmark suffix.
            return [
                _FakeCase("test_cc_abs_benchmark"),
                _FakeCase("test_cc_add_nan_inf"),
            ]

        fake_backend = types.ModuleType("onnx_light.onnx.backend")
        fake_backend.TestMode = _FakeTestMode
        fake_backend.collect_test_cases = _fake_collect_test_cases

        saved_light = sys.modules.get("onnx_light")
        saved_onnx_pkg = sys.modules.get("onnx_light.onnx")
        saved_backend = sys.modules.get("onnx_light.onnx.backend")
        orig_to_onnx = rlb._onnx_light_model_to_onnx
        orig_cc_tensor_to_numpy = rlb._cc_tensor_to_numpy
        rlb._onnx_light_model_to_onnx = lambda m: m
        rlb._cc_tensor_to_numpy = (
            lambda t: np.frombuffer(t.raw_data(), dtype=np.float32).reshape(t.shape)
        )
        try:
            sys.modules["onnx_light"] = types.ModuleType("onnx_light")
            sys.modules["onnx_light.onnx"] = types.ModuleType("onnx_light.onnx")
            sys.modules["onnx_light.onnx.backend"] = fake_backend
            discovered = rlb.discover_node_tests("node")
        finally:
            if saved_light is None:
                sys.modules.pop("onnx_light", None)
            else:
                sys.modules["onnx_light"] = saved_light
            if saved_onnx_pkg is None:
                sys.modules.pop("onnx_light.onnx", None)
            else:
                sys.modules["onnx_light.onnx"] = saved_onnx_pkg
            if saved_backend is None:
                sys.modules.pop("onnx_light.onnx.backend", None)
            else:
                sys.modules["onnx_light.onnx.backend"] = saved_backend
            rlb._onnx_light_model_to_onnx = orig_to_onnx
            rlb._cc_tensor_to_numpy = orig_cc_tensor_to_numpy

        self.assertEqual([d["name"] for d in discovered], ["test_cc_abs_benchmark"])

    def test_include_big_true_is_passed(self):
        import types

        call_kwargs = {}

        def _fake_collect(**kwargs):
            call_kwargs.update(kwargs)
            return {}

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = _fake_collect
        saved = sys.modules.get("onnx_light.onnx_lib.backend.test.case")
        try:
            sys.modules["onnx_light.onnx_lib.backend.test.case"] = fake_module
            rlb.discover_node_tests("node")
        finally:
            if saved is None:
                sys.modules.pop("onnx_light.onnx_lib.backend.test.case", None)
            else:
                sys.modules["onnx_light.onnx_lib.backend.test.case"] = saved

        self.assertIn("include_big", call_kwargs, "include_big keyword not passed")
        self.assertTrue(call_kwargs["include_big"], "include_big must be True")

    def test_kind_filter_applied(self):
        import types

        class _FakeTC:
            kind = "node"
            model = object()
            data_sets = [([], [])]
            model_dir = None
            tag = ""

        class _FakeTCOther:
            kind = "simple"
            model = object()
            data_sets = [([], [])]
            model_dir = None
            tag = ""

        cases = {"test_a": _FakeTC(), "test_b": _FakeTCOther()}

        def _fake_collect(include_big=False):
            return cases

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = _fake_collect
        saved = sys.modules.get("onnx_light.onnx_lib.backend.test.case")

        # patch _onnx_light_model_to_onnx and _onnx_light_tensor_to_numpy
        orig_to_onnx = rlb._onnx_light_model_to_onnx
        orig_to_np = rlb._onnx_light_tensor_to_numpy
        rlb._onnx_light_model_to_onnx = lambda m: m
        rlb._onnx_light_tensor_to_numpy = lambda a: a
        try:
            sys.modules["onnx_light.onnx_lib.backend.test.case"] = fake_module
            discovered = rlb.discover_node_tests("node")
        finally:
            if saved is None:
                sys.modules.pop("onnx_light.onnx_lib.backend.test.case", None)
            else:
                sys.modules["onnx_light.onnx_lib.backend.test.case"] = saved
            rlb._onnx_light_model_to_onnx = orig_to_onnx
            rlb._onnx_light_tensor_to_numpy = orig_to_np

        names = [d["name"] for d in discovered]
        self.assertIn("test_a", names)
        self.assertNotIn("test_b", names)


if __name__ == "__main__":
    unittest.main()
