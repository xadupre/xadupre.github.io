"""Tests for ``scripts.record_onnx_backend_test_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_backend_test_coverage as rbc  # noqa: E402

try:  # ``onnx_light`` is only installed in the coverage workflows, not in CI.
    from onnx_light.tools import to_svg as _to_svg  # noqa: F401

    _HAS_ONNX_LIGHT = True
except Exception:  # noqa: BLE001 - any import failure disables the SVG tests
    _HAS_ONNX_LIGHT = False


class TestRecordOnnxBackendTestCoverage(unittest.TestCase):
    def test_stringify_error_truncates_and_takes_first_line(self):
        self.assertEqual(rbc._stringify_error(None), "")
        self.assertEqual(rbc._stringify_error("boom"), "boom")
        self.assertEqual(rbc._stringify_error("boom\nrest"), "boom")
        long = "x" * 500
        out = rbc._stringify_error(long)
        # Over-long single lines are truncated in the middle so the head and
        # tail both survive; the result stays bounded at the max length.
        self.assertEqual(len(out), 300)
        self.assertIn(" ... ", out)
        self.assertTrue(out.startswith("x"))
        self.assertTrue(out.endswith("x"))

    def test_stringify_error_keeps_onnxruntime_cause_in_tail(self):
        # ``onnxruntime`` reports the human-readable cause at the very end of a
        # long single line, behind a file path and a verbose C++ signature. The
        # informative tail must survive truncation so the dashboard explains
        # *why* a test (e.g. ``test_attention_4d_diff_heads_mask4d_padded_kv``)
        # fails instead of only showing the function signature.
        ort_error = (
            "[ONNXRuntimeError] : 1 : FAIL : Non-zero status code returned "
            "while running Attention node. Name:'' Status Message: "
            "/onnxruntime_src/onnxruntime/core/providers/cpu/llm/"
            "attention_helper.h:146 onnxruntime::common::Status "
            "onnxruntime::attention_helper::ComputeOutputShapeForAttention("
            "const onnxruntime::Tensor*, const onnxruntime::Tensor*) "
            "attn_mask->Shape()[attn_mask->Shape().NumDimensions() - 1] == "
            "parameters.total_sequence_length was false. inconsistent "
            "total_sequence_length (between attn_mask and past_key and "
            "past_value)"
        )
        out = rbc._stringify_error(ort_error)
        self.assertLessEqual(len(out), 300)
        self.assertIn("Attention node", out)
        self.assertIn("inconsistent total_sequence_length", out)

    def test_row_from_results_includes_errors_only_when_present(self):
        row = rbc._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {
                    "success": False,
                    "error": "boom",
                    "error_step": "run",
                },
                "onnx_light": {"success": True, "error": "", "error_step": ""},
            },
            versions={
                "onnxruntime": "1.20.0",
                "onnx": "1.17.0",
                "onnx_light": "0.1.0",
            },
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertEqual(row["name"], "test_relu")
        self.assertTrue(row["onnxruntime"])
        self.assertFalse(row["reference"])
        self.assertTrue(row["onnx_light"])
        self.assertNotIn("onnxruntime_error", row)
        self.assertNotIn("onnxruntime_error_step", row)
        self.assertEqual(row["reference_error"], "boom")
        self.assertEqual(row["reference_error_step"], "run")
        self.assertNotIn("onnx_light_error", row)
        # Passing backend records its last-pass date + matching package version.
        self.assertEqual(row["onnxruntime_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["onnxruntime_last_pass_version"], "1.20.0")
        self.assertEqual(row["onnx_light_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["onnx_light_last_pass_version"], "0.1.0")
        # Failing backend has no recorded last-pass when there is no history.
        self.assertNotIn("reference_last_pass_date", row)
        self.assertNotIn("reference_last_pass_version", row)

    def test_row_from_results_carries_over_previous_last_pass_on_failure(self):
        previous = {
            "name": "test_relu",
            "reference_last_pass_date": "2024-01-02T03:04:05Z",
            "reference_last_pass_version": "1.16.0",
        }
        row = rbc._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {
                    "success": False,
                    "error": "boom",
                    "error_step": "run",
                },
                "onnx_light": {
                    "success": False,
                    "error": "boom",
                    "error_step": "run",
                },
            },
            previous=previous,
            versions={
                "onnxruntime": "1.20.0",
                "onnx": "1.17.0",
                "onnx_light": "0.1.0",
            },
            now_iso="2024-05-06T07:08:09Z",
        )
        # Current pass refreshes the onnxruntime entry, prior reference pass is kept.
        self.assertEqual(row["onnxruntime_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["onnxruntime_last_pass_version"], "1.20.0")
        self.assertEqual(row["reference_last_pass_date"], "2024-01-02T03:04:05Z")
        self.assertEqual(row["reference_last_pass_version"], "1.16.0")

    @unittest.skipUnless(_HAS_ONNX_LIGHT, "onnx_light is required for to_svg")
    def test_build_graph_renders_svg_with_onnx_light(self):
        import onnx
        from onnx import helper

        x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4])
        w = helper.make_tensor_value_info("w", onnx.TensorProto.FLOAT, [4, 2])
        y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3, 2])
        matmul = helper.make_node("MatMul", ["x", "w"], ["m"])
        relu = helper.make_node("Relu", ["m"], ["y"], name="act")
        graph = helper.make_graph([matmul, relu], "g", [x, w], [y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        result = rbc.build_graph(model)
        # ``build_graph`` now delegates to ``onnx_light.tools.to_svg`` and
        # stores the resulting standalone SVG document under ``"svg"``.
        self.assertIn("svg", result)
        svg = result["svg"]
        self.assertIsInstance(svg, str)
        self.assertTrue(svg.lstrip().startswith("<svg"))
        self.assertIn("</svg>", svg)
        # The operator names appear in the rendered SVG text.
        self.assertIn("MatMul", svg)
        self.assertIn("Relu", svg)

    @unittest.skipUnless(_HAS_ONNX_LIGHT, "onnx_light is required for to_svg")
    def test_build_graph_renders_initializers_in_svg(self):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])
        y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])
        const = numpy_helper.from_array(np.ones((2,), dtype=np.float32), name="b")
        add = helper.make_node("Add", ["x", "b"], ["y"])
        graph = helper.make_graph(
            [add],
            "g",
            [x, helper.make_tensor_value_info("b", onnx.TensorProto.FLOAT, [2])],
            [y],
            initializer=[const],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        result = rbc.build_graph(model)
        svg = result["svg"]
        self.assertTrue(svg.lstrip().startswith("<svg"))
        self.assertIn("Add", svg)
        self.assertIn("b", svg)

    def test_row_from_results_includes_and_carries_over_graph(self):
        graph = {
            "inputs": [{"name": "x", "type": "float[2]"}],
            "outputs": [{"name": "y", "type": "float[2]"}],
            "nodes": [{"op_type": "Relu", "inputs": ["x"], "outputs": ["y"]}],
        }
        results = {
            "onnxruntime": {"success": True, "error": "", "error_step": ""},
            "reference": {"success": True, "error": "", "error_step": ""},
            "onnx_light": {"success": True, "error": "", "error_step": ""},
        }
        row = rbc._row_from_results("test_relu", results, graph=graph)
        self.assertEqual(row["graph"], graph)

        # When the current run cannot build a graph, the previous one is kept.
        carried = rbc._row_from_results(
            "test_relu", results, previous={"graph": graph}, graph=None
        )
        self.assertEqual(carried["graph"], graph)

        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": [("in_a", "out_a")]},
            {"name": "test_b", "model": "model_b", "data_sets": [("in_b", "out_b")]},
            {"name": "test_c", "model": "model_c", "data_sets": [("in_c", "out_c")]},
        ]
        # Map of (model, backend) -> result dict
        ok = {"success": True, "error": "", "error_step": ""}
        outcomes = {
            ("model_a", "onnxruntime"): ok,
            ("model_a", "reference"): ok,
            ("model_a", "onnx_light"): ok,
            ("model_a", "onnx_light_cpu"): ok,
            ("model_b", "onnxruntime"): ok,
            ("model_b", "reference"): {
                "success": False,
                "error": "not implemented",
                "error_step": "run",
            },
            ("model_b", "onnx_light"): ok,
            ("model_b", "onnx_light_cpu"): ok,
            ("model_c", "onnxruntime"): {
                "success": False,
                "error": "kernel missing",
                "error_step": "load",
            },
            ("model_c", "reference"): ok,
            ("model_c", "onnx_light"): {
                "success": False,
                "error": "kernel missing in onnx-light",
                "error_step": "run",
            },
            ("model_c", "onnx_light_cpu"): {
                "success": False,
                "error": "no onnx-light-cpu kernel ran",
                "error_step": "run",
            },
        }

        def fake_run(model, data_sets, backend, rtol, atol):
            return outcomes[(model, backend)]

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {"onnx": "1.0"},
        )

        self.assertEqual(payload["kind"], "node")
        self.assertEqual(payload["versions"], {"onnx": "1.0"})
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 2, "fail": 1},
                "reference": {"pass": 2, "fail": 1},
                "onnx_light": {"pass": 2, "fail": 1},
                "onnx_light_cpu": {"pass": 2, "fail": 1},
            },
        )
        names = [row["name"] for row in payload["tests"]]
        self.assertEqual(names, ["test_a", "test_b", "test_c"])
        by_name = {row["name"]: row for row in payload["tests"]}
        self.assertTrue(by_name["test_a"]["onnxruntime"])
        self.assertTrue(by_name["test_a"]["reference"])
        self.assertTrue(by_name["test_a"]["onnx_light"])
        self.assertFalse(by_name["test_b"]["reference"])
        self.assertEqual(by_name["test_b"]["reference_error"], "not implemented")
        self.assertEqual(by_name["test_b"]["reference_error_step"], "run")
        self.assertTrue(by_name["test_b"]["onnx_light"])
        self.assertFalse(by_name["test_c"]["onnxruntime"])
        self.assertEqual(by_name["test_c"]["onnxruntime_error_step"], "load")
        self.assertFalse(by_name["test_c"]["onnx_light"])
        self.assertEqual(
            by_name["test_c"]["onnx_light_error"], "kernel missing in onnx-light"
        )

    def test_build_payload_honours_limit(self):
        tests = [
            {"name": f"test_{i}", "model": f"model_{i}", "data_sets": []}
            for i in range(5)
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            return {"success": True, "error": "", "error_step": ""}

        payload = rbc.build_payload(
            kind="node",
            limit=2,
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(len(payload["tests"]), 2)
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 2, "fail": 0},
                "reference": {"pass": 2, "fail": 0},
                "onnx_light": {"pass": 2, "fail": 0},
                "onnx_light_cpu": {"pass": 2, "fail": 0},
            },
        )

    def test_build_payload_defers_cpu_until_after_all_baselines(self):
        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": []},
            {"name": "test_b", "model": "model_b", "data_sets": []},
        ]
        calls = []

        def fake_run(model, data_sets, backend, rtol, atol):
            calls.append((model, backend))
            return {"success": True, "error": "", "error_step": ""}

        rbc.build_payload(
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )

        first_cpu = next(i for i, (_, backend) in enumerate(calls)
                         if backend == "onnx_light_cpu")
        self.assertTrue(
            all(backend != "onnx_light_cpu" for _, backend in calls[:first_cpu])
        )
        self.assertEqual(
            calls[first_cpu:],
            [
                ("model_a", "onnx_light_cpu"),
                ("model_b", "onnx_light_cpu"),
            ],
        )

    def test_build_payload_captures_unhandled_runner_exceptions(self):
        tests = [{"name": "boom", "model": "model_boom", "data_sets": []}]

        def fake_run(model, data_sets, backend, rtol, atol):
            raise RuntimeError("unexpected")

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertFalse(row["onnxruntime"])
        self.assertFalse(row["reference"])
        self.assertFalse(row["onnx_light"])
        self.assertFalse(row["onnx_light_cpu"])
        self.assertEqual(row["onnxruntime_error"], "unexpected")
        self.assertEqual(row["reference_error_step"], "run")
        self.assertEqual(row["onnx_light_error_step"], "run")
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 0, "fail": 1},
                "reference": {"pass": 0, "fail": 1},
                "onnx_light": {"pass": 0, "fail": 1},
                "onnx_light_cpu": {"pass": 0, "fail": 1},
            },
        )

    def test_build_payload_isolates_native_cpu_failures(self):
        tests = [{"name": "crash", "model": "model", "data_sets": []}]
        crashed = {
            "success": False,
            "error": "onnx-light-cpu worker crashed with exit code -11",
            "error_step": "run",
            "elapsed_s": 0.0,
        }

        def fake_run(model, data_sets, backend, rtol, atol):
            self.assertNotEqual(backend, "onnx_light_cpu")
            return {"success": True, "error": "", "error_step": ""}

        with mock.patch.object(
            rbc, "_run_cpu_tests_isolated", return_value=[crashed]
        ) as isolated:
            payload = rbc.build_payload(
                discover=lambda kind: tests,
                run=fake_run,
                versions=lambda: {},
                isolate_cpu=True,
            )

        isolated.assert_called_once_with(tests, rbc.DEFAULT_RTOL, rbc.DEFAULT_ATOL)
        row = payload["tests"][0]
        self.assertFalse(row["onnx_light_cpu"])
        self.assertEqual(row["onnx_light_cpu_error"], crashed["error"])
        self.assertEqual(row["onnx_light_cpu_error_step"], "run")

    def test_build_payload_carries_previous_last_pass_for_failing_tests(self):
        import datetime as dt

        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": []},
            {"name": "test_b", "model": "model_b", "data_sets": []},
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            if model == "model_a":
                return {"success": True, "error": "", "error_step": ""}
            return {"success": False, "error": "boom", "error_step": "run"}

        previous = {
            "tests": [
                {
                    "name": "test_b",
                    "onnxruntime_last_pass_date": "2024-01-01T00:00:00Z",
                    "onnxruntime_last_pass_version": "1.18.0",
                    "reference_last_pass_date": "2024-02-02T00:00:00Z",
                    "reference_last_pass_version": "1.16.0",
                },
            ],
        }

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now=dt.datetime(2024, 5, 6, 7, 8, 9, tzinfo=dt.timezone.utc),
            previous=previous,
        )
        by_name = {row["name"]: row for row in payload["tests"]}
        # Currently-passing test gets a fresh last-pass timestamp.
        self.assertEqual(
            by_name["test_a"]["onnxruntime_last_pass_date"],
            "2024-05-06T07:08:09Z",
        )
        self.assertEqual(by_name["test_a"]["onnxruntime_last_pass_version"], "1.20.0")
        self.assertEqual(by_name["test_a"]["reference_last_pass_version"], "1.17.0")
        # Currently-failing test keeps the previously recorded last-pass info.
        self.assertEqual(
            by_name["test_b"]["onnxruntime_last_pass_date"],
            "2024-01-01T00:00:00Z",
        )
        self.assertEqual(by_name["test_b"]["onnxruntime_last_pass_version"], "1.18.0")
        self.assertEqual(
            by_name["test_b"]["reference_last_pass_date"],
            "2024-02-02T00:00:00Z",
        )

    def test_load_previous_payload_handles_missing_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.json")
            self.assertEqual(rbc.load_previous_payload(missing), {})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("not json")
            self.assertEqual(rbc.load_previous_payload(bad), {})
            ok = os.path.join(tmp, "ok.json")
            with open(ok, "w", encoding="utf-8") as fh:
                json.dump({"tests": [{"name": "x"}]}, fh)
            self.assertEqual(rbc.load_previous_payload(ok), {"tests": [{"name": "x"}]})

    def test_run_test_with_backend_unknown_backend(self):
        result = rbc.run_test_with_backend(None, [], "totally-unknown")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "load")
        self.assertIn("unknown backend", result["error"])

    def test_backends_include_onnx_light_reference_evaluator(self):
        """``onnx_light`` must be one of the recorded backends."""
        self.assertIn("onnx_light", rbc.BACKENDS)
        self.assertIn("onnx_light", rbc._BACKEND_FACTORIES)
        self.assertEqual(rbc.BACKEND_PACKAGE["onnx_light"], "onnx_light")

    def test_backends_include_onnx_light_cpu(self):
        self.assertIn("onnx_light_cpu", rbc.BACKENDS)
        self.assertIn("onnx_light_cpu", rbc._BACKEND_FACTORIES)
        self.assertEqual(
            rbc.BACKEND_PACKAGE["onnx_light_cpu"], "onnx_light_cpu"
        )

    def test_cpu_runner_enables_kernel_usage_recording(self):
        import types

        events = []
        cpu = types.ModuleType("onnx_light_cpu")
        cpu.__path__ = []
        cpu.register_kernels = lambda: events.append("register")
        cpu.clear_used_kernel_names = lambda: events.append("clear")
        cpu.used_kernel_names = lambda: (
            events.append("used") or ["onnx_light_cpu::Abs"]
        )
        cpu_py = types.ModuleType("onnx_light_cpu.onnx_py")
        cpu_py.__path__ = []
        cpu_register = types.ModuleType("onnx_light_cpu.onnx_py._cpuregister")
        cpu_register.set_kernel_usage_recording = (
            lambda enabled: events.append(("recording", enabled))
        )
        modules = {
            "onnx_light_cpu": cpu,
            "onnx_light_cpu.onnx_py": cpu_py,
            "onnx_light_cpu.onnx_py._cpuregister": cpu_register,
        }
        saved = {name: sys.modules.get(name) for name in modules}
        was_registered = rbc._CPU_KERNELS_REGISTERED
        try:
            sys.modules.update(modules)
            rbc._CPU_KERNELS_REGISTERED = False
            with mock.patch.object(
                rbc,
                "_run_with_onnx_light",
                return_value=lambda inputs: events.append("run") or inputs,
            ):
                runner = rbc._run_with_onnx_light_cpu("abs-model")
                self.assertEqual(runner(["input"]), ["input"])
        finally:
            rbc._CPU_KERNELS_REGISTERED = was_registered
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(
            events,
            [
                "register",
                ("recording", True),
                "clear",
                "run",
                "used",
                ("recording", False),
            ],
        )

    def test_run_with_onnx_light_uses_onnx_light_reference_evaluator(self):
        """``_run_with_onnx_light`` builds and drives the onnx-light evaluator.

        ``onnx-light`` is not installed in the unit test environment, so the
        test injects a fake ``onnx_light.onnx.reference`` module exposing a
        ``ReferenceEvaluator`` mock and checks that the factory feeds it the
        serialised model bytes (since ``onnx-light`` ships its own
        ``ModelProto`` type, distinct from ``onnx.ModelProto``).
        """
        import types

        import numpy as np
        import onnx
        from onnx import helper

        node = helper.make_node("Identity", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        constructed: dict = {}

        class _FakeEvaluator:
            def __init__(self, proto):
                constructed["proto"] = proto

            def run(self, output_names, feeds):
                constructed["feeds"] = feeds
                return [feeds["x"] * 2]

        fake_reference = types.ModuleType("onnx_light.onnx.reference")
        fake_reference.ReferenceEvaluator = _FakeEvaluator
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx", types.ModuleType("onnx_light.onnx")),
            ("onnx_light.onnx.reference", fake_reference),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            runner = rbc._run_with_onnx_light(model)
            inputs = [np.array([1.0, 2.0], dtype=np.float32)]
            actual = runner(inputs)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        # The evaluator must be built from serialised bytes so it sees a
        # proto of its own (onnx-light's) ``ModelProto`` type.
        self.assertEqual(constructed["proto"], model.SerializeToString())
        np.testing.assert_array_equal(constructed["feeds"]["x"], inputs[0])
        np.testing.assert_array_equal(actual[0], np.array([2.0, 4.0], dtype=np.float32))

    def test_run_with_onnx_light_expands_map_inputs_to_keys_and_values(self):
        import types

        import numpy as np
        import onnx
        from onnx import helper

        node = helper.make_node("Identity", ["x"], ["y"])
        map_type = helper.make_map_type_proto(
            onnx.TensorProto.INT64,
            helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, []),
        )
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_value_info("x", map_type)],
            [helper.make_value_info("y", map_type)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        constructed: dict = {}

        class _FakeEvaluator:
            input_names = ["x_keys", "x_values"]

            def __init__(self, proto):
                constructed["proto"] = proto

            def run(self, output_names, feeds):
                constructed["feeds"] = feeds
                return [feeds["x_values"]]

        fake_reference = types.ModuleType("onnx_light.onnx.reference")
        fake_reference.ReferenceEvaluator = _FakeEvaluator
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx", types.ModuleType("onnx_light.onnx")),
            ("onnx_light.onnx.reference", fake_reference),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            runner = rbc._run_with_onnx_light(model)
            actual = runner([{2: 1.0, 5: -3.0}])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(constructed["proto"], model.SerializeToString())
        np.testing.assert_array_equal(constructed["feeds"]["x_keys"], np.array([2, 5]))
        np.testing.assert_array_equal(
            constructed["feeds"]["x_values"], np.array([1.0, -3.0])
        )
        np.testing.assert_array_equal(actual[0], np.array([1.0, -3.0]))

    def test_run_with_onnx_light_expands_map_inputs_with_named_input_descriptors(self):
        import types

        import numpy as np
        import onnx
        from onnx import helper

        node = helper.make_node("Identity", ["x"], ["y"])
        map_type = helper.make_map_type_proto(
            onnx.TensorProto.INT64,
            helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, []),
        )
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_value_info("x", map_type)],
            [helper.make_value_info("y", map_type)],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        constructed: dict = {}

        class _FakeInput:
            def __init__(self, name):
                self.name = name

        class _FakeEvaluator:
            input_names = [_FakeInput("x_keys"), _FakeInput("x_values")]

            def __init__(self, proto):
                constructed["proto"] = proto

            def run(self, output_names, feeds):
                constructed["feeds"] = feeds
                return [feeds["x_values"]]

        fake_reference = types.ModuleType("onnx_light.onnx.reference")
        fake_reference.ReferenceEvaluator = _FakeEvaluator
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx", types.ModuleType("onnx_light.onnx")),
            ("onnx_light.onnx.reference", fake_reference),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            runner = rbc._run_with_onnx_light(model)
            actual = runner([{2: 1.0, 5: -3.0}])
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(constructed["proto"], model.SerializeToString())
        np.testing.assert_array_equal(constructed["feeds"]["x_keys"], np.array([2, 5]))
        np.testing.assert_array_equal(
            constructed["feeds"]["x_values"], np.array([1.0, -3.0])
        )
        np.testing.assert_array_equal(actual[0], np.array([1.0, -3.0]))

    def test_row_from_results_includes_tag_when_provided(self):
        row = rbc._row_from_results(
            "test_qlinearmatmul",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {"success": True, "error": "", "error_step": ""},
            },
            versions={"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
            tag="quantization",
        )
        self.assertEqual(row["tag"], "quantization")

    def test_row_from_results_omits_empty_tag(self):
        row = rbc._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {"success": True, "error": "", "error_step": ""},
            },
            versions={"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertNotIn("tag", row)

    def test_row_from_results_carries_previous_tag_when_current_missing(self):
        row = rbc._row_from_results(
            "test_qlinearmatmul",
            {
                "onnxruntime": {"success": False, "error": "boom", "error_step": "run"},
                "reference": {"success": False, "error": "boom", "error_step": "run"},
            },
            previous={"name": "test_qlinearmatmul", "tag": "quantization"},
            versions={"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertEqual(row["tag"], "quantization")

    def test_build_payload_propagates_tag_from_discover(self):
        tests = [
            {
                "name": "test_a",
                "model": "model_a",
                "data_sets": [],
                "tag": "inference",
            },
            {
                "name": "test_b",
                "model": "model_b",
                "data_sets": [],
                "tag": "quantization",
            },
            {"name": "test_c", "model": "model_c", "data_sets": []},
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            return {"success": True, "error": "", "error_step": ""}

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        by_name = {row["name"]: row for row in payload["tests"]}
        self.assertEqual(by_name["test_a"]["tag"], "inference")
        self.assertEqual(by_name["test_b"]["tag"], "quantization")
        self.assertNotIn("tag", by_name["test_c"])

    def test_discover_node_tests_loads_from_onnx_light(self):
        """``discover_node_tests`` must materialise onnx-light test cases."""
        import types

        import numpy as np
        import onnx
        from onnx import helper

        # Build a tiny in-memory ONNX model representing ``y = Relu(x)`` so
        # the fake ``collect_test_case`` mirrors the shape of onnx-light's
        # output (TestCase with a ``ModelProto`` and ``data_sets``).
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        inputs = [np.array([-1.0, 2.0], dtype=np.float32)]
        outputs = [np.array([0.0, 2.0], dtype=np.float32)]

        node_tc = types.SimpleNamespace(
            name="test_relu_light",
            kind="node",
            tag="inference",
            model=model,
            data_sets=[(inputs, outputs)],
            model_dir=None,
        )
        # A test from a different kind must be filtered out.
        simple_tc = types.SimpleNamespace(
            name="test_simple_other",
            kind="simple",
            model=model,
            data_sets=[(inputs, outputs)],
            model_dir=None,
        )
        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_relu_light": node_tc,
            "test_simple_other": simple_tc,
        }
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
            (
                "onnx_light.onnx_lib.backend",
                types.ModuleType("onnx_light.onnx_lib.backend"),
            ),
            (
                "onnx_light.onnx_lib.backend.test",
                types.ModuleType("onnx_light.onnx_lib.backend.test"),
            ),
            ("onnx_light.onnx_lib.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            discovered = rbc.discover_node_tests(kind="node")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(len(discovered), 1)
        entry = discovered[0]
        self.assertEqual(entry["name"], "test_relu_light")
        # The model is kept in memory as an ``onnx.ModelProto``.
        self.assertIsInstance(entry["model"], onnx.ModelProto)
        self.assertEqual(entry["model"].graph.node[0].op_type, "Relu")
        # Data sets are kept in memory as numpy arrays, ready to feed to
        # the backends without any disk round-trip.
        self.assertEqual(len(entry["data_sets"]), 1)
        loaded_inputs, loaded_outputs = entry["data_sets"][0]
        np.testing.assert_array_equal(loaded_inputs[0], inputs[0])
        np.testing.assert_array_equal(loaded_outputs[0], outputs[0])
        # The tag attached to the onnx-light test case is propagated so
        # the dashboard can group rows by tag.
        self.assertEqual(entry["tag"], "inference")

    def test_discover_node_tests_loads_data_sets_from_disk_when_model_set(self):
        """When a test case carries a model but no data sets, load data from disk.

        The tiny-LLM shape-inference tests shipped with onnx-light populate
        the ``model`` attribute in-memory but leave ``data_sets`` empty;
        the matching test_data_set_* directories live on disk under
        ``model_dir``.  ``discover_node_tests`` must fill in the missing
        data sets from disk so ``run_test_with_backend`` does not report
        "no test_data_set_* directory found".
        """
        import types

        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        inp_arr = np.array([-1.0, 2.0], dtype=np.float32)
        out_arr = np.array([0.0, 2.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            # Write model.onnx (not used by this code path, but present for
            # completeness – only the data sets are missing in-memory).
            onnx.save(model, os.path.join(tmp, "model.onnx"))
            ds_dir = os.path.join(tmp, "test_data_set_0")
            os.makedirs(ds_dir)
            with open(os.path.join(ds_dir, "input_0.pb"), "wb") as fh:
                fh.write(numpy_helper.from_array(inp_arr, name="x").SerializeToString())
            with open(os.path.join(ds_dir, "output_0.pb"), "wb") as fh:
                fh.write(numpy_helper.from_array(out_arr, name="y").SerializeToString())

            # The test case has a model in memory but NO data sets; the data
            # sets must be loaded from model_dir.
            tc = types.SimpleNamespace(
                name="test_tiny_llm",
                kind="model",
                tag="inference",
                model=model,
                data_sets=[],
                model_dir=tmp,
            )

            fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
            fake_module.collect_test_case = lambda include_big=False: {"test_tiny_llm": tc}
            parents = [
                ("onnx_light", types.ModuleType("onnx_light")),
                ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
                (
                    "onnx_light.onnx_lib.backend",
                    types.ModuleType("onnx_light.onnx_lib.backend"),
                ),
                (
                    "onnx_light.onnx_lib.backend.test",
                    types.ModuleType("onnx_light.onnx_lib.backend.test"),
                ),
                ("onnx_light.onnx_lib.backend.test.case", fake_module),
            ]
            saved = {name: sys.modules.get(name) for name, _ in parents}
            original_model_to_onnx = rbc._onnx_light_model_to_onnx
            rbc._onnx_light_model_to_onnx = lambda m: m
            try:
                for name, mod in parents:
                    sys.modules[name] = mod
                discovered = rbc.discover_node_tests(kind="model")
            finally:
                rbc._onnx_light_model_to_onnx = original_model_to_onnx
                for name, mod in saved.items():
                    if mod is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = mod

        self.assertEqual(len(discovered), 1)
        entry = discovered[0]
        self.assertEqual(entry["name"], "test_tiny_llm")
        # Data sets must have been loaded from disk.
        self.assertEqual(len(entry["data_sets"]), 1)
        loaded_inputs, loaded_outputs = entry["data_sets"][0]
        np.testing.assert_array_equal(loaded_inputs[0], inp_arr)
        np.testing.assert_array_equal(loaded_outputs[0], out_arr)

    def test_discover_node_tests_skips_when_no_data_sets_on_disk(self):
        """Test cases with no in-memory data sets and no test_data_set_* dirs are skipped.

        Shape-inference tests like ``test_cc_shape_inference_tiny_llm_inlined``
        carry a model in-memory but have neither in-memory ``data_sets`` nor
        ``test_data_set_*`` directories on disk.  ``discover_node_tests`` must
        silently skip them instead of returning an entry with an empty
        ``data_sets`` list that would later cause ``run_test_with_backend`` to
        report "no test_data_set_* directory found" as a load failure for every
        backend.
        """
        import types

        import onnx
        from onnx import helper

        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

        with tempfile.TemporaryDirectory() as tmp:
            # model_dir exists but contains no test_data_set_* subdirectories.
            tc = types.SimpleNamespace(
                name="test_cc_shape_inference_tiny_llm_inlined",
                kind="model",
                tag="inference",
                model=model,
                data_sets=[],
                model_dir=tmp,
            )

            fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
            fake_module.collect_test_case = lambda include_big=False: {
                "test_cc_shape_inference_tiny_llm_inlined": tc
            }
            parents = [
                ("onnx_light", types.ModuleType("onnx_light")),
                ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
                (
                    "onnx_light.onnx_lib.backend",
                    types.ModuleType("onnx_light.onnx_lib.backend"),
                ),
                (
                    "onnx_light.onnx_lib.backend.test",
                    types.ModuleType("onnx_light.onnx_lib.backend.test"),
                ),
                ("onnx_light.onnx_lib.backend.test.case", fake_module),
            ]
            saved = {name: sys.modules.get(name) for name, _ in parents}
            original_model_to_onnx = rbc._onnx_light_model_to_onnx
            rbc._onnx_light_model_to_onnx = lambda m: m
            try:
                for name, mod in parents:
                    sys.modules[name] = mod
                discovered = rbc.discover_node_tests(kind="model")
            finally:
                rbc._onnx_light_model_to_onnx = original_model_to_onnx
                for name, mod in saved.items():
                    if mod is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = mod

        # The test case must be omitted entirely, not included with empty data sets.
        self.assertEqual(discovered, [])

    def test_normalize_kinds_accepts_various_shapes(self):
        self.assertEqual(rbc._normalize_kinds(None), ())
        self.assertEqual(rbc._normalize_kinds(""), ())
        self.assertEqual(rbc._normalize_kinds("node"), ("node",))
        self.assertEqual(rbc._normalize_kinds("node, model"), ("node", "model"))
        self.assertEqual(rbc._normalize_kinds(["node", "model"]), ("node", "model"))
        # Duplicates are dropped, preserving first-seen order.
        self.assertEqual(
            rbc._normalize_kinds(("node,model", "node")), ("node", "model")
        )

    def test_default_kind_includes_node_and_model(self):
        self.assertEqual(rbc.DEFAULT_KINDS, ("node", "model"))
        self.assertEqual(rbc._normalize_kinds(rbc.DEFAULT_KIND), ("node", "model"))

    def test_discover_node_tests_filters_multiple_kinds(self):
        """``discover_node_tests`` keeps every case whose kind matches.

        In particular the ``test_cc_shape_inference_*`` family ships with
        ``kind="model"`` (see ``onnx-light``'s
        ``onnx_backend_test/cases_for_shapes/inference/``) and must be
        included in the backend-test-coverage page alongside the
        single-node ``kind="node"`` tests so the dashboard reports
        backend-execution status for the shape-inference cases too
        (issue #352).
        """
        import types

        class Case:
            def __init__(self, name, kind, tag=""):
                self.name = name
                self.kind = kind
                self.tag = tag
                self.model = "model_proto"
                self.data_sets = [([1], [1])]
                self.model_dir = None

        cases = {
            "test_node": Case("test_node", "node"),
            "test_cc_shape_inference_x": Case(
                "test_cc_shape_inference_x", "model", "inference"
            ),
            "test_simple": Case("test_simple", "simple"),
        }

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: cases
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
            (
                "onnx_light.onnx_lib.backend",
                types.ModuleType("onnx_light.onnx_lib.backend"),
            ),
            (
                "onnx_light.onnx_lib.backend.test",
                types.ModuleType("onnx_light.onnx_lib.backend.test"),
            ),
            ("onnx_light.onnx_lib.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        original_model_to_onnx = rbc._onnx_light_model_to_onnx
        original_tensor_to_numpy = rbc._onnx_light_tensor_to_numpy
        rbc._onnx_light_model_to_onnx = lambda m: m
        rbc._onnx_light_tensor_to_numpy = lambda a: a
        try:
            for name, mod in parents:
                sys.modules[name] = mod

            # Default kind keeps both ``node`` and ``model`` cases.
            discovered_default = rbc.discover_node_tests()
            self.assertEqual(
                [d["name"] for d in discovered_default],
                ["test_cc_shape_inference_x", "test_node"],
            )

            # Comma-separated string filter.
            discovered_pair = rbc.discover_node_tests(kind="node,model")
            self.assertEqual(
                sorted(d["name"] for d in discovered_pair),
                ["test_cc_shape_inference_x", "test_node"],
            )

            # Single-kind filter still works (backwards-compatible).
            discovered_single = rbc.discover_node_tests(kind="model")
            self.assertEqual(
                [d["name"] for d in discovered_single],
                ["test_cc_shape_inference_x"],
            )

            # Iterable filter.
            discovered_iter = rbc.discover_node_tests(kind=["simple"])
            self.assertEqual([d["name"] for d in discovered_iter], ["test_simple"])
        finally:
            rbc._onnx_light_model_to_onnx = original_model_to_onnx
            rbc._onnx_light_tensor_to_numpy = original_tensor_to_numpy
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_compare_outputs_detects_shape_mismatch(self):
        import numpy as np

        msg = rbc._compare_outputs(
            [np.zeros((2, 3))], [np.zeros((2, 2))], rtol=1e-3, atol=1e-4
        )
        self.assertIsNotNone(msg)
        self.assertIn("shape mismatch", msg)

    def test_compare_outputs_detects_count_mismatch(self):
        import numpy as np

        msg = rbc._compare_outputs(
            [np.zeros(3), np.zeros(3)],
            [np.zeros(3)],
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertIsNotNone(msg)
        self.assertIn("count mismatch", msg)

    def test_compare_outputs_accepts_close_values(self):
        import numpy as np

        self.assertIsNone(
            rbc._compare_outputs(
                [np.array([1.0, 2.0])],
                [np.array([1.0 + 1e-7, 2.0])],
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_compare_outputs_handles_string_outputs(self):
        import numpy as np

        self.assertIsNone(
            rbc._compare_outputs(
                [np.array(["a", "b"])],
                [np.array(["a", "b"])],
                rtol=1e-3,
                atol=1e-4,
            )
        )
        msg = rbc._compare_outputs(
            [np.array(["a", "b"])],
            [np.array(["a", "c"])],
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertIsNotNone(msg)

    def test_compare_outputs_handles_sequence_outputs(self):
        import numpy as np

        expected = [[np.array([1.0, 2.0]), np.array([3.0])]]
        actual = [[np.array([1.0, 2.0]), np.array([3.0])]]
        self.assertIsNone(rbc._compare_outputs(expected, actual, rtol=1e-3, atol=1e-4))

        mismatched = [[np.array([1.0, 2.0]), np.array([9.0])]]
        msg = rbc._compare_outputs(expected, mismatched, rtol=1e-3, atol=1e-4)
        self.assertIsNotNone(msg)

        shorter = [[np.array([1.0, 2.0])]]
        msg = rbc._compare_outputs(expected, shorter, rtol=1e-3, atol=1e-4)
        self.assertIsNotNone(msg)
        self.assertIn("length mismatch", msg)

    def test_compare_outputs_handles_optional_none_outputs(self):
        import numpy as np

        self.assertIsNone(rbc._compare_outputs([None], [None], rtol=1e-3, atol=1e-4))
        msg = rbc._compare_outputs([None], [np.array([1.0])], rtol=1e-3, atol=1e-4)
        self.assertIsNotNone(msg)
        self.assertIn("None", msg)

    def test_compare_outputs_reports_precise_numeric_mismatch(self):
        import numpy as np

        msg = rbc._compare_outputs(
            [np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])],
            [np.array([-2.0, -2.0, -1.0, 0.0, 1.0, 1.0, 1.0])],
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertIsNotNone(msg)
        # The message must surface the precise statistics, not just the
        # generic "Not equal to tolerance" header.
        self.assertIn("Mismatched elements", msg)
        self.assertIn("Max absolute difference", msg)
        self.assertNotIn("Not equal to tolerance", msg)

    def test_compare_outputs_accepts_matching_sub_byte_int(self):
        import numpy as np

        try:
            import ml_dtypes
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("ml_dtypes is not installed")

        values = [7, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        expected = np.array(values, dtype=ml_dtypes.int4)
        actual = np.array(values, dtype=ml_dtypes.int4)
        self.assertIsNone(
            rbc._compare_outputs([expected], [actual], rtol=1e-3, atol=1e-4)
        )

    def test_compare_outputs_reports_exact_sub_byte_int_difference(self):
        import numpy as np

        try:
            import ml_dtypes
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("ml_dtypes is not installed")

        # ``int4`` arithmetic wraps modulo 16, so the true difference between
        # ``-8`` and ``5`` (13) would wrap to 3 without widening. Widening the
        # packed sub-byte dtype must surface the exact magnitude. ``5`` is an
        # interior value (not a saturation bound), so the disagreement is a
        # genuine mismatch rather than a spec-undefined out-of-range cast.
        expected = np.array([-8], dtype=ml_dtypes.int4)
        actual = np.array([5], dtype=ml_dtypes.int4)
        msg = rbc._compare_outputs([expected], [actual], rtol=1e-3, atol=1e-4)
        self.assertIsNotNone(msg)
        self.assertIn("Max absolute difference", msg)
        self.assertIn("13", msg)

    def test_compare_outputs_accepts_saturating_sub_byte_int_cast(self):
        import numpy as np

        try:
            import ml_dtypes
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("ml_dtypes is not installed")

        # The ONNX ``Cast`` spec leaves float -> fixed-point conversions
        # undefined when the source is out of range. The bundled backend test
        # data wraps around while a spec-compliant runtime may saturate to the
        # representable bound; the comparison must treat that as a match.
        for dtype, lo, hi in (
            (ml_dtypes.int4, -8, 7),
            (ml_dtypes.uint4, 0, 15),
            (ml_dtypes.uint2, 0, 3),
        ):
            with self.subTest(dtype=dtype.__name__):
                source = np.arange(lo - 2, hi + 3).astype(np.float32)
                expected = source.astype(dtype)  # wrap-around (test data)
                actual = np.clip(source, lo, hi).astype(dtype)  # saturating
                # The two encodings must genuinely differ to exercise the
                # tolerance rather than accidentally agree.
                self.assertTrue(
                    np.any(
                        np.asarray(expected).astype(np.int64)
                        != np.asarray(actual).astype(np.int64)
                    )
                )
                self.assertIsNone(
                    rbc._compare_outputs([expected], [actual], rtol=1e-3, atol=1e-4)
                )

    def test_compare_outputs_reports_interior_sub_byte_int_mismatch(self):
        import numpy as np

        try:
            import ml_dtypes
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("ml_dtypes is not installed")

        # A disagreement where the actual value is *not* a saturation bound
        # cannot be explained by an out-of-range cast and must be reported.
        expected = np.array([3, 1], dtype=ml_dtypes.int4)
        actual = np.array([3, 2], dtype=ml_dtypes.int4)
        msg = rbc._compare_outputs([expected], [actual], rtol=1e-3, atol=1e-4)
        self.assertIsNotNone(msg)
        self.assertIn("mismatch", msg)

    def test_load_test_data_sets_decodes_sequence_and_optional(self):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper

        tensor = numpy_helper.from_array(
            np.array([1.0, 2.0], dtype=np.float32), name="t"
        )
        seq = onnx.SequenceProto()
        seq.name = "s"
        seq.elem_type = onnx.SequenceProto.TENSOR
        seq.tensor_values.extend(
            [numpy_helper.from_array(np.array([3.0], dtype=np.float32))]
        )

        tensor_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [2])
        seq_type = helper.make_sequence_type_proto(tensor_type)
        opt_type = helper.make_optional_type_proto(tensor_type)

        model = helper.make_model(
            helper.make_graph(
                nodes=[],
                name="g",
                inputs=[
                    helper.make_value_info("tensor_in", tensor_type),
                    helper.make_value_info("seq_in", seq_type),
                ],
                outputs=[helper.make_value_info("opt_out", opt_type)],
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            ds_dir = os.path.join(tmp, "test_data_set_0")
            os.makedirs(ds_dir)
            with open(os.path.join(ds_dir, "input_0.pb"), "wb") as fh:
                fh.write(tensor.SerializeToString())
            with open(os.path.join(ds_dir, "input_1.pb"), "wb") as fh:
                fh.write(seq.SerializeToString())
            # A populated optional output decodes to its tensor value.
            opt_out = onnx.OptionalProto()
            opt_out.name = "opt_out"
            opt_out.elem_type = onnx.OptionalProto.TENSOR
            opt_out.tensor_value.CopyFrom(
                numpy_helper.from_array(np.array([5.0], dtype=np.float32))
            )
            with open(os.path.join(ds_dir, "output_0.pb"), "wb") as fh:
                fh.write(opt_out.SerializeToString())

            data_sets = rbc._load_test_data_sets(tmp, model)

        self.assertEqual(len(data_sets), 1)
        inputs, outputs = data_sets[0]
        self.assertIsInstance(inputs[0], np.ndarray)
        self.assertIsInstance(inputs[1], list)
        np.testing.assert_array_equal(inputs[1][0], np.array([3.0], dtype=np.float32))
        np.testing.assert_array_equal(outputs[0], np.array([5.0], dtype=np.float32))

    def test_onnx_light_tensor_to_numpy_decodes_sequence_and_optional(self):
        import numpy as np
        import onnx
        from onnx import numpy_helper

        # A SequenceProto must decode to a list of numpy arrays so it can be
        # compared against the sequence (list) produced by the runners, rather
        # than being garbled into a non-sequence tensor.
        seq = onnx.SequenceProto()
        seq.name = "s"
        seq.elem_type = onnx.SequenceProto.TENSOR
        seq.tensor_values.extend(
            [
                numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32)),
                numpy_helper.from_array(np.array([3.0], dtype=np.float32)),
            ]
        )
        seq_value = rbc._onnx_light_tensor_to_numpy(seq)
        self.assertIsInstance(seq_value, list)
        self.assertEqual(len(seq_value), 2)
        np.testing.assert_array_equal(
            seq_value[0], np.array([1.0, 2.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(seq_value[1], np.array([3.0], dtype=np.float32))

        # A populated OptionalProto decodes to its tensor value.
        opt = onnx.OptionalProto()
        opt.name = "o"
        opt.elem_type = onnx.OptionalProto.TENSOR
        opt.tensor_value.CopyFrom(
            numpy_helper.from_array(np.array([5.0], dtype=np.float32))
        )
        opt_value = rbc._onnx_light_tensor_to_numpy(opt)
        np.testing.assert_array_equal(opt_value, np.array([5.0], dtype=np.float32))

        # A plain TensorProto still decodes to a numpy array.
        tensor = numpy_helper.from_array(np.array([7.0], dtype=np.float32))
        tensor_value = rbc._onnx_light_tensor_to_numpy(tensor)
        self.assertIsInstance(tensor_value, np.ndarray)
        np.testing.assert_array_equal(tensor_value, np.array([7.0], dtype=np.float32))

    def test_write_payload_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "onnx-light", "backend_test_coverage.json")
            payload = {"date": "2024-01-01T00:00:00Z", "tests": []}
            rbc.write_payload(json_path, payload)
            with open(json_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), payload)

    def test_main_writes_cache_file(self):
        original_build = rbc.build_payload

        def fake_build(**kwargs):
            return {
                "date": "2024-01-01T00:00:00Z",
                "kind": kwargs.get("kind", "node"),
                "tolerances": {"rtol": 1e-3, "atol": 1e-4},
                "versions": {},
                "totals": {
                    "onnxruntime": {"pass": 1, "fail": 0},
                    "reference": {"pass": 1, "fail": 0},
                    "onnx_light": {"pass": 1, "fail": 0},
                },
                "tests": [
                    {
                        "name": "test_x",
                        "onnxruntime": True,
                        "reference": True,
                        "onnx_light": True,
                    }
                ],
            }

        rbc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rbc.main(["--cache-dir", tmp])
                self.assertEqual(code, 0)
                with open(
                    os.path.join(tmp, "onnx-light", "backend_test_coverage.json"),
                    encoding="utf-8",
                ) as fh:
                    payload = json.load(fh)
                self.assertEqual(payload["tests"][0]["name"], "test_x")
                self.assertEqual(payload["kind"], rbc.DEFAULT_KIND)
        finally:
            rbc.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = rbc.build_payload

        def fake_build(**kwargs):
            raise RuntimeError("boom")

        rbc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rbc.main(["--cache-dir", tmp])
                self.assertEqual(code, 1)
                self.assertFalse(
                    os.path.exists(
                        os.path.join(tmp, "onnx-light", "backend_test_coverage.json")
                    )
                )
        finally:
            rbc.build_payload = original_build

    def test_run_test_with_backend_returns_elapsed_s(self):
        result = rbc.run_test_with_backend(None, [], "totally-unknown")
        self.assertIn("elapsed_s", result)
        self.assertIsInstance(result["elapsed_s"], float)

    def test_row_from_results_includes_elapsed_s(self):
        results = {
            "onnxruntime": {
                "success": True,
                "error": "",
                "error_step": "",
                "elapsed_s": 0.1,
            },
            "reference": {
                "success": True,
                "error": "",
                "error_step": "",
                "elapsed_s": 0.2,
            },
            "onnx_light": {
                "success": True,
                "error": "",
                "error_step": "",
                "elapsed_s": 0.3,
            },
        }
        row = rbc._row_from_results("test_relu", results)
        self.assertAlmostEqual(row["onnxruntime_elapsed_s"], 0.1)
        self.assertAlmostEqual(row["reference_elapsed_s"], 0.2)
        self.assertAlmostEqual(row["onnx_light_elapsed_s"], 0.3)
        self.assertAlmostEqual(row["elapsed_s"], 0.6, places=5)

    def test_build_payload_includes_slowest_tests(self):
        tests = [
            {"name": f"test_{i}", "model": f"model_{i}", "data_sets": []}
            for i in range(5)
        ]
        elapsed_map = {
            "model_0": 0.5,
            "model_1": 0.1,
            "model_2": 1.2,
            "model_3": 0.3,
            "model_4": 0.8,
        }

        def fake_run(model, data_sets, backend, rtol, atol):
            return {
                "success": True,
                "error": "",
                "error_step": "",
                "elapsed_s": elapsed_map[model],
            }

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertIn("slowest_tests", payload)
        slowest = payload["slowest_tests"]
        # All 5 tests fit in the top 20.
        self.assertEqual(len(slowest), 5)
        # Sorted by descending total elapsed time.
        self.assertEqual(slowest[0]["name"], "test_2")
        self.assertEqual(slowest[1]["name"], "test_4")
        self.assertEqual(slowest[2]["name"], "test_0")
        for entry in slowest:
            self.assertIn("elapsed_s", entry)
            self.assertIn("name", entry)

    def test_build_payload_slowest_tests_capped_at_20(self):
        tests = [
            {"name": f"test_{i}", "model": f"model_{i}", "data_sets": []}
            for i in range(25)
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            return {"success": True, "error": "", "error_step": "", "elapsed_s": 0.1}

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertLessEqual(len(payload["slowest_tests"]), 20)


if __name__ == "__main__":
    unittest.main()
