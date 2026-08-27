"""Tests for the onnx-light-cpu benchmark recorder."""

from __future__ import annotations

import datetime as dt
import os
import sys
import types
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_cpu_benchmark as rcb


class TestDiscovery(unittest.TestCase):
    def test_uses_only_cpu_benchmark_cases(self):
        calls = []
        module = types.ModuleType("onnx_light_cpu")
        module.register_backend_test_cases = lambda: calls.append("registered")
        original = sys.modules.get("onnx_light_cpu")
        collect = rcb._collect_benchmark_cases
        rcb._collect_benchmark_cases = lambda: [
            types.SimpleNamespace(name="test_cpu_abs_benchmark", kind="node"),
            types.SimpleNamespace(name="test_cc_abs_benchmark", kind="node"),
            types.SimpleNamespace(name="test_cpu_abs_float32", kind="node"),
        ]
        sys.modules["onnx_light_cpu"] = module
        try:
            tests = rcb.discover_benchmark_tests()
        finally:
            rcb._collect_benchmark_cases = collect
            if original is None:
                del sys.modules["onnx_light_cpu"]
            else:
                sys.modules["onnx_light_cpu"] = original
        self.assertEqual(calls, ["registered"])
        self.assertEqual([test["name"] for test in tests], ["test_cpu_abs_benchmark"])
        self.assertNotIn("model", tests[0])


class TestLoading(unittest.TestCase):
    def test_loads_one_benchmark_case_by_exact_name(self):
        calls = []
        benchmark_mode = object()
        backend = types.ModuleType("onnx_light.onnx.backend")
        backend.TestMode = types.SimpleNamespace(BENCHMARK=benchmark_mode)
        backend.collect_test_cases_by_name = lambda pattern, **kwargs: (
            calls.append((pattern, kwargs)) or [types.SimpleNamespace(model="cc-model")]
        )
        modules = {
            "onnx_light": types.ModuleType("onnx_light"),
            "onnx_light.onnx": types.ModuleType("onnx_light.onnx"),
            "onnx_light.onnx.backend": backend,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                rcb.rlb, "_onnx_light_model_to_onnx", lambda model: f"onnx-{model}"
            ),
            patch.object(rcb.rlb, "_cc_data_sets_to_python", lambda case: ["data"]),
        ):
            loaded = rcb._load_benchmark_test("test_cpu_abs[1]_benchmark")

        self.assertEqual(
            calls,
            [
                (
                    r"^test_cpu_abs\[1\]_benchmark$",
                    {"include_big": True, "mode": benchmark_mode},
                )
            ],
        )
        self.assertEqual(
            loaded,
            {
                "name": "test_cpu_abs[1]_benchmark",
                "model": "onnx-cc-model",
                "data_sets": ["data"],
            },
        )


class TestRows(unittest.TestCase):
    def test_first_input_type_is_separate_from_signature(self):
        class Array:
            def __init__(self, dtype, shape):
                self.dtype = types.SimpleNamespace(name=dtype)
                self.shape = shape

        inputs = rcb._format_inputs([Array("float32", (2, 3)), Array("int64", (3,))])
        row = rcb._row(
            inputs,
            {"success": True, "avg_ms": 1.0},
            {"success": True, "avg_ms": 2.0},
        )
        self.assertEqual(row["inputs"], "float32[2x3], int64[3]")
        self.assertEqual(row["input_type"], "float32")
        self.assertEqual(row["speedup_cpu"], 2.0)

    def test_groups_dimensions_by_operator_and_first_input_type(self):
        measurements = [
            {
                "operator": "Abs",
                "test_name": "test_cpu_abs_n65536_benchmark",
                "row": {
                    "inputs": "float32[65536]",
                    "input_type": "float32",
                    "input_elements": 65536,
                    "speedup_cpu": 2.0,
                },
            },
            {
                "operator": "Abs",
                "test_name": "test_cpu_abs_n1024_benchmark",
                "row": {
                    "inputs": "float32[1024]",
                    "input_type": "float32",
                    "input_elements": 1024,
                    "speedup_cpu": 1.0,
                },
            },
            {
                "operator": "Abs",
                "test_name": "test_cpu_abs_float64_benchmark",
                "row": {
                    "inputs": "float64[1024]",
                    "input_type": "float64",
                    "input_elements": 1024,
                    "speedup_cpu": 0.5,
                },
            },
        ]

        examples = rcb._group_measurements(measurements)

        self.assertEqual(len(examples), 2)
        float32 = examples[0]
        self.assertEqual(float32["name"], "Abs_float32_benchmark")
        self.assertEqual(
            [row["inputs"] for row in float32["rows"]],
            ["float32[1024]", "float32[65536]"],
        )
        self.assertEqual(
            [row["test_name"] for row in float32["rows"]],
            [
                "test_cpu_abs_n1024_benchmark",
                "test_cpu_abs_n65536_benchmark",
            ],
        )
        self.assertEqual(float32["summary"]["inputs"], 2)
        self.assertEqual(float32["summary"]["avg_speedup_cpu"], 1.5)
        self.assertEqual(float32["summary"]["min_speedup_cpu"], 1.0)
        self.assertEqual(float32["summary"]["max_speedup_cpu"], 2.0)
        self.assertEqual(examples[1]["rows"][0]["input_type"], "float64")


class TestPayload(unittest.TestCase):
    def test_payload_uses_discovered_tests(self):
        calls = {}

        def run(
            tests,
            n_warmup,
            n_measure,
            max_warmup_time_s,
            max_repeat_time_s,
        ):
            calls["tests"] = tests
            calls["n_warmup"] = n_warmup
            calls["max_warmup_time_s"] = max_warmup_time_s
            calls["max_repeat_time_s"] = max_repeat_time_s
            return []

        cpu = types.ModuleType("onnx_light_cpu.onnx_py._cpukernels")
        cpu.detect_simd_level = lambda: 3
        original = sys.modules.get(cpu.__name__)
        sys.modules[cpu.__name__] = cpu
        try:
            payload = rcb.build_payload(
                discover=lambda kind: [{"name": "test_cpu_abs_benchmark"}],
                run=run,
                versions=dict,
                now=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
            )
        finally:
            if original is None:
                del sys.modules[cpu.__name__]
            else:
                sys.modules[cpu.__name__] = original
        self.assertEqual(calls["tests"], [{"name": "test_cpu_abs_benchmark"}])
        self.assertEqual(calls["n_warmup"], 2)
        self.assertEqual(calls["max_warmup_time_s"], 0.05)
        self.assertEqual(calls["max_repeat_time_s"], rcb.MAX_REPEAT_TIME_S)
        self.assertEqual(payload["max_warmup_time_s"], 0.05)
        self.assertEqual(payload["max_repeat_time_s"], 0.2)
        self.assertEqual(payload["simd_name"], "AVX2")
        self.assertEqual(payload["date"], "2026-01-02T00:00:00Z")

    def test_run_tests_uses_global_backend_phases(self):
        model = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                node=[types.SimpleNamespace(op_type="Abs", domain="")]
            )
        )
        tests = [
            {
                "name": f"test_cpu_abs_{index}_benchmark",
                "model": model,
                "data_sets": [
                    (([types.SimpleNamespace(dtype="float32", shape=(1,))]), [])
                ],
            }
            for index in range(2)
        ]
        calls = []

        def run(
            model,
            data_sets,
            backend,
            n_warmup,
            n_measure,
            max_warmup_time_s,
            max_repeat_time_s,
        ):
            calls.append((backend, n_warmup, max_warmup_time_s, max_repeat_time_s))
            return {"success": True, "avg_ms": 1.0}

        with patch.object(rcb.rlb, "_log") as log:
            rcb.run_tests(tests, run=run)

        self.assertEqual(
            calls,
            [
                ("onnx_light_cpu", 2, 0.05, 0.2),
                ("onnx_light_cpu", 2, 0.05, 0.2),
                ("onnxruntime", 2, 0.05, 0.2),
                ("onnxruntime", 2, 0.05, 0.2),
            ],
        )
        self.assertEqual(
            [call.args[0] for call in log.call_args_list],
            [
                "Benchmarking 1/2 tests (onnx_light_cpu): test_cpu_abs_0_benchmark",
                "Benchmarking 2/2 tests (onnx_light_cpu): test_cpu_abs_1_benchmark",
                "Benchmarking 1/2 tests (onnxruntime): test_cpu_abs_0_benchmark",
                "Benchmarking 2/2 tests (onnxruntime): test_cpu_abs_1_benchmark",
            ],
        )

    def test_run_tests_reloads_lazy_cases_for_each_backend(self):
        names = ["test_cpu_abs_1_benchmark", "test_cpu_abs_2_benchmark"]
        loads = []
        calls = []

        def load(name):
            loads.append(name)
            return {
                "name": name,
                "model": types.SimpleNamespace(
                    graph=types.SimpleNamespace(
                        node=[types.SimpleNamespace(op_type="Abs", domain="")]
                    )
                ),
                "data_sets": [
                    ([types.SimpleNamespace(dtype="float32", shape=(1,))], [])
                ],
            }

        def run(model, data_sets, backend, **kwargs):
            calls.append(backend)
            return {"success": True, "avg_ms": 1.0}

        rcb.run_tests([{"name": name} for name in names], run=run, load=load)

        self.assertEqual(loads, names + names)
        self.assertEqual(
            calls,
            ["onnx_light_cpu", "onnx_light_cpu", "onnxruntime", "onnxruntime"],
        )


if __name__ == "__main__":
    unittest.main()
