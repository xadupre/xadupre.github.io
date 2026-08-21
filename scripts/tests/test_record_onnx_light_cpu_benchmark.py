"""Tests for the onnx-light-cpu benchmark recorder."""

from __future__ import annotations

import datetime as dt
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_cpu_benchmark as rcb  # noqa: E402


class TestDiscovery(unittest.TestCase):
    def test_uses_only_cpu_benchmark_cases(self):
        calls = []
        module = types.ModuleType("onnx_light_cpu")
        module.register_backend_test_cases = lambda: calls.append("registered")
        original = sys.modules.get("onnx_light_cpu")
        discover = rcb.rlb.discover_node_tests
        rcb.rlb.discover_node_tests = lambda kind: [
            {"name": "test_cpu_abs_benchmark"},
            {"name": "test_cc_abs_benchmark"},
            {"name": "test_cpu_abs_float32"},
        ]
        sys.modules["onnx_light_cpu"] = module
        try:
            tests = rcb.discover_benchmark_tests()
        finally:
            rcb.rlb.discover_node_tests = discover
            if original is None:
                del sys.modules["onnx_light_cpu"]
            else:
                sys.modules["onnx_light_cpu"] = original
        self.assertEqual(calls, ["registered"])
        self.assertEqual([test["name"] for test in tests], ["test_cpu_abs_benchmark"])


class TestRows(unittest.TestCase):
    def test_first_input_type_is_separate_from_signature(self):
        class Array:
            def __init__(self, dtype, shape):
                self.dtype = types.SimpleNamespace(name=dtype)
                self.shape = shape

        inputs = rcb._format_inputs(
            [Array("float32", (2, 3)), Array("int64", (3,))]
        )
        row = rcb._row(
            inputs,
            {"success": True, "avg_ms": 1.0},
            {"success": True, "avg_ms": 2.0},
        )
        self.assertEqual(row["inputs"], "float32[2x3], int64[3]")
        self.assertEqual(row["input_type"], "float32")
        self.assertEqual(row["speedup_cpu"], 2.0)


class TestPayload(unittest.TestCase):
    def test_payload_uses_discovered_tests(self):
        calls = {}

        def run(tests, n_warmup, n_measure):
            calls["tests"] = tests
            return []

        cpu = types.ModuleType("onnx_light_cpu.onnx_py._cpukernels")
        cpu.detect_simd_level = lambda: 3
        original = sys.modules.get(cpu.__name__)
        sys.modules[cpu.__name__] = cpu
        try:
            payload = rcb.build_payload(
                discover=lambda kind: [{"name": "test_cpu_abs_benchmark"}],
                run=run,
                versions=lambda: {},
                now=dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc),
            )
        finally:
            if original is None:
                del sys.modules[cpu.__name__]
            else:
                sys.modules[cpu.__name__] = original
        self.assertEqual(calls["tests"], [{"name": "test_cpu_abs_benchmark"}])
        self.assertEqual(payload["simd_name"], "AVX2")
        self.assertEqual(payload["date"], "2026-01-02T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
