"""Tests for ``scripts.record_onnx_light_cpu_examples_benchmark``."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_light_cpu_examples_benchmark as rce  # noqa: E402


class TestRowFromTimes(unittest.TestCase):
    def test_all_backends_and_speedup(self):
        row = rce._row_from_times(
            100,
            {
                "numpy": 0.001,
                "onnx_light_cpu": 0.025,
                "onnxruntime": 0.05,
            },
        )
        self.assertEqual(row["size"], 100)
        self.assertEqual(row["numpy_ms"], 0.001)
        self.assertNotIn("onnx_light_ms", row)
        self.assertEqual(row["onnx_light_cpu_ms"], 0.025)
        self.assertEqual(row["onnxruntime_ms"], 0.05)
        # speedup_cpu = onnxruntime / onnx_light_cpu = 0.05 / 0.025 = 2.0
        self.assertEqual(row["speedup_cpu"], 2.0)

    def test_missing_backend_is_omitted(self):
        row = rce._row_from_times(
            100, {"onnx_light_cpu": 1.0, "onnxruntime": 2.0}
        )
        self.assertNotIn("numpy_ms", row)
        self.assertNotIn("onnx_light_ms", row)
        self.assertEqual(row["speedup_cpu"], 2.0)

    def test_no_speedup_without_both_sides(self):
        row = rce._row_from_times(100, {"onnxruntime": 2.0})
        self.assertNotIn("speedup_cpu", row)

    def test_no_speedup_when_cpu_zero(self):
        row = rce._row_from_times(100, {"onnx_light_cpu": 0.0, "onnxruntime": 2.0})
        self.assertNotIn("speedup_cpu", row)

    def test_size_is_int(self):
        row = rce._row_from_times(10**8, {"onnx_light_cpu": 1.0, "onnxruntime": 1.0})
        self.assertIsInstance(row["size"], int)
        self.assertEqual(row["size"], 10**8)


class TestSummarizeExample(unittest.TestCase):
    def test_summary_stats(self):
        rows = [
            rce._row_from_times(1, {"onnx_light_cpu": 1.0, "onnxruntime": 2.0}),
            rce._row_from_times(2, {"onnx_light_cpu": 4.0, "onnxruntime": 2.0}),
        ]
        summary = rce._summarize_example(rows)
        self.assertEqual(summary["sizes"], 2)
        self.assertEqual(summary["cpu_succeeded"], 2)
        self.assertEqual(summary["max_speedup_cpu"], 2.0)
        self.assertEqual(summary["min_speedup_cpu"], 0.5)
        self.assertEqual(summary["avg_speedup_cpu"], round((2.0 + 0.5) / 2, 4))

    def test_summary_without_speedups(self):
        rows = [rce._row_from_times(1, {"numpy": 1.0})]
        summary = rce._summarize_example(rows)
        self.assertEqual(summary["sizes"], 1)
        self.assertEqual(summary["cpu_succeeded"], 0)
        self.assertNotIn("avg_speedup_cpu", summary)


class TestDefaultExamples(unittest.TestCase):
    def test_examples_shape(self):
        examples = rce.default_examples()
        names = [e["name"] for e in examples]
        self.assertEqual(names, ["abs", "exp", "log", "not", "gemm"])
        for example in examples:
            for key in (
                "title",
                "op",
                "source",
                "xlabel",
                "make_model",
                "size_grid",
                "make_inputs",
                "numpy_op",
                "repeat_for",
                "kernel_name",
            ):
                self.assertIn(key, example, f"{example['name']} missing {key}")
            self.assertTrue(example["size_grid"])

    def test_max_size_caps_grid(self):
        examples = rce.default_examples(
            max_abs_size=10000, max_gemm_size=64, max_unary_size=1000
        )
        abs_ex = next(e for e in examples if e["name"] == "abs")
        gemm_ex = next(e for e in examples if e["name"] == "gemm")
        unary = [e for e in examples if e["name"] in {"exp", "log", "not"}]
        self.assertTrue(all(s <= 10000 for s in abs_ex["size_grid"]))
        self.assertTrue(all(s <= 64 for s in gemm_ex["size_grid"]))
        self.assertTrue(all(s <= 1000 for e in unary for s in e["size_grid"]))



    def test_abs_inputs_and_numpy_op(self):
        abs_ex = rce._abs_example(max_size=1000)
        feeds = abs_ex["make_inputs"](abs_ex["size_grid"][0])
        self.assertIn("X", feeds)
        result = abs_ex["numpy_op"](feeds)
        # numpy abs is always non-negative.
        self.assertTrue((result >= 0).all())

    def test_gemm_inputs_and_numpy_op(self):
        gemm_ex = rce._gemm_example(max_size=32)
        feeds = gemm_ex["make_inputs"](16)
        self.assertIn("A", feeds)
        self.assertIn("B", feeds)
        result = gemm_ex["numpy_op"](feeds)
        self.assertEqual(result.shape, (16, 16))

    def test_unary_inputs_and_numpy_ops(self):
        examples = {
            e["name"]: e for e in rce.default_examples(max_unary_size=100)
        }
        exp_values = examples["exp"]["make_inputs"](100)["X"]
        log_values = examples["log"]["make_inputs"](100)["X"]
        not_values = examples["not"]["make_inputs"](100)["X"]
        self.assertEqual(exp_values.dtype.name, "float32")
        self.assertTrue((log_values > 0).all())
        self.assertEqual(not_values.dtype.name, "bool")
        for name in ("exp", "log", "not"):
            example = examples[name]
            feeds = example["make_inputs"](100)
            self.assertEqual(example["numpy_op"](feeds).shape, (100,))


class TestMeasure(unittest.TestCase):
    def test_median_of_calls(self):
        counter = {"n": 0}

        def func():
            counter["n"] += 1

        value = rce.measure(func, repeat=5, warmup=2)
        self.assertGreaterEqual(value, 0.0)
        # 2 warm-up + 5 timed = 7 calls.
        self.assertEqual(counter["n"], 7)

    def test_together_rotates_all_calls(self):
        calls = []
        values = rce.measure_together(
            (lambda: calls.append("a"), lambda: calls.append("b")),
            repeat=3,
            warmup=1,
        )
        self.assertEqual(calls, ["a", "b", "a", "b", "b", "a", "a", "b"])
        self.assertEqual(len(values), 2)


class TestBuildPayload(unittest.TestCase):
    def _fake_run(self, examples, n_warmup, n_measure):
        rows = [rce._row_from_times(1, {"onnx_light_cpu": 1.0, "onnxruntime": 2.0})]
        results = [
            {
                "name": "abs",
                "title": "Abs",
                "op": "Abs",
                "source": "plot_abs_benchmark.py",
                "xlabel": "array size (elements)",
                "size_key": "size",
                "backends": list(rce.BENCHMARK_BACKENDS),
                "rows": rows,
                "summary": rce._summarize_example(rows),
            }
        ]
        meta = {"simd_level": 3, "simd_name": "AVX2"}
        return results, meta

    def test_payload_structure(self):
        payload = rce.build_payload(
            run=self._fake_run,
            versions=lambda: {"numpy": "1.0"},
            now=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
        )
        self.assertEqual(payload["date"], "2024-01-02T03:04:05Z")
        self.assertEqual(payload["n_warmup"], rce.N_WARMUP)
        self.assertEqual(payload["n_measure"], rce.N_MEASURE)
        self.assertEqual(payload["versions"], {"numpy": "1.0"})
        self.assertEqual(payload["simd_level"], 3)
        self.assertEqual(payload["simd_name"], "AVX2")
        self.assertEqual(len(payload["examples"]), 1)
        self.assertEqual(payload["examples"][0]["name"], "abs")

    def test_examples_are_passed_through(self):
        captured = {}

        def fake_run(examples, n_warmup, n_measure):
            captured["examples"] = examples
            captured["n_warmup"] = n_warmup
            captured["n_measure"] = n_measure
            return [], {}

        sentinel = [{"name": "custom"}]
        rce.build_payload(
            examples=sentinel,
            n_warmup=1,
            n_measure=2,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertIs(captured["examples"], sentinel)
        self.assertEqual(captured["n_warmup"], 1)
        self.assertEqual(captured["n_measure"], 2)


class TestWritePayload(unittest.TestCase):
    def test_roundtrip(self):
        payload = {"date": "2024-01-01T00:00:00Z", "examples": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnx-light-cpu", "examples_benchmark.json")
            rce.write_payload(path, payload)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded, payload)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        args = rce.parse_args([])
        self.assertEqual(args.cache_dir, "cache_data")
        self.assertEqual(args.n_warmup, rce.N_WARMUP)
        self.assertEqual(args.n_measure, rce.N_MEASURE)
        self.assertIsNone(args.max_abs_size)
        self.assertIsNone(args.max_gemm_size)
        self.assertIsNone(args.max_unary_size)

    def test_overrides(self):
        args = rce.parse_args(
            [
                "--n-warmup",
                "1",
                "--n-measure",
                "2",
                "--max-abs-size",
                "100",
                "--max-unary-size",
                "1000",
            ]
        )
        self.assertEqual(args.n_warmup, 1)
        self.assertEqual(args.n_measure, 2)
        self.assertEqual(args.max_abs_size, 100)
        self.assertEqual(args.max_unary_size, 1000)


class TestMain(unittest.TestCase):
    def test_main_writes_file(self):
        rows = [rce._row_from_times(1, {"onnx_light_cpu": 1.0, "onnxruntime": 2.0})]

        def fake_build(**kwargs):
            return {
                "date": "2024-01-01T00:00:00Z",
                "n_warmup": 3,
                "n_measure": 10,
                "versions": {},
                "examples": [
                    {"name": "abs", "rows": rows, "summary": {}}
                ],
            }

        original = rce.build_payload
        rce.build_payload = fake_build  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc = rce.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(
                    tmp, "onnx-light-cpu", "examples_benchmark.json"
                )
                self.assertTrue(os.path.exists(path))
        finally:
            rce.build_payload = original  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
