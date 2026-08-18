"""Tests for ``scripts.record_onnx_light_cpu_examples_benchmark``."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import types
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


class TestReferenceRunner(unittest.TestCase):
    def test_passes_numpy_feeds_directly_to_public_evaluator(self):
        calls = {}

        class _ReferenceEvaluator:
            def __init__(self, model):
                calls["model"] = model

            def run(self, output_names, feeds):
                calls["output_names"] = output_names
                calls["feeds"] = feeds
                return ["output"]

        reference = types.ModuleType("onnx_light.onnx.reference")
        reference.ReferenceEvaluator = _ReferenceEvaluator
        onnx = types.ModuleType("onnx_light.onnx")
        onnx.reference = reference
        onnx_light = types.ModuleType("onnx_light")
        onnx_light.onnx = onnx
        modules = {
            "onnx_light": onnx_light,
            "onnx_light.onnx": onnx,
            "onnx_light.onnx.reference": reference,
        }
        saved = {name: sys.modules.get(name) for name in modules}
        model = object()
        feeds = {"X": object()}
        try:
            sys.modules.update(modules)
            runner = rce._make_reference_runner(model)
            result = runner(feeds)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(result, ["output"])
        self.assertIs(calls["model"], model)
        self.assertIsNone(calls["output_names"])
        self.assertIs(calls["feeds"], feeds)


class TestOnnxruntimeRunnerIrVersion(unittest.TestCase):
    """``_make_onnxruntime_session`` clamps too-new IR versions and retries."""

    class _FakeModel:
        def __init__(self, ir_version=0):
            self.ir_version = ir_version

        def SerializeToString(self):
            return json.dumps({"ir_version": self.ir_version}).encode("utf-8")

        def ParseFromString(self, data):
            self.ir_version = json.loads(data.decode("utf-8"))["ir_version"]

    def _install_fakes(self, max_ir):
        created = []

        class _Session:
            def __init__(self, serialized, providers=None):
                ir = json.loads(serialized.decode("utf-8"))["ir_version"]
                created.append(ir)
                if ir > max_ir:
                    raise RuntimeError(
                        f"Unsupported model IR version: {ir}, "
                        f"max supported IR version: {max_ir}"
                    )
                self.ir_version = ir

            def run(self, output_names, feeds):
                return ["output"]

        onnxruntime = types.ModuleType("onnxruntime")
        onnxruntime.InferenceSession = _Session

        onnx = types.ModuleType("onnx_light.onnx")
        onnx.ModelProto = TestOnnxruntimeRunnerIrVersion._FakeModel
        onnx_light = types.ModuleType("onnx_light")
        onnx_light.onnx = onnx

        modules = {
            "onnxruntime": onnxruntime,
            "onnx_light": onnx_light,
            "onnx_light.onnx": onnx,
        }
        return modules, created

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

    def test_supported_ir_version_loads_directly(self):
        modules, created = self._install_fakes(max_ir=13)
        model = self._FakeModel(ir_version=10)
        runner = self._with_modules(
            modules, lambda: rce._make_onnxruntime_runner(model)
        )
        self.assertEqual(created, [10])
        self.assertEqual(runner({"X": 1}), ["output"])

    def test_too_new_ir_version_is_clamped_and_retried(self):
        modules, created = self._install_fakes(max_ir=13)
        model = self._FakeModel(ir_version=14)
        runner = self._with_modules(
            modules, lambda: rce._make_onnxruntime_runner(model)
        )
        # First attempt with IR 14 fails, retried at the max supported IR 13.
        self.assertEqual(created, [14, 13])
        self.assertEqual(runner({"X": 1}), ["output"])

    def test_unrelated_error_is_not_swallowed(self):
        modules, _ = self._install_fakes(max_ir=13)

        class _BrokenSession:
            def __init__(self, serialized, providers=None):
                raise RuntimeError("some other failure")

        modules["onnxruntime"].InferenceSession = _BrokenSession
        model = self._FakeModel(ir_version=14)
        with self.assertRaises(RuntimeError) as ctx:
            self._with_modules(
                modules, lambda: rce._make_onnxruntime_runner(model)
            )
        self.assertIn("some other failure", str(ctx.exception))


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


class TestDiscoverBigModels(unittest.TestCase):
    def _patch_rlb(self, fake):
        saved = sys.modules.get("record_onnx_light_benchmark")
        sys.modules["record_onnx_light_benchmark"] = fake
        return saved

    def _restore_rlb(self, saved):
        if saved is None:
            sys.modules.pop("record_onnx_light_benchmark", None)
        else:
            sys.modules["record_onnx_light_benchmark"] = saved

    def test_delegates_and_caps(self):
        fake = types.ModuleType("record_onnx_light_benchmark")
        fake.discover_node_tests = lambda kind: [
            {"name": f"m{i}"} for i in range(5)
        ]
        saved = self._patch_rlb(fake)
        try:
            tests = rce.discover_big_models(max_big_models=2)
        finally:
            self._restore_rlb(saved)
        self.assertEqual([t["name"] for t in tests], ["m0", "m1"])

    def test_returns_empty_on_error(self):
        fake = types.ModuleType("record_onnx_light_benchmark")

        def _raise(kind):
            raise ImportError("onnx_light missing")

        fake.discover_node_tests = _raise
        saved = self._patch_rlb(fake)
        try:
            tests = rce.discover_big_models()
        finally:
            self._restore_rlb(saved)
        self.assertEqual(tests, [])


class TestRunBigModels(unittest.TestCase):
    def test_skips_models_without_cpu_kernel(self):
        import numpy as np

        class _Node:
            def __init__(self, op_type):
                self.op_type = op_type

        class _Graph:
            def __init__(self, ops):
                self.node = [_Node(op) for op in ops]

        class _Model:
            def __init__(self, ops):
                self.graph = _Graph(ops)

        data_sets = [([np.zeros(4, dtype=np.float32)], [np.zeros(4, dtype=np.float32)])]
        tests = [
            {"name": "abs_benchmark", "model": _Model(["Abs"]), "data_sets": data_sets},
            {"name": "add_benchmark", "model": _Model(["Add"]), "data_sets": data_sets},
            {"name": "no_data", "model": _Model(["Abs"]), "data_sets": []},
        ]

        def fake_run_benchmark(model, ds, backend, n_warmup, n_measure):
            op = model.graph.node[0].op_type
            if backend == "onnx_light_cpu":
                if op == "Abs":
                    return {"success": True, "avg_ms": 1.0}
                return {"success": False, "error": "no onnx-light-cpu kernel ran"}
            return {"success": True, "avg_ms": 2.0}

        results = rce.run_big_models(tests, run_benchmark=fake_run_benchmark)
        self.assertEqual(len(results), 1)
        example = results[0]
        self.assertEqual(example["name"], "abs_benchmark")
        self.assertEqual(example["op"], "Abs")
        self.assertEqual(example["backends"], ["onnx_light_cpu", "onnxruntime"])
        row = example["rows"][0]
        # symbolic cost = 4 inputs + 4 outputs = 8 elements.
        self.assertEqual(row["size"], 8)
        self.assertEqual(row["onnx_light_cpu_ms"], 1.0)
        self.assertEqual(row["onnxruntime_ms"], 2.0)
        self.assertEqual(row["speedup_cpu"], 2.0)
        self.assertNotIn("numpy_ms", row)

    def test_missing_onnxruntime_still_reports_cpu(self):
        import numpy as np

        class _Model:
            class graph:  # noqa: N801
                node = [type("N", (), {"op_type": "Gemm"})()]

        data_sets = [([np.zeros((2, 2), dtype=np.float32)], [])]
        tests = [{"name": "gemm_benchmark", "model": _Model(), "data_sets": data_sets}]

        def fake_run_benchmark(model, ds, backend, n_warmup, n_measure):
            if backend == "onnx_light_cpu":
                return {"success": True, "avg_ms": 3.0}
            return {"success": False, "error": "unsupported"}

        results = rce.run_big_models(tests, run_benchmark=fake_run_benchmark)
        self.assertEqual(len(results), 1)
        row = results[0]["rows"][0]
        self.assertEqual(row["onnx_light_cpu_ms"], 3.0)
        self.assertNotIn("onnxruntime_ms", row)
        self.assertNotIn("speedup_cpu", row)


class TestBuildPayloadBigModels(unittest.TestCase):
    def test_big_models_appended_to_examples(self):
        def fake_run(examples, n_warmup, n_measure):
            rows = [rce._row_from_times(1, {"onnx_light_cpu": 1.0, "onnxruntime": 2.0})]
            return [{"name": "abs", "rows": rows, "summary": {}}], {}

        captured = {}

        def fake_discover(kind, max_big_models):
            captured["kind"] = kind
            captured["max"] = max_big_models
            return [{"name": "big"}]

        def fake_run_big(tests, n_warmup, n_measure):
            captured["tests"] = tests
            return [{"name": "gemm_benchmark", "rows": [], "summary": {}}]

        payload = rce.build_payload(
            run=fake_run,
            discover_big=fake_discover,
            run_big=fake_run_big,
            max_big_models=7,
            versions=lambda: {},
        )
        names = [e["name"] for e in payload["examples"]]
        self.assertEqual(names, ["abs", "gemm_benchmark"])
        self.assertEqual(captured["max"], 7)
        self.assertEqual(captured["kind"], "node")
        self.assertEqual(captured["tests"], [{"name": "big"}])

    def test_big_models_can_be_disabled(self):
        def fake_run(examples, n_warmup, n_measure):
            return [{"name": "abs", "rows": [], "summary": {}}], {}

        def fake_discover(kind, max_big_models):
            raise AssertionError("discover should not be called when disabled")

        payload = rce.build_payload(
            run=fake_run,
            include_big_models=False,
            discover_big=fake_discover,
            versions=lambda: {},
        )
        self.assertEqual([e["name"] for e in payload["examples"]], ["abs"])


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
        self.assertTrue(args.include_big_models)
        self.assertIsNone(args.max_big_models)

    def test_big_model_overrides(self):
        args = rce.parse_args(["--no-big-models", "--max-big-models", "3"])
        self.assertFalse(args.include_big_models)
        self.assertEqual(args.max_big_models, 3)

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
