"""Tests for ``scripts.record_onnx_shape_inference_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_shape_inference_coverage as rsi  # noqa: E402


def _make_simple_model():
    """Build a tiny ``onnx.ModelProto`` exercising shape inference.

    The model contains a single ``Identity`` node so the official ONNX
    shape inference is enough to recover both the intermediate
    ``value_info`` and the graph output shape from the input shape.
    """
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    mid = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
    nodes = [
        helper.make_node("Identity", ["X"], ["Y"]),
        helper.make_node("Identity", ["Y"], ["Z"]),
    ]
    graph = helper.make_graph(nodes, "simple", [inp], [out], value_info=[mid])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 7
    return model


class TestSnapshotAndStrip(unittest.TestCase):
    def test_snapshot_records_producing_op_type(self):
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        mid = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        nodes = [
            helper.make_node("Relu", ["X"], ["Y"]),
            helper.make_node("Identity", ["Y"], ["Z"]),
        ]
        graph = helper.make_graph(nodes, "ops", [inp], [out], value_info=[mid])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        op_types = {s["name"]: s["op_type"] for s in snap}
        self.assertEqual(op_types, {"Y": "Relu", "Z": "Identity"})

    def test_snapshot_captures_outputs_and_value_info(self):
        model = _make_simple_model()
        snap = rsi.snapshot_intermediates(model)
        names = sorted((s["name"], s["kind"]) for s in snap)
        self.assertEqual(names, [("Y", "value_info"), ("Z", "output")])
        for entry in snap:
            self.assertTrue(entry["has_shape"])
            self.assertEqual(entry["shape"], [2, 3])
            self.assertEqual(entry["elem_type"], 1)  # FLOAT

    def test_snapshot_orders_entries_by_node_order(self):
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
        b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
        c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])
        out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        # Declare nodes in computation order but value_info in a
        # different order to ensure snapshot follows node order.
        nodes = [
            helper.make_node("Identity", ["X"], ["A"]),
            helper.make_node("Identity", ["A"], ["B"]),
            helper.make_node("Identity", ["B"], ["C"]),
            helper.make_node("Identity", ["C"], ["Z"]),
        ]
        graph = helper.make_graph(
            nodes, "ordered", [inp], [out], value_info=[c, a, b]
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        self.assertEqual(
            [(s["name"], s["kind"]) for s in snap],
            [
                ("A", "value_info"),
                ("B", "value_info"),
                ("C", "value_info"),
                ("Z", "output"),
            ],
        )

    def test_snapshot_preserves_symbolic_dim_names(self):
        import onnx
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])
        out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"])],
            "sym",
            [inp],
            [out],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        by_name = {s["name"]: s for s in snap}
        self.assertEqual(by_name["Y"]["shape"], ["N", 3])
        # Named symbolic dims must match themselves (and not be flagged
        # as a mismatch) when scoring inferred shapes.
        details = rsi._compare_snapshot_with_model(snap, model)
        self.assertTrue(all(d["ok"] for d in details), details)

    def test_snapshot_inputs_captures_graph_inputs(self):
        model = _make_simple_model()
        inputs = rsi.snapshot_inputs(model)
        self.assertEqual([(i["name"], i["kind"]) for i in inputs], [("X", "input")])
        entry = inputs[0]
        self.assertTrue(entry["has_shape"])
        self.assertEqual(entry["shape"], [2, 3])
        self.assertEqual(entry["elem_type"], 1)  # FLOAT

    def test_strip_shapes_clears_value_info_and_output_shapes(self):
        model = _make_simple_model()
        stripped = rsi.strip_shapes(model)
        for vi in list(stripped.graph.output) + list(stripped.graph.value_info):
            tt = vi.type.tensor_type
            self.assertEqual(tt.elem_type, 1)
            self.assertFalse(
                tt.HasField("shape"),
                f"shape should be stripped on {vi.name!r}",
            )
        # The original model is untouched.
        for vi in list(model.graph.output) + list(model.graph.value_info):
            self.assertTrue(vi.type.tensor_type.HasField("shape"))


class TestCompareSnapshotWithModel(unittest.TestCase):
    def test_matching_shapes_are_scored_ok(self):
        model = _make_simple_model()
        snap = rsi.snapshot_intermediates(model)
        details = rsi._compare_snapshot_with_model(snap, model)
        self.assertEqual(len(details), len(snap))
        self.assertTrue(all(d["ok"] for d in details), details)

    def test_missing_shape_is_flagged_when_expected(self):
        model = _make_simple_model()
        snap = rsi.snapshot_intermediates(model)
        stripped = rsi.strip_shapes(model)
        details = rsi._compare_snapshot_with_model(snap, stripped)
        # Every intermediate had a shape originally; stripped model has
        # none, so all entries must be flagged as not ok.
        self.assertFalse(any(d["ok"] for d in details), details)
        for d in details:
            self.assertIn("no shape inferred", d["reason"])

    def test_dim_mismatch_is_flagged(self):
        model = _make_simple_model()
        snap = rsi.snapshot_intermediates(model)
        # Tamper with the output shape to simulate a wrong inference.
        wrong = type(model)()
        wrong.CopyFrom(model)
        wrong.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 99
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertFalse(by_name["Z"]["ok"])
        self.assertIn("dim[1]", by_name["Z"]["reason"])
        self.assertTrue(by_name["Y"]["ok"])

    def test_missing_value_info_is_flagged(self):
        model = _make_simple_model()
        snap = rsi.snapshot_intermediates(model)
        # Drop the intermediate value_info entry entirely.
        wrong = type(model)()
        wrong.CopyFrom(model)
        while len(wrong.graph.value_info):
            wrong.graph.value_info.pop()
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertFalse(by_name["Y"]["ok"])
        self.assertIn("missing from graph", by_name["Y"]["reason"])

    def _make_symbolic_model(self, exp_name, got_name):
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [exp_name, 3])
        out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [exp_name, 3])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"])],
            "sym",
            [inp],
            [out],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        # Rewrite the inferred dim_param to ``got_name`` (or to a concrete
        # value when an int is given) to simulate a different inference
        # result for the dynamic dimension.
        wrong = type(model)()
        wrong.CopyFrom(model)
        out_dim0 = wrong.graph.output[0].type.tensor_type.shape.dim[0]
        out_dim0.Clear()
        if isinstance(got_name, str):
            out_dim0.dim_param = got_name
        else:
            out_dim0.dim_value = got_name
        return snap, wrong

    def test_symbolic_dim_name_mismatch_is_flagged(self):
        snap, wrong = self._make_symbolic_model("N", "M")
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertFalse(by_name["Y"]["ok"])
        self.assertIn("dim[0]", by_name["Y"]["reason"])

    def test_symbolic_vs_concrete_dim_is_flagged(self):
        snap, wrong = self._make_symbolic_model("N", 4)
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertFalse(by_name["Y"]["ok"])
        self.assertIn("dim[0]", by_name["Y"]["reason"])

    def test_concrete_vs_symbolic_dim_is_flagged(self):
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        out = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Y"])],
            "g",
            [inp],
            [out],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        wrong = type(model)()
        wrong.CopyFrom(model)
        d0 = wrong.graph.output[0].type.tensor_type.shape.dim[0]
        d0.Clear()
        d0.dim_param = "N"
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertFalse(by_name["Y"]["ok"])
        self.assertIn("dim[0]", by_name["Y"]["reason"])


class TestRunTestWithBackend(unittest.TestCase):
    def test_run_with_official_onnx(self):
        model = _make_simple_model()
        expected = rsi.snapshot_intermediates(model)
        info = rsi.run_test_with_backend(model, expected, "onnx")
        self.assertTrue(info["success"], info)
        self.assertEqual(info["correct"], info["total"])
        self.assertEqual(info["correct"], len(expected))

    def test_unknown_backend(self):
        info = rsi.run_test_with_backend(None, [{}], "unknown")
        self.assertFalse(info["success"])
        self.assertEqual(info["error_step"], "load")

    def test_empty_expected(self):
        info = rsi.run_test_with_backend(None, [], "onnx")
        self.assertFalse(info["success"])
        self.assertEqual(info["error_step"], "load")


class TestRowFromResults(unittest.TestCase):
    def test_records_per_runtime_counts_and_last_pass(self):
        expected = [
            {
                "name": "Y",
                "kind": "value_info",
                "elem_type": 1,
                "has_shape": True,
                "shape": [2, 3],
            },
        ]
        results = {
            "onnx-light": {
                "success": True,
                "error": "",
                "error_step": "",
                "correct": 1,
                "total": 1,
                "details": [{"name": "Y", "ok": True}],
            },
            "onnx": {
                "success": False,
                "error": "boom",
                "error_step": "run",
                "correct": 0,
                "total": 1,
                "details": [],
            },
            "onnx-shape-inference": {
                "success": False,
                "error": "0/1 matched",
                "error_step": "compare",
                "correct": 0,
                "total": 1,
                "details": [{"name": "Y", "ok": False}],
            },
        }
        row = rsi._row_from_results(
            "test_a",
            expected,
            results,
            versions={
                "onnx_light": "0.1",
                "onnx": "1.17",
                "onnx_shape_inference": "0.0.1",
            },
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertEqual(row["name"], "test_a")
        self.assertEqual(row["expected"][0]["name"], "Y")
        self.assertTrue(row["runtimes"]["onnx-light"]["success"])
        self.assertEqual(
            row["runtimes"]["onnx-light"]["last_pass_date"], "2024-05-06T07:08:09Z"
        )
        self.assertEqual(row["runtimes"]["onnx-light"]["last_pass_version"], "0.1")
        self.assertFalse(row["runtimes"]["onnx"]["success"])
        self.assertEqual(row["runtimes"]["onnx"]["error"], "boom")
        self.assertNotIn("last_pass_date", row["runtimes"]["onnx"])

    def test_includes_inputs_when_provided(self):
        expected = [
            {
                "name": "Y",
                "kind": "value_info",
                "elem_type": 1,
                "has_shape": True,
                "shape": [2, 3],
            },
        ]
        inputs = [
            {
                "name": "X",
                "kind": "input",
                "elem_type": 1,
                "has_shape": True,
                "shape": [2, 3],
            },
        ]
        results = {
            backend: {
                "success": True,
                "error": "",
                "error_step": "",
                "correct": 1,
                "total": 1,
                "details": [],
            }
            for backend in rsi.BACKENDS
        }
        row = rsi._row_from_results(
            "test_a",
            expected,
            results,
            inputs=inputs,
        )
        self.assertEqual(len(row["inputs"]), 1)
        self.assertEqual(row["inputs"][0]["name"], "X")
        self.assertEqual(row["inputs"][0]["kind"], "input")
        self.assertEqual(row["inputs"][0]["shape"], [2, 3])

    def test_inputs_default_to_empty_list(self):
        row = rsi._row_from_results("t", [], {})
        self.assertEqual(row["inputs"], [])

    def test_carries_over_previous_last_pass_on_failure(self):
        expected = []
        results = {
            backend: {
                "success": False,
                "error": "x",
                "error_step": "run",
                "correct": 0,
                "total": 0,
                "details": [],
            }
            for backend in rsi.BACKENDS
        }
        previous = {
            "name": "test_a",
            "runtimes": {
                "onnx": {
                    "last_pass_date": "2024-01-02T03:04:05Z",
                    "last_pass_version": "1.16.0",
                }
            },
        }
        row = rsi._row_from_results(
            "test_a",
            expected,
            results,
            previous=previous,
            versions={"onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertEqual(
            row["runtimes"]["onnx"]["last_pass_date"], "2024-01-02T03:04:05Z"
        )
        self.assertEqual(
            row["runtimes"]["onnx"]["last_pass_version"], "1.16.0"
        )
        self.assertNotIn("last_pass_date", row["runtimes"]["onnx-light"])


class TestBuildPayload(unittest.TestCase):
    def test_aggregates_totals_across_backends(self):
        tests = [
            {"name": "test_a", "model": "model_a", "expected": [{"name": "Y"}]},
            {"name": "test_b", "model": "model_b", "expected": [{"name": "Y"}, {"name": "Z"}]},
        ]
        outcomes = {
            ("model_a", "onnx-light"): {"success": True, "correct": 1, "total": 1},
            ("model_a", "onnx-light-onnx-optim"): {"success": True, "correct": 1, "total": 1},
            ("model_a", "onnx"): {"success": True, "correct": 1, "total": 1},
            ("model_a", "onnx-shape-inference"): {
                "success": False, "correct": 0, "total": 1, "error": "x", "error_step": "run",
            },
            ("model_b", "onnx-light"): {"success": False, "correct": 1, "total": 2, "error": "1/2"},
            ("model_b", "onnx-light-onnx-optim"): {"success": True, "correct": 2, "total": 2},
            ("model_b", "onnx"): {"success": True, "correct": 2, "total": 2},
            ("model_b", "onnx-shape-inference"): {
                "success": False, "correct": 0, "total": 2, "error": "x", "error_step": "run",
            },
        }

        def fake_run(model, expected, backend):
            base = {"correct": 0, "total": len(expected), "details": [], "error": "", "error_step": ""}
            base.update(outcomes[(model, backend)])
            return base

        payload = rsi.build_payload(
            tag="inference",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {"onnx": "1.0"},
        )
        self.assertEqual(payload["tag"], "inference")
        self.assertEqual(payload["totals"]["onnx"], {
            "correct": 3, "total": 3, "tests_pass": 2, "tests_fail": 0,
        })
        self.assertEqual(payload["totals"]["onnx-light"], {
            "correct": 2, "total": 3, "tests_pass": 1, "tests_fail": 1,
        })
        self.assertEqual(payload["totals"]["onnx-light-onnx-optim"], {
            "correct": 3, "total": 3, "tests_pass": 2, "tests_fail": 0,
        })
        self.assertEqual(payload["totals"]["onnx-shape-inference"], {
            "correct": 0, "total": 3, "tests_pass": 0, "tests_fail": 2,
        })
        names = [r["name"] for r in payload["tests"]]
        self.assertEqual(names, ["test_a", "test_b"])

    def test_limit_caps_tests(self):
        tests = [
            {"name": f"t{i}", "model": f"m{i}", "expected": [{"name": "Y"}]}
            for i in range(5)
        ]

        def fake_run(model, expected, backend):
            return {"success": True, "correct": 1, "total": 1, "details": [], "error": "", "error_step": ""}

        payload = rsi.build_payload(
            tag="inference",
            limit=2,
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(len(payload["tests"]), 2)

    def test_captures_unhandled_runner_exception(self):
        tests = [{"name": "boom", "model": "m", "expected": [{"name": "Y"}]}]

        def fake_run(model, expected, backend):
            raise RuntimeError("kaboom")

        payload = rsi.build_payload(
            tag="inference",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        for backend in rsi.BACKENDS:
            self.assertFalse(row["runtimes"][backend]["success"])
            self.assertEqual(row["runtimes"][backend]["error"], "kaboom")


class TestMermaid(unittest.TestCase):
    def test_model_to_mermaid_returns_flowchart(self):
        model = _make_simple_model()
        out = rsi.model_to_mermaid(model)
        self.assertIsInstance(out, str)
        # The helper is now self-contained (only depends on ``onnx``),
        # so it must always return a non-empty Mermaid ``flowchart TD``
        # block for a valid model.
        self.assertTrue(out.startswith("flowchart TD"))
        # Inputs, the two Identity nodes and the output all appear.
        self.assertIn("X", out)
        self.assertIn("Identity", out)
        self.assertIn("Z", out)

    def test_model_to_mermaid_returns_empty_on_invalid_model(self):
        # Non-model inputs are tolerated and produce an empty string.
        self.assertEqual(rsi.model_to_mermaid(None), "")
        self.assertEqual(rsi.model_to_mermaid("not a model"), "")

    def test_model_to_mermaid_escapes_quotes_in_names(self):
        import onnx
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info('X"weird', TensorProto.FLOAT, [1])
        out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1])
        graph = helper.make_graph(
            [helper.make_node("Identity", ['X"weird'], ["Z"])],
            "weird",
            [inp],
            [out],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 7
        rendered = rsi.model_to_mermaid(model)
        # The double-quote is escaped so it cannot terminate the Mermaid label.
        self.assertNotIn('X"weird"', rendered)
        self.assertIn("&quot;weird", rendered)

    def test_row_includes_mermaid_when_provided(self):
        row = rsi._row_from_results(
            "t",
            [],
            {b: {"success": True, "correct": 0, "total": 0, "details": []} for b in rsi.BACKENDS},
            mermaid="flowchart TD\nA-->B",
        )
        self.assertEqual(row["mermaid"], "flowchart TD\nA-->B")

    def test_row_preserves_previous_mermaid_when_missing(self):
        previous = {"mermaid": "flowchart TD\nX-->Y"}
        row = rsi._row_from_results(
            "t",
            [],
            {b: {"success": True, "correct": 0, "total": 0, "details": []} for b in rsi.BACKENDS},
            previous=previous,
            mermaid="",
        )
        self.assertEqual(row["mermaid"], "flowchart TD\nX-->Y")

    def test_row_omits_mermaid_when_absent(self):
        row = rsi._row_from_results(
            "t",
            [],
            {b: {"success": True, "correct": 0, "total": 0, "details": []} for b in rsi.BACKENDS},
            mermaid="",
        )
        self.assertNotIn("mermaid", row)

    def test_build_payload_propagates_mermaid(self):
        tests = [
            {
                "name": "test_a",
                "model": "m",
                "expected": [{"name": "Y"}],
                "mermaid": "flowchart TD\nA-->B",
            }
        ]

        def fake_run(model, expected, backend):
            return {
                "success": True, "correct": 1, "total": 1,
                "details": [], "error": "", "error_step": "",
            }

        payload = rsi.build_payload(
            tag="inference",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(payload["tests"][0]["mermaid"], "flowchart TD\nA-->B")


class TestMain(unittest.TestCase):
    def test_main_writes_payload(self):
        original_build = rsi.build_payload

        def fake_build(**kwargs):
            return {
                "date": "2024-01-01T00:00:00Z",
                "tag": kwargs.get("tag", "inference"),
                "versions": {},
                "totals": {b: {"correct": 0, "total": 0, "tests_pass": 0, "tests_fail": 0} for b in rsi.BACKENDS},
                "tests": [],
            }

        try:
            rsi.build_payload = fake_build
            with tempfile.TemporaryDirectory() as tmp:
                code = rsi.main(["--cache-dir", tmp])
                self.assertEqual(code, 0)
                p = os.path.join(tmp, "onnx-light", "shape_inference_coverage.json")
                self.assertTrue(os.path.exists(p))
                with open(p) as fh:
                    data = json.load(fh)
                self.assertEqual(data["tag"], rsi.DEFAULT_TAG)
        finally:
            rsi.build_payload = original_build

    def test_main_returns_error_on_exception(self):
        original_build = rsi.build_payload

        def fake_build(**kwargs):
            raise RuntimeError("nope")

        try:
            rsi.build_payload = fake_build
            with tempfile.TemporaryDirectory() as tmp:
                code = rsi.main(["--cache-dir", tmp])
                self.assertEqual(code, 1)
                self.assertFalse(
                    os.path.exists(
                        os.path.join(tmp, "onnx-light", "shape_inference_coverage.json")
                    )
                )
        finally:
            rsi.build_payload = original_build


class TestTagFiltering(unittest.TestCase):
    def test_default_tag_includes_inference_and_local_function(self):
        self.assertEqual(rsi.DEFAULT_TAGS, ("inference", "local_function"))
        self.assertEqual(
            rsi._normalize_tags(rsi.DEFAULT_TAG),
            ("inference", "local_function"),
        )

    def test_normalize_tags_accepts_various_shapes(self):
        self.assertEqual(rsi._normalize_tags(None), ())
        self.assertEqual(rsi._normalize_tags(""), ())
        self.assertEqual(rsi._normalize_tags("inference"), ("inference",))
        self.assertEqual(
            rsi._normalize_tags("inference, local_function"),
            ("inference", "local_function"),
        )
        self.assertEqual(
            rsi._normalize_tags(["inference", "local_function"]),
            ("inference", "local_function"),
        )
        self.assertEqual(
            rsi._normalize_tags(("inference,local_function", "extra")),
            ("inference", "local_function", "extra"),
        )

    def test_discover_inference_tests_filters_multiple_tags(self):
        class Case:
            def __init__(self, name, tag, model):
                self.name = name
                self.tag = tag
                self.model = model

        cases = {
            "a": Case("a", "inference", "model_a"),
            "b": Case("b", "local_function", "model_b"),
            "c": Case("c", "other", "model_c"),
        }

        import types

        fake_module = types.ModuleType("onnx_light.backend.test.case")
        fake_module.collect_test_case = lambda: cases
        parent_pkg = types.ModuleType("onnx_light.backend.test")
        parent_pkg.case = fake_module
        backend_pkg = types.ModuleType("onnx_light.backend")
        backend_pkg.test = parent_pkg
        root_pkg = types.ModuleType("onnx_light")
        root_pkg.backend = backend_pkg

        saved = {}
        for name in (
            "onnx_light",
            "onnx_light.backend",
            "onnx_light.backend.test",
            "onnx_light.backend.test.case",
        ):
            saved[name] = sys.modules.get(name)
        sys.modules["onnx_light"] = root_pkg
        sys.modules["onnx_light.backend"] = backend_pkg
        sys.modules["onnx_light.backend.test"] = parent_pkg
        sys.modules["onnx_light.backend.test.case"] = fake_module

        original_to_onnx = rsi._onnx_light_model_to_onnx
        original_snapshot = rsi.snapshot_intermediates
        original_snapshot_inputs = rsi.snapshot_inputs
        original_mermaid = rsi.model_to_mermaid

        rsi._onnx_light_model_to_onnx = lambda m: m
        rsi.snapshot_intermediates = lambda m: [{"name": "Y"}]
        rsi.snapshot_inputs = lambda m: []
        rsi.model_to_mermaid = lambda m: ""

        try:
            discovered = rsi.discover_inference_tests("inference,local_function")
            self.assertEqual([d["name"] for d in discovered], ["a", "b"])

            discovered_single = rsi.discover_inference_tests("inference")
            self.assertEqual([d["name"] for d in discovered_single], ["a"])

            discovered_list = rsi.discover_inference_tests(
                ["local_function", "other"]
            )
            self.assertEqual([d["name"] for d in discovered_list], ["b", "c"])
        finally:
            rsi._onnx_light_model_to_onnx = original_to_onnx
            rsi.snapshot_intermediates = original_snapshot
            rsi.snapshot_inputs = original_snapshot_inputs
            rsi.model_to_mermaid = original_mermaid
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_build_payload_records_joined_tag(self):
        tests = [{"name": "t", "model": "m", "expected": [{"name": "Y"}]}]

        def fake_run(model, expected, backend):
            return {
                "success": True,
                "correct": 1,
                "total": 1,
                "details": [],
                "error": "",
                "error_step": "",
            }

        payload = rsi.build_payload(
            tag=["inference", "local_function"],
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(payload["tag"], "inference, local_function")


if __name__ == "__main__":
    unittest.main()
