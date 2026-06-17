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

    def test_snapshot_includes_unannotated_intermediates(self):
        """Node outputs without ``value_info`` must still appear in the
        snapshot as informational ``kind == "intermediate"`` entries so
        the detailed report can show what each backend inferred for
        them (e.g. the shape produced by a ``Shape`` operator)."""
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, ["N", 3]
        )
        out = helper.make_tensor_value_info(
            "Y", TensorProto.INT64, [2]
        )
        nodes = [
            helper.make_node("Shape", ["X"], ["shp"]),
            helper.make_node("Identity", ["shp"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "unannotated", [inp], [out])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        by_name = {s["name"]: s for s in snap}
        # The intermediate ``shp`` produced by the ``Shape`` node must
        # be present even though the model has no ``value_info`` for
        # it. It carries no expectation (``elem_type is None``) and is
        # ordered after its producer.
        self.assertIn("shp", by_name)
        self.assertEqual(by_name["shp"]["kind"], "intermediate")
        self.assertEqual(by_name["shp"]["op_type"], "Shape")
        self.assertIsNone(by_name["shp"]["elem_type"])
        self.assertFalse(by_name["shp"]["has_shape"])
        self.assertEqual(by_name["shp"]["shape"], [])
        # And ordering still follows node declaration order.
        self.assertEqual([s["name"] for s in snap], ["shp", "Y"])

    def test_compare_treats_intermediate_entries_as_informational(self):
        """Entries with ``elem_type is None`` are surfaced in
        ``details`` together with the inferred values but are never
        flagged as a mismatch."""
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, ["N", 3]
        )
        out = helper.make_tensor_value_info(
            "Y", TensorProto.INT64, [2]
        )
        nodes = [
            helper.make_node("Shape", ["X"], ["shp"]),
            helper.make_node("Identity", ["shp"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "unannotated", [inp], [out])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        import onnx.shape_inference

        inferred = onnx.shape_inference.infer_shapes(model)
        details = rsi._compare_snapshot_with_model(snap, inferred)
        by_name = {d["name"]: d for d in details}
        # The informational entry is marked ok and carries the
        # inferred type/shape so the dashboard can display them.
        self.assertTrue(by_name["shp"]["ok"])
        self.assertIsNone(by_name["shp"]["expected_elem_type"])
        self.assertFalse(by_name["shp"]["expected_has_shape"])
        self.assertEqual(by_name["shp"]["elem_type"], int(TensorProto.INT64))
        self.assertTrue(by_name["shp"]["has_shape"])

    def test_run_excludes_intermediate_entries_from_score(self):
        """``run_test_with_backend`` must not count informational
        intermediates in ``correct``/``total``: the score should reflect
        only entries that carry a real expectation."""
        from onnx import TensorProto, helper

        inp = helper.make_tensor_value_info(
            "X", TensorProto.FLOAT, ["N", 3]
        )
        out = helper.make_tensor_value_info(
            "Y", TensorProto.INT64, [2]
        )
        nodes = [
            helper.make_node("Shape", ["X"], ["shp"]),
            helper.make_node("Identity", ["shp"], ["Y"]),
        ]
        graph = helper.make_graph(nodes, "unannotated", [inp], [out])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7
        snap = rsi.snapshot_intermediates(model)
        info = rsi.run_test_with_backend(model, snap, "onnx")
        # ``snap`` contains 2 entries (informational ``shp`` + ``Y``)
        # but only ``Y`` is scored.
        self.assertEqual(len(snap), 2)
        self.assertEqual(info["total"], 1)
        self.assertEqual(info["correct"], 1)
        self.assertTrue(info["success"])
        # Both entries are still surfaced in ``details``.
        names = sorted(d["name"] for d in info["details"])
        self.assertEqual(names, ["Y", "shp"])

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

    def test_strip_shapes_keep_outputs_only_clears_value_info(self):
        model = _make_simple_model()
        stripped = rsi.strip_shapes(model, keep_outputs=True)
        # value_info shapes are stripped...
        for vi in stripped.graph.value_info:
            tt = vi.type.tensor_type
            self.assertEqual(tt.elem_type, 1)
            self.assertFalse(
                tt.HasField("shape"),
                f"value_info shape should be stripped on {vi.name!r}",
            )
        # ...but output shapes are preserved as a prefill hint.
        for vi in stripped.graph.output:
            self.assertTrue(
                vi.type.tensor_type.HasField("shape"),
                f"output shape should be preserved on {vi.name!r}",
            )
        # The original model is untouched.
        for vi in list(model.graph.output) + list(model.graph.value_info):
            self.assertTrue(vi.type.tensor_type.HasField("shape"))


def _make_model_with_subgraph():
    """Build a model whose ``If`` node carries non-trivial subgraphs.

    The ``then`` branch has an intermediate ``value_info`` (``tmid``) and
    an output (``tout``); the ``else`` branch only has an output
    (``eout``). Used to check that subgraph shapes are snapshotted,
    stripped and scored like the main graph's.
    """
    from onnx import TensorProto, helper

    th_mid = helper.make_tensor_value_info("tmid", TensorProto.FLOAT, [2, 3])
    th_out = helper.make_tensor_value_info("tout", TensorProto.FLOAT, [2, 3])
    then_g = helper.make_graph(
        [
            helper.make_node("Identity", ["X"], ["tmid"]),
            helper.make_node("Identity", ["tmid"], ["tout"]),
        ],
        "then",
        [],
        [th_out],
        value_info=[th_mid],
    )
    el_out = helper.make_tensor_value_info("eout", TensorProto.FLOAT, [2, 3])
    else_g = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["eout"])],
        "else",
        [],
        [el_out],
    )
    cond = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "If", ["cond"], ["Y"], then_branch=then_g, else_branch=else_g
    )
    graph = helper.make_graph([node], "main", [cond, X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 7
    return model


class TestSubgraphCoverage(unittest.TestCase):
    def test_snapshot_includes_subgraph_value_info_and_outputs(self):
        model = _make_model_with_subgraph()
        snap = rsi.snapshot_intermediates(model)
        by_name = {s["name"]: s for s in snap}
        # Subgraph intermediate ``value_info`` is snapshotted.
        self.assertIn("tmid", by_name)
        self.assertEqual(by_name["tmid"]["kind"], "value_info")
        self.assertEqual(by_name["tmid"]["shape"], [2, 3])
        # Subgraph outputs of both branches are snapshotted.
        self.assertIn("tout", by_name)
        self.assertEqual(by_name["tout"]["kind"], "output")
        self.assertIn("eout", by_name)
        self.assertEqual(by_name["eout"]["kind"], "output")
        # The main graph output is still present.
        self.assertIn("Y", by_name)

    def test_strip_shapes_clears_subgraph_shapes(self):
        model = _make_model_with_subgraph()
        stripped = rsi.strip_shapes(model)
        for graph in rsi._iter_subgraphs(stripped.graph):
            if graph.name in ("then", "else"):
                for vi in list(graph.value_info) + list(graph.output):
                    self.assertFalse(
                        vi.type.tensor_type.HasField("shape"),
                        f"subgraph shape should be stripped on {vi.name!r}",
                    )
        # The original model is untouched.
        for graph in rsi._iter_subgraphs(model.graph):
            for vi in list(graph.value_info) + list(graph.output):
                self.assertTrue(vi.type.tensor_type.HasField("shape"))

    def test_subgraph_shapes_are_scored_against_inferred_model(self):
        model = _make_model_with_subgraph()
        snap = rsi.snapshot_intermediates(model)
        stripped = rsi.strip_shapes(model)
        inferred = rsi._run_onnx(stripped)
        details = {
            d["name"]: d for d in rsi._compare_snapshot_with_model(snap, inferred)
        }
        for name in ("tmid", "tout", "eout"):
            self.assertIn(name, details)
            self.assertTrue(details[name]["ok"], details[name])
            self.assertEqual(details[name]["shape"], [2, 3])


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

    def test_symbolic_dim_name_ignores_whitespace(self):
        # ``"a + b"`` and ``"a+b"`` describe the same symbolic dim;
        # different shape-inference implementations format expression
        # dims with different spacing, so whitespace must be stripped
        # from both sides before comparing.
        snap, wrong = self._make_symbolic_model("a + b", "a+b")
        details = rsi._compare_snapshot_with_model(snap, wrong)
        by_name = {d["name"]: d for d in details}
        self.assertTrue(by_name["Y"]["ok"], by_name["Y"].get("reason"))

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

    def test_ort_transformers_backend_registered(self):
        # The symbolic shape inference shipped in ``onnxruntime.transformers``
        # is exposed as a dedicated backend in the coverage script.
        self.assertIn("ort-transformers", rsi.BACKENDS)
        self.assertIn("ort-transformers", rsi._BACKEND_RUNNERS)
        self.assertEqual(
            rsi.BACKEND_PACKAGE["ort-transformers"], "onnxruntime"
        )

    def test_yobx_backend_registered(self):
        # The shape inference shipped in ``yet-another-onnx-builder``
        # (``yobx.xshape.BasicShapeBuilder``) is exposed as a dedicated
        # backend in the coverage script.
        self.assertIn("yobx", rsi.BACKENDS)
        self.assertIn("yobx", rsi._BACKEND_RUNNERS)
        self.assertEqual(rsi.BACKEND_PACKAGE["yobx"], "yobx")

    def test_empty_expected(self):
        info = rsi.run_test_with_backend(None, [], "onnx")
        self.assertFalse(info["success"])
        self.assertEqual(info["error_step"], "load")

    def test_onnx_light_optim_keeps_output_shapes_as_prefill_hint(self):
        # ``onnx-light-optim`` opts into ``prefill_with_value_info_output``
        # so it must receive the graph output shapes as anchors. Capture
        # the model handed to the runner and assert the output shapes are
        # preserved while ``graph.value_info`` shapes are stripped.
        model = _make_simple_model()
        expected = rsi.snapshot_intermediates(model)
        captured = {}

        def fake_runner(stripped):
            captured["model"] = stripped
            return stripped

        original = rsi._BACKEND_RUNNERS["onnx-light-optim"]
        rsi._BACKEND_RUNNERS["onnx-light-optim"] = fake_runner
        try:
            rsi.run_test_with_backend(model, expected, "onnx-light-optim")
        finally:
            rsi._BACKEND_RUNNERS["onnx-light-optim"] = original

        stripped = captured["model"]
        for vi in stripped.graph.output:
            self.assertTrue(
                vi.type.tensor_type.HasField("shape"),
                f"output shape should be preserved on {vi.name!r}",
            )
        for vi in stripped.graph.value_info:
            self.assertFalse(
                vi.type.tensor_type.HasField("shape"),
                f"value_info shape should be stripped on {vi.name!r}",
            )


class TestDropShapelessValueInfo(unittest.TestCase):
    def test_drops_value_info_without_shape_keeps_shaped_ones(self):
        # ``strip_shapes(keep_outputs=True)`` leaves intermediate
        # ``value_info`` entries with an ``elem_type`` but no shape. The
        # ``onnx-light-optim`` prefill reads every ``value_info`` entry, so
        # those shapeless entries must be removed before inference to avoid
        # ``Optional field 'shape' has no value.``.
        from onnx import TensorProto, helper

        shaped = helper.make_tensor_value_info("kept", TensorProto.FLOAT, [2, 3])
        shapeless = helper.make_value_info(
            "stripped",
            helper.make_tensor_type_proto(TensorProto.FLOAT, shape=None),
        )
        inp = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2, 3])
        out = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2, 3])
        graph = helper.make_graph(
            [helper.make_node("Identity", ["X"], ["Z"])],
            "g",
            [inp],
            [out],
            value_info=[shaped, shapeless],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 7

        rsi._drop_shapeless_value_info(model)

        names = [vi.name for vi in model.graph.value_info]
        self.assertEqual(names, ["kept"])
        self.assertTrue(
            model.graph.value_info[0].type.tensor_type.HasField("shape")
        )


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
            ("model_a", "onnx-light-optim"): {"success": True, "correct": 1, "total": 1},
            ("model_a", "onnx"): {"success": True, "correct": 1, "total": 1},
            ("model_a", "onnx-shape-inference"): {
                "success": False, "correct": 0, "total": 1, "error": "x", "error_step": "run",
            },
            ("model_a", "ort-transformers"): {
                "success": True, "correct": 1, "total": 1,
            },
            ("model_b", "onnx-light"): {"success": False, "correct": 1, "total": 2, "error": "1/2"},
            ("model_b", "onnx-light-optim"): {"success": True, "correct": 2, "total": 2},
            ("model_b", "onnx"): {"success": True, "correct": 2, "total": 2},
            ("model_b", "onnx-shape-inference"): {
                "success": False, "correct": 0, "total": 2, "error": "x", "error_step": "run",
            },
            ("model_b", "ort-transformers"): {
                "success": False, "correct": 1, "total": 2, "error": "1/2",
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
        self.assertEqual(payload["totals"]["onnx-light-optim"], {
            "correct": 3, "total": 3, "tests_pass": 2, "tests_fail": 0,
        })
        self.assertEqual(payload["totals"]["onnx-shape-inference"], {
            "correct": 0, "total": 3, "tests_pass": 0, "tests_fail": 2,
        })
        self.assertEqual(payload["totals"]["ort-transformers"], {
            "correct": 2, "total": 3, "tests_pass": 1, "tests_fail": 1,
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
    def test_default_tag_includes_shape_local_function_and_inference(self):
        self.assertEqual(rsi.DEFAULT_TAGS, ("shape", "local_function", "inference"))
        self.assertEqual(
            rsi._normalize_tags(rsi.DEFAULT_TAG),
            ("shape", "local_function", "inference"),
        )

    def test_normalize_tags_accepts_various_shapes(self):
        self.assertEqual(rsi._normalize_tags(None), ())
        self.assertEqual(rsi._normalize_tags(""), ())
        self.assertEqual(rsi._normalize_tags("shape"), ("shape",))
        self.assertEqual(rsi._normalize_tags("inference"), ("inference",))
        self.assertEqual(
            rsi._normalize_tags("shape, local_function"),
            ("shape", "local_function"),
        )
        self.assertEqual(
            rsi._normalize_tags(["shape", "local_function"]),
            ("shape", "local_function"),
        )
        self.assertEqual(
            rsi._normalize_tags(("shape,local_function", "extra")),
            ("shape", "local_function", "extra"),
        )

    def test_discover_inference_tests_filters_multiple_tags(self):
        class Case:
            def __init__(self, name, tag, model):
                self.name = name
                self.tag = tag
                self.model = model

        cases = {
            "a": Case("a", "shape", "model_a"),
            "b": Case("b", "local_function", "model_b"),
            "c": Case("c", "other", "model_c"),
            "d": Case("d", ("misc", "local_function"), "model_d"),
            "e": Case("e", "misc, inference", "model_e"),
        }

        import types

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda: cases
        parent_pkg = types.ModuleType("onnx_light.onnx_lib.backend.test")
        parent_pkg.case = fake_module
        backend_pkg = types.ModuleType("onnx_light.onnx_lib.backend")
        backend_pkg.test = parent_pkg
        onnx_lib_pkg = types.ModuleType("onnx_light.onnx_lib")
        onnx_lib_pkg.backend = backend_pkg
        root_pkg = types.ModuleType("onnx_light")
        root_pkg.onnx_lib = onnx_lib_pkg

        saved = {}
        for name in (
            "onnx_light",
            "onnx_light.onnx_lib",
            "onnx_light.onnx_lib.backend",
            "onnx_light.onnx_lib.backend.test",
            "onnx_light.onnx_lib.backend.test.case",
        ):
            saved[name] = sys.modules.get(name)
        sys.modules["onnx_light"] = root_pkg
        sys.modules["onnx_light.onnx_lib"] = onnx_lib_pkg
        sys.modules["onnx_light.onnx_lib.backend"] = backend_pkg
        sys.modules["onnx_light.onnx_lib.backend.test"] = parent_pkg
        sys.modules["onnx_light.onnx_lib.backend.test.case"] = fake_module

        original_to_onnx = rsi._onnx_light_model_to_onnx
        original_snapshot = rsi.snapshot_intermediates
        original_snapshot_inputs = rsi.snapshot_inputs
        original_mermaid = rsi.model_to_mermaid

        rsi._onnx_light_model_to_onnx = lambda m: m
        rsi.snapshot_intermediates = lambda m: [{"name": "Y"}]
        rsi.snapshot_inputs = lambda m: []
        rsi.model_to_mermaid = lambda m: ""

        try:
            discovered = rsi.discover_inference_tests("shape,local_function")
            self.assertEqual([d["name"] for d in discovered], ["a", "b", "d"])

            discovered_single = rsi.discover_inference_tests("shape")
            self.assertEqual([d["name"] for d in discovered_single], ["a"])

            discovered_list = rsi.discover_inference_tests(
                ["local_function", "other"]
            )
            self.assertEqual([d["name"] for d in discovered_list], ["b", "c", "d"])
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
            tag=["shape", "local_function"],
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(payload["tag"], "shape, local_function")

    def test_build_payload_records_backend_versions(self):
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
            tag="inference",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {
                "onnx": "1.17.0",
                "onnx_light": "0.2",
                "onnxruntime": "1.20.0",
            },
        )
        self.assertEqual(
            payload["backend_versions"],
            {
                "onnx-light": "0.2",
                "onnx-light-optim": "0.2",
                "onnx": "1.17.0",
                "ort-transformers": "1.20.0",
            },
        )

    def test_backend_versions_from_map(self):
        self.assertEqual(rsi.backend_versions_from_map({}), {})
        self.assertEqual(
            rsi.backend_versions_from_map({"yobx": "3.1", "onnx": "1.18.0"}),
            {"onnx": "1.18.0", "yobx": "3.1"},
        )


if __name__ == "__main__":
    unittest.main()
