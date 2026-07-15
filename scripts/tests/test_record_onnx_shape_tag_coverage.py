"""Tests for ``scripts.record_onnx_shape_tag_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_shape_tag_coverage as stc  # noqa: E402


class _FakeMeta:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class _FakeNode:
    def __init__(self, op_type, metadata=None):
        self.op_type = op_type
        self.metadata_props = [_FakeMeta(k, v) for k, v in (metadata or {}).items()]
        self.input = []
        self.output = []


class _FakeValueInfo:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata_props = [_FakeMeta(k, v) for k, v in (metadata or {}).items()]


class _FakeTensorProto:
    def __init__(self, name, metadata=None):
        self.name = name
        self.metadata_props = [_FakeMeta(k, v) for k, v in (metadata or {}).items()]


class _FakeGraph:
    def __init__(self, nodes, inputs=None, outputs=None, initializers=None, value_info=None):
        self.node = nodes
        self.input = inputs or []
        self.output = outputs or []
        self.initializer = initializers or []
        self.value_info = value_info or []


class _FakeModel:
    def __init__(self, nodes, inputs=None, outputs=None, initializers=None, value_info=None):
        self.graph = _FakeGraph(
            nodes, inputs=inputs, outputs=outputs, initializers=initializers, value_info=value_info
        )


class _FakeTestCase:
    def __init__(self, name, model, tag=""):
        self.name = name
        self.model = model
        self.tag = tag


class TestRecordOnnxShapeTagCoverage(unittest.TestCase):
    def test_node_metadata_filters_unrelated_keys(self):
        node = _FakeNode(
            "Shape",
            {
                "onnx_light.node_tag": "shape",
                "onnx_light.value_tags": "shape",
                "onnx_light.inplace_reuse": "0:0:equal",
                "ignored": "x",
            },
        )
        self.assertEqual(
            stc._node_metadata(node),
            {
                "onnx_light.node_tag": "shape",
                "onnx_light.value_tags": "shape",
            },
        )

    def test_value_metadata_filters_unrelated_keys(self):
        vi = _FakeValueInfo(
            "X",
            {
                "onnx_light.value_tags": "shape",
                "onnx_light.node_tag": "something",
                "ignored": "x",
            },
        )
        self.assertEqual(
            stc._value_metadata(vi),
            {"onnx_light.value_tags": "shape"},
        )

    def test_graph_value_snapshot_collects_inputs_outputs_initializers(self):
        inp = _FakeValueInfo("X", {"onnx_light.value_tags": "weight"})
        out = _FakeValueInfo("Y", {})
        init = _FakeTensorProto("W", {"onnx_light.value_tags": "weight"})
        val = _FakeValueInfo("Z", {"onnx_light.value_tags": "shape"})
        model = _FakeModel([], inputs=[inp], outputs=[out], initializers=[init], value_info=[val])
        snapshot = stc._graph_value_snapshot(model)
        names = [s["name"] for s in snapshot]
        self.assertIn("X", names)
        self.assertIn("Y", names)
        self.assertIn("W", names)
        self.assertIn("Z", names)
        x_entry = next(s for s in snapshot if s["name"] == "X")
        self.assertEqual(x_entry["kind"], "input")
        self.assertEqual(x_entry["metadata"], {"onnx_light.value_tags": "weight"})
        y_entry = next(s for s in snapshot if s["name"] == "Y")
        self.assertEqual(y_entry["kind"], "output")
        self.assertEqual(y_entry["metadata"], {})
        w_entry = next(s for s in snapshot if s["name"] == "W")
        self.assertEqual(w_entry["kind"], "initializer")
        self.assertEqual(w_entry["metadata"], {"onnx_light.value_tags": "weight"})
        z_entry = next(s for s in snapshot if s["name"] == "Z")
        self.assertEqual(z_entry["kind"], "result")
        self.assertEqual(z_entry["metadata"], {"onnx_light.value_tags": "shape"})

    def test_graph_value_snapshot_excludes_initializer_from_inputs(self):
        inp = _FakeValueInfo("X")
        init = _FakeTensorProto("W")
        # When the same name appears in both input and initializer, it should not
        # show up as an "input" kind (only as "initializer").
        class FakeGraph:
            node = []
            input = [_FakeValueInfo("W"), inp]
            output = []
            initializer = [init]
            value_info = []
        class FakeModel:
            graph = FakeGraph()
        snapshot = stc._graph_value_snapshot(FakeModel())
        kinds = {s["name"]: s["kind"] for s in snapshot}
        self.assertEqual(kinds.get("W"), "initializer")
        self.assertEqual(kinds.get("X"), "input")

    def test_clear_node_metadata_removes_entries(self):
        node = _FakeNode("Shape", {"onnx_light.node_tag": "shape"})
        stc._clear_node_metadata(node)
        self.assertEqual(node.metadata_props, [])

    def test_score_test_counts_nodes_and_metadata(self):
        row = stc._score_test(
            "test_cc_shape_tag_shape_reshape",
            expected_nodes=[
                {},
                {"onnx_light.node_tag": "shape"},
                {"onnx_light.node_tag": "shape", "onnx_light.value_tags": "shape"},
            ],
            actual_nodes=[
                {},
                {"onnx_light.node_tag": "shape"},
                {"onnx_light.node_tag": "axes", "onnx_light.value_tags": "shape"},
            ],
            node_ops=["Abs", "Shape", "Reshape"],
        )
        self.assertFalse(row["success"])
        self.assertEqual(row["matched_nodes"], 2)
        self.assertEqual(row["total_nodes"], 3)
        self.assertEqual(row["matched_metadata"], 2)
        self.assertEqual(row["total_metadata"], 3)
        self.assertEqual(row["matched_values"], 0)
        self.assertEqual(row["total_values"], 0)
        self.assertEqual(row["nodes"][2]["op_type"], "Reshape")
        self.assertNotIn("mermaid", row)
        self.assertIn("values", row)
        self.assertEqual(row["values"], [])

    def test_score_test_includes_values_section(self):
        expected_values = [
            {"name": "X", "kind": "input", "metadata": {"onnx_light.value_tags": "shape"}},
            {"name": "Y", "kind": "output", "metadata": {}},
        ]
        actual_values = [
            {"name": "X", "kind": "input", "metadata": {"onnx_light.value_tags": "shape"}},
            {"name": "Y", "kind": "output", "metadata": {"onnx_light.value_tags": "axes"}},
        ]
        row = stc._score_test(
            "test_values",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=expected_values,
            actual_values=actual_values,
        )
        self.assertIn("values", row)
        self.assertEqual(len(row["values"]), 2)
        x_val = next(v for v in row["values"] if v["name"] == "X")
        self.assertEqual(x_val["kind"], "input")
        self.assertTrue(x_val["success"])
        y_val = next(v for v in row["values"] if v["name"] == "Y")
        self.assertEqual(y_val["kind"], "output")
        self.assertFalse(y_val["success"])
        self.assertEqual(y_val["expected"], {})
        self.assertEqual(y_val["actual"], {"onnx_light.value_tags": "axes"})
        # One value matches (X), one does not (Y); test fails due to Y mismatch
        self.assertEqual(row["matched_values"], 1)
        self.assertEqual(row["total_values"], 2)
        self.assertFalse(row["success"])

    def test_score_test_values_preserves_order(self):
        expected_values = [
            {"name": "A", "kind": "input", "metadata": {"onnx_light.value_tags": "shape"}},
            {"name": "B", "kind": "output", "metadata": {"onnx_light.value_tags": "shape"}},
        ]
        row = stc._score_test(
            "test_order",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=expected_values,
            actual_values=expected_values,
        )
        names = [v["name"] for v in row["values"]]
        self.assertEqual(names, ["A", "B"])

    def test_score_test_values_missing_value_tags_fails(self):
        row = stc._score_test(
            "test_values_missing_tag",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=[{"name": "R", "kind": "result", "metadata": {}}],
            actual_values=[{"name": "R", "kind": "result", "metadata": {}}],
        )
        self.assertFalse(row["success"])
        self.assertEqual(row["matched_values"], 0)
        self.assertEqual(row["total_values"], 1)
        self.assertFalse(row["values"][0]["success"])

    def test_score_test_includes_mermaid_when_provided(self):
        row = stc._score_test(
            "test_with_mermaid",
            expected_nodes=[{"onnx_light.node_tag": "shape"}],
            actual_nodes=[{"onnx_light.node_tag": "shape"}],
            node_ops=["Shape"],
            mermaid="flowchart TD\n    A --> B",
        )
        self.assertTrue(row["success"])
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    A --> B")

    def test_score_test_includes_graph_svg_when_provided(self):
        row = stc._score_test(
            "test_with_svg",
            expected_nodes=[{"onnx_light.node_tag": "shape"}],
            actual_nodes=[{"onnx_light.node_tag": "shape"}],
            node_ops=["Shape"],
            graph={"svg": "<svg><rect/></svg>"},
        )
        self.assertIn("graph", row)
        self.assertEqual(row["graph"], {"svg": "<svg><rect/></svg>"})

    def test_score_test_keeps_node_input_output_info(self):
        row = stc._score_test(
            "test_with_io",
            expected_nodes=[{"onnx_light.node_tag": "shape"}],
            actual_nodes=[{"onnx_light.node_tag": "shape"}],
            node_ops=["Shape"],
            node_inputs=[["X"]],
            node_outputs=[["S"]],
        )
        self.assertEqual(row["nodes"][0]["inputs"], ["X"])
        self.assertEqual(row["nodes"][0]["outputs"], ["S"])

    def test_score_test_omits_mermaid_when_empty(self):
        row = stc._score_test(
            "test_no_mermaid",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            mermaid="",
        )
        self.assertNotIn("mermaid", row)

    def test_score_test_omits_graph_without_svg(self):
        row = stc._score_test(
            "test_no_graph",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            graph={"nodes": []},
        )
        self.assertNotIn("graph", row)

    def test_score_test_missing_metadata_flag_when_no_expected_tags(self):
        """When all nodes have empty expected AND actual metadata, missing_metadata=True and test fails."""
        row = stc._score_test(
            "test_no_metadata",
            expected_nodes=[{}, {}],
            actual_nodes=[{}, {}],
            node_ops=["Relu", "Abs"],
        )
        self.assertTrue(row["missing_metadata"])
        self.assertFalse(row["success"])
        self.assertEqual(row["total_metadata"], 0)
        self.assertEqual(row["matched_metadata"], 0)
        # Nodes themselves still appear to match (both empty)
        self.assertEqual(row["total_nodes"], 2)

    def test_score_test_no_missing_metadata_flag_when_expected_tags_present(self):
        """When at least one node has expected metadata, missing_metadata=False."""
        row = stc._score_test(
            "test_has_metadata",
            expected_nodes=[{"onnx_light.node_tag": "shape"}, {}],
            actual_nodes=[{"onnx_light.node_tag": "shape"}, {}],
            node_ops=["Shape", "Relu"],
        )
        self.assertFalse(row["missing_metadata"])
        self.assertTrue(row["success"])
        self.assertEqual(row["total_metadata"], 1)
        self.assertEqual(row["matched_metadata"], 1)

    def test_score_test_no_missing_metadata_flag_when_value_tags_present(self):
        """Value-level tags should count as meaningful metadata for shape-tag coverage."""
        row = stc._score_test(
            "test_has_value_metadata",
            expected_nodes=[{}],
            actual_nodes=[{}],
            node_ops=["Identity"],
            expected_values=[
                {
                    "name": "X",
                    "kind": "input",
                    "metadata": {"onnx_light.value_tags": "shape"},
                }
            ],
            actual_values=[
                {
                    "name": "X",
                    "kind": "input",
                    "metadata": {"onnx_light.value_tags": "shape"},
                }
            ],
        )
        self.assertFalse(row["missing_metadata"])
        self.assertTrue(row["success"])
        self.assertEqual(row["matched_values"], 1)
        self.assertEqual(row["total_values"], 1)

    def test_score_test_no_missing_metadata_flag_when_error_set(self):
        """When error is already set, missing_metadata stays False even with no metadata."""
        row = stc._score_test(
            "test_error",
            expected_nodes=[{}, {}],
            actual_nodes=[{}, {}],
            node_ops=["Relu", "Abs"],
            error="some exception",
        )
        self.assertFalse(row["missing_metadata"])
        # Error flag is separate; success was already False due to the error
        self.assertFalse(row["success"])

    def test_score_test_no_missing_metadata_when_no_nodes(self):
        """When there are no nodes at all, missing_metadata remains False (nothing to check)."""
        row = stc._score_test(
            "test_empty_nodes",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
        )
        self.assertFalse(row["missing_metadata"])
        self.assertTrue(row["success"])

    def test_build_payload_counts_missing_metadata_as_fail(self):
        """Tests with no expected metadata on any node are counted as failures in totals."""
        tests = [
            {
                "name": "test_no_tags",
                "model": "model_no_tags",
                "expected_nodes": [{}, {}],
                "node_ops": ["Relu", "Abs"],
            },
        ]

        def fake_run(model):
            return {"actual_nodes": [{}, {}]}

        payload = stc.build_payload(
            tag="shape_tag",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(payload["totals"]["tests"], {"pass": 0, "fail": 1})
        row = payload["tests"][0]
        self.assertFalse(row["success"])
        self.assertTrue(row["missing_metadata"])

    def test_build_payload_passes_values(self):
        expected_values = [{"name": "X", "kind": "input", "metadata": {"onnx_light.value_tags": "shape"}}]
        tests = [
            {
                "name": "test_vals",
                "model": "model_v",
                "expected_nodes": [{"onnx_light.node_tag": "shape"}],
                "node_ops": ["Shape"],
                "expected_values": expected_values,
            }
        ]

        def fake_run(model):
            return {
                "actual_nodes": [{"onnx_light.node_tag": "shape"}],
                "actual_values": [{"name": "X", "kind": "input", "metadata": {"onnx_light.value_tags": "shape"}}],
            }

        payload = stc.build_payload(
            tag="shape_tag",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertIn("values", row)
        self.assertEqual(len(row["values"]), 1)
        self.assertEqual(row["values"][0]["name"], "X")
        self.assertTrue(row["values"][0]["success"])

    def test_build_payload_passes_mermaid(self):
        tests = [
            {
                "name": "test_mermaid",
                "model": "model_m",
                "expected_nodes": [{"onnx_light.node_tag": "shape"}],
                "node_ops": ["Shape"],
                "mermaid": "flowchart TD\n    in_X --> op_Shape --> out_S",
                "graph": {"svg": "<svg><g/></svg>"},
            }
        ]

        def fake_run(model):
            return {"actual_nodes": [{"onnx_light.node_tag": "shape"}]}

        payload = stc.build_payload(
            tag="shape_tag",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    in_X --> op_Shape --> out_S")
        self.assertEqual(row["graph"], {"svg": "<svg><g/></svg>"})

    def test_build_payload_aggregates_totals(self):
        tests = [
            {
                "name": "test_a",
                "model": "model_a",
                "expected_nodes": [
                    {},
                    {"onnx_light.node_tag": "shape"},
                ],
                "node_ops": ["Abs", "Shape"],
            },
            {
                "name": "test_b",
                "model": "model_b",
                "expected_nodes": [
                    {"onnx_light.value_tags": "shape"},
                ],
                "node_ops": ["Reshape"],
            },
        ]

        def fake_run(model):
            if model == "model_a":
                return {
                    "actual_nodes": [
                        {},
                        {"onnx_light.node_tag": "shape"},
                    ]
                }
            return {"actual_nodes": [{"onnx_light.value_tags": "axes"}]}

        payload = stc.build_payload(
            tag="shape_tag",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {"onnx_light": "0.1.0"},
        )

        self.assertEqual(payload["tag"], "shape_tag")
        self.assertEqual(payload["versions"], {"onnx_light": "0.1.0"})
        self.assertEqual(
            payload["totals"],
            {
                "tests": {"pass": 1, "fail": 1},
                "nodes": {"pass": 2, "fail": 1},
                "metadata": {"pass": 1, "fail": 1},
                "values": {"pass": 0, "fail": 0},
            },
        )
        self.assertEqual([row["name"] for row in payload["tests"]], ["test_a", "test_b"])

    def test_build_payload_captures_runner_exception(self):
        tests = [
            {
                "name": "boom",
                "model": "model_boom",
                "expected_nodes": [{"onnx_light.node_tag": "shape"}],
                "node_ops": ["Shape"],
            }
        ]

        def fake_run(model):
            raise RuntimeError("unexpected")

        payload = stc.build_payload(
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )

        self.assertEqual(payload["totals"]["tests"], {"pass": 0, "fail": 1})
        row = payload["tests"][0]
        self.assertEqual(row["name"], "boom")
        self.assertEqual(row["error"], "unexpected")
        self.assertFalse(row["success"])

    def test_main_writes_cache_file(self):
        sample_payload = {
            "date": "2026-06-30T00:00:00Z",
            "tag": "shape_tag",
            "versions": {"onnx_light": "0.1.0"},
            "totals": {
                "tests": {"pass": 1, "fail": 0},
                "nodes": {"pass": 2, "fail": 0},
                "metadata": {"pass": 2, "fail": 0},
                "values": {"pass": 5, "fail": 0},
            },
            "tests": [{"name": "test_a", "success": True}],
        }
        original_build = stc.build_payload
        try:
            stc.build_payload = lambda **kwargs: sample_payload
            with tempfile.TemporaryDirectory() as tmp:
                rc = stc.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(tmp, "onnx-light", "shape_tag_coverage.json")
                self.assertTrue(os.path.isfile(path))
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.assertEqual(payload, sample_payload)
        finally:
            stc.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = stc.build_payload
        try:
            def fake_build(**kwargs):
                raise RuntimeError("boom")

            stc.build_payload = fake_build
            self.assertEqual(stc.main([]), 1)
        finally:
            stc.build_payload = original_build


    def test_discover_includes_test_with_metadata_despite_wrong_tag(self):
        """Tests with METADATA_KEYS metadata are kept even if their tag doesn't match."""
        import sys
        import types

        node_with_meta = _FakeNode(
            "Shape", {"onnx_light.node_tag": "shape"}
        )
        tc_meta = _FakeTestCase(
            "test_tiny_llm",
            _FakeModel([node_with_meta]),
            tag="model",
        )
        node_no_meta = _FakeNode("Relu")
        tc_no_meta = _FakeTestCase(
            "test_no_meta",
            _FakeModel([node_no_meta]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda: {
            "test_tiny_llm": tc_meta,
            "test_no_meta": tc_no_meta,
        }
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
            ("onnx_light.onnx_lib.backend", types.ModuleType("onnx_light.onnx_lib.backend")),
            ("onnx_light.onnx_lib.backend.test", types.ModuleType("onnx_light.onnx_lib.backend.test")),
            ("onnx_light.onnx_lib.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            discovered = stc.discover_shape_tag_tests(tag="shape_tag")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertIn("test_tiny_llm", names)
        self.assertNotIn("test_no_meta", names)

    def test_discover_includes_test_with_value_metadata_despite_wrong_tag(self):
        """Value-level metadata should keep a test even when its tag does not match."""
        import sys
        import types

        tc_value_meta = _FakeTestCase(
            "test_value_meta",
            _FakeModel(
                [_FakeNode("Identity")],
                inputs=[
                    _FakeValueInfo(
                        "X", {"onnx_light.value_tags": "shape"}
                    )
                ],
            ),
            tag="model",
        )
        tc_no_meta = _FakeTestCase(
            "test_no_meta",
            _FakeModel([_FakeNode("Relu")]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda: {
            "test_value_meta": tc_value_meta,
            "test_no_meta": tc_no_meta,
        }
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_lib", types.ModuleType("onnx_light.onnx_lib")),
            ("onnx_light.onnx_lib.backend", types.ModuleType("onnx_light.onnx_lib.backend")),
            ("onnx_light.onnx_lib.backend.test", types.ModuleType("onnx_light.onnx_lib.backend.test")),
            ("onnx_light.onnx_lib.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            discovered = stc.discover_shape_tag_tests(tag="shape_tag")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertIn("test_value_meta", names)
        self.assertNotIn("test_no_meta", names)


if __name__ == "__main__":
    unittest.main()
