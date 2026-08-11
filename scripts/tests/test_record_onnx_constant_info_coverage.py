"""Tests for ``scripts.record_onnx_constant_info_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_constant_info_coverage as cic  # noqa: E402

CONSTANT = cic.CONSTANT_METADATA_KEY


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
    def __init__(
        self,
        nodes,
        inputs=None,
        outputs=None,
        initializers=None,
        value_info=None,
        metadata=None,
    ):
        self.node = nodes
        self.input = inputs or []
        self.output = outputs or []
        self.initializer = initializers or []
        self.value_info = value_info or []
        self.metadata_props = [
            _FakeMeta(k, v) for k, v in (metadata or {}).items()
        ]


class _FakeModel:
    def __init__(
        self,
        nodes,
        inputs=None,
        outputs=None,
        initializers=None,
        value_info=None,
        metadata=None,
    ):
        self.graph = _FakeGraph(
            nodes,
            inputs=inputs,
            outputs=outputs,
            initializers=initializers,
            value_info=value_info,
            metadata=metadata,
        )


class _FakeTestCase:
    def __init__(self, name, model, tag=""):
        self.name = name
        self.model = model
        self.tag = tag


class TestRecordOnnxConstantInfoCoverage(unittest.TestCase):
    def test_node_metadata_filters_unrelated_keys(self):
        node = _FakeNode(
            "Add",
            {
                CONSTANT: "1",
                "onnx_light.node_tag": "weight",
                "onnx_light.inplace_reuse": "0:0:equal",
                "ignored": "x",
            },
        )
        self.assertEqual(cic._node_metadata(node), {CONSTANT: "1"})

    def test_value_metadata_filters_unrelated_keys(self):
        vi = _FakeValueInfo(
            "X",
            {
                CONSTANT: "1",
                "onnx_light.value_tag": "weight",
                "ignored": "x",
            },
        )
        self.assertEqual(cic._value_metadata(vi), {CONSTANT: "1"})

    def test_graph_value_snapshot_collects_inputs_outputs_initializers(self):
        inp = _FakeValueInfo("X", {})
        out = _FakeValueInfo("Y", {})
        init = _FakeTensorProto("W", {CONSTANT: "1"})
        val = _FakeValueInfo("Z", {CONSTANT: "1"})
        model = _FakeModel(
            [], inputs=[inp], outputs=[out], initializers=[init], value_info=[val]
        )
        snapshot = cic._graph_value_snapshot(model)
        names = [s["name"] for s in snapshot]
        self.assertIn("X", names)
        self.assertIn("Y", names)
        self.assertIn("W", names)
        self.assertIn("Z", names)
        x_entry = next(s for s in snapshot if s["name"] == "X")
        self.assertEqual(x_entry["kind"], "input")
        self.assertEqual(x_entry["metadata"], {})
        w_entry = next(s for s in snapshot if s["name"] == "W")
        self.assertEqual(w_entry["kind"], "initializer")
        self.assertEqual(w_entry["metadata"], {CONSTANT: "1"})
        z_entry = next(s for s in snapshot if s["name"] == "Z")
        self.assertEqual(z_entry["kind"], "result")
        self.assertEqual(z_entry["metadata"], {CONSTANT: "1"})

    def test_graph_value_snapshot_excludes_initializer_from_inputs(self):
        inp = _FakeValueInfo("X")
        init = _FakeTensorProto("W")

        class FakeGraph:
            node = []
            input = [_FakeValueInfo("W"), inp]
            output = []
            initializer = [init]
            value_info = []

        class FakeModel:
            graph = FakeGraph()

        snapshot = cic._graph_value_snapshot(FakeModel())
        kinds = {s["name"]: s["kind"] for s in snapshot}
        self.assertEqual(kinds.get("W"), "initializer")
        self.assertEqual(kinds.get("X"), "input")

    def test_graph_value_snapshot_merges_value_info_into_input_output(self):
        # onnx-light may store constant metadata for graph inputs/outputs in
        # graph.value_info (same name). Those entries should be merged into the
        # input/output entry rather than appearing as a separate result.
        inp = _FakeValueInfo("X", {})
        out = _FakeValueInfo("Y", {})
        vi_x = _FakeValueInfo("X", {CONSTANT: "1"})
        vi_y = _FakeValueInfo("Y", {CONSTANT: "1"})
        model = _FakeModel(
            [],
            inputs=[inp],
            outputs=[out],
            value_info=[vi_x, vi_y],
        )
        snapshot = cic._graph_value_snapshot(model)
        names = [s["name"] for s in snapshot]
        self.assertEqual(names.count("X"), 1)
        self.assertEqual(names.count("Y"), 1)
        x_entry = next(s for s in snapshot if s["name"] == "X")
        self.assertEqual(x_entry["kind"], "input")
        self.assertEqual(x_entry["metadata"], {CONSTANT: "1"})
        y_entry = next(s for s in snapshot if s["name"] == "Y")
        self.assertEqual(y_entry["kind"], "output")
        self.assertEqual(y_entry["metadata"], {CONSTANT: "1"})

    def test_clear_node_metadata_removes_entries(self):
        node = _FakeNode("Add", {CONSTANT: "1"})
        cic._clear_node_metadata(node)
        self.assertEqual(node.metadata_props, [])

    def test_score_test_counts_nodes_and_metadata(self):
        row = cic._score_test(
            "test_cc_constant_add_chain",
            expected_nodes=[
                {CONSTANT: "1"},
                {},
            ],
            actual_nodes=[
                {CONSTANT: "1"},
                {CONSTANT: "1"},
            ],
            node_ops=["Add", "Add"],
        )
        self.assertFalse(row["success"])
        self.assertEqual(row["matched_nodes"], 1)
        self.assertEqual(row["total_nodes"], 2)
        self.assertEqual(row["matched_metadata"], 1)
        self.assertEqual(row["total_metadata"], 2)
        self.assertEqual(row["matched_values"], 0)
        self.assertEqual(row["total_values"], 0)
        self.assertNotIn("mermaid", row)
        self.assertIn("values", row)
        self.assertEqual(row["values"], [])

    def test_score_test_includes_values_section(self):
        expected_values = [
            {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}},
            {"name": "Y", "kind": "output", "metadata": {}},
        ]
        actual_values = [
            {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}},
            {"name": "Y", "kind": "output", "metadata": {CONSTANT: "1"}},
        ]
        row = cic._score_test(
            "test_values",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=expected_values,
            actual_values=actual_values,
        )
        self.assertIn("values", row)
        # Y is non-constant on the expected side but constant on the actual side,
        # so it carries a signal and is scored (as a mismatch).
        self.assertEqual(len(row["values"]), 2)
        c_val = next(v for v in row["values"] if v["name"] == "C")
        self.assertEqual(c_val["kind"], "initializer")
        self.assertTrue(c_val["success"])
        y_val = next(v for v in row["values"] if v["name"] == "Y")
        self.assertFalse(y_val["success"])
        self.assertEqual(y_val["expected"], {})
        self.assertEqual(y_val["actual"], {CONSTANT: "1"})
        self.assertEqual(row["matched_values"], 1)
        self.assertEqual(row["total_values"], 2)
        self.assertFalse(row["success"])

    def test_score_test_values_preserves_order(self):
        expected_values = [
            {"name": "A", "kind": "initializer", "metadata": {CONSTANT: "1"}},
            {"name": "B", "kind": "result", "metadata": {CONSTANT: "1"}},
        ]
        row = cic._score_test(
            "test_order",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=expected_values,
            actual_values=expected_values,
        )
        names = [v["name"] for v in row["values"]]
        self.assertEqual(names, ["A", "B"])

    def test_score_test_non_constant_values_are_ignored(self):
        row = cic._score_test(
            "test_values_no_constant",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=[{"name": "R", "kind": "result", "metadata": {}}],
            actual_values=[{"name": "R", "kind": "result", "metadata": {}}],
        )
        self.assertTrue(row["success"])
        self.assertEqual(row["matched_values"], 0)
        self.assertEqual(row["total_values"], 0)
        self.assertEqual(row["values"], [])

    def test_score_test_includes_mermaid_when_provided(self):
        row = cic._score_test(
            "test_with_mermaid",
            expected_nodes=[{CONSTANT: "1"}],
            actual_nodes=[{CONSTANT: "1"}],
            node_ops=["Constant"],
            mermaid="flowchart TD\n    A --> B",
        )
        self.assertTrue(row["success"])
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    A --> B")

    def test_score_test_includes_graph_svg_when_provided(self):
        row = cic._score_test(
            "test_with_svg",
            expected_nodes=[{CONSTANT: "1"}],
            actual_nodes=[{CONSTANT: "1"}],
            node_ops=["Constant"],
            graph={"svg": "<svg><rect/></svg>"},
        )
        self.assertIn("graph", row)
        self.assertEqual(row["graph"], {"svg": "<svg><rect/></svg>"})

    def test_score_test_keeps_node_input_output_info(self):
        row = cic._score_test(
            "test_with_io",
            expected_nodes=[{CONSTANT: "1"}],
            actual_nodes=[{CONSTANT: "1"}],
            node_ops=["Add"],
            node_inputs=[["C", "C"]],
            node_outputs=[["D"]],
        )
        self.assertEqual(row["nodes"][0]["inputs"], ["C", "C"])
        self.assertEqual(row["nodes"][0]["outputs"], ["D"])

    def test_score_test_omits_mermaid_when_empty(self):
        row = cic._score_test(
            "test_no_mermaid",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            mermaid="",
        )
        self.assertNotIn("mermaid", row)

    def test_score_test_omits_graph_without_svg(self):
        row = cic._score_test(
            "test_no_graph",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            graph={"nodes": []},
        )
        self.assertNotIn("graph", row)

    def test_score_test_missing_metadata_flag_when_no_expected_tags(self):
        """When all nodes have empty expected AND actual metadata, missing_metadata=True and test fails."""
        row = cic._score_test(
            "test_no_metadata",
            expected_nodes=[{}, {}],
            actual_nodes=[{}, {}],
            node_ops=["Relu", "Abs"],
        )
        self.assertTrue(row["missing_metadata"])
        self.assertFalse(row["success"])
        self.assertEqual(row["total_metadata"], 0)
        self.assertEqual(row["matched_metadata"], 0)
        self.assertEqual(row["total_nodes"], 2)

    def test_score_test_no_missing_metadata_flag_when_expected_tags_present(self):
        """When at least one node has expected metadata, missing_metadata=False."""
        row = cic._score_test(
            "test_has_metadata",
            expected_nodes=[{CONSTANT: "1"}, {}],
            actual_nodes=[{CONSTANT: "1"}, {}],
            node_ops=["Constant", "Relu"],
        )
        self.assertFalse(row["missing_metadata"])
        self.assertTrue(row["success"])
        self.assertEqual(row["total_metadata"], 1)
        self.assertEqual(row["matched_metadata"], 1)

    def test_score_test_no_missing_metadata_flag_when_value_tags_present(self):
        """Value-level constant flags should count as meaningful metadata."""
        row = cic._score_test(
            "test_has_value_metadata",
            expected_nodes=[{}],
            actual_nodes=[{}],
            node_ops=["Identity"],
            expected_values=[
                {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}}
            ],
            actual_values=[
                {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}}
            ],
        )
        self.assertFalse(row["missing_metadata"])
        self.assertTrue(row["success"])
        self.assertEqual(row["matched_values"], 1)
        self.assertEqual(row["total_values"], 1)

    def test_score_test_no_missing_metadata_flag_when_error_set(self):
        """When error is already set, missing_metadata stays False even with no metadata."""
        row = cic._score_test(
            "test_error",
            expected_nodes=[{}, {}],
            actual_nodes=[{}, {}],
            node_ops=["Relu", "Abs"],
            error="some exception",
        )
        self.assertFalse(row["missing_metadata"])
        self.assertFalse(row["success"])

    def test_score_test_no_missing_metadata_when_no_nodes(self):
        """When there are no nodes at all, missing_metadata remains False (nothing to check)."""
        row = cic._score_test(
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

        payload = cic.build_payload(
            tag="constant",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(payload["totals"]["tests"], {"pass": 0, "fail": 1})
        row = payload["tests"][0]
        self.assertFalse(row["success"])
        self.assertTrue(row["missing_metadata"])

    def test_build_payload_passes_values(self):
        expected_values = [
            {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}}
        ]
        tests = [
            {
                "name": "test_vals",
                "model": "model_v",
                "expected_nodes": [{CONSTANT: "1"}],
                "node_ops": ["Constant"],
                "expected_values": expected_values,
            }
        ]

        def fake_run(model):
            return {
                "actual_nodes": [{CONSTANT: "1"}],
                "actual_values": [
                    {"name": "C", "kind": "initializer", "metadata": {CONSTANT: "1"}}
                ],
            }

        payload = cic.build_payload(
            tag="constant",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertIn("values", row)
        self.assertEqual(len(row["values"]), 1)
        self.assertEqual(row["values"][0]["name"], "C")
        self.assertTrue(row["values"][0]["success"])

    def test_build_payload_passes_mermaid(self):
        tests = [
            {
                "name": "test_mermaid",
                "model": "model_m",
                "expected_nodes": [{CONSTANT: "1"}],
                "node_ops": ["Constant"],
                "mermaid": "flowchart TD\n    init_C --> op_Add --> out_D",
                "graph": {"svg": "<svg><g/></svg>"},
            }
        ]

        def fake_run(model):
            return {"actual_nodes": [{CONSTANT: "1"}]}

        payload = cic.build_payload(
            tag="constant",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertIn("mermaid", row)
        self.assertEqual(
            row["mermaid"], "flowchart TD\n    init_C --> op_Add --> out_D"
        )
        self.assertEqual(row["graph"], {"svg": "<svg><g/></svg>"})

    def test_build_payload_aggregates_totals(self):
        tests = [
            {
                "name": "test_a",
                "model": "model_a",
                "expected_nodes": [
                    {CONSTANT: "1"},
                    {},
                ],
                "node_ops": ["Constant", "Add"],
            },
            {
                "name": "test_b",
                "model": "model_b",
                "expected_nodes": [
                    {CONSTANT: "1"},
                ],
                "node_ops": ["Add"],
            },
        ]

        def fake_run(model):
            if model == "model_a":
                return {
                    "actual_nodes": [
                        {CONSTANT: "1"},
                        {},
                    ]
                }
            return {"actual_nodes": [{}]}

        payload = cic.build_payload(
            tag="constant",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {"onnx_light": "0.1.0"},
        )

        self.assertEqual(payload["tag"], "constant")
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
        self.assertEqual(
            [row["name"] for row in payload["tests"]], ["test_a", "test_b"]
        )

    def test_build_payload_captures_runner_exception(self):
        tests = [
            {
                "name": "boom",
                "model": "model_boom",
                "expected_nodes": [{CONSTANT: "1"}],
                "node_ops": ["Constant"],
            }
        ]

        def fake_run(model):
            raise RuntimeError("unexpected")

        payload = cic.build_payload(
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
            "tag": "constant",
            "versions": {"onnx_light": "0.1.0"},
            "totals": {
                "tests": {"pass": 1, "fail": 0},
                "nodes": {"pass": 2, "fail": 0},
                "metadata": {"pass": 2, "fail": 0},
                "values": {"pass": 5, "fail": 0},
            },
            "tests": [{"name": "test_a", "success": True}],
        }
        original_build = cic.build_payload
        try:
            cic.build_payload = lambda **kwargs: sample_payload
            with tempfile.TemporaryDirectory() as tmp:
                rc = cic.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(
                    tmp, "onnx-light", "constant_info_coverage.json"
                )
                self.assertTrue(os.path.isfile(path))
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.assertEqual(payload, sample_payload)
        finally:
            cic.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = cic.build_payload
        try:

            def fake_build(**kwargs):
                raise RuntimeError("boom")

            cic.build_payload = fake_build
            self.assertEqual(cic.main([]), 1)
        finally:
            cic.build_payload = original_build

    def test_discover_includes_test_with_metadata_despite_wrong_tag(self):
        """Tests with METADATA_KEYS metadata are kept even if their tag doesn't match."""
        import types

        node_with_meta = _FakeNode("Constant", {CONSTANT: "1"})
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
        fake_module.collect_test_case = lambda include_big=False: {
            "test_tiny_llm": tc_meta,
            "test_no_meta": tc_no_meta,
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
            discovered = cic.discover_constant_info_tests(tag="constant")
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
        import types

        tc_value_meta = _FakeTestCase(
            "test_value_meta",
            _FakeModel(
                [_FakeNode("Identity")],
                initializers=[_FakeTensorProto("C", {CONSTANT: "1"})],
            ),
            tag="model",
        )
        tc_no_meta = _FakeTestCase(
            "test_no_meta",
            _FakeModel([_FakeNode("Relu")]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_value_meta": tc_value_meta,
            "test_no_meta": tc_no_meta,
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
            discovered = cic.discover_constant_info_tests(tag="constant")
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
