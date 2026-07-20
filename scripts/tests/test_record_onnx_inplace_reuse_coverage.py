"""Tests for ``scripts.record_onnx_inplace_reuse_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_inplace_reuse_coverage as ric  # noqa: E402


class _FakeMeta:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class _FakeNode:
    def __init__(self, op_type, metadata=None):
        self.op_type = op_type
        self.metadata_props = [_FakeMeta(k, v) for k, v in (metadata or {}).items()]


class _FakeGraph:
    def __init__(self, nodes):
        self.node = nodes


class _FakeModel:
    def __init__(self, nodes):
        self.graph = _FakeGraph(nodes)


class _FakeTestCase:
    def __init__(self, name, model, tag=""):
        self.name = name
        self.model = model
        self.tag = tag


class TestRecordOnnxInplaceReuseCoverage(unittest.TestCase):
    def test_node_metadata_filters_unrelated_keys(self):
        node = _FakeNode(
            "Abs",
            {
                "onnx_light.inplace_reuse": "0:0:equal",
                "onnx_light.release_after": "A",
                "ignored": "x",
            },
        )
        self.assertEqual(
            ric._node_metadata(node),
            {
                "onnx_light.inplace_reuse": "0:0:equal",
            },
        )

    def test_clear_node_metadata_removes_entries(self):
        node = _FakeNode("Abs", {"onnx_light.inplace_reuse": "0:0:equal"})
        ric._clear_node_metadata(node)
        self.assertEqual(node.metadata_props, [])

    def test_score_test_counts_nodes_and_metadata(self):
        row = ric._score_test(
            "test_cc_shape_inference_inplace_reuse",
            expected_nodes=[
                {},
                {"onnx_light.inplace_reuse": "0:0:equal"},
                {"onnx_light.inplace_reuse": "0:0:equal"},
            ],
            actual_nodes=[
                {},
                {"onnx_light.inplace_reuse": "0:0:equal"},
                {"onnx_light.inplace_reuse": "0:0:move"},
            ],
            node_ops=["Abs", "Abs", "Abs"],
        )
        self.assertFalse(row["success"])
        self.assertEqual(row["matched_nodes"], 2)
        self.assertEqual(row["total_nodes"], 3)
        self.assertEqual(row["matched_metadata"], 1)
        self.assertEqual(row["total_metadata"], 2)
        self.assertEqual(row["nodes"][2]["op_type"], "Abs")
        self.assertNotIn("mermaid", row)

    def test_score_test_includes_mermaid_when_provided(self):
        row = ric._score_test(
            "test_with_mermaid",
            expected_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            actual_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            node_ops=["Abs"],
            mermaid="flowchart TD\n    A --> B",
        )
        self.assertTrue(row["success"])
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    A --> B")

    def test_score_test_includes_graph_svg_when_provided(self):
        row = ric._score_test(
            "test_with_svg",
            expected_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            actual_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            node_ops=["Abs"],
            graph={"svg": "<svg><rect/></svg>"},
        )
        self.assertIn("graph", row)
        self.assertEqual(row["graph"], {"svg": "<svg><rect/></svg>"})

    def test_score_test_keeps_node_input_output_info(self):
        row = ric._score_test(
            "test_with_io",
            expected_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            actual_nodes=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            node_ops=["Abs"],
            node_inputs=[["X"]],
            node_outputs=[["Y"]],
        )
        self.assertEqual(row["nodes"][0]["inputs"], ["X"])
        self.assertEqual(row["nodes"][0]["outputs"], ["Y"])

    def test_score_test_omits_mermaid_when_empty(self):
        row = ric._score_test(
            "test_no_mermaid",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            mermaid="",
        )
        self.assertNotIn("mermaid", row)

    def test_score_test_omits_graph_without_svg(self):
        row = ric._score_test(
            "test_no_graph",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            graph={"nodes": []},
        )
        self.assertNotIn("graph", row)

    def test_build_payload_passes_mermaid(self):
        tests = [
            {
                "name": "test_mermaid",
                "model": "model_m",
                "expected_nodes": [{"onnx_light.inplace_reuse": "0:0:equal"}],
                "node_ops": ["Add"],
                "mermaid": "flowchart TD\n    in_X --> op_Add --> out_Y",
                "graph": {"svg": "<svg><g/></svg>"},
            }
        ]

        def fake_run(model):
            return {"actual_nodes": [{"onnx_light.inplace_reuse": "0:0:equal"}]}

        payload = ric.build_payload(
            tag="inplace",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    in_X --> op_Add --> out_Y")
        self.assertEqual(row["graph"], {"svg": "<svg><g/></svg>"})

    def test_build_payload_aggregates_totals(self):
        tests = [
            {
                "name": "test_a",
                "model": "model_a",
                "expected_nodes": [
                    {},
                    {"onnx_light.inplace_reuse": "0:0:equal"},
                ],
                "node_ops": ["Abs", "Abs"],
            },
            {
                "name": "test_b",
                "model": "model_b",
                "expected_nodes": [
                    {"onnx_light.inplace_reuse": "0:0:equal"},
                ],
                "node_ops": ["Reshape"],
            },
        ]

        def fake_run(model):
            if model == "model_a":
                return {
                    "actual_nodes": [
                        {},
                        {"onnx_light.inplace_reuse": "0:0:equal"},
                    ]
                }
            return {"actual_nodes": [{"onnx_light.inplace_reuse": "0:0:copy"}]}

        payload = ric.build_payload(
            tag="inplace",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {"onnx_light": "0.1.0"},
        )

        self.assertEqual(payload["tag"], "inplace")
        self.assertEqual(payload["versions"], {"onnx_light": "0.1.0"})
        self.assertEqual(
            payload["totals"],
            {
                "tests": {"pass": 1, "fail": 1},
                "nodes": {"pass": 2, "fail": 1},
                "metadata": {"pass": 1, "fail": 1},
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
                "expected_nodes": [{"onnx_light.inplace_reuse": "0:0:equal"}],
                "node_ops": ["Abs"],
            }
        ]

        def fake_run(model):
            raise RuntimeError("unexpected")

        payload = ric.build_payload(
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
            "date": "2026-06-26T00:00:00Z",
            "tag": "inplace",
            "versions": {"onnx_light": "0.1.0"},
            "totals": {
                "tests": {"pass": 1, "fail": 0},
                "nodes": {"pass": 2, "fail": 0},
                "metadata": {"pass": 2, "fail": 0},
            },
            "tests": [{"name": "test_a", "success": True}],
        }
        original_build = ric.build_payload
        try:
            ric.build_payload = lambda **kwargs: sample_payload
            with tempfile.TemporaryDirectory() as tmp:
                rc = ric.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(tmp, "onnx-light", "inplace_reuse_coverage.json")
                self.assertTrue(os.path.isfile(path))
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.assertEqual(payload, sample_payload)
        finally:
            ric.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = ric.build_payload
        try:

            def fake_build(**kwargs):
                raise RuntimeError("boom")

            ric.build_payload = fake_build
            self.assertEqual(ric.main([]), 1)
        finally:
            ric.build_payload = original_build

    def test_score_test_scores_graph_input_metadata(self):
        """Graph-level input metadata is scored alongside node metadata."""
        row = ric._score_test(
            "test_with_inputs",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_inputs=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            actual_inputs=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            graph_input_names=["X"],
        )
        self.assertTrue(row["success"])
        self.assertEqual(len(row["inputs"]), 1)
        self.assertEqual(row["inputs"][0]["name"], "X")
        self.assertTrue(row["inputs"][0]["success"])
        self.assertEqual(row["matched_metadata"], 1)
        self.assertEqual(row["total_metadata"], 1)

    def test_score_test_fails_on_input_metadata_mismatch(self):
        row = ric._score_test(
            "test_input_mismatch",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_inputs=[{"onnx_light.inplace_reuse": "0:0:equal"}],
            actual_inputs=[{"onnx_light.inplace_reuse": "0:0:greater"}],
            graph_input_names=["X"],
        )
        self.assertFalse(row["success"])
        self.assertFalse(row["inputs"][0]["success"])
        self.assertEqual(row["matched_metadata"], 0)
        self.assertEqual(row["total_metadata"], 1)

    def test_score_test_omits_inputs_key_when_none(self):
        """``inputs`` key is absent when no graph-input metadata is provided."""
        row = ric._score_test(
            "test_no_inputs",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
        )
        self.assertNotIn("inputs", row)

    def test_discover_includes_big_test_without_tag_or_metadata(self):
        """Tests with ``_big_`` in their name are always included."""
        import sys
        import types

        node_plain = _FakeNode("MatMul")
        tc_big = _FakeTestCase(
            "test_cc_shape_inference_big_qwen3_4_layers_like",
            _FakeModel([node_plain]),
            tag="model",
        )
        tc_small = _FakeTestCase(
            "test_no_meta_no_tag",
            _FakeModel([node_plain]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_cc_shape_inference_big_qwen3_4_layers_like": tc_big,
            "test_no_meta_no_tag": tc_small,
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
            discovered = ric.discover_inplace_tests(tag="inplace")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertIn("test_cc_shape_inference_big_qwen3_4_layers_like", names)
        self.assertNotIn("test_no_meta_no_tag", names)

    def test_discover_includes_graph_input_metadata(self):
        """``expected_inputs`` captures metadata from ``graph.input`` entries."""
        import sys
        import types

        # A ValueInfoProto-like fake with metadata on graph.input
        vi_with_meta = _FakeNode(
            "input", {"onnx_light.inplace_reuse": "0:0:equal"}
        )
        vi_with_meta.name = "X"

        class _FakeGraphWithInputs:
            def __init__(self, nodes, inputs):
                self.node = nodes
                self.input = inputs

        class _FakeModelWithInputs:
            def __init__(self, nodes, inputs):
                self.graph = _FakeGraphWithInputs(nodes, inputs)

        tc = _FakeTestCase(
            "test_big_with_input_meta",
            _FakeModelWithInputs([_FakeNode("Abs")], [vi_with_meta]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_big_with_input_meta": tc,
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
            discovered = ric.discover_inplace_tests(tag="inplace")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(len(discovered), 1, (
            "Test with graph-input metadata should be included even when its "
            "tag does not match and no node carries metadata."
        ))
        entry = discovered[0]
        self.assertEqual(entry["name"], "test_big_with_input_meta")
        self.assertEqual(entry["expected_inputs"], [{"onnx_light.inplace_reuse": "0:0:equal"}])
        self.assertEqual(entry["graph_input_names"], ["X"])

    def test_discover_includes_test_with_metadata_despite_wrong_tag(self):
        """Tests with METADATA_KEYS metadata are kept even if their tag doesn't match."""
        import sys
        import types

        node_with_meta = _FakeNode("Abs", {"onnx_light.inplace_reuse": "0:0:equal"})
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
            discovered = ric.discover_inplace_tests(tag="inplace")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertIn("test_tiny_llm", names)
        self.assertNotIn("test_no_meta", names)


if __name__ == "__main__":
    unittest.main()
