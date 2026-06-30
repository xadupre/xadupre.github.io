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
        self.assertEqual(row["nodes"][2]["op_type"], "Reshape")
        self.assertNotIn("mermaid", row)

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


if __name__ == "__main__":
    unittest.main()
