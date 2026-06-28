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
                "onnx_light.release_after": "A",
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
                {"onnx_light.release_after": "B"},
            ],
            actual_nodes=[
                {},
                {"onnx_light.inplace_reuse": "0:0:equal"},
                {"onnx_light.release_after": "A"},
            ],
            node_ops=["Abs", "Abs", "Abs"],
        )
        self.assertFalse(row["success"])
        self.assertEqual(row["matched_nodes"], 2)
        self.assertEqual(row["total_nodes"], 3)
        self.assertEqual(row["matched_metadata"], 1)
        self.assertEqual(row["total_metadata"], 2)
        self.assertEqual(row["nodes"][2]["op_type"], "Abs")

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
                    {"onnx_light.release_after": "X"},
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
            return {"actual_nodes": [{"onnx_light.release_after": "Y"}]}

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
        self.assertEqual([row["name"] for row in payload["tests"]], ["test_a", "test_b"])

    def test_build_payload_captures_runner_exception(self):
        tests = [
            {
                "name": "boom",
                "model": "model_boom",
                "expected_nodes": [{"onnx_light.release_after": "A"}],
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


if __name__ == "__main__":
    unittest.main()
