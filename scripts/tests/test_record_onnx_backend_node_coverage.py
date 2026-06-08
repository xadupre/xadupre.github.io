"""Tests for ``scripts.record_onnx_backend_node_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_backend_node_coverage as rbn  # noqa: E402


class TestRecordOnnxBackendNodeCoverage(unittest.TestCase):
    def test_backends_include_four_runtimes(self):
        self.assertEqual(
            rbn.BACKENDS, ("onnxruntime", "reference", "onnx_light", "yobx")
        )
        for backend in rbn.BACKENDS:
            self.assertIn(backend, rbn.BACKEND_PACKAGE)
            self.assertIn(backend, rbn._BACKEND_FACTORIES)

    def test_row_from_results_handles_four_backends(self):
        row = rbn._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {"success": True, "error": "", "error_step": ""},
                "onnx_light": {
                    "success": False,
                    "error": "kernel missing",
                    "error_step": "load",
                },
                "yobx": {"success": True, "error": "", "error_step": ""},
            },
            versions={
                "onnxruntime": "1.20.0",
                "onnx": "1.17.0",
                "onnx_light": "0.1.0",
                "yobx": "0.2.0",
            },
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertTrue(row["onnxruntime"])
        self.assertTrue(row["reference"])
        self.assertFalse(row["onnx_light"])
        self.assertTrue(row["yobx"])
        self.assertEqual(row["onnx_light_error"], "kernel missing")
        self.assertEqual(row["onnx_light_error_step"], "load")
        self.assertEqual(row["yobx_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["yobx_last_pass_version"], "0.2.0")
        self.assertNotIn("onnx_light_last_pass_date", row)

    def test_row_from_results_carries_over_previous_yobx_last_pass(self):
        previous = {
            "name": "test_relu",
            "yobx_last_pass_date": "2024-01-02T03:04:05Z",
            "yobx_last_pass_version": "0.1.0",
        }
        row = rbn._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {"success": True, "error": "", "error_step": ""},
                "onnx_light": {"success": True, "error": "", "error_step": ""},
                "yobx": {
                    "success": False,
                    "error": "boom",
                    "error_step": "run",
                },
            },
            previous=previous,
            versions={"yobx": "0.2.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertFalse(row["yobx"])
        self.assertEqual(row["yobx_last_pass_date"], "2024-01-02T03:04:05Z")
        self.assertEqual(row["yobx_last_pass_version"], "0.1.0")

    def test_build_payload_runs_every_backend_and_aggregates_totals(self):
        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": [("in_a", "out_a")]},
            {"name": "test_b", "model": "model_b", "data_sets": [("in_b", "out_b")]},
        ]
        ok = {"success": True, "error": "", "error_step": ""}
        ko = {"success": False, "error": "boom", "error_step": "run"}
        outcomes = {
            ("model_a", "onnxruntime"): ok,
            ("model_a", "reference"): ok,
            ("model_a", "onnx_light"): ok,
            ("model_a", "yobx"): ok,
            ("model_b", "onnxruntime"): ok,
            ("model_b", "reference"): ok,
            ("model_b", "onnx_light"): ko,
            ("model_b", "yobx"): ko,
        }

        def fake_run(model, data_sets, backend, rtol, atol):
            return outcomes[(model, backend)]

        payload = rbn.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {"onnx": "1.0", "yobx": "0.2.0"},
        )

        self.assertEqual(payload["kind"], "node")
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 2, "fail": 0},
                "reference": {"pass": 2, "fail": 0},
                "onnx_light": {"pass": 1, "fail": 1},
                "yobx": {"pass": 1, "fail": 1},
            },
        )
        by_name = {row["name"]: row for row in payload["tests"]}
        self.assertTrue(by_name["test_a"]["yobx"])
        self.assertFalse(by_name["test_b"]["yobx"])
        self.assertEqual(by_name["test_b"]["yobx_error"], "boom")
        self.assertEqual(by_name["test_b"]["yobx_error_step"], "run")

    def test_run_test_with_backend_unknown_backend(self):
        result = rbn.run_test_with_backend(
            model=object(), data_sets=[([1], [1])], backend="bogus"
        )
        self.assertFalse(result["success"])
        self.assertIn("unknown backend", result["error"])
        self.assertEqual(result["error_step"], "load")

    def test_run_test_with_backend_no_data_sets(self):
        result = rbn.run_test_with_backend(
            model=object(), data_sets=[], backend="onnxruntime"
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "load")
        self.assertIn("test_data_set", result["error"])

    def test_main_writes_json_to_expected_path(self):
        tests = [
            {"name": "test_x", "model": "model_x", "data_sets": [("i", "o")]},
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            return {"success": True, "error": "", "error_step": ""}

        with tempfile.TemporaryDirectory() as tmp:
            payload = rbn.build_payload(
                kind="node",
                discover=lambda kind: tests,
                run=fake_run,
                versions=lambda: {"onnx": "1.0"},
            )
            json_path = os.path.join(tmp, "onnx", "backend_node_coverage.json")
            rbn.write_payload(json_path, payload)
            self.assertTrue(os.path.exists(json_path))
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual([row["name"] for row in data["tests"]], ["test_x"])
            for backend in rbn.BACKENDS:
                self.assertIn(backend, data["totals"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
