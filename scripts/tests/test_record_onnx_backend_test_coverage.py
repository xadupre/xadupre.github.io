"""Tests for ``scripts.record_onnx_backend_test_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_backend_test_coverage as rbc  # noqa: E402


class TestRecordOnnxBackendTestCoverage(unittest.TestCase):
    def test_stringify_error_truncates_and_takes_first_line(self):
        self.assertEqual(rbc._stringify_error(None), "")
        self.assertEqual(rbc._stringify_error("boom"), "boom")
        self.assertEqual(rbc._stringify_error("boom\nrest"), "boom")
        long = "x" * 500
        out = rbc._stringify_error(long)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(len(out), 300)

    def test_row_from_results_includes_errors_only_when_present(self):
        row = rbc._row_from_results(
            "test_relu",
            {
                "onnxruntime": {"success": True, "error": "", "error_step": ""},
                "reference": {
                    "success": False,
                    "error": "boom",
                    "error_step": "run",
                },
            },
        )
        self.assertEqual(row["name"], "test_relu")
        self.assertTrue(row["onnxruntime"])
        self.assertFalse(row["reference"])
        self.assertNotIn("onnxruntime_error", row)
        self.assertNotIn("onnxruntime_error_step", row)
        self.assertEqual(row["reference_error"], "boom")
        self.assertEqual(row["reference_error_step"], "run")

    def test_build_payload_runs_every_backend_and_aggregates_totals(self):
        tests = [
            {"name": "test_a", "model_dir": "/fake/a"},
            {"name": "test_b", "model_dir": "/fake/b"},
            {"name": "test_c", "model_dir": "/fake/c"},
        ]
        # Map of (model_dir, backend) -> result dict
        outcomes = {
            ("/fake/a", "onnxruntime"): {"success": True, "error": "", "error_step": ""},
            ("/fake/a", "reference"): {"success": True, "error": "", "error_step": ""},
            ("/fake/b", "onnxruntime"): {"success": True, "error": "", "error_step": ""},
            ("/fake/b", "reference"): {
                "success": False,
                "error": "not implemented",
                "error_step": "run",
            },
            ("/fake/c", "onnxruntime"): {
                "success": False,
                "error": "kernel missing",
                "error_step": "load",
            },
            ("/fake/c", "reference"): {"success": True, "error": "", "error_step": ""},
        }

        def fake_run(model_dir, backend, rtol, atol):
            return outcomes[(model_dir, backend)]

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {"onnx": "1.0"},
        )

        self.assertEqual(payload["kind"], "node")
        self.assertEqual(payload["versions"], {"onnx": "1.0"})
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 2, "fail": 1},
                "reference": {"pass": 2, "fail": 1},
            },
        )
        names = [row["name"] for row in payload["tests"]]
        self.assertEqual(names, ["test_a", "test_b", "test_c"])
        by_name = {row["name"]: row for row in payload["tests"]}
        self.assertTrue(by_name["test_a"]["onnxruntime"])
        self.assertTrue(by_name["test_a"]["reference"])
        self.assertFalse(by_name["test_b"]["reference"])
        self.assertEqual(by_name["test_b"]["reference_error"], "not implemented")
        self.assertEqual(by_name["test_b"]["reference_error_step"], "run")
        self.assertFalse(by_name["test_c"]["onnxruntime"])
        self.assertEqual(by_name["test_c"]["onnxruntime_error_step"], "load")

    def test_build_payload_honours_limit(self):
        tests = [
            {"name": "test_%d" % i, "model_dir": "/fake/%d" % i}
            for i in range(5)
        ]

        def fake_run(model_dir, backend, rtol, atol):
            return {"success": True, "error": "", "error_step": ""}

        payload = rbc.build_payload(
            kind="node",
            limit=2,
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        self.assertEqual(len(payload["tests"]), 2)
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 2, "fail": 0},
                "reference": {"pass": 2, "fail": 0},
            },
        )

    def test_build_payload_captures_unhandled_runner_exceptions(self):
        tests = [{"name": "boom", "model_dir": "/fake/boom"}]

        def fake_run(model_dir, backend, rtol, atol):
            raise RuntimeError("unexpected")

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {},
        )
        row = payload["tests"][0]
        self.assertFalse(row["onnxruntime"])
        self.assertFalse(row["reference"])
        self.assertEqual(row["onnxruntime_error"], "unexpected")
        self.assertEqual(row["reference_error_step"], "run")
        self.assertEqual(
            payload["totals"],
            {
                "onnxruntime": {"pass": 0, "fail": 1},
                "reference": {"pass": 0, "fail": 1},
            },
        )

    def test_run_test_with_backend_unknown_backend(self):
        result = rbc.run_test_with_backend("/does/not/matter", "totally-unknown")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "load")
        self.assertIn("unknown backend", result["error"])

    def test_compare_outputs_detects_shape_mismatch(self):
        import numpy as np

        msg = rbc._compare_outputs(
            [np.zeros((2, 3))], [np.zeros((2, 2))], rtol=1e-3, atol=1e-4
        )
        self.assertIsNotNone(msg)
        self.assertIn("shape mismatch", msg)

    def test_compare_outputs_detects_count_mismatch(self):
        import numpy as np

        msg = rbc._compare_outputs(
            [np.zeros(3), np.zeros(3)],
            [np.zeros(3)],
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertIsNotNone(msg)
        self.assertIn("count mismatch", msg)

    def test_compare_outputs_accepts_close_values(self):
        import numpy as np

        self.assertIsNone(
            rbc._compare_outputs(
                [np.array([1.0, 2.0])],
                [np.array([1.0 + 1e-7, 2.0])],
                rtol=1e-3,
                atol=1e-4,
            )
        )

    def test_compare_outputs_handles_string_outputs(self):
        import numpy as np

        self.assertIsNone(
            rbc._compare_outputs(
                [np.array(["a", "b"])],
                [np.array(["a", "b"])],
                rtol=1e-3,
                atol=1e-4,
            )
        )
        msg = rbc._compare_outputs(
            [np.array(["a", "b"])],
            [np.array(["a", "c"])],
            rtol=1e-3,
            atol=1e-4,
        )
        self.assertIsNotNone(msg)

    def test_write_payload_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "onnx-light", "backend_test_coverage.json")
            payload = {"date": "2024-01-01T00:00:00Z", "tests": []}
            rbc.write_payload(json_path, payload)
            with open(json_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), payload)

    def test_main_writes_cache_file(self):
        original_build = rbc.build_payload

        def fake_build(**kwargs):
            return {
                "date": "2024-01-01T00:00:00Z",
                "kind": kwargs.get("kind", "node"),
                "tolerances": {"rtol": 1e-3, "atol": 1e-4},
                "versions": {},
                "totals": {
                    "onnxruntime": {"pass": 1, "fail": 0},
                    "reference": {"pass": 1, "fail": 0},
                },
                "tests": [
                    {"name": "test_x", "onnxruntime": True, "reference": True}
                ],
            }

        rbc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rbc.main(["--cache-dir", tmp])
                self.assertEqual(code, 0)
                with open(
                    os.path.join(tmp, "onnx-light", "backend_test_coverage.json"),
                    encoding="utf-8",
                ) as fh:
                    payload = json.load(fh)
                self.assertEqual(payload["tests"][0]["name"], "test_x")
                self.assertEqual(payload["kind"], "node")
        finally:
            rbc.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = rbc.build_payload

        def fake_build(**kwargs):
            raise RuntimeError("boom")

        rbc.build_payload = fake_build
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rbc.main(["--cache-dir", tmp])
                self.assertEqual(code, 1)
                self.assertFalse(
                    os.path.exists(
                        os.path.join(
                            tmp, "onnx-light", "backend_test_coverage.json"
                        )
                    )
                )
        finally:
            rbc.build_payload = original_build


if __name__ == "__main__":
    unittest.main()
