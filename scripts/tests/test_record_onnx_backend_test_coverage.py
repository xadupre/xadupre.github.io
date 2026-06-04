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
            versions={"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        self.assertEqual(row["name"], "test_relu")
        self.assertTrue(row["onnxruntime"])
        self.assertFalse(row["reference"])
        self.assertNotIn("onnxruntime_error", row)
        self.assertNotIn("onnxruntime_error_step", row)
        self.assertEqual(row["reference_error"], "boom")
        self.assertEqual(row["reference_error_step"], "run")
        # Passing backend records its last-pass date + matching package version.
        self.assertEqual(row["onnxruntime_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["onnxruntime_last_pass_version"], "1.20.0")
        # Failing backend has no recorded last-pass when there is no history.
        self.assertNotIn("reference_last_pass_date", row)
        self.assertNotIn("reference_last_pass_version", row)

    def test_row_from_results_carries_over_previous_last_pass_on_failure(self):
        previous = {
            "name": "test_relu",
            "reference_last_pass_date": "2024-01-02T03:04:05Z",
            "reference_last_pass_version": "1.16.0",
        }
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
            previous=previous,
            versions={"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now_iso="2024-05-06T07:08:09Z",
        )
        # Current pass refreshes the onnxruntime entry, prior reference pass is kept.
        self.assertEqual(row["onnxruntime_last_pass_date"], "2024-05-06T07:08:09Z")
        self.assertEqual(row["onnxruntime_last_pass_version"], "1.20.0")
        self.assertEqual(row["reference_last_pass_date"], "2024-01-02T03:04:05Z")
        self.assertEqual(row["reference_last_pass_version"], "1.16.0")

    def test_build_payload_runs_every_backend_and_aggregates_totals(self):
        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": [("in_a", "out_a")]},
            {"name": "test_b", "model": "model_b", "data_sets": [("in_b", "out_b")]},
            {"name": "test_c", "model": "model_c", "data_sets": [("in_c", "out_c")]},
        ]
        # Map of (model, backend) -> result dict
        outcomes = {
            ("model_a", "onnxruntime"): {
                "success": True,
                "error": "",
                "error_step": "",
            },
            ("model_a", "reference"): {"success": True, "error": "", "error_step": ""},
            ("model_b", "onnxruntime"): {
                "success": True,
                "error": "",
                "error_step": "",
            },
            ("model_b", "reference"): {
                "success": False,
                "error": "not implemented",
                "error_step": "run",
            },
            ("model_c", "onnxruntime"): {
                "success": False,
                "error": "kernel missing",
                "error_step": "load",
            },
            ("model_c", "reference"): {"success": True, "error": "", "error_step": ""},
        }

        def fake_run(model, data_sets, backend, rtol, atol):
            return outcomes[(model, backend)]

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
            {"name": f"test_{i}", "model": f"model_{i}", "data_sets": []}
            for i in range(5)
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
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
        tests = [{"name": "boom", "model": "model_boom", "data_sets": []}]

        def fake_run(model, data_sets, backend, rtol, atol):
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

    def test_build_payload_carries_previous_last_pass_for_failing_tests(self):
        import datetime as dt

        tests = [
            {"name": "test_a", "model": "model_a", "data_sets": []},
            {"name": "test_b", "model": "model_b", "data_sets": []},
        ]

        def fake_run(model, data_sets, backend, rtol, atol):
            if model == "model_a":
                return {"success": True, "error": "", "error_step": ""}
            return {"success": False, "error": "boom", "error_step": "run"}

        previous = {
            "tests": [
                {
                    "name": "test_b",
                    "onnxruntime_last_pass_date": "2024-01-01T00:00:00Z",
                    "onnxruntime_last_pass_version": "1.18.0",
                    "reference_last_pass_date": "2024-02-02T00:00:00Z",
                    "reference_last_pass_version": "1.16.0",
                },
            ],
        }

        payload = rbc.build_payload(
            kind="node",
            discover=lambda kind: tests,
            run=fake_run,
            versions=lambda: {"onnxruntime": "1.20.0", "onnx": "1.17.0"},
            now=dt.datetime(2024, 5, 6, 7, 8, 9, tzinfo=dt.timezone.utc),
            previous=previous,
        )
        by_name = {row["name"]: row for row in payload["tests"]}
        # Currently-passing test gets a fresh last-pass timestamp.
        self.assertEqual(
            by_name["test_a"]["onnxruntime_last_pass_date"],
            "2024-05-06T07:08:09Z",
        )
        self.assertEqual(by_name["test_a"]["onnxruntime_last_pass_version"], "1.20.0")
        self.assertEqual(by_name["test_a"]["reference_last_pass_version"], "1.17.0")
        # Currently-failing test keeps the previously recorded last-pass info.
        self.assertEqual(
            by_name["test_b"]["onnxruntime_last_pass_date"],
            "2024-01-01T00:00:00Z",
        )
        self.assertEqual(by_name["test_b"]["onnxruntime_last_pass_version"], "1.18.0")
        self.assertEqual(
            by_name["test_b"]["reference_last_pass_date"],
            "2024-02-02T00:00:00Z",
        )

    def test_load_previous_payload_handles_missing_and_malformed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.json")
            self.assertEqual(rbc.load_previous_payload(missing), {})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("not json")
            self.assertEqual(rbc.load_previous_payload(bad), {})
            ok = os.path.join(tmp, "ok.json")
            with open(ok, "w", encoding="utf-8") as fh:
                json.dump({"tests": [{"name": "x"}]}, fh)
            self.assertEqual(rbc.load_previous_payload(ok), {"tests": [{"name": "x"}]})

    def test_run_test_with_backend_unknown_backend(self):
        result = rbc.run_test_with_backend(None, [], "totally-unknown")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_step"], "load")
        self.assertIn("unknown backend", result["error"])

    def test_discover_node_tests_loads_from_onnx_light(self):
        """``discover_node_tests`` must materialise onnx-light test cases."""
        import types

        import numpy as np
        import onnx
        from onnx import helper

        # Build a tiny in-memory ONNX model representing ``y = Relu(x)`` so
        # the fake ``collect_test_case`` mirrors the shape of onnx-light's
        # output (TestCase with a ``ModelProto`` and ``data_sets``).
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
        inputs = [np.array([-1.0, 2.0], dtype=np.float32)]
        outputs = [np.array([0.0, 2.0], dtype=np.float32)]

        node_tc = types.SimpleNamespace(
            name="test_relu_light",
            kind="node",
            model=model,
            data_sets=[(inputs, outputs)],
            model_dir=None,
        )
        # A test from a different kind must be filtered out.
        simple_tc = types.SimpleNamespace(
            name="test_simple_other",
            kind="simple",
            model=model,
            data_sets=[(inputs, outputs)],
            model_dir=None,
        )
        fake_module = types.ModuleType("onnx_light.backend.test.case")
        fake_module.collect_test_case = lambda: {
            "test_relu_light": node_tc,
            "test_simple_other": simple_tc,
        }
        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.backend", types.ModuleType("onnx_light.backend")),
            (
                "onnx_light.backend.test",
                types.ModuleType("onnx_light.backend.test"),
            ),
            ("onnx_light.backend.test.case", fake_module),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            discovered = rbc.discover_node_tests(kind="node")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        self.assertEqual(len(discovered), 1)
        entry = discovered[0]
        self.assertEqual(entry["name"], "test_relu_light")
        # The model is kept in memory as an ``onnx.ModelProto``.
        self.assertIsInstance(entry["model"], onnx.ModelProto)
        self.assertEqual(entry["model"].graph.node[0].op_type, "Relu")
        # Data sets are kept in memory as numpy arrays, ready to feed to
        # the backends without any disk round-trip.
        self.assertEqual(len(entry["data_sets"]), 1)
        loaded_inputs, loaded_outputs = entry["data_sets"][0]
        np.testing.assert_array_equal(loaded_inputs[0], inputs[0])
        np.testing.assert_array_equal(loaded_outputs[0], outputs[0])

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
                "tests": [{"name": "test_x", "onnxruntime": True, "reference": True}],
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
                        os.path.join(tmp, "onnx-light", "backend_test_coverage.json")
                    )
                )
        finally:
            rbc.build_payload = original_build


if __name__ == "__main__":
    unittest.main()
