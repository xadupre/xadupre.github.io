"""Tests for ``scripts.record_onnx_release_after_coverage``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_release_after_coverage as rac  # noqa: E402


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
    def __init__(self, nodes, inputs=None, outputs=None, initializers=None):
        self.node = nodes
        self.input = inputs or []
        self.output = outputs or []
        self.initializer = initializers or []
        self.value_info = []


class _FakeModel:
    def __init__(self, nodes, inputs=None, outputs=None, initializers=None):
        self.graph = _FakeGraph(
            nodes, inputs=inputs, outputs=outputs, initializers=initializers
        )


class _FakeTestCase:
    def __init__(self, name, model, tag=""):
        self.name = name
        self.model = model
        self.tag = tag


class TestRecordOnnxReleaseAfterCoverage(unittest.TestCase):
    def test_node_metadata_filters_unrelated_keys(self):
        node = _FakeNode(
            "Abs",
            {
                "onnx_light.inplace_reuse": "0:0:equal",
                "onnx_light.release_after": "A",
                "onnx_light.not_used_after": "X;W",
                "ignored": "x",
            },
        )
        self.assertEqual(
            rac._node_metadata(node),
            {
                "onnx_light.release_after": "A",
                "onnx_light.not_used_after": "X;W",
            },
        )

    def test_value_metadata_filters_unrelated_keys(self):
        vi = _FakeValueInfo(
            "X",
            {
                "onnx_light.value_tags": "shape",
                "onnx_light.release_after": "A",
                "onnx_light.unlocked": "Y",
                "ignored": "x",
            },
        )
        self.assertEqual(
            rac._value_metadata(vi),
            {
                "onnx_light.value_tags": "shape",
                "onnx_light.release_after": "A",
                "onnx_light.unlocked": "Y",
            },
        )

    def test_graph_value_snapshot_collects_inputs_outputs_initializers(self):
        inp = _FakeValueInfo("X", {"onnx_light.value_tags": "weight"})
        out = _FakeValueInfo("Y", {})
        init = _FakeTensorProto("W", {"onnx_light.value_tags": "weight"})
        model = _FakeModel([], inputs=[inp], outputs=[out], initializers=[init])
        snapshot = rac._graph_value_snapshot(model)
        names = [s["name"] for s in snapshot]
        self.assertIn("X", names)
        self.assertIn("Y", names)
        self.assertIn("W", names)
        x_entry = next(s for s in snapshot if s["name"] == "X")
        self.assertEqual(x_entry["kind"], "input")
        y_entry = next(s for s in snapshot if s["name"] == "Y")
        self.assertEqual(y_entry["kind"], "output")
        w_entry = next(s for s in snapshot if s["name"] == "W")
        self.assertEqual(w_entry["kind"], "initializer")

    def test_graph_value_snapshot_ignores_initializer_metadata(self):
        inp = _FakeValueInfo("X", {"onnx_light.unlocked": "A"})
        init = _FakeTensorProto("W", {"onnx_light.release_after": "B"})
        model = _FakeModel([], inputs=[inp], initializers=[init])
        snapshot = rac._graph_value_snapshot(model)
        x_entry = next(s for s in snapshot if s["name"] == "X")
        w_entry = next(s for s in snapshot if s["name"] == "W")
        self.assertEqual(x_entry["metadata"], {"onnx_light.unlocked": "A"})
        # Initializers are TensorProto, not ValueInfoProto: their own
        # metadata_props must be ignored.
        self.assertEqual(w_entry["metadata"], {})

    def test_graph_value_snapshot_merges_initializer_metadata_from_value_info(self):
        init = _FakeTensorProto("W", {"onnx_light.release_after": "ignored"})
        model = _FakeModel([], initializers=[init])
        model.graph.value_info = [
            _FakeValueInfo("W", {"onnx_light.release_after": "B"})
        ]
        snapshot = rac._graph_value_snapshot(model)
        w_entry = next(s for s in snapshot if s["name"] == "W")
        self.assertEqual(w_entry["metadata"], {"onnx_light.release_after": "B"})

    def test_graph_value_snapshot_merges_output_metadata_from_value_info(self):
        out = _FakeValueInfo("Y", {})
        model = _FakeModel([], outputs=[out])
        model.graph.value_info = [
            _FakeValueInfo("Y", {"onnx_light.value_tags": "axes"})
        ]

        snapshot = rac._graph_value_snapshot(model)
        y_entry = next(s for s in snapshot if s["name"] == "Y")
        self.assertEqual(y_entry["kind"], "output")
        self.assertEqual(y_entry["metadata"], {"onnx_light.value_tags": "axes"})

    def test_clear_node_metadata_removes_entries(self):
        node = _FakeNode("Abs", {"onnx_light.release_after": "A"})
        rac._clear_node_metadata(node)
        self.assertEqual(node.metadata_props, [])

    def test_score_test_counts_nodes_and_metadata(self):
        row = rac._score_test(
            "test_cc_shape_inference_release_after",
            expected_nodes=[
                {},
                {"onnx_light.release_after": "A"},
                {"onnx_light.release_after": "B"},
            ],
            actual_nodes=[
                {},
                {"onnx_light.release_after": "A"},
                {"onnx_light.release_after": "C"},
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
        self.assertIn("values", row)
        self.assertEqual(row["values"], [])

    def test_score_test_includes_values_section(self):
        expected_values = [
            {
                "name": "X",
                "kind": "input",
                "metadata": {"onnx_light.value_tags": "shape"},
            },
            {"name": "Y", "kind": "output", "metadata": {}},
        ]
        actual_values = [
            {
                "name": "X",
                "kind": "input",
                "metadata": {"onnx_light.value_tags": "shape"},
            },
            {
                "name": "Y",
                "kind": "output",
                "metadata": {"onnx_light.value_tags": "axes"},
            },
        ]
        row = rac._score_test(
            "test_vals",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            expected_values=expected_values,
            actual_values=actual_values,
        )
        self.assertIn("values", row)
        self.assertEqual(len(row["values"]), 2)
        x_val = next(v for v in row["values"] if v["name"] == "X")
        self.assertTrue(x_val["success"])
        y_val = next(v for v in row["values"] if v["name"] == "Y")
        self.assertFalse(y_val["success"])

    def test_score_test_includes_mermaid_when_provided(self):
        row = rac._score_test(
            "test_with_mermaid",
            expected_nodes=[{"onnx_light.release_after": "A"}],
            actual_nodes=[{"onnx_light.release_after": "A"}],
            node_ops=["Abs"],
            mermaid="flowchart TD\n    A --> B",
        )
        self.assertTrue(row["success"])
        self.assertIn("mermaid", row)
        self.assertEqual(row["mermaid"], "flowchart TD\n    A --> B")

    def test_score_test_includes_graph_svg_when_provided(self):
        row = rac._score_test(
            "test_with_svg",
            expected_nodes=[{"onnx_light.release_after": "A"}],
            actual_nodes=[{"onnx_light.release_after": "A"}],
            node_ops=["Abs"],
            graph={"svg": "<svg><rect/></svg>"},
        )
        self.assertIn("graph", row)
        self.assertEqual(row["graph"], {"svg": "<svg><rect/></svg>"})

    def test_score_test_keeps_node_input_output_info(self):
        row = rac._score_test(
            "test_with_io",
            expected_nodes=[{"onnx_light.release_after": "A"}],
            actual_nodes=[{"onnx_light.release_after": "A"}],
            node_ops=["Abs"],
            node_inputs=[["X"]],
            node_outputs=[["Y"]],
        )
        self.assertEqual(row["nodes"][0]["inputs"], ["X"])
        self.assertEqual(row["nodes"][0]["outputs"], ["Y"])

    def test_score_test_omits_mermaid_when_empty(self):
        row = rac._score_test(
            "test_no_mermaid",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            mermaid="",
        )
        self.assertNotIn("mermaid", row)

    def test_score_test_omits_graph_without_svg(self):
        row = rac._score_test(
            "test_no_graph",
            expected_nodes=[],
            actual_nodes=[],
            node_ops=[],
            graph={"nodes": []},
        )
        self.assertNotIn("graph", row)

    def test_build_payload_passes_values(self):
        expected_values = [
            {
                "name": "X",
                "kind": "input",
                "metadata": {"onnx_light.value_tags": "shape"},
            }
        ]
        tests = [
            {
                "name": "test_vals",
                "model": "model_v",
                "expected_nodes": [{"onnx_light.release_after": "A"}],
                "node_ops": ["Abs"],
                "expected_values": expected_values,
            }
        ]

        def fake_run(model):
            return {
                "actual_nodes": [{"onnx_light.release_after": "A"}],
                "actual_values": [
                    {
                        "name": "X",
                        "kind": "input",
                        "metadata": {"onnx_light.value_tags": "shape"},
                    }
                ],
            }

        payload = rac.build_payload(
            tag="release_after",
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
                "expected_nodes": [{"onnx_light.release_after": "A"}],
                "node_ops": ["Add"],
                "mermaid": "flowchart TD\n    in_X --> op_Add --> out_Y",
                "graph": {"svg": "<svg><g/></svg>"},
            }
        ]

        def fake_run(model):
            return {"actual_nodes": [{"onnx_light.release_after": "A"}]}

        payload = rac.build_payload(
            tag="release_after",
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
                    {"onnx_light.release_after": "A"},
                ],
                "node_ops": ["Abs", "Abs"],
            },
            {
                "name": "test_b",
                "model": "model_b",
                "expected_nodes": [
                    {"onnx_light.release_after": "B"},
                ],
                "node_ops": ["Reshape"],
            },
        ]

        def fake_run(model):
            if model == "model_a":
                return {
                    "actual_nodes": [
                        {},
                        {"onnx_light.release_after": "A"},
                    ]
                }
            return {"actual_nodes": [{"onnx_light.release_after": "X"}]}

        payload = rac.build_payload(
            tag="release_after",
            discover=lambda tag: tests,
            run=fake_run,
            versions=lambda: {"onnx_light": "0.1.0"},
        )

        self.assertEqual(payload["tag"], "release_after")
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
                "expected_nodes": [{"onnx_light.release_after": "A"}],
                "node_ops": ["Abs"],
            }
        ]

        def fake_run(model):
            raise RuntimeError("unexpected")

        payload = rac.build_payload(
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
            "tag": "release_after",
            "versions": {"onnx_light": "0.1.0"},
            "totals": {
                "tests": {"pass": 1, "fail": 0},
                "nodes": {"pass": 2, "fail": 0},
                "metadata": {"pass": 2, "fail": 0},
            },
            "tests": [{"name": "test_a", "success": True}],
        }
        original_build = rac.build_payload
        try:
            rac.build_payload = lambda **kwargs: sample_payload
            with tempfile.TemporaryDirectory() as tmp:
                rc = rac.main(["--cache-dir", tmp])
                self.assertEqual(rc, 0)
                path = os.path.join(tmp, "onnx-light", "release_after_coverage.json")
                self.assertTrue(os.path.isfile(path))
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.assertEqual(payload, sample_payload)
        finally:
            rac.build_payload = original_build

    def test_main_returns_one_on_failure(self):
        original_build = rac.build_payload
        try:

            def fake_build(**kwargs):
                raise RuntimeError("boom")

            rac.build_payload = fake_build
            self.assertEqual(rac.main([]), 1)
        finally:
            rac.build_payload = original_build

    def test_discover_excludes_test_with_metadata_and_wrong_tag(self):
        """Metadata must not bypass the requested backend-case tag."""
        import sys
        import types

        node_with_meta = _FakeNode("Abs", {"onnx_light.release_after": "A"})
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
            discovered = rac.discover_release_after_tests(tag="release_after")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertNotIn("test_tiny_llm", names)
        self.assertNotIn("test_no_meta", names)

    def test_discover_includes_test_with_not_used_after_only(self):
        """Cases whose only node metadata is ``not_used_after`` are kept.

        The release-after algorithm annotates nodes with both
        ``onnx_light.release_after`` and ``onnx_light.not_used_after``; a model
        may carry only the latter (e.g. a single-node graph whose inputs reach
        their last use but which produces no released intermediate). Such a
        case must still appear on the release-after coverage page.
        """
        import sys
        import types

        node_with_meta = _FakeNode("Add", {"onnx_light.not_used_after": "X;W"})
        tc_meta = _FakeTestCase(
            "test_cc_release_initializer_add",
            _FakeModel([node_with_meta]),
            tag="release",
        )
        node_no_meta = _FakeNode("Relu")
        tc_no_meta = _FakeTestCase(
            "test_no_meta",
            _FakeModel([node_no_meta]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_cc_release_initializer_add": tc_meta,
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
            discovered = rac.discover_release_after_tests(tag="release_after")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        by_name = {d["name"]: d for d in discovered}
        self.assertIn("test_cc_release_initializer_add", by_name)
        self.assertNotIn("test_no_meta", by_name)
        self.assertEqual(
            by_name["test_cc_release_initializer_add"]["expected_nodes"],
            [{"onnx_light.not_used_after": "X;W"}],
        )

    def test_discover_excludes_test_with_value_metadata_and_wrong_tag(self):
        """Value metadata must not bypass the requested backend-case tag."""
        import sys
        import types

        # A model with no node-level metadata but with VALUE_METADATA_KEYS on a value
        vi_with_meta = _FakeValueInfo("X", {"onnx_light.value_tags": "shape"})
        node_plain = _FakeNode("Relu")
        model_with_value_meta = _FakeModel([node_plain], inputs=[vi_with_meta])

        tc_value_meta = _FakeTestCase(
            "test_cc_shape_inference_big_qwen3",
            model_with_value_meta,
            tag="model",
        )
        node_no_meta = _FakeNode("Abs")
        tc_no_meta = _FakeTestCase(
            "test_plain_no_meta",
            _FakeModel([node_no_meta]),
            tag="model",
        )

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: {
            "test_cc_shape_inference_big_qwen3": tc_value_meta,
            "test_plain_no_meta": tc_no_meta,
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
            discovered = rac.discover_release_after_tests(tag="release_after")
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        names = [d["name"] for d in discovered]
        self.assertNotIn("test_cc_shape_inference_big_qwen3", names)
        self.assertNotIn("test_plain_no_meta", names)

    def test_release_after_tag_accepts_release_case_alias(self):
        self.assertTrue(rac._matches_requested_tags(("release_after",), ("release",)))
        self.assertFalse(
            rac._matches_requested_tags(("release_after",), ("inference",))
        )

    def test_run_release_after_analysis_uses_onnx_core_module(self):
        """``run_release_after_analysis`` imports
        ``onnx_light.onnx_core.shape_inference`` and drives it with the
        module-level ``compute_shape_model(ctx, model)`` API rather than the
        (non-existent) ``onnx_light.onnx_optim`` submodule.
        """
        import sys
        import types

        calls = []

        class _FakeShapesContext:
            pass

        class _FakeComputeContext:
            def __init__(self):
                self.memory = [1, 2, 3]

            def compute_inplace_reuse_graph(self, graph, ctx):
                calls.append(("compute_inplace_reuse_graph", graph, ctx))

            def write_to_metadata(self, graph):
                calls.append(("write_to_metadata", graph))

        def _fake_compute_shape_model(ctx, model):
            calls.append(("compute_shape_model", ctx, model))

        si = types.ModuleType("onnx_light.onnx_core.shape_inference")
        si.ShapesContext = _FakeShapesContext
        si.ComputeContext = _FakeComputeContext
        si.compute_shape_model = _fake_compute_shape_model

        core = types.ModuleType("onnx_light.onnx_core")
        core.shape_inference = si

        work = _FakeModel([_FakeNode("Abs")])

        parents = [
            ("onnx_light", types.ModuleType("onnx_light")),
            ("onnx_light.onnx_core", core),
            ("onnx_light.onnx_core.shape_inference", si),
        ]
        saved = {name: sys.modules.get(name) for name, _ in parents}
        saved_clone = rac._clone_model
        try:
            for name, mod in parents:
                sys.modules[name] = mod
            rac._clone_model = lambda model: work
            result = rac.run_release_after_analysis(object())
        finally:
            rac._clone_model = saved_clone
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

        # ``compute_shape_model`` must be called as a module-level function with
        # ``(ctx, model)`` and before the in-place reuse computation.
        self.assertEqual(calls[0][0], "compute_shape_model")
        self.assertIsInstance(calls[0][1], _FakeShapesContext)
        self.assertIs(calls[0][2], work)
        names = [c[0] for c in calls]
        self.assertEqual(
            names,
            ["compute_shape_model", "compute_inplace_reuse_graph", "write_to_metadata"],
        )
        self.assertEqual(result["memory"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
