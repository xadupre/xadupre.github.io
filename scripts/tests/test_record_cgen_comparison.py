"""Tests for ``scripts.record_cgen_comparison``."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import unittest.mock as mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_cgen_comparison as rcc  # noqa: E402

try:
    import onnx
    from onnx import helper, TensorProto

    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

# Opset version used in test model helpers
_TEST_OPSET_VERSION = 20


class TestParseSupportOps(unittest.TestCase):
    def test_basic_table(self):
        content = (
            "| Operator | Supported |\n"
            "| --- | --- |\n"
            "| Abs | ✅ |\n"
            "| Affine | ❌ |\n"
            "| ai.onnx.ml.LabelEncoder | ✅ |\n"
        )
        rows = rcc.parse_support_ops(content)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], {"domain": "ai.onnx", "name": "Abs", "in_cgen": True})
        self.assertEqual(
            rows[1], {"domain": "ai.onnx", "name": "Affine", "in_cgen": False}
        )
        self.assertEqual(
            rows[2], {"domain": "ai.onnx.ml", "name": "LabelEncoder", "in_cgen": True}
        )

    def test_skips_header_row(self):
        content = "| Operator | Supported |\n| Abs | ✅ |\n"
        rows = rcc.parse_support_ops(content)
        # "Operator" should be skipped
        self.assertNotIn("Operator", [r["name"] for r in rows])


class TestMergeRows(unittest.TestCase):
    def _make_light(self, name, in_onnx_light=True, tests=5):
        return {
            "domain": "ai.onnx",
            "name": name,
            "in_onnx_light": in_onnx_light,
            "onnx_light_backend_tests": tests,
        }

    def _make_cgen(self, name, in_cgen=True):
        return {"domain": "ai.onnx", "name": name, "in_cgen": in_cgen}

    def test_merge_both_sides(self):
        light = [self._make_light("Abs"), self._make_light("Conv")]
        cgen = [self._make_cgen("Abs"), self._make_cgen("Add")]
        rows = rcc.merge_rows(cgen, light)
        by_name = {r["name"]: r for r in rows}
        self.assertIn("Abs", by_name)
        self.assertIn("Conv", by_name)
        self.assertIn("Add", by_name)
        self.assertTrue(by_name["Abs"]["in_onnx_light"])
        self.assertTrue(by_name["Abs"]["in_cgen"])
        self.assertTrue(by_name["Conv"]["in_onnx_light"])
        self.assertFalse(by_name["Conv"]["in_cgen"])
        self.assertFalse(by_name["Add"]["in_onnx_light"])
        self.assertTrue(by_name["Add"]["in_cgen"])

    def test_merge_with_source_maps(self):
        light = [self._make_light("Abs")]
        cgen = [self._make_cgen("Abs")]
        onnx_light_map = {"abs": "https://raw.example.com/abs.cc"}
        cgen_map = {"abs": "https://raw.example.com/abs_op.c.j2"}
        rows = rcc.merge_rows(cgen, light, onnx_light_map, cgen_map)
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["onnx_light_source_url"], "https://raw.example.com/abs.cc"
        )
        self.assertEqual(
            rows[0]["cgen_source_url"], "https://raw.example.com/abs_op.c.j2"
        )

    def test_merge_with_source_code_map(self):
        light = [self._make_light("Abs")]
        cgen = [self._make_cgen("Abs")]
        cgen_code_map = {("ai.onnx", "Abs"): "/* generated C source */"}
        rows = rcc.merge_rows(cgen, light, cgen_source_code_map=cgen_code_map)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cgen_source_code"], "/* generated C source */")

    def test_merge_no_source_code_when_not_in_map(self):
        light = [self._make_light("Abs")]
        cgen = [self._make_cgen("Abs")]
        rows = rcc.merge_rows(cgen, light, cgen_source_code_map={})
        self.assertNotIn("cgen_source_code", rows[0])

    def test_merge_no_source_url_when_not_found(self):
        light = [self._make_light("Abs")]
        cgen = []
        rows = rcc.merge_rows(cgen, light, onnx_light_source_map={}, cgen_source_map={})
        self.assertNotIn("onnx_light_source_url", rows[0])
        self.assertNotIn("cgen_source_url", rows[0])

    def test_merge_no_source_maps(self):
        light = [self._make_light("Abs")]
        cgen = [self._make_cgen("Abs")]
        rows = rcc.merge_rows(cgen, light)
        self.assertNotIn("onnx_light_source_url", rows[0])
        self.assertNotIn("cgen_source_url", rows[0])


class TestBuildOnnxLightSourceMap(unittest.TestCase):
    def _entry(self, path):
        return {"type": "blob", "path": path}

    def test_extracts_kernel_files(self):
        tree = [
            self._entry(
                "onnx_light/onnx_extensions/kernels/kernels/math/kernel_abs.cc"
            ),
            self._entry(
                "onnx_light/onnx_extensions/kernels/kernels/nn/kernel_conv.cc"
            ),
            self._entry(
                "onnx_light/onnx_extensions/kernels/kernels/math/include_math_kernels.h"
            ),
            self._entry("some/other/file.cc"),
        ]
        result = rcc.build_onnx_light_source_map(tree)
        self.assertIn("abs", result)
        self.assertIn("conv", result)
        self.assertNotIn("include_math_kernels", result)
        self.assertNotIn("some", result)
        self.assertIn(
            "raw.githubusercontent.com/xadupre/onnx-light/main/",
            result["abs"],
        )

    def test_ignores_non_blob_entries(self):
        tree = [
            {
                "type": "tree",
                "path": "onnx_light/onnx_extensions/kernels/kernels/math",
            }
        ]
        result = rcc.build_onnx_light_source_map(tree)
        self.assertEqual(result, {})


class TestBuildCgenSourceMap(unittest.TestCase):
    def _entry(self, path):
        return {"type": "blob", "path": path}

    def test_extracts_template_files(self):
        tree = [
            self._entry("src/emx_onnx_cgen/templates/conv_op.c.j2"),
            self._entry("src/emx_onnx_cgen/templates/batch_norm_op.c.j2"),
            self._entry("src/emx_onnx_cgen/templates/README.md"),
        ]
        result = rcc.build_cgen_source_map(tree)
        self.assertIn("conv", result)
        self.assertIn("batch_norm", result)
        self.assertNotIn("readme", result)
        self.assertIn(
            "raw.githubusercontent.com/emmtrix/emx-onnx-cgen/main/",
            result["conv"],
        )


class TestFindSourceUrls(unittest.TestCase):
    def setUp(self):
        self.onnx_light_map = {
            "abs": "https://raw/abs.cc",
            "averagepool": "https://raw/averagepool.cc",
            "batchnormalization": "https://raw/batchnorm.cc",
        }
        self.cgen_map = {
            "conv": "https://raw/conv.c.j2",
            "batch_norm": "https://raw/batch_norm.c.j2",
        }

    def test_exact_lowercase_match(self):
        url = rcc.find_onnx_light_source_url("Abs", self.onnx_light_map)
        self.assertEqual(url, "https://raw/abs.cc")

    def test_exact_lowercase_match_multiword(self):
        url = rcc.find_onnx_light_source_url("AveragePool", self.onnx_light_map)
        self.assertEqual(url, "https://raw/averagepool.cc")

    def test_camel_to_snake_fallback(self):
        # "batch_norm" is not in onnx_light_map, but exact lowercase is
        url = rcc.find_onnx_light_source_url("BatchNormalization", self.onnx_light_map)
        self.assertEqual(url, "https://raw/batchnorm.cc")

    def test_not_found_returns_none(self):
        url = rcc.find_onnx_light_source_url("UnknownOp", self.onnx_light_map)
        self.assertIsNone(url)

    def test_cgen_snake_case_match(self):
        url = rcc.find_cgen_source_url("BatchNorm", self.cgen_map)
        self.assertEqual(url, "https://raw/batch_norm.c.j2")

    def test_cgen_exact_match(self):
        url = rcc.find_cgen_source_url("conv", self.cgen_map)
        self.assertEqual(url, "https://raw/conv.c.j2")

    def test_cgen_not_found(self):
        url = rcc.find_cgen_source_url("UnknownOp", self.cgen_map)
        self.assertIsNone(url)


class TestCamelToSnake(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(rcc._camel_to_snake("Abs"), "abs")
        self.assertEqual(
            rcc._camel_to_snake("BatchNormalization"), "batch_normalization"
        )
        self.assertEqual(rcc._camel_to_snake("AveragePool"), "average_pool")
        self.assertEqual(rcc._camel_to_snake("Conv"), "conv")


class TestComputeTotals(unittest.TestCase):
    def test_all_categories(self):
        rows = [
            {"in_onnx_light": True, "in_cgen": True},
            {"in_onnx_light": True, "in_cgen": False},
            {"in_onnx_light": False, "in_cgen": True},
            {"in_onnx_light": False, "in_cgen": False},
        ]
        totals = rcc.compute_totals(rows)
        self.assertEqual(totals["onnx_light"], 2)
        self.assertEqual(totals["cgen"], 2)
        self.assertEqual(totals["both"], 1)
        self.assertEqual(totals["only_onnx_light"], 1)
        self.assertEqual(totals["only_cgen"], 1)
        self.assertEqual(totals["neither"], 1)


@unittest.skipUnless(_HAS_ONNX, "onnx is required for these tests")
class TestBuildOpToTestModelMap(unittest.TestCase):
    def _make_model(self, op_type: str, domain: str = ""):
        node = helper.make_node(op_type, ["x"], ["y"])
        node.domain = domain
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", _TEST_OPSET_VERSION)]
        )

    def test_single_node_models_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test_abs directory
            abs_dir = os.path.join(tmpdir, "test_abs")
            os.makedirs(abs_dir)
            onnx.save(self._make_model("Abs"), os.path.join(abs_dir, "model.onnx"))

            result = rcc.build_op_to_test_model_map(tmpdir)
            self.assertIn(("ai.onnx", "Abs"), result)
            self.assertTrue(result[("ai.onnx", "Abs")].endswith("model.onnx"))

    def test_multi_node_models_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a two-node model → should be skipped
            multi_dir = os.path.join(tmpdir, "test_multi")
            os.makedirs(multi_dir)
            node1 = helper.make_node("Abs", ["x"], ["y"])
            node2 = helper.make_node("Relu", ["y"], ["z"])
            graph = helper.make_graph(
                [node1, node2],
                "g",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])],
            )
            model = helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", _TEST_OPSET_VERSION)]
            )
            onnx.save(model, os.path.join(multi_dir, "model.onnx"))

            result = rcc.build_op_to_test_model_map(tmpdir)
            # Neither Abs nor Relu should be indexed from this multi-node model
            self.assertNotIn(("ai.onnx", "Abs"), result)
            self.assertNotIn(("ai.onnx", "Relu"), result)

    def test_missing_model_file_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Directory without model.onnx
            empty_dir = os.path.join(tmpdir, "test_noop")
            os.makedirs(empty_dir)
            result = rcc.build_op_to_test_model_map(tmpdir)
            self.assertEqual(result, {})


@unittest.skipUnless(_HAS_ONNX, "onnx is required for these tests")
class TestMaterializeOnnxLightNodeDir(unittest.TestCase):
    def _make_model(self, op_type: str):
        node = helper.make_node(op_type, ["x"], ["y"])
        graph = helper.make_graph(
            [node],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", _TEST_OPSET_VERSION)]
        )

    def test_writes_single_node_models_from_catalog(self):
        import sys
        import types

        abs_model = self._make_model("Abs")
        multi_node = helper.make_node("Relu", ["x"], ["t"])
        multi_node2 = helper.make_node("Abs", ["t"], ["y"])
        multi_graph = helper.make_graph(
            [multi_node, multi_node2],
            "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        multi_model = helper.make_model(
            multi_graph, opset_imports=[helper.make_opsetid("", _TEST_OPSET_VERSION)]
        )

        cases = {
            "test_abs": types.SimpleNamespace(model=abs_model),
            "test_multi": types.SimpleNamespace(model=multi_model),
            "test_no_model": types.SimpleNamespace(model=None),
        }

        fake_module = types.ModuleType("onnx_light.onnx_lib.backend.test.case")
        fake_module.collect_test_case = lambda include_big=False: cases
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
            with tempfile.TemporaryDirectory() as tmpdir:
                result = rcc._materialize_onnx_light_node_dir(tmpdir)
                self.assertEqual(result, tmpdir)
                # Only the single-node case is materialized.
                self.assertTrue(
                    os.path.exists(os.path.join(tmpdir, "test_abs", "model.onnx"))
                )
                self.assertFalse(os.path.isdir(os.path.join(tmpdir, "test_multi")))
                self.assertFalse(os.path.isdir(os.path.join(tmpdir, "test_no_model")))
                # The layout is understood by build_op_to_test_model_map.
                op_map = rcc.build_op_to_test_model_map(tmpdir)
                self.assertIn(("ai.onnx", "Abs"), op_map)
        finally:
            for name, mod in saved.items():
                if mod is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = mod

    def test_returns_none_when_onnx_light_missing(self):
        import sys

        saved = sys.modules.get("onnx_light")
        # Insert a sentinel that has no backend.test.case submodule so the
        # import inside _materialize_onnx_light_node_dir fails cleanly.
        import types

        sys.modules["onnx_light"] = types.ModuleType("onnx_light")
        sys.modules.pop("onnx_light.onnx_lib.backend.test.case", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                self.assertIsNone(rcc._materialize_onnx_light_node_dir(tmpdir))
        finally:
            if saved is None:
                sys.modules.pop("onnx_light", None)
            else:
                sys.modules["onnx_light"] = saved


class TestGenerateCgenSourceForOp(unittest.TestCase):
    def test_returns_none_when_tool_missing(self):
        with mock.patch("shutil.which", return_value=None):
            result = rcc.generate_cgen_source_for_op("/nonexistent/model.onnx")
        self.assertIsNone(result)

    def test_returns_none_on_compile_failure(self):
        with mock.patch("shutil.which", return_value="/usr/bin/emx-onnx-cgen"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=1)
                result = rcc.generate_cgen_source_for_op("/nonexistent/model.onnx")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
