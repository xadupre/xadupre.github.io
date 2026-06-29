"""Tests for ``scripts.record_cgen_comparison``."""

from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_cgen_comparison as rcc  # noqa: E402


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
        self.assertEqual(rows[1], {"domain": "ai.onnx", "name": "Affine", "in_cgen": False})
        self.assertEqual(rows[2], {"domain": "ai.onnx.ml", "name": "LabelEncoder", "in_cgen": True})

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
        self.assertEqual(rows[0]["onnx_light_source_url"], "https://raw.example.com/abs.cc")
        self.assertEqual(rows[0]["cgen_source_url"], "https://raw.example.com/abs_op.c.j2")

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
            self._entry("onnx_light/onnx_kernels/kernels/math/kernel_abs.cc"),
            self._entry("onnx_light/onnx_kernels/kernels/nn/kernel_conv.cc"),
            self._entry("onnx_light/onnx_kernels/kernels/math/include_math_kernels.h"),
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
        tree = [{"type": "tree", "path": "onnx_light/onnx_kernels/kernels/math"}]
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
        self.assertEqual(rcc._camel_to_snake("BatchNormalization"), "batch_normalization")
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


if __name__ == "__main__":
    unittest.main()
