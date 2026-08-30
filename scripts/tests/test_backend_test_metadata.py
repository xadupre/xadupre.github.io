"""Tests for backend-test metadata compatibility helpers."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backend_test_metadata import kind_name, tag_name


class TestMetadataName(unittest.TestCase):
    def test_preserves_legacy_strings(self):
        self.assertEqual(kind_name("node"), "node")
        self.assertEqual(tag_name("inference"), "inference")
        self.assertEqual(tag_name(None), "")

    def test_converts_native_enums_with_onnx_light_api(self):
        node = object()
        inference = object()
        backend = types.ModuleType(
            "onnx_light.onnx_py._onnxpybackend.backend_test"
        )
        backend.test_case_kind_name = lambda value: (
            "node" if value is node else "model"
        )
        backend.test_case_tag_name = lambda value: (
            "inference" if value is inference else ""
        )
        modules = {
            "onnx_light": types.ModuleType("onnx_light"),
            "onnx_light.onnx_py": types.ModuleType("onnx_light.onnx_py"),
            "onnx_light.onnx_py._onnxpybackend": types.ModuleType(
                "onnx_light.onnx_py._onnxpybackend"
            ),
            "onnx_light.onnx_py._onnxpybackend.backend_test": backend,
        }
        with patch.dict(sys.modules, modules):
            self.assertEqual(kind_name(node), "node")
            self.assertEqual(tag_name(inference), "inference")


if __name__ == "__main__":
    unittest.main()
