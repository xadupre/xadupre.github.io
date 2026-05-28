"""Tests for ``scripts.record_onnx_stats``."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_stats as ros  # noqa: E402


class TestRecordOnnxStats(unittest.TestCase):
    def test_python_version_from_filename(self):
        self.assertEqual(
            ros._python_version_from_filename(
                "onnx-1.21.0-cp312-abi3-manylinux_2_27_x86_64.whl"
            ),
            (3, 12),
        )
        self.assertEqual(
            ros._python_version_from_filename(
                "onnx-1.21.0-cp39-cp39-manylinux_2_17_x86_64.whl"
            ),
            (3, 9),
        )
        self.assertIsNone(
            ros._python_version_from_filename("onnx-1.21.0.tar.gz")
        )

    def test_pick_latest_linux_wheel(self):
        files = [
            {
                "packagetype": "sdist",
                "filename": "onnx-1.21.0.tar.gz",
                "size": 1,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp310-cp310-manylinux_2_27_x86_64.whl",
                "size": 100,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-abi3-manylinux_2_27_x86_64.whl",
                "size": 200,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp314-cp314t-manylinux_2_27_x86_64.whl",
                "size": 300,
            },
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-cp312-win_amd64.whl",
                "size": 999,
            },
        ]
        wheel = ros.pick_latest_linux_wheel(files)
        self.assertIsNotNone(wheel)
        self.assertEqual(wheel["size"], 300)
        self.assertIn("cp314", wheel["filename"])

    def test_pick_latest_linux_wheel_no_match(self):
        files = [
            {
                "packagetype": "bdist_wheel",
                "filename": "onnx-1.21.0-cp312-cp312-win_amd64.whl",
                "size": 1,
            },
        ]
        self.assertIsNone(ros.pick_latest_linux_wheel(files))

    def test_append_row_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "onnx", "stats.csv")
            row = {field: f"v-{field}" for field in ros.CSV_FIELDS}
            ros.append_row(csv_path, row)
            ros.append_row(csv_path, row)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["filename"], "v-filename")

    def test_count_supported_types_excludes_undefined(self):
        # The actual count depends on the installed onnx version, but it must
        # be at least the number of types present in onnx 1.0 (UNDEFINED +
        # 16 known types -> at least 16 after excluding UNDEFINED).
        n = ros.count_supported_types()
        self.assertGreaterEqual(n, 16)

    def test_count_node_test_cases_positive(self):
        # The installed onnx package ships hundreds of node test cases.
        self.assertGreater(ros.count_node_test_cases(), 0)


if __name__ == "__main__":
    unittest.main()
