"""Tests for ``scripts.record_pypi_downloads``."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_pypi_downloads as rpd  # noqa: E402


class TestRecordPypiDownloads(unittest.TestCase):
    def test_default_packages_cover_issue_list(self):
        expected = {
            "onnx",
            "onnxruntime",
            "onnxruntime-genai",
            "skl2onnx",
            "onnxmltools",
            "onnxscript",
            "ir-py",
            "tf2onnx",
        }
        self.assertEqual(set(rpd.DEFAULT_PACKAGES), expected)

    def test_build_row_uses_data_fields(self):
        data = {"last_day": 1, "last_week": 23, "last_month": 456}
        row = rpd.build_row("onnx", data)
        self.assertEqual(row["package"], "onnx")
        self.assertEqual(row["last_day"], "1")
        self.assertEqual(row["last_week"], "23")
        self.assertEqual(row["last_month"], "456")
        # ``date`` should be an ISO-8601 UTC timestamp ending with ``Z``.
        self.assertTrue(row["date"].endswith("Z"))
        self.assertEqual(set(row), set(rpd.CSV_FIELDS))

    def test_build_row_handles_missing_fields(self):
        row = rpd.build_row("ir-py", {})
        self.assertEqual(row["last_day"], "")
        self.assertEqual(row["last_week"], "")
        self.assertEqual(row["last_month"], "")

    def test_append_row_creates_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "onnx", "downloads.csv")
            row = {field: f"v-{field}" for field in rpd.CSV_FIELDS}
            rpd.append_row(csv_path, row)
            rpd.append_row(csv_path, row)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["package"], "v-package")
            self.assertEqual(rows[0]["last_week"], "v-last_week")

    def test_record_package_writes_csv(self):
        captured: list[str] = []

        def fake_fetch(package):
            captured.append(package)
            return {"last_day": 10, "last_week": 70, "last_month": 300}

        original = rpd.fetch_recent_downloads
        rpd.fetch_recent_downloads = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                row = rpd.record_package("skl2onnx", tmp)
                self.assertEqual(captured, ["skl2onnx"])
                self.assertEqual(row["last_week"], "70")
                csv_path = os.path.join(tmp, "skl2onnx", "downloads.csv")
                self.assertTrue(os.path.isfile(csv_path))
                with open(csv_path, encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["package"], "skl2onnx")
                self.assertEqual(rows[0]["last_month"], "300")
        finally:
            rpd.fetch_recent_downloads = original

    def test_main_iterates_over_packages_and_handles_failures(self):
        calls: list[str] = []

        def fake_fetch(package):
            calls.append(package)
            if package == "ir-py":
                raise RuntimeError("boom")
            return {"last_day": 1, "last_week": 2, "last_month": 3}

        original = rpd.fetch_recent_downloads
        rpd.fetch_recent_downloads = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rpd.main(
                    [
                        "--cache-dir",
                        tmp,
                        "--package",
                        "onnx",
                        "--package",
                        "ir-py",
                        "--package",
                        "tf2onnx",
                    ]
                )
                self.assertEqual(code, 1)
                self.assertEqual(calls, ["onnx", "ir-py", "tf2onnx"])
                self.assertTrue(
                    os.path.isfile(os.path.join(tmp, "onnx", "downloads.csv"))
                )
                self.assertTrue(
                    os.path.isfile(os.path.join(tmp, "tf2onnx", "downloads.csv"))
                )
                self.assertFalse(
                    os.path.isfile(os.path.join(tmp, "ir-py", "downloads.csv"))
                )
        finally:
            rpd.fetch_recent_downloads = original

    def test_main_returns_zero_on_success(self):
        def fake_fetch(package):
            return {"last_day": 1, "last_week": 2, "last_month": 3}

        original = rpd.fetch_recent_downloads
        rpd.fetch_recent_downloads = fake_fetch
        try:
            with tempfile.TemporaryDirectory() as tmp:
                code = rpd.main(
                    ["--cache-dir", tmp, "--package", "onnxscript"]
                )
                self.assertEqual(code, 0)
        finally:
            rpd.fetch_recent_downloads = original


if __name__ == "__main__":
    unittest.main()
