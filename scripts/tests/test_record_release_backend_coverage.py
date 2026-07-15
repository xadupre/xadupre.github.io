"""Tests for ``scripts.record_release_backend_coverage``."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_release_backend_coverage as rbc  # noqa: E402

_SAMPLE_PAYLOAD = {
    "date": "2026-06-29T12:30:07Z",
    "kind": "node",
    "tolerances": {"rtol": 1e-5, "atol": 1e-5},
    "versions": {
        "numpy": "2.5.0",
        "onnx": "1.22.0",
        "onnx_light": "0.1.2",
        "onnxruntime": "1.27.0",
        "yobx": "0.1.0",
    },
    "totals": {
        "onnxruntime": {"pass": 1465, "fail": 300},
        "reference": {"pass": 1742, "fail": 23},
        "onnx_light": {"pass": 1751, "fail": 14},
        "yobx": {"pass": 1742, "fail": 23},
    },
    "tests": [],
}


class TestBuildRow(unittest.TestCase):
    def test_extracts_all_fields(self):
        row = rbc.build_row(_SAMPLE_PAYLOAD)
        self.assertEqual(row["date"], "2026-06-29T12:30:07Z")
        self.assertEqual(row["onnxruntime_version"], "1.27.0")
        self.assertEqual(row["onnx_version"], "1.22.0")
        self.assertEqual(row["onnx_light_version"], "0.1.2")
        self.assertEqual(row["yobx_version"], "0.1.0")
        self.assertEqual(row["n_tests"], "1765")
        self.assertEqual(row["onnxruntime_pass"], "1465")
        self.assertEqual(row["onnxruntime_fail"], "300")
        self.assertEqual(row["reference_pass"], "1742")
        self.assertEqual(row["reference_fail"], "23")
        self.assertEqual(row["onnx_light_pass"], "1751")
        self.assertEqual(row["onnx_light_fail"], "14")
        self.assertEqual(row["yobx_pass"], "1742")
        self.assertEqual(row["yobx_fail"], "23")

    def test_missing_versions_are_empty_strings(self):
        payload = {**_SAMPLE_PAYLOAD, "versions": {}}
        row = rbc.build_row(payload)
        self.assertEqual(row["onnxruntime_version"], "")
        self.assertEqual(row["onnx_version"], "")
        self.assertEqual(row["onnx_light_version"], "")
        self.assertEqual(row["yobx_version"], "")

    def test_n_tests_falls_back_to_second_backend_when_first_is_zero(self):
        payload = {
            **_SAMPLE_PAYLOAD,
            "totals": {
                "onnxruntime": {"pass": 0, "fail": 0},
                "reference": {"pass": 10, "fail": 2},
                "onnx_light": {"pass": 9, "fail": 3},
                "yobx": {"pass": 8, "fail": 4},
            },
        }
        row = rbc.build_row(payload)
        self.assertEqual(row["n_tests"], "12")


class TestAppendRow(unittest.TestCase):
    def test_creates_file_with_header_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "out", "data.csv")
            row = rbc.build_row(_SAMPLE_PAYLOAD)
            rbc.append_row(csv_path, row)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["date"], "2026-06-29T12:30:07Z")
            self.assertEqual(rows[0]["n_tests"], "1765")

    def test_appends_to_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            row1 = rbc.build_row({**_SAMPLE_PAYLOAD, "date": "2026-01-01T00:00:00Z"})
            row2 = rbc.build_row({**_SAMPLE_PAYLOAD, "date": "2026-01-02T00:00:00Z"})
            rbc.append_row(csv_path, row1)
            rbc.append_row(csv_path, row2)
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["date"], "2026-01-01T00:00:00Z")
            self.assertEqual(rows[1]["date"], "2026-01-02T00:00:00Z")


class TestReadExistingDates(unittest.TestCase):
    def test_returns_empty_list_when_file_missing(self):
        self.assertEqual(rbc.read_existing_dates("/nonexistent/path.csv"), [])

    def test_returns_dates_from_existing_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=rbc.CSV_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {f: "" for f in rbc.CSV_FIELDS} | {"date": "2026-06-29T00:00:00Z"}
                )
                writer.writerow(
                    {f: "" for f in rbc.CSV_FIELDS} | {"date": "2026-06-30T00:00:00Z"}
                )
            dates = rbc.read_existing_dates(csv_path)
            self.assertEqual(dates, ["2026-06-29T00:00:00Z", "2026-06-30T00:00:00Z"])


class TestMain(unittest.TestCase):
    def test_main_appends_row_from_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "backend_node_coverage.json")
            csv_path = os.path.join(tmp, "release_backend_coverage.csv")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(_SAMPLE_PAYLOAD, fh)
            rc = rbc.main(["--json-path", json_path, "--csv-path", csv_path])
            self.assertEqual(rc, 0)
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["date"], "2026-06-29T12:30:07Z")

    def test_main_skips_duplicate_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            json_path = os.path.join(tmp, "backend_node_coverage.json")
            csv_path = os.path.join(tmp, "release_backend_coverage.csv")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(_SAMPLE_PAYLOAD, fh)
            rbc.main(["--json-path", json_path, "--csv-path", csv_path])
            rc = rbc.main(["--json-path", json_path, "--csv-path", csv_path])
            self.assertEqual(rc, 0)
            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            # Must still be only one row (idempotent).
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
