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

import record_release_backend_coverage as rrc  # noqa: E402


def _make_payload(
    date="2024-01-15T10:00:00Z",
    versions=None,
    totals=None,
) -> dict:
    """Build a minimal coverage JSON payload."""
    if versions is None:
        versions = {
            "numpy": "1.26.0",
            "onnx": "1.17.0",
            "onnx_light": "0.1.0",
            "onnxruntime": "1.20.0",
        }
    if totals is None:
        totals = {
            "onnxruntime": {"pass": 1400, "fail": 200},
            "reference": {"pass": 1550, "fail": 50},
            "onnx_light": {"pass": 1560, "fail": 40},
        }
    return {
        "date": date,
        "kind": "node",
        "versions": versions,
        "totals": totals,
        "tests": [],
    }


class TestCsvColumns(unittest.TestCase):
    def test_three_backends(self):
        backends = ["onnxruntime", "reference", "onnx_light"]
        cols = rrc._csv_columns(backends)
        self.assertEqual(cols[0], "date")
        self.assertIn("onnxruntime_version", cols)
        self.assertIn("onnx_version", cols)
        self.assertIn("onnx_light_version", cols)
        self.assertIn("n_tests", cols)
        self.assertIn("onnxruntime_pass", cols)
        self.assertIn("onnxruntime_fail", cols)
        self.assertIn("reference_pass", cols)
        self.assertIn("onnx_light_pass", cols)
        # yobx should not appear for three-backend payloads
        self.assertNotIn("yobx_version", cols)
        self.assertNotIn("yobx_pass", cols)

    def test_four_backends(self):
        backends = ["onnxruntime", "reference", "onnx_light", "yobx"]
        cols = rrc._csv_columns(backends)
        self.assertIn("yobx_version", cols)
        self.assertIn("yobx_pass", cols)
        self.assertIn("yobx_fail", cols)

    def test_column_order_starts_with_date(self):
        cols = rrc._csv_columns(["onnxruntime", "reference", "onnx_light"])
        self.assertEqual(cols[0], "date")


class TestBuildRow(unittest.TestCase):
    def test_three_backend_row(self):
        payload = _make_payload()
        cols, row = rrc.build_row(payload)
        self.assertEqual(row["date"], "2024-01-15T10:00:00Z")
        self.assertEqual(row["onnxruntime_version"], "1.20.0")
        self.assertEqual(row["onnx_version"], "1.17.0")
        self.assertEqual(row["onnx_light_version"], "0.1.0")
        self.assertEqual(row["n_tests"], "1600")  # 1400+200
        self.assertEqual(row["onnxruntime_pass"], "1400")
        self.assertEqual(row["onnxruntime_fail"], "200")
        self.assertEqual(row["reference_pass"], "1550")
        self.assertEqual(row["onnx_light_pass"], "1560")
        self.assertEqual(row["onnx_light_fail"], "40")

    def test_four_backend_row(self):
        payload = _make_payload(
            versions={
                "numpy": "1.26.0",
                "onnx": "1.17.0",
                "onnx_light": "0.1.0",
                "onnxruntime": "1.20.0",
                "yobx": "0.0.1",
            },
            totals={
                "onnxruntime": {"pass": 1400, "fail": 200},
                "reference": {"pass": 1550, "fail": 50},
                "onnx_light": {"pass": 1560, "fail": 40},
                "yobx": {"pass": 1555, "fail": 45},
            },
        )
        cols, row = rrc.build_row(payload)
        self.assertEqual(row["yobx_version"], "0.0.1")
        self.assertEqual(row["yobx_pass"], "1555")
        self.assertEqual(row["yobx_fail"], "45")
        self.assertIn("yobx_pass", cols)

    def test_columns_match_row_keys(self):
        payload = _make_payload()
        cols, row = rrc.build_row(payload)
        for col in cols:
            self.assertIn(col, row, f"column {col!r} missing from row")

    def test_n_tests_is_sum_of_first_backend(self):
        payload = _make_payload(
            totals={
                "onnxruntime": {"pass": 700, "fail": 100},
                "reference": {"pass": 750, "fail": 50},
                "onnx_light": {"pass": 760, "fail": 40},
            }
        )
        _, row = rrc.build_row(payload)
        # 700 + 100 = 800
        self.assertEqual(row["n_tests"], "800")


class TestReadExistingDates(unittest.TestCase):
    def test_empty_file_returns_empty_set(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("date,n_tests\n")
            path = fh.name
        try:
            dates = rrc.read_existing_dates(path)
            self.assertEqual(dates, set())
        finally:
            os.unlink(path)

    def test_reads_existing_dates(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as fh:
            fh.write("date,n_tests\n")
            fh.write("2024-01-01T00:00:00Z,100\n")
            fh.write("2024-02-01T00:00:00Z,110\n")
            path = fh.name
        try:
            dates = rrc.read_existing_dates(path)
            self.assertIn("2024-01-01T00:00:00Z", dates)
            self.assertIn("2024-02-01T00:00:00Z", dates)
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty_set(self):
        dates = rrc.read_existing_dates("/tmp/does_not_exist_xyzzy.csv")
        self.assertEqual(dates, set())


class TestAppendRow(unittest.TestCase):
    def test_writes_header_and_row_when_file_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "coverage.csv")
            payload = _make_payload()
            cols, row = rrc.build_row(payload)
            written = rrc.append_row(csv_path, cols, row)
            self.assertTrue(written)
            with open(csv_path, encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]["date"], "2024-01-15T10:00:00Z")

    def test_skips_duplicate_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "coverage.csv")
            payload = _make_payload()
            cols, row = rrc.build_row(payload)
            # First write.
            rrc.append_row(csv_path, cols, row, existing_dates=set())
            existing = rrc.read_existing_dates(csv_path)
            # Second write with same date — should be skipped.
            written = rrc.append_row(csv_path, cols, row, existing_dates=existing)
            self.assertFalse(written)
            with open(csv_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            # Only 1 data row (header + 1 row = 2 lines).
            self.assertEqual(len(lines), 2)

    def test_appends_new_row_on_different_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "coverage.csv")
            p1 = _make_payload(date="2024-01-15T10:00:00Z")
            p2 = _make_payload(date="2024-02-15T10:00:00Z")
            cols1, row1 = rrc.build_row(p1)
            cols2, row2 = rrc.build_row(p2)
            rrc.append_row(csv_path, cols1, row1, existing_dates=set())
            existing = rrc.read_existing_dates(csv_path)
            rrc.append_row(csv_path, cols2, row2, existing_dates=existing)
            with open(csv_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            # Header + 2 data rows.
            self.assertEqual(len(lines), 3)


class TestLoadJson(unittest.TestCase):
    def test_round_trip(self):
        payload = _make_payload()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(payload, fh)
            path = fh.name
        try:
            loaded = rrc.load_json(path)
            self.assertEqual(loaded["date"], payload["date"])
            self.assertEqual(loaded["totals"], payload["totals"])
        finally:
            os.unlink(path)


class TestMainCli(unittest.TestCase):
    def test_main_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "coverage.json")
            csv_path = os.path.join(tmpdir, "release_coverage.csv")
            payload = _make_payload()
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            result = rrc.main(
                ["--json-path", json_path, "--csv-path", csv_path]
            )
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))
            self.assertEqual(len(reader), 1)
            self.assertEqual(reader[0]["date"], "2024-01-15T10:00:00Z")

    def test_main_skips_duplicate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "coverage.json")
            csv_path = os.path.join(tmpdir, "release_coverage.csv")
            payload = _make_payload()
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            rrc.main(["--json-path", json_path, "--csv-path", csv_path])
            # Call again — should not add a second row.
            rrc.main(["--json-path", json_path, "--csv-path", csv_path])
            with open(csv_path, encoding="utf-8") as fh:
                lines = fh.readlines()
            self.assertEqual(len(lines), 2)  # header + 1 data row


if __name__ == "__main__":
    unittest.main()
