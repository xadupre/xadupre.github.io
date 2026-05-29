"""Tests for ``scripts.record_yobx_sklearn_coverage``."""

from __future__ import annotations

import csv
import datetime as dt
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_yobx_sklearn_coverage as rysc  # noqa: E402


def _make_row(predictable: bool, has_converter: bool) -> dict:
    return {
        "name": "Foo",
        "predictable": predictable,
        "yobx": (lambda: None) if has_converter else None,
    }


class TestRecordYobxSklearnCoverage(unittest.TestCase):
    def test_summarize_rows(self):
        rows = [
            _make_row(True, True),
            _make_row(True, False),
            _make_row(False, False),
            _make_row(True, True),
        ]
        summary = rysc.summarize_rows(rows)
        self.assertEqual(
            summary,
            {"n_estimators": 4, "n_predictable": 3, "n_converted": 2},
        )

    def test_summarize_rows_empty(self):
        self.assertEqual(
            rysc.summarize_rows([]),
            {"n_estimators": 0, "n_predictable": 0, "n_converted": 0},
        )

    def test_coverage_pct(self):
        self.assertEqual(rysc._coverage_pct(0, 0), 0.0)
        self.assertEqual(rysc._coverage_pct(0, 10), 0.0)
        self.assertEqual(rysc._coverage_pct(5, 10), 50.0)
        self.assertAlmostEqual(rysc._coverage_pct(1, 3), 33.333333, places=4)

    def test_format_iso_naive(self):
        formatted = rysc._format_iso(dt.datetime(2026, 1, 2, 3, 4, 5))
        self.assertEqual(formatted, "2026-01-02T03:04:05Z")

    def test_is_available_missing(self):
        # Use a deliberately missing module name to ensure detection works.
        original = rysc._REQUIRED_MODULE.copy()
        try:
            rysc._REQUIRED_MODULE["nope"] = "this_module_does_not_exist_xyz_42"
            self.assertFalse(rysc._is_available("nope"))
        finally:
            rysc._REQUIRED_MODULE.clear()
            rysc._REQUIRED_MODULE.update(original)

    def test_is_available_present(self):
        # ``sys`` always imports successfully.
        original = rysc._REQUIRED_MODULE.copy()
        try:
            rysc._REQUIRED_MODULE["fake"] = "sys"
            self.assertTrue(rysc._is_available("fake"))
        finally:
            rysc._REQUIRED_MODULE.clear()
            rysc._REQUIRED_MODULE.update(original)

    def test_append_rows_creates_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sub", "sklearn_coverage.csv")
            row = {
                "date": "2026-01-02T03:04:05Z",
                "library": "sklearn",
                "sklearn_version": "1.8.0",
                "yobx_version": "0.1.0",
                "n_estimators": "200",
                "n_predictable": "150",
                "n_converted": "60",
                "coverage_pct": "40.00",
            }
            written = rysc.append_rows(csv_path, [row])
            self.assertEqual(written, 1)
            with open(csv_path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0], row)

    def test_append_rows_appends_without_duplicating_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sklearn_coverage.csv")
            base = {
                "date": "2026-01-02T03:04:05Z",
                "library": "sklearn",
                "sklearn_version": "1.8.0",
                "yobx_version": "0.1.0",
                "n_estimators": "1",
                "n_predictable": "1",
                "n_converted": "1",
                "coverage_pct": "100.00",
            }
            other = dict(base)
            other["library"] = "xgboost"
            rysc.append_rows(csv_path, [base])
            rysc.append_rows(csv_path, [other])
            with open(csv_path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertEqual(content.count("date,library"), 1)
            with open(csv_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["library"], "sklearn")
            self.assertEqual(rows[1]["library"], "xgboost")

    def test_append_rows_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "sklearn_coverage.csv")
            self.assertEqual(rysc.append_rows(csv_path, []), 0)
            self.assertFalse(os.path.exists(csv_path))

    def test_csv_fields(self):
        # The CSV fields must stay in sync with the dashboard expectations.
        self.assertEqual(
            rysc.CSV_FIELDS,
            (
                "date",
                "library",
                "sklearn_version",
                "yobx_version",
                "n_estimators",
                "n_predictable",
                "n_converted",
                "coverage_pct",
            ),
        )


if __name__ == "__main__":
    unittest.main()
