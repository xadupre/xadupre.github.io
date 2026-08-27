"""Tests for the plot_onnx_time history recorder."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_onnx_time as rot  # noqa: E402


class TestRecordOnnxTime(unittest.TestCase):
    def test_main_defaults_to_current_platform(self):
        html = (
            "<pre>load/1filex1/onnx avg=1.0 ms median=1.0 ms "
            "max=1.0 ms std=0.0 ms</pre>"
        )
        with tempfile.TemporaryDirectory() as temp:
            html_path = os.path.join(temp, "plot_onnx_time.html")
            csv_path = os.path.join(temp, "onnx_time.csv")
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write(html)
            args = [
                "record_onnx_time.py",
                html_path,
                "--output",
                csv_path,
                "--commit",
                "abc",
                "--run-id",
                "123",
            ]

            with patch.object(sys, "argv", args), patch.object(
                rot.platform, "platform", return_value="Test Platform"
            ):
                rot.main()

            with open(csv_path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["machine"], "Test Platform")

    def test_extract_timings_decodes_html_and_converts_ms(self):
        html = """<pre>load/1filex1/onnx  avg=12.5 ms median=12.0 ms
max=13.0 ms std=0.5 ms
load/1filex1/onnxlight avg=4.0 ms median=3.9 ms max=4.2 ms std=0.1 ms</pre>"""
        rows = rot.extract_timings(html)
        self.assertEqual([r["name"] for r in rows], ["load/1filex1/onnxlight"])
        self.assertEqual(rows[0]["avg"], 0.004)

    def test_append_snapshot_and_skip_duplicate_run(self):
        html = (
            "<pre>load/1filex1/onnx avg=12.5 ms median=12.0 ms "
            "max=13.0 ms std=0.5 ms</pre>"
        )
        with tempfile.TemporaryDirectory() as temp:
            html_path = os.path.join(temp, "plot_onnx_time.html")
            csv_path = os.path.join(temp, "data", "onnx_time.csv")
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write(html)

            self.assertEqual(
                rot.append_snapshot(
                    html_path,
                    csv_path,
                    "2026-08-21T00:00:00Z",
                    "abc",
                    "123",
                    "Linux X64 / Test CPU",
                ),
                1,
            )
            self.assertEqual(
                rot.append_snapshot(
                    html_path,
                    csv_path,
                    "2026-08-22T00:00:00Z",
                    "abc",
                    "123",
                    "Linux X64 / Test CPU",
                ),
                0,
            )
            with open(csv_path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["commit"], "abc")
            self.assertEqual(rows[0]["run_id"], "123")
            self.assertEqual(rows[0]["machine"], "Linux X64 / Test CPU")
            self.assertEqual(float(rows[0]["avg"]), 0.0125)

    def test_empty_page_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            html_path = os.path.join(temp, "plot_onnx_time.html")
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write("<html></html>")
            with self.assertRaisesRegex(ValueError, "No benchmark timings"):
                rot.append_snapshot(
                    html_path,
                    os.path.join(temp, "out.csv"),
                    "date",
                    "abc",
                    "123",
                    "machine",
                )

    def test_append_snapshot_upgrades_legacy_csv(self):
        html = (
            "<pre>load/1filex1/onnx avg=1.0 ms median=1.0 ms "
            "max=1.0 ms std=0.0 ms</pre>"
        )
        with tempfile.TemporaryDirectory() as temp:
            html_path = os.path.join(temp, "plot_onnx_time.html")
            csv_path = os.path.join(temp, "onnx_time.csv")
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write(html)
            with open(csv_path, "w", encoding="utf-8") as stream:
                stream.write(
                    "date,commit,run_id,name,avg,median,max,std\n"
                    "date,abc,old,old,1,1,1,0\n"
                )

            rot.append_snapshot(
                html_path, csv_path, "date", "abc", "new", "Linux X64 / Test CPU"
            )

            with open(csv_path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["machine"], "not recorded")
            self.assertEqual(rows[1]["machine"], "Linux X64 / Test CPU")

    def test_output_may_be_a_bare_filename(self):
        html = (
            "<pre>load/1filex1/onnx avg=1.0 ms median=1.0 ms "
            "max=1.0 ms std=0.0 ms</pre>"
        )
        with tempfile.TemporaryDirectory() as temp:
            html_path = os.path.join(temp, "plot_onnx_time.html")
            with open(html_path, "w", encoding="utf-8") as stream:
                stream.write(html)
            previous = os.getcwd()
            try:
                os.chdir(temp)
                self.assertEqual(
                    rot.append_snapshot(
                        html_path, "out.csv", "date", "abc", "123", "machine"
                    ),
                    1,
                )
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
