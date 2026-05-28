"""Tests for ``scripts.record_build_durations``."""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_build_durations as rbd  # noqa: E402


class TestRecordBuildDurations(unittest.TestCase):
    def test_parse_and_format_iso(self):
        parsed = rbd._parse_iso("2024-01-02T03:04:05Z")
        self.assertEqual(parsed.tzinfo, dt.timezone.utc)
        self.assertEqual(rbd._format_iso(parsed), "2024-01-02T03:04:05Z")

    def test_run_to_row_skips_unfinished(self):
        run = {"id": 1, "status": "in_progress", "conclusion": None}
        self.assertIsNone(rbd.run_to_row(run))

    def test_run_to_row_computes_duration(self):
        run = {
            "id": 42,
            "name": "Build docs",
            "event": "workflow_dispatch",
            "status": "completed",
            "conclusion": "success",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:01:30Z",
            "head_sha": "abc",
        }
        row = rbd.run_to_row(run)
        self.assertEqual(row["run_id"], "42")
        self.assertEqual(row["duration_seconds"], "90")
        self.assertEqual(row["workflow"], "Build docs")

    def test_append_rows_and_read_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "build_durations.csv")
            added = rbd.append_rows(
                path,
                [
                    {
                        "run_id": "1",
                        "workflow": "w",
                        "event": "push",
                        "status": "completed",
                        "conclusion": "success",
                        "created_at": "2024-05-01T10:00:00Z",
                        "updated_at": "2024-05-01T10:02:00Z",
                        "duration_seconds": "120",
                        "head_sha": "sha1",
                    }
                ],
            )
            self.assertEqual(added, 1)
            seen, latest = rbd.read_existing(path)
            self.assertEqual(seen, {"1"})
            self.assertEqual(rbd._format_iso(latest), "2024-05-01T10:00:00Z")

    def test_determine_since_uses_latest(self):
        latest = dt.datetime(2024, 5, 1, tzinfo=dt.timezone.utc)
        self.assertEqual(rbd.determine_since(latest, 6), latest)

    def test_determine_since_defaults_to_months(self):
        since = rbd.determine_since(None, 6)
        now = dt.datetime.now(tz=dt.timezone.utc)
        delta = now - since
        # 6 months -> ~180 days. Allow a small margin.
        self.assertGreater(delta.days, 175)
        self.assertLess(delta.days, 185)


if __name__ == "__main__":
    unittest.main()
