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

    def test_iter_workflow_runs_splits_saturated_windows(self):
        # Simulate a repository with > 1000 runs per week so that the initial
        # weekly windows saturate the GitHub 1000-result cap. The fake
        # ``_fetch_runs_window`` returns a unique id per (window, index) tuple
        # and reports saturation whenever the window covers more than one day.
        calls: list[tuple[dt.datetime, dt.datetime]] = []

        def fake_fetch(repo, start, end, token):
            calls.append((start, end))
            window = end - start
            if window > dt.timedelta(days=1):
                runs = [{"id": f"{start.isoformat()}-{i}"} for i in range(1000)]
                return runs, True
            runs = [{"id": f"{start.isoformat()}-{i}"} for i in range(3)]
            return runs, False

        original = rbd._fetch_runs_window
        rbd._fetch_runs_window = fake_fetch
        try:
            since = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
            until = dt.datetime(2024, 1, 15, tzinfo=dt.timezone.utc)
            ids = [
                str(r["id"])
                for r in rbd.iter_workflow_runs(
                    "owner/repo", since, token=None, until=until,
                    initial_window_days=7,
                )
            ]
        finally:
            rbd._fetch_runs_window = original

        # All windows should eventually be split down to <= 1 day and yield
        # 3 fake runs per leaf window. The exact count depends on the
        # recursive halving; the key properties are that all ids are unique
        # and we retrieved more than the 1000-result API cap would have
        # allowed with a single query.
        self.assertGreater(len(ids), 30)
        self.assertEqual(len(ids), len(set(ids)))
        # The initial saturated windows must have been split.
        small_calls = [c for c in calls if (c[1] - c[0]) <= dt.timedelta(days=1)]
        self.assertGreater(len(small_calls), 10)

    def test_safe_job_filename(self):
        self.assertEqual(rbd.safe_job_filename("build"), "build")
        self.assertEqual(
            rbd.safe_job_filename("build (ubuntu-latest, 3.12)"),
            "build_ubuntu-latest_3.12",
        )
        self.assertEqual(rbd.safe_job_filename("a/b\\c"), "a_b_c")
        self.assertEqual(rbd.safe_job_filename(""), "job")
        self.assertEqual(rbd.safe_job_filename("///"), "job")

    def test_job_to_row_skips_unfinished(self):
        job = {"id": 1, "status": "in_progress", "conclusion": None}
        self.assertIsNone(rbd.job_to_row(job))

    def test_job_to_row_computes_duration(self):
        run = {"id": 10, "name": "Build docs", "head_sha": "abc"}
        job = {
            "id": 99,
            "run_id": 10,
            "name": "build (ubuntu-latest)",
            "status": "completed",
            "conclusion": "success",
            "started_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T00:02:30Z",
        }
        row = rbd.job_to_row(job, run)
        self.assertEqual(row["job_id"], "99")
        self.assertEqual(row["run_id"], "10")
        self.assertEqual(row["workflow"], "Build docs")
        self.assertEqual(row["job_name"], "build (ubuntu-latest)")
        self.assertEqual(row["duration_seconds"], "150")
        self.assertEqual(row["head_sha"], "abc")

    def test_record_jobs_for_run_creates_file_per_job(self):
        run = {"id": 42, "name": "CI", "head_sha": "sha42"}
        fake_jobs = [
            {
                "id": 1,
                "run_id": 42,
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T00:01:00Z",
            },
            {
                "id": 2,
                "run_id": 42,
                "name": "test (ubuntu-latest, 3.12)",
                "status": "completed",
                "conclusion": "failure",
                "started_at": "2024-01-01T00:01:00Z",
                "completed_at": "2024-01-01T00:03:00Z",
            },
            {
                "id": 3,
                "run_id": 42,
                "name": "test (ubuntu-latest, 3.12)",
                "status": "in_progress",
                "conclusion": None,
                "started_at": "2024-01-01T00:01:00Z",
                "completed_at": None,
            },
        ]

        def fake_iter(run_id, repo, token):
            self.assertEqual(run_id, "42")
            for j in fake_jobs:
                yield j

        original = rbd.iter_run_jobs
        rbd.iter_run_jobs = fake_iter
        try:
            with tempfile.TemporaryDirectory() as tmp:
                added = rbd.record_jobs_for_run(run, "owner/myrepo", tmp, None)
                self.assertEqual(added, 2)
                jobs_dir = os.path.join(tmp, "myrepo", "jobs")
                build_path = os.path.join(jobs_dir, "build.csv")
                test_path = os.path.join(
                    jobs_dir, "test_ubuntu-latest_3.12.csv"
                )
                self.assertTrue(os.path.exists(build_path))
                self.assertTrue(os.path.exists(test_path))
                # Re-running with the same jobs should not duplicate them.
                added_again = rbd.record_jobs_for_run(
                    run, "owner/myrepo", tmp, None
                )
                self.assertEqual(added_again, 0)
                seen = rbd.read_existing_jobs(build_path)
                self.assertEqual(seen, {"1"})
                seen = rbd.read_existing_jobs(test_path)
                self.assertEqual(seen, {"2"})
        finally:
            rbd.iter_run_jobs = original


if __name__ == "__main__":
    unittest.main()
