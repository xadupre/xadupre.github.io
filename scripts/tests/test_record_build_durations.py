"""Tests for ``scripts.record_build_durations``."""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import unittest
import urllib.error

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


    def test_record_jobs_for_run_saves_partial_on_failure(self):
        run = {"id": 7, "name": "CI", "head_sha": "sha7"}
        fake_jobs = [
            {
                "id": 1,
                "run_id": 7,
                "name": "build",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T00:01:00Z",
            },
            {
                "id": 2,
                "run_id": 7,
                "name": "test",
                "status": "completed",
                "conclusion": "success",
                "started_at": "2024-01-01T00:01:00Z",
                "completed_at": "2024-01-01T00:02:00Z",
            },
        ]

        def fake_iter(run_id, repo, token):
            yield fake_jobs[0]
            yield fake_jobs[1]
            raise RuntimeError("simulated API failure")

        original = rbd.iter_run_jobs
        rbd.iter_run_jobs = fake_iter
        try:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    rbd.record_jobs_for_run(run, "owner/myrepo", tmp, None)
                # Even though the iterator raised, the rows collected
                # before the failure must have been flushed to disk.
                jobs_dir = os.path.join(tmp, "myrepo", "jobs")
                build_path = os.path.join(jobs_dir, "build.csv")
                test_path = os.path.join(jobs_dir, "test.csv")
                self.assertTrue(os.path.exists(build_path))
                self.assertTrue(os.path.exists(test_path))
                self.assertEqual(rbd.read_existing_jobs(build_path), {"1"})
                self.assertEqual(rbd.read_existing_jobs(test_path), {"2"})
        finally:
            rbd.iter_run_jobs = original

    def test_read_cached_runs_from_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = os.path.join(tmp, "jobs")
            # Missing directory -> empty result.
            seen, latest = rbd.read_cached_runs_from_jobs(jobs_dir)
            self.assertEqual(seen, set())
            self.assertIsNone(latest)

            os.makedirs(jobs_dir)
            rbd._append_rows(
                os.path.join(jobs_dir, "build.csv"),
                [
                    {
                        "job_id": "j1",
                        "run_id": "100",
                        "workflow": "CI",
                        "job_name": "build",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2024-01-01T00:00:00Z",
                        "completed_at": "2024-01-01T00:01:00Z",
                        "duration_seconds": "60",
                        "head_sha": "sha1",
                    },
                    {
                        "job_id": "j2",
                        "run_id": "101",
                        "workflow": "CI",
                        "job_name": "build",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2024-02-15T08:30:00Z",
                        "completed_at": "2024-02-15T08:32:00Z",
                        "duration_seconds": "120",
                        "head_sha": "sha2",
                    },
                ],
                rbd.JOB_CSV_FIELDS,
            )
            rbd._append_rows(
                os.path.join(jobs_dir, "test.csv"),
                [
                    {
                        "job_id": "j3",
                        "run_id": "101",
                        "workflow": "CI",
                        "job_name": "test",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2024-02-15T08:31:00Z",
                        "completed_at": "2024-02-15T08:35:00Z",
                        "duration_seconds": "240",
                        "head_sha": "sha2",
                    },
                ],
                rbd.JOB_CSV_FIELDS,
            )
            # Non-CSV files and nested directories must be ignored.
            with open(os.path.join(jobs_dir, "index.json"), "w") as fh:
                fh.write("{}")
            os.makedirs(os.path.join(jobs_dir, "nested"))

            seen, latest = rbd.read_cached_runs_from_jobs(jobs_dir)
            self.assertEqual(seen, {"100", "101"})
            self.assertEqual(rbd._format_iso(latest), "2024-02-15T08:31:00Z")

    def test_process_repo_reuses_jobs_cache(self):
        # Reproduce the scenario from the issue: ``build_durations.csv``
        # is missing but the per-job CSVs are present. The script must
        # use them to skip refetching runs whose jobs are already cached.
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = os.path.join(tmp, "myrepo", "jobs")
            os.makedirs(jobs_dir)
            rbd._append_rows(
                os.path.join(jobs_dir, "build.csv"),
                [
                    {
                        "job_id": "j1",
                        "run_id": "100",
                        "workflow": "CI",
                        "job_name": "build",
                        "status": "completed",
                        "conclusion": "success",
                        "started_at": "2024-05-01T10:00:00Z",
                        "completed_at": "2024-05-01T10:01:00Z",
                        "duration_seconds": "60",
                        "head_sha": "sha1",
                    }
                ],
                rbd.JOB_CSV_FIELDS,
            )

            jobs_calls: list[str] = []

            def fake_iter_runs(repo, since, token, until=None):
                # Yield the already-cached run plus a brand-new one.
                yield {
                    "id": 100,
                    "name": "CI",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2024-05-01T09:59:00Z",
                    "updated_at": "2024-05-01T10:02:00Z",
                    "head_sha": "sha1",
                }
                yield {
                    "id": 200,
                    "name": "CI",
                    "event": "push",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2024-05-02T09:00:00Z",
                    "updated_at": "2024-05-02T09:03:00Z",
                    "head_sha": "sha2",
                }

            def fake_iter_jobs(run_id, repo, token):
                jobs_calls.append(run_id)
                yield {
                    "id": int(run_id) * 10,
                    "run_id": int(run_id),
                    "name": "build",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2024-05-02T09:00:30Z",
                    "completed_at": "2024-05-02T09:02:30Z",
                }

            orig_runs = rbd.iter_workflow_runs
            orig_jobs = rbd.iter_run_jobs
            rbd.iter_workflow_runs = fake_iter_runs
            rbd.iter_run_jobs = fake_iter_jobs
            try:
                added = rbd.process_repo("owner/myrepo", tmp, months=6, token=None)
            finally:
                rbd.iter_workflow_runs = orig_runs
                rbd.iter_run_jobs = orig_jobs

            # Only the new run should have triggered a jobs API call.
            self.assertEqual(jobs_calls, ["200"])
            # And only that new run should have been appended.
            self.assertEqual(added, 1)


        with tempfile.TemporaryDirectory() as tmp:
            # Missing directory -> 0 entries, no file.
            missing = os.path.join(tmp, "missing")
            self.assertEqual(rbd.write_jobs_index(missing), 0)
            self.assertFalse(
                os.path.exists(os.path.join(missing, "index.json"))
            )

            jobs_dir = os.path.join(tmp, "jobs")
            os.makedirs(jobs_dir)
            # Two CSVs, one non-CSV file, one nested dir -> ignored.
            for name in ("b.csv", "a.csv", "notes.txt"):
                with open(os.path.join(jobs_dir, name), "w") as fh:
                    fh.write("x")
            os.makedirs(os.path.join(jobs_dir, "sub"))
            with open(
                os.path.join(jobs_dir, "sub", "ignored.csv"), "w"
            ) as fh:
                fh.write("x")

            n = rbd.write_jobs_index(jobs_dir)
            self.assertEqual(n, 2)
            index_path = os.path.join(jobs_dir, "index.json")
            with open(index_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            # The CSVs are sorted alphabetically and the non-CSV / nested
            # files are excluded.
            self.assertEqual(payload, {"jobs": ["a.csv", "b.csv"]})

    def test_main_continues_after_repo_failure(self):
        calls: list[str] = []

        def fake_process(repo, cache_dir, months, token):
            calls.append(repo)
            if repo == "owner/bad":
                raise urllib.error.HTTPError(
                    "http://x", 500, "boom", hdrs=None, fp=None
                )
            return 1

        original = rbd.process_repo
        rbd.process_repo = fake_process
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc = rbd.main(
                    [
                        "--cache-dir",
                        tmp,
                        "--repo",
                        "owner/good1",
                        "--repo",
                        "owner/bad",
                        "--repo",
                        "owner/good2",
                    ]
                )
        finally:
            rbd.process_repo = original
        # All three repos must have been attempted, and the overall exit
        # code must be 0 because at least one repository succeeded.
        self.assertEqual(calls, ["owner/good1", "owner/bad", "owner/good2"])
        self.assertEqual(rc, 0)

    def test_process_repo_writes_jobs_index_even_on_fetch_error(self):
        # Regression test: when ``iter_workflow_runs`` raises partway
        # through (e.g. a transient GitHub API error), ``process_repo``
        # must still refresh ``jobs/index.json`` so the dashboard can
        # discover the per-job CSV files that have already been cached.
        with tempfile.TemporaryDirectory() as tmp:
            jobs_dir = os.path.join(tmp, "myrepo", "jobs")
            os.makedirs(jobs_dir)
            for name in ("build.csv", "test.csv"):
                with open(os.path.join(jobs_dir, name), "w") as fh:
                    fh.write(
                        ",".join(rbd.JOB_CSV_FIELDS) + "\n"
                    )

            def fake_iter_runs(repo, since, token, until=None):
                raise urllib.error.HTTPError(
                    "http://x", 502, "bad gateway", hdrs=None, fp=None
                )
                yield  # pragma: no cover - generator marker

            orig_runs = rbd.iter_workflow_runs
            rbd.iter_workflow_runs = fake_iter_runs
            try:
                with self.assertRaises(urllib.error.HTTPError):
                    rbd.process_repo(
                        "owner/myrepo", tmp, months=6, token=None
                    )
            finally:
                rbd.iter_workflow_runs = orig_runs

            index_path = os.path.join(jobs_dir, "index.json")
            self.assertTrue(os.path.exists(index_path))
            with open(index_path, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual(payload, {"jobs": ["build.csv", "test.csv"]})

    def test_dashboard_job_duration_graphs_use_minutes(self):
        root = os.path.dirname(os.path.dirname(HERE))
        pages = [
            os.path.join(root, "dashboard", "onnx", "build-durations.html"),
            os.path.join(root, "dashboard", "onnx-light", "build-durations.html"),
            os.path.join(
                root,
                "dashboard",
                "yet-another-onnx-builder",
                "build-durations.html",
            ),
        ]
        for path in pages:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn('duration: dur / 60', content)
            self.assertIn('text: "duration (min)"', content)
            self.assertIn('+ " min"', content)
            self.assertNotIn('text: "duration (s)"', content)
            self.assertNotIn('+ " s"', content)


if __name__ == "__main__":
    unittest.main()
