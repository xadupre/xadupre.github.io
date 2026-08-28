"""Tests for ``scripts.record_pr_activity``."""

from __future__ import annotations

import csv
import datetime as dt
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_pr_activity as rpa


class TestRecordPrActivity(unittest.TestCase):
    def test_dashboard_and_home_page_are_wired(self):
        root = os.path.dirname(os.path.dirname(HERE))
        page = os.path.join(root, "dashboard", "onnxruntime", "pr-activity.html")
        with open(page, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn("open_prs", text)
        self.assertIn("merged_prs_7d", text)
        self.assertIn("avg_open_age_days", text)
        self.assertIn("loadChartJs()", text)

        with open(os.path.join(root, "index.html"), encoding="utf-8") as stream:
            index = stream.read()
        self.assertIn('href="dashboard/onnxruntime/pr-activity.html"', index)
        self.assertIn("record_onnxruntime_pr_activity.yml", index)

    def test_workflow_runs_the_recorder(self):
        root = os.path.dirname(os.path.dirname(HERE))
        workflow = os.path.join(
            root, ".github", "workflows", "record_onnxruntime_pr_activity.yml"
        )
        with open(workflow, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn('cron: "53 4 * * 1"', text)
        self.assertIn("python -u scripts/record_pr_activity.py", text)
        self.assertIn("cache_data/onnxruntime/pr_activity.csv", text)

    def test_collect_snapshot(self):
        now = dt.datetime(2026, 8, 28, 8, tzinfo=dt.timezone.utc)
        responses = {
            "open": [
                {
                    "created_at": "2026-08-18T08:00:00Z",
                    "updated_at": "2026-08-28T07:00:00Z",
                },
                {
                    "created_at": "2026-08-08T08:00:00Z",
                    "updated_at": "2026-08-27T07:00:00Z",
                },
            ],
            "closed": [
                {
                    "merged_at": "2026-08-27T08:00:00Z",
                    "updated_at": "2026-08-27T08:00:00Z",
                },
                {
                    "merged_at": None,
                    "updated_at": "2026-08-26T08:00:00Z",
                },
                {
                    "merged_at": "2026-08-20T07:59:00Z",
                    "updated_at": "2026-08-20T07:59:00Z",
                },
            ],
        }

        original = rpa.iter_pulls
        rpa.iter_pulls = lambda repo, state, token: iter(responses[state])
        try:
            snapshot = rpa.collect_snapshot("owner/repo", None, now)
        finally:
            rpa.iter_pulls = original

        self.assertEqual(snapshot["date"], "2026-08-28T08:00:00Z")
        self.assertEqual(snapshot["open_prs"], "2")
        self.assertEqual(snapshot["merged_prs_7d"], "1")
        self.assertEqual(snapshot["avg_open_age_days"], "15.00")

    def test_iter_pulls_handles_pagination(self):
        calls = []

        def fake_request(url, token):
            page = int(url.rsplit("page=", 1)[1])
            calls.append(page)
            return ([{"number": page}] * (100 if page == 1 else 1), {})

        original = rpa._request
        rpa._request = fake_request
        try:
            pulls = list(rpa.iter_pulls("owner/repo", "open", None))
        finally:
            rpa._request = original
        self.assertEqual(len(pulls), 101)
        self.assertEqual(calls, [1, 2])

    def test_write_snapshot_replaces_same_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnxruntime", "pr_activity.csv")
            first = {
                "date": "2026-08-28T08:00:00Z",
                "open_prs": "100",
                "merged_prs_7d": "20",
                "avg_open_age_days": "30.00",
            }
            second = dict(first, date="2026-08-28T09:00:00Z", open_prs="101")
            rpa.write_snapshot(path, first)
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [second])

    def test_default_repository_is_onnxruntime(self):
        self.assertEqual(rpa.DEFAULT_REPO, "microsoft/onnxruntime")


if __name__ == "__main__":
    unittest.main()
