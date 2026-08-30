"""Tests for ``scripts.record_pr_activity``."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_pr_activity as rpa


class TestRecordPrActivity(unittest.TestCase):
    def test_dashboard_and_home_page_are_wired(self):
        root = os.path.dirname(os.path.dirname(HERE))
        for project in ("onnx", "onnxruntime"):
            page = os.path.join(root, "dashboard", project, "pr-activity.html")
            with open(page, encoding="utf-8") as stream:
                text = stream.read()
            self.assertIn("open_prs", text)
            self.assertIn("opened_prs_7d", text)
            self.assertIn("closed_prs_7d", text)
            self.assertIn("merged_prs_7d", text)
            self.assertIn("avg_open_age_days", text)
            self.assertIn("loadChartJs()", text)
            self.assertIn('unit: "day"', text)
            self.assertIn('id="chartOpen"', text)
            self.assertIn('id="chartActivity"', text)
            self.assertIn('id="chartAge"', text)
            self.assertIn(
                'for (const id of ["chartOpen", "chartActivity", "chartAge"])', text
            )
            self.assertRegex(
                text,
                r'(?s)getElementById\("chartOpen"\).*?'
                r'label: "open PRs".*?'
                r'getElementById\("chartActivity"\).*?'
                r'label: "opened in preceding 7 days"',
            )
            open_chart = re.search(
                r'(?s)getElementById\("chartOpen"\)(.*?)'
                r'getElementById\("chartActivity"\)',
                text,
            )
            self.assertIsNotNone(open_chart)
            self.assertNotIn("preceding 7 days", open_chart.group(1))
            self.assertIn("open_pulls.json", text)
            self.assertIn("10 latest open pull requests", text)
            self.assertIn("10 oldest open pull requests", text)

        with open(os.path.join(root, "index.html"), encoding="utf-8") as stream:
            index = stream.read()
        self.assertIn('href="dashboard/onnx/pr-activity.html"', index)
        self.assertIn('href="dashboard/onnxruntime/pr-activity.html"', index)
        self.assertIn("record_onnx_pr_activity.yml", index)
        self.assertIn("record_onnxruntime_pr_activity.yml", index)

    def test_workflow_runs_the_recorder(self):
        root = os.path.dirname(os.path.dirname(HERE))
        workflow = os.path.join(
            root, ".github", "workflows", "record_onnxruntime_pr_activity.yml"
        )
        with open(workflow, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn('cron: "53 4 * * *"', text)
        self.assertIn("python -u scripts/record_pr_activity.py", text)
        self.assertIn("cache_data/onnxruntime/pr_activity.csv", text)
        self.assertIn("cache_data/onnxruntime/open_pulls.json", text)

        onnx_workflow = os.path.join(
            root, ".github", "workflows", "record_onnx_pr_activity.yml"
        )
        with open(onnx_workflow, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn('cron: "7 5 * * *"', text)
        self.assertIn(
            "python -u scripts/record_pr_activity.py --repo onnx/onnx", text
        )
        self.assertIn("cache_data/onnx/pr_activity.csv", text)
        self.assertIn("cache_data/onnx/open_pulls.json", text)

    def test_collect_snapshot(self):
        now = dt.datetime(2026, 8, 28, 8, tzinfo=dt.timezone.utc)
        responses = {
            "open": [
                {
                    "created_at": "2026-08-25T08:00:00Z",
                    "updated_at": "2026-08-28T07:00:00Z",
                },
                {
                    "created_at": "2026-08-08T08:00:00Z",
                    "updated_at": "2026-08-27T07:00:00Z",
                },
            ],
            "closed": [
                {
                    "created_at": "2026-08-25T08:00:00Z",
                    "closed_at": "2026-08-27T08:00:00Z",
                    "merged_at": "2026-08-27T08:00:00Z",
                    "updated_at": "2026-08-27T08:00:00Z",
                },
                {
                    "created_at": "2026-08-23T08:00:00Z",
                    "closed_at": "2026-08-26T08:00:00Z",
                    "merged_at": None,
                    "updated_at": "2026-08-26T08:00:00Z",
                },
                {
                    "created_at": "2026-08-15T08:00:00Z",
                    "closed_at": "2026-08-20T07:59:00Z",
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
        self.assertEqual(snapshot["opened_prs_7d"], "3")
        self.assertEqual(snapshot["closed_prs_7d"], "2")
        self.assertEqual(snapshot["merged_prs_7d"], "1")
        self.assertEqual(snapshot["avg_open_age_days"], "11.50")

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
                "opened_prs_7d": "10",
                "closed_prs_7d": "15",
                "merged_prs_7d": "20",
                "avg_open_age_days": "30.00",
            }
            second = dict(first, date="2026-08-28T09:00:00Z", open_prs="101")
            rpa.write_snapshot(path, first)
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [second])

    def test_write_snapshot_preserves_prior_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "onnxruntime", "pr_activity.csv")
            first = {
                "date": "2026-08-27T08:00:00Z",
                "open_prs": "100",
                "opened_prs_7d": "10",
                "closed_prs_7d": "15",
                "merged_prs_7d": "20",
                "avg_open_age_days": "30.00",
            }
            second = dict(first, date="2026-08-28T08:00:00Z", open_prs="101")
            rpa.write_snapshot(path, first)
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [first, second])

    def test_write_open_pull_tables_sorts_by_creation_date(self):
        pulls = [
            {
                "number": number,
                "title": f"PR {number}",
                "user": {"login": f"user{number}"},
                "created_at": f"2026-08-{number:02d}T00:00:00Z",
            }
            for number in range(1, 13)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "repo", "open_pulls.json")
            rpa.write_open_pull_tables(path, pulls)
            with open(path, encoding="utf-8") as stream:
                tables = json.load(stream)
        self.assertEqual([pull["number"] for pull in tables["latest"]], list(range(12, 2, -1)))
        self.assertEqual([pull["number"] for pull in tables["oldest"]], list(range(1, 11)))

    def test_default_repository_is_onnxruntime(self):
        self.assertEqual(rpa.DEFAULT_REPO, "microsoft/onnxruntime")


if __name__ == "__main__":
    unittest.main()
