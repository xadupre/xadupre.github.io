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
    def test_dashboard_age_series_use_separate_axes(self):
        root = os.path.dirname(os.path.dirname(HERE))
        for project, options in (("onnx", "options"), ("onnxruntime", "commonOptions")):
            with self.subTest(project=project):
                page = os.path.join(root, "dashboard", project, "pr-activity.html")
                with open(page, encoding="utf-8") as stream:
                    text = stream.read()
                age_chart = text.split('getElementById("chartAge"), {', 1)[1].split(
                    'const body = document.getElementById("observationsBody")', 1
                )[0]
                self.assertIn('data: points("age"), yAxisID: "y"', age_chart)
                self.assertIn(
                    'data: points("mergedAge"), yAxisID: "yMerged"', age_chart
                )
                self.assertIn(f"...{options},", age_chart)
                self.assertIn(f"...{options}.scales,", age_chart)
                for axis, position, label in (
                    ("y", "left", "average open age (days)"),
                    ("yMerged", "right", "average merged age (365d, days)"),
                ):
                    self.assertRegex(
                        age_chart,
                        rf'{axis}: \{{\s*\.\.\.{options}\.scales\.y,\s*'
                        rf'position: "{position}",\s*'
                        rf'title: \{{ display: true, text: "{re.escape(label)}"',
                    )
                self.assertIn("grid: { drawOnChartArea: false }", age_chart)
                self.assertNotIn("yMerged", text.split(
                    'getElementById("chartAge"), {', 1
                )[0])

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
            self.assertIn("opened_prs_1d", text)
            self.assertIn("closed_prs_1d", text)
            self.assertIn("merged_prs_1d", text)
            self.assertIn("avg_open_age_days", text)
            self.assertIn("avg_merged_age_days_365d", text)
            self.assertNotIn("avg_merged_age_days_7d", text)
            self.assertIn('id="latestMergedAge"', text)
            self.assertIn('data: points("mergedAge")', text)
            self.assertIn('latest.mergedAge === null ? "N/A"', text)
            self.assertIn('row.mergedAge === null ? "N/A"', text)
            self.assertIn("average merged age (365d, days)", text)
            self.assertIn("rolling year (365 days)", text)
            self.assertIn("loadChartJs()", text)
            self.assertIn('unit: "day"', text)
            self.assertIn('id="chartOpen"', text)
            self.assertIn('id="chartActivity"', text)
            self.assertIn('id="chartAge"', text)
            self.assertIn('id="chartAgeDistribution"', text)
            self.assertIn("ages_days", text)
            self.assertIn(
                'for (const id of ["chartOpen", "chartActivity", "chartAge"])', text
            )
            self.assertRegex(
                text,
                r'(?s)getElementById\("chartOpen"\).*?'
                r'label: "open PRs".*?'
                r'getElementById\("chartActivity"\).*?'
                r'label: "opened per day".*?'
                r'label: "opened 7-day moving average"',
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
        self.assertIn("repository: xadupre/cache_data", text)
        self.assertIn(
            'bash scripts/commit_cache_data.sh cache_data '
            '"Update onnxruntime PR activity cache"',
            text,
        )

        onnx_workflow = os.path.join(
            root, ".github", "workflows", "record_onnx_pr_activity.yml"
        )
        with open(onnx_workflow, encoding="utf-8") as stream:
            text = stream.read()
        self.assertIn('cron: "7 5 * * *"', text)
        self.assertIn(
            "python -u scripts/record_pr_activity.py --repo onnx/onnx", text
        )
        self.assertIn("repository: xadupre/cache_data", text)
        self.assertIn(
            'bash scripts/commit_cache_data.sh cache_data '
            '"Update onnx PR activity cache"',
            text,
        )

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
        self.assertEqual(snapshot["opened_prs_1d"], "0")
        self.assertEqual(snapshot["closed_prs_1d"], "1")
        self.assertEqual(snapshot["merged_prs_1d"], "1")
        self.assertEqual(snapshot["avg_open_age_days"], "11.50")
        self.assertEqual(snapshot["avg_merged_age_days_365d"], "3.50")

    def test_merged_age_uses_merge_time_and_year_window(self):
        now = dt.datetime(2026, 8, 28, 8, tzinfo=dt.timezone.utc)
        pulls = [
            {
                "created_at": created,
                "merged_at": merged,
                "closed_at": "2026-08-28T08:00:00Z",
                "updated_at": "2026-08-28T08:00:00Z",
            }
            for created, merged in [
                ("2025-08-08T08:00:00Z", "2025-08-28T08:00:00Z"),
                ("2026-08-27T08:00:00Z", "2026-08-27T20:00:00Z"),
                ("2025-07-01T08:00:00Z", "2025-08-28T07:59:59Z"),
                ("2026-08-01T08:00:00Z", None),
                (None, "2026-08-28T08:00:00Z"),
                ("invalid", "2026-08-28T08:00:00Z"),
                ("2026-08-28T08:00:00Z", "2026-08-27T08:00:00Z"),
                ("2026-08-01T08:00:00Z", "invalid"),
                ("2026-08-01T08:00:00Z", "2026-08-28T08:00:01Z"),
            ]
        ]
        original = rpa.iter_pulls
        rpa.iter_pulls = lambda repo, state, token: iter(pulls)
        try:
            snapshot = rpa.collect_snapshot("owner/repo", None, now, open_pulls=[])
        finally:
            rpa.iter_pulls = original
        self.assertEqual(snapshot["merged_prs_7d"], "5")
        self.assertEqual(snapshot["avg_merged_age_days_365d"], "10.25")

    def test_merged_age_without_valid_ages_is_blank(self):
        original = rpa.iter_pulls
        try:
            for pulls in ([], [{"merged_at": "2026-08-28T08:00:00Z"}]):
                with self.subTest(pulls=pulls):
                    rpa.iter_pulls = lambda repo, state, token: iter(pulls)
                    snapshot = rpa.collect_snapshot(
                        "owner/repo", None,
                        dt.datetime(2026, 8, 28, 8, tzinfo=dt.timezone.utc),
                        open_pulls=[],
                    )
                    self.assertEqual(snapshot["avg_merged_age_days_365d"], "")
        finally:
            rpa.iter_pulls = original

    def test_merged_age_window_drops_expired_merges_and_adds_new_merges(self):
        original = rpa.iter_pulls
        try:
            for year in (2024, 2026):
                with self.subTest(year=year):
                    now = dt.datetime(year, 8, 28, 8, tzinfo=dt.timezone.utc)

                    def pull(merged, age):
                        return {
                            "created_at": rpa._format_iso(merged - dt.timedelta(days=age)),
                            "merged_at": rpa._format_iso(merged),
                            "closed_at": rpa._format_iso(merged),
                            "updated_at": rpa._format_iso(merged),
                        }

                    pulls = [
                        pull(now - dt.timedelta(days=30), 10),
                        pull(now - dt.timedelta(days=365), 20),
                        pull(now - dt.timedelta(days=365, seconds=1), 100),
                    ]

                    def iter_pulls(repo, state, token):
                        yield from pulls
                        self.fail("Collection must stop once updates predate the year")

                    rpa.iter_pulls = iter_pulls
                    first = rpa.collect_snapshot("owner/repo", None, now, open_pulls=[])
                    self.assertEqual(first["avg_merged_age_days_365d"], "15.00")
                    self.assertEqual(first["merged_prs_7d"], "0")
                    self.assertEqual(first["merged_prs_1d"], "0")

                    pulls.insert(0, pull(now + dt.timedelta(hours=12), 6))
                    second = rpa.collect_snapshot(
                        "owner/repo", None, now + dt.timedelta(days=1), open_pulls=[]
                    )
                    self.assertEqual(second["avg_merged_age_days_365d"], "8.00")
                    self.assertEqual(second["merged_prs_7d"], "1")
                    self.assertEqual(second["merged_prs_1d"], "1")

                    expired = rpa.collect_snapshot(
                        "owner/repo", None, now + dt.timedelta(days=366), open_pulls=[]
                    )
                    self.assertEqual(expired["avg_merged_age_days_365d"], "")
        finally:
            rpa.iter_pulls = original

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
                "opened_prs_1d": "1",
                "closed_prs_1d": "2",
                "merged_prs_1d": "3",
                "avg_open_age_days": "30.00",
                "avg_merged_age_days_365d": "2.50",
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
                "opened_prs_1d": "1",
                "closed_prs_1d": "2",
                "merged_prs_1d": "3",
                "avg_open_age_days": "30.00",
                "avg_merged_age_days_365d": "2.50",
            }
            second = dict(first, date="2026-08-28T08:00:00Z", open_prs="101")
            rpa.write_snapshot(path, first)
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [first, second])

    def test_write_snapshot_preserves_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pr_activity.csv")
            legacy_fields = [
                field for field in rpa.CSV_FIELDS if field != "avg_merged_age_days_365d"
            ]
            first = dict.fromkeys(legacy_fields, "0")
            first["date"] = "2026-08-27T08:00:00Z"
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow(first)
            second = dict(
                first, date="2026-08-28T08:00:00Z", avg_merged_age_days_365d="2.50"
            )
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [dict(first, avg_merged_age_days_365d=""), second])

    def test_write_snapshot_does_not_relabel_seven_day_averages_as_yearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pr_activity.csv")
            legacy_fields = [
                "avg_merged_age_days_7d" if field == "avg_merged_age_days_365d" else field
                for field in rpa.CSV_FIELDS
            ]
            first = dict.fromkeys(legacy_fields, "0")
            first.update(date="2026-08-27T08:00:00Z", avg_merged_age_days_7d="2.50")
            with open(path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow(first)
            second = dict.fromkeys(rpa.CSV_FIELDS, "0")
            second.update(date="2026-08-28T08:00:00Z", avg_merged_age_days_365d="15.00")
            rpa.write_snapshot(path, second)
            rpa.write_snapshot(path, second)
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, list(rpa.CSV_FIELDS))
            first.pop("avg_merged_age_days_7d")
            self.assertEqual(rows, [dict(first, avg_merged_age_days_365d=""), second])

    def test_write_open_pull_tables_sorts_by_creation_date_and_records_ages(self):
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
            rpa.write_open_pull_tables(
                path, pulls, now=dt.datetime(2026, 8, 13, tzinfo=dt.timezone.utc)
            )
            with open(path, encoding="utf-8") as stream:
                tables = json.load(stream)
        self.assertEqual([pull["number"] for pull in tables["latest"]], list(range(12, 2, -1)))
        self.assertEqual([pull["number"] for pull in tables["oldest"]], list(range(1, 11)))
        self.assertEqual(tables["ages_days"], [13.0 - number for number in range(1, 13)])

    def test_default_repository_is_onnxruntime(self):
        self.assertEqual(rpa.DEFAULT_REPO, "microsoft/onnxruntime")


if __name__ == "__main__":
    unittest.main()
