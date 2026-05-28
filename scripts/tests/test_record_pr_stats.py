"""Tests for ``scripts.record_pr_stats``."""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile
import unittest
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import record_pr_stats as rps  # noqa: E402


class TestRecordPrStats(unittest.TestCase):
    def test_is_copilot_login(self):
        self.assertTrue(rps.is_copilot_login("Copilot"))
        self.assertTrue(rps.is_copilot_login("copilot"))
        self.assertTrue(rps.is_copilot_login("copilot-swe-agent[bot]"))
        self.assertTrue(rps.is_copilot_login("github-copilot[bot]"))
        self.assertFalse(rps.is_copilot_login("xadupre"))
        self.assertFalse(rps.is_copilot_login("github-actions[bot]"))
        self.assertFalse(rps.is_copilot_login(None))
        self.assertFalse(rps.is_copilot_login(""))

    def test_append_rows_and_read_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "pr_stats.csv")
            added = rps.append_rows(
                path,
                [
                    {
                        "pr_number": "12",
                        "title": "Fix bug",
                        "author": "alice",
                        "merged_at": "2024-05-01T10:00:00Z",
                        "commits": "3",
                        "comments": "5",
                        "copilot_comments": "1",
                        "copilot_commits": "0",
                    }
                ],
            )
            self.assertEqual(added, 1)
            seen, latest = rps.read_existing_prs(path)
            self.assertEqual(seen, {"12"})
            self.assertEqual(rps._format_iso(latest), "2024-05-01T10:00:00Z")

    def test_append_rows_creates_empty_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "pr_stats.csv")
            self.assertEqual(rps.append_rows(path, []), 0)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                header = fh.readline().strip().split(",")
            self.assertEqual(tuple(header), rps.PR_CSV_FIELDS)

    def test_determine_since_uses_latest(self):
        latest = dt.datetime(2024, 5, 1, tzinfo=dt.timezone.utc)
        self.assertEqual(rps.determine_since(latest, 6), latest)

    def test_determine_since_defaults_to_months(self):
        since = rps.determine_since(None, 6)
        now = dt.datetime.now(tz=dt.timezone.utc)
        delta = now - since
        self.assertGreater(delta.days, 175)
        self.assertLess(delta.days, 185)

    def test_fetch_pr_stats_counts_commits_and_comments(self):
        pr = {
            "number": 42,
            "title": "Implement feature\nwith newline",
            "user": {"login": "alice"},
            "merged_at": "2024-06-01T12:00:00Z",
        }
        # Simulate paginated endpoints. The keys are the URL prefixes that
        # ``fetch_pr_stats`` will request.
        responses = {
            "/repos/owner/repo/pulls/42/commits": [
                {"author": {"login": "alice"}, "committer": {"login": "alice"}},
                {
                    "author": {"login": "copilot-swe-agent[bot]"},
                    "committer": {"login": "alice"},
                },
                {"author": None, "committer": {"login": "Copilot"}},
            ],
            "/repos/owner/repo/issues/42/comments": [
                {"user": {"login": "alice"}},
                {"user": {"login": "Copilot"}},
            ],
            "/repos/owner/repo/pulls/42/comments": [
                {"user": {"login": "bob"}},
                {"user": {"login": "alice"}},
                {"user": {"login": "copilot-swe-agent[bot]"}},
            ],
        }

        def fake_request(url, token):
            # Strip query string before lookup.
            base = url.split("?", 1)[0]
            base = base.replace(rps.GITHUB_API, "")
            # ``page=`` appears after ``per_page=`` in the URL, so use
            # rsplit to extract the actual page number.
            page_str = url.rsplit("page=", 1)[1].split("&", 1)[0]
            page = int(page_str)
            data = responses.get(base, [])
            # Return all items on page 1 (small fixtures), empty after.
            if page == 1:
                return data, {}
            return [], {}

        original = rps._request
        rps._request = fake_request
        try:
            row = rps.fetch_pr_stats("owner/repo", pr, token=None)
        finally:
            rps._request = original
        self.assertEqual(row["pr_number"], "42")
        self.assertEqual(row["author"], "alice")
        self.assertEqual(row["merged_at"], "2024-06-01T12:00:00Z")
        # Newlines in titles must be flattened so CSV stays well-formed.
        self.assertNotIn("\n", row["title"])
        self.assertEqual(row["commits"], "3")
        # 2 commits whose author/committer login contains "copilot".
        self.assertEqual(row["copilot_commits"], "2")
        # 2 issue comments + 3 review comments = 5 total.
        self.assertEqual(row["comments"], "5")
        # 1 copilot issue comment + 1 copilot review comment.
        self.assertEqual(row["copilot_comments"], "2")

    def test_iter_closed_pulls_stops_when_older_than_since(self):
        # Build two pages. Page 1 has 100 PRs (a full page) so the
        # iterator continues to page 2; page 2 contains only PRs older
        # than ``since``, which must cause the iterator to stop without
        # requesting page 3.
        page1 = [
            {
                "number": 200 - i,
                "updated_at": "2024-06-10T00:00:00Z",
                "merged_at": "2024-06-10T00:00:00Z",
            }
            for i in range(100)
        ]
        page2 = [
            {
                "number": 50,
                "updated_at": "2023-01-01T00:00:00Z",
                "merged_at": "2023-01-01T00:00:00Z",
            },
        ]
        pages = [page1, page2, []]
        calls: list[int] = []

        def fake_request(url, token):
            page = int(url.rsplit("page=", 1)[1].split("&", 1)[0])
            calls.append(page)
            idx = page - 1
            if idx >= len(pages):
                return [], {}
            # Pages may be partial; emulate ``per_page=100`` by padding when
            # the test wants more pages. Here all pages are short so the
            # iterator stops naturally.
            return pages[idx], {}

        original = rps._request
        rps._request = fake_request
        try:
            since = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
            prs = list(rps.iter_closed_pulls("owner/repo", None, since))
        finally:
            rps._request = original
        numbers = [pr["number"] for pr in prs]
        # Page 1's 100 PRs are after ``since`` and are yielded; page 2
        # contains only a PR older than ``since`` (not yielded), and
        # because every PR on that page was old the iterator stops
        # without requesting page 3.
        self.assertEqual(len(numbers), 100)
        self.assertNotIn(50, numbers)
        self.assertEqual(calls, [1, 2])

    def test_main_continues_after_repo_failure(self):
        calls: list[str] = []

        def fake_process(repo, cache_dir, months, token):
            calls.append(repo)
            if repo == "owner/bad":
                raise urllib.error.HTTPError(
                    "http://x", 500, "boom", hdrs=None, fp=None
                )
            return 1

        original = rps.process_repo
        rps.process_repo = fake_process
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rc = rps.main(
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
            rps.process_repo = original
        self.assertEqual(calls, ["owner/good1", "owner/bad", "owner/good2"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
