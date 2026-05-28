"""Record statistics on merged pull requests for the tracked repositories.

For each tracked repository, this script queries the GitHub REST API for
pull requests that have been merged since the last date recorded in the
cache and appends the new rows to the corresponding CSV file under
``cache_data/<repo>/pr_stats.csv``.

The CSV columns are::

    pr_number,title,author,merged_at,commits,comments,copilot_comments,copilot_commits

Where ``comments`` counts both PR review comments (inline comments left on
the diff) and issue-style comments on the PR conversation, and the
``copilot_*`` columns count those whose author login looks like a GitHub
Copilot bot (``Copilot``, ``copilot-swe-agent[bot]``, ...).

The script is designed to be run from a GitHub Actions workflow. It reads
the ``GITHUB_TOKEN`` (or ``GH_TOKEN``) environment variable when present
in order to authenticate API requests and benefit from the higher rate
limits.

Usage::

    python scripts/record_pr_stats.py [--cache-dir DIR] [--months N]
        [--repo owner/name [--repo owner/name ...]]

Default tracked repositories are the same as for
``record_build_durations.py``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, Iterator

# Reuse helpers from ``record_build_durations`` so we keep a single place
# for the HTTP plumbing and ISO-8601 helpers.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from record_build_durations import (  # noqa: E402
    GITHUB_API,
    _format_iso,
    _log,
    _parse_iso,
    _request,
)


DEFAULT_REPOS = (
    "xadupre/onnx-light",
    "xadupre/yet-another-onnx-builder",
    "xadupre/yet-another-onnxruntime-extensions",
)

PR_CSV_FIELDS = (
    "pr_number",
    "title",
    "author",
    "merged_at",
    "commits",
    "comments",
    "copilot_comments",
    "copilot_commits",
)


def is_copilot_login(login: str | None) -> bool:
    """Return ``True`` when ``login`` looks like a GitHub Copilot bot.

    Copilot-authored content appears under a handful of related logins
    (``Copilot``, ``copilot-swe-agent[bot]``, ``github-copilot[bot]``,
    ...). Rather than enumerating them, treat any login containing the
    substring ``copilot`` (case insensitive) as Copilot.
    """
    if not login:
        return False
    return "copilot" in login.lower()


def read_existing_prs(csv_path: str) -> tuple[set[str], dt.datetime | None]:
    """Return the set of already recorded PR numbers and the most recent merged date."""
    seen: set[str] = set()
    latest: dt.datetime | None = None
    if not os.path.exists(csv_path):
        return seen, latest
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pr_number = row.get("pr_number")
            if pr_number:
                seen.add(pr_number)
            merged = row.get("merged_at")
            if merged:
                try:
                    parsed = _parse_iso(merged)
                except ValueError:
                    continue
                if latest is None or parsed > latest:
                    latest = parsed
    return seen, latest


def determine_since(latest: dt.datetime | None, months: int) -> dt.datetime:
    """Pick the lower bound of the query based on the existing cache."""
    now = dt.datetime.now(tz=dt.timezone.utc)
    default = now - dt.timedelta(days=months * 30)
    if latest is None:
        return default
    # Always re-fetch from the latest recorded date so that PRs whose
    # numbers were missed (e.g. merged out-of-order) are still considered.
    return latest


def iter_closed_pulls(
    repo: str, token: str | None, since: dt.datetime
) -> Iterator[dict]:
    """Yield closed pull requests of ``repo`` updated on or after ``since``.

    The GitHub ``/pulls`` endpoint does not support a ``since`` filter, so
    we sort the results by ``updated`` descending and stop as soon as a
    page's PRs are all older than ``since``.
    """
    page = 1
    per_page = 100
    while True:
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": str(per_page),
            "page": str(page),
        }
        url = (
            f"{GITHUB_API}/repos/{repo}/pulls?"
            + urllib.parse.urlencode(params)
        )
        payload, _ = _request(url, token)
        if not isinstance(payload, list) or not payload:
            return
        stop = True
        for pr in payload:
            updated_at = pr.get("updated_at")
            if updated_at:
                try:
                    parsed = _parse_iso(updated_at)
                except ValueError:
                    parsed = None
                if parsed is None or parsed >= since:
                    stop = False
                    yield pr
                    continue
                # ``parsed < since``: do not yield, but keep scanning the
                # rest of this page in case the sort is not perfectly
                # monotonic.
                continue
            stop = False
            yield pr
        if stop:
            return
        if len(payload) < per_page:
            return
        page += 1


def _count_paginated(url_base: str, token: str | None) -> int:
    """Return the total number of items behind a paginated GET endpoint."""
    page = 1
    per_page = 100
    total = 0
    while True:
        sep = "&" if "?" in url_base else "?"
        url = f"{url_base}{sep}per_page={per_page}&page={page}"
        payload, _ = _request(url, token)
        if not isinstance(payload, list) or not payload:
            return total
        total += len(payload)
        if len(payload) < per_page:
            return total
        page += 1


def _iter_paginated(url_base: str, token: str | None) -> Iterator[dict]:
    """Yield every item from a paginated GET endpoint."""
    page = 1
    per_page = 100
    while True:
        sep = "&" if "?" in url_base else "?"
        url = f"{url_base}{sep}per_page={per_page}&page={page}"
        payload, _ = _request(url, token)
        if not isinstance(payload, list) or not payload:
            return
        for item in payload:
            yield item
        if len(payload) < per_page:
            return
        page += 1


def fetch_pr_stats(repo: str, pr: dict, token: str | None) -> dict:
    """Return the per-PR statistics for the merged pull request ``pr``."""
    pr_number = pr.get("number")
    commits_total = 0
    copilot_commits = 0
    for commit in _iter_paginated(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/commits", token
    ):
        commits_total += 1
        author = commit.get("author") or {}
        committer = commit.get("committer") or {}
        if is_copilot_login(author.get("login")) or is_copilot_login(
            committer.get("login")
        ):
            copilot_commits += 1

    comments_total = 0
    copilot_comments = 0
    for comment in _iter_paginated(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments", token
    ):
        comments_total += 1
        user = comment.get("user") or {}
        if is_copilot_login(user.get("login")):
            copilot_comments += 1
    for comment in _iter_paginated(
        f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/comments", token
    ):
        comments_total += 1
        user = comment.get("user") or {}
        if is_copilot_login(user.get("login")):
            copilot_comments += 1

    user = pr.get("user") or {}
    return {
        "pr_number": str(pr_number) if pr_number is not None else "",
        "title": (pr.get("title") or "").replace("\r", " ").replace("\n", " "),
        "author": user.get("login") or "",
        "merged_at": pr.get("merged_at") or "",
        "commits": str(commits_total),
        "comments": str(comments_total),
        "copilot_comments": str(copilot_comments),
        "copilot_commits": str(copilot_commits),
    }


def append_rows(csv_path: str, rows: Iterable[dict]) -> int:
    """Append ``rows`` to ``csv_path``, creating the file with a header if needed."""
    rows = list(rows)
    if not rows:
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=PR_CSV_FIELDS)
                writer.writeheader()
        return 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PR_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _log(f"saved {len(rows)} row(s) to {csv_path}")
    return len(rows)


def process_repo(
    repo: str, cache_dir: str, months: int, token: str | None
) -> int:
    """Fetch new merged PRs for ``repo`` and append them to the cache file."""
    repo_name = repo.split("/", 1)[-1]
    csv_path = os.path.join(cache_dir, repo_name, "pr_stats.csv")
    seen, latest = read_existing_prs(csv_path)
    since = determine_since(latest, months)
    _log(
        f"[{repo}] cache file: {csv_path} "
        f"({len(seen)} PR(s) already recorded, "
        f"latest merged={_format_iso(latest) if latest else 'none'})"
    )
    _log(f"[{repo}] fetching closed PRs updated since {_format_iso(since)}")
    started = dt.datetime.now(tz=dt.timezone.utc)
    new_rows: list[dict] = []
    processed = 0
    try:
        for pr in iter_closed_pulls(repo, token, since):
            processed += 1
            merged_at = pr.get("merged_at")
            if not merged_at:
                # Closed but not merged.
                continue
            pr_number = str(pr.get("number", ""))
            if not pr_number or pr_number in seen:
                continue
            try:
                merged_dt = _parse_iso(merged_at)
            except ValueError:
                merged_dt = None
            if merged_dt is not None and merged_dt < since:
                # Older than the lower bound: do not record again.
                continue
            _log(
                f"[{repo}] new merged PR #{pr_number} "
                f"by {(pr.get('user') or {}).get('login')!r}; "
                "fetching its commits and comments..."
            )
            try:
                row = fetch_pr_stats(repo, pr, token)
            except Exception as exc:
                # Recording stats for one PR must not abort the whole
                # repository: log the failure and keep going.
                print(
                    f"[{repo}] failed to fetch stats for PR #{pr_number}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            new_rows.append(row)
            seen.add(pr_number)
            _log(
                f"[{repo}]   PR #{pr_number}: "
                f"commits={row['commits']} comments={row['comments']} "
                f"copilot_commits={row['copilot_commits']} "
                f"copilot_comments={row['copilot_comments']}"
            )
    finally:
        # Always flush whatever we managed to collect.
        try:
            added = append_rows(csv_path, new_rows)
        except Exception as exc:  # pragma: no cover - defensive
            added = 0
            print(
                f"[{repo}] failed to save {len(new_rows)} PR row(s) to "
                f"{csv_path}: {exc}",
                file=sys.stderr,
            )
    elapsed = (dt.datetime.now(tz=dt.timezone.utc) - started).total_seconds()
    _log(
        f"[{repo}] processed {processed} closed PR(s) from GitHub in "
        f"{elapsed:.1f}s; appended {added} new PR(s) to {csv_path}"
    )
    return added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache_data"),
        help="Path to the cache_data directory (defaults to ../cache_data).",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months to look back when the cache is empty (default: 6).",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="Repository to track (owner/name). May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    repos = args.repos or list(DEFAULT_REPOS)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    _log("record_pr_stats.py starting")
    _log(f"  cache directory : {args.cache_dir}")
    _log(f"  months fallback : {args.months}")
    _log(f"  repositories    : {', '.join(repos)}")
    if not token:
        _log("  authentication  : anonymous (no GITHUB_TOKEN/GH_TOKEN set)")
        print("warning: no GITHUB_TOKEN/GH_TOKEN set; using anonymous requests.")
    else:
        _log("  authentication  : using GITHUB_TOKEN/GH_TOKEN")

    overall_started = dt.datetime.now(tz=dt.timezone.utc)
    total = 0
    failures = 0
    for index, repo in enumerate(repos, start=1):
        _log(f"==> [{index}/{len(repos)}] processing repository {repo}")
        try:
            total += process_repo(repo, args.cache_dir, args.months, token)
        except urllib.error.HTTPError as exc:
            failures += 1
            print(
                f"[{repo}] HTTP error {exc.code}: {exc.reason}; "
                "continuing with next repository.",
                file=sys.stderr,
            )
            _log(
                f"[{repo}] aborted with HTTP {exc.code}: {exc.reason}; "
                "moving on to next repository"
            )
        except Exception as exc:
            failures += 1
            print(
                f"[{repo}] unexpected error: {type(exc).__name__}: {exc}; "
                "continuing with next repository.",
                file=sys.stderr,
            )
            _log(
                f"[{repo}] aborted with {type(exc).__name__}: {exc}; "
                "moving on to next repository"
            )
    overall_elapsed = (
        dt.datetime.now(tz=dt.timezone.utc) - overall_started
    ).total_seconds()
    _log(
        f"Done. {total} new PR(s) recorded in total across {len(repos)} "
        f"repository(ies) in {overall_elapsed:.1f}s "
        f"({failures} repository failure(s))."
    )
    print(f"Done. {total} new PR(s) recorded in total.")
    if failures and failures == len(repos):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
