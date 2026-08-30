"""Record a daily snapshot of pull request activity for ONNX Runtime.

The snapshot contains the current number of open pull requests; the numbers
opened, closed, and merged during the preceding seven days; and the average
age in days of the pull requests that are still open. Rows are stored in
``cache_data/onnxruntime/pr_activity.csv``.

Usage::

    python scripts/record_pr_activity.py [--cache-dir DIR]
        [--repo owner/name]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
from collections.abc import Iterator

from record_build_durations import GITHUB_API, _format_iso, _log, _parse_iso, _request

DEFAULT_REPO = "microsoft/onnxruntime"
CSV_FIELDS = (
    "date",
    "open_prs",
    "opened_prs_7d",
    "closed_prs_7d",
    "merged_prs_7d",
    "avg_open_age_days",
)


def iter_pulls(repo: str, state: str, token: str | None) -> Iterator[dict]:
    """Yield pull requests in descending update order."""
    page = 1
    per_page = 100
    while True:
        params = {
            "state": state,
            "sort": "updated",
            "direction": "desc",
            "per_page": str(per_page),
            "page": str(page),
        }
        url = f"{GITHUB_API}/repos/{repo}/pulls?" + urllib.parse.urlencode(params)
        payload, _ = _request(url, token)
        if not isinstance(payload, list) or not payload:
            return
        yield from payload
        if len(payload) < per_page:
            return
        page += 1


def _is_at_or_after(value: str | None, since: dt.datetime) -> bool:
    """Return whether an ISO timestamp is on or after ``since``."""
    if not value:
        return False
    try:
        return _parse_iso(value) >= since
    except ValueError:
        return False


def collect_snapshot(
    repo: str,
    token: str | None,
    now: dt.datetime | None = None,
    open_pulls: list[dict] | None = None,
) -> dict[str, str]:
    """Collect the current PR activity metrics for ``repo``."""
    if now is None:
        now = dt.datetime.now(tz=dt.timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    else:
        now = now.astimezone(dt.timezone.utc)

    if open_pulls is None:
        open_pulls = list(iter_pulls(repo, "open", token))
    ages = []
    for pr in open_pulls:
        created_at = pr.get("created_at")
        if not created_at:
            continue
        try:
            created = _parse_iso(created_at)
        except ValueError:
            continue
        ages.append(max(0.0, (now - created).total_seconds() / 86400))

    since = now - dt.timedelta(days=7)
    opened = 0
    closed = 0
    merged = 0
    for pr in open_pulls:
        if _is_at_or_after(pr.get("created_at"), since):
            opened += 1
    for pr in iter_pulls(repo, "closed", token):
        updated_at = pr.get("updated_at")
        if updated_at:
            try:
                if _parse_iso(updated_at) < since:
                    break
            except ValueError:
                pass
        if _is_at_or_after(pr.get("created_at"), since):
            opened += 1
        if _is_at_or_after(pr.get("closed_at"), since):
            closed += 1
        if _is_at_or_after(pr.get("merged_at"), since):
            merged += 1

    average_age = sum(ages) / len(ages) if ages else 0.0
    return {
        "date": _format_iso(now),
        "open_prs": str(len(open_pulls)),
        "opened_prs_7d": str(opened),
        "closed_prs_7d": str(closed),
        "merged_prs_7d": str(merged),
        "avg_open_age_days": f"{average_age:.2f}",
    }


def write_snapshot(csv_path: str, snapshot: dict[str, str]) -> None:
    """Insert or replace the snapshot for its UTC calendar date."""
    rows: list[dict[str, str]] = []
    snapshot_day = snapshot["date"][:10]
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if (row.get("date") or "")[:10] != snapshot_day
            ]
    rows.append(snapshot)
    rows.sort(key=lambda row: row["date"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_open_pull_tables(json_path: str, pulls: list[dict]) -> None:
    """Write the newest and oldest currently open pull requests."""

    def row(pull: dict) -> dict:
        user = pull.get("user")
        return {
            "number": pull.get("number"),
            "title": pull.get("title", ""),
            "user": user.get("login", "") if isinstance(user, dict) else "",
            "created_at": pull.get("created_at", ""),
        }

    ordered = sorted(pulls, key=lambda pull: pull.get("created_at") or "")
    payload = {
        "latest": [row(pull) for pull in ordered[-10:][::-1]],
        "oldest": [row(pull) for pull in ordered[:10]],
    }
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache_data"),
        help="Path to the cache_data directory.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository to track.")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo_name = args.repo.split("/", 1)[-1]
    csv_path = os.path.join(args.cache_dir, repo_name, "pr_activity.csv")
    json_path = os.path.join(args.cache_dir, repo_name, "open_pulls.json")
    _log(f"collecting daily PR activity for {args.repo}")
    open_pulls = list(iter_pulls(args.repo, "open", token))
    snapshot = collect_snapshot(args.repo, token, open_pulls=open_pulls)
    write_snapshot(csv_path, snapshot)
    write_open_pull_tables(json_path, open_pulls)
    _log(
        f"saved {csv_path}: open={snapshot['open_prs']}, "
        f"opened_7d={snapshot['opened_prs_7d']}, "
        f"closed_7d={snapshot['closed_prs_7d']}, "
        f"merged_7d={snapshot['merged_prs_7d']}, "
        f"average_open_age={snapshot['avg_open_age_days']} days"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
