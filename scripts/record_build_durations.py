"""Record GitHub Actions build durations for the tracked repositories.

For each tracked repository, this script queries the GitHub Actions REST API
for workflow runs created since the last date recorded in the cache and
appends the new rows to the corresponding CSV file under
``cache_data/<repo>/build_durations.csv``.

The CSV columns are::

    run_id,workflow,event,status,conclusion,created_at,updated_at,duration_seconds,head_sha

The script is designed to be run from a GitHub Actions workflow. It reads the
``GITHUB_TOKEN`` (or ``GH_TOKEN``) environment variable when present in order
to authenticate API requests and benefit from the higher rate limits.

Usage::

    python scripts/record_build_durations.py [--cache-dir DIR] [--months N]
        [--repo owner/name [--repo owner/name ...]]

Default tracked repositories are the four repositories whose documentation is
published on https://xadupre.github.io/.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable, Iterator


DEFAULT_REPOS = (
    "xadupre/onnx-light",
    "xadupre/yet-another-onnx-builder",
    "xadupre/yet-another-onnxruntime-extensions",
)

CSV_FIELDS = (
    "run_id",
    "workflow",
    "event",
    "status",
    "conclusion",
    "created_at",
    "updated_at",
    "duration_seconds",
    "head_sha",
)

GITHUB_API = "https://api.github.com"


def _parse_iso(value: str) -> dt.datetime:
    """Parse an ISO 8601 timestamp returned by the GitHub API."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return dt.datetime.fromisoformat(value)


def _format_iso(value: dt.datetime) -> str:
    """Format a UTC datetime as the ISO 8601 string used in the CSV files."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    else:
        value = value.astimezone(dt.timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def read_existing(csv_path: str) -> tuple[set[str], dt.datetime | None]:
    """Return the set of already recorded run ids and the most recent date."""
    seen: set[str] = set()
    latest: dt.datetime | None = None
    if not os.path.exists(csv_path):
        return seen, latest
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_id = row.get("run_id")
            if run_id:
                seen.add(run_id)
            created = row.get("created_at")
            if created:
                try:
                    parsed = _parse_iso(created)
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
    # Always re-fetch from the latest recorded date so that runs whose status
    # changed (e.g. from "in_progress" to "completed") are refreshed.
    return latest


def _request(url: str, token: str | None) -> tuple[dict, dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xadupre.github.io-record-build-durations",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - api.github.com
        payload = json.loads(resp.read().decode("utf-8"))
        return payload, dict(resp.headers)


# GitHub's ``/actions/runs`` endpoint silently caps the total number of
# returned results to 1000 (10 pages of 100 items), regardless of how many runs
# actually match the query. For repositories with a lot of activity this means
# a single ``created>=since`` query covering several months only returns the
# most recent ~1000 runs, hiding the older history. To work around this we
# split the requested time range into smaller windows and recursively split
# any window that hits the cap.
_RUNS_RESULT_CAP = 1000


def _fetch_runs_window(
    repo: str,
    start: dt.datetime,
    end: dt.datetime,
    token: str | None,
) -> tuple[list[dict], bool]:
    """Fetch all runs of ``repo`` created in ``[start, end]``.

    Returns the list of runs and a flag indicating whether the GitHub
    1000-result cap was hit (in which case the caller should split the window).
    """
    created_filter = _format_iso(start) + ".." + _format_iso(end)
    runs: list[dict] = []
    page = 1
    per_page = 100
    while True:
        params = {
            "created": created_filter,
            "per_page": str(per_page),
            "page": str(page),
        }
        url = (
            f"{GITHUB_API}/repos/{repo}/actions/runs?"
            + urllib.parse.urlencode(params, safe=":>=<.")
        )
        payload, _ = _request(url, token)
        page_runs = payload.get("workflow_runs", [])
        if not page_runs:
            break
        runs.extend(page_runs)
        if len(page_runs) < per_page:
            break
        if len(runs) >= _RUNS_RESULT_CAP:
            # We reached the API cap; the window most likely contains more
            # runs than we got back. Signal the caller to split.
            return runs, True
        page += 1
    return runs, False


def iter_workflow_runs(
    repo: str,
    since: dt.datetime,
    token: str | None,
    until: dt.datetime | None = None,
    initial_window_days: int = 7,
) -> Iterator[dict]:
    """Yield workflow runs of ``repo`` created on or after ``since``.

    The query is split into windows of ``initial_window_days`` days. Any
    window that hits GitHub's 1000-result cap is recursively split in half
    (down to a minimum of one hour) so that the full history can be
    retrieved even for very active repositories.
    """
    if until is None:
        until = dt.datetime.now(tz=dt.timezone.utc)
    if since > until:
        return
    window = dt.timedelta(days=max(initial_window_days, 1))
    min_window = dt.timedelta(hours=1)
    # Walk from the most recent window backwards so that any incremental
    # progress is biased towards the freshest data.
    pending: list[tuple[dt.datetime, dt.datetime]] = []
    cursor = until
    while cursor > since:
        start = max(since, cursor - window)
        pending.append((start, cursor))
        cursor = start
    seen_ids: set[str] = set()
    while pending:
        start, end = pending.pop(0)
        runs, capped = _fetch_runs_window(repo, start, end, token)
        if capped and (end - start) > min_window:
            mid = start + (end - start) / 2
            # Re-queue the two halves (newest first) and skip yielding the
            # truncated page; the split queries will return the full data.
            pending.insert(0, (mid, end))
            pending.insert(1, (start, mid))
            continue
        for run in runs:
            run_id = str(run.get("id", ""))
            if run_id and run_id in seen_ids:
                continue
            if run_id:
                seen_ids.add(run_id)
            yield run


def run_to_row(run: dict) -> dict | None:
    """Convert a workflow run payload into a CSV row.

    Returns ``None`` for runs that have not finished yet so that they are
    re-fetched on the next invocation when their duration is known.
    """
    conclusion = run.get("conclusion")
    status = run.get("status")
    if status != "completed" or not conclusion:
        return None
    created_at = run.get("created_at")
    updated_at = run.get("updated_at")
    if not created_at or not updated_at:
        return None
    try:
        duration = (_parse_iso(updated_at) - _parse_iso(created_at)).total_seconds()
    except ValueError:
        return None
    return {
        "run_id": str(run.get("id", "")),
        "workflow": run.get("name") or "",
        "event": run.get("event") or "",
        "status": status,
        "conclusion": conclusion,
        "created_at": created_at,
        "updated_at": updated_at,
        "duration_seconds": f"{duration:.0f}",
        "head_sha": run.get("head_sha") or "",
    }


def append_rows(csv_path: str, rows: Iterable[dict]) -> int:
    """Append ``rows`` to ``csv_path``, creating the file with a header if needed."""
    rows = list(rows)
    if not rows:
        # Still make sure the file exists with the proper header so that the
        # cache directory is present in the repository.
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
        return 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def process_repo(
    repo: str, cache_dir: str, months: int, token: str | None
) -> int:
    """Fetch new runs for ``repo`` and append them to the cache file."""
    repo_name = repo.split("/", 1)[-1]
    csv_path = os.path.join(cache_dir, repo_name, "build_durations.csv")
    seen, latest = read_existing(csv_path)
    since = determine_since(latest, months)
    print(f"[{repo}] fetching runs since {_format_iso(since)}")
    new_rows: list[dict] = []
    for run in iter_workflow_runs(repo, since, token):
        run_id = str(run.get("id", ""))
        if not run_id or run_id in seen:
            continue
        row = run_to_row(run)
        if row is None:
            continue
        new_rows.append(row)
        seen.add(run_id)
    added = append_rows(csv_path, new_rows)
    print(f"[{repo}] appended {added} new run(s) to {csv_path}")
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
    if not token:
        print("warning: no GITHUB_TOKEN/GH_TOKEN set; using anonymous requests.")

    total = 0
    for repo in repos:
        try:
            total += process_repo(repo, args.cache_dir, args.months, token)
        except urllib.error.HTTPError as exc:
            print(f"[{repo}] HTTP error {exc.code}: {exc.reason}", file=sys.stderr)
            return 1
    print(f"Done. {total} new run(s) recorded in total.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
