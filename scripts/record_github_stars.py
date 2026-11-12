"""Record daily GitHub star counts for the release-download dashboard projects.

Run with ``--cache-dir DIR`` and repeat ``--repo owner/name`` to select
repositories explicitly. History starts with the first collection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path

from record_build_durations import _request
from record_github_release_downloads import API, discover_repositories


def record_stars(
    cache_dir: str,
    token: str = "",
    repositories: list[str] | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Keep one snapshot per UTC day; failed collections leave history intact."""
    selected = sorted(
        set(discover_repositories(token) if repositories is None else repositories)
    )
    if not selected:
        raise ValueError("No repositories selected")
    counts = {}
    for repository in selected:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError(f"Invalid repository: {repository!r}")
        payload, _ = _request(f"{API}/repos/{repository}", token)
        count = payload.get("stargazers_count")
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid star count for {repository}: {count!r}")
        counts[repository] = count

    path = Path(cache_dir) / "github-stars.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    history = {row["date"]: row for row in previous.get("history", [])}
    timestamp = now or dt.datetime.now(dt.UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.UTC)
    timestamp = timestamp.astimezone(dt.UTC)
    day = timestamp.strftime("%Y-%m-%d")
    history[day] = {
        "date": day,
        "stars": {**history.get(day, {}).get("stars", {}), **counts},
    }
    data = {
        "updated_at": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "history": [history[key] for key in sorted(history)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Recorded star counts for {len(counts)} repositories in {path}", flush=True)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", default=str(Path(__file__).resolve().parents[1] / "cache_data")
    )
    parser.add_argument("--repo", action="append", help="Repository (repeatable).")
    args = parser.parse_args(argv)
    record_stars(
        args.cache_dir,
        os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN", ""),
        args.repo,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
