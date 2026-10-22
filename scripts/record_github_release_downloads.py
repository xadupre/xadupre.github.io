"""Record daily GitHub release download counts for ONNX and LLM repositories.

Discover public xadupre onnx-light and yet-another projects, and include ONNX
upstreams, llama.cpp and vLLM. Only attached release assets have counters;
GitHub-generated source archives are excluded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path

from record_onnxruntime_release_sizes import fetch_pages

API = "https://api.github.com"
HOME = Path(__file__).resolve().parents[1] / "index.html"
EXTRA_UPSTREAMS = (
    "microsoft/onnxruntime-genai",
    "microsoft/onnxruntime-extensions",
    "ggml-org/llama.cpp",
    "vllm-project/vllm",
)


def is_owner_project(name: str) -> bool:
    """Limit owner discovery to the requested project families."""
    return name == "onnx-light" or name.startswith(("onnx-light-", "yet-another-"))


def discover_repositories(token: str, home: Path = HOME) -> list[str]:
    """Combine public owner repositories with ONNX projects on the home page."""
    repositories = {
        item["full_name"]
        for item in fetch_pages(f"{API}/users/xadupre/repos", token)
        if not item["private"] and is_owner_project(item["name"])
    }
    links = re.findall(
        r'href="https://github\.com/([^/"?#]+/[^/"?#]+)"',
        home.read_text(encoding="utf-8"),
    )
    repositories.update(
        name
        for name in links
        if (
            is_owner_project(name.split("/")[1])
            if name.startswith("xadupre/")
            else "onnx" in name.split("/")[1].lower()
        )
    )
    repositories.update(EXTRA_UPSTREAMS)
    return sorted(repositories)


def collect_repository(repository: str, token: str) -> dict:
    """Read every public release and every uploaded asset, including previews."""
    url = f"{API}/repos/{repository}/releases"
    releases = []
    for release in fetch_pages(url, token):
        if release["draft"]:
            continue
        assets = []
        for asset in fetch_pages(f"{url}/{int(release['id'])}/assets", token):
            if asset["state"] != "uploaded":
                continue
            count = asset["download_count"]
            if type(count) is not int or count < 0:
                raise ValueError(f"Invalid download count for {repository}: {asset!r}")
            assets.append(
                {
                    "id": asset["id"],
                    "name": asset["name"],
                    "download_count": count,
                }
            )
        releases.append(
            {
                "id": release["id"],
                "tag": release["tag_name"],
                "published_at": release["published_at"],
                "prerelease": release["prerelease"],
                "total": sum(asset["download_count"] for asset in assets),
                "assets": sorted(assets, key=lambda asset: asset["id"]),
            }
        )
    releases.sort(key=lambda release: (release["published_at"], release["id"]))
    return {
        "repository": repository,
        "total": sum(release["total"] for release in releases),
        "releases": releases,
    }


def record_downloads(
    cache_dir: str,
    token: str = "",
    repositories: list[str] | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Publish a complete snapshot and upsert one daily UTC total per repository."""
    selected = sorted(
        set(discover_repositories(token) if repositories is None else repositories)
    )
    if not selected:
        raise RuntimeError("No ONNX repositories selected")
    current = []
    for repository in selected:
        print(f"Collecting release downloads for {repository}", flush=True)
        current.append(collect_repository(repository, token))

    path = Path(cache_dir) / "github-release-downloads.json"
    history = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        history = {entry["date"]: entry for entry in previous["history"]}
    timestamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    day = timestamp.strftime("%Y-%m-%d")
    history[day] = {
        "date": day,
        "totals": {item["repository"]: item["total"] for item in current},
        "assets": {
            item["repository"]: {
                str(asset["id"]): asset["download_count"]
                for release in item["releases"]
                for asset in release["assets"]
            }
            for item in current
        },
    }
    result = {
        "updated_at": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repositories": current,
        "history": [history[date] for date in sorted(history)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"Recorded {len(current)} repositories in {path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache_data")
    parser.add_argument(
        "--repo",
        action="append",
        help="Explicit owner/name; repeat to select repositories.",
    )
    args = parser.parse_args(argv)
    record_downloads(args.cache_dir, os.environ.get("GITHUB_TOKEN", ""), args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
