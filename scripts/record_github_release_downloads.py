"""Record daily GitHub release download counts for ONNX and LLM repositories.

Discover public xadupre onnx-light, yet-another and arranged-agents projects,
and include ONNX upstreams, llama.cpp and vLLM. Only attached release assets
have counters; GitHub-generated source archives are excluded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
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
    if name == "onnx-light-com-microsoft":
        return False
    return name in ("onnx-light", "arranged-agents") or name.startswith(
        ("onnx-light-", "yet-another-")
    )


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
        release_assets = release["assets"]
        # Avoid one API request per release unless the embedded list may be full.
        if len(release_assets) >= 100:
            release_assets = fetch_pages(f"{url}/{int(release['id'])}/assets", token)
        assets = []
        for asset in release_assets:
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
    """Publish a small summary and one compressed history file per repository."""
    selected = sorted(
        set(discover_repositories(token) if repositories is None else repositories)
    )
    if not selected:
        raise RuntimeError("No ONNX repositories selected")
    current = []
    for repository in selected:
        print(f"Collecting release downloads for {repository}", flush=True)
        current.append(collect_repository(repository, token))

    legacy_path = Path(cache_dir) / "github-release-downloads.json"
    compressed_path = legacy_path.with_suffix(".json.gz")
    directory = Path(cache_dir) / "github-release-downloads"
    path = directory / "index.json"
    legacy = None
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    elif compressed_path.exists():
        with gzip.open(compressed_path, "rt", encoding="utf-8") as stream:
            previous = legacy = json.load(stream)
    elif legacy_path.exists():
        previous = legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    else:
        previous = {"history": []}
    history = {
        entry["date"]: {"date": entry["date"], "totals": entry["totals"]}
        for entry in previous["history"]
    }
    timestamp = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC)
    day = timestamp.strftime("%Y-%m-%d")
    history[day] = {
        "date": day,
        "totals": {item["repository"]: item["total"] for item in current},
    }
    updated_at = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    projects = {item["repository"]: item for item in current}
    if legacy is not None:
        for item in legacy.get("repositories", []):
            projects.setdefault(item["repository"], item)
        for entry in legacy["history"]:
            for repository in (
                entry.get("totals", {}).keys() | entry.get("assets", {}).keys()
            ):
                projects.setdefault(
                    repository, {"repository": repository, "total": 0, "releases": []}
                )
    pending = []
    for repository, item in sorted(projects.items()):
        # GitHub owner/repository names cannot contain path separators or dot segments.
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository) or (
            repository.split("/")[1] in (".", "..")
        ):
            raise ValueError(f"Invalid repository name: {repository!r}")
        project_path = directory / f"{repository}.json.gz"
        if legacy is not None:
            entries = []
            for entry in legacy["history"]:
                row = {"date": entry["date"], "totals": {}}
                if repository in entry.get("totals", {}):
                    row["totals"][repository] = entry["totals"][repository]
                if repository in entry.get("assets", {}):
                    row["assets"] = {repository: entry["assets"][repository]}
                if row["totals"] or "assets" in row:
                    entries.append(row)
        elif project_path.exists():
            with gzip.open(project_path, "rt", encoding="utf-8") as stream:
                entries = json.load(stream)["history"]
        else:
            if any(
                row["repository"] == repository
                for row in previous.get("repositories", [])
            ):
                raise FileNotFoundError(
                    f"Missing release download history: {project_path}"
                )
            entries = []
        project_history = {entry["date"]: entry for entry in entries}
        if repository in selected:
            project_history[day] = {
                "date": day,
                "totals": {repository: item["total"]},
                "assets": {
                    repository: {
                        str(asset["id"]): asset["download_count"]
                        for release in item["releases"]
                        for asset in release["assets"]
                    }
                },
            }
        project = {
            "updated_at": (
                legacy.get("updated_at", updated_at)
                if legacy is not None and repository not in selected
                else updated_at
            ),
            "repositories": [item],
            "history": [project_history[date] for date in sorted(project_history)],
        }
        project_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = project_path.with_suffix(".gz.tmp")
        temporary.write_bytes(
            gzip.compress(
                (json.dumps(project, separators=(",", ":")) + "\n").encode("utf-8"),
                mtime=0,
            )
        )
        pending.append((temporary, project_path))
    result = {
        "updated_at": updated_at,
        "repositories": [
            {
                "repository": item["repository"],
                "total": item["total"],
                "file": f"{item['repository']}.json.gz",
            }
            for item in current
        ],
        "history": [history[date] for date in sorted(history)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    for source, destination in pending:
        source.replace(destination)
    temporary.replace(path)
    legacy_path.unlink(missing_ok=True)
    compressed_path.unlink(missing_ok=True)
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
