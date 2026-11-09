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
    """Publish plain JSON metadata and one observation file per project and UTC day."""
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
    migrated = {}
    obsolete = [legacy_path, compressed_path]
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
        for repository in projects:
            entries = []
            for entry in legacy["history"]:
                row = {"date": entry["date"], "totals": {}}
                if repository in entry.get("totals", {}):
                    row["totals"][repository] = entry["totals"][repository]
                if repository in entry.get("assets", {}):
                    row["assets"] = {repository: entry["assets"][repository]}
                if row["totals"] or "assets" in row:
                    entries.append(row)
            migrated[repository] = {
                "updated_at": legacy.get("updated_at", updated_at),
                "history": entries,
            }
    elif previous.get("format") != "daily-json-v1":
        # Include archived projects which no longer appear in the summary index.
        for source in sorted(directory.glob("*/*.json.gz")):
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                project = json.load(stream)
            item = project["repositories"][0]
            repository = item["repository"]
            projects.setdefault(repository, item)
            migrated[repository] = project
            obsolete.append(source)
        for item in previous.get("repositories", []):
            if item["repository"] not in migrated:
                raise FileNotFoundError(
                    f"Missing release download history: {item['repository']}"
                )

    pending = []

    def stage_json(destination: Path, data: dict) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        pending.append((temporary, destination))

    for repository, item in sorted(projects.items()):
        # GitHub owner/repository names cannot contain path separators or dot segments.
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository) or (
            repository.split("/")[1] in (".", "..")
        ):
            raise ValueError(f"Invalid repository name: {repository!r}")
        project_path = directory / repository / "latest.json"
        if repository in migrated:
            project_history = {
                entry["date"]: entry for entry in migrated[repository]["history"]
            }
            dates = set(project_history)
        elif project_path.exists():
            dates = set(json.loads(project_path.read_text(encoding="utf-8"))["dates"])
            project_history = {}
        else:
            if any(
                row["repository"] == repository
                for row in previous.get("repositories", [])
            ):
                raise FileNotFoundError(
                    f"Missing release download history: {project_path}"
                )
            dates = set()
            project_history = {}
        if repository in selected:
            dates.add(day)
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
        for date, entry in sorted(project_history.items()):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError(f"Invalid observation date: {date!r}")
            stage_json(project_path.parent / f"{date}.json", entry)
        stage_json(
            project_path,
            {
                "updated_at": (
                    migrated[repository]["updated_at"]
                    if repository in migrated and repository not in selected
                    else updated_at
                ),
                "repositories": [item],
                "dates": sorted(dates),
            },
        )
    result = {
        "format": "daily-json-v1",
        "updated_at": updated_at,
        "repositories": [
            {
                "repository": item["repository"],
                "total": item["total"],
                "file": f"{item['repository']}/latest.json",
            }
            for item in current
        ],
        "history": [history[date] for date in sorted(history)],
    }
    stage_json(path, result)
    for source, destination in pending:
        source.replace(destination)
    for source in obsolete:
        source.unlink(missing_ok=True)
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
