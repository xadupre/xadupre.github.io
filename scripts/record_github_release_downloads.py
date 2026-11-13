"""Record daily GitHub release download counts for ONNX and LLM repositories.

Discover public xadupre onnx-light, yet-another and devdash-agents projects,
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
    return name in ("onnx-light", "devdash-agents") or name.startswith(
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


def project_state(item: dict) -> dict:
    """Separate release metadata and cumulative asset counters for comparison."""
    return {
        "releases": {
            str(release["id"]): {
                key: release[key] for key in ("tag", "published_at", "prerelease")
            }
            for release in item["releases"]
        },
        "assets": {
            str(asset["id"]): {
                "release_id": str(release["id"]),
                "download_count": asset["download_count"],
            }
            for release in item["releases"]
            for asset in release["assets"]
        },
    }


def apply_daily_delta(state: dict, record: dict) -> None:
    """Replay metadata changes, signed counter changes and explicit removals."""
    for asset_id in record["removed_assets"]:
        del state["assets"][asset_id]
    for release_id in record["removed_releases"]:
        del state["releases"][release_id]
    state["releases"].update(record["releases"])
    for asset_id, change in record["assets"].items():
        previous = state["assets"].get(asset_id, {"download_count": 0})
        delta = change.get("downloads", 0)
        count = previous["download_count"] + delta
        if type(delta) is not int or count < 0:
            raise ValueError(f"Invalid asset download delta: {asset_id}: {change!r}")
        state["assets"][asset_id] = {
            **previous,
            **{key: value for key, value in change.items() if key != "downloads"},
            "download_count": count,
        }


def daily_delta(
    repository: str, day: str, previous: dict, current: dict, observed: bool
) -> dict:
    """Record only changed metadata/counters; the first observation is a baseline."""
    assets = {}
    for asset_id, asset in sorted(current["assets"].items()):
        old = previous["assets"].get(asset_id, {})
        if asset != old:
            change = {
                key: value
                for key, value in asset.items()
                if key != "download_count" and old.get(key) != value
            }
            delta = asset["download_count"] - old.get("download_count", 0)
            if delta:
                change["downloads"] = delta
            assets[asset_id] = change
    return {
        "date": day,
        "repository": repository,
        "observed": observed,
        "releases": {
            key: value
            for key, value in sorted(current["releases"].items())
            if previous["releases"].get(key) != value
        },
        "assets": assets,
        "removed_assets": sorted(previous["assets"].keys() - current["assets"].keys()),
        "removed_releases": sorted(
            previous["releases"].keys() - current["releases"].keys()
        ),
    }


def record_downloads(
    cache_dir: str,
    token: str = "",
    repositories: list[str] | None = None,
    now: dt.datetime | None = None,
) -> dict:
    """Publish daily delta journals, without a growing current-metadata file."""
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
        for repository, item in projects.items():
            entries = []
            for entry in legacy["history"]:
                row = {"date": entry["date"], "totals": {}}
                if repository in entry.get("totals", {}):
                    row["totals"][repository] = entry["totals"][repository]
                if repository in entry.get("assets", {}):
                    row["assets"] = {repository: entry["assets"][repository]}
                if row["totals"] or "assets" in row:
                    entries.append(row)
            old_item = next(
                (
                    value
                    for value in legacy.get("repositories", [])
                    if value["repository"] == repository
                ),
                item,
            )
            migrated[repository] = {"history": entries, "repositories": [old_item]}
    elif previous.get("format") != "daily-delta-v1":
        # Include archived projects which no longer appear in the summary index.
        for source in sorted(directory.glob("*/*.json.gz")):
            with gzip.open(source, "rt", encoding="utf-8") as stream:
                project = json.load(stream)
            item = project["repositories"][0]
            repository = item["repository"]
            projects.setdefault(repository, item)
            migrated[repository] = project
            obsolete.append(source)
        for source in sorted(directory.glob("*/*/latest.json")):
            project = json.loads(source.read_text(encoding="utf-8"))
            item = project["repositories"][0]
            repository = item["repository"]
            projects.setdefault(repository, item)
            project["history"] = [
                json.loads((source.parent / f"{date}.json").read_text(encoding="utf-8"))
                for date in project["dates"]
            ]
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

    dates_by_project = {}
    previous_projects = {
        item["repository"]: item for item in previous.get("repositories", [])
    }
    for repository, item in sorted(projects.items()):
        # GitHub owner/repository names cannot contain path separators or dot segments.
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository) or (
            repository.split("/")[1] in (".", "..")
        ):
            raise ValueError(f"Invalid repository name: {repository!r}")
        project_path = directory / repository
        names_path = project_path / "assets.json"
        previous_names = (
            json.loads(names_path.read_text(encoding="utf-8"))
            if names_path.exists()
            else {}
        )
        names = previous_names.copy()
        if repository in migrated:
            for release in migrated[repository]["repositories"][0]["releases"]:
                names.update(
                    {str(asset["id"]): asset["name"] for asset in release["assets"]}
                )
        for release in item["releases"]:
            names.update(
                {str(asset["id"]): asset["name"] for asset in release["assets"]}
            )
        if names != previous_names or not names_path.exists():
            stage_json(names_path, names)
        state = {"releases": {}, "assets": {}}
        records = {}
        if repository in migrated:
            metadata = project_state(migrated[repository]["repositories"][0])
            for entry in sorted(
                migrated[repository]["history"], key=lambda row: row["date"]
            ):
                date = entry["date"]
                if repository in selected and date == day:
                    continue
                observed = repository in entry.get("assets", {})
                target = {"releases": {}, "assets": {}}
                if observed:
                    target["assets"] = {
                        asset_id: {
                            **metadata["assets"].get(asset_id, {}),
                            "download_count": count,
                        }
                        for asset_id, count in entry["assets"][repository].items()
                    }
                else:
                    target["assets"] = state["assets"].copy()
                referenced_releases = {
                    asset.get("release_id") for asset in target["assets"].values()
                }
                target["releases"] = {
                    key: value
                    for key, value in metadata["releases"].items()
                    if value["published_at"][:10] <= date or key in referenced_releases
                }
                record = daily_delta(repository, date, state, target, observed)
                if repository in entry.get("totals", {}):
                    record["total"] = entry["totals"][repository]
                records[date] = record
                state = target
        else:
            dates = previous_projects.get(repository, {}).get("dates")
            if dates is None:
                dates = sorted(
                    path.stem for path in project_path.glob("????-??-??.json")
                )
            for date in dates:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                    raise ValueError(f"Invalid observation date: {date!r}")
                if date > day:
                    raise ValueError(f"Cannot insert an observation before {date}")
                if date == day:
                    continue
                record = json.loads(
                    (project_path / f"{date}.json").read_text(encoding="utf-8")
                )
                if record["date"] != date or record["repository"] != repository:
                    raise ValueError(
                        f"Invalid daily observation: {project_path}/{date}"
                    )
                apply_daily_delta(state, record)
            dates_by_project[repository] = dates
        if repository in selected:
            records[day] = daily_delta(
                repository, day, state, project_state(item), observed=True
            )
            records[day]["total"] = item["total"]
        dates_by_project[repository] = sorted(
            set(dates_by_project.get(repository, [])) | records.keys()
        )
        for date, entry in sorted(records.items()):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                raise ValueError(f"Invalid observation date: {date!r}")
            stage_json(project_path / f"{date}.json", entry)
    result = {
        "format": "daily-delta-v1",
        "updated_at": updated_at,
        "repositories": [
            {
                "repository": item["repository"],
                "total": item["total"],
                "directory": item["repository"],
                "dates": dates_by_project[item["repository"]],
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
