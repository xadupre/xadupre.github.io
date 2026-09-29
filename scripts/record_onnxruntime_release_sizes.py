"""Record ONNX Runtime binary download sizes from GitHub release metadata.

Stable releases are backfilled by publication date. Sizes are the compressed
asset sizes reported by GitHub, not installed library sizes. No binaries are
downloaded. Run with ``--cache-dir DIR`` to change the default ``cache_data``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

RELEASES_URL = "https://api.github.com/repos/microsoft/onnxruntime/releases"
BINARY_SUFFIXES = (".zip", ".tgz", ".tar.gz", ".whl", ".nupkg", ".aar", ".jar")


def fetch_pages(url: str, token: str = "") -> list[dict]:
    """Fetch every page of a GitHub release or release-asset collection."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xadupre.github.io-record-onnxruntime-release-sizes",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    result = []
    page = 1
    while True:
        request = urllib.request.Request(
            f"{url}?per_page=100&page={page}", headers=headers
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            items = json.load(response)
        if not isinstance(items, list):
            raise ValueError("Expected a GitHub API collection")
        result.extend(items)
        if len(items) < 100:
            return result
        page += 1


def build_rows(release: dict, assets: list[dict]) -> list[dict]:
    """Keep uploaded binary packages, retaining platform/provider variants."""
    if release.get("draft") or release.get("prerelease"):
        return []
    tag = release["tag_name"]
    version = tag.removeprefix("v")
    rows = []
    for asset in assets:
        name = asset["name"]
        if asset["state"] != "uploaded" or not name.endswith(BINARY_SUFFIXES):
            continue
        rows.append({
            "date": release["published_at"],
            "version": tag,
            "name": name,
            "package": name.replace(version, "{version}") if version else name,
            "size": int(asset["size"]),
        })
    return rows


def record_sizes(cache_dir: str, token: str = "") -> list[dict]:
    """Refresh the release history only after all API requests succeed."""
    rows = {}
    for release in fetch_pages(RELEASES_URL, token):
        if release.get("draft") or release.get("prerelease"):
            continue
        assets = fetch_pages(
            f"{RELEASES_URL}/{int(release['id'])}/assets", token
        )
        for row in build_rows(release, assets):
            rows[(row["version"], row["name"])] = row
    if not rows:
        raise RuntimeError("No ONNX Runtime release binaries found")
    history = sorted(
        rows.values(), key=lambda row: (row["date"], row["version"], row["name"])
    )
    path = Path(cache_dir) / "onnxruntime" / "release_sizes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {len(history)} release asset sizes in {path}", flush=True)
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache_data")
    args = parser.parse_args(argv)
    record_sizes(args.cache_dir, os.environ.get("GITHUB_TOKEN", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
