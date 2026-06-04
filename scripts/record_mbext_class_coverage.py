"""Record the list of classes defined in ``mbext`` and ``onnxruntime-genai``.

This script lists every Python class defined inside

* https://github.com/xadupre/mbext/tree/main/modelbuilder
* https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models

and writes the union of the class names to
``cache_data/mbext/class_coverage.json``. For each class the JSON records
whether the class is present in either project. The dashboard at
``dashboard/mbext/class-coverage.html`` consumes this file to render the
comparison table requested in the tracking issue.

The script only uses the GitHub REST and ``raw.githubusercontent.com`` URLs
so that it can run from a stock GitHub Actions runner without cloning the
upstream repositories.

Usage::

    python scripts/record_mbext_class_coverage.py [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

GITHUB_CONTENTS_URL = (
    "https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
)
GITHUB_COMMIT_URL = (
    "https://api.github.com/repos/{owner}/{repo}/commits"
    "?path={path}&sha={ref}&per_page=1"
)

# Projects whose class lists are compared on the dashboard.
PROJECTS: tuple[dict[str, str], ...] = (
    {
        "key": "mbext",
        "owner": "xadupre",
        "repo": "mbext",
        "path": "modelbuilder",
        "ref": "main",
    },
    {
        "key": "ort_genai",
        "owner": "microsoft",
        "repo": "onnxruntime-genai",
        "path": "src/python/py/models",
        "ref": "main",
    },
)


def _log(message: str) -> None:
    """Print ``message`` prefixed with a UTC timestamp."""
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


def _format_iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    else:
        value = value.astimezone(dt.timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request(url: str) -> bytes:
    """Fetch ``url`` and return the response body.

    Adds an ``Authorization`` header when ``GITHUB_TOKEN`` is set in the
    environment so that the GitHub API rate limit is the higher authenticated
    one when the script runs from a CI workflow.
    """
    headers = {
        "User-Agent": "xadupre.github.io-record-mbext-class-coverage",
        "Accept": "application/vnd.github+json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - github.com
        return resp.read()


def list_python_files(
    owner: str, repo: str, path: str, ref: str
) -> list[dict[str, str]]:
    """Return ``{"path", "download_url"}`` entries for every ``.py`` file.

    Recurses into subdirectories. ``__init__.py`` is included like any other
    Python file so that classes re-exported from a package's top-level file
    are not silently dropped.
    """
    url = GITHUB_CONTENTS_URL.format(owner=owner, repo=repo, path=path, ref=ref)
    entries = json.loads(_request(url).decode("utf-8"))
    if not isinstance(entries, list):
        raise RuntimeError(
            f"Unexpected GitHub contents payload for {owner}/{repo}/{path}: "
            f"{entries!r}"
        )
    files: list[dict[str, str]] = []
    for entry in entries:
        kind = entry.get("type")
        if kind == "file" and entry.get("name", "").endswith(".py"):
            files.append(
                {
                    "path": entry["path"],
                    "download_url": entry["download_url"],
                }
            )
        elif kind == "dir":
            files.extend(list_python_files(owner, repo, entry["path"], ref))
    return files


def extract_class_names(source: str) -> list[str]:
    """Return the names of every top-level or nested ``class`` defined in ``source``.

    Only classes that actually appear in the AST are returned, which means
    classes hidden behind ``if TYPE_CHECKING`` or similar guarded blocks are
    still detected.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        _log(f"WARNING: failed to parse source ({exc}); skipping.")
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
    return names


def collect_classes(project: dict[str, str]) -> dict[str, list[str]]:
    """Return a mapping ``class name -> list of files`` for ``project``."""
    files = list_python_files(
        project["owner"], project["repo"], project["path"], project["ref"]
    )
    classes: dict[str, list[str]] = {}
    for file_entry in files:
        _log(f"Fetching {file_entry['path']}...")
        source = _request(file_entry["download_url"]).decode("utf-8")
        for name in extract_class_names(source):
            classes.setdefault(name, []).append(file_entry["path"])
    return classes


def fetch_latest_commit(owner: str, repo: str, path: str, ref: str) -> str | None:
    """Return the SHA of the most recent commit touching ``path``."""
    url = GITHUB_COMMIT_URL.format(owner=owner, repo=repo, path=path, ref=ref)
    try:
        payload = json.loads(_request(url).decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        _log(f"WARNING: cannot fetch latest commit for {owner}/{repo}: {exc}")
        return None
    if isinstance(payload, list) and payload:
        return payload[0].get("sha")
    return None


def build_payload(
    projects: tuple[dict[str, str], ...] = PROJECTS,
    now: dt.datetime | None = None,
) -> dict[str, object]:
    """Collect the class lists for every project and merge them.

    The returned dictionary is the structure persisted as JSON.
    """
    per_project_classes: dict[str, dict[str, list[str]]] = {}
    per_project_meta: dict[str, dict[str, str | None]] = {}
    for project in projects:
        _log(
            f"Collecting classes for {project['owner']}/{project['repo']}/"
            f"{project['path']}..."
        )
        per_project_classes[project["key"]] = collect_classes(project)
        per_project_meta[project["key"]] = {
            "owner": project["owner"],
            "repo": project["repo"],
            "path": project["path"],
            "ref": project["ref"],
            "commit": fetch_latest_commit(
                project["owner"],
                project["repo"],
                project["path"],
                project["ref"],
            ),
        }

    all_names: set[str] = set()
    for classes in per_project_classes.values():
        all_names.update(classes)

    rows = []
    for name in sorted(all_names):
        row: dict[str, object] = {"name": name}
        for key, classes in per_project_classes.items():
            row[f"in_{key}"] = name in classes
            files = classes.get(name)
            if files:
                # Sort and de-duplicate so that the cached order is stable.
                row[f"{key}_files"] = sorted(set(files))
        rows.append(row)

    return {
        "date": _format_iso(now or dt.datetime.now(tz=dt.timezone.utc)),
        "projects": per_project_meta,
        "totals": {key: len(classes) for key, classes in per_project_classes.items()},
        "classes": rows,
    }


def write_payload(json_path: str, payload: dict[str, object]) -> None:
    """Write ``payload`` to ``json_path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the JSON cache (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_payload()
    except (urllib.error.URLError, RuntimeError, ValueError) as exc:
        _log(f"ERROR: failed to record mbext class coverage: {exc}")
        return 1
    json_path = os.path.join(args.cache_dir, "mbext", "class_coverage.json")
    write_payload(json_path, payload)
    _log(
        f"Wrote {len(payload['classes'])} class entries to {json_path} "
        f"(totals={payload['totals']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
