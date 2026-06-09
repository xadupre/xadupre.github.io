"""Record wheel sizes published by the ``build_release.yml`` workflow.

This script queries the GitHub Actions REST API for every completed and
successful run of a workflow (``build_release.yml`` of
``xadupre/onnx-light`` by default), downloads each artifact attached to the
run, and records one CSV row per ``.whl`` file found inside the artifact
zip archive. The resulting CSV is consumed by
``dashboard/onnx-light/package-size.html`` to plot the evolution of the
binary size of each wheel over time, side by side with the existing
shared-library size chart.

The CSV columns are::

    date,commit,run_id,size,name

where:

* ``date`` is the ISO 8601 UTC timestamp of the workflow run creation,
* ``commit`` is the head SHA of the run,
* ``run_id`` is the GitHub Actions workflow run id (used to skip runs that
  have already been processed),
* ``size`` is the size in bytes of the ``.whl`` file as stored inside the
  artifact zip archive,
* ``name`` is the wheel file name (e.g.
  ``onnx_light-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl``).

Usage::

    python scripts/record_wheel_sizes.py [--cache-dir DIR] [--repo owner/name]
        [--workflow build_release.yml] [--months N] [--max-runs N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Iterable, Iterator

DEFAULT_REPO = "xadupre/onnx-light"
DEFAULT_WORKFLOW = "build_release.yml"

CSV_FIELDS = ("date", "commit", "run_id", "size", "name")

GITHUB_API = "https://api.github.com"


def _log(message: str) -> None:
    """Print ``message`` prefixed with a UTC timestamp."""
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


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


def read_existing(csv_path: str) -> set[str]:
    """Return the set of run ids already recorded in ``csv_path``."""
    seen: set[str] = set()
    if not os.path.exists(csv_path):
        return seen
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            run_id = row.get("run_id")
            if run_id:
                seen.add(run_id)
    return seen


def append_rows(csv_path: str, rows: Iterable[dict]) -> int:
    """Append ``rows`` to ``csv_path``, creating the file with a header if needed."""
    rows = list(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    if not rows:
        if not file_exists:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
        return 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _log(f"saved {len(rows)} row(s) to {csv_path}")
    return len(rows)


def _request(url: str, token: str | None) -> tuple[dict, dict]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xadupre.github.io-record-wheel-sizes",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - api.github.com
        payload = json.loads(resp.read().decode("utf-8"))
        return payload, dict(resp.headers)


def _download(url: str, token: str | None) -> bytes:
    """Download a binary resource from the GitHub API.

    Follows redirects (the artifact zip endpoint redirects to a temporary
    signed URL on GitHub's storage backend).
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "xadupre.github.io-record-wheel-sizes",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - api.github.com
        return resp.read()


def iter_workflow_runs(
    repo: str,
    workflow: str,
    since: dt.datetime,
    token: str | None,
    max_runs: int | None = None,
) -> Iterator[dict]:
    """Yield workflow runs of ``workflow`` in ``repo`` created on or after ``since``.

    Runs are yielded from the most recent to the oldest. ``workflow`` may be
    a numeric workflow id or a workflow file name (e.g. ``build_release.yml``).
    The iteration stops once a run older than ``since`` is seen or once
    ``max_runs`` runs have been yielded.
    """
    page = 1
    per_page = 100
    yielded = 0
    while True:
        params = {
            "per_page": str(per_page),
            "page": str(page),
            "status": "completed",
        }
        url = (
            f"{GITHUB_API}/repos/{repo}/actions/workflows/"
            f"{urllib.parse.quote(workflow, safe='')}/runs?"
            + urllib.parse.urlencode(params)
        )
        payload, _ = _request(url, token)
        page_runs = payload.get("workflow_runs", [])
        if not page_runs:
            return
        for run in page_runs:
            created = run.get("created_at")
            if created:
                try:
                    if _parse_iso(created) < since:
                        return
                except ValueError:
                    pass
            yield run
            yielded += 1
            if max_runs is not None and yielded >= max_runs:
                return
        if len(page_runs) < per_page:
            return
        page += 1


def list_run_artifacts(
    repo: str, run_id: str, token: str | None
) -> list[dict]:
    """Return the list of artifacts attached to ``run_id`` in ``repo``."""
    artifacts: list[dict] = []
    page = 1
    per_page = 100
    while True:
        params = {"per_page": str(per_page), "page": str(page)}
        url = (
            f"{GITHUB_API}/repos/{repo}/actions/runs/{run_id}/artifacts?"
            + urllib.parse.urlencode(params)
        )
        payload, _ = _request(url, token)
        page_artifacts = payload.get("artifacts", [])
        if not page_artifacts:
            break
        artifacts.extend(page_artifacts)
        if len(page_artifacts) < per_page:
            break
        page += 1
    return artifacts


def extract_wheel_sizes(zip_bytes: bytes) -> list[tuple[str, int]]:
    """Return ``(wheel_name, size_in_bytes)`` pairs for every ``.whl`` in the zip.

    Wheel files are themselves zip archives and are stored uncompressed
    inside the artifact zip, so ``file_size`` from the zip metadata is
    representative of the wheel's on-disk size.
    """
    results: list[tuple[str, int]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if not name.lower().endswith(".whl"):
                continue
            results.append((name, int(info.file_size)))
    return results


def process_run(
    run: dict, repo: str, token: str | None
) -> list[dict]:
    """Return CSV rows for every wheel artifact attached to ``run``.

    Returns an empty list when the run is not completed, did not succeed,
    or has no wheel artifacts. Artifacts whose download fails are skipped
    with a warning so that one broken artifact does not abort the rest of
    the run.
    """
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        return []
    run_id = str(run.get("id", ""))
    if not run_id:
        return []
    created = run.get("created_at") or ""
    commit = run.get("head_sha") or ""
    artifacts = list_run_artifacts(repo, run_id, token)
    rows: list[dict] = []
    for artifact in artifacts:
        if artifact.get("expired"):
            continue
        url = artifact.get("archive_download_url")
        if not url:
            continue
        try:
            data = _download(url, token)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            print(
                f"[{repo}] run {run_id}: failed to download artifact "
                f"{artifact.get('name')!r}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        try:
            wheels = extract_wheel_sizes(data)
        except zipfile.BadZipFile as exc:
            print(
                f"[{repo}] run {run_id}: artifact {artifact.get('name')!r} "
                f"is not a valid zip: {exc}",
                file=sys.stderr,
            )
            continue
        for name, size in wheels:
            rows.append(
                {
                    "date": created,
                    "commit": commit,
                    "run_id": run_id,
                    "size": str(size),
                    "name": name,
                }
            )
    return rows


def process_repo(
    repo: str,
    workflow: str,
    cache_dir: str,
    months: int,
    token: str | None,
    max_runs: int | None = None,
) -> int:
    """Fetch new wheel sizes for ``repo`` and append them to the cache file.

    Returns the number of new CSV rows appended.
    """
    repo_name = repo.split("/", 1)[-1]
    csv_path = os.path.join(cache_dir, repo_name, "wheel_sizes.csv")
    seen = read_existing(csv_path)
    since = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=months * 30)
    _log(
        f"[{repo}] cache file: {csv_path} ({len(seen)} run(s) already recorded)"
    )
    _log(f"[{repo}] fetching {workflow!r} runs since {_format_iso(since)}")
    new_rows: list[dict] = []
    processed = 0
    try:
        for run in iter_workflow_runs(repo, workflow, since, token, max_runs):
            processed += 1
            run_id = str(run.get("id", ""))
            if not run_id or run_id in seen:
                continue
            _log(
                f"[{repo}] processing run {run_id} "
                f"(commit={(run.get('head_sha') or '')[:7]}, "
                f"created={run.get('created_at')})"
            )
            try:
                rows = process_run(run, repo, token)
            except urllib.error.HTTPError as exc:
                print(
                    f"[{repo}] HTTP error while processing run {run_id}: "
                    f"{exc.code} {exc.reason}",
                    file=sys.stderr,
                )
                continue
            new_rows.extend(rows)
            seen.add(run_id)
            _log(
                f"[{repo}]   recorded {len(rows)} wheel(s) for run {run_id}"
            )
    finally:
        added = append_rows(csv_path, new_rows)
    _log(
        f"[{repo}] processed {processed} run(s) from GitHub; "
        f"appended {added} new wheel row(s) to {csv_path}"
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
        "--repo",
        default=DEFAULT_REPO,
        help=f"Repository to query (owner/name, default: {DEFAULT_REPO}).",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help=(
            "Workflow file name or numeric id to query "
            f"(default: {DEFAULT_WORKFLOW})."
        ),
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months to look back when collecting runs (default: 6).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of runs inspected per invocation.",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    _log("record_wheel_sizes.py starting")
    _log(f"  cache directory : {args.cache_dir}")
    _log(f"  repository      : {args.repo}")
    _log(f"  workflow        : {args.workflow}")
    _log(f"  months          : {args.months}")
    if args.max_runs is not None:
        _log(f"  max runs        : {args.max_runs}")
    if not token:
        _log("  authentication  : anonymous (no GITHUB_TOKEN/GH_TOKEN set)")
        print("warning: no GITHUB_TOKEN/GH_TOKEN set; using anonymous requests.")
    else:
        _log("  authentication  : using GITHUB_TOKEN/GH_TOKEN")

    try:
        added = process_repo(
            args.repo,
            args.workflow,
            args.cache_dir,
            args.months,
            token,
            args.max_runs,
        )
    except urllib.error.HTTPError as exc:
        print(
            f"[{args.repo}] HTTP error {exc.code}: {exc.reason}",
            file=sys.stderr,
        )
        return 1
    _log(f"Done. {added} new wheel row(s) recorded.")
    print(f"Done. {added} new wheel row(s) recorded.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
