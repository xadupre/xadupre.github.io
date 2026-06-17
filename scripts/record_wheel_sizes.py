"""Record wheel and shared-library sizes from the ``build_release.yml`` workflow.

This script queries the GitHub Actions REST API for every completed and
successful run of a workflow (``build_release.yml`` of
``xadupre/onnx-light`` by default), downloads each artifact attached to the
run **once**, and records, from that single download:

* one row per ``.whl`` file found inside the artifact, into
  ``wheel_sizes.csv`` (the "Wheel size over time" series), and
* one row per shared library (``.so`` / versioned ``.so.N`` / ``.pyd`` /
  ``.dylib``) found inside those wheels (or directly in the artifact), into
  ``so_sizes.csv`` (the "Binary size over time" series).

Both series feed ``dashboard/onnx-light/package-size.html``. Recording both
from the same artifact download means the shared-library chart is backfilled
over the same ``--months`` look-back window as the wheel chart, instead of
only showing data accumulated since the inline workflow snapshot step first
ran.

``wheel_sizes.csv`` columns are::

    date,commit,run_id,size,name

and ``so_sizes.csv`` columns (matching the inline workflow snapshot step) are::

    date,commit,size,name

where:

* ``date`` is the ISO 8601 UTC timestamp of the workflow run creation,
* ``commit`` is the head SHA of the run,
* ``run_id`` is the GitHub Actions workflow run id (used to skip wheel rows
  for runs that have already been processed),
* ``size`` is the size in bytes of the ``.whl`` file (resp. shared library)
  as stored inside the artifact zip archive,
* ``name`` is the wheel (resp. shared library) file name.

Wheel rows are de-duplicated by ``run_id`` and shared-library rows by
``commit`` (so that commits already snapshotted by the inline workflow step
are not recorded twice).

Usage::

    python scripts/record_wheel_sizes.py [--cache-dir DIR] [--repo owner/name]
        [--workflow build_release.yml] [--months N] [--max-runs N]
        [--wheel-csv-name wheel_sizes.csv] [--skip-so]
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

# ``wheel_sizes.csv`` schema (one row per ``.whl`` file).
CSV_FIELDS = ("date", "commit", "run_id", "size", "name")

# ``so_sizes.csv`` schema (one row per shared library). This matches the
# inline "Record onnx_py shared library sizes" workflow step so that rows from
# either source can live in the same file.
SO_CSV_FIELDS = ("date", "commit", "size", "name")

# Extensions of the compiled shared libraries tracked by the binary-size chart.
# ``.so.<version>`` files (e.g. ``libfoo.so.1``) are matched separately below.
SHARED_LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib")

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


def _is_shared_library(name: str) -> bool:
    """Return ``True`` when ``name`` looks like a compiled shared library."""
    lowered = name.lower()
    if lowered.endswith(SHARED_LIBRARY_SUFFIXES):
        return True
    # Versioned ELF shared objects such as ``liblib_onnx_lib.so.1`` or
    # ``libfoo.so.1.2``: there is a ``.so.`` segment followed by digits.
    marker = ".so."
    idx = lowered.rfind(marker)
    if idx != -1:
        tail = lowered[idx + len(marker):]
        first = tail.split(".", 1)[0]
        if first.isdigit():
            return True
    return False


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


def read_existing_commits(csv_path: str) -> set[str]:
    """Return the set of commit SHAs already recorded in ``csv_path``."""
    seen: set[str] = set()
    if not os.path.exists(csv_path):
        return seen
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            commit = row.get("commit")
            if commit:
                seen.add(commit)
    return seen


def append_rows(
    csv_path: str, rows: Iterable[dict], fields: tuple[str, ...] = CSV_FIELDS
) -> int:
    """Append ``rows`` to ``csv_path``, creating the file with a header if needed."""
    rows = list(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    if not rows:
        if not file_exists:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
        return 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
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


def _extract_shared_libraries_from_wheel(wheel_bytes: bytes) -> list[tuple[str, int]]:
    """Return ``(name, size)`` for every shared library inside a wheel."""
    results: list[tuple[str, int]] = []
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        for info in wheel.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if name and _is_shared_library(name):
                results.append((name, int(info.file_size)))
    return results


def extract_shared_library_sizes(zip_bytes: bytes) -> list[tuple[str, int]]:
    """Return ``(name, size)`` pairs for every shared library in an artifact.

    Artifacts produced by ``build_release.yml`` store one or more ``.whl``
    files (which are themselves zip archives) without further compression.
    Shared libraries are looked up inside each wheel; an artifact that uploads
    shared libraries directly (not wrapped in a wheel) is also supported.

    When the same library name appears more than once (e.g. across several
    wheels in the same artifact) the largest reported size is kept so that the
    series is deterministic regardless of archive ordering.
    """
    sizes: dict[str, int] = {}

    def _record(name: str, size: int) -> None:
        previous = sizes.get(name)
        if previous is None or size > previous:
            sizes[name] = size

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = os.path.basename(info.filename)
            if name.lower().endswith(".whl"):
                try:
                    inner = _extract_shared_libraries_from_wheel(zf.read(info))
                except zipfile.BadZipFile:
                    # A wheel that cannot be opened as a zip carries no
                    # readable shared libraries; skip it without aborting the
                    # rest of the artifact (the wheel-size series, which only
                    # reads outer-zip metadata, is unaffected).
                    continue
                for so_name, so_size in inner:
                    _record(so_name, so_size)
            elif name and _is_shared_library(name):
                _record(name, int(info.file_size))
    return sorted(sizes.items())


def process_run(
    run: dict, repo: str, token: str | None
) -> tuple[list[dict], list[dict]]:
    """Return ``(wheel_rows, so_rows)`` for every artifact attached to ``run``.

    Both lists are empty when the run is not completed or has no matching
    artifacts. Runs whose conclusion is not ``success`` are still inspected
    because the upstream workflow may have uploaded artifacts before a later
    step failed; skipping them would silently hide otherwise-valid size
    measurements from the dashboard. Each artifact is downloaded once and both
    the wheel sizes and the shared-library sizes contained in it are recorded.
    Artifacts whose download fails are skipped with a warning so that one
    broken artifact does not abort the rest of the run.
    """
    if run.get("status") != "completed":
        return [], []
    run_id = str(run.get("id", ""))
    if not run_id:
        return [], []
    created = run.get("created_at") or ""
    commit = run.get("head_sha") or ""
    artifacts = list_run_artifacts(repo, run_id, token)
    wheel_rows: list[dict] = []
    so_sizes: dict[str, int] = {}
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
            libraries = extract_shared_library_sizes(data)
        except zipfile.BadZipFile as exc:
            print(
                f"[{repo}] run {run_id}: artifact {artifact.get('name')!r} "
                f"is not a valid zip: {exc}",
                file=sys.stderr,
            )
            continue
        for name, size in wheels:
            wheel_rows.append(
                {
                    "date": created,
                    "commit": commit,
                    "run_id": run_id,
                    "size": str(size),
                    "name": name,
                }
            )
        for name, size in libraries:
            previous = so_sizes.get(name)
            if previous is None or size > previous:
                so_sizes[name] = size
    so_rows = [
        {
            "date": created,
            "commit": commit,
            "size": str(size),
            "name": name,
        }
        for name, size in sorted(so_sizes.items())
    ]
    return wheel_rows, so_rows


def process_repo(
    repo: str,
    workflow: str,
    cache_dir: str,
    months: int,
    token: str | None,
    max_runs: int | None = None,
    wheel_csv_name: str = "wheel_sizes.csv",
    skip_so: bool = False,
) -> tuple[int, int]:
    """Fetch new wheel and shared-library sizes for ``repo`` and append them.

    Returns ``(wheel_rows_added, so_rows_added)``. Each artifact is downloaded
    once and feeds both ``wheel_csv_name`` (deduplicated by ``run_id``) and
    ``so_sizes.csv`` (deduplicated by ``commit``).

    ``wheel_csv_name`` selects which file the wheel rows are written to (it
    defaults to ``wheel_sizes.csv``); pointing a separate workflow such as
    ``build_reduced_wheel.yml`` at ``wheel_sizes_reduced.csv`` keeps the
    reduced-wheel series from being merged into the full-wheel one even though
    both wheels share the same file name. When ``skip_so`` is true the
    shared-library series is not recorded at all, so a reduced build whose
    compiled extensions differ from the released ones does not pollute
    ``so_sizes.csv``.
    """
    repo_name = repo.split("/", 1)[-1]
    wheel_csv = os.path.join(cache_dir, repo_name, wheel_csv_name)
    so_csv = os.path.join(cache_dir, repo_name, "so_sizes.csv")
    seen_runs = read_existing(wheel_csv)
    seen_commits = set() if skip_so else read_existing_commits(so_csv)
    since = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=months * 30)
    _log(
        f"[{repo}] wheel cache: {wheel_csv} ({len(seen_runs)} run(s) recorded); "
        f"so cache: {so_csv} ({len(seen_commits)} commit(s) recorded"
        f"{', skipped' if skip_so else ''})"
    )
    _log(f"[{repo}] fetching {workflow!r} runs since {_format_iso(since)}")
    new_wheel_rows: list[dict] = []
    new_so_rows: list[dict] = []
    processed = 0
    try:
        for run in iter_workflow_runs(repo, workflow, since, token, max_runs):
            processed += 1
            run_id = str(run.get("id", ""))
            commit = run.get("head_sha") or ""
            need_wheel = bool(run_id) and run_id not in seen_runs
            need_so = (
                not skip_so and bool(commit) and commit not in seen_commits
            )
            if not need_wheel and not need_so:
                continue
            _log(
                f"[{repo}] processing run {run_id} "
                f"(commit={commit[:7]}, created={run.get('created_at')})"
            )
            try:
                wheel_rows, so_rows = process_run(run, repo, token)
            except urllib.error.HTTPError as exc:
                print(
                    f"[{repo}] HTTP error while processing run {run_id}: "
                    f"{exc.code} {exc.reason}",
                    file=sys.stderr,
                )
                continue
            if need_wheel:
                new_wheel_rows.extend(wheel_rows)
                seen_runs.add(run_id)
                _log(f"[{repo}]   recorded {len(wheel_rows)} wheel(s) for run {run_id}")
            if need_so and so_rows:
                new_so_rows.extend(so_rows)
                seen_commits.add(commit)
                _log(
                    f"[{repo}]   recorded {len(so_rows)} shared librar(ies) "
                    f"for commit {commit[:7]}"
                )
    finally:
        added_wheels = append_rows(wheel_csv, new_wheel_rows)
        added_so = 0 if skip_so else append_rows(so_csv, new_so_rows, SO_CSV_FIELDS)
    _log(
        f"[{repo}] processed {processed} run(s) from GitHub; appended "
        f"{added_wheels} wheel row(s) to {wheel_csv} and "
        f"{added_so} shared-library row(s) to {so_csv}"
    )
    return added_wheels, added_so


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
    parser.add_argument(
        "--wheel-csv-name",
        default="wheel_sizes.csv",
        help=(
            "Name of the wheel CSV file written under "
            "<cache-dir>/<repo>/ (default: wheel_sizes.csv). Use e.g. "
            "wheel_sizes_reduced.csv to record the reduced wheel separately."
        ),
    )
    parser.add_argument(
        "--skip-so",
        action="store_true",
        help=(
            "Do not record shared-library sizes to so_sizes.csv. Useful when "
            "the inspected workflow builds a reduced wheel whose compiled "
            "extensions differ from the released ones."
        ),
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    _log("record_wheel_sizes.py starting")
    _log(f"  cache directory : {args.cache_dir}")
    _log(f"  repository      : {args.repo}")
    _log(f"  workflow        : {args.workflow}")
    _log(f"  months          : {args.months}")
    _log(f"  wheel csv name  : {args.wheel_csv_name}")
    _log(f"  skip so series  : {args.skip_so}")
    if args.max_runs is not None:
        _log(f"  max runs        : {args.max_runs}")
    if not token:
        _log("  authentication  : anonymous (no GITHUB_TOKEN/GH_TOKEN set)")
        print("warning: no GITHUB_TOKEN/GH_TOKEN set; using anonymous requests.")
    else:
        _log("  authentication  : using GITHUB_TOKEN/GH_TOKEN")

    try:
        added_wheels, added_so = process_repo(
            args.repo,
            args.workflow,
            args.cache_dir,
            args.months,
            token,
            args.max_runs,
            wheel_csv_name=args.wheel_csv_name,
            skip_so=args.skip_so,
        )
    except urllib.error.HTTPError as exc:
        print(
            f"[{args.repo}] HTTP error {exc.code}: {exc.reason}",
            file=sys.stderr,
        )
        return 1
    _log(
        f"Done. {added_wheels} new wheel row(s) and "
        f"{added_so} new shared-library row(s) recorded."
    )
    print(
        f"Done. {added_wheels} new wheel row(s) and "
        f"{added_so} new shared-library row(s) recorded."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
