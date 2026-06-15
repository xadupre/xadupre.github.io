"""Backfill shared-library (binary) sizes from ``build_release.yml`` artifacts.

The ``Binary size over time`` chart on
``dashboard/onnx-light/package-size.html`` is fed by
``cache_data/onnx-light/so_sizes.csv``. Historically that file was only
populated by the inline ``Record onnx_py shared library sizes`` step of the
``record_size_onnx_light`` workflow, which records a single snapshot of the
current build. As a result the chart could only ever show data starting from
the first time that step ran, rather than the 6 months of history the sibling
*wheel*-size series already collects through ``record_wheel_sizes.py``.

This script closes that gap: it queries the GitHub Actions REST API for every
completed run of a workflow (``build_release.yml`` of ``xadupre/onnx-light`` by
default) created within the look-back window, downloads each artifact, and
records one CSV row per shared library (``.so`` / ``.pyd`` / ``.dylib``) found
inside the wheels stored in the artifact (or directly inside the artifact). The
resulting rows extend ``so_sizes.csv`` backwards so that the binary-size chart
can show up to ``--months`` (6 by default) of history.

The CSV columns match the schema produced by the inline workflow step so that
the dashboard can consume rows from either source interchangeably::

    date,commit,size,name

where:

* ``date`` is the ISO 8601 UTC timestamp of the workflow run creation,
* ``commit`` is the head SHA of the run,
* ``size`` is the size in bytes of the shared library as stored inside the
  artifact / wheel zip archive,
* ``name`` is the shared library file name (e.g.
  ``_onnxpykernels.cpython-312-x86_64-linux-gnu.so``).

Rows are de-duplicated by ``commit`` so that re-running the script is safe and
so that commits already recorded by the inline workflow step are not recorded a
second time.

Usage::

    python scripts/record_so_sizes.py [--cache-dir DIR] [--repo owner/name]
        [--workflow build_release.yml] [--months N] [--max-runs N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import sys
import urllib.error
import zipfile
from typing import Iterable

# Reuse the HTTP plumbing, ISO-8601 helpers and run/artifact iteration already
# implemented (and tested) for the wheel-size recorder so that there is a
# single place for the GitHub Actions REST API logic.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from record_wheel_sizes import (  # noqa: E402
    DEFAULT_REPO,
    DEFAULT_WORKFLOW,
    _download,
    _format_iso,
    _log,
    iter_workflow_runs,
    list_run_artifacts,
)

# ``so_sizes.csv`` is shared with the inline workflow step, which writes exactly
# these four columns. The schema must not change (extra columns would produce
# ragged rows once both sources append to the same file).
CSV_FIELDS = ("date", "commit", "size", "name")

# Extensions of the compiled shared libraries tracked by the binary-size chart.
# ``.so.<version>`` files (e.g. ``libfoo.so.1``) are matched separately below.
SHARED_LIBRARY_SUFFIXES = (".so", ".pyd", ".dylib")


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


def _extract_from_wheel(wheel_bytes: bytes) -> list[tuple[str, int]]:
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
                for so_name, so_size in _extract_from_wheel(zf.read(info)):
                    _record(so_name, so_size)
            elif name and _is_shared_library(name):
                _record(name, int(info.file_size))
    return sorted(sizes.items())


def process_run(run: dict, repo: str, token: str | None) -> list[dict]:
    """Return CSV rows for every shared library attached to ``run``.

    Returns an empty list when the run is not completed or has no shared
    libraries. Runs whose conclusion is not ``success`` are still inspected
    because the upstream workflow may have uploaded artifacts before a later
    step failed. Artifacts whose download fails are skipped with a warning so
    that one broken artifact does not abort the rest of the run.
    """
    if run.get("status") != "completed":
        return []
    run_id = str(run.get("id", ""))
    if not run_id:
        return []
    created = run.get("created_at") or ""
    commit = run.get("head_sha") or ""
    artifacts = list_run_artifacts(repo, run_id, token)
    sizes: dict[str, int] = {}
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
            libraries = extract_shared_library_sizes(data)
        except zipfile.BadZipFile as exc:
            print(
                f"[{repo}] run {run_id}: artifact {artifact.get('name')!r} "
                f"is not a valid zip: {exc}",
                file=sys.stderr,
            )
            continue
        for name, size in libraries:
            previous = sizes.get(name)
            if previous is None or size > previous:
                sizes[name] = size
    return [
        {
            "date": created,
            "commit": commit,
            "size": str(size),
            "name": name,
        }
        for name, size in sorted(sizes.items())
    ]


def process_repo(
    repo: str,
    workflow: str,
    cache_dir: str,
    months: int,
    token: str | None,
    max_runs: int | None = None,
) -> int:
    """Backfill shared-library sizes for ``repo`` and append them to the cache.

    Returns the number of new CSV rows appended.
    """
    repo_name = repo.split("/", 1)[-1]
    csv_path = os.path.join(cache_dir, repo_name, "so_sizes.csv")
    seen_commits = read_existing_commits(csv_path)
    since = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(days=months * 30)
    _log(
        f"[{repo}] cache file: {csv_path} "
        f"({len(seen_commits)} commit(s) already recorded)"
    )
    _log(f"[{repo}] fetching {workflow!r} runs since {_format_iso(since)}")
    new_rows: list[dict] = []
    processed = 0
    try:
        for run in iter_workflow_runs(repo, workflow, since, token, max_runs):
            processed += 1
            commit = run.get("head_sha") or ""
            if not commit or commit in seen_commits:
                continue
            _log(
                f"[{repo}] processing run {run.get('id')} "
                f"(commit={commit[:7]}, created={run.get('created_at')})"
            )
            try:
                rows = process_run(run, repo, token)
            except urllib.error.HTTPError as exc:
                print(
                    f"[{repo}] HTTP error while processing run "
                    f"{run.get('id')}: {exc.code} {exc.reason}",
                    file=sys.stderr,
                )
                continue
            if not rows:
                continue
            new_rows.extend(rows)
            seen_commits.add(commit)
            _log(
                f"[{repo}]   recorded {len(rows)} shared librar(ies) "
                f"for commit {commit[:7]}"
            )
    finally:
        added = append_rows(csv_path, new_rows)
    _log(
        f"[{repo}] processed {processed} run(s) from GitHub; "
        f"appended {added} new shared-library row(s) to {csv_path}"
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
    _log("record_so_sizes.py starting")
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
    _log(f"Done. {added} new shared-library row(s) recorded.")
    print(f"Done. {added} new shared-library row(s) recorded.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
