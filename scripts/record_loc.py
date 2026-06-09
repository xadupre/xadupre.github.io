"""Record the number of lines of Python and C++ source code in a checkout.

This script walks a source tree and counts, per language, the number of
source files and the total number of lines (including blank lines and
comments) they contain. It then appends a single row per language to a CSV
file so that the evolution of the codebase size over time can be plotted by
the dashboard.

The CSV columns are::

    date,commit,language,files,lines

The script is intentionally dependency-free so that it can be invoked from
any GitHub Actions workflow without having to install an external tool such
as ``cloc`` or ``tokei``. Generated artefacts (``build/``, ``dist/``,
``__pycache__/`` ...) and version-control / virtual-environment
directories (``.git``, ``.venv`` ...) are skipped so that only first-party
source code is counted.

Usage::

    python scripts/record_loc.py --source-dir /path/to/onnx-light \\
        --output cache_data/onnx-light/loc.csv

When ``--commit`` is omitted, the script attempts to read the current
commit SHA from ``git`` inside ``--source-dir``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
from typing import Iterable

CSV_FIELDS = ("date", "commit", "language", "files", "lines")

# Mapping from language name to the set of file extensions (lower-case,
# including the leading dot) that belong to it.
LANGUAGES: dict[str, tuple[str, ...]] = {
    "Python": (".py",),
    "C++": (
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".c++",
        ".h",
        ".hh",
        ".hpp",
        ".hxx",
        ".h++",
        ".inl",
        ".ipp",
        ".tcc",
    ),
}

# Directories that should never be descended into when walking the source
# tree. These typically contain build artefacts, virtual environments or
# third-party vendored code that is not part of the project itself.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        "_build",
        ".eggs",
        ".cache",
        "third_party",
        "third-party",
        "external",
        "vendor",
    }
)


def _ext_to_language() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for language, exts in LANGUAGES.items():
        for ext in exts:
            mapping[ext] = language
    return mapping


def count_lines(path: str) -> int:
    """Return the number of lines in *path*.

    The file is opened in binary mode so that a stray decoding error in a
    source file does not interrupt the whole count.
    """
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def count_source_tree(source_dir: str) -> dict[str, dict[str, int]]:
    """Count files and lines per language under *source_dir*.

    Returns a dictionary mapping each language name in :data:`LANGUAGES`
    to a ``{"files": int, "lines": int}`` dictionary. Languages with no
    source file are still included with zero values so that the resulting
    CSV always has a stable shape.
    """
    ext_to_language = _ext_to_language()
    totals: dict[str, dict[str, int]] = {
        language: {"files": 0, "lines": 0} for language in LANGUAGES
    }
    for root, dirs, files in os.walk(source_dir):
        # Skip the directories listed above. ``dirs[:] = ...`` mutates the
        # list in place which is required for ``os.walk`` to honour the
        # filtering.
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            language = ext_to_language.get(ext)
            if language is None:
                continue
            try:
                lines = count_lines(os.path.join(root, name))
            except OSError:
                # Broken symlinks or unreadable files are ignored rather
                # than aborting the whole run.
                continue
            totals[language]["files"] += 1
            totals[language]["lines"] += lines
    return totals


def _git_head_sha(source_dir: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_dir,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.decode("ascii", errors="replace").strip()


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_rows(
    output: str,
    totals: dict[str, dict[str, int]],
    date_iso: str,
    commit: str,
) -> list[dict[str, str]]:
    """Append one row per language to the CSV at *output*.

    The CSV header is written automatically when *output* does not exist
    yet. Returns the list of rows that were appended.
    """
    rows: list[dict[str, str]] = []
    for language in LANGUAGES:
        entry = totals.get(language, {"files": 0, "lines": 0})
        rows.append(
            {
                "date": date_iso,
                "commit": commit,
                "language": language,
                "files": str(entry["files"]),
                "lines": str(entry["lines"]),
            }
        )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    write_header = not os.path.exists(output) or os.path.getsize(output) == 0
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory of the source tree whose lines of code to count.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the CSV file that the new rows will be appended to.",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help=(
            "Commit SHA to record. Defaults to the output of "
            "'git rev-parse HEAD' inside --source-dir."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "ISO-8601 timestamp to record (defaults to the current UTC time)."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_dir = os.path.abspath(args.source_dir)
    if not os.path.isdir(source_dir):
        print(f"--source-dir {source_dir!r} is not a directory.", file=sys.stderr)
        return 2

    commit = args.commit if args.commit is not None else _git_head_sha(source_dir)
    date_iso = args.date if args.date is not None else _now_iso()

    totals = count_source_tree(source_dir)
    rows = append_rows(args.output, totals, date_iso=date_iso, commit=commit)
    for row in rows:
        print(
            "Recorded {language}: {files} files, {lines} lines at {commit}.".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
