"""Record the number of lines of Python and C++ source code in a checkout.

This script walks a source tree and counts, per language, the number of
source files and the total number of lines (including blank lines and
comments) they contain. It then appends a single row per language to a CSV
file so that the evolution of the codebase size over time can be plotted by
the dashboard.

The CSV columns are::

    date,commit,language,files,lines,code_lines,comment_lines

``lines`` is the total number of physical lines (including blanks and
comments) while ``code_lines`` and ``comment_lines`` count only lines
that contain source code or comments respectively. A line that mixes
code and a trailing comment is counted as a code line only.

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

CSV_FIELDS = (
    "date",
    "commit",
    "language",
    "files",
    "lines",
    "code_lines",
    "comment_lines",
)

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
    """Return the total number of physical lines in *path*.

    The file is opened in binary mode so that a stray decoding error in a
    source file does not interrupt the whole count.
    """
    total, _, _ = classify_file(path, language=None)
    return total


def _classify_python_line(stripped: bytes) -> tuple[bool, bool]:
    """Return ``(has_code, has_comment)`` for a single Python line.

    The line is expected to have been stripped of surrounding whitespace.
    Blank lines yield ``(False, False)``. Triple-quoted strings used as
    docstrings are counted as code because tracking string state across
    lines reliably would require a full tokeniser.
    """
    if not stripped:
        return False, False
    if stripped.startswith(b"#"):
        return False, True
    # A trailing ``# ...`` comment after code keeps ``has_code`` True; we
    # only need a best-effort detection of the ``#`` character outside of
    # string literals.
    has_comment = False
    in_single = False
    in_double = False
    i = 0
    n = len(stripped)
    while i < n:
        c = stripped[i:i + 1]
        if in_single:
            if c == b"\\" and i + 1 < n:
                i += 2
                continue
            if c == b"'":
                in_single = False
            i += 1
            continue
        if in_double:
            if c == b"\\" and i + 1 < n:
                i += 2
                continue
            if c == b'"':
                in_double = False
            i += 1
            continue
        if c == b"'":
            in_single = True
        elif c == b'"':
            in_double = True
        elif c == b"#":
            has_comment = True
            break
        i += 1
    return True, has_comment


def _classify_cpp_lines(data: bytes) -> tuple[int, int, int]:
    """Return ``(total, code, comment)`` line counts for C/C++ *data*.

    The classifier is intentionally lightweight: it tracks ``/* ... */``
    block comments across lines and ``// ...`` line comments, and skips
    over string and character literals so that ``//`` inside a string
    does not start a comment.
    """
    total = 0
    code_lines = 0
    comment_lines = 0
    in_block = False
    n = len(data)
    i = 0
    # Process line by line while remembering block-comment state.
    while i <= n:
        # Find end of current line.
        j = data.find(b"\n", i)
        if j == -1:
            line = data[i:]
            advance = n
            if not line and i == n:
                break
        else:
            line = data[i:j]
            advance = j + 1
        total += 1
        has_code = False
        has_comment = False
        k = 0
        m = len(line)
        in_string = 0  # 0 = none, 1 = ", 2 = '
        while k < m:
            c = line[k:k + 1]
            if in_block:
                has_comment = True
                if c == b"*" and k + 1 < m and line[k + 1:k + 2] == b"/":
                    in_block = False
                    k += 2
                    continue
                k += 1
                continue
            if in_string:
                if c == b"\\" and k + 1 < m:
                    k += 2
                    continue
                if (in_string == 1 and c == b'"') or (
                    in_string == 2 and c == b"'"
                ):
                    in_string = 0
                k += 1
                continue
            if c == b"/" and k + 1 < m and line[k + 1:k + 2] == b"/":
                has_comment = True
                break
            if c == b"/" and k + 1 < m and line[k + 1:k + 2] == b"*":
                has_comment = True
                in_block = True
                k += 2
                continue
            if c in (b" ", b"\t", b"\r"):
                k += 1
                continue
            has_code = True
            if c == b'"':
                in_string = 1
            elif c == b"'":
                in_string = 2
            k += 1
        if has_code:
            code_lines += 1
        elif has_comment:
            comment_lines += 1
        if j == -1:
            break
        i = advance
    return total, code_lines, comment_lines


def classify_file(path: str, language: str | None) -> tuple[int, int, int]:
    """Return ``(total, code, comment)`` line counts for *path*.

    When *language* is ``None`` only the total number of physical lines
    is reported and ``code`` / ``comment`` are returned as ``0``.
    """
    with open(path, "rb") as f:
        data = f.read()
    if language == "C++":
        return _classify_cpp_lines(data)
    if not data:
        return 0, 0, 0
    # Split keeping behaviour consistent with iterating over the file:
    # a trailing newline does not introduce an extra empty line.
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    total = len(lines)
    if language != "Python":
        return total, 0, 0
    code_lines = 0
    comment_lines = 0
    for raw in lines:
        stripped = raw.strip()
        has_code, has_comment = _classify_python_line(stripped)
        if has_code:
            code_lines += 1
        elif has_comment:
            comment_lines += 1
    return total, code_lines, comment_lines


def count_source_tree(source_dir: str) -> dict[str, dict[str, int]]:
    """Count files and lines per language under *source_dir*.

    Returns a dictionary mapping each language name in :data:`LANGUAGES`
    to a ``{"files": int, "lines": int, "code_lines": int,
    "comment_lines": int}`` dictionary. Languages with no source file
    are still included with zero values so that the resulting CSV
    always has a stable shape.
    """
    ext_to_language = _ext_to_language()
    totals: dict[str, dict[str, int]] = {
        language: {
            "files": 0,
            "lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
        }
        for language in LANGUAGES
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
                total, code, comment = classify_file(
                    os.path.join(root, name), language=language
                )
            except OSError:
                # Broken symlinks or unreadable files are ignored rather
                # than aborting the whole run.
                continue
            totals[language]["files"] += 1
            totals[language]["lines"] += total
            totals[language]["code_lines"] += code
            totals[language]["comment_lines"] += comment
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
        entry = totals.get(
            language,
            {"files": 0, "lines": 0, "code_lines": 0, "comment_lines": 0},
        )
        rows.append(
            {
                "date": date_iso,
                "commit": commit,
                "language": language,
                "files": str(entry["files"]),
                "lines": str(entry["lines"]),
                "code_lines": str(entry.get("code_lines", 0)),
                "comment_lines": str(entry.get("comment_lines", 0)),
            }
        )
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    write_header = not os.path.exists(output) or os.path.getsize(output) == 0
    if not write_header:
        _migrate_csv_header(output)
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def _migrate_csv_header(path: str) -> None:
    """Rewrite *path* so that its header matches :data:`CSV_FIELDS`.

    Existing rows are preserved; columns introduced after the file was
    first created are back-filled with empty strings so that the dashboard
    can still parse them.
    """
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        if tuple(header) == CSV_FIELDS:
            return
        existing_rows = list(reader)
    indices = {name: header.index(name) if name in header else -1 for name in CSV_FIELDS}
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for row in existing_rows:
            writer.writerow(
                [row[indices[name]] if indices[name] != -1 and indices[name] < len(row) else ""
                 for name in CSV_FIELDS]
            )


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
            "Recorded {language}: {files} files, {lines} lines "
            "({code_lines} code, {comment_lines} comment) at {commit}.".format(
                **row
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
