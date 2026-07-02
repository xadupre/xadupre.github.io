"""Record per-snapshot backend coverage totals into a CSV.

Reads an existing backend coverage JSON file (produced by
``record_onnx_backend_test_coverage.py`` or
``record_onnx_backend_node_coverage.py``) and appends one row to a
per-snapshot CSV so that coverage can be tracked over time and displayed
on a release-coverage dashboard.

The CSV columns are auto-detected from the backends present in the JSON
``totals`` section.  Both three-backend (onnx-light) and four-backend
(onnx node) payloads are supported.

Usage::

    python scripts/record_release_backend_coverage.py \\
        --json-path cache_data/onnx-light/backend_test_coverage.json \\
        --csv-path  cache_data/onnx-light/release_backend_coverage.csv

If the CSV already contains a row for the same ``date`` (the ISO-8601
timestamp written by the JSON recorder), the script exits without writing
anything to avoid duplicates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple


# Ordered list of backends that may appear in a coverage payload, together
# with the package whose version is reported under the matching key.
_BACKEND_PACKAGE_ORDER: Tuple[Tuple[str, str], ...] = (
    ("onnxruntime", "onnxruntime"),
    ("reference", "onnx"),
    ("onnx_light", "onnx_light"),
    ("yobx", "yobx"),
)


def _csv_columns(backends: List[str]) -> List[str]:
    """Return the ordered CSV column names for ``backends``."""
    cols = ["date"]
    for backend, pkg in _BACKEND_PACKAGE_ORDER:
        if pkg in ("onnx_light", "yobx", "onnxruntime", "onnx"):
            # Version columns: one per unique package across all backends.
            pass
    # Version columns – emit one per *package* (not per backend) in a stable
    # order so that the header is predictable.
    version_pkgs_seen: List[str] = []
    for backend, pkg in _BACKEND_PACKAGE_ORDER:
        if backend in backends and pkg not in version_pkgs_seen:
            version_pkgs_seen.append(pkg)
    for pkg in version_pkgs_seen:
        cols.append(f"{pkg}_version")
    # n_tests is the total number of tests exercised.
    cols.append("n_tests")
    # Per-backend pass/fail counts.
    for backend in backends:
        cols.append(f"{backend}_pass")
        cols.append(f"{backend}_fail")
    return cols


def load_json(json_path: str) -> Dict:
    """Load and return the JSON payload from ``json_path``."""
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def read_existing_dates(csv_path: str) -> set:
    """Return the set of ``date`` values already present in ``csv_path``."""
    if not os.path.exists(csv_path):
        return set()
    dates: set = set()
    try:
        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                d = row.get("date", "").strip()
                if d:
                    dates.add(d)
    except (OSError, csv.Error):
        pass
    return dates


def build_row(payload: Dict) -> Tuple[List[str], Dict[str, str]]:
    """Extract a CSV row from ``payload``.

    Returns ``(columns, row_dict)`` where ``columns`` is the ordered list
    of column names and ``row_dict`` maps each column name to its string
    value.
    """
    date: str = payload.get("date", "")
    versions: Dict[str, str] = payload.get("versions") or {}
    totals: Dict[str, Dict[str, int]] = payload.get("totals") or {}

    # Determine which backends are present in the payload, preserving the
    # canonical order defined in ``_BACKEND_PACKAGE_ORDER``.
    backends = [b for b, _ in _BACKEND_PACKAGE_ORDER if b in totals]

    columns = _csv_columns(backends)

    row: Dict[str, str] = {"date": date}

    # Version columns – map package name → version string.
    version_pkgs_seen: List[str] = []
    for backend, pkg in _BACKEND_PACKAGE_ORDER:
        if backend in backends and pkg not in version_pkgs_seen:
            version_pkgs_seen.append(pkg)
    for pkg in version_pkgs_seen:
        row[f"{pkg}_version"] = versions.get(pkg, "")

    # Total test count: sum of pass+fail for the first backend (all backends
    # see the same tests, so they all have the same total).
    first_backend = backends[0] if backends else None
    if first_backend:
        first_totals = totals.get(first_backend, {})
        n_tests = first_totals.get("pass", 0) + first_totals.get("fail", 0)
    else:
        n_tests = 0
    row["n_tests"] = str(n_tests)

    # Per-backend pass/fail.
    for backend in backends:
        bt = totals.get(backend, {})
        row[f"{backend}_pass"] = str(bt.get("pass", 0))
        row[f"{backend}_fail"] = str(bt.get("fail", 0))

    return columns, row


def append_row(
    csv_path: str,
    columns: List[str],
    row: Dict[str, str],
    existing_dates: Optional[set] = None,
) -> bool:
    """Append ``row`` to ``csv_path`` if its date is not already present.

    Returns ``True`` when the row was written, ``False`` when it was
    skipped because the date was already present.
    """
    date = row.get("date", "")
    if existing_dates is not None and date in existing_dates:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})
    return True


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the backend coverage JSON file to read.",
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the per-snapshot CSV file to append to.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    payload = load_json(args.json_path)
    existing_dates = read_existing_dates(args.csv_path)
    columns, row = build_row(payload)

    written = append_row(args.csv_path, columns, row, existing_dates)
    if written:
        print(
            f"Appended row for {row.get('date')} to {args.csv_path}.",
            flush=True,
        )
    else:
        print(
            f"Row for {row.get('date')} already present in {args.csv_path}; "
            "nothing written.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
