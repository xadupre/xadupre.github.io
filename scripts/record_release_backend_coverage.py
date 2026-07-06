"""Append a summary row to the backend-coverage time-series CSV.

After ``record_onnx_backend_node_coverage.py`` writes a fresh
``backend_node_coverage.json`` snapshot, this script reads that snapshot
and appends one row to the matching ``release_backend_coverage.csv`` time
series so that the dashboard can plot pass-rate trends over time.

The CSV columns are::

    date,onnxruntime_version,onnx_version,onnx_light_version,yobx_version,
    n_tests,
    onnxruntime_pass,onnxruntime_fail,
    reference_pass,reference_fail,
    onnx_light_pass,onnx_light_fail,
    yobx_pass,yobx_fail

If the CSV already contains a row with the same ``date`` value the script
exits without writing anything (idempotent re-runs).

Usage::

    python scripts/record_release_backend_coverage.py \\
        --json-path cache_data/onnx/backend_node_coverage.json \\
        --csv-path  cache_data/onnx/release_backend_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from typing import Dict, List, Optional

CSV_FIELDS: tuple = (
    "date",
    "onnxruntime_version",
    "onnx_version",
    "onnx_light_version",
    "yobx_version",
    "n_tests",
    "onnxruntime_pass",
    "onnxruntime_fail",
    "reference_pass",
    "reference_fail",
    "onnx_light_pass",
    "onnx_light_fail",
    "yobx_pass",
    "yobx_fail",
)

BACKENDS: tuple = ("onnxruntime", "reference", "onnx_light", "yobx")


def _log(message: str) -> None:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


def load_json(json_path: str) -> Dict:
    """Load and return the JSON payload from ``json_path``."""
    with open(json_path, encoding="utf-8") as fh:
        return json.load(fh)


def build_row(payload: Dict) -> Dict[str, str]:
    """Extract one CSV row from a ``backend_node_coverage.json`` payload."""
    date = payload.get("date", "")
    versions: Dict = payload.get("versions", {})
    totals: Dict = payload.get("totals", {})

    # Derive n_tests from the first backend that has data; all backends run
    # the same test suite so pass+fail is the same for every backend.
    n_tests = 0
    for backend in BACKENDS:
        bt = totals.get(backend, {})
        candidate = bt.get("pass", 0) + bt.get("fail", 0)
        if candidate:
            n_tests = candidate
            break

    row: Dict[str, str] = {"date": date}
    row["onnxruntime_version"] = versions.get("onnxruntime", "")
    row["onnx_version"] = versions.get("onnx", "")
    row["onnx_light_version"] = versions.get("onnx_light", "")
    row["yobx_version"] = versions.get("yobx", "")
    row["n_tests"] = str(n_tests)
    for backend in BACKENDS:
        bt = totals.get(backend, {})
        row[f"{backend}_pass"] = str(bt.get("pass", 0))
        row[f"{backend}_fail"] = str(bt.get("fail", 0))
    return row


def read_existing_dates(csv_path: str) -> List[str]:
    """Return the list of ``date`` values already recorded in ``csv_path``."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [r["date"] for r in reader if r.get("date")]


def append_row(csv_path: str, row: Dict[str, str]) -> None:
    """Append ``row`` to ``csv_path``, creating the file with a header if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the backend_node_coverage.json snapshot to read.",
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the release_backend_coverage.csv file to append to.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _log(f"Loading {args.json_path} ...")
    payload = load_json(args.json_path)
    row = build_row(payload)
    date = row.get("date", "")
    existing = read_existing_dates(args.csv_path)
    if date and date in existing:
        _log(f"Row for {date} already present in {args.csv_path}; skipping.")
        return 0
    append_row(args.csv_path, row)
    _log(f"Appended row for {date} to {args.csv_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
