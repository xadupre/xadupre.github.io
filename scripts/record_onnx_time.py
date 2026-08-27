"""Append the timings printed by ``plot_onnx_time`` to a historical CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
from html.parser import HTMLParser

CSV_FIELDS = (
    "date",
    "commit",
    "run_id",
    "machine",
    "name",
    "avg",
    "median",
    "max",
    "std",
)
_TIMING = re.compile(
    r"^(?P<name>\S+)\s+avg=(?P<avg>[\d.]+) ms "
    r"median=(?P<median>[\d.]+) ms max=(?P<max>[\d.]+) ms "
    r"std=(?P<std>[\d.]+) ms$",
    re.MULTILINE,
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def extract_timings(html: str) -> list[dict[str, str | float]]:
    """Extract one row per printed benchmark result from gallery HTML."""
    parser = _TextExtractor()
    parser.feed(html)
    rows: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for match in _TIMING.finditer("".join(parser.parts)):
        name = match.group("name")
        if name in seen:
            continue
        seen.add(name)
        row: dict[str, str | float] = {"name": name}
        row.update(
            {
                key: float(match.group(key)) / 1000
                for key in ("avg", "median", "max", "std")
            }
        )
        rows.append(row)
    return rows


def append_snapshot(
    html_path: str,
    csv_path: str,
    date: str,
    commit: str,
    run_id: str,
    machine: str,
) -> int:
    """Append a snapshot unless ``run_id`` is already present in the CSV."""
    existing: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_rows = list(reader)
            existing = {row["run_id"] for row in existing_rows}
        if "machine" not in (reader.fieldnames or ()):
            with open(csv_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in existing_rows:
                    writer.writerow({"machine": "not recorded", **row})
    if run_id in existing:
        return 0

    with open(html_path, encoding="utf-8") as stream:
        rows = extract_timings(stream.read())
    if not rows:
        raise ValueError(f"No benchmark timings found in {html_path!r}.")

    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            row = {
                key: f"{value:.9g}" if isinstance(value, float) else value
                for key, value in row.items()
            }
            writer.writerow(
                {
                    "date": date,
                    "commit": commit,
                    "run_id": run_id,
                    "machine": machine,
                    **row,
                }
            )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", help="generated plot_onnx_time.html")
    parser.add_argument("--output", required=True, help="historical CSV path")
    parser.add_argument("--commit", required=True, help="onnx-light source commit")
    parser.add_argument("--run-id", required=True, help="documentation workflow run id")
    parser.add_argument("--machine", required=True, help="benchmark machine description")
    parser.add_argument(
        "--date",
        default=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        help="snapshot date (default: current UTC time)",
    )
    args = parser.parse_args()
    count = append_snapshot(
        args.html, args.output, args.date, args.commit, args.run_id, args.machine
    )
    print(f"Recorded {count} timing row(s).")


if __name__ == "__main__":
    main()
