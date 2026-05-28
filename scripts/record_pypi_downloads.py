"""Record weekly PyPI download statistics for ONNX-related packages.

The script queries `pypistats.org <https://pypistats.org>`_ for each of the
packages listed in :data:`DEFAULT_PACKAGES` and appends one row per package
to ``cache_data/<package>/downloads.csv``. The cached columns are::

    date,package,last_day,last_week,last_month

``date`` is the UTC timestamp at which the data was fetched. The other
columns mirror the ``data`` block returned by the ``/recent`` endpoint of
pypistats.org, which reports the number of downloads in the last day,
the last week and the last month.

Usage::

    python scripts/record_pypi_downloads.py [--cache-dir DIR] \
        [--package NAME ...]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request


PYPISTATS_RECENT_URL = "https://pypistats.org/api/packages/{package}/recent"

DEFAULT_PACKAGES: tuple[str, ...] = (
    "onnx",
    "onnxruntime",
    "onnxruntime-genai",
    "skl2onnx",
    "onnxmltools",
    "onnxscript",
    "ir-py",
    "tf2onnx",
)

CSV_FIELDS: tuple[str, ...] = (
    "date",
    "package",
    "last_day",
    "last_week",
    "last_month",
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


def fetch_recent_downloads(package: str) -> dict:
    """Return the ``data`` block of pypistats ``/recent`` for ``package``."""
    url = PYPISTATS_RECENT_URL.format(package=package)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "xadupre.github.io-record-pypi-downloads"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - pypistats.org
        payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected pypistats payload for {package!r}: {payload!r}"
        )
    return data


def build_row(
    package: str, data: dict, now: dt.datetime | None = None
) -> dict[str, str]:
    """Build the CSV row for ``package`` from a pypistats ``data`` block."""
    return {
        "date": _format_iso(now or dt.datetime.now(tz=dt.timezone.utc)),
        "package": package,
        "last_day": str(data.get("last_day", "")),
        "last_week": str(data.get("last_week", "")),
        "last_month": str(data.get("last_month", "")),
    }


def append_row(csv_path: str, row: dict[str, str]) -> None:
    """Append ``row`` to ``csv_path``, creating the file (with header) if needed."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_package(package: str, cache_dir: str) -> dict[str, str]:
    """Fetch the recent download counts for ``package`` and cache them."""
    _log(f"Fetching pypistats /recent for {package}...")
    data = fetch_recent_downloads(package)
    row = build_row(package, data)
    csv_path = os.path.join(cache_dir, package, "downloads.csv")
    append_row(csv_path, row)
    _log(f"Appended row to {csv_path}: {row}")
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help=(
            "PyPI package to inspect. May be passed multiple times. "
            "Defaults to the ONNX-related packages listed in the script."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the CSV cache (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    packages = tuple(args.packages) if args.packages else DEFAULT_PACKAGES
    failures: list[str] = []
    for package in packages:
        try:
            record_package(package, args.cache_dir)
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            _log(f"ERROR: failed to record downloads for {package}: {exc}")
            failures.append(package)
    if failures:
        _log(f"Completed with failures for: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
