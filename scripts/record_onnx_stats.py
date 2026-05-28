"""Record weekly statistics about the ``onnx`` PyPI package.

The script fetches the latest released version of ``onnx`` from PyPI, picks
the wheel built for the most recent CPython on Linux x86_64, and appends a
row to ``cache_data/onnx/stats.csv`` with the following columns::

    date,filename,name,wheel_size,version,last_opset,n_operators,
    n_test_cases,n_supported_types

The ``onnx`` package itself must be importable so that the schema/opset/test
case counts can be derived from the installed library. The wheel size is read
from the PyPI metadata, so the wheel itself does not need to be downloaded.

Usage::

    python scripts/record_onnx_stats.py [--cache-dir DIR] [--package NAME]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from typing import Iterable


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"

CSV_FIELDS = (
    "date",
    "filename",
    "name",
    "wheel_size",
    "version",
    "last_opset",
    "n_operators",
    "n_test_cases",
    "n_supported_types",
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


def fetch_pypi_metadata(package: str) -> dict:
    """Return the PyPI JSON metadata for ``package``."""
    url = PYPI_JSON_URL.format(package=package)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "xadupre.github.io-record-onnx-stats"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - pypi.org
        return json.loads(resp.read().decode("utf-8"))


# A manylinux x86_64 wheel filename looks like::
#
#     onnx-1.21.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
#
# The first ``cpXY`` tag indicates the CPython version the wheel targets.
_CPYTHON_TAG_RE = re.compile(r"cp(\d{2,3})")


def _python_version_from_filename(filename: str) -> tuple[int, int] | None:
    """Extract the ``(major, minor)`` CPython version from a wheel filename."""
    match = _CPYTHON_TAG_RE.search(filename)
    if not match:
        return None
    tag = match.group(1)
    # ``cp310`` -> (3, 10); ``cp39`` -> (3, 9); ``cp314`` -> (3, 14).
    if len(tag) == 2:
        return int(tag[0]), int(tag[1])
    return int(tag[0]), int(tag[1:])


def pick_latest_linux_wheel(files: Iterable[dict]) -> dict | None:
    """Pick the manylinux x86_64 wheel with the highest CPython version.

    ``files`` is the list of release artifacts returned by PyPI. Only entries
    with ``packagetype == 'bdist_wheel'`` and a ``manylinux``/``x86_64`` tag
    are considered. Returns ``None`` if no suitable wheel is found.
    """
    best: dict | None = None
    best_version: tuple[int, int] = (0, 0)
    for entry in files:
        if entry.get("packagetype") != "bdist_wheel":
            continue
        filename = entry.get("filename", "")
        if "manylinux" not in filename or "x86_64" not in filename:
            continue
        version = _python_version_from_filename(filename)
        if version is None:
            continue
        if version > best_version:
            best = entry
            best_version = version
    return best


def count_supported_types() -> int:
    """Return the number of defined ``onnx.TensorProto`` data types.

    The ``UNDEFINED`` placeholder is excluded so that the counter reflects the
    number of types ONNX actually supports.
    """
    from onnx import TensorProto

    return sum(1 for name, _ in TensorProto.DataType.items() if name != "UNDEFINED")


def count_node_test_cases() -> int:
    """Return the number of node test cases shipped with ``onnx``.

    Each subdirectory of ``onnx/backend/test/data/node`` whose name starts
    with ``test_`` corresponds to one node test case.
    """
    import onnx

    root = os.path.join(os.path.dirname(onnx.__file__), "backend", "test", "data", "node")
    if not os.path.isdir(root):
        return 0
    return sum(
        1
        for entry in os.listdir(root)
        if entry.startswith("test_") and os.path.isdir(os.path.join(root, entry))
    )


def collect_local_stats() -> dict[str, str]:
    """Collect the stats that require the installed ``onnx`` package."""
    import onnx
    import onnx.defs

    return {
        "version": onnx.__version__,
        "last_opset": str(onnx.defs.onnx_opset_version()),
        "n_operators": str(len(onnx.defs.get_all_schemas())),
        "n_test_cases": str(count_node_test_cases()),
        "n_supported_types": str(count_supported_types()),
    }


def build_row(package: str, metadata: dict, now: dt.datetime | None = None) -> dict[str, str]:
    """Build the CSV row for the latest release of ``package``."""
    version = metadata["info"]["version"]
    files = metadata.get("releases", {}).get(version, [])
    wheel = pick_latest_linux_wheel(files)
    if wheel is None:
        raise RuntimeError(
            f"No manylinux x86_64 wheel found for {package} {version}."
        )
    local = collect_local_stats()
    if local["version"] != version:
        _log(
            f"WARNING: installed {package} {local['version']} differs from "
            f"latest PyPI version {version}; the wheel size belongs to "
            f"the PyPI version while the other stats reflect the installed one."
        )
    return {
        "date": _format_iso(now or dt.datetime.now(tz=dt.timezone.utc)),
        "filename": wheel["filename"],
        "name": package,
        "wheel_size": str(wheel["size"]),
        "version": version,
        "last_opset": local["last_opset"],
        "n_operators": local["n_operators"],
        "n_test_cases": local["n_test_cases"],
        "n_supported_types": local["n_supported_types"],
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        default="onnx",
        help="PyPI package name to inspect (default: %(default)s).",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the CSV cache (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _log(f"Fetching PyPI metadata for {args.package}...")
    metadata = fetch_pypi_metadata(args.package)
    row = build_row(args.package, metadata)
    csv_path = os.path.join(args.cache_dir, args.package, "stats.csv")
    append_row(csv_path, row)
    _log(f"Appended row to {csv_path}: {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
