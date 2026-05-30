"""Record weekly scikit-learn coverage stats for ``yet-another-onnx-builder``.

The script imports :mod:`yobx.sklearn` (which registers all converters) and
calls :func:`yobx.sklearn.register.get_sklearn_estimator_coverage` for each
known library to compute coverage statistics. One row per ``(date, library)``
pair is appended to ``cache_data/yet-another-onnx-builder/sklearn_coverage.csv``
with the following columns::

    date,library,sklearn_version,yobx_version,n_estimators,
    n_predictable,n_converted,coverage_pct

``coverage_pct`` is computed as ``n_converted / n_predictable * 100`` (with a
value of ``0`` when no predictable estimator is found) and matches the
percentage displayed at the bottom of the upstream
``docs/coverage/sklearn/supported_converters`` page.

Libraries whose optional dependencies are not installed are skipped silently
so that the workflow only records data for what is available in the
environment.

Usage::

    python scripts/record_yobx_sklearn_coverage.py [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import os
import sys
from typing import Iterable


CSV_FIELDS = (
    "date",
    "library",
    "sklearn_version",
    "yobx_version",
    "n_estimators",
    "n_predictable",
    "n_converted",
    "coverage_pct",
)

# Libraries supported by ``yobx.sklearn.register.get_sklearn_estimator_coverage``.
DEFAULT_LIBRARIES: tuple[str, ...] = (
    "category_encoders",
    "imblearn",
    "lightgbm",
    "sklearn",
    "sksurv",
    "statsmodels",
    "xgboost",
)

# Python distribution name that needs to be importable for a given library.
_REQUIRED_MODULE = {
    "category_encoders": "category_encoders",
    "imblearn": "imblearn",
    "lightgbm": "lightgbm",
    "sklearn": "sklearn",
    "sksurv": "sksurv",
    "statsmodels": "statsmodels",
    "xgboost": "xgboost",
}


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


def _is_available(library: str) -> bool:
    """Return ``True`` if the third-party ``library`` can be imported."""
    module = _REQUIRED_MODULE.get(library, library)
    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - any import error means "not available"
        return False
    return True


def summarize_rows(rows: Iterable[dict]) -> dict[str, int]:
    """Aggregate the rows returned by ``get_sklearn_estimator_coverage``.

    Returns a dict with ``n_estimators``, ``n_predictable`` and
    ``n_converted``. ``n_predictable`` counts estimators which expose at
    least one of the methods returned by :func:`sklearn_exportable_methods`,
    and ``n_converted`` counts those for which a yobx converter is
    registered (the ``yobx`` key is not ``None``).
    """
    n_estimators = 0
    n_predictable = 0
    n_converted = 0
    for row in rows:
        n_estimators += 1
        if row.get("predictable"):
            n_predictable += 1
        if row.get("yobx") is not None:
            n_converted += 1
    return {
        "n_estimators": n_estimators,
        "n_predictable": n_predictable,
        "n_converted": n_converted,
    }


def _coverage_pct(n_converted: int, n_predictable: int) -> float:
    if n_predictable <= 0:
        return 0.0
    return n_converted / n_predictable * 100.0


def build_rows(
    libraries: Iterable[str], now: dt.datetime | None = None
) -> list[dict[str, str]]:
    """Build the CSV rows for the supplied ``libraries``.

    Importing :mod:`yobx.sklearn` exposes the public API but does **not**
    register the converters. :func:`yobx.sklearn.register_sklearn_converters`
    must be called before computing the coverage so that
    ``SKLEARN_CONVERTERS`` is fully populated.
    """
    import sklearn

    import yobx
    import yobx.sklearn
    from yobx.sklearn.register import get_sklearn_estimator_coverage

    # Importing :mod:`yobx.sklearn` only exposes the public API; the
    # converters themselves are populated lazily by
    # :func:`register_sklearn_converters`. It must be called explicitly,
    # otherwise ``SKLEARN_CONVERTERS`` stays empty and every library is
    # reported with a coverage of 0%.
    yobx.sklearn.register_sklearn_converters()

    date = _format_iso(now or dt.datetime.now(tz=dt.timezone.utc))
    sklearn_version = sklearn.__version__
    yobx_version = getattr(yobx, "__version__", "")

    rows: list[dict[str, str]] = []
    for library in libraries:
        if not _is_available(library):
            _log(f"Skipping {library}: package is not installed.")
            continue
        try:
            details = get_sklearn_estimator_coverage(libraries=library, rst=False)
        except Exception as exc:  # noqa: BLE001
            _log(f"Skipping {library}: coverage failed ({exc!r}).")
            continue
        summary = summarize_rows(details)
        pct = _coverage_pct(summary["n_converted"], summary["n_predictable"])
        rows.append(
            {
                "date": date,
                "library": library,
                "sklearn_version": sklearn_version,
                "yobx_version": yobx_version,
                "n_estimators": str(summary["n_estimators"]),
                "n_predictable": str(summary["n_predictable"]),
                "n_converted": str(summary["n_converted"]),
                "coverage_pct": f"{pct:.2f}",
            }
        )
    return rows


def append_rows(csv_path: str, rows: Iterable[dict[str, str]]) -> int:
    """Append ``rows`` to ``csv_path``, creating the file (with header) if needed.

    Returns the number of rows actually written.
    """
    rows = list(rows)
    if not rows:
        return 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the CSV cache (default: %(default)s).",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        default=list(DEFAULT_LIBRARIES),
        help="Libraries to inspect (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _log(f"Collecting yobx scikit-learn coverage for libraries={args.libraries}...")
    rows = build_rows(args.libraries)
    csv_path = os.path.join(
        args.cache_dir, "yet-another-onnx-builder", "sklearn_coverage.csv"
    )
    n = append_rows(csv_path, rows)
    _log(f"Appended {n} row(s) to {csv_path}.")
    for row in rows:
        _log(f"  {row}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
