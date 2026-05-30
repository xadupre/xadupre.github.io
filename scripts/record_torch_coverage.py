"""Record the torch exporter coverage snapshot for ``yet-another-onnx-builder``.

This script imports ``yobx`` (yet-another-onnx-builder) and runs the
:func:`yobx.torch.testing.model_eval_cases.evaluation` helper over all
discovered test cases for a fixed set of torch-to-ONNX exporters. The
resulting per-(case, exporter, dynamic) status is serialised to
``cache_data/yet-another-onnx-builder/torch_coverage.json``.

The JSON file is consumed by
``dashboard/yet-another-onnx-builder/torch-coverage.html`` to render a
sortable table comparing how the ``yobx`` torch exporter performs against
the alternative torch-to-ONNX exporters.

Usage::

    python scripts/record_torch_coverage.py [--cache-dir DIR]
        [--exporter NAME ...] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from typing import Any, Dict, List, Tuple


DEFAULT_EXPORTERS: Tuple[str, ...] = (
    "yobx",
    "yobx-new-tracing",
    "export-tracing",
    "export-strict",
    "export-nostrict",
    "export-nostrict-decall",
    "dynamo",
)


def _log(message: str) -> None:
    """Print ``message`` prefixed with a UTC timestamp."""
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in ("yobx", "torch", "onnx", "onnxruntime", "onnxscript"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string for an exporter error message."""
    if value is None:
        return ""
    text = str(value)
    # Keep the message reasonably short for the JSON snapshot. Errors raised
    # by the exporters can be quite verbose (full graphs, stack traces, ...).
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > 400:
        text = text[:397] + "..."
    return text


def _normalise_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pick a JSON-serialisable subset of the fields returned by ``evaluation``."""
    success = result.get("success")
    if isinstance(success, bool):
        success_int = 1 if success else 0
    elif success is None:
        success_int = None
    else:
        try:
            success_int = int(success)
        except (TypeError, ValueError):
            success_int = None

    def _to_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "name": result.get("name"),
        "exporter": result.get("exporter"),
        "dynamic": 1 if result.get("dynamic") else 0,
        "success": success_int,
        "error_step": result.get("error_step") or "",
        "error": _stringify_error(result.get("error")),
        "abs": _to_float(result.get("abs")),
        "rel": _to_float(result.get("rel")),
    }


def run_coverage(
    exporters: Tuple[str, ...],
    dynamic: Tuple[bool, ...] = (False, True),
    limit: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run ``evaluation`` for all discovered cases and the given ``exporters``."""
    # Import lazily so that ``--help`` works without the heavy ``torch`` stack.
    from yobx.torch.testing.model_eval_cases import discover, evaluation

    all_cases = discover()
    if limit is not None:
        items = sorted(all_cases.items())[:limit]
        cases = dict(items)
    else:
        cases = dict(sorted(all_cases.items()))

    _log(
        f"Evaluating {len(cases)} cases against {len(exporters)} exporters "
        f"(dynamic={list(dynamic)})..."
    )
    raw_results = evaluation(
        exporters=exporters, dynamic=dynamic, cases=cases, verbose=0, quiet=True
    )
    results = [_normalise_result(r) for r in raw_results]
    return results, list(cases.keys())


def build_payload(
    results: List[Dict[str, Any]],
    case_names: List[str],
    exporters: Tuple[str, ...],
    dynamic: Tuple[bool, ...],
    commit: str | None,
) -> Dict[str, Any]:
    """Assemble the JSON snapshot to be written to disk."""
    totals: Dict[str, Dict[str, int]] = {}
    for exporter in exporters:
        totals[exporter] = {"success": 0, "failure": 0, "total": 0}
    for row in results:
        exporter = row["exporter"]
        bucket = totals.setdefault(
            exporter, {"success": 0, "failure": 0, "total": 0}
        )
        bucket["total"] += 1
        if row["success"] == 1:
            bucket["success"] += 1
        else:
            bucket["failure"] += 1

    return {
        "date": dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit or "",
        "versions": collect_versions(),
        "exporters": list(exporters),
        "dynamic": [bool(d) for d in dynamic],
        "cases": list(case_names),
        "totals": totals,
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the JSON cache (default: %(default)s).",
    )
    parser.add_argument(
        "--repo",
        default="yet-another-onnx-builder",
        help="Sub-directory of the cache to write into (default: %(default)s).",
    )
    parser.add_argument(
        "--exporter",
        action="append",
        dest="exporters",
        help=(
            "Exporter to evaluate. May be repeated. "
            f"Defaults to: {', '.join(DEFAULT_EXPORTERS)}."
        ),
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("YOBX_COMMIT", ""),
        help="Commit SHA of yet-another-onnx-builder used for the snapshot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to the first N cases (mainly for local testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exporters = tuple(args.exporters) if args.exporters else DEFAULT_EXPORTERS
    dynamic: Tuple[bool, ...] = (False, True)

    results, case_names = run_coverage(exporters, dynamic, limit=args.limit)
    payload = build_payload(results, case_names, exporters, dynamic, args.commit)

    out_dir = os.path.join(args.cache_dir, args.repo)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "torch_coverage.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    _log(
        f"Wrote {out_path}: {len(results)} results, "
        f"{len(case_names)} cases, {len(exporters)} exporters."
    )
    for exporter, bucket in payload["totals"].items():
        _log(
            f"  {exporter}: {bucket['success']}/{bucket['total']} succeeded "
            f"({bucket['failure']} failed)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
