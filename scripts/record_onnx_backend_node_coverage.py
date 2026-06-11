"""Record the backend node test coverage of four ONNX runtimes.

The script walks every backend node test bundled with the installed
``onnx`` package (collected via
``onnx.backend.test.loader.load_model_tests``), runs each one against:

* ``onnxruntime`` (CPU execution provider),
* the ONNX Python reference implementation (``onnx.reference``),
* the ``onnx-light`` reference implementation backed by the C++
  ``KernelDispatchTable`` (``onnx_light.reference``) and
* the extended ``yobx`` reference implementation
  (``yobx.reference.ExtendedReferenceEvaluator``).

The pass/fail status of every (test, runtime) combination is persisted
to ``cache_data/onnx/backend_node_coverage.json``. The dashboard at
``dashboard/onnx/backend-test-coverage.html`` consumes that file to
render the table and pass ratios.

This is the counterpart of
:mod:`scripts.record_onnx_backend_test_coverage`, which uses the
``onnx-light`` test discovery (``onnx_light.onnx_lib.backend.test.case``) and
only compares three runtimes. The two scripts share most of their
implementation; the differences are the discovery source and the list
of backends.

Usage::

    python scripts/record_onnx_backend_node_coverage.py [--cache-dir DIR]
        [--kind node] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# Reuse helpers from the sibling onnx-light recorder so the two scripts
# stay in lock-step (error formatting, payload structure, last-pass
# carry-over, comparison tolerances, ...).
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import record_onnx_backend_test_coverage as _base  # noqa: E402

DEFAULT_RTOL = _base.DEFAULT_RTOL
DEFAULT_ATOL = _base.DEFAULT_ATOL
_log = _base._log
_format_iso = _base._format_iso
_stringify_error = _base._stringify_error
_compare_outputs = _base._compare_outputs
_model_input_names = _base._model_input_names
_load_test_data_sets = _base._load_test_data_sets
_run_with_onnxruntime = _base._run_with_onnxruntime
_run_with_reference = _base._run_with_reference
_run_with_onnx_light = _base._run_with_onnx_light
load_previous_payload = _base.load_previous_payload
write_payload = _base.write_payload

BACKENDS: Tuple[str, ...] = ("onnxruntime", "reference", "onnx_light", "yobx")

# Package whose version is recorded alongside the ``last_pass`` date for
# each backend.
BACKEND_PACKAGE: Dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "reference": "onnx",
    "onnx_light": "onnx_light",
    "yobx": "yobx",
}


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "onnx_light", "yobx", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def discover_node_tests(kind: str = "node") -> List[Dict[str, Any]]:
    """Return ``[{"name", "model", "data_sets", "tag"}, ...]`` for every test.

    The tests are loaded from ``onnx.backend.test.loader.load_model_tests``
    which ships with the installed ``onnx`` package. ``kind`` selects the
    test group (``node``, ``simple``, ``pytorch-converted``,
    ``pytorch-operator`` or ``real``); the default ``node`` matches the
    tests exercised by ``onnx``'s reference implementation.
    """
    import onnx
    from onnx.backend.test.loader import load_model_tests

    tests = load_model_tests(kind=kind)
    discovered: List[Dict[str, Any]] = []
    for tc in tests:
        name = getattr(tc, "name", None)
        if not name:
            continue
        model_dir = getattr(tc, "model_dir", None)
        model = getattr(tc, "model", None)
        data_sets = getattr(tc, "data_sets", None)
        if model is None and model_dir:
            try:
                model = onnx.load(os.path.join(str(model_dir), "model.onnx"))
            except Exception:  # noqa: BLE001 - skip unreadable models
                continue
        if data_sets is None and model_dir:
            try:
                data_sets = _load_test_data_sets(str(model_dir))
            except Exception:  # noqa: BLE001 - skip unreadable data sets
                data_sets = []
        if model is None:
            continue
        converted_data_sets: List[Tuple[List[Any], List[Any]]] = [
            (list(inputs), list(outputs)) for inputs, outputs in (data_sets or [])
        ]
        tag = getattr(tc, "kind", None) or kind or ""
        discovered.append(
            {
                "name": str(name),
                "model": model,
                "data_sets": converted_data_sets,
                "tag": str(tag),
            }
        )
    discovered.sort(key=lambda d: d["name"])
    return discovered


def _run_with_yobx(model) -> Callable[[List[Any]], List[Any]]:
    """Run ``model`` with ``yobx.reference.ExtendedReferenceEvaluator``.

    ``yobx`` extends :class:`onnx.reference.ReferenceEvaluator` with
    additional contrib-op kernels while keeping the same
    ``sess.run(None, {...})`` interface.
    """
    from yobx.reference import ExtendedReferenceEvaluator

    evaluator = ExtendedReferenceEvaluator(model)
    input_names = _model_input_names(model)

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(evaluator.run(None, feeds))

    return _run


_BACKEND_FACTORIES: Dict[str, Callable[[Any], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _run_with_onnxruntime,
    "reference": _run_with_reference,
    "onnx_light": _run_with_onnx_light,
    "yobx": _run_with_yobx,
}


def run_test_with_backend(
    model: Any,
    data_sets: List[Tuple[List[Any], List[Any]]],
    backend: str,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Dict[str, Any]:
    """Run a single backend test against ``backend`` and return its status.

    The returned dictionary has the same shape as
    :func:`record_onnx_backend_test_coverage.run_test_with_backend`:
    ``{"success": bool, "error": str, "error_step": str}`` where
    ``error_step`` is ``"load"``, ``"run"`` or ``"compare"``.
    """
    factory = _BACKEND_FACTORIES.get(backend)
    if factory is None:
        return {
            "success": False,
            "error": f"unknown backend: {backend}",
            "error_step": "load",
        }
    if not data_sets:
        return {
            "success": False,
            "error": "no test_data_set_* directory found",
            "error_step": "load",
        }
    try:
        runner = factory(model)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "load",
        }
    for inputs, expected in data_sets:
        try:
            actual = runner(inputs)
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": _stringify_error(exc),
                "error_step": "run",
            }
        mismatch = _compare_outputs(expected, actual, rtol=rtol, atol=atol)
        if mismatch is not None:
            return {
                "success": False,
                "error": mismatch,
                "error_step": "compare",
            }
    return {"success": True, "error": "", "error_step": ""}


def _row_from_results(
    name: str,
    results: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]] = None,
    versions: Optional[Dict[str, str]] = None,
    now_iso: Optional[str] = None,
    tag: str = "",
) -> Dict[str, Any]:
    """Build a dashboard row, carrying over per-backend ``last_pass`` info."""
    versions = versions or {}
    previous = previous or {}
    row: Dict[str, Any] = {"name": name}
    if tag:
        row["tag"] = tag
    elif previous.get("tag"):
        row["tag"] = previous["tag"]
    for backend in BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        row[backend] = success
        error = _stringify_error(info.get("error"))
        if error:
            row[f"{backend}_error"] = error
        step = info.get("error_step") or ""
        if step:
            row[f"{backend}_error_step"] = step
        if success and now_iso is not None:
            row[f"{backend}_last_pass_date"] = now_iso
            pkg = BACKEND_PACKAGE.get(backend)
            version = versions.get(pkg) if pkg else None
            if version:
                row[f"{backend}_last_pass_version"] = version
        else:
            prev_date = previous.get(f"{backend}_last_pass_date")
            if prev_date:
                row[f"{backend}_last_pass_date"] = prev_date
            prev_version = previous.get(f"{backend}_last_pass_version")
            if prev_version:
                row[f"{backend}_last_pass_version"] = prev_version
    return row


def _index_previous_rows(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = payload.get("tests") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            name = row.get("name")
            if isinstance(name, str):
                indexed[name] = row
    return indexed


def build_payload(
    kind: str = "node",
    limit: Optional[int] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    discover: Callable[[str], List[Dict[str, Any]]] = discover_node_tests,
    run: Callable[..., Dict[str, Any]] = run_test_with_backend,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover all tests, run them on every backend and return a payload."""
    if versions is None:
        versions = collect_versions
    tests = discover(kind)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    _log(f"Discovered {len(tests)} {kind} backend tests.")

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)
    version_map = versions()
    previous_rows = _index_previous_rows(previous or {})

    rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {
        backend: {"pass": 0, "fail": 0} for backend in BACKENDS
    }
    for idx, test in enumerate(tests):
        name = test["name"]
        model = test["model"]
        data_sets = test["data_sets"]
        results: Dict[str, Dict[str, Any]] = {}
        for backend in BACKENDS:
            try:
                info = run(model, data_sets, backend, rtol=rtol, atol=atol)
            except Exception as exc:  # noqa: BLE001
                _log(
                    f"Unhandled error for {name} on {backend}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                info = {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "run",
                }
            results[backend] = info
            bucket = "pass" if info.get("success") else "fail"
            totals[backend][bucket] += 1
        rows.append(
            _row_from_results(
                name,
                results,
                previous=previous_rows.get(name),
                versions=version_map,
                now_iso=now_iso,
                tag=str(test.get("tag", "") or ""),
            )
        )
        if (idx + 1) % 50 == 0:
            _log(f"Ran {idx + 1}/{len(tests)} tests.")

    return {
        "date": now_iso,
        "kind": kind,
        "tolerances": {"rtol": rtol, "atol": atol},
        "versions": version_map,
        "totals": totals,
        "tests": rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the JSON cache (default: %(default)s).",
    )
    parser.add_argument(
        "--kind",
        default="node",
        help=(
            "Backend test group to run (default: %(default)s). "
            "Common values: node, simple, pytorch-converted, "
            "pytorch-operator, real."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of tests executed (useful for debugging).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for output comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance for output comparison (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(args.cache_dir, "onnx", "backend_node_coverage.json")
    previous = load_previous_payload(json_path)
    try:
        payload = build_payload(
            kind=args.kind,
            limit=args.limit,
            rtol=args.rtol,
            atol=args.atol,
            previous=previous,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record backend node test coverage: {exc}")
        traceback.print_exc()
        return 1
    write_payload(json_path, payload)
    _log(
        f"Wrote {len(payload['tests'])} test entries to {json_path} "
        f"(totals={payload['totals']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
