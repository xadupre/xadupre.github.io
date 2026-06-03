"""Record the backend node test coverage of ``onnxruntime`` and the ONNX
Python reference implementation.

The script walks every backend node test bundled with the installed
``onnx`` package (the same tests that are exercised by ``onnx-light``),
runs each one against:

* ``onnxruntime`` (CPU execution provider) and
* the ONNX Python reference implementation (``onnx.reference``),

and records whether the produced outputs match the expected ones. The
resulting per-test status is persisted to
``cache_data/onnx-light/backend_test_coverage.json``. The dashboard at
``dashboard/onnx-light/backend-test-coverage.html`` consumes that file to
render the table and pass ratio requested in the tracking issue.

Usage::

    python scripts/record_onnx_backend_test_coverage.py [--cache-dir DIR]
        [--kind node] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple


BACKENDS: Tuple[str, ...] = ("onnxruntime", "reference")

# Default numerical tolerances when comparing produced outputs with the
# expected ones. ``onnxruntime`` and the reference implementation are not
# always bit-identical (different math libraries, different summation
# orders, ...), so we use a generous tolerance that still catches real
# regressions.
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-4


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


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string representation of an error."""
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > 300:
        text = text[:297] + "..."
    return text


def discover_node_tests(kind: str = "node") -> List[Dict[str, str]]:
    """Return ``[{"name", "model_dir"}, ...]`` for every backend test.

    The tests are loaded from ``onnx.backend.test`` which ships with the
    installed ``onnx`` package. ``kind`` selects the test group (``node``,
    ``simple``, ``pytorch-converted``, ``pytorch-operator`` or ``real``).
    The default ``node`` matches the tests exercised by ``onnx-light``'s
    reference implementation.
    """
    from onnx.backend.test.loader import load_model_tests

    tests = load_model_tests(kind=kind)
    discovered: List[Dict[str, str]] = []
    for tc in tests:
        model_dir = getattr(tc, "model_dir", None)
        name = getattr(tc, "name", None)
        if not model_dir or not name:
            continue
        discovered.append({"name": str(name), "model_dir": str(model_dir)})
    discovered.sort(key=lambda d: d["name"])
    return discovered


def _load_tensor(path: str):
    import onnx
    from onnx import numpy_helper

    tensor = onnx.TensorProto()
    with open(path, "rb") as fh:
        tensor.ParseFromString(fh.read())
    return numpy_helper.to_array(tensor)


def _load_test_data_sets(model_dir: str) -> List[Tuple[List[Any], List[Any]]]:
    """Return ``[(inputs, expected_outputs), ...]`` for ``model_dir``.

    Each test directory contains one or more ``test_data_set_<n>``
    sub-directories with ``input_<i>.pb`` and ``output_<j>.pb`` files
    storing serialised ``TensorProto`` messages.
    """
    data_sets: List[Tuple[List[Any], List[Any]]] = []
    for name in sorted(os.listdir(model_dir)):
        if not name.startswith("test_data_set_"):
            continue
        ds_path = os.path.join(model_dir, name)
        if not os.path.isdir(ds_path):
            continue
        inputs: List[Any] = []
        i = 0
        while True:
            p = os.path.join(ds_path, "input_%d.pb" % i)
            if not os.path.exists(p):
                break
            inputs.append(_load_tensor(p))
            i += 1
        outputs: List[Any] = []
        j = 0
        while True:
            p = os.path.join(ds_path, "output_%d.pb" % j)
            if not os.path.exists(p):
                break
            outputs.append(_load_tensor(p))
            j += 1
        data_sets.append((inputs, outputs))
    return data_sets


def _model_input_names(model) -> List[str]:
    """Return the names of the graph inputs that are not initializers."""
    initializer_names = {init.name for init in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def _compare_outputs(
    expected: List[Any],
    actual: List[Any],
    rtol: float,
    atol: float,
) -> Optional[str]:
    """Return ``None`` if the outputs match, otherwise an error string."""
    import numpy as np

    if len(expected) != len(actual):
        return (
            "output count mismatch: expected %d, got %d"
            % (len(expected), len(actual))
        )
    for idx, (exp, act) in enumerate(zip(expected, actual)):
        exp_arr = np.asarray(exp)
        act_arr = np.asarray(act)
        if exp_arr.shape != act_arr.shape:
            return (
                "output %d shape mismatch: expected %s, got %s"
                % (idx, exp_arr.shape, act_arr.shape)
            )
        if exp_arr.dtype.kind in ("U", "S", "O") or act_arr.dtype.kind in (
            "U",
            "S",
            "O",
        ):
            if not np.array_equal(exp_arr, act_arr):
                return "output %d value mismatch" % idx
            continue
        try:
            np.testing.assert_allclose(
                act_arr, exp_arr, rtol=rtol, atol=atol, equal_nan=True
            )
        except AssertionError as exc:
            return "output %d mismatch (%s)" % (idx, _stringify_error(exc))
    return None


def _run_with_onnxruntime(model_dir: str) -> Callable[[List[Any]], List[Any]]:
    import onnxruntime

    model_path = os.path.join(model_dir, "model.onnx")
    sess = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(sess.run(None, feeds))

    return _run


def _run_with_reference(model_dir: str) -> Callable[[List[Any]], List[Any]]:
    import onnx
    from onnx.reference import ReferenceEvaluator

    model = onnx.load(os.path.join(model_dir, "model.onnx"))
    evaluator = ReferenceEvaluator(model)
    input_names = _model_input_names(model)

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(evaluator.run(None, feeds))

    return _run


_BACKEND_FACTORIES: Dict[str, Callable[[str], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _run_with_onnxruntime,
    "reference": _run_with_reference,
}


def run_test_with_backend(
    model_dir: str,
    backend: str,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Dict[str, Any]:
    """Run a single backend test against ``backend``.

    The returned dictionary has the following structure::

        {"success": bool, "error": str, "error_step": str}

    ``error_step`` is either ``"load"`` (failure when instantiating the
    backend session/evaluator), ``"run"`` (failure when executing the
    model) or ``"compare"`` (failure when comparing outputs).
    """
    factory = _BACKEND_FACTORIES.get(backend)
    if factory is None:
        return {
            "success": False,
            "error": "unknown backend: %s" % backend,
            "error_step": "load",
        }

    try:
        data_sets = _load_test_data_sets(model_dir)
    except Exception as exc:  # noqa: BLE001 - dataset corruption is a failure
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "load",
        }
    if not data_sets:
        return {
            "success": False,
            "error": "no test_data_set_* directory found",
            "error_step": "load",
        }

    try:
        runner = factory(model_dir)
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
    name: str, results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"name": name}
    for backend in BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        row[backend] = success
        error = _stringify_error(info.get("error"))
        if error:
            row["%s_error" % backend] = error
        step = info.get("error_step") or ""
        if step:
            row["%s_error_step" % backend] = step
    return row


def build_payload(
    kind: str = "node",
    limit: Optional[int] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    discover: Callable[[str], List[Dict[str, str]]] = discover_node_tests,
    run: Callable[..., Dict[str, Any]] = run_test_with_backend,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """Discover all tests, run them on every backend and return a payload."""
    if versions is None:
        versions = collect_versions
    tests = discover(kind)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    _log("Discovered %d %s backend tests." % (len(tests), kind))

    rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {
        backend: {"pass": 0, "fail": 0} for backend in BACKENDS
    }
    for idx, test in enumerate(tests):
        name = test["name"]
        model_dir = test["model_dir"]
        results: Dict[str, Dict[str, Any]] = {}
        for backend in BACKENDS:
            try:
                info = run(model_dir, backend, rtol=rtol, atol=atol)
            except Exception as exc:  # noqa: BLE001
                # Defensive guard: the runner is expected to capture its
                # own exceptions, but we never want a single broken test
                # to abort the whole snapshot.
                _log(
                    "Unhandled error for %s on %s: %s\n%s"
                    % (name, backend, exc, traceback.format_exc())
                )
                info = {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "run",
                }
            results[backend] = info
            bucket = "pass" if info.get("success") else "fail"
            totals[backend][bucket] += 1
        rows.append(_row_from_results(name, results))
        if (idx + 1) % 50 == 0:
            _log("Ran %d/%d tests." % (idx + 1, len(tests)))

    return {
        "date": _format_iso(now or dt.datetime.now(tz=dt.timezone.utc)),
        "kind": kind,
        "tolerances": {"rtol": rtol, "atol": atol},
        "versions": versions(),
        "totals": totals,
        "tests": rows,
    }


def write_payload(json_path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``json_path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


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
    try:
        payload = build_payload(
            kind=args.kind,
            limit=args.limit,
            rtol=args.rtol,
            atol=args.atol,
        )
    except Exception as exc:  # noqa: BLE001
        _log("ERROR: failed to record backend test coverage: %s" % exc)
        traceback.print_exc()
        return 1
    json_path = os.path.join(
        args.cache_dir, "onnx-light", "backend_test_coverage.json"
    )
    write_payload(json_path, payload)
    _log(
        "Wrote %d test entries to %s (totals=%s)."
        % (len(payload["tests"]), json_path, payload["totals"])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
