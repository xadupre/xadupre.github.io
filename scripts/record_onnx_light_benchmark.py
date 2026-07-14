"""Benchmark ``onnx-light`` vs ``onnxruntime`` on the backend test cases.

The script discovers every backend node test bundled with the installed
``onnx-light`` package (via
``onnx_light.onnx_lib.backend.test.case.collect_test_case``) and measures
the processing time of both ``onnxruntime`` and the ``onnx-light``
reference implementation backed by its C++ ``KernelDispatchTable``.

For each test the measurement protocol is:

1. Load / compile the model once (not timed).
2. Run :data:`N_WARMUP` iterations to prime the JIT / kernel cache.
3. Run :data:`N_MEASURE` iterations and record the wall-clock time of each.
4. Report the per-backend **average**, **min** and **max** execution time
   (in milliseconds) and the **speedup** defined as::

       speedup = onnxruntime_avg_ms / onnx_light_avg_ms

   A speedup > 1 indicates that ``onnx-light`` is faster than
   ``onnxruntime`` on that test case.

The resulting payload is persisted to
``cache_data/onnx-light/benchmark.json``. The dashboard at
``dashboard/onnx-light/benchmark.html`` renders that file as a sortable,
searchable table with per-row collapsible SVG graphs.

Usage::

    python scripts/record_onnx_light_benchmark.py [--cache-dir DIR]
        [--kind node] [--limit N] [--n-warmup N] [--n-measure N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Backends exercised by the benchmark. Only ``onnxruntime`` and
#: ``onnx_light`` are timed; the reference implementation is intentionally
#: omitted because it is Python-only and not representative of production
#: performance.
BENCHMARK_BACKENDS: Tuple[str, ...] = ("onnxruntime", "onnx_light")

BACKEND_PACKAGE: Dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "onnx_light": "onnx_light",
}

#: Default number of warm-up iterations (not timed) run before measurement.
N_WARMUP: int = 3

#: Default number of timed iterations used to compute the average time.
N_MEASURE: int = 10

DEFAULT_KIND: str = "node"


# ---------------------------------------------------------------------------
# Helpers shared with record_onnx_backend_test_coverage
# ---------------------------------------------------------------------------

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


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string representation of an error."""
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    max_len = 300
    if len(text) > max_len:
        head_len = 180
        ellipsis = " ... "
        tail_len = max_len - head_len - len(ellipsis)
        text = text[:head_len].rstrip() + ellipsis + text[-tail_len:].lstrip()
    return text


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "onnx_light", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


# ---------------------------------------------------------------------------
# Re-use discover helpers from the coverage script
# ---------------------------------------------------------------------------

# Import shared helpers from record_onnx_backend_test_coverage so there is
# a single source of truth for test discovery and model conversion.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from record_onnx_backend_test_coverage import (  # noqa: E402
    _onnx_light_model_to_onnx,
    _onnx_light_tensor_to_numpy,
    _model_input_names,
    _normalize_kinds,
    _load_test_data_sets,
    build_graph,
    discover_node_tests,
)


# ---------------------------------------------------------------------------
# Backend runners (load once, then call N times)
# ---------------------------------------------------------------------------

def _make_onnxruntime_runner(model) -> Callable[[List[Any]], List[Any]]:
    import onnxruntime

    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(sess.run(None, feeds))

    return _run


def _make_onnx_light_runner(model) -> Callable[[List[Any]], List[Any]]:
    from onnx_light.onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model.SerializeToString())
    input_names = _model_input_names(model)
    evaluator_input_names = getattr(evaluator, "input_names", None)

    def _run(inputs: List[Any]) -> List[Any]:
        import numpy as np

        feeds: Dict[str, Any] = {}
        for name, value in zip(input_names, inputs):
            map_keys_name = f"{name}_keys"
            map_values_name = f"{name}_values"
            if (
                isinstance(value, dict)
                and evaluator_input_names
                and map_keys_name in evaluator_input_names
                and map_values_name in evaluator_input_names
            ):
                items = list(value.items())
                feeds[map_keys_name] = np.asarray([k for k, _ in items])
                feeds[map_values_name] = np.asarray([v for _, v in items])
                continue
            feeds[name] = value
        return list(evaluator.run(None, feeds))

    return _run


_RUNNER_FACTORIES: Dict[str, Callable[[Any], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _make_onnxruntime_runner,
    "onnx_light": _make_onnx_light_runner,
}


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def run_benchmark(
    model: Any,
    data_sets: List[Tuple[List[Any], List[Any]]],
    backend: str,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
) -> Dict[str, Any]:
    """Run a benchmark for ``backend`` on ``model`` / ``data_sets``.

    Returns a dictionary with the following keys:

    * ``success`` – ``True`` when every iteration completed without error.
    * ``error`` – human-readable error string when ``success`` is ``False``.
    * ``error_step`` – ``"load"``, ``"warmup"`` or ``"measure"``.
    * ``avg_ms`` – average per-dataset execution time in milliseconds.
    * ``min_ms`` – minimum per-dataset execution time in milliseconds.
    * ``max_ms`` – maximum per-dataset execution time in milliseconds.
    * ``n_warmup`` – number of warm-up iterations actually run.
    * ``n_measure`` – number of timed iterations actually run.
    """
    factory = _RUNNER_FACTORIES.get(backend)
    if factory is None:
        return {
            "success": False,
            "error": f"unknown backend: {backend}",
            "error_step": "load",
        }

    if not data_sets:
        return {
            "success": False,
            "error": "no test_data_set found",
            "error_step": "load",
        }

    # --- Load / compile the model -------------------------------------------
    try:
        runner = factory(model)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "load",
        }

    # --- Warm-up iterations (not timed) -------------------------------------
    for _ in range(n_warmup):
        for inputs, _ in data_sets:
            try:
                runner(inputs)
            except Exception as exc:  # noqa: BLE001
                return {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "warmup",
                    "n_warmup": 0,
                    "n_measure": 0,
                }

    # --- Timed measurement iterations ---------------------------------------
    times_ms: List[float] = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        for inputs, _ in data_sets:
            try:
                runner(inputs)
            except Exception as exc:  # noqa: BLE001
                return {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "measure",
                    "n_warmup": n_warmup,
                    "n_measure": len(times_ms),
                }
        elapsed_ms = (time.perf_counter() - t0) * 1_000
        times_ms.append(elapsed_ms)

    if not times_ms:
        return {
            "success": False,
            "error": "no timing samples collected",
            "error_step": "measure",
            "n_warmup": n_warmup,
            "n_measure": 0,
        }

    avg_ms = sum(times_ms) / len(times_ms)
    return {
        "success": True,
        "error": "",
        "error_step": "",
        "avg_ms": round(avg_ms, 6),
        "min_ms": round(min(times_ms), 6),
        "max_ms": round(max(times_ms), 6),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
    }


def _row_from_results(
    name: str,
    results: Dict[str, Dict[str, Any]],
    tag: str = "",
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a dashboard row from per-backend benchmark results."""
    row: Dict[str, Any] = {"name": name}
    if tag:
        row["tag"] = tag
    if graph is not None:
        row["graph"] = graph

    for backend in BENCHMARK_BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        row[f"{backend}_success"] = success
        error = _stringify_error(info.get("error"))
        if error:
            row[f"{backend}_error"] = error
        step = info.get("error_step") or ""
        if step and step != "":
            row[f"{backend}_error_step"] = step
        for metric in ("avg_ms", "min_ms", "max_ms"):
            v = info.get(metric)
            if v is not None:
                row[f"{backend}_{metric}"] = v

    # Compute speedup = onnxruntime_avg_ms / onnx_light_avg_ms.
    # A value > 1 means onnx-light is faster than onnxruntime.
    ort_avg = results.get("onnxruntime", {}).get("avg_ms")
    light_avg = results.get("onnx_light", {}).get("avg_ms")
    ort_ok = results.get("onnxruntime", {}).get("success", False)
    light_ok = results.get("onnx_light", {}).get("success", False)
    if ort_ok and light_ok and ort_avg is not None and light_avg is not None and light_avg > 0:
        row["speedup"] = round(ort_avg / light_avg, 4)

    return row


# ---------------------------------------------------------------------------
# Top-level payload builder
# ---------------------------------------------------------------------------

def build_payload(
    kind: str = DEFAULT_KIND,
    limit: Optional[int] = None,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    discover: Callable[[str], List[Dict[str, Any]]] = discover_node_tests,
    run: Callable[..., Dict[str, Any]] = run_benchmark,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """Discover all tests, run benchmarks on every backend and return a payload."""
    if versions is None:
        versions = collect_versions

    tests = discover(kind)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    _log(f"Discovered {len(tests)} {kind!r} backend tests to benchmark.")

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)
    version_map = versions()

    rows: List[Dict[str, Any]] = []
    for idx, test in enumerate(tests):
        name = test["name"]
        model = test["model"]
        data_sets = test["data_sets"]
        tag = str(test.get("tag", "") or "")

        try:
            graph = build_graph(model)
        except Exception:  # noqa: BLE001
            graph = None

        results: Dict[str, Dict[str, Any]] = {}
        for backend in BENCHMARK_BACKENDS:
            try:
                info = run(
                    model,
                    data_sets,
                    backend,
                    n_warmup=n_warmup,
                    n_measure=n_measure,
                )
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

        rows.append(_row_from_results(name, results, tag=tag, graph=graph))

        if (idx + 1) % 50 == 0:
            _log(f"Benchmarked {idx + 1}/{len(tests)} tests.")

    # Summary stats across all tests that both backends succeeded on.
    both_ok = [
        r for r in rows
        if r.get("onnxruntime_success") and r.get("onnx_light_success")
    ]
    speedups = [r["speedup"] for r in both_ok if "speedup" in r]
    summary: Dict[str, Any] = {"both_succeeded": len(both_ok), "total": len(rows)}
    if speedups:
        summary["avg_speedup"] = round(sum(speedups) / len(speedups), 4)
        summary["min_speedup"] = round(min(speedups), 4)
        summary["max_speedup"] = round(max(speedups), 4)

    return {
        "date": now_iso,
        "kind": kind,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "versions": version_map,
        "summary": summary,
        "tests": rows,
    }


def write_payload(json_path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``json_path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_previous_payload(json_path: str) -> Dict[str, Any]:
    """Return the previously written payload, or an empty dict if absent."""
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the JSON cache (default: %(default)s).",
    )
    parser.add_argument(
        "--kind",
        default=DEFAULT_KIND,
        help=(
            "Backend test group(s) to benchmark (default: %(default)s). "
            "Accepts a single value or a comma-separated list."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of tests benchmarked (useful for debugging).",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=N_WARMUP,
        help=f"Number of warm-up iterations before timing (default: {N_WARMUP}).",
    )
    parser.add_argument(
        "--n-measure",
        type=int,
        default=N_MEASURE,
        help=f"Number of timed iterations per test (default: {N_MEASURE}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(args.cache_dir, "onnx-light", "benchmark.json")
    try:
        payload = build_payload(
            kind=args.kind,
            limit=args.limit,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record benchmark: {exc}")
        traceback.print_exc()
        return 1
    write_payload(json_path, payload)
    summary = payload.get("summary", {})
    _log(
        f"Wrote {len(payload['tests'])} test entries to {json_path} "
        f"(summary={summary})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
