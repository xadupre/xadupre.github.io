"""Benchmark ``onnx-light`` vs ``onnxruntime`` on the backend test cases.

The script discovers every benchmark-sized backend node test bundled with
the installed ``onnx-light`` package and measures the processing time of
both ``onnxruntime`` and the ``onnx-light`` reference implementation
backed by its C++ ``KernelDispatchTable``.

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
# Test-discovery helpers (self-contained, no dependency on other scripts)
# ---------------------------------------------------------------------------


def _cc_tensor_to_numpy(tensor):
    """Convert a C++ backend-test tensor into a numpy value."""
    import numpy as np
    import onnx

    try:
        import ml_dtypes  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        ml_dtypes = None

    if int(tensor.data_type) == int(onnx.TensorProto.STRING):
        values = tensor.string_data()
        arr = np.array(values, dtype=object)
        return arr.reshape(tuple(int(d) for d in tensor.shape))

    dtype_map = {
        int(onnx.TensorProto.FLOAT): np.float32,
        int(onnx.TensorProto.DOUBLE): np.float64,
        int(onnx.TensorProto.INT32): np.int32,
        int(onnx.TensorProto.INT64): np.int64,
        int(onnx.TensorProto.UINT8): np.uint8,
        int(onnx.TensorProto.INT8): np.int8,
        int(onnx.TensorProto.BOOL): np.bool_,
        int(onnx.TensorProto.UINT16): np.uint16,
        int(onnx.TensorProto.INT16): np.int16,
        int(onnx.TensorProto.UINT32): np.uint32,
        int(onnx.TensorProto.UINT64): np.uint64,
        int(onnx.TensorProto.FLOAT16): np.float16,
    }
    if ml_dtypes is not None:
        optional_ml_dtypes = (
            (onnx.TensorProto.BFLOAT16, "bfloat16"),
            (onnx.TensorProto.FLOAT8E4M3FN, "float8_e4m3fn"),
            (onnx.TensorProto.FLOAT8E4M3FNUZ, "float8_e4m3fnuz"),
            (onnx.TensorProto.FLOAT8E5M2, "float8_e5m2"),
            (onnx.TensorProto.FLOAT8E5M2FNUZ, "float8_e5m2fnuz"),
            (onnx.TensorProto.FLOAT8E8M0, "float8_e8m0fnu"),
        )
        for onnx_type, attr_name in optional_ml_dtypes:
            dtype = getattr(ml_dtypes, attr_name, None)
            if dtype is not None:
                dtype_map[int(onnx_type)] = dtype

    dtype = dtype_map.get(int(tensor.data_type))
    if dtype is None:
        raise NotImplementedError(
            f"Cannot convert benchmark input tensor with data_type={tensor.data_type}."
        )
    arr = np.frombuffer(tensor.raw_data(), dtype=dtype)
    return arr.reshape(tuple(int(d) for d in tensor.shape))


def _cc_data_sets_to_python(test_case) -> List[Tuple[List[Any], List[Any]]]:
    """Convert C++ backend-test datasets into positional python inputs."""

    graph_inputs = list(test_case.model.graph.input)
    data_sets: List[Tuple[List[Any], List[Any]]] = []
    for ds in test_case.data_sets:
        by_name = {_tensor.name: _cc_tensor_to_numpy(_tensor) for _tensor in ds.inputs}
        maps_by_name = {m.name: m for m in ds.maps} if getattr(ds, "maps", None) else {}
        inputs: List[Any] = []
        for gi in graph_inputs:
            if gi.type.has_map_type():
                if gi.name in maps_by_name:
                    m = maps_by_name[gi.name]
                    inputs.append(_cc_tensor_to_numpy(m.keys))
                    inputs.append(_cc_tensor_to_numpy(m.values))
                    continue
                keys_arr = by_name.get(f"{gi.name}_keys")
                values_arr = by_name.get(f"{gi.name}_values")
                if keys_arr is None or values_arr is None:
                    inputs.append(by_name.get(gi.name))
                    continue
                inputs.append(keys_arr)
                inputs.append(values_arr)
                continue
            inputs.append(by_name.get(gi.name))
        data_sets.append((inputs, []))
    return data_sets


def _discover_benchmark_mode_tests(kind: str) -> Optional[List[Dict[str, Any]]]:
    """Discover benchmark-sized backend tests when onnx-light exposes them."""
    try:
        from onnx_light.onnx.backend import TestMode, collect_test_cases
    except (ImportError, AttributeError):
        return None

    try:
        cases = collect_test_cases(include_big=True, mode=TestMode.BENCHMARK)
    except TypeError:
        return None

    kinds = _normalize_kinds(kind)
    discovered: List[Dict[str, Any]] = []
    for tc in cases:
        name = getattr(tc, "name", "")
        if not name:
            continue
        case_kind = getattr(tc, "kind", None)
        if kinds and case_kind not in kinds:
            continue
        model = getattr(tc, "model", None)
        if model is None:
            continue
        onnx_model = _onnx_light_model_to_onnx(model)
        data_sets = _cc_data_sets_to_python(tc)
        if not data_sets:
            continue
        tag = getattr(tc, "tag", None) or ""
        discovered.append(
            {
                "name": str(name),
                "model": onnx_model,
                "data_sets": data_sets,
                "tag": str(tag),
            }
        )
    discovered.sort(key=lambda d: d["name"])
    return discovered


def _normalize_kinds(kind) -> Tuple[str, ...]:
    """Normalize a ``kind`` filter into a tuple of non-empty kind names."""
    if kind is None:
        return ()
    items: List[str] = []
    if isinstance(kind, str):
        items.extend(piece.strip() for piece in kind.split(","))
    else:
        for entry in kind:
            if entry is None:
                continue
            items.extend(piece.strip() for piece in str(entry).split(","))
    seen: Dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return tuple(seen)


def _onnx_light_model_to_onnx(model):
    """Convert an ``onnx-light`` ``ModelProto`` into an ``onnx`` ``ModelProto``."""
    import onnx

    if isinstance(model, onnx.ModelProto):
        return model
    out = onnx.ModelProto()
    out.ParseFromString(model.SerializeToString())
    return out


def _onnx_light_tensor_to_numpy(arr):
    """Convert an ``onnx-light`` tensor / numpy array to a numpy value."""
    import numpy as np

    if isinstance(arr, np.ndarray):
        return arr
    if isinstance(arr, (list, tuple)):
        return [_onnx_light_tensor_to_numpy(a) for a in arr]
    if hasattr(arr, "SerializeToString"):
        import onnx
        from onnx import numpy_helper

        content = arr.SerializeToString()
        proto_name = type(arr).__name__
        if proto_name == "SequenceProto":
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(content)
            return numpy_helper.to_list(sequence)
        if proto_name == "OptionalProto":
            optional = onnx.OptionalProto()
            optional.ParseFromString(content)
            return numpy_helper.to_optional(optional)
        tensor = onnx.TensorProto()
        tensor.ParseFromString(content)
        return numpy_helper.to_array(tensor)
    return np.asarray(arr)


def _load_proto(path: str, type_proto: Any = None):
    """Load a serialised proto from ``path`` as a numpy value."""
    import onnx
    from onnx import numpy_helper

    with open(path, "rb") as fh:
        content = fh.read()

    if type_proto is not None:
        if type_proto.HasField("sequence_type"):
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(content)
            return numpy_helper.to_list(sequence)
        if type_proto.HasField("optional_type"):
            optional = onnx.OptionalProto()
            optional.ParseFromString(content)
            return numpy_helper.to_optional(optional)

    tensor = onnx.TensorProto()
    tensor.ParseFromString(content)
    return numpy_helper.to_array(tensor)


def _load_test_data_sets(
    model_dir: str, model: Any = None
) -> List[Tuple[List[Any], List[Any]]]:
    """Return ``[(inputs, expected_outputs), ...]`` for ``model_dir``."""
    input_types: List[Any] = []
    output_types: List[Any] = []
    if model is not None:
        input_types = [inp.type for inp in model.graph.input]
        output_types = [out.type for out in model.graph.output]

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
            p = os.path.join(ds_path, f"input_{i}.pb")
            if not os.path.exists(p):
                break
            type_proto = input_types[i] if i < len(input_types) else None
            inputs.append(_load_proto(p, type_proto))
            i += 1
        outputs: List[Any] = []
        j = 0
        while True:
            p = os.path.join(ds_path, f"output_{j}.pb")
            if not os.path.exists(p):
                break
            type_proto = output_types[j] if j < len(output_types) else None
            outputs.append(_load_proto(p, type_proto))
            j += 1
        data_sets.append((inputs, outputs))
    return data_sets


def _model_input_names(model) -> List[str]:
    """Return the names of the graph inputs that are not initializers."""
    initializer_names = {init.name for init in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def build_graph(model) -> Dict[str, Any]:
    """Return an SVG rendering of ``model``'s graph."""
    from onnx_light.tools import to_svg

    return {"svg": to_svg(model)}


def discover_node_tests(kind: str = DEFAULT_KIND) -> List[Dict[str, Any]]:
    """Return ``[{"name", "model", "data_sets", "tag"}, ...]`` for every backend test.

    When available, benchmark-sized C++ backend tests are preferred so the
    dashboard measures the dedicated benchmark corpus rather than the
    standard correctness-only cases. Older ``onnx-light`` builds fall back
    to ``onnx_light.onnx_lib.backend.test.case.collect_test_case`` with
    ``include_big=True``.
    """
    benchmark_cases = _discover_benchmark_mode_tests(kind)
    if benchmark_cases is not None:
        return benchmark_cases

    from onnx_light.onnx_lib.backend.test.case import collect_test_case

    kinds = _normalize_kinds(kind)
    cases = collect_test_case(include_big=True)
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_kind = getattr(tc, "kind", None)
        if kinds and case_kind not in kinds:
            continue
        model = getattr(tc, "model", None)
        data_sets = getattr(tc, "data_sets", None) or []
        existing_dir = getattr(tc, "model_dir", None)
        if existing_dir and (model is None or not data_sets):
            import onnx

            if model is None:
                model = onnx.load(os.path.join(str(existing_dir), "model.onnx"))
            if not data_sets:
                data_sets = _load_test_data_sets(str(existing_dir), model)
        if model is None or not data_sets:
            continue
        onnx_model = _onnx_light_model_to_onnx(model)
        converted_data_sets: List[Tuple[List[Any], List[Any]]] = [
            (
                [_onnx_light_tensor_to_numpy(a) for a in inputs],
                [_onnx_light_tensor_to_numpy(a) for a in outputs],
            )
            for inputs, outputs in data_sets
        ]
        tag = getattr(tc, "tag", None) or ""
        discovered.append(
            {
                "name": str(name),
                "model": onnx_model,
                "data_sets": converted_data_sets,
                "tag": str(tag),
            }
        )
    discovered.sort(key=lambda d: d["name"])
    return discovered

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

    def _run(inputs: List[Any]) -> List[Any]:
        import numpy as np

        feeds: Dict[str, Any] = {}
        for name, value in zip(input_names, inputs):
            if isinstance(value, dict):
                map_keys_name = f"{name}_keys"
                map_values_name = f"{name}_values"
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
        if step:
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
    if (
        ort_ok
        and light_ok
        and ort_avg is not None
        and light_avg is not None
        and light_avg > 0
    ):
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
        r for r in rows if r.get("onnxruntime_success") and r.get("onnx_light_success")
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
