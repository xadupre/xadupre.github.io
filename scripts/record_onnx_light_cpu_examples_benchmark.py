"""Benchmark ``onnxruntime`` vs ``onnx-light-cpu`` on the onnx-light-cpu examples.

The ``onnx-light-cpu`` documentation ships a gallery of runnable benchmark
examples that compare the SIMD-accelerated CPU kernels against
``onnxruntime`` (and, for context, plain ``numpy`` and ``onnx-light``'s own
un-accelerated reference kernels). This script reproduces those benchmark
examples so their results can be published on a dashboard rather than only
living as static images in the rendered gallery.

The dashboard records five ``onnxruntime`` vs ``onnx-light-cpu`` benchmarks:

``abs`` (``plot_abs_benchmark.py``)
    A single ``Abs`` node evaluated on a 1-D ``float32`` array whose length
    sweeps from a hundred to a hundred million elements.

``gemm`` (``plot_gemm_benchmark.py``)
    A single ``Gemm`` node computing ``Y = A @ B`` for square ``float32``
    matrices of increasing size.

``exp``, ``log`` and ``not``
    Single-node vector benchmarks for the additional elementwise kernels
    accelerated by ``onnx-light-cpu``. ``Exp`` uses bounded ``float32`` values,
    ``Log`` uses strictly positive ``float32`` values and ``Not`` uses boolean
    values.

For every example and every input size the script measures the wall-clock
time of:

* **onnxruntime** - the same single-node ONNX model run through an
  ``onnxruntime`` ``InferenceSession`` (CPU execution provider).
* **onnx-light-cpu** - the same model evaluated by ``onnx-light``'s
  ``ReferenceEvaluator`` after :func:`onnx_light_cpu.register_kernels` has
  installed the SIMD-accelerated kernels into onnx-light's shared C++ dispatch
  table, so every matching node dispatches to the SIMD kernel.
* **onnx-light** - onnx-light's own un-accelerated reference kernel, measured
  *before* ``register_kernels()`` is called (registration is process-wide and
  irreversible), as a baseline for what onnx-light-cpu adds on top of it.
* **numpy** - the equivalent :mod:`numpy` operation, as a reference baseline.

Each measurement runs :data:`N_WARMUP` untimed warm-up calls and then retains
the median of :data:`N_MEASURE` timed repetitions, mirroring the ``measure``
helper the examples use. The headline metric is
``speedup_cpu = onnxruntime_ms / onnx_light_cpu_ms``: a value above ``1`` means
onnx-light-cpu is faster than onnxruntime on that input.

The resulting payload is persisted to
``cache_data/onnx-light-cpu/examples_benchmark.json`` and rendered by
``dashboard/onnx-light-cpu/examples-benchmark.html``.

Usage::

    python scripts/record_onnx_light_cpu_examples_benchmark.py [--cache-dir DIR]
        [--n-warmup N] [--n-measure N] [--max-abs-size N]
        [--max-unary-size N] [--max-gemm-size N]
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

#: Backends reported by the dashboard. ``onnxruntime`` and ``onnx_light_cpu``
#: are the headline comparison; ``onnx_light`` (built-in) and ``numpy`` are
#: included as context, exactly like the gallery examples.
BENCHMARK_BACKENDS: Tuple[str, ...] = (
    "numpy",
    "onnx_light",
    "onnx_light_cpu",
    "onnxruntime",
)

#: Default number of untimed warm-up calls run before each measurement.
N_WARMUP: int = 3

#: Default number of timed repetitions whose median is retained.
N_MEASURE: int = 10

#: SIMD level names reported by ``onnx-light-cpu`` (``detect_simd_level``).
_SIMD_NAMES: Dict[int, str] = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}


# ---------------------------------------------------------------------------
# Small helpers
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


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "onnx_light", "onnx_light_cpu", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def measure(func: Callable[[], Any], repeat: int, warmup: int = N_WARMUP) -> float:
    """Return the median wall-clock time (seconds) of ``func`` over ``repeat`` calls.

    ``warmup`` untimed calls prime any caches first. This mirrors the
    ``measure`` helper used by the onnx-light-cpu gallery examples.
    """
    import numpy as np

    for _ in range(max(0, warmup)):
        func()
    timings: List[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings))


def measure_together(
    funcs: Tuple[Callable[[], Any], ...],
    repeat: int,
    warmup: int = N_WARMUP,
) -> Tuple[float, ...]:
    """Measure callables in a rotating order and return their median times."""
    import numpy as np

    timings: Tuple[List[float], ...] = tuple([] for _ in funcs)
    for iteration in range(max(0, warmup)):
        for offset in range(len(funcs)):
            funcs[(iteration + offset) % len(funcs)]()
    for iteration in range(max(1, repeat)):
        for offset in range(len(funcs)):
            index = (iteration + offset) % len(funcs)
            start = time.perf_counter()
            funcs[index]()
            timings[index].append(time.perf_counter() - start)
    return tuple(float(np.median(values)) for values in timings)


# ---------------------------------------------------------------------------
# Example (benchmark scenario) definitions
# ---------------------------------------------------------------------------


def _make_abs_model():
    """Return the single-node ``Abs`` model used by ``plot_abs_benchmark``."""
    from onnx_light.onnx import TensorProto, checker, helper

    graph = helper.make_graph(
        [helper.make_node("Abs", ["X"], ["Y"])],
        "abs_bench",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


def _make_unary_model(op_type: str, boolean: bool = False):
    """Return a single-node unary model with a dynamic vector input."""
    from onnx_light.onnx import TensorProto, checker, helper

    dtype = TensorProto.BOOL if boolean else TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node(op_type, ["X"], ["Y"])],
        f"{op_type.lower()}_bench",
        [helper.make_tensor_value_info("X", dtype, ["N"])],
        [helper.make_tensor_value_info("Y", dtype, ["N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


def _make_gemm_model():
    """Return the single-node ``Gemm`` model used by ``plot_gemm_benchmark``."""
    from onnx_light.onnx import TensorProto, checker, helper

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=1.0, beta=1.0)],
        "gemm_bench",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", "K"]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, ["K", "N"]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


def _abs_example(max_size: Optional[int] = None) -> Dict[str, Any]:
    """Return the ``abs`` benchmark scenario description.

    ``max_size`` optionally caps the size grid so the benchmark stays within a
    time budget (useful for debugging / CI); ``None`` keeps the example's full
    grid.
    """
    import numpy as np

    size_grid = [10**k for k in range(2, 9)]
    if max_size is not None:
        size_grid = [s for s in size_grid if s <= max_size] or size_grid[:1]

    rng = np.random.default_rng(0)

    def make_inputs(size: int) -> Dict[str, Any]:
        inp = rng.uniform(-100.0, 100.0, size=size).astype(np.float32)
        return {"X": inp}

    def numpy_op(feeds: Dict[str, Any]) -> Any:
        return np.abs(feeds["X"])

    def repeat_for(size: int) -> int:
        return max(7, min(200, 2_000_000 // size))

    return {
        "name": "abs",
        "title": "Abs: onnxruntime vs onnx-light-cpu",
        "op": "Abs",
        "source": "plot_abs_benchmark.py",
        "xlabel": "array size (elements)",
        "size_key": "size",
        "make_model": _make_abs_model,
        "size_grid": size_grid,
        # ``onnx-light`` built-in is measured across the whole grid for Abs.
        "builtin_sizes": list(size_grid),
        "make_inputs": make_inputs,
        "numpy_op": numpy_op,
        "repeat_for": repeat_for,
        "kernel_name": "onnx_light_cpu::Abs",
    }


def _gemm_example(max_size: Optional[int] = None) -> Dict[str, Any]:
    """Return the ``gemm`` benchmark scenario description."""
    import numpy as np

    size_grid = [16, 32, 64, 128, 256, 512]
    if max_size is not None:
        size_grid = [s for s in size_grid if s <= max_size] or size_grid[:1]

    # The built-in reference Gemm kernel grows much faster than the other
    # back-ends, so the example only measures it for all but the two largest
    # sizes to keep the benchmark's runtime reasonable.
    builtin_sizes = size_grid[:-2] if len(size_grid) > 2 else list(size_grid)

    rng = np.random.default_rng(0)

    def make_inputs(size: int) -> Dict[str, Any]:
        a = rng.standard_normal((size, size)).astype(np.float32)
        b = rng.standard_normal((size, size)).astype(np.float32)
        return {"A": a, "B": b}

    def numpy_op(feeds: Dict[str, Any]) -> Any:
        return feeds["A"] @ feeds["B"]

    def repeat_for(size: int) -> int:
        return max(7, min(100, 20_000_000 // (size * size * size)))

    return {
        "name": "gemm",
        "title": "Gemm: onnxruntime vs onnx-light-cpu",
        "op": "Gemm",
        "source": "plot_gemm_benchmark.py",
        "xlabel": "matrix size N (N x N)",
        "size_key": "size",
        "make_model": _make_gemm_model,
        "size_grid": size_grid,
        "builtin_sizes": builtin_sizes,
        "make_inputs": make_inputs,
        "numpy_op": numpy_op,
        "repeat_for": repeat_for,
        "kernel_name": "onnx_light_cpu::Gemm",
    }


def _unary_example(
    name: str, op_type: str, max_size: Optional[int] = None
) -> Dict[str, Any]:
    """Return an elementwise unary benchmark scenario description."""
    import numpy as np

    size_grid = [10**k for k in range(2, 9)]
    if max_size is not None:
        size_grid = [s for s in size_grid if s <= max_size] or size_grid[:1]

    rng = np.random.default_rng(0)

    def make_inputs(size: int) -> Dict[str, Any]:
        if op_type == "Exp":
            values = rng.uniform(-10.0, 10.0, size=size).astype(np.float32)
        elif op_type == "Log":
            values = rng.uniform(1e-4, 100.0, size=size).astype(np.float32)
        else:
            values = rng.integers(0, 2, size=size).astype(np.bool_)
        return {"X": values}

    def numpy_op(feeds: Dict[str, Any]) -> Any:
        if op_type == "Exp":
            return np.exp(feeds["X"])
        if op_type == "Log":
            return np.log(feeds["X"])
        return np.logical_not(feeds["X"])

    def repeat_for(size: int) -> int:
        return max(7, min(200, 2_000_000 // size))

    return {
        "name": name,
        "title": f"{op_type}: onnxruntime vs onnx-light-cpu",
        "op": op_type,
        "source": "record_onnx_light_cpu_examples_benchmark.py",
        "xlabel": "array size (elements)",
        "size_key": "size",
        "make_model": lambda: _make_unary_model(op_type, op_type == "Not"),
        "size_grid": size_grid,
        "builtin_sizes": list(size_grid),
        "make_inputs": make_inputs,
        "numpy_op": numpy_op,
        "repeat_for": repeat_for,
        "kernel_name": f"onnx_light_cpu::{op_type}",
    }


def default_examples(
    max_abs_size: Optional[int] = None,
    max_gemm_size: Optional[int] = None,
    max_unary_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the benchmark scenarios displayed on the examples dashboard."""
    return [
        _abs_example(max_abs_size),
        _unary_example("exp", "Exp", max_unary_size),
        _unary_example("log", "Log", max_unary_size),
        _unary_example("not", "Not", max_unary_size),
        _gemm_example(max_gemm_size),
    ]


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------


def _make_onnxruntime_runner(model) -> Callable[[Dict[str, Any]], Any]:
    import onnxruntime

    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )

    def _run(feeds: Dict[str, Any]) -> Any:
        return sess.run(None, feeds)

    return _run


def _make_reference_runner(model) -> Callable[[Dict[str, Any]], Any]:
    """Return a runner backed by onnx-light's ``ReferenceEvaluator``.

    Whether the SIMD-accelerated onnx-light-cpu kernels are used depends solely
    on whether :func:`onnx_light_cpu.register_kernels` has been called
    (process-wide) before the evaluator's first ``run``; the caller controls
    that ordering.
    """
    from onnx_light.onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)

    def _run(feeds: Dict[str, Any]) -> Any:
        return evaluator.run(None, feeds)

    return _run


def detect_simd() -> Tuple[Optional[int], str]:
    """Return ``(level, name)`` for the CPU SIMD level onnx-light-cpu detects."""
    from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level

    level = int(detect_simd_level())
    return level, _SIMD_NAMES.get(level, str(level))


def run_examples(
    examples: List[Dict[str, Any]],
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    measure_fn: Callable[..., float] = measure,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run every scenario and return ``(example_results, meta)``.

    ``register_kernels()`` overrides onnx-light's process-wide dispatch table
    irreversibly, so the un-accelerated ``onnx_light`` baseline is primed for
    every example *first*. After registration, cached built-in sessions are
    measured together with onnx-light-cpu / onnxruntime / numpy in rotating
    order so the speed-up ratios compare equivalent samples.
    """
    from onnx_light_cpu import (
        clear_used_kernel_names,
        register_kernels,
        used_kernel_names,
    )
    from onnx_light_cpu.onnx_py._cpuregister import set_kernel_usage_recording

    meta: Dict[str, Any] = {}
    level, simd_name = detect_simd()
    meta["simd_level"] = level
    meta["simd_name"] = simd_name

    # --- Pass 1: prime onnx-light built-in before registration --------------
    builtin_runners: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    for example in examples:
        name = example["name"]
        model = example["make_model"]()
        runner = _make_reference_runner(model)
        builtin_runners[name] = runner
        if example["builtin_sizes"]:
            runner(example["make_inputs"](example["builtin_sizes"][0]))
        _log(f"Primed onnx-light built-in baseline for {name!r}.")

    # --- Register the SIMD kernels once (process-wide, irreversible) --------
    register_kernels()

    # --- Pass 2: onnxruntime, onnx-light-cpu and numpy ----------------------
    results: List[Dict[str, Any]] = []
    for example in examples:
        name = example["name"]
        model = example["make_model"]()
        cpu_runner = _make_reference_runner(model)
        ort_runner = _make_onnxruntime_runner(example["make_model"]())

        # Confirm the model dispatches to the onnx-light-cpu kernel rather than
        # onnx-light's built-in kernel before timing it.
        probe = example["make_inputs"](example["size_grid"][0])
        clear_used_kernel_names()
        cpu_runner(probe)
        used = list(used_kernel_names())
        expected_kernel = example.get("kernel_name")
        if expected_kernel is not None and used != [expected_kernel]:
            raise RuntimeError(
                f"{name}: expected dispatch to {expected_kernel!r} but recorded {used!r}"
            )
        # Usage recording takes a mutex on every call; disable it for timing.
        set_kernel_usage_recording(False)

        rows: List[Dict[str, Any]] = []
        for size in example["size_grid"]:
            feeds = example["make_inputs"](size)
            repeat = example["repeat_for"](size)

            funcs = (
                lambda feeds=feeds, numpy_op=example["numpy_op"]: numpy_op(feeds),
                lambda feeds=feeds, runner=cpu_runner: runner(feeds),
                lambda feeds=feeds, runner=ort_runner: runner(feeds),
            )
            has_builtin = size in example["builtin_sizes"]
            if has_builtin:
                builtin_runner = builtin_runners[name]
                funcs += (
                    lambda feeds=feeds, runner=builtin_runner: runner(feeds),
                )
            measured = measure_together(funcs, repeat, warmup=n_warmup)
            numpy_ms, cpu_ms, ort_ms = (value * 1e3 for value in measured[:3])

            times: Dict[str, Optional[float]] = {
                "numpy": numpy_ms,
                "onnx_light_cpu": cpu_ms,
                "onnxruntime": ort_ms,
            }
            times["onnx_light"] = measured[3] * 1e3 if has_builtin else None
            rows.append(_row_from_times(size, times))

        set_kernel_usage_recording(True)

        results.append(
            {
                "name": name,
                "title": example["title"],
                "op": example["op"],
                "source": example["source"],
                "xlabel": example["xlabel"],
                "size_key": example["size_key"],
                "backends": list(BENCHMARK_BACKENDS),
                "rows": rows,
                "summary": _summarize_example(rows),
            }
        )
        _log(f"Benchmarked example {name!r} on {len(rows)} sizes.")

    return results, meta


# ---------------------------------------------------------------------------
# Row / summary builders (pure, unit-tested)
# ---------------------------------------------------------------------------


def _round_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _row_from_times(size: int, times: Dict[str, Optional[float]]) -> Dict[str, Any]:
    """Build a dashboard row from per-backend millisecond timings for one size."""
    row: Dict[str, Any] = {"size": int(size)}
    for backend in BENCHMARK_BACKENDS:
        value = times.get(backend)
        if value is not None:
            row[f"{backend}_ms"] = _round_ms(value)

    ort = times.get("onnxruntime")
    cpu = times.get("onnx_light_cpu")
    if ort is not None and cpu is not None and cpu > 0:
        # speedup_cpu > 1 means onnx-light-cpu is faster than onnxruntime.
        row["speedup_cpu"] = round(ort / cpu, 4)
    return row


def _summarize_example(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the onnxruntime vs onnx-light-cpu speed-ups across sizes."""
    speedups = [r["speedup_cpu"] for r in rows if r.get("speedup_cpu") is not None]
    summary: Dict[str, Any] = {"sizes": len(rows), "cpu_succeeded": len(speedups)}
    if speedups:
        summary["avg_speedup_cpu"] = round(sum(speedups) / len(speedups), 4)
        summary["min_speedup_cpu"] = round(min(speedups), 4)
        summary["max_speedup_cpu"] = round(max(speedups), 4)
    return summary


# ---------------------------------------------------------------------------
# Top-level payload builder
# ---------------------------------------------------------------------------


def build_payload(
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    max_abs_size: Optional[int] = None,
    max_gemm_size: Optional[int] = None,
    max_unary_size: Optional[int] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    run: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]] = run_examples,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """Run every example benchmark and return the dashboard payload."""
    if versions is None:
        versions = collect_versions
    if examples is None:
        examples = default_examples(max_abs_size, max_gemm_size, max_unary_size)

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)

    example_results, meta = run(
        examples, n_warmup=n_warmup, n_measure=n_measure
    )

    payload: Dict[str, Any] = {
        "date": now_iso,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "versions": versions(),
        "examples": example_results,
    }
    payload.update(meta)
    return payload


def write_payload(json_path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``json_path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


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
        "--n-warmup",
        type=int,
        default=N_WARMUP,
        help=f"Number of warm-up iterations before timing (default: {N_WARMUP}).",
    )
    parser.add_argument(
        "--n-measure",
        type=int,
        default=N_MEASURE,
        help=f"Number of timed repetitions per size (default: {N_MEASURE}).",
    )
    parser.add_argument(
        "--max-abs-size",
        type=int,
        default=None,
        help="Optionally cap the largest Abs array size benchmarked.",
    )
    parser.add_argument(
        "--max-gemm-size",
        type=int,
        default=None,
        help="Optionally cap the largest Gemm matrix size benchmarked.",
    )
    parser.add_argument(
        "--max-unary-size",
        type=int,
        default=None,
        help="Optionally cap the largest Exp, Log and Not array size benchmarked.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(
        args.cache_dir, "onnx-light-cpu", "examples_benchmark.json"
    )
    try:
        payload = build_payload(
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
            max_abs_size=args.max_abs_size,
            max_gemm_size=args.max_gemm_size,
            max_unary_size=args.max_unary_size,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record examples benchmark: {exc}")
        traceback.print_exc()
        return 1
    write_payload(json_path, payload)
    n_examples = len(payload.get("examples", []))
    _log(f"Wrote {n_examples} example benchmark(s) to {json_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
