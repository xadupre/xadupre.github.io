"""Benchmark ``onnx-light-cpu``'s ``Gemm`` across dtypes and kernel code paths.

The ``onnx-light-cpu`` documentation ships a gallery example,
``plot_gemm_dtype_benchmark.py``, that measures the SIMD-accelerated ``Gemm``
kernel for its three supported element types -- ``float32``, ``float16`` and
``bfloat16`` -- on one representative shape per internal code path
(``single-tile``, ``K-chunked``, ``multi-panel`` and ``skinny-M/wide-N``). This
is the *second* onnx-light-cpu benchmark reproduced for publication on a
dashboard, alongside
``scripts/record_onnx_light_cpu_examples_benchmark.py`` (the ``Abs`` / ``Gemm``
size-sweep gallery examples).

For every shape the script measures the wall-clock time of:

* **onnx-light-cpu** -- onnx-light's ``ReferenceEvaluator`` after
  :func:`onnx_light_cpu.register_kernels` installs the SIMD-accelerated kernels
  into onnx-light's shared C++ dispatch table, for ``float32``, ``float16`` and
  ``bfloat16``. ``float16``/``bfloat16`` have no dedicated micro-kernel: the
  kernel widens them to ``float32``, reuses the ``float32`` SIMD routine and
  rounds back, so their time is the ``float32`` compute plus that overhead.
* **onnxruntime** -- the same single-node model run through an ``onnxruntime``
  ``InferenceSession`` (CPU execution provider) for ``float32`` and
  ``float16``. onnxruntime's CPU provider does not implement ``Gemm`` for
  ``bfloat16``, so no ``bfloat16`` result is recorded for it.
* **onnx-light** -- onnx-light's own un-accelerated reference ``Gemm`` kernel,
  measured *before* ``register_kernels()`` (registration is process-wide and
  irreversible), for ``float32`` only and only on the two lighter shapes
  (``single-tile`` and ``K-chunked``): it is dramatically slower on the heavier
  shapes and would dwarf every other measurement.

Each measurement runs :data:`N_WARMUP` untimed warm-up calls and then retains
the median of :data:`N_MEASURE` timed repetitions, mirroring the ``measure``
helper the example uses. The headline metric is
``speedup_cpu = onnxruntime_float32_ms / onnx_light_cpu_float32_ms``: a value
above ``1`` means onnx-light-cpu is faster than onnxruntime on that shape.

The resulting payload is persisted to
``cache_data/onnx-light-cpu/dtype_benchmark.json`` and rendered by
``dashboard/onnx-light-cpu/dtype-benchmark.html``.

Usage::

    python scripts/record_onnx_light_cpu_dtype_benchmark.py [--cache-dir DIR]
        [--n-warmup N] [--n-measure N] [--max-size N]
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

#: Element types the onnx-light-cpu ``Gemm`` kernel supports.
CPU_DTYPES: Tuple[str, ...] = ("float32", "float16", "bfloat16")

#: Element types ``onnxruntime``'s CPU ``Gemm`` implements (no ``bfloat16``).
ORT_DTYPES: Tuple[str, ...] = ("float32", "float16")

#: The un-accelerated ``onnx-light`` built-in reference kernel only supports
#: ``float32``.
BUILTIN_DTYPE: str = "float32"

#: Ordered ``(key, label)`` pairs for the per-shape millisecond columns the
#: dashboard renders, one per backend/dtype combination.
SERIES: Tuple[Tuple[str, str], ...] = (
    ("onnx_light_cpu_float32", "onnx-light-cpu float32"),
    ("onnx_light_cpu_float16", "onnx-light-cpu float16"),
    ("onnx_light_cpu_bfloat16", "onnx-light-cpu bfloat16"),
    ("onnxruntime_float32", "onnxruntime float32"),
    ("onnxruntime_float16", "onnxruntime float16"),
    ("onnx_light_float32", "onnx-light (built-in) float32"),
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
    for name in ("onnx", "onnxruntime", "onnx_light", "onnx_light_cpu", "numpy", "ml_dtypes"):
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
# Shape (kernel code path) definitions
# ---------------------------------------------------------------------------

#: One representative ``(label, M, N, K)`` shape per ``Gemm`` code path. The
#: labels mirror ``plot_gemm_dtype_benchmark.py``; ``kGemmTileM == 64``,
#: ``kGemmTileN == 256`` and ``kGemmTileK == 256`` so every shape sits clearly
#: on one side of those thresholds.
_DEFAULT_SHAPES: Tuple[Tuple[str, int, int, int], ...] = (
    ("single-tile", 64, 64, 64),
    ("K-chunked", 32, 32, 2048),
    ("multi-panel", 512, 512, 128),
    ("skinny-M/wide-N", 4, 4096, 128),
)

#: The ``onnx-light`` built-in reference kernel is only measured on these two
#: lighter shapes; it is far too slow on the heavier ones for a fair
#: comparison.
_BUILTIN_SHAPE_LABELS: Tuple[str, ...] = ("single-tile", "K-chunked")


def default_shapes(max_size: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return the ``Gemm`` shapes benchmarked, one per kernel code path.

    ``max_size`` optionally drops shapes whose largest dimension exceeds it so
    the benchmark stays within a time budget (useful for debugging / CI);
    ``None`` keeps the example's full set. At least one shape is always kept.
    """
    shapes: List[Dict[str, Any]] = []
    for label, m, n, k in _DEFAULT_SHAPES:
        if max_size is not None and max(m, n, k) > max_size:
            continue
        shapes.append(
            {
                "label": label,
                "M": m,
                "N": n,
                "K": k,
                "has_builtin": label in _BUILTIN_SHAPE_LABELS,
            }
        )
    if not shapes:
        label, m, n, k = _DEFAULT_SHAPES[0]
        shapes.append(
            {
                "label": label,
                "M": m,
                "N": n,
                "K": k,
                "has_builtin": label in _BUILTIN_SHAPE_LABELS,
            }
        )
    return shapes


def repeat_for(m: int, n: int, k: int) -> int:
    """Return the timed-repetition count for a ``(M, N, K)`` shape."""
    return max(7, min(50, 200_000_000 // (m * n * k + 1)))


# ---------------------------------------------------------------------------
# Model / input builders
# ---------------------------------------------------------------------------


def _tensor_proto_dtype(label: str):
    """Return the ``TensorProto`` element type for a dtype ``label``."""
    from onnx_light.onnx import TensorProto

    return {
        "float32": TensorProto.FLOAT,
        "float16": TensorProto.FLOAT16,
        "bfloat16": TensorProto.BFLOAT16,
    }[label]


def _np_dtype(label: str):
    """Return the numpy / ``ml_dtypes`` dtype for a dtype ``label``."""
    import ml_dtypes
    import numpy as np

    return {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": ml_dtypes.bfloat16,
    }[label]


def _make_gemm_model(label: str):
    """Return the single-node ``Gemm`` model for one dtype ``label``."""
    from onnx_light.onnx import checker, helper

    tp = _tensor_proto_dtype(label)
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["A", "B"], ["Y"], alpha=1.0, beta=1.0)],
        "gemm_dtype_bench",
        [
            helper.make_tensor_value_info("A", tp, ["M", "K"]),
            helper.make_tensor_value_info("B", tp, ["K", "N"]),
        ],
        [helper.make_tensor_value_info("Y", tp, ["M", "N"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    checker.check_model(model)
    return model


def _make_inputs(label: str, m: int, n: int, k: int) -> Dict[str, Any]:
    """Return random ``A``/``B`` feeds of dtype ``label`` for a shape."""
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)
    np_dtype = _np_dtype(label)
    return {"A": a.astype(np_dtype), "B": b.astype(np_dtype)}


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


def run_benchmark(
    shapes: List[Dict[str, Any]],
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    measure_fn: Callable[..., float] = measure,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run every shape and return ``(rows, meta)``.

    ``register_kernels()`` overrides onnx-light's process-wide dispatch table
    irreversibly, so the un-accelerated ``onnx_light`` built-in baseline is
    measured for every eligible shape *first*, then the kernels are registered
    once, then the onnx-light-cpu / onnxruntime curves are measured.
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

    # --- Pass 1: onnx-light built-in float32 baseline (before registration) --
    builtin_times: Dict[str, float] = {}
    builtin_model = _make_gemm_model(BUILTIN_DTYPE)
    for shape in shapes:
        if not shape["has_builtin"]:
            continue
        label, m, n, k = shape["label"], shape["M"], shape["N"], shape["K"]
        runner = _make_reference_runner(builtin_model)
        feeds = _make_inputs(BUILTIN_DTYPE, m, n, k)
        repeat = repeat_for(m, n, k)
        builtin_times[label] = measure_fn(
            lambda feeds=feeds, runner=runner: runner(feeds), repeat, warmup=n_warmup
        )
        _log(f"Measured onnx-light built-in baseline for shape {label!r}.")

    # --- Register the SIMD kernels once (process-wide, irreversible) ---------
    register_kernels()

    # --- Pass 2: onnx-light-cpu (all dtypes) and onnxruntime -----------------
    cpu_runners = {label: _make_reference_runner(_make_gemm_model(label)) for label in CPU_DTYPES}
    ort_runners = {label: _make_onnxruntime_runner(_make_gemm_model(label)) for label in ORT_DTYPES}

    # Confirm the onnx-light-cpu path dispatches to the SIMD ``Gemm`` kernel
    # (identified by the library-qualified name it records) before timing it.
    probe = _make_inputs("float32", 2, 2, 2)
    clear_used_kernel_names()
    cpu_runners["float32"](probe)
    used = list(used_kernel_names())
    if used != ["onnx_light_cpu::Gemm"]:
        raise RuntimeError(
            f"expected dispatch to 'onnx_light_cpu::Gemm' but recorded {used!r}"
        )
    # Usage recording takes a mutex on every call; disable it for timing.
    set_kernel_usage_recording(False)

    rows: List[Dict[str, Any]] = []
    for shape in shapes:
        label, m, n, k = shape["label"], shape["M"], shape["N"], shape["K"]
        repeat = repeat_for(m, n, k)
        times: Dict[str, Optional[float]] = {}
        feeds_by_dtype = {
            dtype: _make_inputs(dtype, m, n, k) for dtype in CPU_DTYPES
        }
        keys = tuple(
            [f"onnx_light_cpu_{dtype}" for dtype in CPU_DTYPES]
            + [f"onnxruntime_{dtype}" for dtype in ORT_DTYPES]
        )
        funcs = tuple(
            [
                (
                    lambda runner=cpu_runners[dtype],
                    feeds=feeds_by_dtype[dtype]: runner(feeds)
                )
                for dtype in CPU_DTYPES
            ]
            + [
                (
                    lambda runner=ort_runners[dtype],
                    feeds=feeds_by_dtype[dtype]: runner(feeds)
                )
                for dtype in ORT_DTYPES
            ]
        )
        measured = measure_together(funcs, repeat, warmup=n_warmup)
        times.update(
            {
                key: elapsed * 1e3
                for key, elapsed in zip(keys, measured, strict=True)
            }
        )

        builtin = builtin_times.get(label)
        times["onnx_light_float32"] = builtin * 1e3 if builtin is not None else None

        rows.append(_row_from_times(shape, times))
        _log(f"Benchmarked shape {label!r} ({m}x{n}x{k}).")

    set_kernel_usage_recording(True)

    return rows, meta


# ---------------------------------------------------------------------------
# Row / summary builders (pure, unit-tested)
# ---------------------------------------------------------------------------


def _round_ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _row_from_times(
    shape: Dict[str, Any], times: Dict[str, Optional[float]]
) -> Dict[str, Any]:
    """Build a dashboard row from per-series millisecond timings for one shape."""
    row: Dict[str, Any] = {
        "shape": shape["label"],
        "M": int(shape["M"]),
        "N": int(shape["N"]),
        "K": int(shape["K"]),
    }
    for key, _label in SERIES:
        value = times.get(key)
        if value is not None:
            row[f"{key}_ms"] = _round_ms(value)

    ort = times.get("onnxruntime_float32")
    cpu = times.get("onnx_light_cpu_float32")
    if ort is not None and cpu is not None and cpu > 0:
        # speedup_cpu > 1 means onnx-light-cpu is faster than onnxruntime.
        row["speedup_cpu"] = round(ort / cpu, 4)
    return row


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the float32 onnxruntime vs onnx-light-cpu speed-ups."""
    speedups = [r["speedup_cpu"] for r in rows if r.get("speedup_cpu") is not None]
    summary: Dict[str, Any] = {"shapes": len(rows), "cpu_succeeded": len(speedups)}
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
    max_size: Optional[int] = None,
    shapes: Optional[List[Dict[str, Any]]] = None,
    run: Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]] = run_benchmark,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
    """Run the dtype benchmark and return the dashboard payload."""
    if versions is None:
        versions = collect_versions
    if shapes is None:
        shapes = default_shapes(max_size)

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)

    rows, meta = run(shapes, n_warmup=n_warmup, n_measure=n_measure)

    payload: Dict[str, Any] = {
        "date": now_iso,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "versions": versions(),
        "title": "Gemm: float32 vs float16 vs bfloat16 across kernel code paths",
        "op": "Gemm",
        "source": "plot_gemm_dtype_benchmark.py",
        "series": [{"key": key, "label": label} for key, label in SERIES],
        "rows": rows,
        "summary": _summarize(rows),
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
        help=f"Number of timed repetitions per shape (default: {N_MEASURE}).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optionally drop shapes whose largest dimension exceeds this value.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(
        args.cache_dir, "onnx-light-cpu", "dtype_benchmark.json"
    )
    try:
        payload = build_payload(
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
            max_size=args.max_size,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record dtype benchmark: {exc}")
        traceback.print_exc()
        return 1
    write_payload(json_path, payload)
    n_rows = len(payload.get("rows", []))
    _log(f"Wrote {n_rows} dtype benchmark shape(s) to {json_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
