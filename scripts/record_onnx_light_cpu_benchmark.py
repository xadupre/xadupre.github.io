"""Benchmark the tagged backend tests provided by ``onnx-light-cpu``.

The script registers onnx-light-cpu's C++ backend-test collector, discovers its
``TestMode.BENCHMARK`` cases through onnx-light, and compares onnx-light-cpu
with onnxruntime.  It never creates models or inputs: every measured test case
comes from the installed onnx-light-cpu source tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import re
import sys
from collections.abc import Callable
from typing import Any

import record_onnx_light_benchmark as rlb

BENCHMARK_BACKENDS = ("onnx_light_cpu", "onnxruntime")
N_WARMUP = rlb.N_WARMUP
N_MEASURE = rlb.N_MEASURE
MAX_REPEAT_TIME_S = rlb.MAX_REPEAT_TIME_S
_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}


def collect_versions() -> dict[str, str]:
    """Return versions of the packages involved in the benchmark."""
    versions: dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "onnx_light", "onnx_light_cpu", "numpy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _collect_benchmark_cases() -> list[Any]:
    from onnx_light.onnx.backend import TestMode, collect_test_cases

    return collect_test_cases(include_big=True, mode=TestMode.BENCHMARK)


def discover_benchmark_tests(kind: str = "node") -> list[dict[str, Any]]:
    """Return only benchmark-tagged cases registered by onnx-light-cpu."""
    from onnx_light_cpu import register_backend_test_cases

    register_backend_test_cases()
    kinds = rlb._normalize_kinds(kind)
    return [
        {"name": str(test.name)}
        for test in _collect_benchmark_cases()
        if str(test.name).startswith("test_cpu_")
        and str(test.name).endswith("_benchmark")
        and (not kinds or getattr(test, "kind", None) in kinds)
    ]


def _load_benchmark_test(name: str) -> dict[str, Any]:
    from onnx_light.onnx.backend import TestMode, collect_test_cases_by_name

    cases = collect_test_cases_by_name(
        f"^{re.escape(name)}$", include_big=True, mode=TestMode.BENCHMARK
    )
    if len(cases) != 1:
        raise RuntimeError(f"Unable to load benchmark test {name!r}.")
    case = cases[0]
    return {
        "name": name,
        "model": rlb._onnx_light_model_to_onnx(case.model),
        "data_sets": rlb._cc_data_sets_to_python(case),
    }


def _first_input_type(signature: Any) -> str:
    """Return the element type of the first input in a dtype/shape signature."""
    first = str(signature).split(",")[0].split("[")[0].strip()
    return first if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first) else ""


def _format_inputs(inputs: Any) -> str:
    """Return a readable dtype/shape signature for typed inputs."""
    parts: list[str] = []
    for value in inputs:
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            continue
        name = str(getattr(dtype, "name", dtype))
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape):
            parts.append(f"{name}[{'x'.join(str(int(dim)) for dim in shape)}]")
        else:
            parts.append(name)
    return ", ".join(parts)


def _row(inputs: str, cpu: dict[str, Any], ort: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "inputs": inputs,
        "input_type": _first_input_type(inputs),
    }
    if cpu.get("success"):
        row["onnx_light_cpu_ms"] = round(float(cpu["avg_ms"]), 6)
    if ort.get("success"):
        row["onnxruntime_ms"] = round(float(ort["avg_ms"]), 6)
    if cpu.get("success") and ort.get("success") and cpu["avg_ms"] > 0:
        row["speedup_cpu"] = round(float(ort["avg_ms"]) / float(cpu["avg_ms"]), 4)
    return row


def _first_input_element_count(inputs: Any) -> int:
    """Returns the element count of the first typed input."""
    for value in inputs:
        if getattr(value, "dtype", None) is None:
            continue
        shape = getattr(value, "shape", None)
        return math.prod(int(dim) for dim in shape) if shape is not None else 0
    return 0


def _group_measurements(
    measurements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Groups measured dimensions by operator and first-input element type."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for measurement in measurements:
        row = measurement["row"]
        operator = measurement["operator"]
        input_type = row["input_type"]
        key = (operator, input_type)
        group = groups.setdefault(
            key,
            {
                "name": f"{operator}_{input_type}_benchmark",
                "title": f"{operator} ({input_type}): onnxruntime vs onnx-light-cpu",
                "op": operator,
                "backends": list(BENCHMARK_BACKENDS),
                "rows": [],
            },
        )
        row["test_name"] = measurement["test_name"]
        group["rows"].append(row)

    examples: list[dict[str, Any]] = []
    for group in groups.values():
        rows = sorted(
            group.pop("rows"),
            key=lambda row: (
                row.get("input_elements", 0),
                row["inputs"],
                row["test_name"],
            ),
        )
        speedups = [row["speedup_cpu"] for row in rows if "speedup_cpu" in row]
        group["rows"] = rows
        group["source"] = f"{len(rows)} onnx-light-cpu benchmark tests"
        group["summary"] = {
            "inputs": len(rows),
            "cpu_succeeded": len(speedups),
            **(
                {
                    "avg_speedup_cpu": round(sum(speedups) / len(speedups), 4),
                    "min_speedup_cpu": min(speedups),
                    "max_speedup_cpu": max(speedups),
                }
                if speedups
                else {}
            ),
        }
        examples.append(group)
    return sorted(
        examples,
        key=lambda example: (
            example["op"].lower(),
            example["rows"][0]["input_type"].lower(),
        ),
    )


def run_tests(
    tests: list[dict[str, Any]],
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    max_repeat_time_s: float = MAX_REPEAT_TIME_S,
    run: Callable[..., dict[str, Any]] = rlb.run_benchmark,
    load: Callable[[str], dict[str, Any]] = _load_benchmark_test,
) -> list[dict[str, Any]]:
    """Benchmark the supplied onnx-light-cpu cases."""
    cpu_results = []
    metadata = []
    total = len(tests)
    for index, test in enumerate(tests, start=1):
        loaded = test if "model" in test else load(test["name"])
        rlb._log(f"Benchmarking {index}/{total} tests (onnx_light_cpu): {test['name']}")
        cpu_results.append(
            run(
                loaded["model"],
                loaded["data_sets"],
                "onnx_light_cpu",
                n_warmup=n_warmup,
                n_measure=n_measure,
                max_repeat_time_s=max_repeat_time_s,
            )
        )
        first_inputs = loaded["data_sets"][0][0]
        metadata.append(
            {
                "operator": rlb._operator_name(loaded["model"]) or "?",
                "inputs": _format_inputs(first_inputs),
                "input_elements": _first_input_element_count(first_inputs),
            }
        )
        del loaded
        gc.collect()
    gc.collect()

    ort_results = []
    for index, test in enumerate(tests, start=1):
        loaded = test if "model" in test else load(test["name"])
        rlb._log(f"Benchmarking {index}/{total} tests (onnxruntime): {test['name']}")
        ort_results.append(
            run(
                loaded["model"],
                loaded["data_sets"],
                "onnxruntime",
                n_warmup=n_warmup,
                n_measure=n_measure,
                max_repeat_time_s=max_repeat_time_s,
            )
        )
        del loaded
        gc.collect()

    measurements: list[dict[str, Any]] = []
    for test, cpu, ort, details in zip(
        tests, cpu_results, ort_results, metadata, strict=True
    ):
        row = _row(details["inputs"], cpu, ort)
        row["input_elements"] = details["input_elements"]
        measurements.append(
            {
                "operator": details["operator"],
                "test_name": test["name"],
                "row": row,
            }
        )
    return _group_measurements(measurements)


def build_payload(
    kind: str = "node",
    limit: int | None = None,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    max_repeat_time_s: float = MAX_REPEAT_TIME_S,
    discover: Callable[[str], list[dict[str, Any]]] = discover_benchmark_tests,
    run: Callable[..., list[dict[str, Any]]] = run_tests,
    versions: Callable[[], dict[str, str]] = collect_versions,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Discover, run, and format the onnx-light-cpu benchmark tests."""
    tests = discover(kind)
    if limit is not None:
        tests = tests[: max(0, limit)]
    from onnx_light_cpu.onnx_py._cpukernels import detect_simd_level

    level = int(detect_simd_level())
    timestamp = now or dt.datetime.now(tz=dt.timezone.utc)
    return {
        "date": timestamp.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "max_repeat_time_s": max_repeat_time_s,
        "versions": versions(),
        "simd_level": level,
        "simd_name": _SIMD_NAMES.get(level, str(level)),
        "examples": run(
            tests,
            n_warmup=n_warmup,
            n_measure=n_measure,
            max_repeat_time_s=max_repeat_time_s,
        ),
    }


def write_payload(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache_data")
    parser.add_argument("--kind", default="node")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-measure", type=int, default=N_MEASURE)
    parser.add_argument("--max-repeat-time", type=float, default=MAX_REPEAT_TIME_S)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        kind=args.kind,
        limit=args.limit,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        max_repeat_time_s=args.max_repeat_time,
    )
    path = os.path.join(args.cache_dir, "onnx-light-cpu", "examples_benchmark.json")
    write_payload(path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
