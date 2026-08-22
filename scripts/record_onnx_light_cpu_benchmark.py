"""Benchmark the tagged backend tests provided by ``onnx-light-cpu``.

The script registers onnx-light-cpu's C++ backend-test collector, discovers its
``TestMode.BENCHMARK`` cases through onnx-light, and compares onnx-light-cpu
with onnxruntime.  It never creates models or inputs: every measured test case
comes from the installed onnx-light-cpu source tree.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from collections.abc import Callable
from typing import Any

import record_onnx_light_benchmark as rlb

BENCHMARK_BACKENDS = ("onnx_light_cpu", "onnxruntime")
N_WARMUP = 3
N_MEASURE = 10
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


def discover_benchmark_tests(kind: str = "node") -> list[dict[str, Any]]:
    """Return only benchmark-tagged cases registered by onnx-light-cpu."""
    from onnx_light_cpu import register_backend_test_cases

    register_backend_test_cases()
    return [
        test
        for test in rlb.discover_node_tests(kind)
        if str(test.get("name", "")).startswith("test_cpu_")
        and str(test.get("name", "")).endswith("_benchmark")
    ]


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
                "test_names": [],
            },
        )
        group["rows"].append(row)
        group["test_names"].append(measurement["test_name"])

    examples: list[dict[str, Any]] = []
    for group in groups.values():
        rows = sorted(
            group.pop("rows"),
            key=lambda row: (row.get("input_elements", 0), row["inputs"]),
        )
        test_names = group.pop("test_names")
        speedups = [row["speedup_cpu"] for row in rows if "speedup_cpu" in row]
        group["rows"] = rows
        group["source"] = f"{len(test_names)} onnx-light-cpu benchmark tests"
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
    run: Callable[..., dict[str, Any]] = rlb.run_benchmark,
) -> list[dict[str, Any]]:
    """Benchmark the supplied onnx-light-cpu cases."""
    measurements: list[dict[str, Any]] = []
    for test in tests:
        model = test["model"]
        data_sets = test["data_sets"]
        cpu = run(
            model,
            data_sets,
            "onnx_light_cpu",
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        ort = run(
            model,
            data_sets,
            "onnxruntime",
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        first_inputs = data_sets[0][0]
        inputs = _format_inputs(first_inputs)
        row = _row(inputs, cpu, ort)
        row["input_elements"] = _first_input_element_count(first_inputs)
        operator = rlb._operator_name(model) or "?"
        measurements.append(
            {
                "operator": operator,
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
        "versions": versions(),
        "simd_level": level,
        "simd_name": _SIMD_NAMES.get(level, str(level)),
        "examples": run(tests, n_warmup=n_warmup, n_measure=n_measure),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        kind=args.kind,
        limit=args.limit,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )
    path = os.path.join(args.cache_dir, "onnx-light-cpu", "examples_benchmark.json")
    write_payload(path, payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
