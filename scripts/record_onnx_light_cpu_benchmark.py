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
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional

import record_onnx_light_benchmark as rlb

BENCHMARK_BACKENDS = ("onnx_light_cpu", "onnxruntime")
N_WARMUP = 3
N_MEASURE = 10
_SIMD_NAMES = {0: "scalar", 1: "SSE2", 2: "AVX", 3: "AVX2", 4: "AVX-512"}


def collect_versions() -> Dict[str, str]:
    """Return versions of the packages involved in the benchmark."""
    versions: Dict[str, str] = {}
    for name in ("onnx", "onnxruntime", "onnx_light", "onnx_light_cpu", "numpy"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def discover_benchmark_tests(kind: str = "node") -> List[Dict[str, Any]]:
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
    parts: List[str] = []
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


def _row(inputs: str, cpu: Dict[str, Any], ort: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
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


def run_tests(
    tests: List[Dict[str, Any]],
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    run: Callable[..., Dict[str, Any]] = rlb.run_benchmark,
) -> List[Dict[str, Any]]:
    """Benchmark the supplied onnx-light-cpu cases."""
    examples: List[Dict[str, Any]] = []
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
        inputs = _format_inputs(data_sets[0][0])
        row = _row(inputs, cpu, ort)
        speedups = [row["speedup_cpu"]] if "speedup_cpu" in row else []
        operator = rlb._operator_name(model) or "?"
        examples.append(
            {
                "name": test["name"],
                "title": f"{operator}: onnxruntime vs onnx-light-cpu",
                "op": operator,
                "source": f"onnx-light-cpu benchmark test ({test['name']})",
                "backends": list(BENCHMARK_BACKENDS),
                "rows": [row],
                "summary": {
                    "inputs": 1,
                    "cpu_succeeded": len(speedups),
                    **(
                        {
                            "avg_speedup_cpu": speedups[0],
                            "min_speedup_cpu": speedups[0],
                            "max_speedup_cpu": speedups[0],
                        }
                        if speedups
                        else {}
                    ),
                },
            }
        )
    return sorted(examples, key=lambda ex: (ex["op"].lower(), ex["name"].lower()))


def build_payload(
    kind: str = "node",
    limit: Optional[int] = None,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    discover: Callable[[str], List[Dict[str, Any]]] = discover_benchmark_tests,
    run: Callable[..., List[Dict[str, Any]]] = run_tests,
    versions: Callable[[], Dict[str, str]] = collect_versions,
    now: Optional[dt.datetime] = None,
) -> Dict[str, Any]:
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


def write_payload(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache_data")
    parser.add_argument("--kind", default="node")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-measure", type=int, default=N_MEASURE)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
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
