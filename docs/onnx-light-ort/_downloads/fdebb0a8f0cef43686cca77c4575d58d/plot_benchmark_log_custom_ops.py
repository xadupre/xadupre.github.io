"""
Benchmark Log: built-in and custom operators
============================================

Compare synchronous CPU calls to ``Log`` using NumPy, ORT's built-in operator,
and the ``onnx_light_cpu`` custom operator (not an execution provider).
Two graph topologies are used: ``X -> Log -> Y`` and
``X -> Log -> ... -> Log -> Y`` with ten consecutive nodes. Both use float32,
dynamic one-dimensional inputs and models built with ``onnx_light.onnx``.

ORT executes each topology with either standard-domain or custom-domain nodes,
using all four synchronous Python calling interfaces: ``run``,
``run_with_ort_values``, ``run_with_iobinding`` and ``run_with_ortvaluevector``.
The standalone onnx-light custom-op host exposes only ``run`` and accepts only
one node. Its ten-node result is therefore unavailable, not ten Python calls
misrepresented as a graph. NumPy provides a baseline with one or ten ufunc calls.

Repeated real-valued logarithms inevitably leave the positive domain in float32.
The ten-node graph produces NaNs: this is a literal Log chain, not a benchmark
of ten finite-input logarithms. Every path is checked against a NumPy reference,
including matching NaNs, before and after timing.
"""

import argparse
from functools import partial
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy
from onnx_light.onnx import TensorProto, checker, helper

# %%
# Construct both graph topologies here, so the downloaded gallery example
# contains the complete benchmark, including its isolated worker.


def make_model(custom, nodes=1):
    domain = "onnx_light_cpu" if custom else ""
    names = ["X", *(f"log_{index}" for index in range(1, nodes)), "Y"]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Log", [names[index]], [names[index + 1]], domain=domain)
                for index in range(nodes)
            ],
            "log-custom-op-benchmark",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N"])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N"])],
        ),
        opset_imports=[helper.make_opsetid(domain, 1 if custom else 18)],
        ir_version=9,
    )
    checker.check_model(model)
    return model.SerializeToString()


def numpy_log(values, nodes):
    for _ in range(nodes):
        values = numpy.log(values)
    return values


# %%
# Prepare reusable inputs and outputs before timing each ORT calling interface.
# Import ORT only inside these functions: standalone workers must not load it.


def ort_runner(session, values, api):
    """Prepares reusable inputs and outputs outside the measured region."""
    import onnxruntime

    feeds = {"X": values}
    if api == "run":
        return lambda: session.run(["Y"], feeds), lambda result: result[0]
    input_value = onnxruntime.OrtValue.ortvalue_from_numpy(values)
    if api == "ortvalues":
        feeds = {"X": input_value}
        return (
            lambda: session.run_with_ort_values(["Y"], feeds),
            lambda result: result[0].numpy(),
        )
    output = numpy.empty_like(values)
    output_value = onnxruntime.OrtValue.ortvalue_from_numpy(output)
    if api == "iobinding":
        binding = session.io_binding()
        binding.bind_ortvalue_input("X", input_value)
        binding.bind_ortvalue_output("Y", output_value)
        return lambda: session.run_with_iobinding(binding), lambda result: output
    if api != "ortvaluevector":
        raise ValueError(f"Unsupported ORT API: {api}")
    from onnxruntime.capi import _pybind_state as C

    inputs = C.OrtValueVector()
    inputs.push_back(input_value._get_c_value())
    outputs = C.OrtValueVector()
    outputs.push_back(output_value._get_c_value())
    devices = [C.OrtDevice(C.OrtDevice.cpu(), C.OrtDevice.default_memory(), 0)]
    options = onnxruntime.RunOptions()
    input_names, output_names = ["X"], ["Y"]
    return (
        lambda: session.run_with_ortvaluevector(
            options, input_names, inputs, output_names, outputs, devices
        ),
        lambda result: output,
    )


# %%
# Check each path against NumPy before and after collecting its timing samples.


def measure(args):
    if args.backend == "standalone" and (args.nodes != 1 or args.api != "run"):
        raise ValueError("The standalone host supports only one node and the run API")
    custom = args.backend != "ort"
    if args.backend == "standalone":
        from onnx_light_ort import CustomOpSession

        session = CustomOpSession(
            make_model(custom, args.nodes), library=args.library, host_library=args.host
        )
        version = None
    elif args.backend == "numpy":
        version = None
    else:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        if custom:
            options.register_custom_ops_library(str(args.library))
        session = onnxruntime.InferenceSession(
            make_model(custom, args.nodes), options, providers=["CPUExecutionProvider"]
        )
        version = onnxruntime.__version__

    rng = numpy.random.default_rng(42)
    rows = []
    for size in args.sizes:
        values = rng.uniform(0.01, 10, size).astype(numpy.float32)
        feeds = {"X": values}
        if args.backend == "standalone":
            run, get_output = partial(session.run, None, feeds), lambda result: result[0]
        elif args.backend == "numpy":
            run = partial(numpy_log, values, args.nodes)

            def get_output(result):
                return result

        else:
            run, get_output = ort_runner(session, values, args.api)
        # Repeated real-valued logarithms eventually leave the positive domain.
        with numpy.errstate(divide="ignore", invalid="ignore"):
            expected = values.copy()
            for _ in range(args.nodes):
                expected = numpy.log(expected.astype(numpy.float64)).astype(numpy.float32)
            for _ in range(args.warmup):
                run()
            numpy.testing.assert_allclose(
                get_output(run()), expected, rtol=2e-6, atol=0, equal_nan=True
            )
            samples = []
            for _ in range(args.repeat):
                begin = time.perf_counter()
                for _ in range(args.number):
                    run()
                samples.append((time.perf_counter() - begin) / args.number)
            numpy.testing.assert_allclose(
                get_output(run()), expected, rtol=2e-6, atol=0, equal_nan=True
            )
        rows.append({"size": size, "samples_s": samples})

    if args.backend == "standalone":
        session.close()
        assert "onnxruntime" not in sys.modules
    assert "onnx" not in sys.modules
    return {
        "backend": args.backend,
        "api": args.api,
        "nodes": args.nodes,
        "ort_version": version,
        "rows": rows,
    }


# %%
# The command-line entry point also serves isolated measurement workers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[1024, 32768, 65536, 262144, 1048576]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--nodes", type=int, choices=[1, 10], default=1)
    parser.add_argument(
        "--api", choices=["run", "ortvalues", "iobinding", "ortvaluevector"], default="run"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend", choices=["ort", "custom", "standalone", "numpy"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(args.sizes) <= 0 or max(args.sizes) > 16 * 1024 * 1024:
        parser.error("sizes must be positive and fit the host's 64 MiB tensor limit")
    if min(args.repeat, args.number, args.trials) <= 0 or args.warmup < 0:
        parser.error("repeat, number and trials must be positive; warmup must be nonnegative")
    args.library = args.library.resolve(strict=True)
    args.host = args.host.resolve(strict=True)
    if args.backend:
        print(json.dumps(measure(args)))
        return

    backends = ["ort", "custom"]
    if args.nodes == 1 and args.api == "run":
        backends.append("standalone")
    results = []
    for trial in range(args.trials):
        offset = trial % len(backends)
        for backend in backends[offset:] + backends[:offset]:
            command = [
                sys.executable,
                str(Path(inspect.getsourcefile(main)).resolve()),
                "--library",
                str(args.library),
                "--host",
                str(args.host),
                "--sizes",
                *(str(size) for size in args.sizes),
                "--warmup",
                str(args.warmup),
                "--repeat",
                str(args.repeat),
                "--number",
                str(args.number),
                "--backend",
                backend,
                "--nodes",
                str(args.nodes),
                "--api",
                args.api,
            ]
            completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
            results.append(json.loads(completed.stdout))

    report = {
        "measurement": f"steady-state {args.api}; includes Python interfaces",
        "nodes": args.nodes,
        "threads": 1,
        "operator": "Log",
        "dtype": "float32",
        "input_distribution": "uniform(0.01, 10), NumPy default_rng seed 42",
        "platform": platform.platform(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": platform.python_version(),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "number": args.number,
        "trials": args.trials,
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    standalone = " CPU custom op / standalone (us) |" if "standalone" in backends else ""
    print(f"| Elements | ORT Log (us) | CPU custom op / ORT (us) | Speedup |{standalone}")
    print("| ---: | ---: | ---: | ---: |" + (" ---: |" if standalone else ""))
    for index, size in enumerate(args.sizes):
        times = {}
        for backend in backends:
            samples = [
                value
                for result in results
                if result["backend"] == backend
                for value in result["rows"][index]["samples_s"]
            ]
            times[backend] = statistics.median(samples)
        standalone_time = f" {times['standalone'] * 1e6:.3f} |" if standalone else ""
        print(
            f"| {size:,} | {times['ort'] * 1e6:.3f} | {times['custom'] * 1e6:.3f} "
            f"| {times['ort'] / times['custom']:.2f}x |{standalone_time}"
        )
    print(
        "Raw samples, when --output is supplied, include all trials; "
        "standalone timings include protective input and output copies."
    )


# %%
# Each measurement runs in a fresh process so the standalone host never imports
# or loads ORT. Library paths are explicit, just like the other gallery examples.
# Sessions, random inputs (seed 42), OrtValues, bindings and preallocated outputs
# are prepared outside timing. ORT uses one thread and sequential execution.
# Each path gets three warmups and five batches of ten calls; report the median
# time per *whole graph*, not per node. Correctness checks and output conversion
# to NumPy are excluded from timing.
#
# ``run`` and ``run_with_ort_values`` allocate outputs; I/O binding and
# ``run_with_ortvaluevector`` reuse preallocated output buffers. The standalone
# wrapper includes protective input and output copies. These are end-to-end
# Python interface timings, not isolated kernel timings or universal speedups.


def run_gallery():
    benchmark = Path(inspect.getsourcefile(run_gallery)).resolve()
    root = benchmark.parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])
    launcher = [sys.executable]
    if sys.platform == "win32":
        launcher.append(str(root / "cmake/run_python_with_dll_paths.py"))
    library = Path(os.environ["ONNX_LIGHT_ORT_PLUGIN_LIBRARY"]).resolve(strict=True)
    host = Path(os.environ["ONNX_LIGHT_ORT_HOST_LIBRARY"]).resolve(strict=True)
    sizes = [1, 1024, 65536]
    apis = ["run", "ortvalues", "iobinding", "ortvaluevector"]
    paths = [
        ("numpy", "run"),
        ("standalone", "run"),
        *((backend, api) for backend in ("ort", "custom") for api in apis),
    ]
    print(f"Python {platform.python_version()}, {platform.platform()}")
    results = []
    for nodes in (1, 10):
        print(f"\n{nodes} Log node(s), median microseconds per graph")
        print(f"{'Backend / API':<34}" + "".join(f"{size:>14,}" for size in sizes))
        for backend, api in paths:
            label = f"{backend} / {api}"
            if backend == "standalone" and nodes != 1:
                print(f"{label:<34}  unavailable (single-node host)")
                continue
            command = [
                *launcher,
                str(benchmark),
                "--library",
                str(library),
                "--host",
                str(host),
                "--sizes",
                *(str(size) for size in sizes),
                "--nodes",
                str(nodes),
                "--api",
                api,
                "--backend",
                backend,
                "--warmup",
                "3",
                "--repeat",
                "5",
                "--number",
                "10",
            ]
            completed = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, text=True, env=environment
            )
            result = json.loads(completed.stdout)
            results.append(result)
            times = [statistics.median(row["samples_s"]) * 1e6 for row in result["rows"]]
            print(f"{label:<34}" + "".join(f"{value:>14.3f}" for value in times))
    ort_version = next(result["ort_version"] for result in results if result["backend"] == "ort")
    print(f"\nONNX Runtime {ort_version}; NumPy and onnx-light model bindings; float32 CPU")
    assert "onnx" not in sys.modules
    return results


# %%
# The full samples remain available in ``results``. No performance thresholds
# are asserted: CPU, SIMD dispatch and ORT version affect these measurements.
if __name__ == "__main__":
    if "__file__" in globals() and len(sys.argv) > 1:
        main()
    else:
        results = run_gallery()
