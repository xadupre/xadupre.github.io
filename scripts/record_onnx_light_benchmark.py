"""Benchmark ``onnx-light`` vs ``onnxruntime`` on the backend test cases.

The script discovers every benchmark-sized backend node test bundled with
the installed ``onnx-light`` package and measures the processing time of
``onnxruntime``, the ``onnx-light`` reference implementation backed by its
C++ ``KernelDispatchTable``, and ``onnx-light`` running with the
``onnx-light-cpu`` SIMD kernels registered on top. The ``onnx-light-cpu``
backend runs the model through the exact same ``onnx-light``
``ReferenceEvaluator`` API as the plain ``onnx-light`` backend, but
first calls ``onnx_light_cpu.register_kernels()`` to install the
SIMD-accelerated ``Abs``/``Exp``/``Log``/``Gemm``/``Not`` kernels into
``onnx-light``'s shared C++ ``KernelDispatchTable`` (replacing the built-in
entries for the default ONNX domain), so every matching node dispatches to the
SIMD-accelerated kernel. That registration is global and irreversible, so the
built-in ``onnx-light`` pass over every test runs before the
``onnx-light-cpu`` pass registers the kernels.

``onnx-light`` runs each model through its public NumPy-based
``ReferenceEvaluator`` API. The evaluator owns a reusable native C++ runner,
resolves each kernel once and replays the cached execution plan on subsequent
runs.

The benchmark runs every test through ``onnx-light``, then every test through
``onnx-light-cpu``, and finally every test through ONNX Runtime. Sessions from
one phase are released before the next phase starts, so each runtime keeps its
default worker-spin policy without perturbing another runtime's measurements.

For each backend and test the measurement protocol is:

1. Load / compile the model once (not timed).
2. Run up to :data:`N_WARMUP` iterations to prime the JIT / kernel cache,
   stopping after :data:`MAX_REPEAT_TIME_S` seconds of cumulative execution.
3. Run up to :data:`N_MEASURE` iterations, also stopping after
   :data:`MAX_REPEAT_TIME_S` seconds, and record the wall-clock time of each.
4. Report the per-backend **average** execution time
   (in milliseconds), computed as a trimmed mean that discards the fastest
   and slowest timed iterations, along with the raw ``min``/``max`` samples,
   and the **speedup** defined as::

       speedup = onnxruntime_avg_ms / onnx_light_avg_ms

   A speedup > 1 indicates that ``onnx-light`` is faster than
   ``onnxruntime`` on that test case.
5. Aggregate three summary averages: ``avg_speedup`` is the unweighted mean
   of every per-test ``speedup`` ratio, while ``avg_speedup_weighted`` is
   ``sum(N_i * speedup_i) / sum(N_i)`` across those same tests, where
   ``N_i`` is a symbolic per-test cost estimate (linear in tensor size for
   ordinary unary/binary kernels and quadratic for matrix multiplication and
   attention, see ``_symbolic_cost``). ``speedup_sum_latency`` is
   ``sum(onnxruntime_avg_ms) / sum(onnx_light_avg_ms)``.
   The unweighted mean treats every kernel equally regardless of its
   actual cost, so an O(1) kernel (``Shape``, ``Reshape``, ...) counts as
   much as an O(n) (``Add``) or O(n^2) (``Gemm``) one; the weighted
   average instead gives more importance to kernels that process more
   data, without depending on any backend's measured execution time.
   Each row also carries an ``operator`` field (its model's node
   ``op_type``(s)), and ``summary["operator_weights"]`` reports, for every
   operator, the total ``N`` weight it contributes (see
   ``_operator_weights``), making the weighting transparent per operator.

The resulting payload is persisted to
``cache_data/onnx-light/benchmark.json``. The dashboard at
``dashboard/onnx-light/benchmark.html`` renders that file as a sortable,
searchable table with per-row collapsible SVG graphs.

Usage::

    python scripts/record_onnx_light_benchmark.py [--cache-dir DIR]
        [--kind node] [--limit N] [--n-warmup N] [--n-measure N]
        [--max-repeat-time SECONDS]
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: Backends exercised by the benchmark. ``onnxruntime``, ``onnx_light`` and
#: ``onnx_light_cpu`` are timed; the reference implementation is intentionally
#: omitted because it is Python-only and not representative of production
#: performance. ``onnx_light_cpu`` runs the model through the *same*
#: ``onnx-light`` ``ReferenceEvaluator`` API as ``onnx_light``, but with
#: the SIMD-accelerated kernels shipped by ``onnx-light-cpu`` installed into
#: onnx-light's shared C++ dispatch table via ``register_kernels``, so it
#: isolates the effect of those kernels within the same engine.
BENCHMARK_BACKENDS: Tuple[str, ...] = (
    "onnxruntime",
    "onnx_light",
    "onnx_light_cpu",
)

BACKEND_PACKAGE: Dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "onnx_light": "onnx_light",
    "onnx_light_cpu": "onnx_light_cpu",
}

CPU_COUNT: int = os.cpu_count() or 1

#: Default number of warm-up iterations (not timed) run before measurement.
N_WARMUP: int = 2 * CPU_COUNT

#: Default number of timed iterations used to compute the average time.
N_MEASURE: int = 10 * CPU_COUNT

MAX_REPEAT_TIME_S: float = 1.0

DEFAULT_KIND: str = "node"

BENCHMARK_EXECUTION_ORDER: Tuple[str, ...] = (
    "onnx_light",
    "onnx_light_cpu",
    "onnxruntime",
)

#: Suffix identifying the genuine benchmark-sized backend test cases. In
#: ``TestMode.BENCHMARK`` mode ``onnx-light`` registers large, benchmark-sized
#: models under names ending with this suffix. Operators that do not provide a
#: benchmark variant still emit their small correctness cases in that mode, so
#: the discovery step filters on this suffix to keep only the benchmark models.
BENCHMARK_NAME_SUFFIX: str = "_benchmark"


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
    for name in ("onnx", "onnxruntime", "onnx_light", "onnx_light_cpu", "numpy"):
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


def _cc_numpy_dtype_for(data_type: int):
    """Return the numpy dtype for a fixed-width ``TensorProto`` data type.

    Returns ``None`` for types without a directly reinterpretable numpy dtype
    (e.g. STRING and packed sub-byte types), which must be handled out of band.
    """
    import numpy as np
    import onnx

    try:
        import ml_dtypes  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        ml_dtypes = None

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
        )
        for onnx_type, attr_name in optional_ml_dtypes:
            dtype = getattr(ml_dtypes, attr_name, None)
            if dtype is not None:
                dtype_map[int(onnx_type)] = dtype

    return dtype_map.get(int(data_type))


def _cc_tensor_to_numpy(tensor):
    """Convert a C++ backend-test tensor into a numpy value."""
    import numpy as np
    import onnx

    if int(tensor.data_type) == int(onnx.TensorProto.STRING):
        values = tensor.string_data()
        arr = np.array(values, dtype=object)
        return arr.reshape(tuple(int(d) for d in tensor.shape))

    dtype = _cc_numpy_dtype_for(int(tensor.data_type))
    if dtype is None:
        raise NotImplementedError(
            f"Cannot convert benchmark input tensor with data_type={tensor.data_type}."
        )
    arr = np.frombuffer(tensor.raw_data(), dtype=dtype)
    return arr.reshape(tuple(int(d) for d in tensor.shape))


def _cc_data_sets_to_python(test_case) -> List[Tuple[List[Any], List[Any]]]:
    """Convert C++ backend-test datasets into ``(inputs, outputs)`` tuples.

    The benchmark only consumes positional inputs, so the output side of
    each tuple is kept empty.
    """

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
        data_sets.append((inputs, []))  # benchmark runs measure execution only
    return data_sets


def _discover_benchmark_mode_tests(kind: str) -> Optional[List[Dict[str, Any]]]:
    """Discover benchmark-sized backend tests when onnx-light exposes them.

    Only cases whose name ends with :data:`BENCHMARK_NAME_SUFFIX` are kept:
    ``TestMode.BENCHMARK`` also returns the standard correctness cases for
    operators that do not provide a benchmark variant, and those tiny models
    are not meaningful to benchmark.
    """
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
        # ``TestMode.BENCHMARK`` still emits the small correctness cases for
        # operators that do not register a benchmark variant. Keep only the
        # genuine benchmark-sized models so the dashboard does not time tiny
        # correctness inputs.
        if not str(name).endswith(BENCHMARK_NAME_SUFFIX):
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


def _type_proto_kind(type_proto: Any) -> str:
    """Return ``"sequence"``, ``"map"`` or ``"tensor"`` for a graph value type.

    Tensor, sequence and map values have different serialized representations
    in backend test datasets. Both the protobuf ``HasField`` API and
    onnx-light's ``has_*_type()`` accessors are supported.
    """
    if type_proto is None:
        return "tensor"
    for field, kind in (("sequence_type", "sequence"), ("map_type", "map")):
        has_field = getattr(type_proto, "HasField", None)
        if callable(has_field):
            try:
                if has_field(field):
                    return kind
            except (ValueError, KeyError):
                pass
        method = getattr(type_proto, f"has_{field}", None)
        if callable(method) and method():
            return kind
    return "tensor"


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
        model.SerializeToString(),
        providers=["CPUExecutionProvider"],
    )
    input_names = [i.name for i in sess.get_inputs()]

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(sess.run(None, feeds))

    return _run


def _make_onnx_light_reference_runner(
    model,
    register: Optional[Callable[[], Any]] = None,
) -> Callable[[List[Any]], List[Any]]:
    """Run ``model`` through onnx-light's public NumPy-based evaluator API.

    ``ReferenceEvaluator`` owns and reuses its native C++ execution runner.
    Inputs are passed directly as NumPy arrays, just as they are for
    ``onnxruntime.InferenceSession.run``; the benchmark does not construct
    runtime tensors or protobuf values itself.

    ``register``, when provided, runs before evaluator construction because
    onnx-light-cpu installs its kernels into the process-wide dispatch table.
    """
    from onnx_light.onnx.reference import ReferenceEvaluator

    if register is not None:
        register()

    evaluator = ReferenceEvaluator(model.SerializeToString())
    input_names = evaluator.input_names

    def _run(inputs: List[Any]) -> List[Any]:
        return list(evaluator.run(None, dict(zip(input_names, inputs))))

    _run.used_kernels = evaluator.used_kernels  # type: ignore[attr-defined]

    return _run


def _make_onnx_light_runner(model) -> Callable[[List[Any]], List[Any]]:
    """Build the onnx-light runner for ``model``.

    Runs ``model`` through onnx-light's public ``ReferenceEvaluator`` API
    (native runner initialized once, then reused), the same API used by the
    ``onnx_light_cpu`` backend. Any failure — whether while building the
    evaluator or while a kernel runs — is surfaced so the benchmark records a
    clear error, exactly like the ``onnxruntime`` and ``onnx_light_cpu``
    backends; there is no fallback to another runner.
    """
    return _make_onnx_light_reference_runner(model)


def _make_onnx_light_cpu_runner(
    model,
    *,
    register_kernels: bool = True,
    verification_state: Optional[Dict[str, bool]] = None,
) -> Callable[[List[Any]], List[Any]]:
    """Build the onnx-light runner with the ``onnx-light-cpu`` SIMD kernels active.

    ``onnx-light-cpu`` ships SIMD-accelerated ``Abs``/``Exp``/``Log``/``Gemm``/
    ``Not`` kernels that override the corresponding built-in entries for the
    default ONNX domain. The kernels are installed process-wide through
    :func:`onnx_light_cpu.register_kernels` before constructing the evaluator.
    The plain ``onnx-light`` backend is benchmarked for every model before that
    global registration occurs, so its measurements retain the built-in
    kernels.

    Importing ``onnx_light_cpu`` — or calling ``register_kernels`` — raises
    ``ImportError`` when ``onnx-light-cpu`` is not installed (or was built
    without the onnx-light integration), so the ``onnx_light_cpu`` backend
    records a clear load error in that case. No fallback is provided: the point
    of this backend is to measure the CPU kernels, so an unavailable
    onnx-light-cpu is surfaced as a load error rather than silently running the
    built-in kernels.

    A model whose operators are *not* overridden by onnx-light-cpu would run
    through the exact same built-in kernels as the plain ``onnx-light`` backend,
    making the measurement meaningless. To guard against that, the returned
    runner checks — on its first invocation — that every kernel that actually
    ran is an onnx-light-cpu kernel (its library-qualified name appears in
    :func:`onnx_light_cpu.registered_kernel_names`), using
    :func:`onnx_light_cpu.used_kernel_names`. It raises ``RuntimeError`` when no
    onnx-light-cpu kernel ran, or when a name that is *not* an onnx-light-cpu
    kernel is recorded, so the backend records an error instead of silently
    reporting built-in-kernel timings as onnx-light-cpu results. Builds that do
    not expose the kernel-name introspection helpers skip the check.

    ``ReferenceEvaluator.used_kernels()`` exposes the normalized
    ``"<domain>:<op_type>"`` identifiers of the operators the evaluator
    executed. The check uses it to ensure every operator that onnx-light-cpu
    overrides was served by an
    onnx-light-cpu kernel. This catches the case where an overridable operator
    silently fell back to a built-in kernel (so onnx-light-cpu is *not* really
    used where it should be), which the process-wide
    :func:`onnx_light_cpu.used_kernel_names` record alone cannot detect.

    ``register_kernels=False`` and a shared ``verification_state`` let callers
    that run a homogeneous benchmark suite pay both process-wide setup costs
    only once. The defaults retain the standalone per-model checks.
    """
    import onnx_light_cpu

    runner = _make_onnx_light_reference_runner(
        model, register=onnx_light_cpu.register_kernels if register_kernels else None
    )

    checked = verification_state if verification_state is not None else {"done": False}
    if checked["done"]:
        return runner

    clear_used_kernel_names = getattr(onnx_light_cpu, "clear_used_kernel_names", None)
    used_kernel_names = getattr(onnx_light_cpu, "used_kernel_names", None)
    registered_kernel_names = getattr(onnx_light_cpu, "registered_kernel_names", None)
    try:
        from onnx_light_cpu.onnx_py._cpuregister import set_kernel_usage_recording
    except ImportError:
        set_kernel_usage_recording = None
    if clear_used_kernel_names is None or used_kernel_names is None:
        checked["done"] = True
        return runner

    # Map of ONNX ``op_type`` -> library-qualified kernel name that
    # onnx-light-cpu overrides, e.g. ``{"Abs": "onnx_light_cpu::Abs"}``. The
    # values are the names each onnx-light-cpu kernel records when it runs; a
    # used name outside this value set means a built-in (non onnx-light-cpu)
    # kernel ran.
    registered = (
        dict(registered_kernel_names()) if callable(registered_kernel_names) else {}
    )
    cpu_kernel_names = set(registered.values())
    overridden_op_types = set(registered)

    def _run_checked(inputs: List[Any]) -> List[Any]:
        if checked["done"]:
            return runner(inputs)
        if set_kernel_usage_recording is not None:
            set_kernel_usage_recording(True)
        clear_used_kernel_names()
        try:
            outputs = runner(inputs)
            used = list(used_kernel_names())
            if not used:
                overridden = sorted(registered) if registered else []
                raise RuntimeError(
                    "no onnx-light-cpu kernel ran for this model; it contains none "
                    f"of the operators overridden by onnx-light-cpu ({overridden})"
                )
            if cpu_kernel_names:
                foreign = sorted(set(used) - cpu_kernel_names)
                if foreign:
                    raise RuntimeError(
                        "kernels that ran are not all from onnx-light-cpu; the "
                        f"following names are not onnx-light-cpu kernels: {foreign}"
                    )
            # Session-scoped cross-check (onnx-light#4391): every operator the
            # session executed that onnx-light-cpu overrides must have been served
            # by an onnx-light-cpu kernel.
            session_used_kernels = getattr(runner, "used_kernels", None)
            if callable(session_used_kernels) and overridden_op_types:
                session_overridable = {
                    identifier.rsplit(":", 1)[-1]
                    for identifier in session_used_kernels()
                    if identifier.rsplit(":", 1)[-1] in overridden_op_types
                }
                served = {
                    op_type for op_type, kernel in registered.items() if kernel in used
                }
                missing = sorted(session_overridable - served)
                if missing:
                    raise RuntimeError(
                        "some operators overridden by onnx-light-cpu ran with the "
                        "built-in kernels instead of the onnx-light-cpu kernels: "
                        f"{missing}"
                    )
            checked["done"] = True
            return outputs
        finally:
            if set_kernel_usage_recording is not None:
                set_kernel_usage_recording(False)

    return _run_checked


_RUNNER_FACTORIES: Dict[str, Callable[[Any], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _make_onnxruntime_runner,
    "onnx_light": _make_onnx_light_runner,
    "onnx_light_cpu": _make_onnx_light_cpu_runner,
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
    max_repeat_time_s: float = MAX_REPEAT_TIME_S,
    max_warmup_time_s: Optional[float] = None,
    runner_factory: Optional[Callable[[Any], Callable[[List[Any]], List[Any]]]] = None,
) -> Dict[str, Any]:
    """Run a benchmark for ``backend`` on ``model`` / ``data_sets``.

    Returns a dictionary with the following keys:

    * ``success`` – ``True`` when every iteration completed without error.
    * ``error`` – human-readable error string when ``success`` is ``False``.
    * ``error_step`` – ``"load"``, ``"warmup"`` or ``"measure"``.
    * ``avg_ms`` – average per-dataset execution time in milliseconds,
      computed as a trimmed mean that excludes the fastest and slowest
      timed iterations when at least three samples are available.
    * ``min_ms`` – fastest timed iteration in milliseconds.
    * ``max_ms`` – slowest timed iteration in milliseconds.
    * ``n_warmup`` – number of warm-up iterations actually run.
    * ``n_measure`` – number of timed iterations actually run.

    ``runner_factory`` overrides the registered backend factory when a caller
    needs to share process-wide setup across several benchmark invocations.
    """
    if max_warmup_time_s is None:
        max_warmup_time_s = max_repeat_time_s

    factory = runner_factory or _RUNNER_FACTORIES.get(backend)
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
    warmup_count = 0
    warmup_elapsed_s = 0.0
    for _ in range(n_warmup):
        t0 = time.perf_counter()
        for inputs, _ in data_sets:
            try:
                runner(inputs)
            except Exception as exc:  # noqa: BLE001
                return {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "warmup",
                    "n_warmup": warmup_count,
                    "n_measure": 0,
                }
        warmup_count += 1
        warmup_elapsed_s += time.perf_counter() - t0
        if warmup_elapsed_s >= max_warmup_time_s:
            break

    # --- Timed measurement iterations ---------------------------------------
    times_ms: List[float] = []
    total_elapsed_s = 0.0
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
                    "n_warmup": warmup_count,
                    "n_measure": len(times_ms),
                }
        elapsed_s = time.perf_counter() - t0
        elapsed_ms = elapsed_s * 1_000
        times_ms.append(elapsed_ms)
        total_elapsed_s += elapsed_s
        if len(times_ms) >= min(3, n_measure) and total_elapsed_s >= max_repeat_time_s:
            break

    if not times_ms:
        return {
            "success": False,
            "error": "no timing samples collected",
            "error_step": "measure",
            "n_warmup": warmup_count,
            "n_measure": 0,
        }

    # Sort the samples so the slowest and fastest iterations (typically caused
    # by GC pauses or scheduler hiccups) can be discarded from the average. The
    # min/max are still reported for reference, but ``avg_ms`` is a trimmed mean
    # that excludes them when there are enough samples to do so.
    sorted_ms = sorted(times_ms)
    if len(sorted_ms) > 2:
        trimmed = sorted_ms[1:-1]
    else:
        trimmed = sorted_ms
    avg_ms = sum(trimmed) / len(trimmed)
    return {
        "success": True,
        "error": "",
        "error_step": "",
        "avg_ms": round(avg_ms, 6),
        "min_ms": round(sorted_ms[0], 6),
        "max_ms": round(sorted_ms[-1], 6),
        "n_warmup": warmup_count,
        "n_measure": len(times_ms),
    }


def _first_input_type(data_sets: List[Tuple[List[Any], List[Any]]]) -> str:
    """Return the element type of a test's first input, e.g. ``"float32"``.

    The type is read from the first input of the first data set. Sequence and
    map inputs (represented as Python lists/tuples of arrays) are descended
    into until an array-like value with a ``dtype`` is found. Returns an empty
    string when no typed input is available.
    """
    if not data_sets:
        return ""
    try:
        inputs, _ = data_sets[0]
    except (ValueError, TypeError):
        return ""
    if not inputs:
        return ""
    value: Any = inputs[0]
    while isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return ""
    return str(getattr(dtype, "name", dtype))


def _count_elements(value: Any) -> int:
    """Return a rough element count for ``value``, used by :func:`_symbolic_cost`."""
    size = getattr(value, "size", None)
    if isinstance(size, int):
        return size
    if isinstance(value, (list, tuple)):
        return sum(_count_elements(v) for v in value)
    return 1


QUADRATIC_COST_OPERATORS = frozenset(
    {
        "Attention",
        "Gemm",
        "MatMul",
        "MatMulInteger",
        "MultiHeadAttention",
        "QAttention",
        "QLinearMatMul",
    }
)


def _cost_complexity(operator: str) -> str:
    """Return the complexity class used to weight ``operator``."""
    operators = set(operator.split("+"))
    return "quadratic" if operators & QUADRATIC_COST_OPERATORS else "linear"


def _declared_output_elements(model: Any) -> int:
    """Return the number of scalar elements declared by ``model``'s graph outputs.

    Benchmark data sets only carry inputs (execution is timed, results are not
    compared), so :func:`_symbolic_cost` falls back to the statically declared
    output shapes. Outputs with a dynamic or missing dimension are ignored.
    """
    try:
        outputs = list(model.graph.output)
    except AttributeError:
        return 0
    total = 0
    for output in outputs:
        try:
            dims = list(output.type.tensor_type.shape.dim)
        except AttributeError:
            continue
        if not dims:
            continue
        size = 1
        for dim in dims:
            value = int(getattr(dim, "dim_value", 0) or 0)
            if value <= 0:
                size = 0
                break
            size *= value
        total += size
    return total


def _symbolic_cost(
    data_sets: List[Tuple[List[Any], List[Any]]], operator: str = "", model: Any = None
) -> Optional[int]:
    """Return a symbolic complexity estimate ``N`` for a test's first data set.

    The base size is the total number of scalar input/output elements. When the
    data set carries no outputs (benchmark runs only feed inputs), the shapes
    declared by ``model``'s graph outputs are used instead, so that operators
    producing far more data than they consume (e.g. ``AffineGrid``) are not
    under-weighted. Unary, binary, and other ordinary kernels receive that
    linear weight. Matrix multiplication and attention kernels receive its
    square, reflecting the requested quadratic cost model. The estimate depends
    only on test shapes, never on a backend's measured execution time.
    """
    if not data_sets:
        return None
    try:
        inputs, outputs = data_sets[0]
    except (ValueError, TypeError):
        return None
    size = _count_elements(list(inputs))
    if outputs:
        size += _count_elements(list(outputs))
    elif model is not None:
        size += _declared_output_elements(model)
    if not size:
        return None
    return size**2 if _cost_complexity(operator) == "quadratic" else size


def _operator_name(model: Any) -> str:
    """Return the ``op_type`` (or ``+``-joined op_types) exercised by ``model``.

    Backend node tests build a graph around the single operator under test
    (occasionally chained with a couple of helper ops), so the graph's node
    ``op_type``s identify which operator a test's :func:`_symbolic_cost`
    weight is attributed to.
    """
    try:
        nodes = model.graph.node
    except AttributeError:
        return ""
    seen: List[str] = []
    for node in nodes:
        op_type = getattr(node, "op_type", "") or ""
        if op_type and op_type not in seen:
            seen.append(op_type)
    return "+".join(seen)


def _operator_weights(rows: List[Dict[str, Any]], cost_key: str = "cost_n") -> List[Dict[str, Any]]:
    """Return per-operator ``{"operator", "weight", "tests"}`` totals, sorted by weight desc.

    Aggregates the symbolic cost ``N`` (see :func:`_symbolic_cost`) of every
    row by its ``operator`` field, making explicit how much each operator
    contributes to :func:`_weighted_avg_speedup`.
    """
    totals: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        operator = row.get("operator") or ""
        weight = row.get(cost_key)
        if not operator or weight is None or weight <= 0:
            continue
        entry = totals.setdefault(operator, {"operator": operator, "weight": 0, "tests": 0})
        entry["weight"] += weight
        entry["tests"] += 1
    return sorted(totals.values(), key=lambda e: e["weight"], reverse=True)


def _weighted_avg_speedup(
    rows: List[Dict[str, Any]], speedup_key: str, cost_key: str = "cost_n"
) -> Optional[float]:
    """Return a cost-weighted average speedup, or ``None`` when unavailable.

    ``summary["avg_speedup"]`` is the unweighted mean of the per-test
    ``speedup`` ratios, so a cheap O(1) kernel (e.g. ``Shape``/``Reshape``)
    counts exactly as much as an O(n^2) kernel (e.g. ``Gemm``) even though the
    latter dominates real-world execution time. This helper instead weighs
    every test by its symbolic cost ``N`` (see :func:`_symbolic_cost`) — an
    estimate of how much data the test processes, not a measured timing —
    computing::

        weighted_speedup = sum(N_i * speedup_i) / sum(N_i)

    which gives more importance to tests that process more data, without
    tying the metric to any one backend's measured execution time.
    """
    weighted_total = 0.0
    weight_total = 0.0
    for row in rows:
        speedup = row.get(speedup_key)
        weight = row.get(cost_key)
        if speedup is None or weight is None or weight <= 0:
            continue
        weighted_total += speedup * weight
        weight_total += weight
    if weight_total <= 0:
        return None
    return round(weighted_total / weight_total, 4)


def _sum_latency_speedup(
    rows: List[Dict[str, Any]], backend_key: str
) -> Optional[float]:
    """Return ``sum(baseline latency) / sum(backend latency)``."""
    baseline_total = 0.0
    backend_total = 0.0
    for row in rows:
        baseline = row.get("onnxruntime_avg_ms")
        backend = row.get(backend_key)
        if baseline is None or backend is None or backend <= 0:
            continue
        baseline_total += baseline
        backend_total += backend
    if backend_total <= 0:
        return None
    return round(baseline_total / backend_total, 4)


def _row_from_results(
    name: str,
    results: Dict[str, Dict[str, Any]],
    tag: str = "",
    graph: Optional[Dict[str, Any]] = None,
    cost_n: Optional[int] = None,
    cost_complexity: str = "",
    operator: str = "",
    input_type: str = "",
) -> Dict[str, Any]:
    """Build a dashboard row from per-backend benchmark results."""
    row: Dict[str, Any] = {"name": name}
    if tag:
        row["tag"] = tag
    if graph is not None:
        row["graph"] = graph
    if cost_n is not None:
        row["cost_n"] = cost_n
    if cost_complexity:
        row["cost_complexity"] = cost_complexity
    if operator:
        row["operator"] = operator
    if input_type:
        row["input_type"] = input_type

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
    cpu_avg = results.get("onnx_light_cpu", {}).get("avg_ms")
    ort_ok = results.get("onnxruntime", {}).get("success", False)
    light_ok = results.get("onnx_light", {}).get("success", False)
    cpu_ok = results.get("onnx_light_cpu", {}).get("success", False)
    if (
        ort_ok
        and light_ok
        and ort_avg is not None
        and light_avg is not None
        and light_avg > 0
    ):
        row["speedup"] = round(ort_avg / light_avg, 4)

    # Compute speedup_cpu = onnxruntime_avg_ms / onnx_light_cpu_avg_ms, mirroring
    # ``speedup`` for the onnx-light runtime running onnx-light-cpu kernels.
    if (
        ort_ok
        and cpu_ok
        and ort_avg is not None
        and cpu_avg is not None
        and cpu_avg > 0
    ):
        row["speedup_cpu"] = round(ort_avg / cpu_avg, 4)

    return row


# ---------------------------------------------------------------------------
# Top-level payload builder
# ---------------------------------------------------------------------------


def build_payload(
    kind: str = DEFAULT_KIND,
    limit: Optional[int] = None,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    max_repeat_time_s: float = MAX_REPEAT_TIME_S,
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

    def _benchmark(name: str, model: Any, data_sets: Any, backend: str) -> Dict[str, Any]:
        try:
            return run(
                model,
                data_sets,
                backend,
                n_warmup=n_warmup,
                n_measure=n_measure,
                max_repeat_time_s=max_repeat_time_s,
            )
        except Exception as exc:  # noqa: BLE001
            _log(
                f"Unhandled error for {name} on {backend}: {exc}\n"
                f"{traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": _stringify_error(exc),
                "error_step": "run",
            }

    per_test: List[Dict[str, Any]] = []
    for test in tests:
        name = test["name"]
        model = test["model"]
        data_sets = test["data_sets"]
        tag = str(test.get("tag", "") or "")

        try:
            graph = build_graph(model)
        except Exception:  # noqa: BLE001
            graph = None

        operator = _operator_name(model)
        cost_complexity = _cost_complexity(operator)
        cost_n = _symbolic_cost(data_sets, operator, model)
        input_type = _first_input_type(data_sets)

        per_test.append(
            {
                "name": name,
                "tag": tag,
                "graph": graph,
                "results": {},
                "cost_n": cost_n,
                "cost_complexity": cost_complexity,
                "operator": operator,
                "input_type": input_type,
            }
        )

    # The plain onnx-light phase must precede onnx-light-cpu because registering
    # the accelerated kernels is process-wide and irreversible. ONNX Runtime is
    # measured last, after all onnx-light sessions have been released.
    for backend in BENCHMARK_EXECUTION_ORDER:
        for idx, (test, entry) in enumerate(zip(tests, per_test)):
            entry["results"][backend] = _benchmark(
                entry["name"], test["model"], test["data_sets"], backend
            )
            if (idx + 1) % 50 == 0:
                _log(f"Benchmarked {idx + 1}/{len(tests)} tests ({backend}).")
        gc.collect()

    rows: List[Dict[str, Any]] = [
        _row_from_results(
            entry["name"],
            entry["results"],
            tag=entry["tag"],
            graph=entry["graph"],
            cost_n=entry["cost_n"],
            cost_complexity=entry["cost_complexity"],
            operator=entry["operator"],
            input_type=entry["input_type"],
        )
        for entry in per_test
    ]

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
    weighted = _weighted_avg_speedup(both_ok, "speedup")
    if weighted is not None:
        summary["avg_speedup_weighted"] = weighted
    sum_latency = _sum_latency_speedup(both_ok, "onnx_light_avg_ms")
    if sum_latency is not None:
        summary["speedup_sum_latency"] = sum_latency
    operator_weights = _operator_weights(both_ok)
    if operator_weights:
        summary["operator_weights"] = operator_weights

    # Summary stats for the onnx-light + onnx-light-cpu kernels backend.
    cpu_ok = [
        r
        for r in rows
        if r.get("onnxruntime_success") and r.get("onnx_light_cpu_success")
    ]
    speedups_cpu = [r["speedup_cpu"] for r in cpu_ok if "speedup_cpu" in r]
    summary["cpu_succeeded"] = len(cpu_ok)
    if speedups_cpu:
        summary["avg_speedup_cpu"] = round(sum(speedups_cpu) / len(speedups_cpu), 4)
        summary["min_speedup_cpu"] = round(min(speedups_cpu), 4)
        summary["max_speedup_cpu"] = round(max(speedups_cpu), 4)
    weighted_cpu = _weighted_avg_speedup(cpu_ok, "speedup_cpu")
    if weighted_cpu is not None:
        summary["avg_speedup_weighted_cpu"] = weighted_cpu
    sum_latency_cpu = _sum_latency_speedup(cpu_ok, "onnx_light_cpu_avg_ms")
    if sum_latency_cpu is not None:
        summary["speedup_sum_latency_cpu"] = sum_latency_cpu

    return {
        "date": now_iso,
        "kind": kind,
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "max_repeat_time_s": max_repeat_time_s,
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
    parser.add_argument(
        "--max-repeat-time",
        type=float,
        default=MAX_REPEAT_TIME_S,
        help=(
            "Maximum cumulative time in seconds for each warm-up and "
            f"measurement phase (default: {MAX_REPEAT_TIME_S})."
        ),
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
            max_repeat_time_s=args.max_repeat_time,
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
