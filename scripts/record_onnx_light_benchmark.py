"""Benchmark ``onnx-light`` vs ``onnxruntime`` on the backend test cases.

The script discovers every benchmark-sized backend node test bundled with
the installed ``onnx-light`` package and measures the processing time of
``onnxruntime``, the ``onnx-light`` reference implementation backed by its
C++ ``KernelDispatchTable``, and ``onnx-light`` running with the
``onnx-light-cpu`` SIMD kernels registered on top. The ``onnx-light-cpu``
backend runs the model through the exact same ``onnx-light``
``RuntimeSession`` execution path as the plain ``onnx-light`` backend, but
first calls ``onnx_light_cpu.register_kernels()`` to install the
SIMD-accelerated ``Abs``/``Exp``/``Log``/``Gemm``/``Not`` kernels into
``onnx-light``'s shared C++ ``KernelDispatchTable`` (replacing the built-in
entries for the default ONNX domain), so every matching node dispatches to the
SIMD-accelerated kernel. That registration is global and irreversible, so the
built-in ``onnx-light`` pass over every test runs before the
``onnx-light-cpu`` pass registers the kernels.

``onnx-light`` runs a model through a reusable ``RuntimeSession`` that
resolves every kernel once and then replays them on each subsequent run, so
the benchmark builds the model's ``ExecutionPlan`` and ``RuntimeSession``
once (outside the timed loop) and reuses them across the measured
iterations.

For each test the measurement protocol is:

1. Load / compile the model once (not timed).
2. Run :data:`N_WARMUP` iterations to prime the JIT / kernel cache.
3. Run :data:`N_MEASURE` iterations and record the wall-clock time of each.
4. Report the per-backend **average** execution time
   (in milliseconds), computed as a trimmed mean that discards the fastest
   and slowest timed iterations, along with the raw ``min``/``max`` samples,
   and the **speedup** defined as::

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

#: Backends exercised by the benchmark. ``onnxruntime``, ``onnx_light`` and
#: ``onnx_light_cpu`` are timed; the reference implementation is intentionally
#: omitted because it is Python-only and not representative of production
#: performance. ``onnx_light_cpu`` runs the model through the *same*
#: ``onnx-light`` ``RuntimeSession`` execution path as ``onnx_light``, but with
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

#: Default number of warm-up iterations (not timed) run before measurement.
N_WARMUP: int = 3

#: Default number of timed iterations used to compute the average time.
N_MEASURE: int = 10

DEFAULT_KIND: str = "node"

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


def _model_input_names(model) -> List[str]:
    """Return the names of the graph inputs that are not initializers."""
    initializer_names = {init.name for init in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializer_names]


def _type_proto_kind(type_proto: Any) -> str:
    """Return ``"sequence"``, ``"map"`` or ``"tensor"`` for a graph value type.

    The low-level ``RuntimeSession`` execution path keeps tensor, sequence and
    map values in separate name-keyed stores on :class:`RuntimeContext`. The
    kind of a graph input/output therefore selects which store the benchmark
    feeds (``set`` / ``put_sequence`` / ``put_map``) and reads back (``get`` /
    ``get_sequence`` / ``get_map``). Both the protobuf ``HasField`` API and
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
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(sess.run(None, feeds))

    return _run


def _runtime_tensor_to_numpy(runtime, numpy_helper, tensor):
    """Convert an onnx-light runtime ``Tensor`` into a :class:`numpy.ndarray`.

    Standard fixed-width dtypes are reinterpreted from the tensor's raw byte
    view returned by ``runtime.tensor_to_numpy`` (no copy); packed sub-byte
    types and strings fall back to the general ``numpy_helper.to_array`` path
    via ``runtime.tensor_to_proto``.
    """
    import numpy as np

    dtype = _cc_numpy_dtype_for(int(tensor.data_type))
    if dtype is not None:
        raw = runtime.tensor_to_numpy(tensor)
        arr = raw.view(dtype)
        return arr.reshape(tuple(int(d) for d in tensor.shape))
    return numpy_helper.to_array(runtime.tensor_to_proto(tensor))


def _make_onnx_light_runtime_session_runner(
    model,
    register: Optional[Callable[[Any], Any]] = None,
) -> Callable[[List[Any]], List[Any]]:
    """Run ``model`` through onnx-light's ``ExecutionPlan`` + ``RuntimeSession``.

    onnx-light now exposes a reusable
    :class:`~onnx_light.onnx_py._onnxpykernels.runtime.RuntimeSession` that
    resolves every kernel once (on its first ``run``) and replays the cached
    kernels on subsequent runs, mirroring how an inference runtime prepares an
    executable graph once and then runs it repeatedly.

    The benchmark builds the plan and session a single time (outside the timed
    loop) so the measured iterations reflect the model's execution rather than
    the one-off kernel initialisation. Only the numeric backend-test corpus is
    benchmarked, so positional inputs are wired to the graph's declared
    (non-initializer) inputs by name.

    ``register``, when provided, is invoked once with the freshly created
    ``RuntimeSession`` before any run so a backend can install kernels *on that
    session* (rather than process-wide). The ``onnx_light_cpu`` backend uses it
    to register its SIMD kernels on the session via
    ``onnx_light_cpu.register_kernels(session)``.
    """
    import numpy as np
    from onnx_light.onnx_lib import ModelProto as _LModelProto
    from onnx_light.onnx_lib import numpy_helper as _lnh
    from onnx_light.onnx_py._onnxpykernels import runtime as _rt

    lmodel = model
    if not isinstance(model, _LModelProto):
        lmodel = _LModelProto()
        lmodel.ParseFromString(model.SerializeToString())

    version = 18
    for opset in lmodel.opset_import:
        if opset.domain in ("", "ai.onnx"):
            version = int(opset.version)
            break

    initializers = list(lmodel.graph.initializer)
    initializer_names = {init.name for init in initializers}
    graph_inputs = [vi for vi in lmodel.graph.input if vi.name not in initializer_names]
    input_names = [vi.name for vi in graph_inputs]
    input_kinds = [_type_proto_kind(getattr(vi, "type", None)) for vi in graph_inputs]
    output_names = [vi.name for vi in lmodel.graph.output]
    output_kinds = [
        _type_proto_kind(getattr(vi, "type", None)) for vi in lmodel.graph.output
    ]

    def _to_tensor(name: str, value: Any):
        return _rt.tensor_from_proto(_lnh.from_array(np.ascontiguousarray(value), name=name))

    # Build the execution plan and the reusable session once; the session
    # caches the resolved kernels after its first ``run`` call.
    plan = _rt.ExecutionPlan(lmodel.graph)
    session = _rt.RuntimeSession(plan)
    if register is not None:
        register(session)

    def _run(inputs: List[Any]) -> List[Any]:
        ctx = _rt.RuntimeContext(_rt.KernelContext(_rt.default_opset(version)))
        # Sequence and map inputs live in dedicated stores on the context, so
        # feed each graph input through the store matching its declared type.
        for name, kind, value in zip(input_names, input_kinds, inputs):
            if kind == "sequence":
                ctx.put_sequence(name, [_to_tensor(name, elem) for elem in value])
            elif kind == "map":
                ctx.put_map(name, value)
            else:
                ctx.set(name, _to_tensor(name, value))
        _rt.register_model_functions(lmodel, ctx)
        for init in initializers:
            if not ctx.has(init.name):
                ctx.set(init.name, _rt.tensor_from_proto(init), "initializer")
        session.run(ctx)
        # Read every graph output back from the store matching its type.
        results: List[Any] = []
        for name, kind in zip(output_names, output_kinds):
            if kind == "sequence":
                results.append(
                    [_runtime_tensor_to_numpy(_rt, _lnh, t) for t in ctx.get_sequence(name)]
                )
            elif kind == "map":
                results.append(ctx.get_map(name))
            else:
                results.append(_runtime_tensor_to_numpy(_rt, _lnh, ctx.get(name)))
        return results

    return _run


def _make_onnx_light_reference_runner(
    model, register: Optional[Callable[[Any], Any]] = None
) -> Callable[[List[Any]], List[Any]]:
    from onnx_light.onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model.SerializeToString())
    if register is not None:
        register(evaluator)
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


def _make_onnx_light_runner(model) -> Callable[[List[Any]], List[Any]]:
    """Build the onnx-light runner for ``model``.

    Prefers the reusable ``RuntimeSession`` execution path (init kernels once,
    run repeatedly). Falls back to the ``ReferenceEvaluator`` wrapper when the
    low-level runtime bindings are unavailable (older onnx-light builds) or the
    model cannot be prepared through them.

    Some models build a ``RuntimeSession`` successfully but only fail once a
    kernel actually runs (e.g. ``CausalConvWithState`` or ops that the
    ``RuntimeSession`` path rejects but the reference implementation accepts).
    To ensure onnx-light still runs whenever it can, the returned runner also
    falls back to the ``ReferenceEvaluator`` on the *first* run that raises,
    then replays subsequent runs through the reference evaluator.
    """
    try:
        session_runner = _make_onnx_light_runtime_session_runner(model)
    except (ImportError, AttributeError, TypeError, ValueError):
        return _make_onnx_light_reference_runner(model)

    state: Dict[str, Any] = {"runner": session_runner}

    def _run(inputs: List[Any]) -> List[Any]:
        runner = state["runner"]
        try:
            return runner(inputs)
        except Exception:  # noqa: BLE001
            if runner is not session_runner:
                # Already on the reference evaluator; propagate the failure.
                raise
            # The RuntimeSession path cannot execute this model. Fall back to
            # the reference evaluator so onnx-light still runs the model.
            reference_runner = _make_onnx_light_reference_runner(model)
            state["runner"] = reference_runner
            return reference_runner(inputs)

    return _run


def _make_onnx_light_cpu_runner(model) -> Callable[[List[Any]], List[Any]]:
    """Build the onnx-light runner with the ``onnx-light-cpu`` SIMD kernels active.

    ``onnx-light-cpu`` ships SIMD-accelerated ``Abs``/``Exp``/``Log``/``Gemm``/
    ``Not`` kernels that override the corresponding built-in entries for the
    default ONNX domain. The kernels are installed *on this benchmark's
    session* — not process-wide — by passing the freshly created
    ``RuntimeSession`` to :func:`onnx_light_cpu.register_kernels` through the
    session runner's ``register`` hook. Scoping the registration to the session
    keeps the plain ``onnx-light`` backend running its built-in kernels and lets
    the model run through the exact same ``RuntimeSession`` execution path, so
    the only difference measured is the SIMD kernels themselves.

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
    """
    import onnx_light_cpu

    runner = _make_onnx_light_runtime_session_runner(
        model, register=onnx_light_cpu.register_kernels
    )

    clear_used_kernel_names = getattr(onnx_light_cpu, "clear_used_kernel_names", None)
    used_kernel_names = getattr(onnx_light_cpu, "used_kernel_names", None)
    registered_kernel_names = getattr(onnx_light_cpu, "registered_kernel_names", None)
    if clear_used_kernel_names is None or used_kernel_names is None:
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

    checked = {"done": False}

    def _run_checked(inputs: List[Any]) -> List[Any]:
        if checked["done"]:
            return runner(inputs)
        clear_used_kernel_names()
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
        checked["done"] = True
        return outputs

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

    # ``onnx_light_cpu`` installs its SIMD kernels into onnx-light's shared C++
    # dispatch table globally and irreversibly (there is no un-register hook), so
    # it is deferred to a second pass: the built-in ``onnx_light`` backend is
    # timed for every test first to keep that baseline unaffected by the
    # registration.
    deferred = "onnx_light_cpu"
    first_pass_backends = [b for b in BENCHMARK_BACKENDS if b != deferred]

    per_test: List[Dict[str, Any]] = []
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
        for backend in first_pass_backends:
            results[backend] = _benchmark(name, model, data_sets, backend)

        per_test.append(
            {"name": name, "tag": tag, "graph": graph, "results": results}
        )

        if (idx + 1) % 50 == 0:
            _log(f"Benchmarked {idx + 1}/{len(tests)} tests (built-in backends).")

    # Second pass: the ``onnx_light_cpu`` backend registers its SIMD kernels on
    # first use, after which every onnx-light run dispatches to them.
    if deferred in BENCHMARK_BACKENDS:
        for idx, (test, entry) in enumerate(zip(tests, per_test)):
            entry["results"][deferred] = _benchmark(
                entry["name"], test["model"], test["data_sets"], deferred
            )
            if (idx + 1) % 50 == 0:
                _log(f"Benchmarked {idx + 1}/{len(tests)} tests (onnx-light-cpu).")

    rows: List[Dict[str, Any]] = [
        _row_from_results(
            entry["name"], entry["results"], tag=entry["tag"], graph=entry["graph"]
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
