"""Record the backend test coverage of ``onnxruntime`` and the ONNX
Python reference implementation.

The script walks every backend test bundled with the installed
``onnx-light`` package (collected via
``onnx_light.onnx_lib.backend.test.case.collect_test_case``), runs each one
against:

* ``onnxruntime`` (CPU execution provider),
* the ONNX Python reference implementation (``onnx.reference``) and
* the ``onnx-light`` reference implementation backed by the C++
  ``KernelDispatchTable`` (``onnx_light.onnx.reference``), and
  * ``onnx-light`` with the optimized ``onnx-light-cpu`` kernels registered,

and records whether the produced outputs match the expected ones. By
default both the ``node`` (single-operator) and ``model`` (multi-node,
including the ``test_cc_shape_inference_*`` family tagged ``inference``)
backend test groups are exercised. The resulting per-test status is
persisted to ``cache_data/onnx-light/backend_test_coverage.json``. The
dashboard at ``dashboard/onnx-light/backend-test-coverage.html``
consumes that file to render the table and pass ratio requested in the
tracking issue.

Usage::

    python scripts/record_onnx_backend_test_coverage.py [--cache-dir DIR]
        [--kind node,model] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing
import os
import queue
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend_test_metadata import kind_name, tag_name

BACKENDS: Tuple[str, ...] = (
    "onnxruntime",
    "reference",
    "onnx_light",
    "onnx_light_cpu",
)

# Package whose version is recorded alongside the ``last_pass`` date for
# each backend. ``onnxruntime`` runs the model with the ``onnxruntime``
# package, the ``reference`` implementation lives in ``onnx`` and
# ``onnx_light`` ships its own C++-backed reference evaluator.
BACKEND_PACKAGE: Dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "reference": "onnx",
    "onnx_light": "onnx_light",
    "onnx_light_cpu": "onnx_light_cpu",
}

# Default numerical tolerances when comparing produced outputs with the
# expected ones. ``onnxruntime`` and the reference implementation are not
# always bit-identical (different math libraries, different summation
# orders, ...), so we use a generous tolerance that still catches real
# regressions.
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-4

# Default backend test groups to run. ``node`` covers the single-node
# operator tests; ``model`` covers the multi-node models bundled with
# ``onnx-light`` (in particular the ``test_cc_shape_inference_*`` family
# tagged ``shape``/``inference``/``local_function``) so the dashboard
# also reports backend-execution status for the shape-inference test
# cases requested by issue #352.
DEFAULT_KINDS: Tuple[str, ...] = ("node", "model")
DEFAULT_KIND: str = ",".join(DEFAULT_KINDS)


def _normalize_kinds(kind) -> Tuple[str, ...]:
    """Normalize a ``kind`` filter into a tuple of non-empty kind names.

    ``kind`` may be ``None``, an empty string (no filter), a single kind
    name, a comma-separated list of kind names or an iterable of kind
    names. Whitespace is stripped and duplicates are removed while
    preserving the first-seen order.
    """
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
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


_ERROR_MAX_LEN = 300
_ERROR_ELLIPSIS = " ... "
# When an error line is longer than ``_ERROR_MAX_LEN`` we keep this many
# characters from the front (enough to identify *what* failed) and fill the
# rest with the tail, because backends such as ``onnxruntime`` append the
# human-readable cause (e.g. "inconsistent total_sequence_length") *after* a
# long file path and C++ type signature. Keeping only the head would hide it.
_ERROR_HEAD_LEN = 180


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string representation of an error.

    Long single-line errors are truncated in the middle rather than at the
    end so that both the head (which usually identifies the failing node /
    operator) and the tail (which often carries the actual cause) survive.
    ``onnxruntime`` in particular reports the informative status message at the
    very end of a long line, behind a verbose C++ type signature, so a plain
    head truncation would drop the part that explains *why* a test fails.
    """
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > _ERROR_MAX_LEN:
        tail_len = _ERROR_MAX_LEN - _ERROR_HEAD_LEN - len(_ERROR_ELLIPSIS)
        head = text[:_ERROR_HEAD_LEN].rstrip()
        tail = text[-tail_len:].lstrip()
        text = head + _ERROR_ELLIPSIS + tail
    return text


def _onnx_light_model_to_onnx(model):
    """Convert an ``onnx-light`` ``ModelProto`` into an ``onnx`` ``ModelProto``.

    ``onnx-light`` exposes its own (protobuf-free) ``ModelProto`` whose
    wire format is compatible with the official ``onnx`` package. The
    conversion goes through ``SerializeToString`` / ``ParseFromString``
    so the returned object is a real ``onnx.ModelProto`` that
    ``onnxruntime`` and ``onnx.reference`` know how to consume.
    """
    import onnx

    if isinstance(model, onnx.ModelProto):
        return model
    out = onnx.ModelProto()
    out.ParseFromString(model.SerializeToString())
    return out


def _onnx_light_tensor_to_numpy(arr):
    """Convert an ``onnx-light`` tensor / numpy array to a numpy value.

    ``arr`` can either be an ``onnx-light`` ``TensorProto``, ``SequenceProto``
    or ``OptionalProto`` (converted by round-tripping its serialised bytes
    through the matching ``onnx`` proto) or a plain numpy-compatible value, in
    which case ``numpy.asarray`` is used. Sequence protos decode to a list of
    numpy arrays and optional protos to either ``None`` or a numpy value, which
    mirrors how ``onnxruntime`` / ``onnx.reference`` represent the corresponding
    computed outputs so the comparison stays apples-to-apples.
    """
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


def discover_node_tests(kind=DEFAULT_KIND) -> List[Dict[str, Any]]:
    """Return ``[{"name", "model", "data_sets"}, ...]`` for every backend test.

    The tests are loaded from ``onnx_light.onnx_lib.backend.test.case`` which
    ships with the installed ``onnx-light`` package via
    :func:`onnx_light.onnx_lib.backend.test.case.collect_test_case`. ``kind``
    selects the test groups; it can be a single kind name (``"node"``,
    ``"simple"``, ``"pytorch-converted"``, ``"pytorch-operator"``,
    ``"real"``, ``"model"``...), a comma-separated list of kind names
    or any iterable of kind names. A test case is retained when its
    ``kind`` attribute matches any of the requested kinds. Passing an
    empty value disables kind filtering and keeps every collected test.

    The default :data:`DEFAULT_KIND` covers both ``node`` (the single
    operator tests exercised by ``onnx-light``'s reference
    implementation) and ``model`` (multi-node models, in particular the
    ``test_cc_shape_inference_*`` family tagged ``inference`` that the
    onnx-light dashboard requested via issue #352).

    Test cases collected by ``onnx-light`` carry their ``ModelProto`` and
    expected input / output tensors in memory. They are converted to the
    official ``onnx`` types via :func:`_onnx_light_model_to_onnx` /
    :func:`_onnx_light_tensor_to_numpy` and returned in memory so the
    rest of the pipeline never has to touch the filesystem. ``real``
    test cases that only carry a ``model_dir`` are loaded from disk on
    the fly into the same in-memory shape.
    """
    from onnx_light.onnx_lib.backend.test.case import collect_test_case

    kinds = _normalize_kinds(kind)
    cases = collect_test_case(include_big=True)
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_kind = kind_name(getattr(tc, "kind", None))
        if kinds and case_kind not in kinds:
            continue
        model = getattr(tc, "model", None)
        data_sets = getattr(tc, "data_sets", None) or []
        existing_dir = getattr(tc, "model_dir", None)
        if existing_dir and (model is None or not data_sets):
            # ``real`` cases (large models fetched on demand) may carry only a
            # ``model_dir``, or may carry an in-memory model but no data sets
            # (e.g. the tiny-LLM shape-inference tests). Load whichever pieces
            # are missing from disk so the runner keeps a single in-memory
            # contract.
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
        tag = tag_name(getattr(tc, "tag", None))
        discovered.append(
            {
                "name": str(name),
                "model": onnx_model,
                "data_sets": converted_data_sets,
                "tag": tag,
            }
        )
    discovered.sort(key=lambda d: d["name"])
    return discovered


def _load_proto(path: str, type_proto: Any = None):
    """Load a serialised proto from ``path`` as a numpy value.

    ``type_proto`` is the matching ``onnx.TypeProto`` taken from the
    model's graph input/output; it determines whether the file stores a
    ``TensorProto``, a ``SequenceProto`` (decoded to a list of numpy
    arrays) or an ``OptionalProto`` (decoded to either ``None`` or a
    numpy value). When ``type_proto`` is missing or carries no usable
    field, the file is parsed as a ``TensorProto`` for backwards
    compatibility.
    """
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


# Backwards-compatible alias: historically only tensors were supported.
_load_tensor = _load_proto


def _load_test_data_sets(
    model_dir: str, model: Any = None
) -> List[Tuple[List[Any], List[Any]]]:
    """Return ``[(inputs, expected_outputs), ...]`` for ``model_dir``.

    Each test directory contains one or more ``test_data_set_<n>``
    sub-directories with ``input_<i>.pb`` and ``output_<j>.pb`` files
    storing serialised protos. Most files hold ``TensorProto`` messages,
    but sequence and optional operator tests store ``SequenceProto`` and
    ``OptionalProto`` messages instead. When ``model`` is provided its
    graph input/output ``TypeProto`` entries are used to decode each file
    with the right proto type; otherwise every file is parsed as a
    ``TensorProto``.
    """
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
    """Return an SVG rendering of ``model``'s graph.

    The conversion is delegated to :func:`onnx_light.tools.to_svg` so the
    dashboard reuses the canonical ``onnx-light`` renderer instead of
    re-implementing the graph layout in JavaScript. The returned mapping
    stores the self-contained ``<svg>`` document under the ``"svg"`` key,
    which the dashboard embeds directly when unfolding a test.
    """
    from onnx_light.tools import to_svg

    return {"svg": to_svg(model)}


# ``ml_dtypes`` packed sub-byte integer dtypes. They expose ``kind == "V"``
# (void) rather than ``"i"`` / ``"u"`` and arithmetic on them wraps around
# inside the narrow range, so they need widening before numeric comparison.
_SUB_BYTE_INT_DTYPES: Tuple[str, ...] = ("int4", "uint4", "int2", "uint2")

# Representable ``[min, max]`` range of each packed sub-byte integer dtype.
# Used to recognise spec-compliant saturating ``Cast`` results (see
# :func:`_compare_value`).
_SUB_BYTE_INT_RANGES: Dict[str, Tuple[int, int]] = {
    "int4": (-8, 7),
    "uint4": (0, 15),
    "int2": (-2, 1),
    "uint2": (0, 3),
}


def _widen_sub_byte_int(arr):
    """Upcast packed sub-byte integer arrays to ``int64`` for comparison.

    ``int4`` / ``uint4`` / ``int2`` / ``uint2`` outputs are stored with
    ``ml_dtypes`` packed dtypes whose element-wise subtraction wraps around
    modulo the narrow range. Promoting both operands to a wide signed integer
    makes :func:`numpy.testing.assert_allclose` compute exact differences (and
    therefore exact mismatch statistics). Arrays of any other dtype are
    returned unchanged.
    """
    import numpy as np

    if arr.dtype.name in _SUB_BYTE_INT_DTYPES:
        return arr.astype(np.int64)
    return arr


def _compare_value(
    exp: Any,
    act: Any,
    rtol: float,
    atol: float,
    label: str,
) -> Optional[str]:
    """Compare a single expected/actual value, recursing into sequences.

    ``exp``/``act`` may be numpy arrays (tensor outputs), Python lists
    (sequence outputs) or ``None`` (an absent optional output).
    """
    import numpy as np

    if exp is None or act is None:
        if exp is None and act is None:
            return None
        return f"{label} value mismatch: one side is None"

    if isinstance(exp, list) or isinstance(act, list):
        if not (isinstance(exp, list) and isinstance(act, list)):
            return f"{label} type mismatch: sequence vs non-sequence"
        if len(exp) != len(act):
            return f"{label} length mismatch: " f"expected {len(exp)}, got {len(act)}"
        for k, (sub_exp, sub_act) in enumerate(zip(exp, act)):
            msg = _compare_value(sub_exp, sub_act, rtol, atol, f"{label}[{k}]")
            if msg is not None:
                return msg
        return None

    exp_arr = np.asarray(exp)
    act_arr = np.asarray(act)
    if exp_arr.shape != act_arr.shape:
        return (
            f"{label} shape mismatch: " f"expected {exp_arr.shape}, got {act_arr.shape}"
        )
    if exp_arr.dtype.kind in ("U", "S", "O") or act_arr.dtype.kind in (
        "U",
        "S",
        "O",
    ):
        if not np.array_equal(exp_arr, act_arr):
            return f"{label} value mismatch"
        return None
    # Sub-byte integer outputs (``int4`` / ``uint4`` / ``int2`` / ``uint2``)
    # are carried by ``ml_dtypes`` packed dtypes. ``assert_allclose`` computes
    # the element-wise difference in the narrow dtype, which overflows modulo
    # 16 / 4 and reports misleading statistics (e.g. an absolute difference of
    # 15 wraps to 1). Widen both sides to a signed 64-bit integer so the
    # comparison and the reported diff are exact.
    sub_byte_range = _SUB_BYTE_INT_RANGES.get(exp_arr.dtype.name)
    exp_arr = _widen_sub_byte_int(exp_arr)
    act_arr = _widen_sub_byte_int(act_arr)
    if sub_byte_range is not None:
        # The ONNX ``Cast`` spec leaves float -> fixed-point conversions
        # *undefined* when the source value is out of range. The bundled
        # backend test data wraps around (numpy ``astype`` semantics) while a
        # spec-compliant runtime may instead saturate to the representable
        # bound. A disagreement is therefore only a genuine failure when the
        # actual value is not such a saturation bound: any element equal to the
        # dtype min/max is a valid saturating result of some out-of-range
        # source consistent with the wrapped reference value.
        lo, hi = sub_byte_range
        diff_mask = exp_arr != act_arr
        if diff_mask.any():
            saturating = (act_arr == lo) | (act_arr == hi)
            if not (diff_mask & ~saturating).any():
                return None
    try:
        np.testing.assert_allclose(
            act_arr, exp_arr, rtol=rtol, atol=atol, equal_nan=True
        )
    except AssertionError as exc:
        return f"{label} mismatch ({_summarize_allclose_error(exc)})"
    return None


def _summarize_allclose_error(exc: AssertionError) -> str:
    """Summarise a ``numpy.testing.assert_allclose`` failure on one line.

    ``assert_allclose`` reports a generic ``Not equal to tolerance`` header
    and puts the informative statistics (how many elements differ and by how
    much) on the following lines. :func:`_stringify_error` keeps only the
    first line, which hides those details, so this helper gathers the
    statistic lines into a single concise, precise message.
    """
    wanted = (
        "Mismatched elements",
        "Max absolute difference",
        "Max relative difference",
    )
    parts = [
        line.strip()
        for line in str(exc).splitlines()
        if line.strip().startswith(wanted)
    ]
    if not parts:
        return _stringify_error(exc)
    summary = "; ".join(parts)
    if len(summary) > 300:
        summary = summary[:297] + "..."
    return summary


def _compare_outputs(
    expected: List[Any],
    actual: List[Any],
    rtol: float,
    atol: float,
) -> Optional[str]:
    """Return ``None`` if the outputs match, otherwise an error string."""
    if len(expected) != len(actual):
        return f"output count mismatch: " f"expected {len(expected)}, got {len(actual)}"
    for idx, (exp, act) in enumerate(zip(expected, actual)):
        msg = _compare_value(exp, act, rtol, atol, f"output {idx}")
        if msg is not None:
            return msg
    return None


def _run_with_onnxruntime(model) -> Callable[[List[Any]], List[Any]]:
    import onnxruntime

    sess = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(sess.run(None, feeds))

    return _run


def _run_with_reference(model) -> Callable[[List[Any]], List[Any]]:
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    input_names = _model_input_names(model)

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(evaluator.run(None, feeds))

    return _run


def _run_with_onnx_light(model) -> Callable[[List[Any]], List[Any]]:
    """Run ``model`` with ``onnx_light.onnx.reference.ReferenceEvaluator``.

    ``onnx_light`` ships its own ``ModelProto`` (and matching
    ``ReferenceEvaluator``) that is wire-format compatible with the
    official ``onnx`` package but distinct at the Python type level. The
    in-memory ``onnx.ModelProto`` produced by :func:`discover_node_tests`
    is therefore serialised and re-parsed by the evaluator so it sees a
    proto of its own type.

    ``ReferenceEvaluator`` executes the model through onnx-light's
    ``ExecutionPlan`` / ``RuntimeSession`` machinery (the same reusable
    runtime session exercised by the C++ backend), converting the feed
    dictionary to/from the runtime ``Tensor`` type.
    """
    from onnx_light.onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model.SerializeToString())
    input_names = _model_input_names(model)
    raw_evaluator_input_names = getattr(evaluator, "input_names", None)
    evaluator_input_names = None
    if raw_evaluator_input_names:
        evaluator_input_names = {
            str(getattr(name, "name", name)) for name in raw_evaluator_input_names
        }

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


_CPU_KERNELS_REGISTERED = False


def _run_with_onnx_light_cpu(model) -> Callable[[List[Any]], List[Any]]:
    """Run with onnx-light-cpu and require an optimized kernel to be used."""
    global _CPU_KERNELS_REGISTERED

    from onnx_light_cpu import (
        clear_used_kernel_names,
        register_kernels,
        used_kernel_names,
    )
    from onnx_light_cpu.onnx_py._cpuregister import set_kernel_usage_recording

    if not _CPU_KERNELS_REGISTERED:
        register_kernels()
        _CPU_KERNELS_REGISTERED = True
    run = _run_with_onnx_light(model)

    def _run(inputs: List[Any]) -> List[Any]:
        set_kernel_usage_recording(True)
        clear_used_kernel_names()
        try:
            outputs = run(inputs)
            used = used_kernel_names()
        finally:
            set_kernel_usage_recording(False)
        if not used:
            raise RuntimeError("no onnx-light-cpu kernel ran")
        return outputs

    return _run


_BACKEND_FACTORIES: Dict[str, Callable[[Any], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _run_with_onnxruntime,
    "reference": _run_with_reference,
    "onnx_light": _run_with_onnx_light,
    "onnx_light_cpu": _run_with_onnx_light_cpu,
}


def run_test_with_backend(
    model: Any,
    data_sets: List[Tuple[List[Any], List[Any]]],
    backend: str,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Dict[str, Any]:
    """Run a single backend test against ``backend``.

    ``model`` is an in-memory ``onnx.ModelProto`` and ``data_sets`` is
    the list of ``(inputs, expected_outputs)`` numpy arrays produced by
    :func:`discover_node_tests`. The returned dictionary has the
    following structure::

        {"success": bool, "error": str, "error_step": str}

    ``error_step`` is either ``"load"`` (failure when instantiating the
    backend session/evaluator), ``"run"`` (failure when executing the
    model) or ``"compare"`` (failure when comparing outputs).
    """
    factory = _BACKEND_FACTORIES.get(backend)
    if factory is None:
        return {
            "success": False,
            "error": f"unknown backend: {backend}",
            "error_step": "load",
            "elapsed_s": 0.0,
        }

    if not data_sets:
        return {
            "success": False,
            "error": "no test_data_set_* directory found",
            "error_step": "load",
            "elapsed_s": 0.0,
        }

    t0 = time.perf_counter()
    try:
        runner = factory(model)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "load",
            "elapsed_s": time.perf_counter() - t0,
        }

    for inputs, expected in data_sets:
        try:
            actual = runner(inputs)
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": _stringify_error(exc),
                "error_step": "run",
                "elapsed_s": time.perf_counter() - t0,
            }
        mismatch = _compare_outputs(expected, actual, rtol=rtol, atol=atol)
        if mismatch is not None:
            return {
                "success": False,
                "error": mismatch,
                "error_step": "compare",
                "elapsed_s": time.perf_counter() - t0,
            }
    return {
        "success": True,
        "error": "",
        "error_step": "",
        "elapsed_s": time.perf_counter() - t0,
    }


def _cpu_worker(task_queue, result_queue, rtol: float, atol: float) -> None:
    """Run onnx-light-cpu tests until stopped or a native kernel crashes."""
    while True:
        task = task_queue.get()
        if task is None:
            return
        index, model, data_sets = task
        result_queue.put(
            (
                index,
                run_test_with_backend(
                    model, data_sets, "onnx_light_cpu", rtol=rtol, atol=atol
                ),
            )
        )


def _run_cpu_tests_isolated(
    tests: List[Dict[str, Any]], rtol: float, atol: float
) -> List[Dict[str, Any]]:
    """Run native CPU kernels out of process so one crash does not abort the job."""
    context = multiprocessing.get_context("spawn")
    results: List[Dict[str, Any]] = []
    process = None
    task_queue = None
    result_queue = None

    def stop_worker() -> None:
        nonlocal process, task_queue, result_queue
        worker = process
        tasks = task_queue
        results_queue = result_queue
        process = task_queue = result_queue = None
        if worker is not None:
            if worker.is_alive():
                tasks.put(None)
                worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            tasks.close()
            results_queue.close()

    try:
        for index, test in enumerate(tests):
            if process is None or not process.is_alive():
                stop_worker()
                task_queue = context.Queue()
                result_queue = context.Queue()
                process = context.Process(
                    target=_cpu_worker,
                    args=(task_queue, result_queue, rtol, atol),
                )
                process.start()

            task_queue.put((index, test["model"], test["data_sets"]))
            while True:
                try:
                    result_index, info = result_queue.get(timeout=0.1)
                    if result_index != index:
                        raise RuntimeError(
                            f"unexpected onnx-light-cpu result index {result_index}"
                        )
                    results.append(info)
                    break
                except queue.Empty:
                    if process.is_alive():
                        continue
                    process.join()
                    results.append(
                        {
                            "success": False,
                            "error": (
                                "onnx-light-cpu worker crashed with exit code "
                                f"{process.exitcode}"
                            ),
                            "error_step": "run",
                            "elapsed_s": 0.0,
                        }
                    )
                    stop_worker()
                    _log(
                        f"onnx-light-cpu worker crashed while running "
                        f"{test['name']}; continuing with the next test."
                    )
                    break
            if (index + 1) % 50 == 0:
                _log(f"Ran {index + 1}/{len(tests)} tests on onnx-light-cpu.")
        return results
    finally:
        stop_worker()


def _row_from_results(
    name: str,
    results: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]] = None,
    versions: Optional[Dict[str, str]] = None,
    now_iso: Optional[str] = None,
    tag: str = "",
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a dashboard row, carrying over per-backend ``last_pass`` info.

    For every backend, when the current run succeeds, ``last_pass_date``
    is set to ``now_iso`` and ``last_pass_version`` to the recorded
    version of the matching package (``onnxruntime`` or ``onnx``). When
    the current run fails, the corresponding values are carried over from
    ``previous`` (the row from a previous snapshot, if any) so the
    dashboard can report when the test last passed.
    """
    versions = versions or {}
    previous = previous or {}
    row: Dict[str, Any] = {"name": name}
    if tag:
        row["tag"] = tag
    elif previous.get("tag"):
        row["tag"] = previous["tag"]
    if graph is not None:
        row["graph"] = graph
    elif previous.get("graph"):
        row["graph"] = previous["graph"]
    total_elapsed: float = 0.0
    for backend in BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        row[backend] = success
        error = _stringify_error(info.get("error"))
        if error:
            row[f"{backend}_error"] = error
        step = info.get("error_step") or ""
        if step:
            row[f"{backend}_error_step"] = step
        elapsed = info.get("elapsed_s")
        if elapsed is not None:
            row[f"{backend}_elapsed_s"] = round(float(elapsed), 6)
            total_elapsed += float(elapsed)
        if success and now_iso is not None:
            row[f"{backend}_last_pass_date"] = now_iso
            pkg = BACKEND_PACKAGE.get(backend)
            version = versions.get(pkg) if pkg else None
            if version:
                row[f"{backend}_last_pass_version"] = version
        else:
            prev_date = previous.get(f"{backend}_last_pass_date")
            if prev_date:
                row[f"{backend}_last_pass_date"] = prev_date
            prev_version = previous.get(f"{backend}_last_pass_version")
            if prev_version:
                row[f"{backend}_last_pass_version"] = prev_version
    if total_elapsed:
        row["elapsed_s"] = round(total_elapsed, 6)
    return row


def load_previous_payload(json_path: str) -> Dict[str, Any]:
    """Return the previously written payload, or an empty dict if absent.

    The recorder uses this to carry over ``last_pass_date`` /
    ``last_pass_version`` entries for tests that fail in the current run
    but passed in a prior one. Any unreadable / malformed file is treated
    as missing so a fresh snapshot can always be produced.
    """
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


def _index_previous_rows(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = payload.get("tests") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            name = row.get("name")
            if isinstance(name, str):
                indexed[name] = row
    return indexed


def build_payload(
    kind: str = "node",
    limit: Optional[int] = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    discover: Callable[[str], List[Dict[str, Any]]] = discover_node_tests,
    run: Callable[..., Dict[str, Any]] = run_test_with_backend,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
    previous: Optional[Dict[str, Any]] = None,
    isolate_cpu: bool = False,
) -> Dict[str, Any]:
    """Discover all tests, run them on every backend and return a payload."""
    if versions is None:
        versions = collect_versions
    tests = discover(kind)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    _log(f"Discovered {len(tests)} {kind} backend tests.")

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)
    version_map = versions()
    previous_rows = _index_previous_rows(previous or {})

    entries: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {
        backend: {"pass": 0, "fail": 0} for backend in BACKENDS
    }
    baseline_backends = tuple(b for b in BACKENDS if b != "onnx_light_cpu")
    for idx, test in enumerate(tests):
        name = test["name"]
        model = test["model"]
        data_sets = test["data_sets"]
        results: Dict[str, Dict[str, Any]] = {}
        try:
            graph = build_graph(model)
        except Exception:  # noqa: BLE001 - graph is a best-effort annotation
            graph = None
        for backend in baseline_backends:
            try:
                info = run(model, data_sets, backend, rtol=rtol, atol=atol)
            except Exception as exc:  # noqa: BLE001
                # Defensive guard: the runner is expected to capture its
                # own exceptions, but we never want a single broken test
                # to abort the whole snapshot.
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
            bucket = "pass" if info.get("success") else "fail"
            totals[backend][bucket] += 1
        entries.append(
            {
                "test": test,
                "results": results,
                "graph": graph,
            }
        )
        if (idx + 1) % 50 == 0:
            _log(f"Ran {idx + 1}/{len(tests)} tests on baseline backends.")

    # Kernel registration is process-wide and irreversible. Run every
    # unmodified onnx-light baseline first, then install onnx-light-cpu once
    # and execute its complete pass so the baseline remains trustworthy. The
    # command-line recorder uses a child process so a native kernel crash can be
    # attributed to one test without losing the rest of the snapshot.
    cpu_infos = _run_cpu_tests_isolated(tests, rtol, atol) if isolate_cpu else None
    for idx, entry in enumerate(entries):
        test = entry["test"]
        if cpu_infos is not None:
            info = cpu_infos[idx]
        else:
            try:
                info = run(
                    test["model"],
                    test["data_sets"],
                    "onnx_light_cpu",
                    rtol=rtol,
                    atol=atol,
                )
            except Exception as exc:  # noqa: BLE001
                _log(
                    f"Unhandled error for {test['name']} on onnx_light_cpu: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                info = {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "run",
                }
        entry["results"]["onnx_light_cpu"] = info
        bucket = "pass" if info.get("success") else "fail"
        totals["onnx_light_cpu"][bucket] += 1
        if cpu_infos is None and (idx + 1) % 50 == 0:
            _log(f"Ran {idx + 1}/{len(tests)} tests on onnx-light-cpu.")

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        test = entry["test"]
        rows.append(
            _row_from_results(
                test["name"],
                entry["results"],
                previous=previous_rows.get(test["name"]),
                versions=version_map,
                now_iso=now_iso,
                tag=str(test.get("tag", "") or ""),
                graph=entry["graph"],
            )
        )

    slowest = sorted(
        (r for r in rows if r.get("elapsed_s")),
        key=lambda r: r["elapsed_s"],
        reverse=True,
    )[:20]
    slowest_tests = [
        {
            k: r[k]
            for k in ["name"]
            + [f"{b}_elapsed_s" for b in BACKENDS if f"{b}_elapsed_s" in r]
            + (["elapsed_s"] if "elapsed_s" in r else [])
        }
        for r in slowest
    ]

    return {
        "date": now_iso,
        "kind": kind,
        "slowest_tests": slowest_tests,
        "tolerances": {"rtol": rtol, "atol": atol},
        "versions": version_map,
        "totals": totals,
        "tests": rows,
    }


def write_payload(json_path: str, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``json_path`` (creating parent directories)."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


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
            "Backend test group(s) to run (default: %(default)s). "
            "Accepts a single value or a comma-separated list. "
            "Common values: node, model, simple, pytorch-converted, "
            "pytorch-operator, real."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of tests executed (useful for debugging).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance for output comparison (default: %(default)s).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance for output comparison (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(args.cache_dir, "onnx-light", "backend_test_coverage.json")
    previous = load_previous_payload(json_path)
    try:
        payload = build_payload(
            kind=args.kind,
            limit=args.limit,
            rtol=args.rtol,
            atol=args.atol,
            previous=previous,
            isolate_cpu=True,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record backend test coverage: {exc}")
        traceback.print_exc()
        return 1
    write_payload(json_path, payload)
    _log(
        f"Wrote {len(payload['tests'])} test entries to {json_path} "
        f"(totals={payload['totals']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
