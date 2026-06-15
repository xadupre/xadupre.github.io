"""Record the backend test coverage of ``onnxruntime`` and the ONNX
Python reference implementation.

The script walks every backend test bundled with the installed
``onnx-light`` package (collected via
``onnx_light.onnx_lib.backend.test.case.collect_test_case``), runs each one
against:

* ``onnxruntime`` (CPU execution provider),
* the ONNX Python reference implementation (``onnx.reference``) and
* the ``onnx-light`` reference implementation backed by the C++
  ``KernelDispatchTable`` (``onnx_light.onnx.reference``),

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
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

BACKENDS: Tuple[str, ...] = ("onnxruntime", "reference", "onnx_light")

# Package whose version is recorded alongside the ``last_pass`` date for
# each backend. ``onnxruntime`` runs the model with the ``onnxruntime``
# package, the ``reference`` implementation lives in ``onnx`` and
# ``onnx_light`` ships its own C++-backed reference evaluator.
BACKEND_PACKAGE: Dict[str, str] = {
    "onnxruntime": "onnxruntime",
    "reference": "onnx",
    "onnx_light": "onnx_light",
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
    for name in ("onnx", "onnxruntime", "onnx_light", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string representation of an error."""
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > 300:
        text = text[:297] + "..."
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
    """Convert an ``onnx-light`` tensor / numpy array to a numpy array.

    ``arr`` can either be an ``onnx-light`` ``TensorProto`` (converted by
    round-tripping its serialised bytes through ``onnx.TensorProto``) or
    a plain numpy-compatible value, in which case ``numpy.asarray`` is
    used.
    """
    import numpy as np

    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "SerializeToString"):
        import onnx
        from onnx import numpy_helper

        tensor = onnx.TensorProto()
        tensor.ParseFromString(arr.SerializeToString())
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
    cases = collect_test_case()
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
        if model is None and existing_dir:
            # ``real`` cases (large models fetched on demand) only carry
            # a ``model_dir``; load the model + data sets into memory so
            # the runner side keeps a single in-memory contract.
            import onnx

            model = onnx.load(os.path.join(str(existing_dir), "model.onnx"))
            data_sets = _load_test_data_sets(str(existing_dir), model)
        if model is None:
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
            return (
                f"{label} length mismatch: "
                f"expected {len(exp)}, got {len(act)}"
            )
        for k, (sub_exp, sub_act) in enumerate(zip(exp, act)):
            msg = _compare_value(sub_exp, sub_act, rtol, atol, f"{label}[{k}]")
            if msg is not None:
                return msg
        return None

    exp_arr = np.asarray(exp)
    act_arr = np.asarray(act)
    if exp_arr.shape != act_arr.shape:
        return (
            f"{label} shape mismatch: "
            f"expected {exp_arr.shape}, got {act_arr.shape}"
        )
    if exp_arr.dtype.kind in ("U", "S", "O") or act_arr.dtype.kind in (
        "U",
        "S",
        "O",
    ):
        if not np.array_equal(exp_arr, act_arr):
            return f"{label} value mismatch"
        return None
    try:
        np.testing.assert_allclose(
            act_arr, exp_arr, rtol=rtol, atol=atol, equal_nan=True
        )
    except AssertionError as exc:
        return f"{label} mismatch ({_stringify_error(exc)})"
    return None


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


def _fixed_point_int_range(dtype) -> Optional[Tuple[int, int]]:
    """Return ``(min, max)`` for a fixed-point integer ``dtype``.

    Handles both the standard NumPy integer types and the sub-byte integer
    types (``int2``/``uint2``/``int4``/``uint4``) provided by ``ml_dtypes``.
    Returns ``None`` for any non-integer dtype (floating point, ``float8``,
    ``float4``, bool, string, ...).
    """
    import numpy as np

    dtype = np.dtype(dtype)
    try:
        info = np.iinfo(dtype)
        return int(info.min), int(info.max)
    except (ValueError, TypeError):
        pass
    try:
        import ml_dtypes

        info = ml_dtypes.iinfo(dtype.type)
        return int(info.min), int(info.max)
    except Exception:  # noqa: BLE001 - not a sub-byte integer dtype
        return None


def _normalize_undefined_cast_outputs(
    model: Any,
    inputs: List[Any],
    expected: List[Any],
    actual: List[Any],
) -> Tuple[List[Any], List[Any]]:
    """Neutralise output elements whose value is undefined per the spec.

    The ONNX ``Cast`` specification states that casting a floating point
    value to a fixed-point (integer) type is *undefined* when the value is
    out of the target type's range. Backends therefore legitimately produce
    different results for those elements (for instance the ONNX reference
    implementation wraps around while ``onnx-light`` saturates), which used
    to flag ``onnx-light`` as failing tests such as
    ``test_cast_FLOAT_to_INT2`` even though it works correctly.

    For a single elementwise ``Cast``/``CastLike`` node casting floats to a
    fixed-point integer type, the out-of-range (and non-finite) output
    elements are zeroed in both the expected and actual tensors so the
    comparison ignores them. The ``(expected, actual)`` pair is returned
    unchanged when the model is not such a cast or has no undefined element.
    """
    import numpy as np

    try:
        nodes = list(model.graph.node)
    except Exception:  # noqa: BLE001 - defensive, model may be unusual
        return expected, actual
    if len(nodes) != 1 or nodes[0].op_type not in ("Cast", "CastLike"):
        return expected, actual
    if not inputs or len(expected) != 1 or len(actual) != 1:
        return expected, actual

    exp = expected[0]
    act = actual[0]
    if not isinstance(exp, np.ndarray) or not isinstance(act, np.ndarray):
        return expected, actual

    rng = _fixed_point_int_range(exp.dtype)
    if rng is None:
        return expected, actual

    src = np.asarray(inputs[0])
    if src.dtype.kind != "f" or src.shape != exp.shape or act.shape != exp.shape:
        return expected, actual

    lo, hi = rng
    src_f = src.astype(np.float64)
    # Casting from float to integer truncates toward zero; an element is
    # well-defined only when that truncated value fits the target range.
    truncated = np.trunc(src_f)
    defined = np.isfinite(src_f) & (truncated >= lo) & (truncated <= hi)
    if defined.all():
        return expected, actual

    exp_fixed = exp.copy()
    act_fixed = act.copy()
    exp_fixed[~defined] = np.zeros((), dtype=exp.dtype)
    act_fixed[~defined] = np.zeros((), dtype=act.dtype)
    new_expected = list(expected)
    new_actual = list(actual)
    new_expected[0] = exp_fixed
    new_actual[0] = act_fixed
    return new_expected, new_actual


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
    """
    from onnx_light.onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model.SerializeToString())
    input_names = _model_input_names(model)

    def _run(inputs: List[Any]) -> List[Any]:
        feeds = {name: value for name, value in zip(input_names, inputs)}
        return list(evaluator.run(None, feeds))

    return _run


_BACKEND_FACTORIES: Dict[str, Callable[[Any], Callable[[List[Any]], List[Any]]]] = {
    "onnxruntime": _run_with_onnxruntime,
    "reference": _run_with_reference,
    "onnx_light": _run_with_onnx_light,
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
        }

    if not data_sets:
        return {
            "success": False,
            "error": "no test_data_set_* directory found",
            "error_step": "load",
        }

    try:
        runner = factory(model)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "load",
        }

    for inputs, expected in data_sets:
        try:
            actual = runner(inputs)
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "error": _stringify_error(exc),
                "error_step": "run",
            }
        expected, actual = _normalize_undefined_cast_outputs(
            model, inputs, expected, actual
        )
        mismatch = _compare_outputs(expected, actual, rtol=rtol, atol=atol)
        if mismatch is not None:
            return {
                "success": False,
                "error": mismatch,
                "error_step": "compare",
            }
    return {"success": True, "error": "", "error_step": ""}


def _row_from_results(
    name: str,
    results: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]] = None,
    versions: Optional[Dict[str, str]] = None,
    now_iso: Optional[str] = None,
    tag: str = "",
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

    rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {
        backend: {"pass": 0, "fail": 0} for backend in BACKENDS
    }
    for idx, test in enumerate(tests):
        name = test["name"]
        model = test["model"]
        data_sets = test["data_sets"]
        results: Dict[str, Dict[str, Any]] = {}
        for backend in BACKENDS:
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
        rows.append(
            _row_from_results(
                name,
                results,
                previous=previous_rows.get(name),
                versions=version_map,
                now_iso=now_iso,
                tag=str(test.get("tag", "") or ""),
            )
        )
        if (idx + 1) % 50 == 0:
            _log(f"Ran {idx + 1}/{len(tests)} tests.")

    return {
        "date": now_iso,
        "kind": kind,
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
