"""Record the shape-inference coverage of the shape-inference
implementations exercised by ``onnx-light`` (``onnx-light`` itself,
``onnx_light.onnx_optim`` — the experimental shape inference shipped
inside ``onnx-light``'s ``onnx_optim`` submodule, the official
``onnx.shape_inference`` and the standalone ``onnx-shape-inference``
package).

The script walks every backend test bundled with the installed
``onnx-light`` package (collected via
``onnx_light.backend.test.case.collect_test_case``) and keeps only the
cases whose ``tag`` is ``"inference"`` (the family of tests dedicated to
shape inference, mirroring
``unittests/backend/test_backend_with_shape_inference.py`` in the
``xadupre/onnx-light`` repository).

For each retained test case the recorded ``graph.output`` (and
intermediate ``graph.value_info``) shapes are snapshotted then stripped
from a working copy of the model. Each candidate shape-inference
implementation is invoked on that stripped model and the produced shapes
are compared with the snapshot. For every intermediate, we report
whether the runtime recovered the expected ``elem_type`` and a
compatible shape (concrete dims must match; ``-1``/symbolic dims are
tolerated).

The aggregated payload is persisted as JSON to
``cache_data/onnx-light/shape_inference_coverage.json``. The dashboard
at ``dashboard/onnx-light/shape-inference-coverage.html`` consumes that
file to render the percentage of correctly inferred shapes for each
runtime and the detailed per-intermediate results for each test.

Usage::

    python scripts/record_onnx_shape_inference_coverage.py \\
        [--cache-dir DIR] [--tag inference] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# Order matters: it drives the column order in the dashboard.
BACKENDS: Tuple[str, ...] = (
    "onnx-light",
    "onnx-light-onnx-optim",
    "onnx",
    "onnx-shape-inference",
)

# Package whose version is recorded alongside the ``last_pass`` date for
# each backend.
BACKEND_PACKAGE: Dict[str, str] = {
    "onnx-light": "onnx_light",
    "onnx-light-onnx-optim": "onnx_light",
    "onnx": "onnx",
    "onnx-shape-inference": "onnx_shape_inference",
}

# Default tag used by ``onnx-light`` to mark backend cases that are
# specifically designed to exercise shape inference.
DEFAULT_TAG = "inference"


def _log(message: str) -> None:
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
    for name in ("onnx", "onnx_light", "onnx_shape_inference", "onnx_ir", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _stringify_error(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > 300:
        text = text[:297] + "..."
    return text


def _onnx_light_model_to_onnx(model):
    """Convert an ``onnx-light`` ``ModelProto`` into an ``onnx`` ``ModelProto``."""
    import onnx

    if isinstance(model, onnx.ModelProto):
        return model
    out = onnx.ModelProto()
    out.ParseFromString(model.SerializeToString())
    return out


def _mermaid_escape(text: str) -> str:
    """Escape ``text`` so it can appear inside a Mermaid ``"..."`` label."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("\"", "&quot;")
        .replace("\n", " ")
    )


def _mermaid_dtype_name(onnx_mod: Any, dtype: int) -> str:
    if not dtype:
        return ""
    try:
        return onnx_mod.TensorProto.DataType.Name(dtype)
    except Exception:  # noqa: BLE001 - unknown enum value
        return str(dtype)


def _mermaid_format_type(onnx_mod: Any, type_proto: Any) -> str:
    """Render an ``onnx.TypeProto`` as ``DTYPE[d0,d1,...]``."""
    if type_proto is None:
        return ""
    tensor_type = getattr(type_proto, "tensor_type", None)
    if tensor_type is None or not getattr(tensor_type, "elem_type", 0):
        return ""
    dtype = _mermaid_dtype_name(onnx_mod, tensor_type.elem_type)
    dims: List[str] = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(str(dim.dim_value))
            elif dim.HasField("dim_param") and dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("?")
    return f"{dtype}[{','.join(dims)}]" if dims else dtype


def _render_model_as_mermaid(model: Any) -> str:
    """Self-contained ``onnx.ModelProto`` to Mermaid ``flowchart TD`` renderer.

    The output is a textual Mermaid graph the dashboard can render
    client-side. Each input is drawn as a stadium-shape node, each
    initializer as a cylinder, each operator as a rectangle labelled
    with its ``op_type`` (and ``name`` when present), and each graph
    output as a stadium-shape node. Edges are labelled with the tensor
    name and, when shape inference succeeds, with their inferred
    ``DTYPE[shape]``.
    """
    import onnx

    if not hasattr(model, "graph"):
        return ""

    annotated = model
    try:
        annotated = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, check_type=False
        )
    except Exception:  # noqa: BLE001 - shape inference is best-effort
        annotated = model

    graph = annotated.graph

    edge_types: Dict[str, str] = {}
    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        label = _mermaid_format_type(onnx, value_info.type)
        if label:
            edge_types[value_info.name] = label

    initializer_names = {init.name for init in graph.initializer}

    used_ids: set = set()

    def _make_id(prefix: str, name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name) or "x"
        base = f"{prefix}_{sanitized}"
        candidate = base
        index = 1
        while candidate in used_ids:
            index += 1
            candidate = f"{base}_{index}"
        used_ids.add(candidate)
        return candidate

    lines: List[str] = ["flowchart TD"]
    tensor_source: Dict[str, str] = {}

    for value_info in graph.input:
        if value_info.name in initializer_names:
            continue
        node_id = _make_id("in", value_info.name)
        type_label = edge_types.get(value_info.name, "")
        label = value_info.name + (f"<br>{type_label}" if type_label else "")
        lines.append(f'    {node_id}(["{_mermaid_escape(label)}"])')
        tensor_source[value_info.name] = node_id

    for initializer in graph.initializer:
        node_id = _make_id("init", initializer.name)
        dtype = _mermaid_dtype_name(onnx, initializer.data_type)
        dims = ",".join(str(d) for d in initializer.dims)
        label = initializer.name + (f"<br>{dtype}[{dims}]" if dtype else "")
        lines.append(f'    {node_id}[("{_mermaid_escape(label)}")]')
        tensor_source[initializer.name] = node_id

    op_ids: List[str] = []
    for index, node in enumerate(graph.node):
        node_id = _make_id("op", node.name or f"{node.op_type}_{index}")
        op_ids.append(node_id)
        label = node.op_type + (f"<br>{node.name}" if node.name else "")
        lines.append(f'    {node_id}["{_mermaid_escape(label)}"]')
        for out_name in node.output:
            if out_name:
                tensor_source.setdefault(out_name, node_id)

    output_entries: List[Tuple[str, str]] = []
    for value_info in graph.output:
        node_id = _make_id("out", value_info.name)
        output_entries.append((value_info.name, node_id))
        type_label = edge_types.get(value_info.name, "")
        label = value_info.name + (f"<br>{type_label}" if type_label else "")
        lines.append(f'    {node_id}(["{_mermaid_escape(label)}"])')

    for node_id, node in zip(op_ids, graph.node):
        for in_name in node.input:
            if not in_name:
                continue
            source_id = tensor_source.get(in_name)
            if not source_id:
                continue
            type_label = edge_types.get(in_name, "")
            edge_label = in_name + (f" : {type_label}" if type_label else "")
            lines.append(
                f'    {source_id} -- "{_mermaid_escape(edge_label)}" --> {node_id}'
            )

    for out_name, out_id in output_entries:
        source_id = tensor_source.get(out_name)
        if not source_id or source_id == out_id:
            continue
        type_label = edge_types.get(out_name, "")
        edge_label = out_name + (f" : {type_label}" if type_label else "")
        lines.append(
            f'    {source_id} -- "{_mermaid_escape(edge_label)}" --> {out_id}'
        )

    return "\n".join(lines)


def model_to_mermaid(model: Any) -> str:
    """Return a Mermaid ``flowchart TD`` string for ``model``.

    Self-contained renderer that only relies on the ``onnx`` package and
    therefore works without any optional dependency. Returns an empty
    string when ``onnx`` cannot be imported, when ``model`` is not a
    usable ``onnx.ModelProto`` or when rendering fails (graph
    visualisation is best-effort, never a hard requirement for the
    coverage data itself).
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        return ""
    try:
        return _render_model_as_mermaid(model)
    except Exception:  # noqa: BLE001 - best effort rendering
        return ""


def discover_inference_tests(tag: str = DEFAULT_TAG) -> List[Dict[str, Any]]:
    """Return the list of backend tests tagged ``tag``.

    Each entry is a dictionary ``{"name", "model", "expected", "mermaid"}``
    where ``model`` is an ``onnx.ModelProto``, ``expected`` is the list
    of snapshotted intermediates (see :func:`snapshot_intermediates`)
    and ``mermaid`` is a Mermaid ``flowchart TD`` rendering of ``model``
    (empty string when rendering fails).
    """
    from onnx_light.backend.test.case import collect_test_case

    cases = collect_test_case()
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_tag = getattr(tc, "tag", "") or ""
        if tag and case_tag != tag:
            continue
        model = getattr(tc, "model", None)
        if model is None:
            continue
        onnx_model = _onnx_light_model_to_onnx(model)
        expected = snapshot_intermediates(onnx_model)
        if not expected:
            # Nothing to validate: the test does not declare any
            # intermediate / output shape to recover, so it cannot be
            # used to score shape inference.
            continue
        inputs = snapshot_inputs(onnx_model)
        discovered.append(
            {
                "name": str(name),
                "model": onnx_model,
                "expected": expected,
                "inputs": inputs,
                "mermaid": model_to_mermaid(onnx_model),
            }
        )
    discovered.sort(key=lambda d: d["name"])
    return discovered


def _dims_of_tensor_type(tensor_type) -> Tuple[bool, List[Any]]:
    """Return ``(has_shape, dims)`` for an ``onnx`` ``TypeProto.Tensor``.

    Concrete dimensions are returned as ``int``. Named symbolic
    dimensions (``dim_param``) are returned as the parameter name
    (``str``) so the dashboard can show the dim name instead of a
    generic ``?``. Fully unknown dimensions (neither ``dim_value`` nor
    ``dim_param`` set) are represented by ``-1``. When the tensor type
    does not carry a ``shape`` field, ``has_shape`` is False and
    ``dims`` is empty.
    """
    if not tensor_type.HasField("shape"):
        return False, []
    dims: List[Any] = []
    for d in tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(str(d.dim_param))
        else:
            dims.append(-1)
    return True, dims


def snapshot_inputs(model) -> List[Dict[str, Any]]:
    """Snapshot the shape information of every graph input.

    Returns a list of dicts ``{"name", "kind", "elem_type", "has_shape",
    "shape"}`` with ``kind == "input"``. Non plain-tensor entries
    (sequence/optional/map) are skipped, matching
    :func:`snapshot_intermediates`. Graph inputs are recorded so the
    dashboard can show, alongside each test, the input shapes that were
    fed to the shape-inference runtimes.
    """
    snapshots: List[Dict[str, Any]] = []
    for vi in model.graph.input:
        if not vi.HasField("type"):
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        has_shape, dims = _dims_of_tensor_type(vi.type.tensor_type)
        snapshots.append(
            {
                "name": vi.name,
                "kind": "input",
                "elem_type": int(vi.type.tensor_type.elem_type),
                "has_shape": has_shape,
                "shape": dims,
            }
        )
    return snapshots


def snapshot_intermediates(model) -> List[Dict[str, Any]]:
    """Snapshot the shape information of every output / value_info.

    Returns a list of dicts ``{"name", "kind", "elem_type", "has_shape",
    "shape"}`` where ``kind`` is either ``"output"`` or ``"value_info"``.
    Non plain-tensor entries (sequence/optional/map) are skipped since
    they cannot be scored against a simple ``elem_type`` + ``shape``
    contract.
    """
    snapshots: List[Dict[str, Any]] = []
    for kind, container in (
        ("output", model.graph.output),
        ("value_info", model.graph.value_info),
    ):
        for vi in container:
            if not vi.HasField("type"):
                continue
            if not vi.type.HasField("tensor_type"):
                continue
            has_shape, dims = _dims_of_tensor_type(vi.type.tensor_type)
            snapshots.append(
                {
                    "name": vi.name,
                    "kind": kind,
                    "elem_type": int(vi.type.tensor_type.elem_type),
                    "has_shape": has_shape,
                    "shape": dims,
                }
            )
    return snapshots


def strip_shapes(model):
    """Return a deep copy of ``model`` with output / value_info shapes stripped.

    Only the ``elem_type`` is kept on plain ``tensor_type`` entries; the
    ``shape`` field is cleared so that shape inference must rebuild it.
    Non plain-tensor entries (sequence/optional/map) are left untouched.
    """
    import onnx

    stripped = onnx.ModelProto()
    stripped.CopyFrom(model)
    for container in (stripped.graph.output, stripped.graph.value_info):
        for vi in container:
            if not vi.HasField("type"):
                continue
            if not vi.type.HasField("tensor_type"):
                continue
            tt = vi.type.tensor_type
            elem_type = tt.elem_type
            vi.type.ClearField("tensor_type")
            vi.type.tensor_type.elem_type = elem_type
    return stripped


def _index_value_infos(model) -> Dict[str, Any]:
    """Return ``{name: ValueInfoProto}`` for outputs + value_info."""
    indexed: Dict[str, Any] = {}
    for vi in model.graph.value_info:
        indexed[vi.name] = vi
    for vi in model.graph.output:
        indexed[vi.name] = vi
    return indexed


def _compare_snapshot_with_model(snapshot, inferred_model) -> List[Dict[str, Any]]:
    """Score each snapshotted intermediate against ``inferred_model``.

    Returns a list of dicts ``{"name", "kind", "ok", "reason",
    "elem_type", "has_shape", "shape"}`` with the inferred information
    (or ``None`` for the inferred fields when the entry could not be
    located).
    """
    by_name = _index_value_infos(inferred_model)
    details: List[Dict[str, Any]] = []
    for entry in snapshot:
        name = entry["name"]
        expected_elem = entry["elem_type"]
        had_shape = entry["has_shape"]
        expected_shape = entry["shape"]
        detail: Dict[str, Any] = {
            "name": name,
            "kind": entry["kind"],
            "expected_elem_type": expected_elem,
            "expected_has_shape": had_shape,
            "expected_shape": list(expected_shape),
            "ok": False,
            "reason": "",
            "elem_type": None,
            "has_shape": False,
            "shape": [],
        }
        vi = by_name.get(name)
        if vi is None:
            detail["reason"] = "missing from graph after inference"
            details.append(detail)
            continue
        if not vi.HasField("type") or not vi.type.HasField("tensor_type"):
            detail["reason"] = "lost tensor_type"
            details.append(detail)
            continue
        tt = vi.type.tensor_type
        inferred_has_shape, inferred_dims = _dims_of_tensor_type(tt)
        detail["elem_type"] = int(tt.elem_type)
        detail["has_shape"] = inferred_has_shape
        detail["shape"] = list(inferred_dims)
        if int(tt.elem_type) != expected_elem:
            detail["reason"] = (
                f"elem_type mismatch: expected {expected_elem}, "
                f"got {int(tt.elem_type)}"
            )
            details.append(detail)
            continue
        if not inferred_has_shape:
            if had_shape:
                detail["reason"] = (
                    f"no shape inferred (expected rank {len(expected_shape)})"
                )
            else:
                detail["ok"] = True
            details.append(detail)
            continue
        if had_shape:
            if len(inferred_dims) != len(expected_shape):
                detail["reason"] = (
                    f"rank mismatch: expected {len(expected_shape)}, "
                    f"got {len(inferred_dims)}"
                )
                details.append(detail)
                continue
            mismatch = None
            for i, (got, exp) in enumerate(zip(inferred_dims, expected_shape)):
                got_concrete = isinstance(got, int) and got >= 0
                exp_concrete = isinstance(exp, int) and exp >= 0
                got_symbolic = isinstance(got, str)
                exp_symbolic = isinstance(exp, str)
                # Unknown expected dim (``-1``) carries no information,
                # so accept whatever was inferred. For every other case a
                # perfect match requires the same concrete value or the
                # same symbolic dim name.
                if exp_concrete and got_concrete:
                    if got != exp:
                        mismatch = (
                            f"dim[{i}] mismatch: expected {exp}, got {got}"
                        )
                        break
                elif exp_symbolic and got_symbolic:
                    if got != exp:
                        mismatch = (
                            f"dim[{i}] mismatch: expected {exp!r}, "
                            f"got {got!r}"
                        )
                        break
                elif exp_concrete and not got_concrete:
                    mismatch = (
                        f"dim[{i}] mismatch: expected {exp}, got {got!r}"
                    )
                    break
                elif exp_symbolic and not got_symbolic:
                    mismatch = (
                        f"dim[{i}] mismatch: expected {exp!r}, got {got}"
                    )
                    break
            if mismatch is not None:
                detail["reason"] = mismatch
                details.append(detail)
                continue
        # No expected shape but one was inferred — that is fine.
        detail["ok"] = True
        details.append(detail)
    return details


def _run_onnx_light(model):
    """Run ``onnx_light.onnx.shape_inference.infer_shapes`` on ``model``."""
    import onnx
    import onnx_light.onnx as onnxl
    import onnx_light.onnx.shape_inference as shape_inference

    light = onnxl.ModelProto()
    light.ParseFromString(model.SerializeToString())
    shape_inference.infer_shapes(light)
    out = onnx.ModelProto()
    out.ParseFromString(light.SerializeToString())
    return out


def _run_onnx_light_onnx_optim(model):
    """Run ``onnx_light.onnx_optim.shape_inference.infer_shapes_model``.

    The experimental shape inference shipped inside ``onnx-light``'s
    ``onnx_optim`` submodule mutates the model in place; we round-trip
    the result back to an ``onnx.ModelProto`` so the comparison helpers
    can score it uniformly.
    """
    import onnx
    import onnx_light.onnx as onnxl
    from onnx_light.onnx_optim.shape_inference import infer_shapes_model

    light = onnxl.ModelProto()
    light.ParseFromString(model.SerializeToString())
    infer_shapes_model(light)
    out = onnx.ModelProto()
    out.ParseFromString(light.SerializeToString())
    return out


def _run_onnx(model):
    """Run ``onnx.shape_inference.infer_shapes`` on ``model``."""
    import onnx.shape_inference

    return onnx.shape_inference.infer_shapes(model)


def _run_onnx_shape_inference(model):
    """Run ``onnx_shape_inference.infer_symbolic_shapes`` on ``model``.

    The package operates on an ``onnx_ir.Model``; we serialise the
    inferred IR model back to an ``onnx.ModelProto`` so the comparison
    helpers can score it uniformly.
    """
    import onnx_ir as ir
    from onnx_shape_inference import infer_symbolic_shapes

    ir_model = ir.serde.deserialize_model(model)
    inferred = infer_symbolic_shapes(ir_model)
    return ir.serde.serialize_model(inferred)


_BACKEND_RUNNERS: Dict[str, Callable[[Any], Any]] = {
    "onnx-light": _run_onnx_light,
    "onnx-light-onnx-optim": _run_onnx_light_onnx_optim,
    "onnx": _run_onnx,
    "onnx-shape-inference": _run_onnx_shape_inference,
}


def run_test_with_backend(
    model: Any,
    expected: List[Dict[str, Any]],
    backend: str,
) -> Dict[str, Any]:
    """Run ``backend`` on a stripped copy of ``model``, score the result.

    Returns a dictionary ``{"success", "error", "error_step", "correct",
    "total", "details"}``. ``success`` is True when shape inference ran
    AND every snapshotted intermediate was recovered correctly.
    """
    runner = _BACKEND_RUNNERS.get(backend)
    if runner is None:
        return {
            "success": False,
            "error": f"unknown backend: {backend}",
            "error_step": "load",
            "correct": 0,
            "total": len(expected),
            "details": [],
        }
    if not expected:
        return {
            "success": False,
            "error": "no intermediate to validate",
            "error_step": "load",
            "correct": 0,
            "total": 0,
            "details": [],
        }
    try:
        stripped = strip_shapes(model)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "strip",
            "correct": 0,
            "total": len(expected),
            "details": [],
        }
    try:
        inferred = runner(stripped)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "run",
            "correct": 0,
            "total": len(expected),
            "details": [],
        }
    try:
        details = _compare_snapshot_with_model(expected, inferred)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "compare",
            "correct": 0,
            "total": len(expected),
            "details": [],
        }
    correct = sum(1 for d in details if d["ok"])
    total = len(details)
    return {
        "success": total > 0 and correct == total,
        "error": "" if correct == total else f"{total - correct}/{total} intermediates mismatched",
        "error_step": "" if correct == total else "compare",
        "correct": correct,
        "total": total,
        "details": details,
    }


def _row_from_results(
    name: str,
    expected: List[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]] = None,
    versions: Optional[Dict[str, str]] = None,
    now_iso: Optional[str] = None,
    inputs: Optional[List[Dict[str, Any]]] = None,
    mermaid: str = "",
) -> Dict[str, Any]:
    """Build a dashboard row carrying over per-backend ``last_pass`` info."""
    versions = versions or {}
    previous = previous or {}
    row: Dict[str, Any] = {
        "name": name,
        "inputs": [
            {
                "name": i.get("name"),
                "kind": i.get("kind", "input"),
                "elem_type": i.get("elem_type"),
                "has_shape": i.get("has_shape", False),
                "shape": list(i.get("shape", [])),
            }
            for i in (inputs or [])
        ],
        "expected": [
            {
                "name": e.get("name"),
                "kind": e.get("kind"),
                "elem_type": e.get("elem_type"),
                "has_shape": e.get("has_shape", False),
                "shape": list(e.get("shape", [])),
            }
            for e in expected
        ],
        "runtimes": {},
    }
    if mermaid:
        row["mermaid"] = mermaid
    elif isinstance(previous, dict):
        # Preserve any previously rendered mermaid graph when the current
        # rendering returned an empty string (for instance when the model
        # could not be parsed), so the dashboard keeps showing the graph
        # for tests that already have one.
        prev_mermaid = previous.get("mermaid")
        if isinstance(prev_mermaid, str) and prev_mermaid:
            row["mermaid"] = prev_mermaid
    for backend in BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        runtime_entry: Dict[str, Any] = {
            "success": success,
            "correct": int(info.get("correct", 0)),
            "total": int(info.get("total", len(expected))),
            "details": info.get("details", []) or [],
        }
        error = _stringify_error(info.get("error"))
        if error:
            runtime_entry["error"] = error
        step = info.get("error_step") or ""
        if step:
            runtime_entry["error_step"] = step
        if success and now_iso is not None:
            runtime_entry["last_pass_date"] = now_iso
            pkg = BACKEND_PACKAGE.get(backend)
            version = versions.get(pkg) if pkg else None
            if version:
                runtime_entry["last_pass_version"] = version
        else:
            prev_runtimes = previous.get("runtimes") if isinstance(previous, dict) else None
            prev_entry = (
                prev_runtimes.get(backend)
                if isinstance(prev_runtimes, dict)
                else None
            )
            if isinstance(prev_entry, dict):
                prev_date = prev_entry.get("last_pass_date")
                if prev_date:
                    runtime_entry["last_pass_date"] = prev_date
                prev_version = prev_entry.get("last_pass_version")
                if prev_version:
                    runtime_entry["last_pass_version"] = prev_version
        row["runtimes"][backend] = runtime_entry
    return row


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
    tag: str = DEFAULT_TAG,
    limit: Optional[int] = None,
    discover: Callable[[str], List[Dict[str, Any]]] = discover_inference_tests,
    run: Callable[..., Dict[str, Any]] = run_test_with_backend,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover tests, run each backend on each test and return a payload."""
    if versions is None:
        versions = collect_versions
    tests = discover(tag)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    _log(f"Discovered {len(tests)} backend tests tagged {tag!r}.")

    now_dt = now or dt.datetime.now(tz=dt.timezone.utc)
    now_iso = _format_iso(now_dt)
    version_map = versions()
    previous_rows = _index_previous_rows(previous or {})

    rows: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {
        backend: {"correct": 0, "total": 0, "tests_pass": 0, "tests_fail": 0}
        for backend in BACKENDS
    }
    for idx, test in enumerate(tests):
        name = test["name"]
        model = test["model"]
        expected = test["expected"]
        inputs = test.get("inputs", [])
        results: Dict[str, Dict[str, Any]] = {}
        for backend in BACKENDS:
            try:
                info = run(model, expected, backend)
            except Exception as exc:  # noqa: BLE001
                _log(
                    f"Unhandled error for {name} on {backend}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                info = {
                    "success": False,
                    "error": _stringify_error(exc),
                    "error_step": "run",
                    "correct": 0,
                    "total": len(expected),
                    "details": [],
                }
            results[backend] = info
            totals[backend]["correct"] += int(info.get("correct", 0))
            totals[backend]["total"] += int(info.get("total", 0))
            if info.get("success"):
                totals[backend]["tests_pass"] += 1
            else:
                totals[backend]["tests_fail"] += 1
        rows.append(
            _row_from_results(
                name,
                expected,
                results,
                previous=previous_rows.get(name),
                versions=version_map,
                now_iso=now_iso,
                inputs=inputs,
                mermaid=test.get("mermaid", ""),
            )
        )
        if (idx + 1) % 25 == 0:
            _log(f"Ran {idx + 1}/{len(tests)} tests.")

    return {
        "date": now_iso,
        "tag": tag,
        "versions": version_map,
        "totals": totals,
        "tests": rows,
    }


def write_payload(json_path: str, payload: Dict[str, Any]) -> None:
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
        "--tag",
        default=DEFAULT_TAG,
        help=(
            "Filter backend cases by their ``tag`` attribute "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of tests executed (useful for debugging).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    json_path = os.path.join(
        args.cache_dir, "onnx-light", "shape_inference_coverage.json"
    )
    previous = load_previous_payload(json_path)
    try:
        payload = build_payload(
            tag=args.tag,
            limit=args.limit,
            previous=previous,
        )
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record shape inference coverage: {exc}")
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
