"""Record the shape-inference coverage of the shape-inference
implementations exercised by ``onnx-light`` (``onnx-light`` itself,
``onnx_light.onnx_optim`` — the experimental shape inference shipped
inside ``onnx-light``'s ``onnx_optim`` submodule, the official
``onnx.shape_inference``, the standalone ``onnx-shape-inference``
package and the symbolic shape inference shipped with
``onnxruntime.transformers``).

The script walks every backend test bundled with the installed
``onnx-light`` package (collected via
``onnx_light.onnx_lib.backend.test.case.collect_test_case``) and keeps only the
cases whose ``tag`` matches one of the requested tags (by default the
``"shape"``, ``"local_function"`` and legacy ``"inference"`` families
of tests dedicated to shape inference, mirroring
``unittests/backend/test_backend_with_shape_inference.py`` in the
``xadupre/onnx-light`` repository).

For each retained test case the recorded ``graph.output`` (and
intermediate ``graph.value_info``) shapes are snapshotted then stripped
from a working copy of the model. Only the main graph's shapes are
snapshotted and scored; shapes carried by subgraphs nested inside
control-flow nodes (``If``/``Loop``/``Scan``) are still stripped from
the working copy but are deliberately **not** compared, since their
intermediates depend on outer-scope inputs the shape-inference passes
cannot always propagate. Each candidate shape-inference
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
import itertools
import json
import os
import re
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend_test_metadata import tag_name

# Order matters: it drives the column order in the dashboard.
BACKENDS: Tuple[str, ...] = (
    "onnx-light",
    "onnx-light-optim",
    "onnx",
    "onnx-shape-inference",
    "ort-transformers",
    "yobx",
)

# Package whose version is recorded alongside the ``last_pass`` date for
# each backend.
BACKEND_PACKAGE: Dict[str, str] = {
    "onnx-light": "onnx_light",
    "onnx-light-optim": "onnx_light",
    "onnx": "onnx",
    "onnx-shape-inference": "onnx_shape_inference",
    "ort-transformers": "onnxruntime",
    "yobx": "yobx",
}

# Default tags used by ``onnx-light`` to mark backend cases that are
# specifically designed to exercise shape inference. A test case is
# selected when its ``tag`` attribute matches any of these values.
DEFAULT_TAGS: Tuple[str, ...] = ("shape", "local_function", "inference")
# Backwards-compatible alias kept for callers that import a single tag.
DEFAULT_TAG = ",".join(DEFAULT_TAGS)


def _normalize_tags(tag) -> Tuple[str, ...]:
    """Normalize a tag filter into a tuple of non-empty tag names.

    Accepts ``None``, a single tag (``str``) – optionally a
    comma-separated list such as ``"inference,local_function"`` – or an
    iterable of strings. Empty entries and surrounding whitespace are
    stripped. An empty result means "do not filter by tag".
    """
    if tag is None:
        return ()
    if isinstance(tag, str):
        parts = tag.split(",")
    else:
        parts = []
        for item in tag:
            if item is None:
                continue
            parts.extend(str(item).split(","))
    return tuple(p.strip() for p in parts if p and p.strip())


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
    for name in (
        "onnx",
        "onnx_light",
        "onnx_shape_inference",
        "onnx_ir",
        "onnxruntime",
        "yobx",
        "numpy",
    ):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def backend_versions_from_map(version_map: Dict[str, str]) -> Dict[str, str]:
    """Map each tested backend to the version of the package it exercises.

    The dashboard lists one column per tested package; this returns the
    version string for every backend in :data:`BACKENDS` whose underlying
    package (see :data:`BACKEND_PACKAGE`) has a known version, so the page
    can display the version of each tested package.
    """
    version_map = version_map or {}
    backend_versions: Dict[str, str] = {}
    for backend in BACKENDS:
        pkg = BACKEND_PACKAGE.get(backend)
        version = version_map.get(pkg) if pkg else None
        if version:
            backend_versions[backend] = version
    return backend_versions


def _stringify_error(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text and isinstance(value, BaseException):
        # Some exceptions carry no message (for instance a bare ``assert``
        # in onnxruntime's symbolic shape inference). Fall back to the
        # exception type so the dashboard reports an explicit reason
        # instead of an empty ``error`` that reads as "not running".
        text = type(value).__name__
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
    return str(text).replace("\\", "\\\\").replace('"', "&quot;").replace("\n", " ")


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

    Subgraphs nested inside control-flow nodes (``If``/``Loop``/``Scan``)
    are rendered as Mermaid ``subgraph`` blocks so their inputs,
    operators and outputs are visible too.
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

    edge_types: Dict[str, str] = {}

    def _collect_types(graph: Any) -> None:
        for value_info in (
            list(graph.input) + list(graph.output) + list(graph.value_info)
        ):
            label = _mermaid_format_type(onnx, value_info.type)
            if label:
                edge_types.setdefault(value_info.name, label)
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("g"):
                    _collect_types(attr.g)
                for sub in attr.graphs:
                    _collect_types(sub)

    _collect_types(annotated.graph)

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
    edges: List[str] = []
    # Per-graph records used in the edge pass: ``(op_nodes, output_entries,
    # scope)`` where ``op_nodes`` is a list of ``(node_id, node)``,
    # ``output_entries`` a list of ``(out_name, out_id)`` and ``scope`` the
    # tensor-name to producing-node-id mapping visible inside that graph.
    # Each subgraph inherits a copy of its parent scope so locally produced
    # tensors shadow outer-scope names (matching ONNX subgraph semantics)
    # without leaking to sibling subgraphs.
    graph_records: List[
        Tuple[List[Tuple[str, Any]], List[Tuple[str, str]], Dict[str, str]]
    ] = []

    def _declare(graph: Any, indent: int, parent_scope: Dict[str, str]) -> None:
        pad = "    " * indent
        # ``local`` holds the producers declared in *this* graph; it is
        # layered over ``parent_scope`` so a locally produced tensor shadows
        # an outer-scope tensor of the same name (ONNX subgraph semantics).
        local: Dict[str, str] = {}
        initializer_names = {init.name for init in graph.initializer}

        for value_info in graph.input:
            if value_info.name in initializer_names:
                continue
            node_id = _make_id("in", value_info.name)
            type_label = edge_types.get(value_info.name, "")
            label = value_info.name + (f"<br>{type_label}" if type_label else "")
            lines.append(f'{pad}{node_id}(["{_mermaid_escape(label)}"])')
            local[value_info.name] = node_id

        for initializer in graph.initializer:
            node_id = _make_id("init", initializer.name)
            dtype = _mermaid_dtype_name(onnx, initializer.data_type)
            dims = ",".join(str(d) for d in initializer.dims)
            label = initializer.name + (f"<br>{dtype}[{dims}]" if dtype else "")
            lines.append(f'{pad}{node_id}[("{_mermaid_escape(label)}")]')
            local[initializer.name] = node_id

        op_nodes: List[Tuple[str, Any]] = []
        subgraphs: List[Tuple[str, str, Any]] = []
        for index, node in enumerate(graph.node):
            node_id = _make_id("op", node.name or f"{node.op_type}_{index}")
            op_nodes.append((node_id, node))
            label = node.op_type + (f"<br>{node.name}" if node.name else "")
            lines.append(f'{pad}{node_id}["{_mermaid_escape(label)}"]')
            for out_name in node.output:
                if out_name:
                    local.setdefault(out_name, node_id)
            for attr in node.attribute:
                attr_graphs: List[Any] = []
                if attr.HasField("g"):
                    attr_graphs.append(attr.g)
                attr_graphs.extend(attr.graphs)
                for sub in attr_graphs:
                    sg_id = _make_id("sg", f"{node.op_type}_{attr.name}")
                    sg_label = f"{node.op_type}.{attr.name}"
                    subgraphs.append((sg_id, sg_label, sub))

        scope = {**parent_scope, **local}

        output_entries: List[Tuple[str, str]] = []
        for value_info in graph.output:
            node_id = _make_id("out", value_info.name)
            output_entries.append((value_info.name, node_id))
            type_label = edge_types.get(value_info.name, "")
            label = value_info.name + (f"<br>{type_label}" if type_label else "")
            lines.append(f'{pad}{node_id}(["{_mermaid_escape(label)}"])')

        graph_records.append((op_nodes, output_entries, scope))

        for sg_id, sg_label, sub in subgraphs:
            lines.append(f'{pad}subgraph {sg_id}["{_mermaid_escape(sg_label)}"]')
            _declare(sub, indent + 1, scope)
            lines.append(f"{pad}end")

    _declare(annotated.graph, 1, {})

    for op_nodes, output_entries, scope in graph_records:
        for node_id, node in op_nodes:
            for in_name in node.input:
                if not in_name:
                    continue
                source_id = scope.get(in_name)
                if not source_id:
                    continue
                type_label = edge_types.get(in_name, "")
                edge_label = in_name + (f" : {type_label}" if type_label else "")
                edges.append(
                    f'    {source_id} -- "{_mermaid_escape(edge_label)}" --> {node_id}'
                )

        for out_name, out_id in output_entries:
            source_id = scope.get(out_name)
            if not source_id or source_id == out_id:
                continue
            type_label = edge_types.get(out_name, "")
            edge_label = out_name + (f" : {type_label}" if type_label else "")
            edges.append(
                f'    {source_id} -- "{_mermaid_escape(edge_label)}" --> {out_id}'
            )

    return "\n".join(lines + edges)


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


def discover_inference_tests(tag=DEFAULT_TAGS) -> List[Dict[str, Any]]:
    """Return the list of backend tests whose ``tag`` matches.

    ``tag`` can be a single tag name, a comma-separated list of tag
    names (e.g. ``"inference,local_function"``) or an iterable of tag
    names. A test case is retained when its ``tag`` attribute matches
    any of the requested tags. Passing an empty value disables tag
    filtering.

    Each entry is a dictionary ``{"name", "model", "expected", "mermaid"}``
    where ``model`` is an ``onnx.ModelProto``, ``expected`` is the list
    of snapshotted intermediates (see :func:`snapshot_intermediates`)
    and ``mermaid`` is a Mermaid ``flowchart TD`` rendering of ``model``
    (empty string when rendering fails).
    """
    from onnx_light.onnx_lib.backend.test.case import collect_test_case

    tags = _normalize_tags(tag)
    cases = collect_test_case(include_big=True)
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_tags = _normalize_tags(tag_name(getattr(tc, "tag", None)))
        if tags and not any(t in tags for t in case_tags):
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


def _iter_subgraphs(graph):
    """Yield ``graph`` followed by every nested subgraph (depth-first).

    Subgraphs are reached through node attributes holding a ``GraphProto``
    (e.g. the ``then_branch``/``else_branch`` of ``If`` or the ``body`` of
    ``Loop``/``Scan``) or a list of ``GraphProto`` (attributes of type
    ``GRAPHS``). The walk recurses so subgraphs nested inside subgraphs are
    visited as well, ensuring their ``value_info``/``output`` shapes are
    seen by the snapshot, strip and comparison helpers.
    """
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _iter_subgraphs(attr.g)
            for sub in attr.graphs:
                yield from _iter_subgraphs(sub)


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

    Returns a list of dicts ``{"name", "kind", "op_type", "elem_type",
    "has_shape", "shape"}`` where ``kind`` is one of ``"output"``,
    ``"value_info"`` or ``"intermediate"``. ``op_type`` is the type of
    the node that produces the tensor (empty string if the tensor is
    not produced by any node, e.g. a graph output directly aliasing an
    input or initializer). Non plain-tensor entries
    (sequence/optional/map) are skipped since they cannot be scored
    against a simple ``elem_type`` + ``shape`` contract.

    Node outputs that have no ``value_info`` in the original model (and
    are not graph outputs) are still recorded with ``kind ==
    "intermediate"``, ``elem_type == None`` and ``has_shape == False``
    so the detailed report can show what each backend inferred for
    them, even though there is no ground truth to compare against. Such
    entries carry no expectation and are not counted towards the
    correctness score (see :func:`_compare_snapshot_with_model`).

    Only the model's main graph is snapshotted. Shapes carried by
    subgraphs nested inside control-flow nodes (``If``/``Loop``/``Scan``)
    are intentionally **not** snapshotted, so they are never scored
    against the runtimes: subgraph intermediates depend on outer-scope
    inputs whose shapes the shape-inference passes cannot always
    propagate, which would otherwise produce spurious mismatches.

    Entries are returned in the order they appear in the model: the
    graph nodes are walked in declaration order and each output they
    produce yields one entry. Graph outputs that are not produced by any
    node are appended at the end, preserving their relative order in
    ``model.graph.output``.
    """
    return _snapshot_graph_intermediates(model.graph)


def _snapshot_graph_intermediates(graph) -> List[Dict[str, Any]]:
    """Snapshot output / value_info shapes for a single ``GraphProto``.

    See :func:`snapshot_intermediates` for the entry format. This helper
    operates on a single graph; only the model's main graph is passed in
    (subgraph shapes are deliberately not snapshotted).
    """
    output_names = {vi.name for vi in graph.output}
    by_name: Dict[str, Tuple[str, Any]] = {}
    for vi in graph.value_info:
        kind = "output" if vi.name in output_names else "value_info"
        by_name[vi.name] = (kind, vi)
    for vi in graph.output:
        by_name[vi.name] = ("output", vi)

    # Walk the graph nodes in declaration order so that each
    # intermediate (and any graph output produced by a node) appears in
    # the order it is computed by the model. Graph outputs that are not
    # produced by any node (rare; e.g. directly aliasing an input or an
    # initializer) are appended at the end, preserving their relative
    # order in ``graph.output``.
    ordered_names: List[str] = []
    seen: set = set()
    op_type_by_name: Dict[str, str] = {}
    # Names of node outputs that have no ``value_info`` in the original
    # model (i.e. unannotated intermediates). These are still surfaced
    # in the snapshot so the detailed report can display the per-backend
    # inferred shape for them, but they carry no expectation.
    unannotated: set = set()
    for node in graph.node:
        for out_name in node.output:
            if not out_name or out_name in seen:
                continue
            op_type_by_name.setdefault(out_name, node.op_type)
            if out_name not in by_name:
                if out_name in unannotated:
                    continue
                unannotated.add(out_name)
            ordered_names.append(out_name)
            seen.add(out_name)
    for vi in graph.output:
        if vi.name in seen or vi.name not in by_name:
            continue
        ordered_names.append(vi.name)
        seen.add(vi.name)

    snapshots: List[Dict[str, Any]] = []
    for name in ordered_names:
        if name in by_name:
            kind, vi = by_name[name]
            if not vi.HasField("type"):
                continue
            if not vi.type.HasField("tensor_type"):
                continue
            has_shape, dims = _dims_of_tensor_type(vi.type.tensor_type)
            snapshots.append(
                {
                    "name": vi.name,
                    "kind": kind,
                    "op_type": op_type_by_name.get(vi.name, ""),
                    "elem_type": int(vi.type.tensor_type.elem_type),
                    "has_shape": has_shape,
                    "shape": dims,
                }
            )
        else:
            # Unannotated node output: report it for visibility but
            # without any expectation. ``_compare_snapshot_with_model``
            # treats entries with ``elem_type is None`` as informational
            # only and they are excluded from the correctness score.
            snapshots.append(
                {
                    "name": name,
                    "kind": "intermediate",
                    "op_type": op_type_by_name.get(name, ""),
                    "elem_type": None,
                    "has_shape": False,
                    "shape": [],
                }
            )
    return snapshots


def strip_shapes(model, keep_outputs: bool = False):
    """Return a deep copy of ``model`` with output / value_info shapes stripped.

    Only the ``elem_type`` is kept on plain ``tensor_type`` entries; the
    ``shape`` field is cleared so that shape inference must rebuild it.
    Non plain-tensor entries (sequence/optional/map) are left untouched.

    When ``keep_outputs`` is ``True``, only ``graph.value_info`` shapes are
    cleared and ``graph.output`` shapes are preserved. This is used to feed
    backends (e.g. ``onnx-light``) that can take advantage of the known
    output shape as a prefill hint when running shape inference.

    Subgraphs nested inside control-flow nodes (``If``/``Loop``/``Scan``)
    are stripped as well so shape inference must rebuild their
    intermediate and output shapes. ``keep_outputs`` only applies to the
    model's top-level outputs; a subgraph's outputs are internal results
    and are always cleared.
    """
    import onnx

    stripped = onnx.ModelProto()
    stripped.CopyFrom(model)
    for index, graph in enumerate(_iter_subgraphs(stripped.graph)):
        if keep_outputs and index == 0:
            containers = (graph.value_info,)
        else:
            containers = (graph.output, graph.value_info)
        for container in containers:
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
    """Return ``{name: ValueInfoProto}`` for outputs + value_info.

    Only the main graph is indexed. Subgraph outputs / value_info are
    intentionally excluded so the comparison never scores a snapshotted
    intermediate against a shape that lives in a control-flow subgraph
    (e.g. when a subgraph reuses a main-graph tensor name).
    """
    indexed: Dict[str, Any] = {}
    graph = model.graph
    for vi in graph.value_info:
        indexed[vi.name] = vi
    for vi in graph.output:
        indexed[vi.name] = vi
    return indexed


_SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _symbolic_dims_equal(got: str, exp: str) -> bool:
    """Return ``True`` when two symbolic dim expressions are equivalent.

    Shape-inference implementations format symbolic dimensions in
    different ways. The cheap comparison strips whitespace so that
    ``"a + b"`` and ``"a+b"`` are treated as equal. When that fails the
    expressions are parsed with ``sympy`` (when available) and compared
    symbolically, so mathematically equivalent forms such as
    ``"2*floor(0.5*H)"`` and ``"2*(H//2)"`` are recognised as equal.

    Floor-division/ceil identities such as ``"-floor(-b/2 - c/2)"`` and
    ``"(1+b+c)//2"`` are integer-valued but ``sympy.simplify`` cannot
    always prove them.  When the symbolic difference does not collapse to
    zero, it is evaluated over a small grid of positive integer dimension
    values; if the difference vanishes everywhere, the expressions are
    treated as equal.
    """

    if "".join(got.split()) == "".join(exp.split()):
        return True

    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
    except ImportError:
        return False

    def _parse(expr: str):
        # Map every identifier to a plain symbol so names such as ``N``
        # are not interpreted as ``sympy`` helpers (e.g. ``sympy.N``).
        names = set(_SYMBOL_RE.findall(expr))
        names.discard("floor")
        local = {name: sympy.Symbol(name) for name in names}
        parsed = parse_expr(expr, local_dict=local)
        # Convert floats (``0.5``) to rationals so ``floor(0.5*H)`` and
        # ``floor(H/2)`` collapse to the same expression.
        return sympy.nsimplify(parsed, rational=True)

    try:
        diff = sympy.simplify(_parse(got) - _parse(exp))
        if diff == 0:
            return True

        # ``sympy.simplify`` leaves floor/ceil identities unproven, so
        # verify the difference numerically over a grid of positive
        # integer dimension values. Bail out when there are too many
        # symbols to keep the combinatorial check cheap.
        symbols = sorted(diff.free_symbols, key=str)
        if not symbols or len(symbols) > 4:
            return False
        for combo in itertools.product(range(1, 11), repeat=len(symbols)):
            if diff.subs(dict(zip(symbols, combo))) != 0:
                return False
        return True
    except Exception:
        # Parsing/simplification can raise a wide range of errors for
        # expressions sympy cannot handle; fall back to "not equal"
        # instead of crashing the coverage run.
        return False


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
        # ``expected_elem is None`` flags a purely informational entry:
        # a node output that has no ``value_info`` in the original model
        # and therefore no ground truth to compare against. We still
        # populate the inferred fields below so the dashboard can show
        # what each backend produced, but we never mark it as a
        # mismatch and ``run_test_with_backend`` excludes it from the
        # correctness score.
        informational = expected_elem is None
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
            if informational:
                detail["ok"] = True
            else:
                detail["reason"] = "missing from graph after inference"
            details.append(detail)
            continue
        if not vi.HasField("type") or not vi.type.HasField("tensor_type"):
            if informational:
                detail["ok"] = True
            else:
                detail["reason"] = "lost tensor_type"
            details.append(detail)
            continue
        tt = vi.type.tensor_type
        inferred_has_shape, inferred_dims = _dims_of_tensor_type(tt)
        detail["elem_type"] = int(tt.elem_type)
        detail["has_shape"] = inferred_has_shape
        detail["shape"] = list(inferred_dims)
        if informational:
            # Purely informational entry: any inferred value is fine.
            detail["ok"] = True
            details.append(detail)
            continue
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
                        mismatch = f"dim[{i}] mismatch: expected {exp}, got {got}"
                        break
                elif exp_symbolic and got_symbolic:
                    # Symbolic dims may carry expressions such as
                    # ``"a + b"`` whose exact spacing varies between
                    # shape-inference implementations. Strip whitespace
                    # before comparing so that ``"a + b"`` and ``"a+b"``
                    # are treated as equal, and fall back to a symbolic
                    # comparison so mathematically equivalent forms such
                    # as ``"2*floor(0.5*H)"`` and ``"2*(H//2)"`` match.
                    if not _symbolic_dims_equal(got, exp):
                        mismatch = (
                            f"dim[{i}] mismatch: expected {exp!r}, " f"got {got!r}"
                        )
                        break
                elif exp_concrete and not got_concrete:
                    mismatch = f"dim[{i}] mismatch: expected {exp}, got {got!r}"
                    break
                elif exp_symbolic and not got_symbolic:
                    mismatch = f"dim[{i}] mismatch: expected {exp!r}, got {got}"
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


def _drop_shapeless_value_info(model):
    """Remove ``graph.value_info`` entries that carry a type but no shape.

    ``strip_shapes(..., keep_outputs=True)`` deliberately leaves the
    intermediate ``value_info`` entries with their ``elem_type`` set but
    no ``shape`` field. The experimental ``onnx_optim`` inference, when
    asked to prefill from ``value_info``/``output`` shapes, reads those
    entries unconditionally and a stripped ``tensor_type`` raises
    ``Optional field 'shape' has no value.``. Dropping the shapeless
    entries lets the inference rebuild them from scratch while still
    anchoring on the preserved ``graph.output`` shapes. Subgraphs are
    cleaned as well so their stripped intermediate ``value_info`` entries
    do not trip the same prefill path.
    """
    for graph in _iter_subgraphs(model.graph):
        keep = [
            vi
            for vi in graph.value_info
            if not (
                vi.type.HasField("tensor_type")
                and not vi.type.tensor_type.HasField("shape")
            )
        ]
        del graph.value_info[:]
        graph.value_info.extend(keep)
    return model


def _run_onnx_light_optim(model):
    """Run the experimental ``onnx-light`` shape inference.

    The Python entry point lives in
    ``onnx_light.onnx_core.shape_inference`` after the refactoring.

    ``prefill_with_value_info_output=True`` lets the inference anchor on
    the model's declared ``graph.output`` shapes (preserved by
    ``strip_shapes(..., keep_outputs=True)``). Without it, data-dependent
    outputs such as ``NonZero`` get freshly generated symbolic dim names
    that would never match the expected ones (e.g.
    ``test_cc_shape_inference_nonzero_chain_named``).

    Because the prefill also visits ``graph.value_info``, the shapeless
    intermediate entries left behind by ``strip_shapes`` must be removed
    first; otherwise the inference raises ``Optional field 'shape' has no
    value.`` when it reads their stripped ``tensor_type``.
    """
    import onnx
    import onnx_light.onnx as onnxl
    from onnx_light.onnx_core.shape_inference import infer_shapes_model

    prepared = onnx.ModelProto()
    prepared.CopyFrom(model)
    _drop_shapeless_value_info(prepared)
    light = onnxl.ModelProto()
    light.ParseFromString(prepared.SerializeToString())
    infer_shapes_model(light, prefill_with_value_info_output=True)
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


def _run_ort_transformers(model):
    """Run the symbolic shape inference shipped with ``onnxruntime.transformers``.

    The implementation lives in ``onnxruntime/tools/symbolic_shape_infer.py``
    and is re-exported via ``onnxruntime.transformers.shape_infer_helper``;
    importing the helper takes care of inserting the ``tools`` directory
    on ``sys.path`` so ``symbolic_shape_infer`` is importable. The
    ``SymbolicShapeInference.infer_shapes`` static method takes an
    ``onnx.ModelProto`` and returns one with shapes filled in.
    """
    # Importing this module has the side-effect of inserting
    # ``onnxruntime/tools`` (or ``onnxruntime/transformers/..``) on
    # ``sys.path`` so that ``symbolic_shape_infer`` becomes importable
    # from a regular ``onnxruntime`` wheel — the helper class itself is
    # not used here.
    import onnxruntime.transformers.shape_infer_helper  # noqa: F401
    from symbolic_shape_infer import SymbolicShapeInference

    return SymbolicShapeInference.infer_shapes(model, auto_merge=True)


def _run_yobx(model):
    """Run ``yobx.xshape.BasicShapeBuilder`` on ``model``.

    :class:`yobx.xshape.BasicShapeBuilder` is the shape-inference engine
    shipped with the ``yet-another-onnx-builder`` project. It walks the
    graph and tracks shapes (potentially symbolic) for every
    intermediate. ``update_shapes`` mutates the input ``ModelProto`` to
    populate ``graph.value_info`` for the intermediates it managed to
    infer; existing ``value_info`` entries (left in place by
    ``strip_shapes`` with only ``elem_type`` set) and ``graph.output``
    entries whose shape was stripped are also refilled here so the
    shared comparison helpers can score them uniformly.
    """
    import onnx
    from yobx.xshape import BasicShapeBuilder

    builder = BasicShapeBuilder()
    builder.run_model(model)
    out = onnx.ModelProto()
    out.CopyFrom(model)
    builder.update_shapes(out)
    # ``update_shapes`` skips both ``graph.output`` (so the caller's
    # declared output shapes are preserved) and any name already
    # appearing in ``graph.value_info`` (so the caller's annotations are
    # preserved). Here ``strip_shapes`` left those entries shape-less,
    # so refill them from the builder when possible.
    for container in (out.graph.value_info, out.graph.output):
        for vi in container:
            if not vi.HasField("type") or not vi.type.HasField("tensor_type"):
                continue
            tt = vi.type.tensor_type
            if tt.HasField("shape") and len(tt.shape.dim) > 0:
                continue
            if not builder.has_shape(vi.name):
                continue
            tt.ClearField("shape")
            for d in builder.get_shape(vi.name):
                new_d = tt.shape.dim.add()
                if isinstance(d, int):
                    # ``BasicShapeBuilder`` uses negative values (typically
                    # ``-1``) as a placeholder for "unknown rank position";
                    # leave the dim empty so the comparison helper treats it
                    # as unknown rather than as a concrete dimension.
                    if d >= 0:
                        new_d.dim_value = d
                else:
                    new_d.dim_param = str(d)
            if not tt.elem_type and builder.has_type(vi.name):
                tt.elem_type = builder.get_type(vi.name)
    return out


_BACKEND_RUNNERS: Dict[str, Callable[[Any], Any]] = {
    "onnx-light": _run_onnx_light,
    "onnx-light-optim": _run_onnx_light_optim,
    "onnx": _run_onnx,
    "onnx-shape-inference": _run_onnx_shape_inference,
    "ort-transformers": _run_ort_transformers,
    "yobx": _run_yobx,
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
    # ``len(expected)`` is used as a sensible upper bound for ``total``
    # on early failure paths. Informational entries (no expectation) are
    # excluded so the count reflects scoring as it would have been.
    scored_total = sum(1 for e in expected if e.get("elem_type") is not None)
    if runner is None:
        return {
            "success": False,
            "error": f"unknown backend: {backend}",
            "error_step": "load",
            "correct": 0,
            "total": scored_total,
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
        # ``onnx-light``'s shape inference can take advantage of the
        # known graph output shapes as a prefill hint: only clean
        # ``graph.value_info`` and keep ``graph.output`` shapes so they
        # are passed through to the backend as initial constraints. This
        # applies to both the ``onnx.shape_inference`` backend and the
        # experimental ``onnx_optim`` one (which opts into the anchors via
        # ``prefill_with_value_info_output=True``).
        keep_outputs = backend in ("onnx-light", "onnx-light-optim")
        stripped = strip_shapes(model, keep_outputs=keep_outputs)
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": _stringify_error(exc),
            "error_step": "strip",
            "correct": 0,
            "total": scored_total,
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
            "total": scored_total,
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
            "total": scored_total,
            "details": [],
        }
    # Entries without an expectation (``expected_elem_type is None``) are
    # purely informational: they show up in ``details`` so the dashboard
    # can display the inferred shape for unannotated intermediates, but
    # they are not counted towards the correctness score.
    scored = [d for d in details if d.get("expected_elem_type") is not None]
    correct = sum(1 for d in scored if d["ok"])
    total = len(scored)
    return {
        "success": total > 0 and correct == total,
        "error": (
            ""
            if correct == total
            else f"{total - correct}/{total} intermediates mismatched"
        ),
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
                "op_type": e.get("op_type", ""),
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
    scored_count = sum(1 for e in expected if e.get("elem_type") is not None)
    for backend in BACKENDS:
        info = results.get(backend, {})
        success = bool(info.get("success"))
        runtime_entry: Dict[str, Any] = {
            "success": success,
            "correct": int(info.get("correct", 0)),
            "total": int(info.get("total", scored_count)),
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
            prev_runtimes = (
                previous.get("runtimes") if isinstance(previous, dict) else None
            )
            prev_entry = (
                prev_runtimes.get(backend) if isinstance(prev_runtimes, dict) else None
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
    tag=DEFAULT_TAGS,
    limit: Optional[int] = None,
    discover: Callable[..., List[Dict[str, Any]]] = discover_inference_tests,
    run: Callable[..., Dict[str, Any]] = run_test_with_backend,
    versions: Optional[Callable[[], Dict[str, str]]] = None,
    now: Optional[dt.datetime] = None,
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover tests, run each backend on each test and return a payload.

    ``tag`` accepts a single tag name, a comma-separated list of tag
    names or an iterable of tag names. See :func:`discover_inference_tests`
    for details.
    """
    if versions is None:
        versions = collect_versions
    tests = discover(tag)
    if limit is not None and limit >= 0:
        tests = tests[:limit]
    tags = _normalize_tags(tag)
    tag_display = ", ".join(tags)
    _log(f"Discovered {len(tests)} backend tests tagged {tag_display!r}.")

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
        "tag": tag_display,
        "versions": version_map,
        "backend_versions": backend_versions_from_map(version_map),
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
            "Filter backend cases by their ``tag`` attribute. Accepts a "
            "single tag or a comma-separated list of tags; a case is "
            "retained when its tag matches any of the provided values "
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
