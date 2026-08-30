"""Record how well onnx-light's in-place metadata algorithm matches the
backend tests tagged ``inplace``.

The script walks the backend tests bundled with the installed ``onnx-light``
package (collected via ``onnx_light.onnx_lib.backend.test.case.collect_test_case``),
keeps only the cases whose ``tag`` matches ``inplace`` by default, reruns the
shape-inference-driven in-place analysis on each model, and compares the
resulting node metadata against the metadata embedded in the test case.

The resulting JSON summary is persisted to
``cache_data/onnx-light/inplace_reuse_coverage.json``. The dashboard at
``dashboard/onnx-light/inplace-reuse-coverage.html`` consumes that file to
show how many test cases, nodes and metadata annotations are recovered exactly.

Usage::

    python scripts/record_onnx_inplace_reuse_coverage.py [--cache-dir DIR]
        [--tag inplace] [--limit N]
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

from backend_test_metadata import tag_name

DEFAULT_TAGS: Tuple[str, ...] = ("inplace",)
DEFAULT_TAG: str = ",".join(DEFAULT_TAGS)
METADATA_KEYS: Tuple[str, ...] = ("onnx_light.inplace_reuse",)


def _normalize_tags(tag) -> Tuple[str, ...]:
    """Normalize a tag filter into a tuple of distinct non-empty tags."""
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
    seen: Dict[str, None] = {}
    for part in parts:
        value = part.strip()
        if value and value not in seen:
            seen[value] = None
    return tuple(seen)


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
    for name in ("onnx", "onnx_light", "numpy"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - optional dependencies at test time
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _clone_model(model):
    """Deep-copy a model proto through its serialized representation."""
    out = type(model)()
    out.ParseFromString(model.SerializeToString())
    return out


def _node_metadata(node) -> Dict[str, str]:
    """Return the metadata keys relevant to the inplace analysis."""
    return {
        str(entry.key): str(entry.value)
        for entry in getattr(node, "metadata_props", [])
        if str(entry.key) in METADATA_KEYS
    }


def _clear_node_metadata(node) -> None:
    """Remove every metadata entry from ``node`` in place."""
    del node.metadata_props[:]


def _node_io(node) -> Tuple[List[str], List[str]]:
    """Return ``(inputs, outputs)`` for one node as display-ready strings."""
    inputs = [str(name) for name in getattr(node, "input", []) if str(name)]
    outputs = [str(name) for name in getattr(node, "output", []) if str(name)]
    return inputs, outputs


# ---------------------------------------------------------------------------
# Mermaid graph rendering (best-effort; failures are silently ignored)
# ---------------------------------------------------------------------------


def _mermaid_escape(text: str) -> str:
    """Escape ``text`` so it can appear inside a Mermaid ``"..."`` label."""
    return str(text).replace("\\", "\\\\").replace('"', "&quot;").replace("\n", " ")


def _mermaid_dtype_name(onnx_mod: Any, dtype: int) -> str:
    if not dtype:
        return ""
    try:
        return onnx_mod.TensorProto.DataType.Name(dtype)
    except Exception:  # noqa: BLE001
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
    """Render ``model`` as a Mermaid ``flowchart TD`` string."""
    import onnx

    if not hasattr(model, "graph"):
        return ""

    annotated = model
    try:
        annotated = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, check_type=False
        )
    except Exception:  # noqa: BLE001
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
    graph_records: List[
        Tuple[List[Tuple[str, Any]], List[Tuple[str, str]], Dict[str, str]]
    ] = []

    def _declare(graph: Any, indent: int, parent_scope: Dict[str, str]) -> None:
        pad = "    " * indent
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

    Returns an empty string when ``onnx`` cannot be imported, when
    ``model`` is not a usable ``onnx.ModelProto`` or when rendering
    fails (best-effort, never a hard requirement).
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        return ""
    try:
        return _render_model_as_mermaid(model)
    except Exception:  # noqa: BLE001
        return ""


def _normalize_graph(graph: Any) -> Dict[str, str]:
    """Return ``{"svg": ...}`` when ``graph`` carries a non-empty SVG string."""
    if graph is None:
        return {}
    if (
        isinstance(graph, dict)
        and isinstance(graph.get("svg"), str)
        and graph.get("svg")
    ):
        return {"svg": graph["svg"]}
    return {}


def model_to_svg_graph(model: Any) -> Dict[str, str]:
    """Return ``{"svg": ...}`` for ``model`` or ``{}`` when unavailable."""
    try:
        from onnx_light.tools import to_svg
    except Exception:  # noqa: BLE001
        return {}
    try:
        svg = to_svg(model)
    except Exception:  # noqa: BLE001
        return {}
    return _normalize_graph({"svg": svg})


def discover_inplace_tests(tag=DEFAULT_TAGS) -> List[Dict[str, Any]]:
    """Return backend tests whose ``tag`` matches ``tag``.

    Each entry is a dictionary with keys ``"name"``, ``"model"``,
    ``"expected_nodes"``, ``"node_ops"``, ``"expected_inputs"``, and
    ``"graph_input_names"``.  ``expected_nodes`` and ``expected_inputs`` hold
    the per-node / per-graph-input metadata subsets relevant to the in-place
    analysis.  Tests with ``"_big_"`` in their name are always included so
    that large model test cases (e.g. qwen3) appear on the coverage page even
    when they do not carry a matching tag.
    """
    from onnx_light.onnx_lib.backend.test.case import collect_test_case

    tags = _normalize_tags(tag)
    cases = collect_test_case(include_big=True)
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_tags = _normalize_tags(tag_name(getattr(tc, "tag", None)))
        model = getattr(tc, "model", None)
        if model is None:
            continue
        nodes = list(getattr(model.graph, "node", []))
        expected_nodes = [_node_metadata(node) for node in nodes]
        graph_inputs = list(getattr(model.graph, "input", []))
        expected_inputs = [_node_metadata(vi) for vi in graph_inputs]
        graph_input_names = [str(getattr(vi, "name", "")) for vi in graph_inputs]
        has_metadata = any(expected_nodes) or any(expected_inputs)
        is_big = "_big_" in str(name)
        if tags and not any(t in tags for t in case_tags) and not has_metadata and not is_big:
            continue
        discovered.append(
            {
                "name": str(name),
                "model": model,
                "expected_nodes": expected_nodes,
                "expected_inputs": expected_inputs,
                "graph_input_names": graph_input_names,
                "node_ops": [str(getattr(node, "op_type", "")) for node in nodes],
                "node_inputs": [_node_io(node)[0] for node in nodes],
                "node_outputs": [_node_io(node)[1] for node in nodes],
                "mermaid": model_to_mermaid(model),
                "graph": model_to_svg_graph(model),
            }
        )
    discovered.sort(key=lambda item: item["name"])
    return discovered


def run_inplace_analysis(model) -> Dict[str, Any]:
    """Run onnx-light's inplace metadata analysis on ``model``."""
    from onnx_light.onnx_core import shape_inference as si

    work = _clone_model(model)
    for node in work.graph.node:
        _clear_node_metadata(node)
    for vi in work.graph.input:
        _clear_node_metadata(vi)

    ctx = si.ShapesContext()
    si.compute_shape_model(ctx, work)

    inplace = si.ComputeContext()
    inplace.compute_inplace_reuse_graph(work.graph, ctx)
    inplace.write_to_metadata(work.graph)

    return {
        "actual_nodes": [_node_metadata(node) for node in work.graph.node],
        "actual_inputs": [_node_metadata(vi) for vi in work.graph.input],
        "memory": list(getattr(inplace, "memory", [])),
    }


def _empty_totals() -> Dict[str, Dict[str, int]]:
    return {
        "tests": {"pass": 0, "fail": 0},
        "nodes": {"pass": 0, "fail": 0},
        "metadata": {"pass": 0, "fail": 0},
    }


def _score_test(
    name: str,
    expected_nodes: List[Dict[str, str]],
    actual_nodes: List[Dict[str, str]],
    node_ops: Optional[List[str]] = None,
    node_inputs: Optional[List[List[str]]] = None,
    node_outputs: Optional[List[List[str]]] = None,
    expected_inputs: Optional[List[Dict[str, str]]] = None,
    actual_inputs: Optional[List[Dict[str, str]]] = None,
    graph_input_names: Optional[List[str]] = None,
    error: str = "",
    memory: Optional[List[Any]] = None,
    mermaid: str = "",
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a single test and return a row dict for the JSON payload.

    The ``"mermaid"`` key is only present in the returned dict when a
    non-empty ``mermaid`` string is provided, so consumers should use
    ``row.get("mermaid", "")`` to retrieve it safely.

    When ``expected_inputs`` / ``actual_inputs`` are provided the
    per-graph-input metadata is scored in the same way as node metadata and
    the results are stored in a top-level ``"inputs"`` list on the row.
    """
    node_ops = list(node_ops or [])
    total_nodes = max(len(expected_nodes), len(actual_nodes), len(node_ops))
    nodes: List[Dict[str, Any]] = []
    matched_nodes = 0
    matched_metadata = 0
    total_metadata = 0
    success = not error
    for index in range(total_nodes):
        expected = dict(expected_nodes[index]) if index < len(expected_nodes) else {}
        actual = dict(actual_nodes[index]) if index < len(actual_nodes) else {}
        op_type = node_ops[index] if index < len(node_ops) else ""
        inputs = (
            list(node_inputs[index])
            if node_inputs is not None and index < len(node_inputs)
            else []
        )
        outputs = (
            list(node_outputs[index])
            if node_outputs is not None and index < len(node_outputs)
            else []
        )
        keys = sorted(set(expected) | set(actual))
        metadata_matches = sum(
            1 for key in keys if expected.get(key) == actual.get(key)
        )
        total_metadata += len(keys)
        matched_metadata += metadata_matches
        node_success = expected == actual
        if node_success:
            matched_nodes += 1
        else:
            success = False
        nodes.append(
            {
                "index": index,
                "op_type": op_type,
                "success": node_success,
                "expected": expected,
                "actual": actual,
                "memory": (
                    memory[index]
                    if memory is not None and index < len(memory)
                    else None
                ),
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    # Score graph-level input metadata (e.g. onnx_light.inplace_reuse on
    # ValueInfoProto entries in graph.input).
    scored_inputs: List[Dict[str, Any]] = []
    if expected_inputs is not None or actual_inputs is not None:
        exp_inp = list(expected_inputs or [])
        act_inp = list(actual_inputs or [])
        inp_names = list(graph_input_names or [])
        total_inp = max(len(exp_inp), len(act_inp))
        for idx in range(total_inp):
            exp = dict(exp_inp[idx]) if idx < len(exp_inp) else {}
            act = dict(act_inp[idx]) if idx < len(act_inp) else {}
            input_name = inp_names[idx] if idx < len(inp_names) else ""
            keys = sorted(set(exp) | set(act))
            metadata_matches = sum(
                1 for key in keys if exp.get(key) == act.get(key)
            )
            total_metadata += len(keys)
            matched_metadata += metadata_matches
            inp_success = exp == act
            if not inp_success:
                success = False
            scored_inputs.append(
                {
                    "index": idx,
                    "name": input_name,
                    "success": inp_success,
                    "expected": exp,
                    "actual": act,
                }
            )

    row: Dict[str, Any] = {
        "name": name,
        "success": success,
        "error": error,
        "matched_nodes": matched_nodes,
        "total_nodes": total_nodes,
        "matched_metadata": matched_metadata,
        "total_metadata": total_metadata,
        "nodes": nodes,
    }
    if scored_inputs:
        row["inputs"] = scored_inputs
    if mermaid:
        row["mermaid"] = mermaid
    normalized_graph = _normalize_graph(graph)
    if normalized_graph:
        row["graph"] = normalized_graph
    return row


def build_payload(
    tag=DEFAULT_TAG,
    *,
    discover: Callable[..., List[Dict[str, Any]]] = discover_inplace_tests,
    run: Callable[[Any], Dict[str, Any]] = run_inplace_analysis,
    versions: Callable[[], Dict[str, str]] = collect_versions,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and return the JSON payload consumed by the dashboard."""
    tests = discover(tag)
    if limit is not None:
        tests = tests[: max(limit, 0)]
    tags = _normalize_tags(tag)
    tag_display = ", ".join(tags)
    _log(f"Discovered {len(tests)} backend tests tagged {tag_display!r}.")

    totals = _empty_totals()
    rows: List[Dict[str, Any]] = []
    for test in tests:
        try:
            info = run(test["model"])
            row = _score_test(
                test["name"],
                list(test.get("expected_nodes", [])),
                list(info.get("actual_nodes", [])),
                node_ops=list(test.get("node_ops", [])),
                node_inputs=list(test.get("node_inputs", [])),
                node_outputs=list(test.get("node_outputs", [])),
                expected_inputs=(
                    list(test["expected_inputs"])
                    if test.get("expected_inputs") is not None
                    else None
                ),
                actual_inputs=(
                    list(info["actual_inputs"])
                    if info.get("actual_inputs") is not None
                    else None
                ),
                graph_input_names=list(test.get("graph_input_names", [])),
                memory=(
                    list(info.get("memory", []))
                    if info.get("memory") is not None
                    else None
                ),
                mermaid=test.get("mermaid", ""),
                graph=test.get("graph"),
            )
        except Exception as exc:  # noqa: BLE001 - keep recording other tests
            _log(f"Unhandled error for {test['name']}: {exc}")
            traceback.print_exc()
            row = _score_test(
                test["name"],
                list(test.get("expected_nodes", [])),
                [],
                node_ops=list(test.get("node_ops", [])),
                node_inputs=list(test.get("node_inputs", [])),
                node_outputs=list(test.get("node_outputs", [])),
                expected_inputs=(
                    list(test["expected_inputs"])
                    if test.get("expected_inputs") is not None
                    else None
                ),
                actual_inputs=None,
                graph_input_names=list(test.get("graph_input_names", [])),
                error=str(exc) or type(exc).__name__,
                mermaid=test.get("mermaid", ""),
                graph=test.get("graph"),
            )

        totals["tests"]["pass" if row["success"] else "fail"] += 1
        totals["nodes"]["pass"] += row["matched_nodes"]
        totals["nodes"]["fail"] += max(row["total_nodes"] - row["matched_nodes"], 0)
        totals["metadata"]["pass"] += row["matched_metadata"]
        totals["metadata"]["fail"] += max(
            row["total_metadata"] - row["matched_metadata"], 0
        )
        rows.append(row)

    now_iso = _format_iso(dt.datetime.now(tz=dt.timezone.utc))
    return {
        "date": now_iso,
        "tag": tag_display,
        "versions": versions(),
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
            "single tag or a comma-separated list of tags (default: %(default)s)."
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
        args.cache_dir, "onnx-light", "inplace_reuse_coverage.json"
    )
    try:
        payload = build_payload(tag=args.tag, limit=args.limit)
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record inplace reuse coverage: {exc}")
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
