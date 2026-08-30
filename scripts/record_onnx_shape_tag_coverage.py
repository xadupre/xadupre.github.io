"""Record how well onnx-light's shape-tag metadata algorithm matches the
backend tests tagged ``shape_tag``.

The script walks the backend tests bundled with the installed ``onnx-light``
package (collected via ``onnx_light.onnx_lib.backend.test.case.collect_test_case``),
keeps only the cases whose ``tag`` matches ``shape_tag`` by default, reruns the
shape-tag analysis on each model, and compares the resulting node metadata against
the metadata embedded in the test case.

The resulting JSON summary is persisted to
``cache_data/onnx-light/shape_tag_coverage.json``. The dashboard at
``dashboard/onnx-light/shape-tag-coverage.html`` consumes that file to
show how many test cases, nodes and metadata annotations are recovered exactly.

Usage::

    python scripts/record_onnx_shape_tag_coverage.py [--cache-dir DIR]
        [--tag shape_tag] [--limit N]
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

DEFAULT_TAGS: Tuple[str, ...] = ("shape_tag",)
DEFAULT_TAG: str = ",".join(DEFAULT_TAGS)
# Metadata keys written by onnx-light's ``write_value_and_node_tags_to_metadata``.
# ``onnx_light.node_tag`` is stored on each node, ``onnx_light.value_tag`` (singular)
# on each value proto (input/output/initializer/value_info) and
# ``onnx_light.value_tags`` (plural) is a graph-level JSON map ``{name: tag}``.
NODE_TAG_METADATA_KEY: str = "onnx_light.node_tag"
VALUE_TAG_METADATA_KEY: str = "onnx_light.value_tag"
VALUE_TAGS_METADATA_KEY: str = "onnx_light.value_tags"
METADATA_KEYS: Tuple[str, ...] = (NODE_TAG_METADATA_KEY,)
VALUE_METADATA_KEYS: Tuple[str, ...] = (VALUE_TAG_METADATA_KEY,)


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
    """Return the metadata keys relevant to the shape-tag analysis."""
    return {
        str(entry.key): str(entry.value)
        for entry in getattr(node, "metadata_props", [])
        if str(entry.key) in METADATA_KEYS
    }


def _clear_node_metadata(node) -> None:
    """Remove every metadata entry from ``node`` in place."""
    del node.metadata_props[:]


def _value_metadata(obj) -> Dict[str, str]:
    """Return VALUE_METADATA_KEYS entries from a value_info or tensor proto."""
    return {
        str(entry.key): str(entry.value)
        for entry in getattr(obj, "metadata_props", [])
        if str(entry.key) in VALUE_METADATA_KEYS
    }


def _graph_level_value_tags(graph) -> Dict[str, str]:
    """Return the graph-level ``value_tags`` JSON aggregate as a ``{name: tag}`` map.

    onnx-light stores the full value-to-tag mapping as a JSON object in the
    graph's ``onnx_light.value_tags`` metadata entry, in addition to the
    per-value ``onnx_light.value_tag`` annotations. Reading this aggregate
    ensures tags that are only recorded at the graph level (for example on
    graph inputs) are not lost. Returns an empty dict when the entry is absent
    or not decodable.
    """
    for entry in getattr(graph, "metadata_props", []):
        if str(entry.key) != VALUE_TAGS_METADATA_KEY:
            continue
        try:
            payload = json.loads(str(entry.value))
        except (json.JSONDecodeError, ValueError, TypeError):
            return {}
        if isinstance(payload, dict):
            return {str(name): str(tag) for name, tag in payload.items()}
        return {}
    return {}


def _graph_value_snapshot(model) -> List[Dict[str, Any]]:
    """Collect value-level metadata for a model's graph inputs, outputs, and initializers.

    Returns a list of ``{"name", "kind", "metadata"}`` dicts where ``kind``
    is ``"input"``, ``"output"``, ``"initializer"``, or ``"result"``.

    Some onnx-light pipelines store value tags for graph inputs/outputs in
    ``value_info`` only; those tags are merged into the matching entry so that
    the snapshot reflects the full set of annotations regardless of where
    onnx-light chose to write them.
    """
    if not hasattr(model, "graph"):
        return []
    graph = model.graph
    init_names = {init.name for init in graph.initializer}
    by_name: Dict[str, Dict[str, Any]] = {}
    result: List[Dict[str, Any]] = []
    for vi in graph.input:
        if vi.name not in init_names:
            row: Dict[str, Any] = {
                "name": vi.name,
                "kind": "input",
                "metadata": _value_metadata(vi),
            }
            result.append(row)
            by_name[vi.name] = row
    for vi in graph.output:
        row = {"name": vi.name, "kind": "output", "metadata": _value_metadata(vi)}
        result.append(row)
        by_name[vi.name] = row
    for init in graph.initializer:
        row = {
            "name": init.name,
            "kind": "initializer",
            "metadata": _value_metadata(init),
        }
        result.append(row)
        by_name[init.name] = row
    for vi in graph.value_info:
        existing = by_name.get(vi.name)
        if existing is not None:
            # Merge value_info metadata into the input/output/initializer entry so
            # that tags stored only in value_info are visible for those values.
            vi_meta = _value_metadata(vi)
            if vi_meta:
                merged = dict(existing.get("metadata", {}))
                for key, value in vi_meta.items():
                    merged.setdefault(key, value)
                existing["metadata"] = merged
        else:
            result.append(
                {"name": vi.name, "kind": "result", "metadata": _value_metadata(vi)}
            )
    # Merge the graph-level ``value_tags`` JSON aggregate so that tags recorded
    # only at the graph level (e.g. on graph inputs) are reflected per value.
    aggregate = _graph_level_value_tags(graph)
    if aggregate:
        known = {row["name"] for row in result}
        for row in result:
            tag = aggregate.get(row["name"])
            if tag and VALUE_TAG_METADATA_KEY not in row["metadata"]:
                merged = dict(row["metadata"])
                merged[VALUE_TAG_METADATA_KEY] = tag
                row["metadata"] = merged
        for name, tag in aggregate.items():
            if name and name not in known:
                result.append(
                    {
                        "name": name,
                        "kind": "result",
                        "metadata": {VALUE_TAG_METADATA_KEY: tag},
                    }
                )
                known.add(name)
    return result


def _has_expected_value_metadata(values: Optional[List[Dict[str, Any]]]) -> bool:
    """Tell if any expected graph value carries shape-tag metadata."""
    return any(value.get("metadata") for value in (values or []))


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


def discover_shape_tag_tests(tag=DEFAULT_TAGS) -> List[Dict[str, Any]]:
    """Return backend tests whose ``tag`` matches ``tag``.

    Each entry is a dictionary ``{"name", "model", "expected_nodes", "node_ops",
    "expected_values"}`` where ``expected_nodes`` is the per-node metadata subset
    relevant to the shape-tag analysis and ``expected_values`` is the per-value
    metadata (inputs, outputs, initializers).
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
        expected_values = _graph_value_snapshot(model)
        has_metadata = any(expected_nodes) or _has_expected_value_metadata(
            expected_values
        )
        if tags and not any(t in tags for t in case_tags) and not has_metadata:
            continue
        discovered.append(
            {
                "name": str(name),
                "model": model,
                "expected_nodes": expected_nodes,
                "node_ops": [str(getattr(node, "op_type", "")) for node in nodes],
                "node_inputs": [_node_io(node)[0] for node in nodes],
                "node_outputs": [_node_io(node)[1] for node in nodes],
                "expected_values": expected_values,
                "mermaid": model_to_mermaid(model),
                "graph": model_to_svg_graph(model),
            }
        )
    discovered.sort(key=lambda item: item["name"])
    return discovered


def run_shape_tag_analysis(model) -> Dict[str, Any]:
    """Run onnx-light's shape-tag metadata analysis on ``model``."""
    from onnx_light.onnx_core.shape_inference import (
        write_value_and_node_tags_to_metadata,
    )

    work = _clone_model(model)
    del work.graph.metadata_props[:]
    for node in work.graph.node:
        _clear_node_metadata(node)
    for vi in (
        list(work.graph.input) + list(work.graph.output) + list(work.graph.value_info)
    ):
        del vi.metadata_props[:]
    for init in work.graph.initializer:
        del init.metadata_props[:]

    write_value_and_node_tags_to_metadata(work.graph)

    return {
        "actual_nodes": [_node_metadata(node) for node in work.graph.node],
        "actual_values": _graph_value_snapshot(work),
    }


def _empty_totals() -> Dict[str, Dict[str, int]]:
    return {
        "tests": {"pass": 0, "fail": 0},
        "nodes": {"pass": 0, "fail": 0},
        "metadata": {"pass": 0, "fail": 0},
        "values": {"pass": 0, "fail": 0},
    }


def _score_test(
    name: str,
    expected_nodes: List[Dict[str, str]],
    actual_nodes: List[Dict[str, str]],
    node_ops: Optional[List[str]] = None,
    node_inputs: Optional[List[List[str]]] = None,
    node_outputs: Optional[List[List[str]]] = None,
    expected_values: Optional[List[Dict[str, Any]]] = None,
    actual_values: Optional[List[Dict[str, Any]]] = None,
    error: str = "",
    mermaid: str = "",
    graph: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score a single test and return a row dict for the JSON payload.

    The ``"mermaid"`` key is only present in the returned dict when a
    non-empty ``mermaid`` string is provided, so consumers should use
    ``row.get("mermaid", "")`` to retrieve it safely.
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
                "inputs": inputs,
                "outputs": outputs,
            }
        )

    # Build value-level comparison (inputs, outputs, initializers)
    values: List[Dict[str, Any]] = []
    matched_values = 0
    total_values = 0
    if expected_values is not None or actual_values is not None:
        exp_map = {v["name"]: v for v in (expected_values or [])}
        act_map = {v["name"]: v for v in (actual_values or [])}
        # dict.fromkeys preserves insertion order while deduplicating names.
        all_names = list(
            dict.fromkeys(
                [v["name"] for v in (expected_values or [])]
                + [v["name"] for v in (actual_values or [])]
            )
        )
        for val_name in all_names:
            exp_entry = exp_map.get(val_name, {})
            act_entry = act_map.get(val_name, {})
            exp_meta = exp_entry.get("metadata", {})
            act_meta = act_entry.get("metadata", {})
            exp_tags = str(exp_meta.get(VALUE_TAG_METADATA_KEY, "")).strip()
            act_tags = str(act_meta.get(VALUE_TAG_METADATA_KEY, "")).strip()
            # Values with no shape-tag metadata on either side carry no signal for
            # shape-tag coverage and should not dilute the ratio.
            if not exp_tags and not act_tags:
                continue
            exp_has_tags = bool(exp_tags)
            act_has_tags = bool(act_tags)
            val_success = exp_meta == act_meta and exp_has_tags and act_has_tags
            total_values += 1
            if val_success:
                matched_values += 1
            else:
                success = False
            values.append(
                {
                    "name": val_name,
                    "kind": exp_entry.get("kind") or act_entry.get("kind", ""),
                    "expected": exp_meta,
                    "actual": act_meta,
                    "success": val_success,
                }
            )

    # When a test has nodes but no metadata was expected or produced for any of
    # them, it trivially passes but carries no signal about shape-tag quality.
    # In shape-tag coverage context this means the expected annotations are
    # missing from the test case, which is itself an error.
    missing_metadata = (
        total_nodes > 0
        and total_metadata == 0
        and not _has_expected_value_metadata(expected_values)
        and not error
    )
    if missing_metadata:
        success = False

    row: Dict[str, Any] = {
        "name": name,
        "success": success,
        "error": error,
        "missing_metadata": missing_metadata,
        "matched_nodes": matched_nodes,
        "total_nodes": total_nodes,
        "matched_metadata": matched_metadata,
        "total_metadata": total_metadata,
        "matched_values": matched_values,
        "total_values": total_values,
        "nodes": nodes,
        "values": values,
    }
    if mermaid:
        row["mermaid"] = mermaid
    normalized_graph = _normalize_graph(graph)
    if normalized_graph:
        row["graph"] = normalized_graph
    return row


def build_payload(
    tag=DEFAULT_TAG,
    *,
    discover: Callable[..., List[Dict[str, Any]]] = discover_shape_tag_tests,
    run: Callable[[Any], Dict[str, Any]] = run_shape_tag_analysis,
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
                expected_values=list(test.get("expected_values", [])),
                actual_values=list(info.get("actual_values", [])),
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
                expected_values=list(test.get("expected_values", [])),
                actual_values=[],
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
        totals["values"]["pass"] += row["matched_values"]
        totals["values"]["fail"] += max(row["total_values"] - row["matched_values"], 0)
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
    json_path = os.path.join(args.cache_dir, "onnx-light", "shape_tag_coverage.json")
    try:
        payload = build_payload(tag=args.tag, limit=args.limit)
    except Exception as exc:  # noqa: BLE001
        _log(f"ERROR: failed to record shape-tag coverage: {exc}")
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
