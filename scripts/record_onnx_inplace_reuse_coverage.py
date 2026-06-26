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
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_TAGS: Tuple[str, ...] = ("inplace",)
DEFAULT_TAG: str = ",".join(DEFAULT_TAGS)
METADATA_KEYS: Tuple[str, ...] = (
    "onnx_light.inplace_reuse",
    "onnx_light.release_after",
    "onnx_light.release_after_shape_tag",
)


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


def discover_inplace_tests(tag=DEFAULT_TAGS) -> List[Dict[str, Any]]:
    """Return backend tests whose ``tag`` matches ``tag``.

    Each entry is a dictionary ``{"name", "model", "expected_nodes", "node_ops"}``
    where ``expected_nodes`` is the per-node metadata subset relevant to the
    in-place analysis.
    """
    from onnx_light.onnx_lib.backend.test.case import collect_test_case

    tags = _normalize_tags(tag)
    cases = collect_test_case()
    discovered: List[Dict[str, Any]] = []
    for name, tc in cases.items():
        if not name:
            continue
        case_tags = _normalize_tags(getattr(tc, "tag", "") or "")
        if tags and not any(t in tags for t in case_tags):
            continue
        model = getattr(tc, "model", None)
        if model is None:
            continue
        nodes = list(getattr(model.graph, "node", []))
        expected_nodes = [_node_metadata(node) for node in nodes]
        discovered.append(
            {
                "name": str(name),
                "model": model,
                "expected_nodes": expected_nodes,
                "node_ops": [str(getattr(node, "op_type", "")) for node in nodes],
            }
        )
    discovered.sort(key=lambda item: item["name"])
    return discovered


def run_inplace_analysis(model) -> Dict[str, Any]:
    """Run onnx-light's inplace metadata analysis on ``model``."""
    from onnx_light.onnx_optim import shape_inference as si

    work = _clone_model(model)
    for node in work.graph.node:
        _clear_node_metadata(node)

    ctx = si.ShapesContext()
    si.compute_shape_model(ctx, work)

    inplace = si.ComputeContext()
    inplace.compute_inplace_reuse_graph(work.graph, ctx)
    inplace.write_to_metadata(work.graph)

    return {
        "actual_nodes": [_node_metadata(node) for node in work.graph.node],
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
    error: str = "",
    memory: Optional[List[Any]] = None,
) -> Dict[str, Any]:
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
        keys = sorted(set(expected) | set(actual))
        metadata_matches = sum(1 for key in keys if expected.get(key) == actual.get(key))
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
                "memory": memory[index] if memory is not None and index < len(memory) else None,
            }
        )

    return {
        "name": name,
        "success": success,
        "error": error,
        "matched_nodes": matched_nodes,
        "total_nodes": total_nodes,
        "matched_metadata": matched_metadata,
        "total_metadata": total_metadata,
        "nodes": nodes,
    }


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
                memory=list(info.get("memory", [])) if info.get("memory") is not None else None,
            )
        except Exception as exc:  # noqa: BLE001 - keep recording other tests
            _log(f"Unhandled error for {test['name']}: {exc}")
            traceback.print_exc()
            row = _score_test(
                test["name"],
                list(test.get("expected_nodes", [])),
                [],
                node_ops=list(test.get("node_ops", [])),
                error=str(exc) or type(exc).__name__,
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
