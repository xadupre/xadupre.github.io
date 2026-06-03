"""Record ``yobx validate`` discrepancy snapshots for selected HuggingFace models.

This script exercises :func:`yobx.torch.validate.validate_model` (the same
entry point used by ``python -m yobx validate``) over a fixed set of model
ids and a fixed set of ONNX exporter configurations. The resulting summary
for each (model, exporter) cell is serialised to
``cache_data/yet-another-onnx-builder/model_validate.json``.

The JSON file is consumed by
``dashboard/yet-another-onnx-builder/model-validate.html`` which renders a
small table showing whether the model still exports without discrepancies
for each exporter and what was the last date the cell was working.

For every cell, the previous snapshot is consulted so that the
``last_working_date`` (and ``last_working_commit``) of a cell is preserved
when a previously working configuration starts failing. The intent is to
make it obvious when an export regressed and when it was last known good.

The run uses ``dtype=float16`` and ``device=cpu`` and exercises the
following exporter configurations:

* ``yobx`` with ``optimization='default'``
* ``dynamo`` with ``optimization='ir'``

Usage::

    python scripts/record_yobx_model_validate.py [--cache-dir DIR]
        [--model ID ...] [--limit N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MODELS: Tuple[str, ...] = (
    "arnir0/Tiny-LLM",
    "microsoft/Phi-4-reasoning",
)

# Each exporter configuration is fully described by a small dict so that the
# dashboard can render meaningful column headers. ``label`` is the unique
# identifier used in the JSON snapshot (both in ``exporters`` and in the
# ``label`` field of each result row).
DEFAULT_EXPORTERS: Tuple[Dict[str, str], ...] = (
    {"label": "yobx", "exporter": "yobx", "optimization": "default"},
    {"label": "dynamo-ir", "exporter": "dynamo", "optimization": "ir"},
)

DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE = "cpu"


def _log(message: str) -> None:
    """Print ``message`` prefixed with a UTC timestamp."""
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


def collect_versions() -> Dict[str, str]:
    """Return the versions of the relevant packages, if importable."""
    versions: Dict[str, str] = {}
    for name in (
        "yobx",
        "torch",
        "transformers",
        "onnx",
        "onnxruntime",
        "onnxscript",
    ):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - best effort, optional packages
            continue
        version = getattr(module, "__version__", None)
        if version:
            versions[name] = str(version)
    return versions


def _stringify_error(value: Any) -> str:
    """Return a short, single-line string for an exporter error message."""
    if value is None:
        return ""
    text = str(value)
    if "\n" in text:
        text = text.splitlines()[0]
    if len(text) > 400:
        text = text[:397] + "..."
    return text


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # JSON does not support Infinity/NaN; coerce to None so the payload
    # remains valid JSON parsable by browsers.
    if not math.isfinite(f):
        return None
    return f


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_get(summary: Any, key: str, default: Any = None) -> Any:
    """Best-effort getter that works for both dicts and ``ValidateSummary``."""
    if summary is None:
        return default
    try:
        return summary.get(key, default)
    except AttributeError:
        pass
    if hasattr(summary, key):
        return getattr(summary, key, default)
    try:
        return summary[key]
    except (TypeError, KeyError, IndexError):
        return default


def is_cell_working(summary: Any) -> bool:
    """Return ``True`` when the export *and* the discrepancy check succeeded.

    ``summary`` may be either a ``ValidateSummary`` instance or a plain dict
    (the latter is used by the tests).
    """
    export = _summary_get(summary, "export")
    discrepancies = _summary_get(summary, "discrepancies")
    if export != "OK":
        return False
    # ``discrepancies`` is only set when ``do_run=True``. If it was not set
    # the export ran but we cannot conclude that the model is "working"
    # numerically, so we conservatively report False.
    return discrepancies == "OK"


def _first_error(summary: Any) -> Tuple[str, str]:
    """Return ``(error_step, error_message)`` for the first failing step."""
    steps = (
        ("config", "error_config"),
        ("tokenizer", "error_tokenizer"),
        ("model", "error_model"),
        ("observer", "error_observer"),
        ("export", "error_export"),
        ("discrepancies", "error_discrepancies"),
    )
    for step, field in steps:
        value = _summary_get(summary, field)
        if value:
            return step, _stringify_error(value)
    return "", ""


def _normalise_result(
    model_id: str,
    exporter_cfg: Dict[str, str],
    summary: Any,
) -> Dict[str, Any]:
    """Pick a JSON-serialisable subset of the fields returned by ``validate_model``."""
    working = is_cell_working(summary)
    error_step, error = _first_error(summary)

    return {
        "model_id": model_id,
        "label": exporter_cfg["label"],
        "exporter": exporter_cfg["exporter"],
        "optimization": exporter_cfg["optimization"],
        "success": 1 if working else 0,
        "export": _summary_get(summary, "export") or "",
        "discrepancies": _summary_get(summary, "discrepancies") or "",
        "discrepancies_ok": _to_int(_summary_get(summary, "discrepancies_ok")),
        "discrepancies_total": _to_int(_summary_get(summary, "discrepancies_total")),
        "discrepancies_max_abs": _to_float(
            _summary_get(summary, "discrepancies_max_abs")
        ),
        "discrepancies_atol": _to_float(_summary_get(summary, "discrepancies_atol")),
        "n_nodes": _to_int(_summary_get(summary, "n_nodes")),
        "top_op_types": _summary_get(summary, "top_op_types") or "",
        "error_step": error_step,
        "error": error,
    }


def _load_existing_cache(path: str) -> Dict[str, Any]:
    """Read the previous snapshot if it exists, returning ``{}`` on any error."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _index_previous_results(payload: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Index previous results by ``(model_id, label)`` for quick lookup."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in payload.get("results", []) or []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("model_id", "")), str(row.get("label", "")))
        out[key] = row
    return out


def merge_last_working(
    row: Dict[str, Any],
    previous_row: Optional[Dict[str, Any]],
    current_date: str,
    current_commit: str,
) -> Dict[str, Any]:
    """Return ``row`` updated with ``last_working_date`` / ``last_working_commit``.

    * If the current run is working, the "last working" timestamp is updated
      to the current run.
    * Otherwise the previous values are carried over (when present), so that
      the dashboard can show the last known good date for a failing cell.
    """
    if row.get("success") == 1:
        row["last_working_date"] = current_date
        row["last_working_commit"] = current_commit or ""
    elif previous_row is not None:
        prev_date = previous_row.get("last_working_date") or ""
        prev_commit = previous_row.get("last_working_commit") or ""
        if prev_date:
            row["last_working_date"] = prev_date
            row["last_working_commit"] = prev_commit
        else:
            row["last_working_date"] = ""
            row["last_working_commit"] = ""
    else:
        row["last_working_date"] = ""
        row["last_working_commit"] = ""
    return row


def run_validate_one(
    model_id: str,
    exporter_cfg: Dict[str, str],
    dtype: str,
    device: str,
    verbose: int = 0,
) -> Any:
    """Run :func:`yobx.torch.validate.validate_model` for one (model, exporter)."""
    # Lazy import so ``--help`` works without the heavy ``torch`` stack.
    from yobx.torch.validate import validate_model

    summary, _data = validate_model(
        model_id=model_id,
        exporter=exporter_cfg["exporter"],
        optimization=exporter_cfg["optimization"],
        dtype=dtype,
        device=device,
        do_run=True,
        quiet=True,
        verbose=verbose,
    )
    return summary


def run_all(
    models: Tuple[str, ...],
    exporters: Tuple[Dict[str, str], ...],
    dtype: str,
    device: str,
    limit: Optional[int] = None,
    verbose: int = 0,
) -> List[Tuple[str, Dict[str, str], Any]]:
    """Run ``validate_model`` for every (model, exporter) combination."""
    items = list(models)
    if limit is not None:
        items = items[:limit]
    out: List[Tuple[str, Dict[str, str], Any]] = []
    for model_id in items:
        for exporter_cfg in exporters:
            _log(
                f"Validating {model_id} with exporter={exporter_cfg['exporter']} "
                f"optimization={exporter_cfg['optimization']}..."
            )
            try:
                summary = run_validate_one(
                    model_id, exporter_cfg, dtype=dtype, device=device, verbose=verbose
                )
            except Exception as exc:  # noqa: BLE001 - we never want to crash CI
                _log(f"  -> raised: {type(exc).__name__}: {exc}")
                # Build a synthetic summary so the dashboard can still display
                # a failed cell with a meaningful error message.
                summary = {
                    "model_id": model_id,
                    "export": "FAILED",
                    "error_export": f"{type(exc).__name__}: {exc}",
                }
            out.append((model_id, exporter_cfg, summary))
    return out


def build_payload(
    raw_results: List[Tuple[str, Dict[str, str], Any]],
    models: Tuple[str, ...],
    exporters: Tuple[Dict[str, str], ...],
    dtype: str,
    device: str,
    commit: Optional[str],
    previous_payload: Optional[Dict[str, Any]] = None,
    now: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the JSON snapshot to be written to disk."""
    current_date = now or dt.datetime.now(tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    previous_index = _index_previous_results(previous_payload or {})

    results: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {}
    for label in (e["label"] for e in exporters):
        totals[label] = {"success": 0, "failure": 0, "total": 0}

    for model_id, exporter_cfg, summary in raw_results:
        row = _normalise_result(model_id, exporter_cfg, summary)
        previous_row = previous_index.get((model_id, exporter_cfg["label"]))
        merge_last_working(row, previous_row, current_date, commit or "")
        results.append(row)
        bucket = totals.setdefault(
            exporter_cfg["label"], {"success": 0, "failure": 0, "total": 0}
        )
        bucket["total"] += 1
        if row["success"] == 1:
            bucket["success"] += 1
        else:
            bucket["failure"] += 1

    return {
        "date": current_date,
        "commit": commit or "",
        "versions": collect_versions(),
        "dtype": dtype,
        "device": device,
        "models": list(models),
        "exporters": [dict(e) for e in exporters],
        "totals": totals,
        "results": results,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default=os.path.join("cache_data"),
        help="Root directory of the JSON cache (default: %(default)s).",
    )
    parser.add_argument(
        "--repo",
        default="yet-another-onnx-builder",
        help="Sub-directory of the cache to write into (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "HuggingFace model id to validate. May be repeated. "
            f"Defaults to: {', '.join(DEFAULT_MODELS)}."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        help=f"Dtype passed to validate_model (default: {DEFAULT_DTYPE}).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device passed to validate_model (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--commit",
        default=os.environ.get("YOBX_COMMIT", ""),
        help="Commit SHA of yet-another-onnx-builder used for the snapshot.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to the first N models (mainly for local testing).",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level forwarded to validate_model (default: 0).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    models = tuple(args.models) if args.models else DEFAULT_MODELS
    exporters = DEFAULT_EXPORTERS

    out_dir = os.path.join(args.cache_dir, args.repo)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_validate.json")

    previous_payload = _load_existing_cache(out_path)

    raw_results = run_all(
        models=models,
        exporters=exporters,
        dtype=args.dtype,
        device=args.device,
        limit=args.limit,
        verbose=args.verbose,
    )
    payload = build_payload(
        raw_results=raw_results,
        models=models,
        exporters=exporters,
        dtype=args.dtype,
        device=args.device,
        commit=args.commit,
        previous_payload=previous_payload,
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")

    _log(
        f"Wrote {out_path}: {len(payload['results'])} results, "
        f"{len(models)} models, {len(exporters)} exporters."
    )
    for label, bucket in payload["totals"].items():
        _log(
            f"  {label}: {bucket['success']}/{bucket['total']} working "
            f"({bucket['failure']} failed)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
