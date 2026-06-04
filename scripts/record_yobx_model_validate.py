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
import time
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE = "cpu"
DEFAULT_ATOL = 0.02
DEFAULT_TASK = "text-generation"

DEFAULT_MODELS: Tuple[Dict[str, Any], ...] = (
    dict(
        model="arnir0/Tiny-LLM",
        dtype="float16",
        atol=0.02,
        device="cpu",
        task="text-generation",
    ),
    dict(
        model="microsoft/Phi-4-reasoning",
        dtype="float16",
        atol=0.02,
        device="cpu",
        task="text-generation",
    ),
)

# Each exporter configuration is fully described by a small dict so that the
# dashboard can render meaningful column headers. ``label`` is the unique
# identifier used in the JSON snapshot (both in ``exporters`` and in the
# ``label`` field of each result row).
DEFAULT_EXPORTERS: Tuple[Dict[str, str], ...] = (
    {"label": "yobx", "exporter": "yobx", "optimization": "default"},
    {"label": "dynamo-ir", "exporter": "dynamo", "optimization": "ir"},
)


def _coerce_model_entry(
    item: Any,
    *,
    default_dtype: str = DEFAULT_DTYPE,
    default_device: str = DEFAULT_DEVICE,
    default_atol: float = DEFAULT_ATOL,
    default_task: str = DEFAULT_TASK,
) -> Dict[str, Any]:
    """Return a fully-populated model entry dict from a string id or dict.

    Accepted forms:

    * ``"org/model"`` – a bare HuggingFace model id; defaults are filled in.
    * ``{"model": ..., "dtype": ..., "device": ..., "atol": ..., "task": ...}``
      – any missing key falls back to the supplied default.
    """
    if isinstance(item, str):
        data: Dict[str, Any] = {"model": item}
    elif isinstance(item, dict):
        data = dict(item)
    else:
        raise TypeError(
            f"Model entry must be a str or dict, got {type(item).__name__}"
        )
    if not data.get("model"):
        raise ValueError(f"Model entry is missing a 'model' field: {item!r}")
    data.setdefault("dtype", default_dtype)
    data.setdefault("device", default_device)
    data.setdefault("atol", default_atol)
    data.setdefault("task", default_task)
    return data


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


def is_cell_working(summary: Any, model_atol: Optional[float] = None) -> bool:
    """Return ``True`` when the export *and* the discrepancy check succeeded.

    ``summary`` may be either a ``ValidateSummary`` instance or a plain dict
    (the latter is used by the tests).

    When ``model_atol`` is provided, a cell is also considered working if the
    export succeeded and the maximum absolute discrepancy is below the
    per-model tolerance, even if ``validate_model`` flagged the run as failed
    because it used a stricter default tolerance.
    """
    export = _summary_get(summary, "export")
    discrepancies = _summary_get(summary, "discrepancies")
    if export != "OK":
        return False
    if discrepancies == "OK":
        return True
    # ``discrepancies`` was set but reported as failed by ``validate_model``.
    # If the caller supplied a per-model ``atol`` that is more permissive than
    # the one used during validation, honour it: a cell where the observed
    # maximum absolute error is within the model's declared tolerance is still
    # considered working for our dashboard.
    if model_atol is not None and discrepancies:
        max_abs = _to_float(_summary_get(summary, "discrepancies_max_abs"))
        if max_abs is not None and max_abs <= model_atol:
            return True
    # ``discrepancies`` is only set when ``do_run=True``. If it was not set
    # the export ran but we cannot conclude that the model is "working"
    # numerically, so we conservatively report False.
    return False


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
    duration_s: Optional[float] = None,
    model_atol: Optional[float] = None,
) -> Dict[str, Any]:
    """Pick a JSON-serialisable subset of the fields returned by ``validate_model``."""
    working = is_cell_working(summary, model_atol=model_atol)
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
        "duration_s": _to_float(duration_s),
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


def _list_xlsx(folder: Optional[str]) -> set:
    """Return the set of all ``.xlsx`` files found recursively under ``folder``."""
    if not folder or not os.path.isdir(folder):
        return set()
    out = set()
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".xlsx"):
                out.add(os.path.join(root, name))
    return out


def _read_yobx_export_duration(xlsx_path: str) -> Optional[float]:
    """Return ``stat_time_export_and_post_processing`` from the ``extra`` sheet.

    The yobx exporter saves a companion ``.xlsx`` next to every exported
    ``.onnx`` file. The ``extra`` sheet of that workbook is a small
    key/value table with scalar metrics recorded during the export.
    Returns ``None`` when the workbook, sheet, key or value cannot be
    read (missing ``openpyxl``, corrupted file, ...).
    """
    try:  # pragma: no cover - exercised when openpyxl is available
        from openpyxl import load_workbook
    except Exception:  # noqa: BLE001 - optional dependency
        return None
    try:
        wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception:  # noqa: BLE001 - never fail the recorder on a bad file
        return None
    try:
        if "extra" not in wb.sheetnames:
            return None
        ws = wb["extra"]
        key = "stat_time_export_and_post_processing"
        # ``extra`` is a two-column key/value table; the first row is a
        # header. We do not assume which column holds the key, so scan both.
        for row in ws.iter_rows(values_only=True):
            if not row:
                continue
            # Find the key cell and pair it with the first non-key cell.
            try:
                idx = row.index(key)
            except ValueError:
                continue
            for j, cell in enumerate(row):
                if j == idx:
                    continue
                value = _to_float(cell)
                if value is not None:
                    return value
            return None
        return None
    finally:
        try:
            wb.close()
        except Exception:  # noqa: BLE001
            pass


def run_validate_one(
    entry: Dict[str, Any],
    exporter_cfg: Dict[str, str],
    verbose: int = 0,
    dump_folder: Optional[str] = None,
    quiet: bool = True,
) -> Any:
    """Run :func:`yobx.torch.validate.validate_model` for one (model, exporter)."""
    # Lazy import so ``--help`` works without the heavy ``torch`` stack.
    from yobx.torch.validate import validate_model

    summary, _data = validate_model(
        model_id=entry["model"],
        exporter=exporter_cfg["exporter"],
        optimization=exporter_cfg["optimization"],
        dtype=entry.get("dtype", DEFAULT_DTYPE),
        device=entry.get("device", DEFAULT_DEVICE),
        do_run=True,
        quiet=quiet,
        verbose=verbose,
        patch="transformers",
        dump_folder=dump_folder,
    )
    return summary


def detect_task(model_id: str) -> str:
    """Return the HuggingFace task detected for ``model_id``.

    Uses :func:`yobx.torch.validate._detect_task` against the model's
    ``AutoConfig`` (e.g. ``"text-generation"``, ``"fill-mask"``,
    ``"image-classification"``, ``"feature-extraction"``).  Returns an
    empty string when the task cannot be detected (missing dependencies,
    network failure, ...).
    """
    try:  # pragma: no cover - exercised via the recording script
        from transformers import AutoConfig
        from yobx.torch.validate import _detect_task

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return _detect_task(config) or ""
    except Exception as exc:  # noqa: BLE001 - never fail the recorder for this
        _log(f"  -> could not detect task for {model_id}: {type(exc).__name__}: {exc}")
        return ""


def run_all(
    models: Tuple[Dict[str, Any], ...],
    exporters: Tuple[Dict[str, str], ...],
    limit: Optional[int] = None,
    verbose: int = 0,
    dump_folder: Optional[str] = None,
    quiet: bool = True,
) -> List[Tuple[str, Dict[str, str], Any, float]]:
    """Run ``validate_model`` for every (model, exporter) combination."""
    import shutil
    import tempfile

    items = list(models)
    if limit is not None:
        items = items[:limit]
    out: List[Tuple[str, Dict[str, str], Any, float]] = []
    for entry in items:
        model_id = entry["model"]
        for exporter_cfg in exporters:
            _log(
                f"Validating {model_id} (dtype={entry.get('dtype')}, "
                f"device={entry.get('device')}) with "
                f"exporter={exporter_cfg['exporter']} "
                f"optimization={exporter_cfg['optimization']}..."
            )
            # For yobx, the export time we want to record is the
            # ``stat_time_export_and_post_processing`` metric stored in
            # the ``extra`` sheet of the workbook saved next to the
            # ONNX file. We therefore make sure that a dump folder is
            # always passed for yobx runs, even when the user did not
            # supply one (in which case we use a per-cell temporary
            # directory which is removed once the metric is read).
            is_yobx = exporter_cfg["exporter"] == "yobx"
            tmp_dump: Optional[str] = None
            effective_dump = dump_folder
            if is_yobx and effective_dump is None:
                tmp_dump = tempfile.mkdtemp(prefix="yobx_dump_")
                effective_dump = tmp_dump
            # Snapshot the xlsx files present before the export so we can
            # detect the workbook(s) the yobx exporter just produced.
            pre_xlsx = _list_xlsx(effective_dump) if is_yobx else set()
            start = time.monotonic()
            try:
                summary = run_validate_one(
                    entry,
                    exporter_cfg,
                    verbose=verbose,
                    dump_folder=effective_dump,
                    quiet=quiet,
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
            duration_s = time.monotonic() - start
            # For yobx, prefer the ``stat_time_export_and_post_processing``
            # metric recorded in the ``extra`` sheet of the generated
            # workbook over the wall-clock time, which also includes the
            # discrepancy check and other unrelated work.
            if is_yobx and effective_dump:
                new_xlsx = _list_xlsx(effective_dump) - pre_xlsx
                def _safe_mtime(p: str) -> float:
                    try:
                        return os.path.getmtime(p)
                    except OSError:
                        return 0.0
                for xlsx_path in sorted(new_xlsx, key=_safe_mtime, reverse=True):
                    metric = _read_yobx_export_duration(xlsx_path)
                    if metric is not None:
                        duration_s = metric
                        break
            if tmp_dump is not None:
                shutil.rmtree(tmp_dump, ignore_errors=True)
            _log(f"  -> done in {duration_s:.2f}s")
            out.append((model_id, exporter_cfg, summary, duration_s))
    return out


def build_payload(
    raw_results: List[Tuple[str, Dict[str, str], Any, float]],
    models: Tuple[Dict[str, Any], ...],
    exporters: Tuple[Dict[str, str], ...],
    dtype: str,
    device: str,
    commit: Optional[str],
    previous_payload: Optional[Dict[str, Any]] = None,
    now: Optional[str] = None,
    tasks: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble the JSON snapshot to be written to disk."""
    current_date = now or dt.datetime.now(tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    previous_index = _index_previous_results(previous_payload or {})
    previous_tasks = (previous_payload or {}).get("tasks") or {}
    # Map model id -> entry dict for per-model lookups.
    entries_by_id: Dict[str, Dict[str, Any]] = {e["model"]: e for e in models}
    model_ids: List[str] = [e["model"] for e in models]
    resolved_tasks: Dict[str, str] = {}
    for m in model_ids:
        entry = entries_by_id.get(m, {})
        if tasks and tasks.get(m):
            resolved_tasks[m] = tasks[m]
        elif entry.get("task"):
            resolved_tasks[m] = entry["task"]
        elif previous_tasks.get(m):
            resolved_tasks[m] = previous_tasks[m]
        else:
            resolved_tasks[m] = ""

    results: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, int]] = {}
    for label in (e["label"] for e in exporters):
        totals[label] = {"success": 0, "failure": 0, "total": 0}

    for item in raw_results:
        if len(item) == 4:
            model_id, exporter_cfg, summary, duration_s = item
        else:
            model_id, exporter_cfg, summary = item
            duration_s = None
        row = _normalise_result(
            model_id,
            exporter_cfg,
            summary,
            duration_s,
            model_atol=_to_float(entries_by_id.get(model_id, {}).get("atol")),
        )
        row["task"] = resolved_tasks.get(model_id, "")
        entry = entries_by_id.get(model_id, {})
        row["dtype"] = entry.get("dtype", dtype)
        row["device"] = entry.get("device", device)
        row["atol"] = _to_float(entry.get("atol"))
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
        "models": model_ids,
        "model_entries": [dict(e) for e in models],
        "tasks": resolved_tasks,
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
            f"Defaults to: {', '.join(e['model'] for e in DEFAULT_MODELS)}."
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
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=True,
        help=(
            "Forward quiet=True to validate_model so per-model output is "
            "suppressed (default)."
        ),
    )
    parser.add_argument(
        "--no-quiet",
        dest="quiet",
        action="store_false",
        help=(
            "Forward quiet=False to validate_model so the underlying "
            "tracebacks and progress output are shown."
        ),
    )
    parser.add_argument(
        "--dump-folder",
        default=None,
        help=(
            "Folder where validate_model dumps its intermediate artefacts "
            "(ONNX files, captured inputs, ...). Intended for local runs. "
            "The folder is created if missing and the script changes its "
            "working directory to it before running so any relative paths "
            "(including --cache-dir) are resolved inside it."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.models:
        models = tuple(
            _coerce_model_entry(
                m, default_dtype=args.dtype, default_device=args.device
            )
            for m in args.models
        )
    else:
        models = tuple(
            _coerce_model_entry(
                e, default_dtype=args.dtype, default_device=args.device
            )
            for e in DEFAULT_MODELS
        )
    exporters = DEFAULT_EXPORTERS

    dump_folder: Optional[str] = None
    if args.dump_folder:
        dump_folder = os.path.abspath(args.dump_folder)
        os.makedirs(dump_folder, exist_ok=True)
        _log(f"Using dump folder: {dump_folder} (chdir into it)")
        os.chdir(dump_folder)

    out_dir = os.path.join(args.cache_dir, args.repo)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_validate.json")

    previous_payload = _load_existing_cache(out_path)

    tasks = {e["model"]: e.get("task") or detect_task(e["model"]) for e in models}

    raw_results = run_all(
        models=models,
        exporters=exporters,
        limit=args.limit,
        verbose=args.verbose,
        dump_folder=dump_folder,
        quiet=args.quiet,
    )
    payload = build_payload(
        raw_results=raw_results,
        models=models,
        exporters=exporters,
        dtype=args.dtype,
        device=args.device,
        commit=args.commit,
        previous_payload=previous_payload,
        tasks=tasks,
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
