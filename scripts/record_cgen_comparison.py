"""Compare the C++ operator implementations in onnx-light with those produced
by the `emx-onnx-cgen <https://github.com/emmtrix/emx-onnx-cgen>`_ code
generator for every ONNX operator.

The script:

1. Downloads ``SUPPORT_OPS.md`` from the ``emmtrix/emx-onnx-cgen`` GitHub
   repository and parses the operator-support table.
2. Reads the latest ``cache_data/onnx-light/schema_comparison.json`` snapshot
   (produced by ``record_size_onnx_light.yml``) to know which operators are
   implemented in onnx-light.
3. Merges the two datasets by ``(domain, operator name)`` and writes the
   result to ``cache_data/onnx-light/cgen_comparison.json``.

The resulting JSON is consumed by
``dashboard/onnx-light/cgen-comparison.html``.

Usage::

    python scripts/record_cgen_comparison.py [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SUPPORT_OPS_URL = (
    "https://raw.githubusercontent.com/emmtrix/emx-onnx-cgen/main/SUPPORT_OPS.md"
)
CGEN_REPO_URL = "https://github.com/emmtrix/emx-onnx-cgen"

# Row pattern: | <operator> | ✅ | or | <operator> | ❌ |
_ROW_RE = re.compile(r"^\|\s*(.+?)\s*\|\s*([✅❌])\s*\|", re.MULTILINE)


def _log(message: str) -> None:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {message}", flush=True)


def _format_iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    else:
        value = value.astimezone(dt.timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_support_ops_md(url: str = SUPPORT_OPS_URL) -> str:
    """Fetch and return the raw content of SUPPORT_OPS.md."""
    _log(f"Fetching {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "xadupre.github.io-record-cgen-comparison"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def parse_support_ops(content: str) -> List[Dict[str, Any]]:
    """Parse the SUPPORT_OPS.md table into a list of operator dicts.

    Each plain operator name (e.g. ``Abs``) is assigned domain ``ai.onnx``.
    Operators prefixed with a dotted namespace (e.g.
    ``ai.onnx.ml.LabelEncoder`` or ``com.microsoft.Attention``) have their
    domain extracted from the prefix and their bare name from the final
    component.
    """
    rows: List[Dict[str, Any]] = []
    for match in _ROW_RE.finditer(content):
        raw_name = match.group(1).strip()
        supported = match.group(2) == "✅"
        # Skip the header row if the regex accidentally captures it
        if raw_name.lower() in ("operator", "---"):
            continue
        if "." in raw_name:
            parts = raw_name.split(".")
            name = parts[-1]
            domain = ".".join(parts[:-1])
        else:
            name = raw_name
            domain = "ai.onnx"
        rows.append({"domain": domain, "name": name, "in_cgen": supported})
    return rows


def load_schema_comparison(json_path: str) -> List[Dict[str, Any]]:
    """Load the onnx-light schema comparison snapshot."""
    if not os.path.exists(json_path):
        _log(f"Schema comparison file not found: {json_path}")
        return []
    with open(json_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload.get("rows", []))


def merge_rows(
    cgen_rows: List[Dict[str, Any]],
    light_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge cgen and onnx-light rows by (domain, name).

    Operators present in only one dataset are still included with the
    missing side set to ``False`` / ``0``.
    """
    # Build lookup for onnx-light data
    light_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in light_rows:
        key = (r.get("domain", "ai.onnx"), r.get("name", ""))
        light_by_key[key] = r

    # Build lookup for cgen data
    cgen_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in cgen_rows:
        key = (r.get("domain", "ai.onnx"), r.get("name", ""))
        cgen_by_key[key] = r

    all_keys = sorted(set(light_by_key) | set(cgen_by_key))
    merged: List[Dict[str, Any]] = []
    for domain, name in all_keys:
        light = light_by_key.get((domain, name), {})
        cgen = cgen_by_key.get((domain, name), {})
        merged.append(
            {
                "domain": domain,
                "name": name,
                "in_onnx_light": bool(light.get("in_onnx_light", False)),
                "in_cgen": bool(cgen.get("in_cgen", False)),
                "onnx_light_backend_tests": int(
                    light.get("onnx_light_backend_tests", 0) or 0
                ),
            }
        )
    return merged


def compute_totals(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute aggregate counts across all merged rows."""
    totals: Dict[str, int] = {
        "onnx_light": 0,
        "cgen": 0,
        "both": 0,
        "only_onnx_light": 0,
        "only_cgen": 0,
        "neither": 0,
    }
    for r in rows:
        in_light = r.get("in_onnx_light", False)
        in_cgen = r.get("in_cgen", False)
        if in_light:
            totals["onnx_light"] += 1
        if in_cgen:
            totals["cgen"] += 1
        if in_light and in_cgen:
            totals["both"] += 1
        elif in_light:
            totals["only_onnx_light"] += 1
        elif in_cgen:
            totals["only_cgen"] += 1
        else:
            totals["neither"] += 1
    return totals


def build_payload(
    schema_json_path: str,
) -> Dict[str, Any]:
    """Fetch and merge all data; return the full payload dict."""
    content = fetch_support_ops_md()
    cgen_rows = parse_support_ops(content)
    _log(
        f"Parsed {len(cgen_rows)} operators from SUPPORT_OPS.md "
        f"({sum(1 for r in cgen_rows if r['in_cgen'])} supported)."
    )

    light_rows = load_schema_comparison(schema_json_path)
    _log(f"Loaded {len(light_rows)} operators from schema_comparison.json.")

    rows = merge_rows(cgen_rows, light_rows)
    totals = compute_totals(rows)

    return {
        "date": _format_iso(dt.datetime.now(tz=dt.timezone.utc)),
        "cgen_url": CGEN_REPO_URL,
        "cgen_support_ops_url": SUPPORT_OPS_URL,
        "totals": totals,
        "rows": rows,
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    schema_json_path = os.path.join(
        args.cache_dir, "onnx-light", "schema_comparison.json"
    )
    json_path = os.path.join(args.cache_dir, "onnx-light", "cgen_comparison.json")

    payload = build_payload(schema_json_path=schema_json_path)
    write_payload(json_path, payload)
    _log(
        f"Wrote {len(payload['rows'])} operator rows to {json_path} "
        f"(totals={payload['totals']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
