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
4. Fetches the GitHub repository trees for both projects and adds
   ``onnx_light_source_url`` / ``cgen_source_url`` fields so the dashboard
   can offer an inline C++ code view per operator.
5. When ``emx-onnx-cgen`` is installed, compiles a representative ONNX
   backend test model for each supported operator (using
   ``emx-onnx-cgen compile``) and stores the generated C source inline in
   the ``cgen_source_code`` field so the dashboard can display it without
   a network request.

The resulting JSON is consumed by
``dashboard/onnx-light/cgen-comparison.html``.

Usage::

    python scripts/record_cgen_comparison.py [--cache-dir DIR]
                                             [--skip-cgen-compile]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

SUPPORT_OPS_URL = (
    "https://raw.githubusercontent.com/emmtrix/emx-onnx-cgen/main/SUPPORT_OPS.md"
)
CGEN_REPO_URL = "https://github.com/emmtrix/emx-onnx-cgen"

# GitHub repository coordinates
ONNX_LIGHT_OWNER = "xadupre"
ONNX_LIGHT_REPO = "onnx-light"
# Kernel sources live under this prefix in the onnx-light repo, grouped into
# per-category subdirectories (e.g. math/, nn/, ...).
# Files are named  kernel_<opname_lower>.cc
ONNX_LIGHT_KERNELS_PREFIX = "onnx_light/onnx_extensions/kernels/kernels/"

CGEN_OWNER = "emmtrix"
CGEN_REPO = "emx-onnx-cgen"
# Jinja2 C templates live here, named  <snake_op>_op.c.j2
CGEN_TEMPLATES_PREFIX = "src/emx_onnx_cgen/templates/"

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


def fetch_github_tree(
    owner: str,
    repo: str,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch the recursive git tree for a GitHub repository.

    Returns a list of tree entries (dicts with at least ``path`` and ``type``
    keys), or an empty list when the request fails (e.g. rate-limited or
    network unavailable).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"
    headers: Dict[str, str] = {
        "User-Agent": "xadupre.github.io-record-cgen-comparison",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"token {token}"
    _log(f"Fetching GitHub tree for {owner}/{repo}")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        entries = data.get("tree", [])
        _log(f"  Got {len(entries)} tree entries for {owner}/{repo}.")
        return entries
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        json.JSONDecodeError,
        OSError,
    ) as exc:
        _log(f"  Warning: failed to fetch tree for {owner}/{repo}: {exc}")
        return []


def build_onnx_light_source_map(
    tree: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Build a map from operator name key (lowercase) to raw GitHub content URL.

    Only files matching ``ONNX_LIGHT_KERNELS_PREFIX*/kernel_<name>.cc`` are
    considered.
    """
    result: Dict[str, str] = {}
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path: str = entry.get("path", "")
        if not path.startswith(ONNX_LIGHT_KERNELS_PREFIX):
            continue
        filename = path.rsplit("/", 1)[-1]
        if not (filename.startswith("kernel_") and filename.endswith(".cc")):
            continue
        op_key = filename[len("kernel_") : -len(".cc")]  # e.g. "abs"
        raw_url = (
            f"https://raw.githubusercontent.com/"
            f"{ONNX_LIGHT_OWNER}/{ONNX_LIGHT_REPO}/main/{path}"
        )
        result[op_key] = raw_url
    return result


def build_cgen_source_map(
    tree: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Build a map from template stem (lowercase, no ``_op.c.j2``) to raw URL.

    Only files under ``CGEN_TEMPLATES_PREFIX`` with the ``_op.c.j2`` suffix
    are considered (e.g. ``conv_op.c.j2`` → key ``conv``).
    """
    result: Dict[str, str] = {}
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path: str = entry.get("path", "")
        if not path.startswith(CGEN_TEMPLATES_PREFIX):
            continue
        filename = path.rsplit("/", 1)[-1]
        if not filename.endswith("_op.c.j2"):
            continue
        stem = filename[: -len("_op.c.j2")]  # e.g. "conv"
        raw_url = (
            f"https://raw.githubusercontent.com/"
            f"{CGEN_OWNER}/{CGEN_REPO}/main/{path}"
        )
        result[stem] = raw_url
    return result


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case (e.g. ``BatchNormalization`` → ``batch_normalization``)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def find_onnx_light_source_url(
    name: str,
    source_map: Dict[str, str],
) -> Optional[str]:
    """Return the raw content URL for the onnx-light kernel implementing *name*.

    Tries an exact lowercase match first, then a snake_case conversion.
    """
    key = name.lower()
    if key in source_map:
        return source_map[key]
    key_snake = _camel_to_snake(name)
    if key_snake in source_map:
        return source_map[key_snake]
    return None


def find_cgen_source_url(
    name: str,
    source_map: Dict[str, str],
) -> Optional[str]:
    """Return the raw content URL for the emx-onnx-cgen template for *name*.

    Tries an exact lowercase match first, then a snake_case conversion.
    """
    key = name.lower()
    if key in source_map:
        return source_map[key]
    key_snake = _camel_to_snake(name)
    if key_snake in source_map:
        return source_map[key_snake]
    return None


def _onnx_backend_test_node_dir() -> Optional[str]:
    """Return the path to the ONNX backend node test-data directory, or None."""
    try:
        import onnx.backend.test.data as _d  # noqa: PLC0415

        # Use __path__ because the package may be a namespace package (__file__
        # is None for namespace packages).
        paths = list(getattr(_d, "__path__", []))
        if not paths:
            # Fallback: try __file__ (works for regular packages)
            f = getattr(_d, "__file__", None)
            if f:
                paths = [os.path.dirname(f)]
        for base in paths:
            candidate = os.path.join(base, "node")
            if os.path.isdir(candidate):
                return candidate
        return None
    except Exception:  # noqa: BLE001
        return None


def _materialize_onnx_light_node_dir(dest_dir: str) -> Optional[str]:
    """Write onnx-light backend node test models under *dest_dir*.

    ``onnx-weekly`` no longer ships the ``onnx.backend.test.data`` directory,
    so the single-node test models used to feed ``emx-onnx-cgen compile`` are
    taken from the ``onnx-light`` catalog instead (collected via
    :func:`onnx_light.onnx_lib.backend.test.case.collect_test_case`).

    Each collected test case whose ``model`` carries exactly one node is
    written to ``<dest_dir>/<name>/model.onnx`` in the same layout that
    :func:`build_op_to_test_model_map` expects. Returns *dest_dir* when at
    least one model was written, otherwise ``None``.
    """
    try:
        import onnx  # noqa: PLC0415
        from onnx_light.onnx_lib.backend.test.case import (  # noqa: PLC0415
            collect_test_case,
        )
    except Exception:  # noqa: BLE001 - onnx-light or onnx not importable
        return None

    written = 0
    cases = collect_test_case(include_big=True)
    for name, tc in cases.items():
        if not name:
            continue
        model = getattr(tc, "model", None)
        if model is None:
            continue
        if not isinstance(model, onnx.ModelProto):
            out = onnx.ModelProto()
            out.ParseFromString(model.SerializeToString())
            model = out
        if len(model.graph.node) != 1:
            continue
        case_dir = os.path.join(dest_dir, str(name))
        os.makedirs(case_dir, exist_ok=True)
        onnx.save(model, os.path.join(case_dir, "model.onnx"))
        written += 1
    return dest_dir if written else None
    test_data_dir: str,
) -> Dict[Tuple[str, str], str]:
    """Scan *test_data_dir* and return a ``(domain, op_name) → model_path`` map.

    Only directories that contain a single-node model are considered so that
    the compiled output is representative of that one operator.  The first
    matching directory (alphabetical order) is used for each ``(domain,
    op_name)`` pair.
    """
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        return {}

    result: Dict[Tuple[str, str], str] = {}
    for dirname in sorted(os.listdir(test_data_dir)):
        model_path = os.path.join(test_data_dir, dirname, "model.onnx")
        if not os.path.exists(model_path):
            continue
        try:
            model = onnx.load(model_path)
            nodes = list(model.graph.node)
            if len(nodes) != 1:
                continue
            node = nodes[0]
            op = node.op_type
            domain = node.domain or "ai.onnx"
            key: Tuple[str, str] = (domain, op)
            if key not in result:
                result[key] = model_path
        except Exception:  # noqa: BLE001
            pass
    return result


def generate_cgen_source_for_op(model_path: str) -> Optional[str]:
    """Compile *model_path* with ``emx-onnx-cgen compile`` and return the C source.

    *model_path* is expected to be an absolute path to an ONNX model file
    from the ONNX backend test-data directory (built by
    :func:`build_op_to_test_model_map`).  It is never derived from external
    or user-controlled input.

    Returns ``None`` when the tool is not available or compilation fails.
    """
    if not shutil.which("emx-onnx-cgen"):
        return None
    # Sanity-check: only compile files that exist and have an .onnx extension.
    # model_path always comes from build_op_to_test_model_map (ONNX test-data
    # directory), never from user input, so subprocess injection is not possible.
    is_absolute = os.path.isabs(model_path)
    has_onnx_extension = model_path.endswith(".onnx")
    file_exists = os.path.isfile(model_path)
    if not (is_absolute and has_onnx_extension and file_exists):
        return None
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "model.c")
        result = subprocess.run(  # noqa: S603  # path is validated above; not user-controlled
            ["emx-onnx-cgen", "compile", model_path, out_path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path, encoding="utf-8") as fh:
            return fh.read()


def build_cgen_source_code_map(
    cgen_rows: List[Dict[str, Any]],
    test_data_dir: str,
) -> Dict[Tuple[str, str], str]:
    """Generate C source for every supported operator and return a lookup map.

    The key is ``(domain, op_name)``; the value is the generated C source
    string.  Operators for which no test model is found or compilation fails
    are silently skipped.
    """
    op_to_model = build_op_to_test_model_map(test_data_dir)
    _log(
        f"Built op→model map with {len(op_to_model)} entries " f"from {test_data_dir}."
    )

    result: Dict[Tuple[str, str], str] = {}
    for row in cgen_rows:
        if not row.get("in_cgen"):
            continue
        domain: str = row.get("domain", "ai.onnx")
        name: str = row.get("name", "")
        key: Tuple[str, str] = (domain, name)
        model_path = op_to_model.get(key)
        if model_path is None:
            continue
        code = generate_cgen_source_for_op(model_path)
        if code is not None:
            result[key] = code
    _log(
        f"Generated emx-onnx-cgen C source for {len(result)} "
        f"of {sum(1 for r in cgen_rows if r.get('in_cgen'))} supported operators."
    )
    return result


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
    onnx_light_source_map: Optional[Dict[str, str]] = None,
    cgen_source_map: Optional[Dict[str, str]] = None,
    cgen_source_code_map: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Dict[str, Any]]:
    """Merge cgen and onnx-light rows by (domain, name).

    Operators present in only one dataset are still included with the
    missing side set to ``False`` / ``0``.

    When *onnx_light_source_map* and/or *cgen_source_map* are provided the
    matching raw-content URL is stored in ``onnx_light_source_url`` /
    ``cgen_source_url`` respectively (``None`` when no match is found).

    When *cgen_source_code_map* is provided the generated C source string is
    stored in ``cgen_source_code`` (keyed by ``(domain, name)``).
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
        row: Dict[str, Any] = {
            "domain": domain,
            "name": name,
            "in_onnx_light": bool(light.get("in_onnx_light", False)),
            "in_cgen": bool(cgen.get("in_cgen", False)),
            "onnx_light_backend_tests": int(
                light.get("onnx_light_backend_tests", 0) or 0
            ),
        }
        if onnx_light_source_map is not None:
            url = find_onnx_light_source_url(name, onnx_light_source_map)
            if url is not None:
                row["onnx_light_source_url"] = url
        if cgen_source_map is not None:
            url = find_cgen_source_url(name, cgen_source_map)
            if url is not None:
                row["cgen_source_url"] = url
        if cgen_source_code_map is not None:
            code = cgen_source_code_map.get((domain, name))
            if code is not None:
                row["cgen_source_code"] = code
        merged.append(row)
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
    github_token: Optional[str] = None,
    skip_cgen_compile: bool = False,
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

    # Attempt to fetch GitHub trees for source-URL lookup (best-effort).
    onnx_light_tree = fetch_github_tree(
        ONNX_LIGHT_OWNER, ONNX_LIGHT_REPO, token=github_token
    )
    onnx_light_source_map = build_onnx_light_source_map(onnx_light_tree)
    _log(f"Built onnx-light source map with {len(onnx_light_source_map)} entries.")

    cgen_tree = fetch_github_tree(CGEN_OWNER, CGEN_REPO, token=github_token)
    cgen_source_map = build_cgen_source_map(cgen_tree)
    _log(f"Built emx-onnx-cgen source map with {len(cgen_source_map)} entries.")

    # Generate C source via emx-onnx-cgen compile (best-effort, optional).
    cgen_source_code_map: Optional[Dict[Tuple[str, str], str]] = None
    if not skip_cgen_compile and shutil.which("emx-onnx-cgen"):
        test_node_dir = _onnx_backend_test_node_dir()
        if test_node_dir:
            cgen_source_code_map = build_cgen_source_code_map(cgen_rows, test_node_dir)
        else:
            # ``onnx-weekly`` no longer bundles ``onnx.backend.test.data``; fall
            # back to the onnx-light backend test catalog and materialize the
            # single-node models to a temporary directory.
            with tempfile.TemporaryDirectory() as tmp_node_dir:
                materialized = _materialize_onnx_light_node_dir(tmp_node_dir)
                if materialized:
                    _log(
                        "ONNX backend test data directory not found; using "
                        "onnx-light backend test models instead."
                    )
                    cgen_source_code_map = build_cgen_source_code_map(
                        cgen_rows, materialized
                    )
                else:
                    _log(
                        "ONNX backend test data directory not found and onnx-light "
                        "backend test models unavailable; skipping emx-onnx-cgen "
                        "compile step."
                    )
    elif skip_cgen_compile:
        _log("Skipping emx-onnx-cgen compile step (--skip-cgen-compile).")
    else:
        _log("emx-onnx-cgen not found on PATH; skipping C source generation.")

    rows = merge_rows(
        cgen_rows,
        light_rows,
        onnx_light_source_map=onnx_light_source_map,
        cgen_source_map=cgen_source_map,
        cgen_source_code_map=cgen_source_code_map,
    )
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
    parser.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN"),
        help=(
            "GitHub personal-access token used for API requests "
            "(default: $GITHUB_TOKEN env var). Increases the unauthenticated "
            "rate limit from 60 to 5,000 requests/hour."
        ),
    )
    parser.add_argument(
        "--skip-cgen-compile",
        action="store_true",
        default=False,
        help=(
            "Skip the ``emx-onnx-cgen compile`` step that generates inline C "
            "source for each supported operator (useful when emx-onnx-cgen is "
            "not installed or to speed up a dry-run)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    schema_json_path = os.path.join(
        args.cache_dir, "onnx-light", "schema_comparison.json"
    )
    json_path = os.path.join(args.cache_dir, "onnx-light", "cgen_comparison.json")

    payload = build_payload(
        schema_json_path=schema_json_path,
        github_token=args.github_token,
        skip_cgen_compile=args.skip_cgen_compile,
    )
    write_payload(json_path, payload)
    _log(
        f"Wrote {len(payload['rows'])} operator rows to {json_path} "
        f"(totals={payload['totals']})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
