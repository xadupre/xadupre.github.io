#!/usr/bin/env bash
# Record a yet-another-onnx-builder model_validate snapshot locally.
#
# This mirrors the steps performed by
# ``.github/workflows/record_yobx_model_validate.yml`` but runs entirely on
# the developer's machine, without committing or pushing anything. The
# resulting snapshot is written to
# ``cache_data/yet-another-onnx-builder/model_validate.json`` (relative to
# this repository) and can then be inspected or committed manually.
#
# Usage::
#
#     scripts/record_yobx_model_validate_local.sh [--yobx-dir DIR]
#         [--ref REF] [--python BIN] [--skip-install]
#         [-- ...extra args forwarded to record_yobx_model_validate.py]
#
# Examples::
#
#     # Use a checkout of yet-another-onnx-builder living next to this repo
#     scripts/record_yobx_model_validate_local.sh \
#         --yobx-dir ../yet-another-onnx-builder
#
#     # Quick smoke test: only record the first model
#     scripts/record_yobx_model_validate_local.sh \
#         --yobx-dir ../yet-another-onnx-builder -- --limit 1
#
# Environment variables:
#   YOBX_DIR    Same as --yobx-dir. Defaults to ``../yet-another-onnx-builder``.
#   PYTHON      Same as --python. Defaults to ``python``.

set -euo pipefail

YOBX_DIR="${YOBX_DIR:-../yet-another-onnx-builder}"
REF=""
PYTHON_BIN="${PYTHON:-python}"
SKIP_INSTALL=0
EXTRA_ARGS=()

usage() {
    sed -n '2,30p' "$0"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yobx-dir)
            YOBX_DIR="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$YOBX_DIR" ]]; then
    echo "yet-another-onnx-builder directory not found: $YOBX_DIR" >&2
    echo "Clone it first, e.g.:" >&2
    echo "    git clone https://github.com/xadupre/yet-another-onnx-builder.git $YOBX_DIR" >&2
    exit 1
fi
YOBX_DIR="$(cd "$YOBX_DIR" && pwd)"

if [[ -n "$REF" ]]; then
    echo ">>> Checking out $REF in $YOBX_DIR"
    git -C "$YOBX_DIR" fetch --tags origin "$REF" || git -C "$YOBX_DIR" fetch origin
    git -C "$YOBX_DIR" checkout "$REF"
fi

if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    if "$PYTHON_BIN" -m yobx --help >/dev/null 2>&1; then
        echo ">>> yobx already installed, skipping install"
    else
        echo ">>> Installing yet-another-onnx-builder from $YOBX_DIR"
        # Mirrors the install step of record_yobx_model_validate.yml. ``pandas``,
        # ``openpyxl`` and ``optree`` are required at runtime by
        # ``validate_model`` and recent ``torch`` releases respectively.
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -e "$YOBX_DIR[torch,transformers,onnxscript]" \
            pandas openpyxl optree
    fi
fi

echo ">>> pip freeze"
"$PYTHON_BIN" -m pip freeze

commit_sha="$(git -C "$YOBX_DIR" rev-parse HEAD)"
echo ">>> Recording model validate snapshot (yobx commit ${commit_sha})"
cd "$SITE_DIR"
YOBX_COMMIT="$commit_sha" "$PYTHON_BIN" -u scripts/record_yobx_model_validate.py \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ">>> Done. Snapshot written under $SITE_DIR/cache_data/yet-another-onnx-builder/"
