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
#         [--dump-folder DIR] [--quiet|--no-quiet]
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
#   DUMP_FOLDER Same as --dump-folder. When set, the recorder script ``chdir``s
#               into this folder and writes intermediate artefacts there.
#   QUIET       Same as --quiet/--no-quiet. Set to ``0`` to forward
#               ``--no-quiet`` to ``record_yobx_model_validate.py`` so the
#               underlying ``validate_model`` output is shown. Defaults to ``1``.

set -euo pipefail

YOBX_DIR="${YOBX_DIR:-../yet-another-onnx-builder}"
REF=""
PYTHON_BIN="${PYTHON:-python}"
SKIP_INSTALL=0
DUMP_FOLDER="${DUMP_FOLDER:-}"
QUIET="${QUIET:-1}"
EXTRA_ARGS=()

usage() {
    sed -n '2,35p' "$0"
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
        --dump-folder)
            DUMP_FOLDER="$2"
            shift 2
            ;;
        --quiet)
            QUIET=1
            shift
            ;;
        --no-quiet)
            QUIET=0
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
        # ``sentencepiece`` and ``tiktoken`` are tokenizer backends required
        # by some HuggingFace models (for example ``mistralai/Mistral-7B-v0.3``)
        # without which ``AutoTokenizer.from_pretrained`` aborts with
        # ``Couldn't instantiate the backend tokenizer from one of:``.
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -e "$YOBX_DIR[torch,transformers,onnxscript]" \
            pandas openpyxl optree sentencepiece tiktoken
        # Install the development version of Olive from main so the
        # ``olive-modelbuilder`` column exercises the latest
        # ``ModelBuilder`` pass. ``onnxruntime-genai`` is the runtime
        # dependency that pass uses to build the ONNX graph.
        "$PYTHON_BIN" -m pip install --upgrade \
            "git+https://github.com/microsoft/Olive.git" onnxruntime-genai
    fi
fi

echo ">>> pip freeze"
"$PYTHON_BIN" -m pip freeze

commit_sha="$(git -C "$YOBX_DIR" rev-parse HEAD)"
echo ">>> Recording model validate snapshot (yobx commit ${commit_sha})"
cd "$SITE_DIR"
DUMP_ARGS=()
if [[ -n "$DUMP_FOLDER" ]]; then
    DUMP_ARGS=(--dump-folder "$DUMP_FOLDER")
    echo ">>> Using dump folder: $DUMP_FOLDER"
fi
QUIET_ARGS=()
if [[ "$QUIET" -eq 0 ]]; then
    QUIET_ARGS=(--no-quiet)
    echo ">>> Forwarding --no-quiet to record_yobx_model_validate.py"
else
    QUIET_ARGS=(--quiet)
fi
YOBX_COMMIT="$commit_sha" "$PYTHON_BIN" -u scripts/record_yobx_model_validate.py \
    ${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"} \
    ${QUIET_ARGS[@]+"${QUIET_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ">>> Done. Snapshot written under $SITE_DIR/cache_data/yet-another-onnx-builder/"
