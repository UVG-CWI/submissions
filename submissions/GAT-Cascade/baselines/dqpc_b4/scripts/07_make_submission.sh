#!/usr/bin/env bash
set -euo pipefail

# Package final colored PLY files into a submission folder and zip archive.
# Default mode follows the organizer's current guidance: zip final PLY files only.
#
# Example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=test
#   export INPUT_ROOT=outputs/dqpc_b4/test_final
#   bash baselines/dqpc_b4/scripts/07_make_submission.sh
#
# Optional full internal package with manifest/runtime/hardware/README:
#   export PACKAGE_MODE=full
#   export VALIDATE_RGB=1
#   bash baselines/dqpc_b4/scripts/07_make_submission.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
INPUT_ROOT="${INPUT_ROOT:-outputs/dqpc_b4/${SPLIT}_final}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
PLY_GLOB="${PLY_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
RUNTIME_GLOB="${RUNTIME_GLOB:-outputs/dqpc_b4/runtime/${SPLIT}_*.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/dqpc_b4/submission_${SPLIT}}"
ZIP_PATH="${ZIP_PATH:-outputs/dqpc_b4/submission_${SPLIT}.zip}"
METHOD_NAME="${METHOD_NAME:-DQPC-B4-PUDense-DAKNN-GQENet}"
PACKAGE_MODE="${PACKAGE_MODE:-ply-only}"
VALIDATE_RGB="${VALIDATE_RGB:-1}"
DQPC_ROOT="${DQPC_ROOT:-}"
EXPECTED_ROOT="${EXPECTED_ROOT:-}"
EXPECTED_GLOB="${EXPECTED_GLOB:-}"
GEOMETRY_REFERENCE_ROOT="${GEOMETRY_REFERENCE_ROOT:-}"
QUALITY_CONFIG="${QUALITY_CONFIG:-}"
XYZ_ATOL="${XYZ_ATOL:-0.0001}"
LIMIT="${LIMIT:-0}"

if [[ -n "$DQPC_ROOT" && -z "$EXPECTED_ROOT" ]]; then
  EXPECTED_ROOT="$(dqpc_default_input_root)"
fi
if [[ -n "$EXPECTED_ROOT" && -z "$EXPECTED_GLOB" ]]; then
  EXPECTED_GLOB="$(dqpc_default_cg_glob "$EXPECTED_ROOT")"
fi
if [[ -z "$GEOMETRY_REFERENCE_ROOT" && -d "outputs/dqpc_b4/${SPLIT}_geometry" ]]; then
  GEOMETRY_REFERENCE_ROOT="outputs/dqpc_b4/${SPLIT}_geometry"
fi
if [[ -z "$QUALITY_CONFIG" && -f "outputs/dqpc_b4/eval/valid_gqenet_blend.json" ]]; then
  QUALITY_CONFIG="outputs/dqpc_b4/eval/valid_gqenet_blend.json"
fi

ARGS=(
  --input-root "$INPUT_ROOT"
  --ply-glob "$PLY_GLOB"
  --runtime-glob "$RUNTIME_GLOB"
  --out-dir "$OUT_DIR"
  --zip-path "$ZIP_PATH"
  --method-name "$METHOD_NAME"
  --package-mode "$PACKAGE_MODE"
  --xyz-atol "$XYZ_ATOL"
  --limit "$LIMIT"
)

if [[ "$VALIDATE_RGB" == "1" ]]; then
  ARGS+=(--validate-rgb)
fi
if [[ -n "$EXPECTED_ROOT" || -n "$EXPECTED_GLOB" ]]; then
  : "${EXPECTED_ROOT:?Set both EXPECTED_ROOT and EXPECTED_GLOB}"
  : "${EXPECTED_GLOB:?Set both EXPECTED_ROOT and EXPECTED_GLOB}"
  ARGS+=(--expected-root "$EXPECTED_ROOT" --expected-glob "$EXPECTED_GLOB")
fi
if [[ -n "$GEOMETRY_REFERENCE_ROOT" ]]; then
  ARGS+=(--geometry-reference-root "$GEOMETRY_REFERENCE_ROOT")
fi
if [[ -n "$QUALITY_CONFIG" ]]; then
  ARGS+=(--quality-config "$QUALITY_CONFIG")
fi

echo "[DQPC-B4] Make submission"
echo "  INPUT_ROOT=$INPUT_ROOT"
echo "  PLY_GLOB=$PLY_GLOB"
echo "  EXPECTED_GLOB=${EXPECTED_GLOB:-<disabled>}"
echo "  GEOMETRY_REFERENCE_ROOT=${GEOMETRY_REFERENCE_ROOT:-<disabled>}"
echo "  QUALITY_CONFIG=${QUALITY_CONFIG:-<disabled>}"
echo "  VALIDATE_RGB=$VALIDATE_RGB"

"$PYTHON_BIN" baselines/dqpc_b4/make_submission.py "${ARGS[@]}"
