#!/usr/bin/env bash
set -euo pipefail

# Run GQE-Net Y/U/V color refinement on PU-Dense + recoloring outputs.
#
# Example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export SPLIT=valid
#   export INPUT_ROOT=outputs/dqpc_b4/valid_colored
#   export MODEL_Y=outputs/dqpc_b4/gqenet_ckpts/y/model_19.pth
#   export MODEL_U=outputs/dqpc_b4/gqenet_ckpts/u/model_19.pth
#   export MODEL_V=outputs/dqpc_b4/gqenet_ckpts/v/model_19.pth
#   export PREDICTION_MODE=auto
#   bash baselines/dqpc_b4/scripts/04_infer_gqenet_dqpc.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${MODEL_Y:?Set MODEL_Y to the trained Y-channel GQE-Net checkpoint}"
: "${MODEL_U:?Set MODEL_U to the trained U-channel GQE-Net checkpoint}"
: "${MODEL_V:?Set MODEL_V to the trained V-channel GQE-Net checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
INPUT_ROOT="${INPUT_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_GLOB="${INPUT_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_gqenet_raw}"
PATCH_SIZE="${PATCH_SIZE:-2048}"
PATCH_STRIDE="${PATCH_STRIDE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CENTER_MODE="${CENTER_MODE:-v3_stride}"
COORD_SCALE="${COORD_SCALE:-1.0}"
PREDICTION_MODE="${PREDICTION_MODE:-auto}"
RESIDUAL="${RESIDUAL:-}"
BLEND_Y="${BLEND_Y:-1.0}"
BLEND_U="${BLEND_U:-1.0}"
BLEND_V="${BLEND_V:-1.0}"
MAX_DELTA_Y="${MAX_DELTA_Y:-0}"
MAX_DELTA_U="${MAX_DELTA_U:-0}"
MAX_DELTA_V="${MAX_DELTA_V:-0}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_gqenet.jsonl}"

if [[ -n "$RESIDUAL" ]]; then
  if [[ "$RESIDUAL" == "1" ]]; then
    PREDICTION_MODE="residual"
  elif [[ "$RESIDUAL" == "0" ]]; then
    PREDICTION_MODE="absolute"
  else
    echo "RESIDUAL must be 0 or 1 when set" >&2
    exit 2
  fi
fi

echo "[DQPC-B4] Infer GQE-Net DQPC"
echo "  INPUT_GLOB=$INPUT_GLOB"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  MODEL_Y=$MODEL_Y"
echo "  MODEL_U=$MODEL_U"
echo "  MODEL_V=$MODEL_V"
echo "  PREDICTION_MODE=$PREDICTION_MODE"
echo "  BLEND_Y/U/V=$BLEND_Y/$BLEND_U/$BLEND_V"
echo "  MAX_DELTA_Y/U/V=$MAX_DELTA_Y/$MAX_DELTA_U/$MAX_DELTA_V"
echo "  CENTER_MODE=$CENTER_MODE"
echo "  SKIP_EXISTING=$SKIP_EXISTING"

ARGS=(
  --gqenet-root external/GQE-Net
  --input-root "$INPUT_ROOT"
  --input-glob "$INPUT_GLOB"
  --out-root "$OUT_ROOT"
  --model-y "$MODEL_Y"
  --model-u "$MODEL_U"
  --model-v "$MODEL_V"
  --patch-size "$PATCH_SIZE"
  --patch-stride "$PATCH_STRIDE"
  --batch-size "$BATCH_SIZE"
  --center-mode "$CENTER_MODE"
  --coord-scale "$COORD_SCALE"
  --prediction-mode "$PREDICTION_MODE"
  --blend-y "$BLEND_Y"
  --blend-u "$BLEND_U"
  --blend-v "$BLEND_V"
  --max-delta-y "$MAX_DELTA_Y"
  --max-delta-u "$MAX_DELTA_U"
  --max-delta-v "$MAX_DELTA_V"
  --limit "$LIMIT"
  --runtime-log "$RUNTIME_LOG"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

"$PYTHON_BIN" baselines/dqpc_b4/infer_gqenet_dqpc.py "${ARGS[@]}"
