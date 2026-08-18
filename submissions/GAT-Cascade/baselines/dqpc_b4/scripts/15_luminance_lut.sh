#!/usr/bin/env bash
set -euo pipefail

# Fit a train/HE luminance LUT or apply it to valid/test colored geometry.
#
# Fit:
#   export MODE=fit
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=train
#   export SOURCE_ROOT=outputs/dqpc_b4/train_colored
#   bash baselines/dqpc_b4/scripts/15_luminance_lut.sh
#
# Apply:
#   export MODE=apply
#   export SPLIT=valid
#   export SOURCE_ROOT=outputs/dqpc_b4/valid_colored
#   export OUT_ROOT=outputs/dqpc_b4/valid_y_lut_raw
#   bash baselines/dqpc_b4/scripts/15_luminance_lut.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-fit}"
SPLIT="${SPLIT:-train}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"
MODEL_PATH="${MODEL_PATH:-outputs/dqpc_b4/y_lut/train_y_lut.json}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_y_lut_raw}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_y_lut.jsonl}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
SOURCE_GLOB="${SOURCE_GLOB:-$(dqpc_default_cg_glob "$SOURCE_ROOT")}"

ARGS=(
  --mode "$MODE"
  --source-root "$SOURCE_ROOT"
  --source-glob "$SOURCE_GLOB"
  --limit "$LIMIT"
)

if [[ "$MODE" == "fit" ]]; then
  : "${DQPC_ROOT:?Set DQPC_ROOT for HE-supervised LUT fitting}"
  GT_ROOT="${GT_ROOT:-$(dqpc_default_input_root)}"
  BINS="${BINS:-256}"
  MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-200000}"
  MAX_DISTANCE="${MAX_DISTANCE:-10}"
  DISTANCE_UNIT="${DISTANCE_UNIT:-mm}"
  MAX_TARGET_DELTA="${MAX_TARGET_DELTA:-80}"
  IDENTITY_PRIOR="${IDENTITY_PRIOR:-500}"
  MAX_CORRECTION="${MAX_CORRECTION:-40}"
  SEED="${SEED:-1}"
  ARGS+=(
    --gt-root "$GT_ROOT"
    --model-out "$MODEL_PATH"
    --bins "$BINS"
    --max-points-per-frame "$MAX_POINTS_PER_FRAME"
    --max-distance "$MAX_DISTANCE"
    --distance-unit "$DISTANCE_UNIT"
    --max-target-delta "$MAX_TARGET_DELTA"
    --identity-prior "$IDENTITY_PRIOR"
    --max-correction "$MAX_CORRECTION"
    --seed "$SEED"
  )
elif [[ "$MODE" == "apply" ]]; then
  STRENGTH="${STRENGTH:-1}"
  MAX_DELTA="${MAX_DELTA:-40}"
  ARGS+=(
    --model-in "$MODEL_PATH"
    --out-root "$OUT_ROOT"
    --runtime-log "$RUNTIME_LOG"
    --strength "$STRENGTH"
    --max-delta "$MAX_DELTA"
  )
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip-existing)
  fi
else
  echo "MODE must be fit or apply" >&2
  exit 2
fi

echo "[DQPC-B4] Luminance LUT"
echo "  MODE=$MODE"
echo "  SOURCE_GLOB=$SOURCE_GLOB"
echo "  MODEL_PATH=$MODEL_PATH"
if [[ "$MODE" == "apply" ]]; then
  echo "  OUT_ROOT=$OUT_ROOT"
fi

"$PYTHON_BIN" baselines/dqpc_b4/luminance_lut.py "${ARGS[@]}"
