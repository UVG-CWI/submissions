#!/usr/bin/env bash
set -euo pipefail

# Train GQE-Net color refinement on DQPC enhanced colored frames.
#
# Run after 02_infer_geometry_and_recolor.sh has produced colored enhanced PLY.
#
# Example for Y channel:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=train
#   export ENHANCED_ROOT=outputs/dqpc_b4/train_colored
#   export HE_ROOT=$DQPC_ROOT/train
#   export CHANNEL=0
#   bash baselines/dqpc_b4/scripts/03_train_gqenet_dqpc.sh
#
# Train all three channels:
#   for CHANNEL in 0 1 2; do
#     export CHANNEL
#     bash baselines/dqpc_b4/scripts/03_train_gqenet_dqpc.sh
#   done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
DQPC_ROOT="${DQPC_ROOT:-}"
SPLIT="${SPLIT:-train}"
ENHANCED_ROOT="${ENHANCED_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
ENHANCED_GLOB="${ENHANCED_GLOB:-$(dqpc_default_cg_glob "$ENHANCED_ROOT")}"
if [[ -n "$DQPC_ROOT" && -d "$DQPC_ROOT/$SPLIT" ]]; then
  DEFAULT_HE_ROOT="$DQPC_ROOT/$SPLIT"
else
  DEFAULT_HE_ROOT="$DQPC_ROOT"
fi
HE_ROOT="${HE_ROOT:-$DEFAULT_HE_ROOT}"
CHANNEL="${CHANNEL:-0}"
PATCH_SIZE="${PATCH_SIZE:-2048}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.0025}"
COORD_SCALE="${COORD_SCALE:-1.0}"
TRAIN_PROFILE="${TRAIN_PROFILE:-v3_compatible}"
case "$TRAIN_PROFILE" in
  v3_compatible)
    DEFAULT_RESIDUAL=0
    DEFAULT_TARGET_MAX_DISTANCE=0
    DEFAULT_FAR_TARGET_WEIGHT=1
    DEFAULT_FRAME_SAMPLING=v3_random
    DEFAULT_MAX_CACHED_FRAMES=8
    ;;
  v3_compatible_fast)
    DEFAULT_RESIDUAL=0
    DEFAULT_TARGET_MAX_DISTANCE=0
    DEFAULT_FAR_TARGET_WEIGHT=1
    DEFAULT_FRAME_SAMPLING=cache_local
    DEFAULT_MAX_CACHED_FRAMES=2
    ;;
  robust_residual)
    DEFAULT_RESIDUAL=1
    DEFAULT_TARGET_MAX_DISTANCE=20
    DEFAULT_FAR_TARGET_WEIGHT=0.2
    DEFAULT_FRAME_SAMPLING=cache_local
    DEFAULT_MAX_CACHED_FRAMES=2
    ;;
  *)
    echo "TRAIN_PROFILE must be v3_compatible, v3_compatible_fast, or robust_residual" >&2
    exit 2
    ;;
esac
RESIDUAL="${RESIDUAL:-$DEFAULT_RESIDUAL}"
TARGET_MAX_DISTANCE="${TARGET_MAX_DISTANCE:-$DEFAULT_TARGET_MAX_DISTANCE}"
FAR_TARGET_WEIGHT="${FAR_TARGET_WEIGHT:-$DEFAULT_FAR_TARGET_WEIGHT}"
DISTANCE_UNIT="${DISTANCE_UNIT:-mm}"
MAX_CACHED_FRAMES="${MAX_CACHED_FRAMES:-$DEFAULT_MAX_CACHED_FRAMES}"
FRAME_SAMPLING="${FRAME_SAMPLING:-$DEFAULT_FRAME_SAMPLING}"
PATCHES_PER_FRAME="${PATCHES_PER_FRAME:-8}"
VAL_ENHANCED_ROOT="${VAL_ENHANCED_ROOT:-}"
VAL_ENHANCED_GLOB="${VAL_ENHANCED_GLOB:-}"
VAL_HE_ROOT="${VAL_HE_ROOT:-}"
VAL_SAMPLES="${VAL_SAMPLES:-512}"
OUT_DIR="${OUT_DIR:-outputs/dqpc_b4/gqenet_ckpts}"
INIT_CKPT="${INIT_CKPT:-}"
SEED="${SEED:-1}"

: "${HE_ROOT:?Set HE_ROOT or DQPC_ROOT so HE targets can be found}"

ARGS=(
  --gqenet-root external/GQE-Net
  --enhanced-root "$ENHANCED_ROOT"
  --enhanced-glob "$ENHANCED_GLOB"
  --he-root "$HE_ROOT"
  --channel "$CHANNEL"
  --patch-size "$PATCH_SIZE"
  --samples-per-epoch "$SAMPLES_PER_EPOCH"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --epochs "$EPOCHS"
  --lr "$LR"
  --coord-scale "$COORD_SCALE"
  --target-max-distance "$TARGET_MAX_DISTANCE"
  --far-target-weight "$FAR_TARGET_WEIGHT"
  --distance-unit "$DISTANCE_UNIT"
  --max-cached-frames "$MAX_CACHED_FRAMES"
  --frame-sampling "$FRAME_SAMPLING"
  --patches-per-frame "$PATCHES_PER_FRAME"
  --out-dir "$OUT_DIR"
  --seed "$SEED"
)
if [[ "$RESIDUAL" == "1" ]]; then
  ARGS+=(--residual)
fi
if [[ -n "$VAL_ENHANCED_ROOT" || -n "$VAL_ENHANCED_GLOB" || -n "$VAL_HE_ROOT" ]]; then
  : "${VAL_ENHANCED_ROOT:?Set VAL_ENHANCED_ROOT, VAL_ENHANCED_GLOB, and VAL_HE_ROOT together}"
  : "${VAL_ENHANCED_GLOB:?Set VAL_ENHANCED_ROOT, VAL_ENHANCED_GLOB, and VAL_HE_ROOT together}"
  : "${VAL_HE_ROOT:?Set VAL_ENHANCED_ROOT, VAL_ENHANCED_GLOB, and VAL_HE_ROOT together}"
  ARGS+=(
    --val-enhanced-root "$VAL_ENHANCED_ROOT"
    --val-enhanced-glob "$VAL_ENHANCED_GLOB"
    --val-he-root "$VAL_HE_ROOT"
    --val-samples "$VAL_SAMPLES"
  )
fi

if [[ -n "$INIT_CKPT" ]]; then
  ARGS+=(--init-ckpt "$INIT_CKPT")
fi

echo "[DQPC-B4] Train GQE-Net DQPC channel $CHANNEL"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  ENHANCED_GLOB=$ENHANCED_GLOB"
echo "  HE_ROOT=$HE_ROOT"
echo "  PATCH_SIZE=$PATCH_SIZE"
echo "  SAMPLES_PER_EPOCH=$SAMPLES_PER_EPOCH"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  EPOCHS=$EPOCHS"
echo "  OUT_DIR=$OUT_DIR"
echo "  TRAIN_PROFILE=$TRAIN_PROFILE"
echo "  RESIDUAL=$RESIDUAL"
echo "  TARGET_MAX_DISTANCE=$TARGET_MAX_DISTANCE"
echo "  FAR_TARGET_WEIGHT=$FAR_TARGET_WEIGHT"
echo "  DISTANCE_UNIT=$DISTANCE_UNIT"
echo "  MAX_CACHED_FRAMES=$MAX_CACHED_FRAMES"
echo "  FRAME_SAMPLING=$FRAME_SAMPLING"
echo "  PATCHES_PER_FRAME=$PATCHES_PER_FRAME"
echo "  VAL_ENHANCED_GLOB=${VAL_ENHANCED_GLOB:-<disabled>}"
echo "  VAL_SAMPLES=$VAL_SAMPLES"
echo "  SEED=$SEED"

"$PYTHON_BIN" baselines/dqpc_b4/train_gqenet_dqpc.py "${ARGS[@]}"
