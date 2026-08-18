#!/usr/bin/env bash
set -euo pipefail

# Train or fine-tune PU-Dense geometry on DQPC CG/CG_aligned/CGv2 -> HE pairs.
#
# Run from any directory; the script will cd to the project root.
#
# Required:
#   DQPC_ROOT: dataset root.
#
# Default expected dataset layout, in priority order:
#   $DQPC_ROOT/train/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
#   $DQPC_ROOT/train/<sequence>/CG_aligned/15fps/*.ply
#   $DQPC_ROOT/train/<sequence>/CGv2_15/*.ply
# Matching HE paths are inferred automatically.
#
# Minimal example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
#
# Debug example, only 20 steps:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export STEPS=20
#   export SAVE_EVERY=20
#   export OUT_DIR=outputs/dqpc_b4/pudense_ckpts_debug
#   bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
#
# Full example with a specific conda Python and optional initial checkpoint:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=train
#   export INIT_CKPT=/path/to/pudense_pretrained.pth
#   export STEPS=2000
#   export SAVE_EVERY=500
#   export OUT_DIR=outputs/dqpc_b4/pudense_ckpts
#   bash baselines/dqpc_b4/scripts/01_train_pudense_dqpc.sh
#
# Override these if the dataset layout is different:
#   export CG_GLOB="/data/DQPC/train/*/consumer-grade_capture_system/CG_aligned/15fps/*.ply"
#   export HE_GLOB="/data/DQPC/train/*/high-end_capture_system/HE/15fps/*.ply"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root, for example /data/UVG-CWI-DQPC}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-train}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_ROOT="${INPUT_ROOT:-$(dqpc_default_input_root)}"
CG_GLOB="${CG_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
HE_GLOB="${HE_GLOB:-}"
VOXEL_SIZE="${VOXEL_SIZE:-auto}"
CROP_SIZE="${CROP_SIZE:-256}"
MAX_CG_POINTS="${MAX_CG_POINTS:-70000}"
MAX_HE_POINTS="${MAX_HE_POINTS:-280000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STEPS="${STEPS:-2000}"
LR="${LR:-0.0008}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LAST_KERNEL_SIZE="${LAST_KERNEL_SIZE:-5}"
OUT_DIR="${OUT_DIR:-outputs/dqpc_b4/pudense_ckpts}"
INIT_CKPT="${INIT_CKPT:-}"
SEED="${SEED:-1}"

ARGS=(
  --pudense-root external/PointCloudUpsampling
  --cg-glob "$CG_GLOB"
  --voxel-size "$VOXEL_SIZE"
  --crop-size "$CROP_SIZE"
  --max-cg-points "$MAX_CG_POINTS"
  --max-he-points "$MAX_HE_POINTS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --steps "$STEPS"
  --lr "$LR"
  --save-every "$SAVE_EVERY"
  --last-kernel-size "$LAST_KERNEL_SIZE"
  --out-dir "$OUT_DIR"
  --seed "$SEED"
)

if [[ -n "$HE_GLOB" ]]; then
  ARGS+=(--he-glob "$HE_GLOB")
fi

if [[ -n "$INIT_CKPT" ]]; then
  ARGS+=(--init-ckpt "$INIT_CKPT")
fi

echo "[DQPC-B4] Train PU-Dense geometry"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  DQPC_ROOT=$DQPC_ROOT"
echo "  SPLIT=$SPLIT"
echo "  INPUT_ROOT=$INPUT_ROOT"
echo "  CG_GLOB=$CG_GLOB"
echo "  HE_GLOB=${HE_GLOB:-<auto from CG/HE token>}"
echo "  VOXEL_SIZE=$VOXEL_SIZE"
echo "  CROP_SIZE=$CROP_SIZE"
echo "  MAX_CG_POINTS=$MAX_CG_POINTS"
echo "  MAX_HE_POINTS=$MAX_HE_POINTS"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  NUM_WORKERS=$NUM_WORKERS"
echo "  STEPS=$STEPS"
echo "  LR=$LR"
echo "  SAVE_EVERY=$SAVE_EVERY"
echo "  OUT_DIR=$OUT_DIR"
echo "  INIT_CKPT=${INIT_CKPT:-<none>}"
echo "  SEED=$SEED"

"$PYTHON_BIN" baselines/dqpc_b4/train_pudense_dqpc.py "${ARGS[@]}"
