#!/usr/bin/env bash
set -euo pipefail

# Run PU-Dense geometry inference, then Gaussian KNN recoloring.
#
# Run from any directory; the script will cd to the project root.
#
# Required:
#   DQPC_ROOT: dataset root.
#   PUDENSE_CKPT: PU-Dense checkpoint produced by 01_train_pudense_dqpc.sh.
#
# Default expected input layout, in priority order:
#   $DQPC_ROOT/valid/<sequence>/consumer-grade_capture_system/CG_aligned/15fps/*.ply
#   $DQPC_ROOT/valid/<sequence>/CG_aligned/15fps/*.ply
#   $DQPC_ROOT/valid/<sequence>/CGv2_15/*.ply
#
# One-frame smoke test example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
#   export SPLIT=valid
#   export LIMIT=1
#   bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
#
# Full validation example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
#   export GEOMETRY_OUT_ROOT=outputs/dqpc_b4/valid_geometry
#   export COLORED_OUT_ROOT=outputs/dqpc_b4/valid_colored
#   export RUNTIME_DIR=outputs/dqpc_b4/runtime
#   bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
#
# Test split example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=test
#   export PUDENSE_CKPT=outputs/dqpc_b4/pudense_ckpts/iter2000.pth
#   bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh
#
# Override these if the dataset layout is different:
#   export INPUT_ROOT=/data/DQPC/valid
#   export CG_GLOB="$INPUT_ROOT/*/consumer-grade_capture_system/CG_aligned/15fps/*.ply"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root, for example /data/UVG-CWI-DQPC}"
: "${PUDENSE_CKPT:?Set PUDENSE_CKPT to a PU-Dense checkpoint, for example outputs/dqpc_b4/pudense_ckpts/iter2000.pth}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_ROOT="${INPUT_ROOT:-$(dqpc_default_input_root)}"
CG_GLOB="${CG_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
VOXEL_SIZE="${VOXEL_SIZE:-auto}"
MAX_POINTS_PER_BLOCK="${MAX_POINTS_PER_BLOCK:-70000}"
UP_RATIO="${UP_RATIO:-4}"
TARGET_POINT_RATIO="${TARGET_POINT_RATIO:-}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-}"
MAX_OUTPUT_POINT_RATIO="${MAX_OUTPUT_POINT_RATIO:-0}"
BLOCK_HALO="${BLOCK_HALO:-32}"
INCLUDE_INPUT="${INCLUDE_INPUT:-1}"
LAST_KERNEL_SIZE="${LAST_KERNEL_SIZE:-5}"
K="${K:-8}"
SIGMA="${SIGMA:-}"
COPY_DISTANCE="${COPY_DISTANCE:-0.5}"
DISTANCE_UNIT="${DISTANCE_UNIT:-mm}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-250000}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
GEOMETRY_OUT_ROOT="${GEOMETRY_OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_geometry}"
COLORED_OUT_ROOT="${COLORED_OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"
RUNTIME_DIR="${RUNTIME_DIR:-outputs/dqpc_b4/runtime}"

mkdir -p "$RUNTIME_DIR"

echo "[DQPC-B4] Infer PU-Dense geometry and recolor"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  DQPC_ROOT=$DQPC_ROOT"
echo "  SPLIT=$SPLIT"
echo "  INPUT_ROOT=$INPUT_ROOT"
echo "  CG_GLOB=$CG_GLOB"
echo "  PUDENSE_CKPT=$PUDENSE_CKPT"
echo "  VOXEL_SIZE=$VOXEL_SIZE"
echo "  MAX_POINTS_PER_BLOCK=$MAX_POINTS_PER_BLOCK"
echo "  UP_RATIO=$UP_RATIO"
echo "  TARGET_POINT_RATIO=${TARGET_POINT_RATIO:-<same as UP_RATIO>}"
echo "  SCORE_THRESHOLD=${SCORE_THRESHOLD:-<top-k>}"
echo "  MAX_OUTPUT_POINT_RATIO=$MAX_OUTPUT_POINT_RATIO"
echo "  BLOCK_HALO=$BLOCK_HALO"
echo "  INCLUDE_INPUT=$INCLUDE_INPUT"
echo "  K=$K"
echo "  SIGMA=${SIGMA:-<auto median kth distance>}"
echo "  COPY_DISTANCE=$COPY_DISTANCE"
echo "  DISTANCE_UNIT=$DISTANCE_UNIT"
echo "  QUERY_CHUNK_SIZE=$QUERY_CHUNK_SIZE"
echo "  LIMIT=$LIMIT"
echo "  SKIP_EXISTING=$SKIP_EXISTING"
echo "  GEOMETRY_OUT_ROOT=$GEOMETRY_OUT_ROOT"
echo "  COLORED_OUT_ROOT=$COLORED_OUT_ROOT"
echo "  RUNTIME_DIR=$RUNTIME_DIR"

INFER_ARGS=(
  --pudense-root external/PointCloudUpsampling \
  --checkpoint "$PUDENSE_CKPT" \
  --cg-glob "$CG_GLOB" \
  --input-root "$INPUT_ROOT" \
  --geometry-out-root "$GEOMETRY_OUT_ROOT" \
  --runtime-log "$RUNTIME_DIR/${SPLIT}_geometry.jsonl" \
  --voxel-size "$VOXEL_SIZE" \
  --max-points-per-block "$MAX_POINTS_PER_BLOCK" \
  --up-ratio "$UP_RATIO" \
  --max-output-point-ratio "$MAX_OUTPUT_POINT_RATIO" \
  --block-halo "$BLOCK_HALO" \
  --last-kernel-size "$LAST_KERNEL_SIZE" \
  --limit "$LIMIT"
)

if [[ "$INCLUDE_INPUT" == "1" ]]; then
  INFER_ARGS+=(--include-input)
fi
if [[ -n "$TARGET_POINT_RATIO" ]]; then
  INFER_ARGS+=(--target-point-ratio "$TARGET_POINT_RATIO")
fi
if [[ -n "$SCORE_THRESHOLD" ]]; then
  INFER_ARGS+=(--score-threshold "$SCORE_THRESHOLD")
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
  INFER_ARGS+=(--skip-existing)
fi

"$PYTHON_BIN" baselines/dqpc_b4/infer_pudense_dqpc.py "${INFER_ARGS[@]}"

RECOLOR_ARGS=(
  --cg-glob "$CG_GLOB"
  --input-root "$INPUT_ROOT"
  --geometry-root "$GEOMETRY_OUT_ROOT"
  --colored-out-root "$COLORED_OUT_ROOT"
  --runtime-log "$RUNTIME_DIR/${SPLIT}_recolor.jsonl"
  --k "$K"
  --copy-distance "$COPY_DISTANCE"
  --distance-unit "$DISTANCE_UNIT"
  --query-chunk-size "$QUERY_CHUNK_SIZE"
  --limit "$LIMIT"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  RECOLOR_ARGS+=(--skip-existing)
fi

if [[ -n "$SIGMA" ]]; then
  RECOLOR_ARGS+=(--sigma "$SIGMA")
fi

"$PYTHON_BIN" baselines/dqpc_b4/recolor_sequence.py "${RECOLOR_ARGS[@]}"
