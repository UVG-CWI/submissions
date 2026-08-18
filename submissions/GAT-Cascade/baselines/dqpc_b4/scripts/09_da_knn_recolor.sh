#!/usr/bin/env bash
set -euo pipefail

# Run DA-KNN recoloring as a stronger replacement for Gaussian recoloring.
#
# Example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export TARGET_ROOT=outputs/dqpc_b4/valid_geometry
#   bash baselines/dqpc_b4/scripts/09_da_knn_recolor.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
SOURCE_ROOT="${SOURCE_ROOT:-$(dqpc_default_input_root)}"
SOURCE_GLOB="${SOURCE_GLOB:-$(dqpc_default_cg_glob "$SOURCE_ROOT")}"
TARGET_ROOT="${TARGET_ROOT:-outputs/dqpc_b4/${SPLIT}_geometry}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_daknn_colored}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_daknn_recolor.jsonl}"
K="${K:-8}"
SIGMA="${SIGMA:-}"
NORMAL_K="${NORMAL_K:-16}"
NORMAL_GAMMA="${NORMAL_GAMMA:-2.0}"
COLOR_GAMMA="${COLOR_GAMMA:-0.5}"
COPY_DISTANCE="${COPY_DISTANCE:-0.5}"
DISTANCE_UNIT="${DISTANCE_UNIT:-mm}"
NORMAL_CHUNK_SIZE="${NORMAL_CHUNK_SIZE:-200000}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-250000}"
COLOR_SCALE_SAMPLE_SIZE="${COLOR_SCALE_SAMPLE_SIZE:-200000}"
COLOR_SCALE_MODE="${COLOR_SCALE_MODE:-exact}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ARGS=(
  --source-glob "$SOURCE_GLOB"
  --source-root "$SOURCE_ROOT"
  --target-root "$TARGET_ROOT"
  --out-root "$OUT_ROOT"
  --runtime-log "$RUNTIME_LOG"
  --k "$K"
  --normal-k "$NORMAL_K"
  --normal-gamma "$NORMAL_GAMMA"
  --color-gamma "$COLOR_GAMMA"
  --copy-distance "$COPY_DISTANCE"
  --distance-unit "$DISTANCE_UNIT"
  --normal-chunk-size "$NORMAL_CHUNK_SIZE"
  --query-chunk-size "$QUERY_CHUNK_SIZE"
  --color-scale-sample-size "$COLOR_SCALE_SAMPLE_SIZE"
  --color-scale-mode "$COLOR_SCALE_MODE"
  --limit "$LIMIT"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

if [[ -n "$SIGMA" ]]; then
  ARGS+=(--sigma "$SIGMA")
fi

echo "[DQPC-B4] DA-KNN recolor"
echo "  SOURCE_GLOB=$SOURCE_GLOB"
echo "  TARGET_ROOT=$TARGET_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  K=$K"
echo "  NORMAL_K=$NORMAL_K"
echo "  COPY_DISTANCE=$COPY_DISTANCE"
echo "  DISTANCE_UNIT=$DISTANCE_UNIT"
echo "  COLOR_SCALE_MODE=$COLOR_SCALE_MODE"
echo "  SKIP_EXISTING=$SKIP_EXISTING"

"$PYTHON_BIN" baselines/dqpc_b4/da_knn_recolor.py "${ARGS[@]}"
