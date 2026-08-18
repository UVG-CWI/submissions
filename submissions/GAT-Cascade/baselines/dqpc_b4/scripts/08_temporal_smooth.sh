#!/usr/bin/env bash
set -euo pipefail

# Apply lightweight temporal color smoothing to a colored output tree.
#
# Example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export SPLIT=valid
#   export INPUT_ROOT=outputs/dqpc_b4/valid_final
#   bash baselines/dqpc_b4/scripts/08_temporal_smooth.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
INPUT_ROOT="${INPUT_ROOT:-outputs/dqpc_b4/${SPLIT}_final}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_GLOB="${INPUT_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_temporal_raw}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_temporal.jsonl}"
ALPHA="${ALPHA:-0.75}"
MAX_DISTANCE="${MAX_DISTANCE:-30}"
DISTANCE_UNIT="${DISTANCE_UNIT:-mm}"
MAX_COLOR_DELTA="${MAX_COLOR_DELTA:-35}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-250000}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

echo "[DQPC-B4] Temporal smoothing"
echo "  INPUT_GLOB=$INPUT_GLOB"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  ALPHA=$ALPHA"
echo "  MAX_DISTANCE=$MAX_DISTANCE"
echo "  DISTANCE_UNIT=$DISTANCE_UNIT"
echo "  MAX_COLOR_DELTA=$MAX_COLOR_DELTA"
echo "  SKIP_EXISTING=$SKIP_EXISTING"

ARGS=(
  --input-root "$INPUT_ROOT" \
  --input-glob "$INPUT_GLOB" \
  --out-root "$OUT_ROOT" \
  --runtime-log "$RUNTIME_LOG" \
  --alpha "$ALPHA" \
  --max-distance "$MAX_DISTANCE" \
  --distance-unit "$DISTANCE_UNIT" \
  --max-color-delta "$MAX_COLOR_DELTA" \
  --query-chunk-size "$QUERY_CHUNK_SIZE" \
  --limit "$LIMIT"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

"$PYTHON_BIN" baselines/dqpc_b4/temporal_smooth.py "${ARGS[@]}"
