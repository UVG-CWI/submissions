#!/usr/bin/env bash
set -euo pipefail

# Basic local evaluation against HE PLY.
#
# Example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export PRED_ROOT=outputs/dqpc_b4/valid_final
#   export RUN_NAME=valid_final
#   bash baselines/dqpc_b4/scripts/06_evaluate_basic.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
RUN_NAME="${RUN_NAME:-${SPLIT}_final}"
PRED_ROOT="${PRED_ROOT:-outputs/dqpc_b4/${SPLIT}_final}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
PRED_GLOB="${PRED_GLOB:-$(dqpc_default_cg_glob "$PRED_ROOT")}"
GT_ROOT="${GT_ROOT:-$(dqpc_default_input_root)}"
OUT_DIR="${OUT_DIR:-outputs/dqpc_b4/eval}"
LIMIT="${LIMIT:-0}"
FSCORE_THRESHOLDS="${FSCORE_THRESHOLDS:-5,10,20,30}"
THRESHOLD_UNIT="${THRESHOLD_UNIT:-mm}"

mkdir -p "$OUT_DIR"

echo "[DQPC-B4] Basic evaluation"
echo "  RUN_NAME=$RUN_NAME"
echo "  PRED_ROOT=$PRED_ROOT"
echo "  PRED_GLOB=$PRED_GLOB"
echo "  GT_ROOT=$GT_ROOT"
echo "  FSCORE_THRESHOLDS=$FSCORE_THRESHOLDS"
echo "  THRESHOLD_UNIT=$THRESHOLD_UNIT"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/evaluate_basic.py \
  --pred-root "$PRED_ROOT" \
  --pred-glob "$PRED_GLOB" \
  --gt-root "$GT_ROOT" \
  --out-jsonl "$OUT_DIR/${RUN_NAME}_basic_eval.jsonl" \
  --summary-json "$OUT_DIR/${RUN_NAME}_basic_eval_summary.json" \
  --fscore-thresholds "$FSCORE_THRESHOLDS" \
  --threshold-unit "$THRESHOLD_UNIT" \
  --limit "$LIMIT"
