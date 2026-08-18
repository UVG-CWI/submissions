#!/usr/bin/env bash
set -euo pipefail

# Official-style geometry evaluation through external/UVG-CWI-Metric.
# This is the preferred geometry summary for validation boards. PCQM remains
# available through 10_evaluate_external.sh as a supplementary perceptual metric.
#
# Example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export PRED_ROOT=outputs/dqpc_b4/valid_final
#   export RUN_NAME=valid_final
#   bash baselines/dqpc_b4/scripts/13_evaluate_uvg_metric.sh

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
METRIC_ROOT="${METRIC_ROOT:-$ROOT_DIR/external/UVG-CWI-Metric}"
OUT_JSONL="${OUT_JSONL:-outputs/dqpc_b4/eval/${RUN_NAME}_uvg_metric_eval.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/dqpc_b4/eval/${RUN_NAME}_uvg_metric_eval_summary.json}"
FSCORE_THRESHOLDS="${FSCORE_THRESHOLDS:-5,10,20,30}"
THRESHOLD_UNIT="${THRESHOLD_UNIT:-mm}"
LIMIT="${LIMIT:-0}"

echo "[DQPC-B4] UVG-CWI Metric evaluation"
echo "  RUN_NAME=$RUN_NAME"
echo "  PRED_ROOT=$PRED_ROOT"
echo "  PRED_GLOB=$PRED_GLOB"
echo "  GT_ROOT=$GT_ROOT"
echo "  METRIC_ROOT=$METRIC_ROOT"
echo "  FSCORE_THRESHOLDS=$FSCORE_THRESHOLDS"
echo "  THRESHOLD_UNIT=$THRESHOLD_UNIT"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/evaluate_uvg_metric.py \
  --pred-root "$PRED_ROOT" \
  --pred-glob "$PRED_GLOB" \
  --gt-root "$GT_ROOT" \
  --metric-root "$METRIC_ROOT" \
  --out-jsonl "$OUT_JSONL" \
  --summary-json "$SUMMARY_JSON" \
  --fscore-thresholds "$FSCORE_THRESHOLDS" \
  --threshold-unit "$THRESHOLD_UNIT" \
  --limit "$LIMIT"
