#!/usr/bin/env bash
set -euo pipefail

# Collect UVG Metric/basic/external/projection summaries into one validation board.
#
# Example:
#   export SPLIT=valid
#   export RUN_NAME=valid_final
#   bash baselines/dqpc_b4/scripts/12_summarize_validation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
RUN_NAME="${RUN_NAME:-${SPLIT}_final}"
EVAL_DIR="${EVAL_DIR:-outputs/dqpc_b4/eval}"
BASIC_SUMMARY="${BASIC_SUMMARY:-$EVAL_DIR/${RUN_NAME}_basic_eval_summary.json}"
UVG_METRIC_SUMMARY="${UVG_METRIC_SUMMARY:-$EVAL_DIR/${RUN_NAME}_uvg_metric_eval_summary.json}"
EXTERNAL_SUMMARY="${EXTERNAL_SUMMARY:-$EVAL_DIR/${RUN_NAME}_external_eval_summary.json}"
PROJECTION_SUMMARY="${PROJECTION_SUMMARY:-$EVAL_DIR/${RUN_NAME}_projection_eval_summary.json}"
RUNTIME_GLOB="${RUNTIME_GLOB:-outputs/dqpc_b4/runtime/${SPLIT}_*.jsonl}"
OUT_JSON="${OUT_JSON:-$EVAL_DIR/${RUN_NAME}_validation_board.json}"
OUT_MD="${OUT_MD:-$EVAL_DIR/${RUN_NAME}_validation_board.md}"

echo "[DQPC-B4] Summarize validation"
echo "  RUN_NAME=$RUN_NAME"
echo "  BASIC_SUMMARY=$BASIC_SUMMARY"
echo "  UVG_METRIC_SUMMARY=$UVG_METRIC_SUMMARY"
echo "  EXTERNAL_SUMMARY=$EXTERNAL_SUMMARY"
echo "  PROJECTION_SUMMARY=$PROJECTION_SUMMARY"
echo "  RUNTIME_GLOB=$RUNTIME_GLOB"
echo "  OUT_JSON=$OUT_JSON"

"$PYTHON_BIN" baselines/dqpc_b4/summarize_validation.py \
  --run-name "$RUN_NAME" \
  --basic-summary "$BASIC_SUMMARY" \
  --uvg-metric-summary "$UVG_METRIC_SUMMARY" \
  --external-summary "$EXTERNAL_SUMMARY" \
  --projection-summary "$PROJECTION_SUMMARY" \
  --runtime-glob "$RUNTIME_GLOB" \
  --out-json "$OUT_JSON" \
  --out-md "$OUT_MD"
