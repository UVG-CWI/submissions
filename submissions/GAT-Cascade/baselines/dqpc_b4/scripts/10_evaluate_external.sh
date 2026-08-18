#!/usr/bin/env bash
set -euo pipefail

# Optional wrappers for external metrics such as MPEG pc_error or PCQM.
#
# Example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export PRED_ROOT=outputs/dqpc_b4/valid_final
#   export RUN_NAME=valid_final
#   export PC_ERROR_BIN=/path/to/pc_error_d
#   bash baselines/dqpc_b4/scripts/10_evaluate_external.sh

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
OUT_JSONL="${OUT_JSONL:-outputs/dqpc_b4/eval/${RUN_NAME}_external_eval.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/dqpc_b4/eval/${RUN_NAME}_external_eval_summary.json}"
PC_ERROR_BIN="${PC_ERROR_BIN:-}"
PC_ERROR_RES="${PC_ERROR_RES:-1024}"
PCQM_BIN="${PCQM_BIN:-}"
if [[ -z "$PCQM_BIN" && -x "$ROOT_DIR/external/PCQM/build/PCQM" ]]; then
  PCQM_BIN="$ROOT_DIR/external/PCQM/build/PCQM"
fi
PCQM_WORK_DIR="${PCQM_WORK_DIR:-outputs/dqpc_b4/eval/pcqm_work}"
LIMIT="${LIMIT:-0}"

ARGS=(
  --pred-root "$PRED_ROOT"
  --pred-glob "$PRED_GLOB"
  --gt-root "$GT_ROOT"
  --out-jsonl "$OUT_JSONL"
  --summary-json "$SUMMARY_JSON"
  --pc-error-res "$PC_ERROR_RES"
  --limit "$LIMIT"
)

if [[ -n "$PC_ERROR_BIN" ]]; then
  ARGS+=(--pc-error-bin "$PC_ERROR_BIN")
fi
if [[ -n "$PCQM_BIN" ]]; then
  ARGS+=(--pcqm-bin "$PCQM_BIN")
  ARGS+=(--pcqm-work-dir "$PCQM_WORK_DIR")
fi

echo "[DQPC-B4] External evaluation"
echo "  RUN_NAME=$RUN_NAME"
echo "  PRED_GLOB=$PRED_GLOB"
echo "  GT_ROOT=$GT_ROOT"
echo "  PC_ERROR_BIN=${PC_ERROR_BIN:-<disabled>}"
echo "  PCQM_BIN=${PCQM_BIN:-<disabled>}"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/evaluate_external.py "${ARGS[@]}"
