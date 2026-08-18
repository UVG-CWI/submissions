#!/usr/bin/env bash
set -euo pipefail

# Six-view projection SSIM and optional LPIPS.
#
# Example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export PRED_ROOT=outputs/dqpc_b4/valid_final
#   export RUN_NAME=valid_final
#   bash baselines/dqpc_b4/scripts/11_evaluate_projection.sh
#
# Optional LPIPS, if the lpips Python package and weights are available:
#   export COMPUTE_LPIPS=1
#   bash baselines/dqpc_b4/scripts/11_evaluate_projection.sh

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
IMAGE_SIZE="${IMAGE_SIZE:-512}"
LIMIT="${LIMIT:-0}"
COMPUTE_LPIPS="${COMPUTE_LPIPS:-0}"

mkdir -p "$OUT_DIR"

ARGS=(
  --pred-root "$PRED_ROOT"
  --pred-glob "$PRED_GLOB"
  --gt-root "$GT_ROOT"
  --out-jsonl "$OUT_DIR/${RUN_NAME}_projection_eval.jsonl"
  --summary-json "$OUT_DIR/${RUN_NAME}_projection_eval_summary.json"
  --image-size "$IMAGE_SIZE"
  --limit "$LIMIT"
)

if [[ "$COMPUTE_LPIPS" == "1" ]]; then
  ARGS+=(--lpips)
fi

echo "[DQPC-B4] Projection evaluation"
echo "  RUN_NAME=$RUN_NAME"
echo "  PRED_GLOB=$PRED_GLOB"
echo "  GT_ROOT=$GT_ROOT"
echo "  IMAGE_SIZE=$IMAGE_SIZE"
echo "  COMPUTE_LPIPS=$COMPUTE_LPIPS"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/evaluate_projection.py "${ARGS[@]}"
