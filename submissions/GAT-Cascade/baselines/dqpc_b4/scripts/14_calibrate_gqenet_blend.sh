#!/usr/bin/env bash
set -euo pipefail

# Calibrate validation-set Y/U/V blend weights, or apply a saved blend config.
#
# Calibration example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   export BASE_ROOT=outputs/dqpc_b4/valid_colored
#   export CANDIDATE_ROOT=outputs/dqpc_b4/valid_gqenet_raw
#   export OUT_ROOT=outputs/dqpc_b4/valid_final
#   bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
#
# Test-set application example:
#   export MODE=apply
#   export SPLIT=test
#   export BASE_ROOT=outputs/dqpc_b4/test_colored
#   export CANDIDATE_ROOT=outputs/dqpc_b4/test_gqenet_raw
#   export CONFIG_IN=outputs/dqpc_b4/eval/valid_gqenet_blend.json
#   export OUT_ROOT=outputs/dqpc_b4/test_final
#   bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-calibrate}"
SPLIT="${SPLIT:-valid}"
BASE_ROOT="${BASE_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/dqpc_b4/${SPLIT}_gqenet_raw}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_final}"
CONFIG_OUT="${CONFIG_OUT:-outputs/dqpc_b4/eval/${SPLIT}_gqenet_blend.json}"
CONFIG_IN="${CONFIG_IN:-}"
STEPS="${STEPS:-101}"
XYZ_ATOL="${XYZ_ATOL:-0.0001}"
LIMIT="${LIMIT:-0}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_gqenet_blend.jsonl}"
MIN_PSNR_Y="${MIN_PSNR_Y:-0}"
MIN_PSNR_U="${MIN_PSNR_U:-0}"
MIN_PSNR_V="${MIN_PSNR_V:-0}"
MIN_GAIN_Y="${MIN_GAIN_Y:-0}"
MIN_GAIN_U="${MIN_GAIN_U:-0}"
MIN_GAIN_V="${MIN_GAIN_V:-0}"
REQUIRE_QUALITY_GATE="${REQUIRE_QUALITY_GATE:-1}"
REQUIRE_COLOR_DRIFT_GUARD="${REQUIRE_COLOR_DRIFT_GUARD:-1}"

source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
BASE_GLOB="${BASE_GLOB:-$(dqpc_default_cg_glob "$BASE_ROOT")}"

ARGS=(
  --base-root "$BASE_ROOT"
  --base-glob "$BASE_GLOB"
  --candidate-root "$CANDIDATE_ROOT"
  --out-root "$OUT_ROOT"
  --runtime-log "$RUNTIME_LOG"
  --steps "$STEPS"
  --xyz-atol "$XYZ_ATOL"
  --min-psnr-y "$MIN_PSNR_Y"
  --min-psnr-u "$MIN_PSNR_U"
  --min-psnr-v "$MIN_PSNR_V"
  --min-gain-y "$MIN_GAIN_Y"
  --min-gain-u "$MIN_GAIN_U"
  --min-gain-v "$MIN_GAIN_V"
  --limit "$LIMIT"
)
if [[ "$REQUIRE_QUALITY_GATE" == "1" ]]; then
  ARGS+=(--require-quality-gate)
fi
if [[ "$REQUIRE_COLOR_DRIFT_GUARD" == "1" ]]; then
  ARGS+=(--require-color-drift-guard)
fi

if [[ "$MODE" == "calibrate" ]]; then
  : "${DQPC_ROOT:?Set DQPC_ROOT for validation-set calibration}"
  GT_ROOT="${GT_ROOT:-$(dqpc_default_input_root)}"
  ARGS+=(--gt-root "$GT_ROOT" --config-out "$CONFIG_OUT")
elif [[ "$MODE" == "apply" ]]; then
  : "${CONFIG_IN:?Set CONFIG_IN to a validation blend JSON for apply mode}"
  ARGS+=(--config-in "$CONFIG_IN")
else
  echo "MODE must be calibrate or apply" >&2
  exit 2
fi

echo "[DQPC-B4] GQE-Net color blend"
echo "  MODE=$MODE"
echo "  BASE_GLOB=$BASE_GLOB"
echo "  CANDIDATE_ROOT=$CANDIDATE_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  CONFIG=${CONFIG_IN:-$CONFIG_OUT}"
echo "  STEPS=$STEPS"
echo "  MIN_PSNR_Y/U/V=$MIN_PSNR_Y/$MIN_PSNR_U/$MIN_PSNR_V"
echo "  MIN_GAIN_Y/U/V=$MIN_GAIN_Y/$MIN_GAIN_U/$MIN_GAIN_V"
echo "  REQUIRE_QUALITY_GATE=$REQUIRE_QUALITY_GATE"
echo "  REQUIRE_COLOR_DRIFT_GUARD=$REQUIRE_COLOR_DRIFT_GUARD"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/calibrate_gqenet_blend.py "${ARGS[@]}"
