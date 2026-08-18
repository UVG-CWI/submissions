#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the UVG-CWI-DQPC dataset root}"
: "${PUDENSE_CKPT:?Set PUDENSE_CKPT to the PU-Dense geometry checkpoint}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-test}"
APPLY_GQENET="${APPLY_GQENET:-auto}"
APPLY_TEMPORAL="${APPLY_TEMPORAL:-1}"
PACKAGE_MODE="${PACKAGE_MODE:-ply-only}"
VALIDATE_RGB="${VALIDATE_RGB:-1}"

export PYTHON_BIN
export DQPC_ROOT
export SPLIT
export PACKAGE_MODE
export VALIDATE_RGB

echo "[GAT-Cascade] Checking environment and dataset layout"
REQUIRE_GT="${REQUIRE_GT:-0}" bash baselines/dqpc_b4/scripts/00_check_env_and_data.sh

echo "[GAT-Cascade] Stage 1: geometry enhancement and color initialization"
bash baselines/dqpc_b4/scripts/02_infer_geometry_and_recolor.sh

FINAL_ROOT="${COLORED_OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_colored}"

if [[ "$APPLY_GQENET" == "auto" ]]; then
  if [[ -n "${MODEL_Y:-}" && -n "${MODEL_U:-}" && -n "${MODEL_V:-}" ]]; then
    APPLY_GQENET=1
  else
    APPLY_GQENET=0
  fi
fi

if [[ "$APPLY_GQENET" == "1" ]]; then
  : "${MODEL_Y:?Set MODEL_Y or set APPLY_GQENET=0}"
  : "${MODEL_U:?Set MODEL_U or set APPLY_GQENET=0}"
  : "${MODEL_V:?Set MODEL_V or set APPLY_GQENET=0}"

  echo "[GAT-Cascade] Stage 2: GQE-Net Y/U/V color refinement"
  INPUT_ROOT="$FINAL_ROOT" OUT_ROOT="outputs/dqpc_b4/${SPLIT}_gqenet_raw" \
    bash baselines/dqpc_b4/scripts/04_infer_gqenet_dqpc.sh

  if [[ -n "${CONFIG_IN:-}" ]]; then
    echo "[GAT-Cascade] Stage 3: apply validation-calibrated color blend"
    MODE=apply \
      BASE_ROOT="$FINAL_ROOT" \
      CANDIDATE_ROOT="outputs/dqpc_b4/${SPLIT}_gqenet_raw" \
      OUT_ROOT="outputs/dqpc_b4/${SPLIT}_final" \
      bash baselines/dqpc_b4/scripts/14_calibrate_gqenet_blend.sh
    FINAL_ROOT="outputs/dqpc_b4/${SPLIT}_final"
  else
    FINAL_ROOT="outputs/dqpc_b4/${SPLIT}_gqenet_raw"
  fi
else
  echo "[GAT-Cascade] Stage 2 skipped: GQE-Net checkpoints not provided"
fi

if [[ "$APPLY_TEMPORAL" == "1" ]]; then
  echo "[GAT-Cascade] Stage 4: temporal color smoothing"
  INPUT_ROOT="$FINAL_ROOT" OUT_ROOT="outputs/dqpc_b4/${SPLIT}_temporal" \
    bash baselines/dqpc_b4/scripts/08_temporal_smooth.sh
  FINAL_ROOT="outputs/dqpc_b4/${SPLIT}_temporal"
fi

echo "[GAT-Cascade] Packaging submission archive"
INPUT_ROOT="$FINAL_ROOT" \
  OUT_DIR="outputs/dqpc_b4/submission_${SPLIT}" \
  ZIP_PATH="outputs/dqpc_b4/submission_${SPLIT}.zip" \
  METHOD_NAME="GAT-Cascade" \
  bash baselines/dqpc_b4/scripts/07_make_submission.sh

echo "[GAT-Cascade] Done: outputs/dqpc_b4/submission_${SPLIT}.zip"
