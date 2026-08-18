#!/usr/bin/env bash
set -euo pipefail

# Copy CG PLY frames to an identity baseline output tree.
#
# Example:
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=valid
#   bash baselines/dqpc_b4/scripts/05_identity_baseline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_ROOT="${INPUT_ROOT:-$(dqpc_default_input_root)}"
CG_GLOB="${CG_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_identity}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_identity.jsonl}"
LIMIT="${LIMIT:-0}"

echo "[DQPC-B4] Identity baseline"
echo "  INPUT_ROOT=$INPUT_ROOT"
echo "  CG_GLOB=$CG_GLOB"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  LIMIT=$LIMIT"

"$PYTHON_BIN" baselines/dqpc_b4/identity_baseline.py \
  --input-root "$INPUT_ROOT" \
  --cg-glob "$CG_GLOB" \
  --out-root "$OUT_ROOT" \
  --runtime-log "$RUNTIME_LOG" \
  --limit "$LIMIT"
