#!/usr/bin/env bash
set -euo pipefail

# Transfer RGB from a strong color result (for example V3) to current geometry.
#
# Example:
#   export DONOR_ROOT=/path/to/V3/valid_final
#   export TARGET_ROOT=outputs/dqpc_b4/valid_geometry
#   export OUT_ROOT=outputs/dqpc_b4/valid_v3_color_current_geometry
#   bash baselines/dqpc_b4/scripts/16_transfer_color_donor.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DONOR_ROOT:?Set DONOR_ROOT to a colored PLY tree such as the V3 result}"
: "${TARGET_ROOT:?Set TARGET_ROOT to the current geometry PLY tree}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid}"
OUT_ROOT="${OUT_ROOT:-outputs/dqpc_b4/${SPLIT}_donor_colored}"
RUNTIME_LOG="${RUNTIME_LOG:-outputs/dqpc_b4/runtime/${SPLIT}_donor_recolor.jsonl}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
DONOR_GLOB="${DONOR_GLOB:-$(dqpc_default_cg_glob "$DONOR_ROOT")}"

ARGS=(
  --cg-glob "$DONOR_GLOB"
  --input-root "$DONOR_ROOT"
  --geometry-root "$TARGET_ROOT"
  --colored-out-root "$OUT_ROOT"
  --runtime-log "$RUNTIME_LOG"
  --k 1
  --copy-distance -1
  --limit "$LIMIT"
)
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

echo "[DQPC-B4] Transfer donor colors to current geometry"
echo "  DONOR_GLOB=$DONOR_GLOB"
echo "  TARGET_ROOT=$TARGET_ROOT"
echo "  OUT_ROOT=$OUT_ROOT"
echo "  mapping=nearest-neighbor RGB"

"$PYTHON_BIN" baselines/dqpc_b4/recolor_sequence.py "${ARGS[@]}"
