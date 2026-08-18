#!/usr/bin/env bash
set -euo pipefail

# Check Python dependencies and DQPC directory layout.
#
# Example:
#   export PYTHON_BIN=/path/to/conda/env/bin/python
#   export DQPC_ROOT=/data/UVG-CWI-DQPC
#   export SPLIT=train
#   bash baselines/dqpc_b4/scripts/00_check_env_and_data.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

: "${DQPC_ROOT:?Set DQPC_ROOT to the dataset root}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-train}"
source "$ROOT_DIR/baselines/dqpc_b4/scripts/_layout.sh"
INPUT_ROOT="${INPUT_ROOT:-$(dqpc_default_input_root)}"
CG_GLOB="${CG_GLOB:-$(dqpc_default_cg_glob "$INPUT_ROOT")}"
HE_GLOB="${HE_GLOB:-}"
if [[ "$SPLIT" == "test" ]]; then
  DEFAULT_REQUIRE_GT=0
else
  DEFAULT_REQUIRE_GT=1
fi
REQUIRE_GT="${REQUIRE_GT:-$DEFAULT_REQUIRE_GT}"
VOXEL_SIZE="${VOXEL_SIZE:-auto}"
OUT_DIR="${OUT_DIR:-outputs/dqpc_b4/checks}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" baselines/dqpc_b4/check_env.py \
  --json-out "$OUT_DIR/env.json"

DATA_ARGS=(
  --dqpc-root "$DQPC_ROOT"
  --split "$SPLIT"
  --cg-glob "$CG_GLOB"
  --voxel-size "$VOXEL_SIZE"
  --json-out "$OUT_DIR/dataset_${SPLIT}.json"
)

if [[ -n "$HE_GLOB" ]]; then
  DATA_ARGS+=(--he-glob "$HE_GLOB")
fi

if [[ "$REQUIRE_GT" == "1" ]]; then
  DATA_ARGS+=(--require-gt)
fi

"$PYTHON_BIN" baselines/dqpc_b4/check_dataset_layout.py "${DATA_ARGS[@]}"
