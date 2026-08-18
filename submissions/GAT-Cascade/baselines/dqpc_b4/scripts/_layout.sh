#!/usr/bin/env bash

# Shared DQPC layout defaults. Source this file after ROOT_DIR/DQPC_ROOT/SPLIT
# are set. The official release has appeared both as split/<sequence>/CG/... and
# as <sequence>/consumer-grade_capture_system/CG_aligned/15fps/...

dqpc_default_input_root() {
  if [[ -d "$DQPC_ROOT/$SPLIT" ]]; then
    printf "%s\n" "$DQPC_ROOT/$SPLIT"
  else
    printf "%s\n" "$DQPC_ROOT"
  fi
}

dqpc_default_cg_glob() {
  local input_root="$1"
  local candidates=(
    "$input_root/*/CG_aligned/15fps/*.ply"
    "$input_root/*/consumer-grade_capture_system/CG_aligned/15fps/*.ply"
    "$input_root/*/CGv2/15fps/*.ply"
    "$input_root/*/CGv2_15/*.ply"
    "$input_root/*/consumer-grade_capture_system/CGv2/15fps/*.ply"
    "$input_root/*/CG/15fps/*.ply"
    "$input_root/*/consumer-grade_capture_system/CG/15fps/*.ply"
  )
  local pattern
  for pattern in "${candidates[@]}"; do
    if compgen -G "$pattern" >/dev/null; then
      printf "%s\n" "$pattern"
      return 0
    fi
  done
  printf "%s\n" "${candidates[0]}"
}
