#!/bin/bash

# Configuration
EVAL_EXE="./evaluate_points"
BASE_DIR="/xrspace2/UVG-CWI-DQPC"
MAX_JOBS=2  # Set the maximum number of parallel instances

# List of sequence names to process
SEQUENCES=(
    "BlueSpeech"
    "BouncingBlue"
    "FitFluencer"
    "Mannequin"
    "PinkNoir"
    "TrumanShow"
    "VirtualLife"
    "BlueVolley"
    "GoodVision"
    "OrangeKettlebell"
    "TicTacToe"
    "VictoryHeart"
)

# Check if executable exists
if [ ! -f "$EVAL_EXE" ]; then
    echo "Error: Executable '$EVAL_EXE' not found."
    exit 1
fi

process_sequence() {
    local seq=$1
    local log_file="${seq}_aligned_output.log"

    # Construct paths
    local CG_PATH="$BASE_DIR/$seq/consumer-grade_capture_system/CG_aligned/15fps"
    local HE_PATH="$BASE_DIR/$seq/high-end_capture_system/HE/15fps"

    # specific check to ensure source directories exist before running
    if [ ! -d "$CG_PATH" ] || [ ! -d "$HE_PATH" ]; then
        echo "  [Skipping] Directory not found for $seq"
        return
    fi

    # Run the command
    if $EVAL_EXE --mode baseline \
        --sequence_name "$seq" \
        --cg_dir "$CG_PATH" \
        --he_dir "$HE_PATH" > "$log_file" 2>&1; then
        echo "  [Success] $seq completed."
    else
        echo "  [Error] $seq failed. See $log_file"
    fi
}

echo "Starting parallel execution (Max jobs: $MAX_JOBS)..."

for seq in "${SEQUENCES[@]}"; do
    # Wait if we have reached the maximum number of background jobs
    while [ "$(jobs -r -p | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done

    process_sequence "$seq" &
done

wait
echo "All evaluations finished."
