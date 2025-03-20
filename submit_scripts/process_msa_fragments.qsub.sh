#!/bin/bash

# Process MSA fragments and cluster with MMSEQS
# Example usage:
# ./process_msa_fragments.sh <input_pattern> <output_dir> [num_tasks]

# Check if required parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_pattern> <output_dir> [num_tasks]"
    echo "  <input_pattern>: Pattern to match parquet input files (e.g., '../data/openfold/uniclust30_filtered_parquet/*.parquet')"
    echo "  <output_dir>: Directory to save output files"
    echo "  [num_tasks]: Optional number of tasks for parallel processing"
    exit 1
fi

INPUT_PATTERN=$1
OUTPUT_DIR=$2
NUM_TASKS=${3:-1}  # Default to 1 task if not provided

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting MSA fragment processing..."
echo "Input pattern: $INPUT_PATTERN"
echo "Output directory: $OUTPUT_DIR"
echo "Number of tasks: $NUM_TASKS"

# If only one task, run directly
if [ "$NUM_TASKS" -eq 1 ]; then
    python data_creation_scripts/process_msa_fragments.py \
        --input_pattern "$INPUT_PATTERN" \
        --output_dir "$OUTPUT_DIR" \
        --threads 20
else
    # Otherwise, run multiple tasks in parallel
    for ((i=0; i<$NUM_TASKS; i++)); do
        echo "Starting task $i/$NUM_TASKS..."
        python data_creation_scripts/process_msa_fragments.py \
            --input_pattern "$INPUT_PATTERN" \
            --output_dir "$OUTPUT_DIR" \
            --threads 20 \
            --task_index $i \
            --num_tasks $NUM_TASKS &
        
        # Sleep a bit to avoid overwhelming the system
        sleep 2
    done
    
    # Wait for all background tasks to complete
    wait
    echo "All tasks completed."
fi

echo "Processing complete. Check $OUTPUT_DIR for results." 