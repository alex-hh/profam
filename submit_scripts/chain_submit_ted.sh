#!/bin/bash
set -euo pipefail

# Submit a chain of dependent training jobs (each limited to 24h) that resume the same run.
# Usage: chain_submit_ted.sh [NUM_JOBS]
#   NUM_JOBS: how many 24h jobs to schedule sequentially (default: 3)
# Optional env:
#   JOB_NAME        - overrides default job name used by sbatch header (default: profam-train)
#   RUN_ID          - set a specific run id; otherwise the first submission will generate one
#   SCRATCH         - if set, used as base for logs/state (matches sbatch script behavior)

NUM_JOBS=${1:-14}
if ! [[ "$NUM_JOBS" =~ ^[0-9]+$ ]] || [ "$NUM_JOBS" -lt 1 ]; then
  echo "NUM_JOBS must be a positive integer" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SBATCH_FILE="$SCRIPT_DIR/slurm_train_ted.sbatch"
if [ ! -f "$SBATCH_FILE" ]; then
  echo "Cannot find sbatch file: $SBATCH_FILE" >&2
  exit 1
fi

JOB_NAME_DEFAULT="profam-train"
JOB_NAME=${JOB_NAME:-$JOB_NAME_DEFAULT}

SCRATCH_BASE=${SCRATCH:-/p/project1/hai_1116/ProFam/logs}
RUN_BASE="$SCRATCH_BASE"
STATE_DIR="$RUN_BASE/state"
mkdir -p "$STATE_DIR"

RUN_ID_FILE="$STATE_DIR/${JOB_NAME}.run_id"
if [ -n "${RUN_ID:-}" ]; then
  echo "$RUN_ID" > "$RUN_ID_FILE"
elif [ -f "$RUN_ID_FILE" ]; then
  RUN_ID=$(cat "$RUN_ID_FILE")
else
  RUN_ID=$(date +"%Y-%m-%d_%H-%M-%S-%N")
  echo "$RUN_ID" > "$RUN_ID_FILE"
fi

echo "Using RUN_ID=$RUN_ID (state: $RUN_ID_FILE)"

submit_with_dep() {
  local dep="$1"
  if [ -n "$dep" ]; then
    sbatch --job-name="$JOB_NAME" --dependency=afterany:"$dep" "$SBATCH_FILE"
  else
    sbatch --job-name="$JOB_NAME" "$SBATCH_FILE"
  fi
}

# Submit first job
SUBMIT_OUT=$(submit_with_dep "")
echo "$SUBMIT_OUT"
JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $4}')
if [ -z "$JOB_ID" ]; then
  echo "Failed to parse first job id" >&2
  exit 1
fi

ALL_JOBS=("$JOB_ID")

# Submit chained follow-ups
for ((i=2; i<=NUM_JOBS; i++)); do
  SUBMIT_OUT=$(submit_with_dep "$JOB_ID")
  echo "$SUBMIT_OUT"
  JOB_ID=$(echo "$SUBMIT_OUT" | awk '{print $4}')
  if [ -z "$JOB_ID" ]; then
    echo "Failed to parse job id at position $i" >&2
    exit 1
  fi
  ALL_JOBS+=("$JOB_ID")
done

echo "Scheduled $NUM_JOBS jobs (sequential). Job IDs: ${ALL_JOBS[*]}"
echo "Hydra/outputs/checkpoints will be under: $RUN_BASE/runs/$RUN_ID"

