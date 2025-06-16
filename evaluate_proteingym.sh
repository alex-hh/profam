#!/usr/bin/env bash
set -euo pipefail

# Usage: ./evaluate_proteingym.sh <remote_list.txt> [<local_base_dir>]
# remote_list.txt should contain lines of the form user@host:/path/to/logs/.../runs
# Example: w-ssh-judew@ssh.host:/home/jovyan/workspace/profam/logs/openfold_fs50_ur90_memmap/runs

PROJECT_ROOT="/mnt/disk2/cath_plm/profam"
REMOTE_LIST_FILE="${1:-remote_list.txt}"
LOCAL_BASE_DIR="${2:-$PROJECT_ROOT/logs/eval_runs}"

mkdir -p "$LOCAL_BASE_DIR"

while IFS= read -r REMOTE_LOGS; do
  # Split host and directory
  REMOTE_HOST="${REMOTE_LOGS%%:*}"
  REMOTE_DIR="${REMOTE_LOGS#*:}"
  echo "Checking for latest checkpoint on $REMOTE_HOST:$REMOTE_DIR"
  # Find latest last.ckpt on remote
  LATEST_CKPT=$(ssh "$REMOTE_HOST" "ls -td $REMOTE_DIR/*/checkpoints/last.ckpt 2>/dev/null | head -1")
  if [ -z "$LATEST_CKPT" ]; then
    echo "  No checkpoint found, skipping."
    continue
  fi
  # Derive run identifier
  RUN_BASE=$(dirname "$(dirname "$LATEST_CKPT")")
  RUN_NAME=$(basename "$RUN_BASE")
  LOCAL_RUN_DIR="$LOCAL_BASE_DIR/$RUN_NAME"
  # Skip if already processed
  if [ -f "$LOCAL_RUN_DIR/checkpoints/last.ckpt" ]; then
    echo "  Already processed $RUN_NAME, skipping."
    continue
  fi
  echo "  New checkpoint: $LATEST_CKPT"
  # Prepare local directories
  mkdir -p "$LOCAL_RUN_DIR/checkpoints" "$LOCAL_RUN_DIR/.hydra"
  # Sync checkpoint and config
  rsync -av "$REMOTE_HOST:$LATEST_CKPT" "$LOCAL_RUN_DIR/checkpoints/"
  rsync -av "$REMOTE_HOST:$RUN_BASE/.hydra/config.yaml" "$LOCAL_RUN_DIR/.hydra/"

  # Run evaluation on ProteinGym only
  echo "  Evaluating run $RUN_NAME on ProteinGym..."
  pushd "$LOCAL_RUN_DIR" >/dev/null
  python "$PROJECT_ROOT/src/train.py" \
    --config-dir ./.hydra --config-name config.yaml \
    experiment=eval_full_gym_only train=false test=true \
    ckpt_path=./checkpoints/last.ckpt \
    hydra.run.dir="$LOCAL_RUN_DIR/eval_results"
  popd >/dev/null

done < "$REMOTE_LIST_FILE"
