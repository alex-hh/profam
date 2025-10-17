#!/bin/bash
#SBATCH -A hai_1116
#SBATCH -p dc-hwai
#SBATCH --job-name=profam-train
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

set -euo pipefail

# Requeue on pre-timeout and forward TERM to child so it can checkpoint
cleanup_and_requeue() {
  echo "[$(date)] SIGUSR1 received: requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || true
  if [ -n "${APP_PID:-}" ]; then
    echo "[$(date)] Sending SIGTERM to child process group -$APP_PID for graceful checkpoint"
    kill -TERM "-\$APP_PID" 2>/dev/null || true
  fi
}

propagate_term() {
  echo "[$(date)] SIGTERM received: forwarding to child and waiting"
  if [ -n "${APP_PID:-}" ]; then
    kill -TERM "-\$APP_PID" 2>/dev/null || true
    wait "$APP_PID" || true
  fi
  exit 0
}

trap cleanup_and_requeue USR1
trap propagate_term TERM

# Create host-side log dir for Slurm stdout/err if not present
mkdir -p /p/project1/hai_1116/ProFam/logs/slurm

# Prefer large scratch for W&B, caches, tmp if available; fallback to project logs
SCRATCH_BASE=${SCRATCH:-/p/project1/hai_1116/ProFam/logs}
RUN_BASE="$SCRATCH_BASE"

# Inside-container target for logs/cache/tmp
CONTAINER_LOGS=/logs

echo "Using RUN_BASE=$RUN_BASE"

# Ensure bind source paths exist
mkdir -p "$RUN_BASE" || true

# Stable run identifier across requeues: saved per job name
STATE_DIR="$RUN_BASE/state"
mkdir -p "$STATE_DIR"
RUN_ID_FILE="$STATE_DIR/${SLURM_JOB_NAME}.run_id"
if [ -f "$RUN_ID_FILE" ]; then
  RUN_ID=$(cat "$RUN_ID_FILE")
else
  RUN_ID=$(date +"%Y-%m-%d_%H-%M-%S-%N")
  echo "$RUN_ID" > "$RUN_ID_FILE"
fi
EXP_DIR_HOST="$RUN_BASE/runs/$RUN_ID"
mkdir -p "$EXP_DIR_HOST/checkpoints"

# If a checkpoint exists, pass it through to resume
CKPT_PATH_HOST="$EXP_DIR_HOST/checkpoints/last.ckpt"
if [ -f "$CKPT_PATH_HOST" ]; then
  CKPT_PATH_IN_CONTAINER="$CONTAINER_LOGS/runs/$RUN_ID/checkpoints/last.ckpt"
else
  CKPT_PATH_IN_CONTAINER=""
fi

# Launch inside Apptainer (background) so traps can run
setsid apptainer exec --nv \
  --bind /p/project1/hai_1116/ProFam/profam:/workspace \
  --bind /p/project1/hai_1116/ProFam/data:/data \
  --bind "$RUN_BASE":$CONTAINER_LOGS \
  --env WANDB_DIR=$CONTAINER_LOGS \
  --env WANDB_CACHE_DIR=$CONTAINER_LOGS/.wandb_cache \
  --env HF_HOME=$CONTAINER_LOGS/.hf \
  --env TRANSFORMERS_CACHE=$CONTAINER_LOGS/.hf \
  --env TORCH_HOME=$CONTAINER_LOGS/.torch \
  --env MPLCONFIGDIR=$CONTAINER_LOGS/.mpl \
  --env XDG_CACHE_HOME=$CONTAINER_LOGS/.cache \
  --env TMPDIR=$CONTAINER_LOGS/tmp \
  --env RUN_ID=$RUN_ID \
  --env CKPT_PATH_IN_CONTAINER="$CKPT_PATH_IN_CONTAINER" \
  /p/project1/hai_1116/ProFam/profam.sif \
  bash -lc '
    set -euo pipefail
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    nvidia-smi
    mkdir -p /logs/tmp /logs/.wandb_cache /logs/.hf /logs/.torch /logs/.mpl /logs/.cache
    cd /workspace
    python src/train.py \
      experiment=replicate_best_training_run_no_funfams \
      logger.wandb.offline=true \
      logger.wandb.save_dir=/logs \
      logger.wandb.log_hydra_config_file=false \
      trainer.devices=auto \
      "hydra.run.dir=/logs/runs/$RUN_ID" \
      ${CKPT_PATH_IN_CONTAINER:+ckpt_path=$CKPT_PATH_IN_CONTAINER}
  '

# Capture PID and wait to propagate signals
APP_PID=$!
wait $APP_PID


