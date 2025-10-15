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

set -euo pipefail

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

# Launch inside Apptainer
apptainer exec --nv \
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
      trainer.val_check_interval=100 \
      "hydra.run.dir=/logs/runs/\${now:%Y-%m-%d}_\${now:%H-%M-%S-%f}"
  '


