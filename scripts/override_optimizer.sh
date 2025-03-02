#!/bin/bash
set -e  # Exit on error

# Define paths and variables
ROOT_DIR=$(pwd)
EXPERIMENT_DIR="${ROOT_DIR}/test_override_optimizer"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
DATA_DIR="../data"
LOG_DIR="${ROOT_DIR}/logs"

# Create necessary directories
mkdir -p "${EXPERIMENT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=== Testing override_optimizer_on_load functionality ==="
echo "Experiment directory: ${EXPERIMENT_DIR}"

# Step 1: Train a tiny model for 1 epoch and save checkpoint
echo "=== Step 1: Initial training run with lr=3e-4 ==="
python src/train.py \
  data=afdb_s50_single \
  data.batch_size=2 \
  data.num_workers=2 \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  trainer.limit_val_batches=5 \
  model=llama_tiny \
  model.lr=3e-4 \
  model.scheduler_name="cosine" \
  model.num_warmup_steps=2 \
  paths.output_dir="${EXPERIMENT_DIR}" \
  paths.data_dir="${DATA_DIR}" \
  logger=wandb \
  callbacks=default_no_shuffle

# Find the checkpoint file
CHECKPOINT_PATH=$(find "${CHECKPOINT_DIR}" -name "last.ckpt" -type f)
if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Error: Checkpoint file not found!"
  exit 1
fi
echo "Checkpoint saved at: ${CHECKPOINT_PATH}"

# Step 2: Resume training without override (should keep original lr)
echo "=== Step 2: Resume training WITHOUT override (lr=1e-5, but should keep original lr=3e-4) ==="
python src/train.py \
  data=afdb_s50_single \
  data.batch_size=2 \
  data.num_workers=2 \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_epochs=2 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  model=llama_tiny \
  model.lr=1e-5 \
  model.scheduler_name="linear" \
  model.num_warmup_steps=1 \
  model.override_optimizer_on_load=False \
  paths.output_dir="${EXPERIMENT_DIR}/resume_without_override" \
  paths.data_dir="${DATA_DIR}" \
  logger=csv \
  callbacks=default_no_shuffle \
  ckpt_path="${CHECKPOINT_PATH}" \
  2>&1 | tee "${LOG_DIR}/resume_without_override.log"

# Step 3: Resume training with override (should use new lr)
echo "=== Step 3: Resume training WITH override (should use new lr=1e-5) ==="
python src/train.py \
  data=afdb_s50_single \
  data.batch_size=2 \
  data.num_workers=2 \
  trainer=gpu \
  trainer.devices=1 \
  trainer.max_epochs=2 \
  trainer.limit_train_batches=5 \
  trainer.limit_val_batches=2 \
  model=llama_tiny \
  model.lr=1e-5 \
  model.scheduler_name="linear" \
  model.num_warmup_steps=1 \
  model.override_optimizer_on_load=True \
  paths.output_dir="${EXPERIMENT_DIR}/resume_with_override" \
  paths.data_dir="${DATA_DIR}" \
  logger=csv \
  callbacks=default_no_shuffle \
  ckpt_path="${CHECKPOINT_PATH}" \
  2>&1 | tee "${LOG_DIR}/resume_with_override.log"

echo "=== Test completed ==="
echo "Check logs in ${LOG_DIR} to verify the learning rates"
echo "Without override: ${LOG_DIR}/resume_without_override.log"
echo "With override: ${LOG_DIR}/resume_with_override.log"