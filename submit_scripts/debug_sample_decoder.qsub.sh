#!/bin/bash

# Evaluate checkpoints on Protein Gym

#$ -l tmem=127.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
# -l hostname=!(bubba-213-1*)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N sample1
#$ -t 1-12
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
nvidia-smi
echo "#################### QSUB SCRIPT START ####################"
cat "$0"
echo "####################  QSUB SCRIPT END  ####################"

conda activate venvPF

ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd "$ROOT_DIR"

set -euo pipefail

# Optional overrides via env vars
TOP_P=${TOP_P:-0.95}
TEMPERATURE=${TEMPERATURE:-}
NUM_SAMPLES=${NUM_SAMPLES:-20}
NUM_PROMPTS_IN_ENSEMBLE=${NUM_PROMPTS_IN_ENSEMBLE:-8}
MAX_TOKENS=${MAX_TOKENS:-8192}
MAX_GENERATED_LENGTH=${MAX_GENERATED_LENGTH:-}
REDUCTION=${REDUCTION:-mean_probs} # mean_probs | sum_log_probs
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}

# Map SGE task ID to (dataset, sampler)
DATASETS=(
  foldseek_val
  foldseek_test
  funfams_val
  funfams_test
  pfam_val
  pfam_test
)
GLOBS=(
  "../data/val_test_v2_fastas/foldseek/val/*.fasta"
  "../data/val_test_v2_fastas/foldseek/test/*.fasta"
  "../data/val_test_v2_fastas/funfams/val/*.fasta"
  "../data/val_test_v2_fastas/funfams/test/*.fasta"
  "../data/val_test_v2_fastas/pfam/val/*.fasta"
  "../data/val_test_v2_fastas/pfam/test/*.fasta"
)
SAMPLERS=(ensemble single)

IDX=$((SGE_TASK_ID - 1))
NUM_DATASETS=${#DATASETS[@]}
DATASET_IDX=$((IDX % NUM_DATASETS))
SAMPLER_IDX=$((IDX / NUM_DATASETS))

DATASET=${DATASETS[$DATASET_IDX]}
GLOB=${GLOBS[$DATASET_IDX]}
SAMPLER=${SAMPLERS[$SAMPLER_IDX]}

# Build descriptive save directory reflecting dataset and settings
TEMP_SEG=""
if [ -n "${TEMPERATURE}" ]; then
  TEMP_SEG="_temp=${TEMPERATURE}"
fi
MAXLEN_SEG=""
if [ -n "${MAX_GENERATED_LENGTH}" ]; then
  MAXLEN_SEG="_maxlen=${MAX_GENERATED_LENGTH}"
fi

SAVE_DIR="../sampling_results/${DATASET}/sampler=${SAMPLER}_tp=${TOP_P}${TEMP_SEG}_ns=${NUM_SAMPLES}_nv=${NUM_PROMPTS_IN_ENSEMBLE}_red=${REDUCTION}${MAXLEN_SEG}"

echo "[SGE_TASK_ID=${SGE_TASK_ID}] dataset=${DATASET} sampler=${SAMPLER}"
echo "glob=${GLOB}"
echo "save_dir=${SAVE_DIR}"

EXTRA_ARGS=()
if [ -n "${TEMPERATURE}" ]; then
  EXTRA_ARGS+=(--temperature "${TEMPERATURE}")
fi
if [ -n "${MAX_GENERATED_LENGTH}" ]; then
  EXTRA_ARGS+=(--max_generated_length "${MAX_GENERATED_LENGTH}")
fi
if [ -n "${CHECKPOINT_DIR}" ]; then
  EXTRA_ARGS+=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi

CMD=(
  python -u scripts/adhoc_analysis/debug_ensemble_decoder.py
  --glob "${GLOB}"
  --save_dir "${SAVE_DIR}"
  --sampler "${SAMPLER}"
  --num_prompts_in_ensemble "${NUM_PROMPTS_IN_ENSEMBLE}"
  --num_samples "${NUM_SAMPLES}"
  --max_tokens "${MAX_TOKENS}"
  --top_p "${TOP_P}"
  --reduction "${REDUCTION}"
  --device cuda
  --dtype bfloat16
  --msa
)

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

date