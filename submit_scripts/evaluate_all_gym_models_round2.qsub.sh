#!/bin/bash

# Evaluate checkpoints on Protein Gym

#$ -l tmem=127.9G
#$ -l h_vmem=127.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N GYM2
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0"
echo "####################  QSUB SCRIPT END  ####################"

conda activate venvPF

ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# List of all model checkpoint directories (indexed 1-26 to match $SGE_TASK_ID)
# -----------------------------------------------------------------------------
DIRS=(
  "logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325"
)

# Validate SGE_TASK_ID and pick the directory
if (( SGE_TASK_ID < 1 || SGE_TASK_ID > ${#DIRS[@]} )); then
  echo "ERROR: SGE_TASK_ID=${SGE_TASK_ID} is out of range (1..${#DIRS[@]})" >&2
  exit 1
fi

DIR_REL="${DIRS[SGE_TASK_ID-1]}"
DIR="${ROOT_DIR}/${DIR_REL}"
NAME="${DIR_REL#logs/saturn_cloud_good_runs/}_GYM_ONLY"

echo "Selected directory: $DIR"
echo "Experiment group : $NAME"

echo "Running evaluation..."

python src/train.py \
--config-dir="${DIR}/.hydra" \
--config-name=gym_config.yaml \
model.scoring_max_tokens=50_000 \
train=false \
test=true \
data.dataset_builders.proteingym.max_tokens_per_example=500000 \
data.dataset_builders.proteingym.dms_ids=null \
data.dataset_builders.proteingym.max_mutated_sequences=3000 \
+data.dataset_builders.proteingym.max_completion_length=null \
+model.gym_results_save_dir="${DIR} \
ckpt_path="${DIR}/checkpoints/last.ckpt"

date
