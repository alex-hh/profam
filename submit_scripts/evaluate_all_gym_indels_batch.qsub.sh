#!/bin/bash

# Evaluate checkpoints on Protein Gym

#$ -l tmem=127.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
#$ -l hostname=!(bubba-213-1*)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N GYMindels1
#$ -t 1-10
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

# -----------------------------------------------------------------------------
# List of all model checkpoint directories (indexed 1-26 to match $SGE_TASK_ID)
# -----------------------------------------------------------------------------


DIR_REL="logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325"
DIR="${ROOT_DIR}/${DIR_REL}"
NAME="${DIR_REL#logs/saturn_cloud_good_runs/}_GYM_ONLY_INDELS"

echo "Selected directory: $DIR"
echo "Experiment group : $NAME"

echo "Running evaluation..."
python src/train.py \
--config-dir="${DIR}/.hydra" \
--config-name=gym_config.yaml \
model.scoring_max_tokens=150_000 \
train=false \
test=true \
data.dataset_builders.proteingym.max_tokens_per_example=1000_000_000 \
data.dataset_builders.proteingym.dms_ids=null \
data.dataset_builders.proteingym.max_mutated_sequences=1000 \
+data.dataset_builders.proteingym.max_completion_length=null \
data.dataset_builders.proteingym.use_filtered_msa=false \
+model.gym_results_save_dir="${DIR}/indels_poet_msa_diversity_weighting_only" \
+data.dataset_builders.proteingym.msa_folder_name=PoET_DMS_msa_files/DMS_substitutions \
+model.gym_subsamples_per_n=120 \
+data.dataset_builders.proteingym.task_index=$((SGE_TASK_ID - 1)) \
+data.dataset_builders.proteingym.num_tasks=10 \
+data.dataset_builders.proteingym.csv_filename="DMS_indels.csv" \
+data.dataset_builders.proteingym.use_msa_seq_weights=true \
ckpt_path="${DIR}/checkpoints/last.ckpt"

date
