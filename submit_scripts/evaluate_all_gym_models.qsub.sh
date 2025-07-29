#!/bin/bash

# Evaluate checkpoints on Protein Gym

#$ -l tmem=127.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N GYM3
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
#   "logs/saturn_cloud_good_runs/0a3sckdo_ted_s100_ff50_ff100_openfold_fs100_fs50_ur90/copied_2025-06-30_15-56/2025-06-23_13-03-29-461041"
#   "logs/saturn_cloud_good_runs/2l63mstm_ff_openfold_fs50_ur90_250m/copied_2025-06-21_11-08/2025-06-11_00-29-03-422898"
#   "logs/saturn_cloud_good_runs/28yyyp8u_ff50_ff100_openfold_fs100_fs50_ur90_553/copied_2025-06-22_09-34/2025-06-17_13-41-16-626286"
#   "logs/saturn_cloud_good_runs/28yyyp8u_ff50_ff100_openfold_fs100_fs50_ur90_553/copied_2025-06-25_07-09/2025-06-17_13-41-16-626286"
#   "logs/saturn_cloud_good_runs/28yyyp8u_ff50_ff100_openfold_fs100_fs50_ur90_553/copied_2025-06-25_14-45/2025-06-17_13-41-16-626286"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-18_22-08"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-20_15-08/2025-06-11_00-09-25-545380"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-21_08-13/2025-06-11_00-09-25-545380"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-22_21-15/2025-06-11_00-09-25-545380"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-24_10-54/2025-06-11_00-09-25-545380"
#   "logs/saturn_cloud_good_runs/60lnxlim_ff50_ff100_of_fs50_ur90/copied_2025-06-24_13-04/2025-06-11_00-09-25-545380"
#   "logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/2025-06-15_22-33"
#   "logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-20_07-35/2025-06-10_22-48-14-455325"
  "logs/saturn_cloud_good_runs/abyoeovl_openfold_fs50_ur90_memmap_251m/copied_2025-06-23_22-18/2025-06-10_22-48-14-455325"
#   "logs/saturn_cloud_good_runs/g4mkecal_ted_s100_ff50_ff100_openfold_fs100_fs50_ur90/copied_2025-06-24_10-47/2025-06-23_09-14-32-431398"
#   "logs/saturn_cloud_good_runs/g4mkecal_ted_s100_ff50_ff100_openfold_fs100_fs50_ur90/copied_2025-06-27_11-20/2025-06-23_09-14-32-431398"
#   "logs/saturn_cloud_good_runs/qrdm0jk9_ff50_ff100_openfold_fs100_fs50_ur90/copied/2025-06-16_20-15-01-754653"
#   "logs/saturn_cloud_good_runs/qrdm0jk9_ff50_ff100_openfold_fs100_fs50_ur90/copied_2025-06-21_11-14/2025-06-16_20-15-01-754653"
#   "logs/saturn_cloud_good_runs/ttzdquol_ff_openfold_fs50_ur90_1b/copied_2025-06-30_16-29/2025-06-15_18-13-45-504178"
#   "logs/saturn_cloud_good_runs/uljreks3_ted_s100_ff50_ff100_openfold_fs100_fs50_ur90_lr0.0004_acc1_wd0.4_pack28000_8GPU_ur90crop320_ur90crop1024-554M/copied_2025-06-30_14-57/2025-06-23_12-25-03-504044"
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
model.scoring_max_tokens=90_000 \
train=false \
test=true \
data.dataset_builders.proteingym.max_tokens_per_example=10_000_000 \
data.dataset_builders.proteingym.dms_ids=null \
data.dataset_builders.proteingym.max_mutated_sequences=3000 \
+data.dataset_builders.proteingym.max_completion_length=1024 \
+model.gym_results_save_dir="${DIR} \
ckpt_path="${DIR}/checkpoints/last.ckpt"

date
