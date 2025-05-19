#!/bin/bash

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a10|a100|a100_80)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N evalGymNormContext
#$ -P cath
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/scripts/adhoc_analysis/eval_ckpt_model_on_gym.py \
--ckpt_path "${ROOT_DIR}/logs/train_foldseekS50_UR90_251m/runs/2025-05-15_freeze/checkpoints/last.ckpt" \
--output_dir "${ROOT_DIR}/results/fseekS50_ur90_model_on_normal_gym_context" \
# --max_context_seqs 0 \
--use_dms_ids \
--max_completion_length 320
date
