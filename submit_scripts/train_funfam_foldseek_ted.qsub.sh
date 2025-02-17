#!/bin/bash

# Train ProFam

#$ -l tmem=191G
#$ -l h_vmem=191G
#$ -l gpu=true
#$ -l gpu_type=(a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N v4_ff_fs_ted
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
export WANDB__SERVICE_WAIT=180
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/train.py \
experiment=train_funfams_foldseek_ted \
trainer=gpu \
logger=wandb \
data.pack_to_max_tokens=140_000 \
data.num_workers=24 \
trainer.val_check_interval=2500 \
ckpt_path=/SAN/orengolab/cath_plm/ProFam/profam/logs/train_ff_fs_ted_pg/runs/2025-01-13_12-07-44-967554/checkpoints/last.ckpt
date
