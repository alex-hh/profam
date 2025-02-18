#!/bin/bash

# Train ProFam

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N Single500m
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
experiment=train_single_seq \
data.num_workers=16 \
trainer.val_check_interval=5000
date
