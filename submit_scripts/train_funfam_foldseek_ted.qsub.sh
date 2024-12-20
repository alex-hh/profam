#!/bin/bash

# Train ProFam

#$ -l tmem=127.9G
#$ -l h_vmem=127.9G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N trainAll
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
logger=stdout \
data.pack_to_max_tokens=90_000 \
data.num_workers=4
date
