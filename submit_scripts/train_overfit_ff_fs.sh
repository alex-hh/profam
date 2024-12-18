#!/bin/bash

# Train ProFam

#$ -l tmem=63.9G
#$ -l h_vmem=63.9G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N trainOverfit
#$ -t 1
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
python ${ROOT_DIR}/src/train.py \
experiment=benchmark/overfit_funfams_foldseek \
trainer=gpu \
logger=wandb \
data.pack_to_max_tokens=45_000 \
data.num_workers=6
date
