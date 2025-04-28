#!/bin/bash

# Train ProFam

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=119:55:30
#$ -S /bin/bash
#$ -N GymMsaTrain
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -P cath
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
export WANDB__SERVICE_WAIT=300
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/train.py \
experiment=train_openfold_foldseekS100S50_ur90_ProtGym \
ckpt_path="logs/train_openfold_clustered_raw_251m/runs/2025-04-20_21-42-36-235228/checkpoints/last.ckpt"
date
