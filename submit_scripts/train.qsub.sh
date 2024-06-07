#!/bin/bash

# Train ProFam

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G
#$ -l gpu=true
#$ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a100|a100_80)
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N trainPF
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
python ${ROOT_DIR}/src/train.py data=data trainer=gpu
date