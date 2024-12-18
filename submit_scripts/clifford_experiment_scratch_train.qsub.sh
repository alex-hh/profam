#!/bin/bash

# Train ProFam

#$ -l tmem=60G
#$ -l gpu=true
#$ -l hostname='clifford*'
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N train_single_seq
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"


nvidia-smi

SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
rsync -av /SAN/orengolab/plm_embeds/profam ${SCRATCH_DIR}/
cd ${SCRATCH_DIR}/profam
echo "Copied ProFam to ${SCRATCH_DIR}/profam/"

rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_representatives ${SCRATCH_DIR}/data/ &

echo "${date} Copied directories to ${SCRATCH_DIR}/data/"
ls ${SCRATCH_DIR}/data/

ROOT_DIR=${SCRATCH_DIR}/profam
cd $ROOT_DIR

# Activate the virtual environment
source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam
echo "Using python from $(which python)"

python ${ROOT_DIR}/src/train.py experiment=train_single_seq paths.data_dir=${SCRATCH_DIR}/data
date
