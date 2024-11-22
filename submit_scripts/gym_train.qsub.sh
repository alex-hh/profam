#!/bin/bash

# Train ProFam

#$ -l tmem=60G
#$ -l gpu=true
#$ -l hostname='bubba*'
## $ -l gpu_type=(rtx3090|rtx4090|a6000|a40|a10|a100|a100_80)
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N trainPF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
# conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR

nvidia-smi

SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
rsync -av /home/ahawkins/profam ${SCRATCH_DIR}/
cd ${SCRATCH_DIR}/profam
echo "Copied ProFam to ${SCRATCH_DIR}/profam/"

rsync -av /SAN/orengolab/cath_plm/ProFam/data/ProteinGym ${SCRATCH_DIR}/data/ &

echo "${date} Copied directories to ${SCRATCH_DIR}/data/"
ls ${SCRATCH_DIR}/data/
source /share/apps/source_files/python/python-3.11.9.source
ROOT_DIR=${SCRATCH_DIR}/profam
cd $ROOT_DIR
echo "${date} Installing requirements"
python3 -m venv ${SCRATCH_DIR}/venv
# Activate the virtual environment
source ${SCRATCH_DIR}/venv/bin/activate
pip install -r ${SCRATCH_DIR}/profam/requirements.txt
echo " ${date} Requirements installed"
echo "Using python from $(which python)"
python ${ROOT_DIR}/src/train.py trainer=gpu experiment=gym_train_multi_msa_gpt2_blat data.num_workers=4
date
