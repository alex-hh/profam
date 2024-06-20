#!/bin/bash
#$ -P cath
#$ -l tmem=16G
#$ -l tscratch=600G
#$ -l hostname=clifford* 
#$ -l gpu=true
#$ -pe gpu 2
#$ -l m_core=8
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N plm2
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"

# copy the datasets to the scratch space
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
rsync -av /SAN/orengolab/cath_plm/ProFam/profam ${SCRATCH_DIR}/
cd ${SCRATCH_DIR}/profam
echo "Copied ProFam to ${SCRATCH_DIR}/profam/"

echo "${date} Copying directories to ${SCRATCH_DIR}/data/"
rsync -av /SAN/orengolab/cath_plm/ProFam/data/ec ${SCRATCH_DIR}/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/data/ted ${SCRATCH_DIR}/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/data/ProteinGym ${SCRATCH_DIR}/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/data/cath ${SCRATCH_DIR}/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/data/openfold ${SCRATCH_DIR}/data/ &
wait
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
python ${ROOT_DIR}/src/train.py data=data trainer=gpu trainer.devices=2 trainer.max_epochs=100
date
