#!/bin/bash
#$ -P cath
#$ -l tmem=16G
#$ -l tscratch=600G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -pe gpu 2
#$ -l m_core=32
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N MistAll2
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
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd /state/partition1/ProFam_data
echo "pwd: $(pwd)" && echo "ls: $(ls)"
cd $ROOT_DIR
conda activate venvPF
source /share/apps/source_files/python/python-3.11.9.source
python ${ROOT_DIR}/src/train.py \
data=data \
data.data_dir=/state/partition1/ProFam_data \
data.batch_size=100 \
trainer=gpu \
trainer.devices=2 \
trainer.max_epochs=100 \

date
