#!/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N shufflePfam
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "qsub script: $0"
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"

conda activate venvPF

ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'

cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/scripts/shuffle_pfam_parquets.py --task_index $((SGE_TASK_ID - 1))
date
