#!/bin/bash

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -l tscratch=10G
# hostname does not contain larry* or arbuckle* or abner*
#$ -l hostname=!(larry*|arbuckle*|abner*)
#$ -N mmseqs_splitV2
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/mmseqs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-335
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"
conda activate venvPF
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p $SCRATCH_DIR
python data_creation_scripts/mmseqs_train_test_split.py \
    --task_index $((SGE_TASK_ID - 1)) \
    --scratch_dir $SCRATCH_DIR
date
