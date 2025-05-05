#!/bin/bash

#$ -l tmem=31.9G
#$ -l h_vmem=31.9G
#$ -l h_rt=91:55:30
#$ -S /bin/bash
# hostname does not contain larry* or arbuckle* or abner*
#$ -l hostname=!(larry*|arbuckle*|abner*)
#$ -N RedRedSplitV2
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/mmseqs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-6
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"
conda activate venvPF
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
python data_creation_scripts/mmseqs_train_test_split_apply_to_redundancy_reduced_datasets.py \
    --task_index $((SGE_TASK_ID - 1))
date
