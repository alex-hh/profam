#!/bin/bash

#$ -l tmem=7.9G
#$ -l h_vmem=7.9G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N clusterFamilies
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/clustering/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-200
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"
conda activate venvPF
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
python /SAN/orengolab/cath_plm/ProFam/profam/data_creation_scripts/cluster_within_families.py \
    --input_pattern "/SAN/orengolab/cath_plm/ProFam/data/ted/s100_parquets/train_val_test_split_hq/train_val_test_split/*/*.parquet" \
    --task_index $((SGE_TASK_ID - 1)) \
    --num_tasks $SGE_TASK_LAST
date