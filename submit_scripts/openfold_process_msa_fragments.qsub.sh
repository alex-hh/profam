#!/bin/bash

#$ -l tmem=3.9G
#$ -l h_vmem=3.9G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -l tscratch=10G
# hostname does not contain larry* or arbuckle*
#$ -l hostname=!(larry*|arbuckle*)
#$ -N openfoldCluster
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/clusteringV2/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-270
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
python /SAN/orengolab/cath_plm/ProFam/profam/data_creation_scripts/openfold_process_msa_fragments.py \
    --input_pattern "/SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_filtered_parquet/*.parquet" \
    --output_dir "/SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_filtered_parquet_fragments_ucl_cluster" \
    --task_index $((SGE_TASK_ID - 1)) \
    --num_tasks $SGE_TASK_LAST \
    --scratch_dir $SCRATCH_DIR
date
