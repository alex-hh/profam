#!/bin/bash
#$ -l h_rt=10:55:30
#$ -l h_vmem=8G
#$ -l tmem=8G
#$ -S /bin/bash
#$ -N index_parquets
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
source ~/source_files/afenv.source
python3 data_creation_scripts/save_parquet_index.py "$@"
rm -rf ${SCRATCH_DIR}/data
date
