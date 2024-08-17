#!/bin/bash
#$ -l tmem=6G
#$ -l h_vmem=6G
#$ -l h_rt=5:55:30
#$ -S /bin/bash
#$ -t 1,
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct_with_af50.py $1 ${SCRATCH_DIR}/data  --skip_af50 --num_processes 1 --minimum_foldseek_cluster_size 10 --parquet_ids $((SGE_TASK_ID - 1))
# TODO: zip the scratch dir?
date
