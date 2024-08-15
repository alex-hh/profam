#!/bin/bash
#$ -l tmem=88G
# $ -l gpu=true
#$ -l h_rt=123:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct_with_af50.py $1 ${SCRATCH_DIR}/data --minimum_foldseek_cluster_size 10
date

# TODO: zip the scratch dir?
# n.b. if i want to restart parallel - start from 10
rm -rf /scratch0/$USER/$JOB_ID
