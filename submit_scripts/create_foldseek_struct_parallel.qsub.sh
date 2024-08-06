#!/bin/bash
#$ -l tmem=48G
#$ -l h_vmem=48G
#$ -l h_rt=24:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -j y
#$ -t 1-250
date
# conda activate venvPF
cd ~/profam
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct.py $1 --parquet_index ${SGE_TASK_ID}
date
