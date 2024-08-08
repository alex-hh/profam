#!/bin/bash
#$ -l tmem=88G
#$ -pe smp 12
#$ -l h_rt=123:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct_with_af50.py $1 --parquet_ids 0
date

# n.b. if i want to restart parallel - start from 10
