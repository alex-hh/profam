#!/bin/bash
#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h_rt=35:55:30
#$ -S /bin/bash
#$ -N reformatFoldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
date
# conda activate venvPF
source /SAN/orengolab/cath_plm/ProFam/pfenv.source
DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 data_creation_scripts/save_parquet_index.py $1
date
