#!/bin/bash
#$ -l tmem=35G
#$ -l h_vmem=35G
#$ -l h_rt=35:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
date
# conda activate venvPF
source /share/apps/source_files/python/python-3.11.9.source
source /SAN/orengolab/cath_plm/ProFam/pfenv/bin/activate
DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 data_creation_scripts/reformat_parquets.py $1
date
