#!/bin/bash
#$ -l tmem=128G
#$ -l h_vmem=128G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
date
# conda activate venvPF
source /SAN/orengolab/cath_plm/ProFam/pfenv.source
export PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data
python3 data_creation_scripts/create_foldseek_with_af50.py $1
date
