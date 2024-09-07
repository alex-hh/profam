#!/bin/bash
#$ -l tmem=128G
#$ -l h_vmem=128G
#$ -l h_rt=12:0:0
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 data_creation_scripts/foldseek/create_foldseek_with_af50.py $1 --skip_af50
date
