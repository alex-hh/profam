#!/bin/bash
#$ -l tmem=108G
#$ -l h_vmem=108G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct.py $1
date
