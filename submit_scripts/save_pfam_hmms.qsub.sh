#!/bin/bash
#$ -l tmem=6G
#$ -l h_vmem=6G
#$ -l h_rt=03:55:30
#$ -S /bin/bash
#$ -N save_pfam_hmms
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
export USE_TORCH=1
source ~/source_files/afenv.source
python3 data_creation_scripts/save_pfam_hmms.py "$@"
# TODO: zip the scratch dir?
date
