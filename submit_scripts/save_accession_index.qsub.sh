#!/bin/bash
#$ -l tmem=88G
#$ -l h_vmem=88G
#$ -l h_rt=128:55:30
#$ -S /bin/bash
#$ -N saveindex
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
date
hostname
# conda activate venvPF
source /share/apps/source_files/python/python-3.11.9.source
source /SAN/orengolab/cath_plm/ProFam/pfenv/bin/activate
export PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data
# source /share/apps/source_files/python/python-3.11.9.source
python3 data_creation_scripts/foldseek/save_accession_index.py $@
date
