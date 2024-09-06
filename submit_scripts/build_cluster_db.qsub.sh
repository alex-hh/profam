#!/bin/bash
#$ -l tmem=128G
#$ -l h_vmem=128G
#$ -l h_rt=128:55:30
#$ -S /bin/bash
#$ -N builddb
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
hostname
# conda activate venvPF
source ~/source_files/afenv.source
# source /share/apps/source_files/python/python-3.11.9.source
python3 data_creation_scripts/foldseek/build_cluster_db.py $@
date
