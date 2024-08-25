#!/bin/bash
#$ -l h_rt=34:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -l gpu=true
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 data_creation_scripts/build_cluster_db.py
date
