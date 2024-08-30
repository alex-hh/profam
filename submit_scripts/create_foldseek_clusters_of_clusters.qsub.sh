#!/bin/bash
#$ -l tmem=30G
#$ -l h_vmem=30G
#$ -l h_rt=64:55:30
#$ -S /bin/bash
#$ -N aug_foldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_clusters_of_clusters.py "$@"
date
