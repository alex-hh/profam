#!/bin/bash
#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=34:55:30
#$ -S /bin/bash
#$ -N aug_foldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source
python3 scripts/summarise_indexed_dataset.py "$@"
date
