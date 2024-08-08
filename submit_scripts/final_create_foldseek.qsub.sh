#!/bin/bash
#$ -P cath
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python data_creation_scripts/final_create_foldseek.py
date
