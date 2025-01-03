#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=11:55:30
#$ -S /bin/bash
#$ -N seq_only_afdb_s50
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python data_creation_scripts/seq_only_afdb_s50.py
date