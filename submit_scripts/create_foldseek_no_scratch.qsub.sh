#!/bin/bash
#$ -P cath
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N foldseek3
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python data_creation_scripts/create_foldseek.py /SAN/orengolab/cath_plm
date
