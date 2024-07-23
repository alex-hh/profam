#!/bin/bash
#$ -P cath
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=11:55:30
#$ -S /bin/bash
#$ -N del
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
cd /SAN/orengolab/cath_plm/ProFam/data/foldseek
find . -maxdepth 1 -name "*.fasta" -exec rm {} \;
date