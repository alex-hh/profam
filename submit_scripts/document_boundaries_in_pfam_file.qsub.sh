#!/bin/bash
#$ -P cath
#$ -l tmem=128G
#$ -l tscratch=6G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -l m_core=64
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N splitPfam
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname
cd /SAN/orengolab/cath_plm/ProFam/profam
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
python data_creation_scripts/document_boundaries_in_pfam_file.py