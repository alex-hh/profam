#!/bin/bash

#$ -l tmem=192G
#$ -l h_vmem=192G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N fastaParquet
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"
conda activate venvPF
python conda activate venvPF/data_creation_scripts/create_uniprot_singe_seq_parquets.py