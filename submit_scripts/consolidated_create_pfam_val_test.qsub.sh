#!/bin/bash

#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N PfamValTest
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"

conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR

# Add ROOT_DIR to PYTHONPATH
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

python ${ROOT_DIR}/data_creation_scripts/pfam/consolidated_create_pfam_val_test.py

date
