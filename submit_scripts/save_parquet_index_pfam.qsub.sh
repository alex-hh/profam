#!/bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N parqIdx
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "qsub script: $0"
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"

conda activate venvPF

ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'

cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/data_creation_scripts/save_parquet_index.py \
"/SAN/orengolab/cath_plm/ProFam/data/pfam/shuffled_parquets/index.csv" \
"/SAN/orengolab/cath_plm/ProFam/data/pfam/shuffled_parquets/*.parquet" \
--identifier_col "fam_id"
date