#!/bin/bash

#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N shuffle
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
base_dir="/SAN/orengolab/cath_plm/ProFam/data/uniprot"
python data_creation_scripts/shuffle_uniprot_parquets.py --input_dir /SAN/orengolab/cath_plm/ProFam/data/uniprot/ordered_parquet --output_dir /SAN/orengolab/cath_plm/ProFam/data/uniprot/shuffled_uniprot_parquets
