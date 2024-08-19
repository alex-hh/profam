#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N create_funfam_parquets
#$ -t 1-20
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
SAVE_DIR="/SAN/orengolab/cath_plm/ProFam/data/funfams/parquets"
cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/data_creation_scripts/create_funfam_parquets.py --task_index $((SGE_TASK_ID - 1)) --num_tasks 20 --save_dir $SAVE_DIR
date

SAVE_SOURCE_PATH=${SAVE_DIR}/parquet_creation_source_code.txt
# if SGE_TASK_ID=1, then save the contents of this script and python script to SAVE_SOURCE_PATH
if [ "$SGE_TASK_ID" -eq 1 ]; then
    echo "##### start #####" > $SAVE_SOURCE_PATH
    cat "$0" >> $SAVE_SOURCE_PATH
    echo "##### Python script #####" >> $SAVE_SOURCE_PATH
    cat "${ROOT_DIR}/create_funfam_parquets.py" >> $SAVE_SOURCE_PATH
fi
