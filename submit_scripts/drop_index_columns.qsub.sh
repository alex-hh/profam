#!/bin/bash
#$ -P cath
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N dropIndexCols
#$ -t 1-200
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
python scripts/adhoc_analysis/drop_index_columns.py --task_index $(($SGE_TASK_ID - 1)) --num_tasks $SGE_TASK_LAST
date
