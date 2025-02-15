#!/bin/bash
#$ -P cath
#$ -l tmem=192G
#$ -l h_vmem=192G
#$ -l h_rt=11:55:30
#$ -S /bin/bash
#$ -N checkParqs5
#$ -l gpu=true
#$ -l hostname="bubba*"
#$ -t 1-10
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
python scripts/adhoc_analysis/MULTITHREAD_validate_cath_split_parquets.py --task_index $SGE_TASK_ID
date
