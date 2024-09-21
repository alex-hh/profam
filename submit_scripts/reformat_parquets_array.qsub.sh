#!/bin/bash
#$ -l tmem=12G
#$ -l h_vmem=12G
#$ -l h_rt=5:55:30
#$ -S /bin/bash
#$ -N reformatFoldseek
#$ -t 1-10000
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
date
# conda activate venvPF
source /SAN/orengolab/cath_plm/ProFam/pfenv.source
DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data python3 data_creation_scripts/reformat_parquets.py $1 --parquet_index $(($SGE_TASK_ID - 1))
date
