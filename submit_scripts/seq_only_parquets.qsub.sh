#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N seq_only_parquets
#$ -t 20
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python data_creation_scripts/seq_only_parquets.py \
--task_index $((SGE_TASK_ID - 1)) \
--num_tasks $SGE_TASK_LAST \
--parquet_dir /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s100_raw/train_val_test_split \
--new_parquet_dir /SAN/orengolab/cath_plm/ProFam/data/foldseek/foldseek_s100_raw/train_val_test_split_seq_only
date
