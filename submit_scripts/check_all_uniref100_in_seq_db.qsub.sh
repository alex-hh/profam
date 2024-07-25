#!/bin/bash
#$ -P cath
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N addSeq
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python scripts/check_all_uniref100_in_seq_db.py
date
