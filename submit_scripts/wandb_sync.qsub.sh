#!/bin/bash

#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=0:55:30
#$ -S /bin/bash
#$ -N wb_sync
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
conda activate venvPF
wandb sync /SAN/orengolab/cath_plm/ProFam/profam/logs/train_ff_fs_ted_pg/runs/2025-01-13_03-22-27-917653/wandb/offline-run-20250113_032451-cwrf1u2s
date
