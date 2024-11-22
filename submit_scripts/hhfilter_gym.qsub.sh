#!/bin/bash

# Run hhfilter to generate smaller proteingym msas

#$ -l tmem=10G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N hhfilter_gym
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y


source ~/source_files/hhsuite.source
./scripts/hhfilter_gym.sh
