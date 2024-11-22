#!/bin/bash

# Train ProFam

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G
#$ -l gpu=true
#$ -l gpu_type=(a6000|a40|a100|a100_80)
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -P cath
#$ -j y
#$ -R y
#$ -N mixtral_sweep
#$ -cwd

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam
which python3
python3 --version

ROOT_DIR='/SAN/orengolab/plm_embeds/profam'
cd $ROOT_DIR
# python src/train.py -m hparams_search=mixtral_sweep_multi_msa experiment=mixtral_multi_msa

# !! Run in the terminal: 1. init wandb
# wandb init

# !! Run in the terminal: 2. choose the sweep config file to create sweep id
# wandb sweep configs/hparams_search/mistral_sweep.yaml

# the previous step will create a sweep id, which is a string of characters, e.g. cath/mistral_sweep/xxxxxxxx
# 3. Run the sweep with the qsub script: wandb agent <replace with the sweep id>
wandb agent cath/mistral_sweep/xxxxxxxx

date