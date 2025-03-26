#!/bin/bash

# Train ProFam

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a100|a100_80)
#$ -l h_rt=119:55:30
#$ -S /bin/bash
#$ -N FamFromS90
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -P cath
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
export WANDB__SERVICE_WAIT=300
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/train.py \
experiment=train_single_seq \
data.num_workers=16 \
trainer.val_check_interval=5000 \
trainer.max_epochs=1000 \
ckpt_path="/SAN/orengolab/cath_plm/ProFam/profam/logs/train_single_seq/runs/2025-03-10_17-46-18-441714/checkpoints/last.ckpt" \
trainer.target_tokens_per_batch=1000000 \
+model.override_optimizer_on_load=true \
model.lr=7e-4 \
model.scheduler_name=cosine_with_min_lr \
model.num_warmup_steps=100 \
model.num_training_steps=200000
date
