#!/bin/bash

# Train ProFam

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=a100_80
#$ -l h_rt=95:55:30
#$ -S /bin/bash
#$ -N UR901bnV4
#$ -P cath
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
export WANDB__SERVICE_WAIT=900
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
python ${ROOT_DIR}/src/train.py \
experiment=train_single_seq \
experiment_group=train_single_seq_ur90_1bn \
data=uniref90_single \
data.num_workers=16 \
trainer.val_check_interval=5000 \
trainer.max_epochs=1000 \
ckpt_path=null \
trainer.target_tokens_per_batch=1000000 \
+model.override_optimizer_on_load=true \
model.lr=1e-3 \
model.scheduler_name=cosine_with_min_lr \
model.num_warmup_steps=100 \
model.num_training_steps=50000 \
model.config.num_hidden_layers=16 \
trainer.devices=1 \
ckpt_path="/SAN/orengolab/cath_plm/ProFam/profam/logs/train_single_seq_ur90_1bn/runs/2025-04-14_19-51-54-868135/checkpoints/last.ckpt" \
data.pack_to_max_tokens=32000
date
