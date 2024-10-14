#!/bin/bash
#$ -P cath
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l hostname=clifford*
#$ -l gpu=true
# -pe gpu 2
#$ -l m_core=8
#$ -l h_rt=95:55:30
#$ -S /bin/bash
#$ -N allNoScratch
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
conda activate venvPF
python ${ROOT_DIR}/src/train.py \
data=foldseek_function_pfam \
trainer=gpu \
trainer.devices=auto \
trainer.max_epochs=1000 \
model=llama_medium \
model.lr=2e-3 \
paths.data_dir="../data" \
logger=wandb \
trainer.val_check_interval=2000 \
data.batch_size=8 \
model.embed_sequence_index=True
date
