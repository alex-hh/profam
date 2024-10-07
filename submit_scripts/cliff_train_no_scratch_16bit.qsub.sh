#!/bin/bash
#$ -P cath
#$ -l tmem=64G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -pe gpu 1
#$ -l m_core=32
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N bit16flash
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
#$ -t 1
date
hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PROJECT_ROOT=$ROOT_DIR
cd $ROOT_DIR
source ../pfenv/bin/activate
echo "Using python from $(which python)"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
python ${ROOT_DIR}/src/train.py \
data=pfam_mix \
data.batch_size=9 \
trainer=gpu \
trainer.devices=auto \
trainer.max_epochs=1000 \
model=llama_medium \
model.lr=1e-3 \
model.optimizer="adamw" \
model.config.attn_implementation="flash_attention_2" \
model.embed_sequence_index="true" \
trainer.val_check_interval=1.0 \
trainer.precision="bf16-true" \
data.num_workers=10 \
data.max_tokens=10000 \
paths.data_dir="/SAN/orengolab/cath_plm/ProFam/data"  \
float32_matmul_precision=high \
callbacks=default_with_shuffle
date