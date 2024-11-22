#!/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -l tscratch=200G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -pe gpu 2
#$ -R y
#$ -N PFam_Scratch
#$ -j y
#$ -cwd
#$ -P cath
#$ -e /SAN/orengolab/plm_embeds/profam/logs
#$ -o /SAN/orengolab/plm_embeds/profam/logs

date
hostname
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"



# Set scratch directory
SCRATCH_DIR=/scratch0
# TMP_DIR=${SCRATCH_DIR}/$USER/$JOB_NAME-$JOB_ID-$SGE_TASK_ID
DATA_DIR=${SCRATCH_DIR}/$USER/

# Function to clean up temporary files
cleanup() {
    echo "[$(date)] Cleaning up temporary files..."
    rm -rf "${DATA_DIR}"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT ERR INT TERM

# Define paths
INPUT_DIR=/SAN/orengolab/cath_plm/ProFam/data/pfam/train_test_split_parquets
PFAM_DIR=${DATA_DIR}/data/pfam
PROTEINGYM_DIR=${DATA_DIR}/data/ProteinGym

# Set error and debugging options
set -e
set -x

# Create necessary directories in scratch space
echo "Creating directories in scratch space... ${DATA_DIR}" 
mkdir -p $DATA_DIR

echo "Creating directories in scratch space... ${PFAM_DIR}" 
mkdir -p $PFAM_DIR

echo "Creating directories in scratch space... ${PROTEINGYM_DIR}" 
mkdir -p $PROTEINGYM_DIR

# Sync input files from funfams_noali to scratch space
# rsync -a $INPUT_DIR/ $SFAM_DIR/
rsync -a $INPUT_DIR ${PFAM_DIR}/
rsync -a /SAN/orengolab/cath_plm/ProFam/data/ProteinGym ${PROTEINGYM_DIR}/

ls $PFAM_DIR

# Run script
source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam

export ROOT_DIR='/SAN/orengolab/plm_embeds/profam'
# export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

cd $ROOT_DIR

python ${ROOT_DIR}/src/train.py \
data=pfam_mix \
data.evaluate_pfam_class=false \
data.data_dir=${DATA_DIR}/data \
trainer=gpu \
trainer.devices=auto \
model=llama_medium \
model.lr=1e-3 \
trainer.val_check_interval=1.0 \
data.num_workers=16 \
data.batch_size=7 \
data.max_tokens=10000 \
trainer.max_epochs=10000

echo "DONE!!!!"
date

# python /SAN/orengolab/plm_embeds/profam/src/train.py \
# data=pfam_mix \
# data.evaluate_pfam_class=false \
# data.data_dir=/SAN/orengolab/cath_plm/ProFam/data \
# trainer=gpu \
# trainer.devices=auto \
# model=llama_medium \
# model.lr=1e-3 \
# trainer.val_check_interval=1.0 \
# data.num_workers=16 \
# data.batch_size=7 \
# data.max_tokens=10000 \
# trainer.max_epochs=10000