#!/bin/bash
#$ -P cath
#$ -l tmem=32G
#$ -l tscratch=200G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -pe gpu 2
#$ -l m_core=12
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N optz
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
#$ -t 1-2
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
conda activate venvPF
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
SCRATCH_DIR="/scratch0/${USER}/data"
# Function to clean up temporary files
cleanup() {
    echo "[$(date)] Cleaning up temporary files..."
    rm -rf "${SCRATCH_DIR}"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT ERR INT TERM

PFAM_DIR="/SAN/orengolab/cath_plm/ProFam/data/pfam/train_test_split_parquets"
GYM_DIR="/SAN/orengolab/cath_plm/ProFam/data/ProteinGym"
mkdir -p $SCRATCH_DIR/pfam
rsync -av $PFAM_DIR $SCRATCH_DIR/pfam/
rsync -av $GYM_DIR $SCRATCH_DIR/

# Set optimizer based on SGE_TASK_ID
if [ "$SGE_TASK_ID" -eq 1 ]; then
    OPTIMIZER="adamw"
elif [ "$SGE_TASK_ID" -eq 2 ]; then
    OPTIMIZER="lion8bit"
else
    echo "Error: Invalid SGE_TASK_ID. Must be 1 or 2."
    exit 1
fi
tree -d $SCRATCH_DIR
echo "ls ${SCRATCH_DIR}/pfam:"
ls ${SCRATCH_DIR}/pfam
echo "Optimizer: $OPTIMIZER"
python ${ROOT_DIR}/src/train.py \
data=pfam_mix \
data.batch_size=2 \
trainer=gpu \
trainer.devices=auto \
trainer.max_epochs=1000 \
model=llama_medium \
model.lr=1e-3 \
model.optimizer=$OPTIMIZER \
trainer.val_check_interval=1.0 \
data.num_workers=8 \
data.max_tokens=10000 \
paths.data_dir=$SCRATCH_DIR \

date
