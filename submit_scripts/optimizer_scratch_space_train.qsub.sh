#!/bin/bash
#$ -l tmem=64G
#$ -l tscratch=200G
#$ -l m_core=32
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -pe gpu 1
#$ -N optimizer_test
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


SCRATCH_DIR="/scratch0/${USER}/data"
export HF_HOME="/scratch0/${USER}/hf"
# Function to clean up temporary files
cleanup() {
    echo "[$(date)] Cleaning up temporary files..."
    rm -rf "${SCRATCH_DIR}"
}

# Trap to ensure cleanup on exit
trap cleanup EXIT ERR INT TERM

# Set error and debugging options
set -e
set -x

# Create necessary directories in scratch space
echo "Creating directories in scratch space... ${SCRATCH_DIR}" 
mkdir -p $SCRATCH_DIR
echo "Creating directories in scratch space... ${HF_HOME}" 
mkdir -p $HF_HOME

PFAM_DIR="/SAN/orengolab/cath_plm/ProFam/data/pfam/train_test_split_parquets"
GYM_DIR="/SAN/orengolab/cath_plm/ProFam/data/ProteinGym"
mkdir -p $SCRATCH_DIR/pfam
# check if scratch directory is created
if [ -d $SCRATCH_DIR/pfam ]; then
    echo "Scrat Pfam directory exists"
else
    echo "Directory does not exist"
    exit 1
fi

# Sync input files from funfams_noali to scratch space
rsync -av $PFAM_DIR $SCRATCH_DIR/pfam/
rsync -av $GYM_DIR $SCRATCH_DIR/


# Set optimizer based on SGE_TASK_ID
if [ "$SGE_TASK_ID" -eq 1 ]; then
    OPTIMIZER="adamw"
elif [ "$SGE_TASK_ID" -eq 2 ]; then
    OPTIMIZER="lion8bit"
    echo "testing lion8bit with python -m bitsandbytes:"
    python -m bitsandbytes
else
    echo "Error: Invalid SGE_TASK_ID. Must be 1 or 2."
    exit 1
fi


echo "ls $SCRATCH_DIR:"
ls $SCRATCH_DIR
echo "ls ${SCRATCH_DIR}/pfam:"
ls ${SCRATCH_DIR}/pfam
echo "Optimizer: $c"

# Run script

# export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
# export PROJECT_ROOT=$ROOT_DIR
# cd $ROOT_DIR
# conda activate venvPF
# export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam
source /share/apps/source_files/cuda/cuda-11.7.source

OPTIMIZER="adamw"

export ROOT_DIR='/SAN/orengolab/plm_embeds/profam'
# export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

cd $ROOT_DIR

python3 ${ROOT_DIR}/src/train.py \
data=pfam_mix \
data.batch_size=10 \
trainer=gpu \
trainer.devices=auto \
trainer.max_epochs=1000 \
model=llama_medium \
model.lr=4e-3 \
model.optimizer=$OPTIMIZER \
trainer.val_check_interval=1.0 \
data.num_workers=30 \
data.max_tokens=10000 \
float32_matmul_precision=high \
callbacks=default_with_shuffle \
paths.data_dir=$SCRATCH_DIR  #"/SAN/orengolab/cath_plm/ProFam/data" \

date


python src/train.py \
data=pfam_mix \
data.evaluate_pfam_class=false \
data.batch_size=10 \
trainer=gpu \
trainer.devices=auto \
trainer.max_epochs=1000 \
model=llama_medium \
model.lr=4e-3 \
model.optimizer=$OPTIMIZER \
trainer.val_check_interval=1.0 \
data.num_workers=30 \
data.max_tokens=10000 \
float32_matmul_precision=high \
callbacks=default_with_shuffle \
paths.data_dir=/SAN/orengolab/cath_plm/ProFam/data
