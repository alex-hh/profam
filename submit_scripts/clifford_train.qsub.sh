#!/bin/bash

# Train ProFam on Clifford
# Clifford has 503GB of RAM
# 4 x A100 80GB, 112 CPUs we request 32 for batch dataloader multiprocessing

#$ -l tmem=256G
#$ -l h_vmem=256G
#$ -l tscratch=1000G
#$ -l hostname=clifford*
#$ -l gpu=true
#$ -l -pe gpu 4
#$ -pe smp 32
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N cliffTrain
#$ -t 1
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

# copy the datasets to the scratch space
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
#date
#rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ec $SCRATCH_DIR/data/
#echo "Copied EC to $SCRATCH_DIR/data/"
#date
#rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ted $SCRATCH_DIR/data/
#echo "Copied TED to $SCRATCH_DIR/data/"
#date
#rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ProteinGym $SCRATCH_DIR/data/
#echo "Copied ProteinGym to $SCRATCH_DIR/data/"
#date
#rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/cath $SCRATCH_DIR/data/
#echo "Copied CATH to $SCRATCH_DIR/data/"
#date
#rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/openfold $SCRATCH_DIR/data/
#echo "Copied OpenFold to $SCRATCH_DIR/data/"
#date
echo "Copying directories to $SCRATCH_DIR/data/"
date
rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ec $SCRATCH_DIR/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ted $SCRATCH_DIR/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/ProteinGym $SCRATCH_DIR/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/cath $SCRATCH_DIR/data/ &
rsync -av /SAN/orengolab/cath_plm/ProFam/profam/data/openfold $SCRATCH_DIR/data/ &
wait
echo "Copied directories to $SCRATCH_DIR/data/"
date
ls $SCRATCH_DIR/data/
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
python ${ROOT_DIR}/src/train.py data=data trainer=clifford
date
