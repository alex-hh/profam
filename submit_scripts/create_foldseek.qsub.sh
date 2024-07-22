#!/bin/bash
#$ -P cath
#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l tscratch=256G
#$ -l h_rt=95:55:30
#$ -S /bin/bash
#$ -N foldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data/
rsync -av /SAN/orengolab/cath_plm/profam_db ${SCRATCH_DIR}/
echo "Copied ProFam to ${SCRATCH_DIR}/profam_db/"
conda activate venvPF
python data_creation_scripts/create_foldseek.py ${SCRATCH_DIR}
date
