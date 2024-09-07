#!/bin/bash
#$ -l h_rt=58:55:30
#$ -l h_vmem=88G
#$ -l tmem=88G
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
cp /SAN/bioinf/afdb_domain/zipmaker/zip_index $SCRATCH_DIR/data/
cp /SAN/orengolab/cath_plm/ProFam/data/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv $SCRATCH_DIR/data/
source ~/source_files/afenv.source
export PATH=/SAN/orengolab/cath_plm/ProFam/foldseek/bin/:$PATH
python3 data_creation_scripts/foldseek/create_foldseek_representatives.py $1 ${SCRATCH_DIR}/data
rm -rf ${SCRATCH_DIR}/data
date
