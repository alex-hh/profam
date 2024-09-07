#!/bin/bash
#$ -l tmem=30G
#$ -l h_vmem=30G
#$ -l h_rt=64:55:30
#$ -l gpu=true
#$ -S /bin/bash
#$ -N aug_foldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source

# copy the datasets to the scratch space
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek_af50 ${SCRATCH_DIR}/
rsync -av /SAN/orengolab/cath_plm/ProFam/data/afdb ${SCRATCH_DIR}/
rsync -av /SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_aug ${SCRATCH_DIR}/
save_dir=${SCRATCH_DIR}/foldseek_af50_aug

python3 data_creation_scripts/foldseek/create_foldseek_clusters_of_clusters.py ${SCRATCH_DIR}/foldseek_af50/index.csv \
    ${SCRATCH_DIR}/foldseek_af50_aug --cluster_path ${SCRATCH_DIR}/afdb/1-AFDBClusters-entryId_repId_taxId.tsv \
    --all_vs_all_path "${SCRATCH_DIR}/afdb/6-all-vs-all-similarity-queryId_targetId_eValue.tsv" "$@"

rsync -av ${SCRATCH_DIR}/foldseek_af50_aug /SAN/orengolab/cath_plm/ProFam/data/
