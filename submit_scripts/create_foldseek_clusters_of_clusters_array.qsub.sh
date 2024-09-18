#!/bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=34:55:30
#$ -l gpu=true
#$ -S /bin/bash
#$ -N aug_foldseek
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -t 1-1
#$ -tc 1000
#$ -j y
date
# conda activate venvPF
source ~/source_files/afenv.source

# copy the datasets to the scratch space
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data
cp $DATA_DIR/afdb/6-all-vs-all-similarity-queryId_targetId_eValue.tsv $SCRATCH_DIR
save_dir=${SCRATCH_DIR}/foldseek_af50_aug
file_prefix=$((SGE_TASK_ID - 1))
output_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_aug/${file_prefix}.parquet"
if [ ! -f $output_file ]; then
    python3 data_creation_scripts/foldseek/create_foldseek_clusters_of_clusters.py \
    "${DATA_DIR}/foldseek_af50_struct/index.csv" "${DATA_DIR}/foldseek_af50_aug" \
    --all_vs_all_path "${SCRATCH_DIR}/6-all-vs-all-similarity-queryId_targetId_eValue.tsv" \
    --parquet_id $file_prefix --with_structure
