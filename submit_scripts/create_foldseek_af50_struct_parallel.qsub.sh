#!/bin/bash
#$ -l tmem=48G
#$ -l h_rt=128:55:30
##$ -l gpu_type=(a100|a100_80)
#$ -l gpu=true
#$ -S /bin/bash
#$ -N foldseek_af50_parallel
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -j y
#$ -l avx2=yes  # for foldmason

date
hostname
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
echo "Created scratch dir"

# copy the datasets to the scratch space
# rsync -av /SAN/orengolab/cath_plm/ProFam/profam ${SCRATCH_DIR}/
# cd ${SCRATCH_DIR}/profam
# echo "Copied ProFam to ${SCRATCH_DIR}/profam/"

# export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
# cd $ROOT_DIR
# conda activate venvPF
# source /share/apps/source_files/python/python-3.11.9.source

cd ~/profam
source ~/source_files/afenv.source

ls /scratch0/$USER/$JOB_ID
cp /SAN/bioinf/afdb_domain/zipmaker/zip_index $SCRATCH_DIR/data/
cp /SAN/orengolab/cath_plm/ProFam/data/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv $SCRATCH_DIR/data/

mkdir ${SCRATCH_DIR}/data/foldseek_af50_struct
export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
python3 data_creation_scripts/foldseek/create_foldseek_struct_with_af50.py $1 ${SCRATCH_DIR}/data --minimum_foldseek_cluster_size 1 \
    --parquet_ids $file_prefix --run_foldmason --num_processes 30 --save_dir $SCRATCH_DIR/data/foldseek_af50_struct
rsync -av $SCRATCH_DIR/data/foldseek_af50_struct /SAN/orengolab/cath_plm/ProFam/data/
rm -rf ${SCRATCH_DIR}/data
