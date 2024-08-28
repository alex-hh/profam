#!/bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=25:55:30
#$ -S /bin/bash
#$ -t 1-4500  # 8000 for 2000000 at 250; 1800/4500 for 450000 at 250/100
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -tc 1000
#$ -j y
date
# conda activate venvPF
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
cp /SAN/bioinf/afdb_domain/zipmaker/zip_index $SCRATCH_DIR/data/
cp /SAN/orengolab/cath_plm/ProFam/data/afdb/5-allmembers-repId-entryId-cluFlag-taxId.tsv $SCRATCH_DIR/data/
source ~/source_files/afenv.source
python3 data_creation_scripts/create_foldseek_struct_with_af50.py $1 ${SCRATCH_DIR}/data --num_processes 1 --minimum_foldseek_cluster_size 1 --parquet_ids $((SGE_TASK_ID - 1)) --run_foldmason
rm -rf ${SCRATCH_DIR}/data
date
