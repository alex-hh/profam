#!/bin/bash
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=38:55:30
#$ -S /bin/bash
#$ -t 1-10000  # 8000 for 2000000 at 250; 1800/4500 for 450000 at 250/100
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -tc 1000
#$ -j y
#$ -l avx2=yes  # for foldmason

date
hostname
file_prefix=$((SGE_TASK_ID - 1))
output_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek_representatives/${file_prefix}.parquet"
if [ ! -f $output_file ]; then
    echo "Output file not found: $output_file"
    SCRATCH_DIR=/scratch0/$USER/$JOB_ID
    mkdir -p ${SCRATCH_DIR}/data
    echo "Created scratch dir"
    ls /scratch0/$USER/$JOB_ID
    source /SAN/orengolab/cath_plm/ProFam/pfenv/bin/activate
    export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
    python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db ${SCRATCH_DIR}/data --minimum_foldseek_cluster_size 1 --parquet_ids $file_prefix --representative_only
    rm -rf ${SCRATCH_DIR}/data
fi
