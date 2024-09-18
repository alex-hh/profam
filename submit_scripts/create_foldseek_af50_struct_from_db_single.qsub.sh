#!/bin/bash
#$ -l tmem=24G
#$ -l h_vmem=24G
#$ -l h_rt=38:55:30
#$ -S /bin/bash
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -j y
#$ -l avx2=yes  # for foldmason

date
hostname
file_prefix=$1
output_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/${file_prefix}.parquet"
echo "Checking for output file $output_file"
if [ ! -f $output_file ]; then
    echo "Output file not found: $output_file"
    SCRATCH_DIR=/scratch0/$USER/$JOB_ID/$SGE_TASK_ID
    mkdir -p ${SCRATCH_DIR}/data
    echo "Created scratch dir"
    ls /scratch0/$USER/$JOB_ID
    source /SAN/orengolab/cath_plm/ProFam/pfenv/bin/activate
    export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
    # foldmason can in theory scale to very large alignments (in paper they align clusters of size ~ 100000)
    python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db ${SCRATCH_DIR}/data --minimum_foldseek_cluster_size 1 --parquet_ids $file_prefix --run_foldmason --max_cluster_size_for_foldmason 10000
    rm -rf ${SCRATCH_DIR}/data/
fi
