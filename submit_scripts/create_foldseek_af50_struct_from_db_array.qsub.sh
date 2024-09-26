#!/bin/bash
#$ -l tmem=12G
#$ -l h_vmem=12G
#$ -l h_rt=38:55:30
#$ -S /bin/bash
#$ -t 1-10000  # 8000 for 2000000 at 250; 1800/4500 for 450000 at 250/100
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/ProFam/ahh/profam
#$ -tc 1000
#$ -j y
#$ -l avx2=yes  # for foldmason
##$ -l tscratch=5G

date
hostname
file_prefix=$((SGE_TASK_ID - 1))
source /share/apps/source_files/python/python-3.11.9.source
source /SAN/orengolab/cath_plm/ProFam/pfenv/bin/activate
export PROFAM_DATA_DIR=/SAN/orengolab/cath_plm/ProFam/data
output_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek_af50_struct/${file_prefix}.parquet"
if [ ! -f $output_file ]; then
    echo "Output file not found: $output_file"
    SCRATCH_DIR=/scratch0/$USER/$JOB_ID/$SGE_TASK_ID
    mkdir -p ${SCRATCH_DIR}/data
    echo "Created scratch dir"
    ls /scratch0/$USER/$JOB_ID
    export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
    # foldmason can in theory scale to very large alignments (in paper they align clusters of size ~ 100000)
    python3 -m data_creation_scripts.foldseek.create_foldseek_struct_from_db ${SCRATCH_DIR}/data --minimum_foldseek_cluster_size 1 --parquet_ids $file_prefix --run_foldmason --max_cluster_size_for_foldmason 10000
    rm -rf ${SCRATCH_DIR}/data/
fi
