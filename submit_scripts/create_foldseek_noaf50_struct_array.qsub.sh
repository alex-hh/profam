#!/bin/bash
#$ -l tmem=10G
#$ -l h_vmem=10G
#$ -l h_rt=25:55:30
#$ -S /bin/bash
#$ -t 1-1  # 8000 for 2000000 at 250; 1800/4500 for 450000 at 250/100
#$ -N foldseekF
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd ~/profam
#$ -tc 100
#$ -j y
date
# conda activate venvPF
# n.b. if we set minimum foldseek cluster size to 1, we might need more memory
SCRATCH_DIR=/scratch0/$USER/$JOB_ID
mkdir -p ${SCRATCH_DIR}/data
source ~/source_files/afenv.source
export PATH=/SAN/orengolab/cath_plm/ProFam/foldmason/bin/:$PATH
python3 data_creation_scripts/create_foldseek_struct_with_af50.py $1 /SAN/orengolab/cath_plm/ProFam/data/foldseek_struct_example  --skip_af50 --num_processes 1 --minimum_foldseek_cluster_size 1 --parquet_ids $((SGE_TASK_ID - 1)) --run_foldmason
# TODO: zip the scratch dir?, remove scratch dir
rm -rf ${SCRATCH_DIR}/data
date
