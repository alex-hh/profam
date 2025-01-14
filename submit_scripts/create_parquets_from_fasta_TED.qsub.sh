#!/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N tedRParquets
#$ -t 1-5
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "qsub script: $0"
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"

conda activate venvPF

ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
SAVE_DIR="/SAN/orengolab/cath_plm/ProFam/data/ted/s100_parquets"
FASTA_GLOB_PATTERN="/cluster/project9/afdb_domain_ext/results/cath_gene3d_hits/ted_domain_sequences_by_sfam/*.fasta"
DS_NAME="ted_100"

cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR
echo "Number of tasks: $SGE_TASK_LAST"
python ${ROOT_DIR}/data_creation_scripts/create_parquets_from_fasta.py \
--task_index $((SGE_TASK_ID - 1)) \
--num_tasks $SGE_TASK_LAST \
--save_dir $SAVE_DIR \
--fasta_glob_pattern "$FASTA_GLOB_PATTERN" \
--ds_name $DS_NAME
date

SAVE_SOURCE_PATH=${SAVE_DIR}/parquet_creation_source_code.txt
# if SGE_TASK_ID=1, then save the contents of this script and python script to SAVE_SOURCE_PATH
if [ "$SGE_TASK_ID" -eq 1 ]; then
    echo "##### start #####" > $SAVE_SOURCE_PATH
    cat "$0" >> $SAVE_SOURCE_PATH
    echo "##### Python script #####" >> $SAVE_SOURCE_PATH
    cat "${ROOT_DIR}/data_creation_scripts/create_parquets_from_fasta.py" >> $SAVE_SOURCE_PATH
fi
