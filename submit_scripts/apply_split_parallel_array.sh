#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N Foldseek_af50_splitParquets
#$ -t 1-20
#$ -P cath
#$ -j y
#$ -R y
#$ -cwd
date
hostname

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
export ROOT_DIR='/SAN/orengolab/plm_embeds/profam'
export PROJECT_ROOT=$ROOT_DIR
cd $ROOT_DIR
source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam
echo "Using python from $(which python)"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

export DATA_DIR='/SAN/orengolab/cath_plm/ProFam'

ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_af50_todo | tail -n 1)

python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path ${DATA_DIR}/data/ted/ted_esmif_accessions_split.json \
    --parquet_dir ${DATA_DIR}/data/foldseek_af50_representatives/ \
    --output_dir ${DATA_DIR}/data/foldseek_af50_representatives/train_val_test_split \
    --splitter FoldSeek_AF50 \
    --split_dataset_id af50_cluster_id \
    --paral_index ${ARRAY_ID}
    echo "completed foldseek_af50_representatives split"
    date
