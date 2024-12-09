#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N $JOB_NAME
#$ -t 1-20
#$ -P cath
#$ -j y
#$ -R y
#$ -cwd
date
hostname

umask 002

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

if [ -z "$CASE_ID" ]; then
    echo "CASE_ID not set. Exiting."
    exit 1
else
    echo "Running with CASE_ID=$CASE_ID"
fi

case $CASE_ID in
1)
    # Foldseek_AF50_Representatives
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_af50_representatives_todo | tail -n 1)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --json_path ${DATA_DIR}/data/ted/ted_esmif_accessions_split.json \
        --parquet_dir ${DATA_DIR}/data/foldseek_af50_representatives/ \
        --output_dir ${DATA_DIR}/data/foldseek_af50_representatives/train_val_test_split \
        --splitter FoldSeek_AF50 \
        --split_dataset_id af50_cluster_id \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_af50_representatives split"
        date
        ;;
2)
    # FoldSeek_Struct
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_struct | tail -n 1)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --json_path ${DATA_DIR}/profam/data/val_test/foldseek_cath_topology_splits.json \
        --parquet_dir ${DATA_DIR}/data/foldseek_struct/ \
        --output_dir ${DATA_DIR}/data/foldseek_struct/train_val_test_split \
        --splitter FoldSeek \
        --split_dataset_id fam_id \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_struct split"
        date
        ;;
3)
    # FoldSeek_AF50_Struct
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_af50_struct | tail -n 1)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --json_path ${DATA_DIR}/profam/data/val_test/foldseek_cath_topology_splits.json \
        --parquet_dir ${DATA_DIR}/data/foldseek_af50_struct/ \
        --output_dir ${DATA_DIR}/data/foldseek_af50_struct/train_val_test_split_array \
        --splitter FoldSeek \
        --split_dataset_id fam_id \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_struct split"
        date
        ;;
4)
    # Foldseek_Representatives
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_representatives | tail -n 1)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --json_path ${DATA_DIR}/profam/data/val_test/foldseek_cath_topology_splits.json \
        --parquet_dir ${DATA_DIR}/data/foldseek_representatives/ \
        --output_dir ${DATA_DIR}/data/foldseek_representatives/train_val_test_split \
        --splitter FoldSeek \
        --split_dataset_id fam_id \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_representatives split"
        date
        ;;
*)
    echo "Invalid SGE_TASK_ID: $SGE_TASK_ID"
    ;;
esac

# Usage:
# qsub -N Foldseek_af50_representatives_splitParquets -v CASE_ID=1 apply_split_parallel_array.sh
# qsub -N Foldseek_struct_splitParquets -v CASE_ID=2 apply_split_parallel_array.sh
# qsub -N Foldseek_af50_struct_splitParquets -v CASE_ID=3 apply_split_parallel_array.sh
# qsub -N Foldseek_representatives_splitParquets -v CASE_ID=4 apply_split_parallel_array.sh
