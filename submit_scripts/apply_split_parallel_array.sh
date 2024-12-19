#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=72:55:30
#$ -S /bin/bash
#$ -N $JOB_NAME
#$ -t 1-20
#$ -P cath
#$ -j y
#$ -R y
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/splitting/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam

date
hostname

umask 002

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PROJECT_ROOT=$ROOT_DIR
cd $ROOT_DIR
source /SAN/orengolab/plm_embeds/cache/miniconda3/miniconda3/bin/activate profam
echo "Using python from $(which python)"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

export DATA_DIR='/SAN/orengolab/cath_plm/ProFam/data'

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
    echo "ARRAY_ID: $ARRAY_ID"
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --parquet_dir ${DATA_DIR}/afdb_s50_single/ \
        --output_dir ${DATA_DIR}/afdb_s50_single/train_val_test_split \
        --splitter FoldSeek_AF50 \
        --paral_index ${ARRAY_ID}
        echo "completed afdb_s50_single split"
        date
        ;;
2)
    # FoldSeek_Struct
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_struct_todo | tail -n 1)
    echo "ARRAY_ID: $ARRAY_ID"
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --parquet_dir ${DATA_DIR}/foldseek/foldseek_s50_struct/ \
        --output_dir ${DATA_DIR}/foldseek/foldseek_s50_struct/train_val_test_split \
        --splitter FoldSeek \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_s50_struct"
        date
        ;;
3)
    # FoldSeek_s100_Struct
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_af50_struct_todo | tail -n 1)
    echo "ARRAY_ID: $ARRAY_ID"
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --parquet_dir ${DATA_DIR}/foldseek/foldseek_s100_struct/ \
        --output_dir ${DATA_DIR}/foldseek/foldseek_s100_struct/train_val_test_split \
        --splitter FoldSeek \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_s100_struct split"
        date
        ;;
4)
    # Foldseek_Representatives
    ARRAY_ID=$(head -n $SGE_TASK_ID ${ROOT_DIR}/submit_scripts/redundant_scripts/foldseek_representatives_todo | tail -n 1)
    echo "ARRAY_ID: $ARRAY_ID"
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
        --parquet_dir ${DATA_DIR}/foldseek/foldseek_reps_single \
        --output_dir ${DATA_DIR}/foldseek/foldseek_reps_single/train_val_test_split \
        --splitter FoldSeek \
        --paral_index ${ARRAY_ID}
        echo "completed foldseek_reps_single split"
        date
        ;;
*)
    echo "Invalid SGE_TASK_ID: $SGE_TASK_ID"
    ;;
esac

# Usage:
# qsub -N Foldseek_af50_representatives_splitParquets -v CASE_ID=1 apply_split_parallel_array.sh
# qsub -N FSs50struct -v CASE_ID=2 apply_split_parallel_array.sh
# qsub -N FSs100struct -v CASE_ID=3 apply_split_parallel_array.sh
# qsub -N FSrepsSingle -v CASE_ID=4 apply_split_parallel_array.sh
