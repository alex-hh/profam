#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N splitParquets
#$ -t 1-9
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname

echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PROJECT_ROOT=$ROOT_DIR
cd $ROOT_DIR
conda activate venvPF
echo "Using python from $(which python)"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

case $SGE_TASK_ID in

1)
    ##### SPLIT TED #####
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/topology_splits.json \
    --parquet_dir ../data/ted/s100_parquets \
    --output_dir ../data/ted/s100_parquets/train_val_test_split \
    --splitter CATH
    echo "completed ted s100 split"
    date
    ;;
2)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/topology_splits.json \
    --parquet_dir ../data/ted/s50_parquets \
    --output_dir ../data/ted/s50_parquets/train_val_test_split \
    --splitter CATH
    echo "completed ted s50 split"
    date
    ;;
3)
    ##### SPLIT FUNFAM #####
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/topology_splits.json \
    --parquet_dir ../data/funfams/s100_noali_parquets \
    --output_dir ../data/funfams/s100_noali_parquets/train_val_test_split \
    --splitter CATH
    echo "completed ff s100 unaligned split"
    date
    ;;
4)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/topology_splits.json \
    --parquet_dir ../data/funfams/s50_parquets \
    --output_dir ../data/funfams/s50_parquets/train_val_test_split \
    --splitter CATH
    echo "completed ff s50 aligned split"
    date
    ;;
5)
    ##### SPLIT FOLDSEEK #####
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/foldseek_cath_topology_splits.json \
    --parquet_dir ../data/foldseek_af50/ \
    --output_dir ../data/foldseek_af50/train_val_test_split \
    --splitter FoldSeek
    echo "completed foldseek_af50 split"
    date
    ;;
6)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/foldseek_cath_topology_splits.json \
    --parquet_dir ../data/foldseek_af50_representatives/ \
    --output_dir ../data/foldseek_af50_representatives/train_val_test_split \
    --splitter FoldSeek
    echo "completed foldseek_af50_representatives split"
    date
    ;;
7)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/foldseek_cath_topology_splits.json \
    --parquet_dir ../data/foldseek_af50_struct/ \
    --output_dir ../data/foldseek_af50_struct/train_val_test_split \
    --splitter FoldSeek
    echo "completed foldseek_af50_struct split"
    date
    ;;
8)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/foldseek_cath_topology_splits.json \
    --parquet_dir ../data/foldseek_representatives/ \
    --output_dir ../data/foldseek_representatives/train_val_test_split \
    --splitter FoldSeek
    echo "completed foldseek_representatives split"
    date
    ;;
9)
    python data_creation_scripts/val_test_split/apply_split_to_parquets.py \
    --json_path data/val_test/foldseek_cath_topology_splits.json \
    --parquet_dir ../data/foldseek_struct/ \
    --output_dir ../data/foldseek_struct/train_val_test_split \
    --splitter FoldSeek
    echo "completed foldseek_struct split"
    date
    ;;
*)
    echo "Invalid SGE_TASK_ID: $SGE_TASK_ID"
    ;;
esac