#$ -l tmem=7G
#$ -l h_vmem=7G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N combineParquets
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
hostname

echo "SGE script: $0" # print the path to this file
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
# export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
export PROJECT_ROOT=$ROOT_DIR
cd $ROOT_DIR
conda activate venvPF
echo "Using python from $(which python)"
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH


python data_creation_scripts/combine_parquet_files_in_directory.py \
    --parquet_dir ../data/openfold/uniclust30_clustered_shuffled_final/train_test_split_v2/val_filtered \
    --max_residue_per_file 20_000_000

python data_creation_scripts/combine_parquet_files_in_directory.py \
    --parquet_dir ../data/openfold/uniclust30_clustered_shuffled_final/train_test_split_v2/test_filtered \
    --max_residue_per_file 20_000_000
