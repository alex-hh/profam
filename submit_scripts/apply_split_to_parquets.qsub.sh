#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N splitParquets
#$ -t 1
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

python data_creation_scripts/val_test_split/apply_split_to_parquets.py