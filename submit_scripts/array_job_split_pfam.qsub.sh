#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=3:55:30
#$ -S /bin/bash
#$ -N V3arrPfam
#$ -t 1-50
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

python