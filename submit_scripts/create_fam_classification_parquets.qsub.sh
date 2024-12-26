#$ -l tmem=64G
#$ -l h_vmem=64G
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N makeFamClassParqs
#$ -t 1
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
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
cd $ROOT_DIR
python ${ROOT_DIR}/data_creation_scripts/create_fam_classification_parquets.py
date
