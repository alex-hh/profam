#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=71:55:30
#$ -S /bin/bash
#$ -N valCount
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
git checkout -b val_test_overlap
python ${ROOT_DIR}/data_creation_scripts/count_all_doc_overlaps_w_pfam_val_test.py
date
