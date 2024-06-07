#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N msaSimCov
#$ -t 100
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
cd $ROOT_DIR
python ${ROOT_DIR}/data_creation/msa_similarity_and_coverage.py --task_index $((SGE_TASK_ID - 1)) --num_tasks 100
date