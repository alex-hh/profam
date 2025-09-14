#$ -S /bin/bash
#$ -l h_rt=23:00:00
#$ -l tmem=50G
#$ -l gpu=true
#$ -l alphafold=yes
#$ -N alphaGen0
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -j y
#$ -t 1

date
hostname
nvidia-smi
echo "#################### QSUB SCRIPT START ####################"
cat "$0"
echo "####################  QSUB SCRIPT END  ####################"


bash /mnt/disk2/cath_plm/profam/submit_scripts/array_run_alphafold-2.3.1_fast.sh $SGE_TASK_ID

date
