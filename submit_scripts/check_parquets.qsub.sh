#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N parqCheck
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python scripts/check_parquets.py
date
