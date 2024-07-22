#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l h_rt=03:55:30
#$ -S /bin/bash
#$ -N drop
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date
conda activate venvPF
python scripts/drop_redundant_tables.py
date