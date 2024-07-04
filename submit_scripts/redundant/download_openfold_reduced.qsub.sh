#$ -l tmem=7G
#$ -l h_vmem=7G
#$ -l h_rt=95:55:30
#$ -S /bin/bash
#$ -N DLopen1
#$ -t 1
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -j y
date
hostname
/share/apps/aws-cli-tools/v2/current/bin/aws s3 sync s3://openfold/uniclust30/ /SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_filtered/ --no-sign-request --size-only --no-progress
date
