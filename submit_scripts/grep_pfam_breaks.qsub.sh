#!/bin/bash
#$ -P cath
#$ -l tmem=8G
#$ -l h_rt=11:55:30
#$ -S /bin/bash
#$ -N grepper
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/data/pfam
#$ -j y
grep -n "^//" /SAN/orengolab/cath_plm/ProFam/data/pfam/Pfam-A.full > pfam_end_grepper.txt