#$ -l tmem=256G
#$ -l h_vmem=256G
#$ -l h_rt=11:55:30
#$ -S /bin/bash
#$ -N mmseqsFULL
#$ -t 1
#$ -o /SAN/bioinf/VoxelDiffOuter/cath_plm/profam/qsub_logs/
#$ -wd /SAN/bioinf/VoxelDiffOuter/cath_plm/
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
ROOT_DIR='/SAN/bioinf/VoxelDiffOuter/cath_plm/data/seq_dbs/uniref50'
cd ${ROOT_DIR}
DB_DIR='uniref50'
mmseqs createdb ${ROOT_DIR}/uniref50.fasta ${ROOT_DIR}/${DB_DIR}
echo "Database created (createdb command completed)"
mmseqs createindex ${ROOT_DIR}/${DB_DIR} ${ROOT_DIR}/tmp --split-memory-limit 25G
echo "Database created and indexed"
date
# test how long it takes to run mmseqs easy-search
mmseqs easy-search /SAN/bioinf/VoxelDiffOuter/cath_plm/data/single_protein.fasta ${ROOT_DIR}/${DB_DIR} search_single_prot_uniref50.m8 tmp --split-memory-limit 25G
echo "Search complete"
date
