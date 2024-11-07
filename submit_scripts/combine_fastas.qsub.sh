#!/bin/bash

#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=92:55:30
#$ -S /bin/bash
#$ -N fastaCombi
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y

date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "#################### QSUB SCRIPT END  ####################"

base_dir="/SAN/orengolab/cath_plm/ProFam/data/uniprot"
# Define input and output paths
input1="${base_dir}/uniprot_sprot.fasta.gz"
input2="${base_dir}/uniprot_trembl.fasta.gz"
output="${base_dir}/combined_sprot_trembl_uniprot.fasta"

# Unzip, extract accessions, and combine both files
{
    zcat "$input1" | awk '/^>/{acc_pos=index(substr($1,5), "|"); print ">" substr($1,5,acc_pos-1)} !/^>/{print}'
    zcat "$input2" | awk '/^>/{acc_pos=index(substr($1,5), "|"); print ">" substr($1,5,acc_pos-1)} !/^>/{print}'
} > "$output"

echo "Combined FASTA file created: $output"