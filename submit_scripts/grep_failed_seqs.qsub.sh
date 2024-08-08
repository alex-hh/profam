#!/bin/bash
#$ -P cath
#$ -l tmem=8G
#$ -l h_vmem=8G
#$ -l h_rt=7:55:30
#$ -S /bin/bash
#$ -N grep_failed
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -j y
date

fail_id_file="/SAN/orengolab/cath_plm/ProFam/data/foldseek/failed_sequences.txt"
uniref_file="/SAN/orengolab/cath_plm/ProFam/data/uniref100/uniref100.fasta"

echo "First few lines of failed_sequences.txt:"
head ${fail_id_file}

echo -e "\nFirst few lines of uniref100.fasta:"
head ${uniref_file}

# grep 100 random sequences from failed_sequences.txt to see if they are in uniref100.fasta
echo -e "\nChecking 100 random sequences from failed_sequences.txt in uniref100.fasta:"

# Get 100 random lines from failed_sequences.txt
random_ids=$(shuf -n 100 ${fail_id_file})

# Loop through each random ID
for id in ${random_ids}; do
    # Search for the ID in uniref100.fasta and store the result
    match=$(grep "^>UniRef100_${id}" ${uniref_file})

    if [ -n "$match" ]; then
        echo "${id}: Found in uniref100.fasta"
        echo "Matching line: ${match}"
    else
        echo "${id}: Not found in uniref100.fasta"
    fi
done
