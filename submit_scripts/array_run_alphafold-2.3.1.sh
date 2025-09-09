#!/bin/bash

# Command script to run alphafold 2.3.1.
# David Gregory 09/02/23.

# To run alphafold 2.3.1 in multimer mode, modifiy the alphafold command below
# to include the options under MULTIMER and remove the options under MONOMER.
# To run alphafold 2.3.1 with its previous (single-chain) behaviour,
# include the options under MONOMER and remove the options under MULTIMER.

# Run the run_alphafold.py script with "--help",
# to see a description of the options available.

# MULTIMER:
# --model_preset=multimer:
# --pdb_seqres_database_path=${AF_DATA}/pdb_seqres/pdb_seqres.txt
# --uniprot_database_path=${AF_DATA}/uniprot/uniprot.fasta
# --is_prokaryote_list=true|false [either "true" or "false"]

# MONOMER:
# --model_preset=monomer:
# --pdb70_database_path=${AF_DATA}/pdb70/pdb70

source /share/apps/AlphaFold2/alphafold-2.3.1_env.source
LINENUM=$1
FASTADIR=/SAN/bioinf/domdet/alphaf/fastas/casp6_test_fasta
FASTAPATH="$FASTADIR/$(sed -n ${LINENUM}p $FASTADIR/fasta_file_list.txt)"
runstart="Run started at: `date`"
echo $FASTAPATH
# ALPHAFOLD COMMAND (modify as necessary):

python3 ${AF_INST}/run_alphafold.py \
   --bfd_database_path=${AF_DATA}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
   --data_dir=${AF_DATA} \
   --fasta_paths=$FASTAPATH \
   --hhblits_binary_path=${AF_DEPS}/hh-suite-3.3.0/bin/hhblits \
   --hhsearch_binary_path=${AF_DEPS}/hh-suite-3.3.0/bin/hhsearch \
   --hmmbuild_binary_path=${AF_DEPS}/hmmer-3.3.2/bin/hmmbuild \
   --hmmsearch_binary_path=${AF_DEPS}/hmmer-3.3.2/bin/hmmsearch \
   --jackhmmer_binary_path=${AF_DEPS}/hmmer-3.3.2/bin/jackhmmer \
   --kalign_binary_path=${AF_DEPS}/kalign-3.3.1/bin/kalign \
   --mgnify_database_path=${AF_DATA}/mgnify/mgy_clusters_2022_05.fa \
   --model_preset=monomer \
   --pdb70_database_path=${AF_DATA}/pdb70/pdb70 \
   --obsolete_pdbs_path=${AF_DATA}/pdb_mmcif/obsolete.dat \
   --output_dir=/SAN/bioinf/domdet/alphaf/outputs2025 \
   --template_mmcif_dir=${AF_DATA}/pdb_mmcif/mmcif_files \
   --uniref30_database_path=${AF_DATA}/uniref30/UniRef30_2021_03 \
   --uniref90_database_path=${AF_DATA}/uniref90/uniref90.fasta \
   --max_template_date=2023-02-14 \
   --run_relax=False \
   --use_gpu_relax=False
echo
echo ${runstart}
echo "Run ended at: `date`"
echo
