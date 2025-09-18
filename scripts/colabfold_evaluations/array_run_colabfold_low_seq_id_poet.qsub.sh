#!/bin/bash

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a10|a100|a100_80)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N CFs50
#$ -P cath
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-10
#$ -j y
date
hostname
echo "#################### QSUB SCRIPT START ####################"
cat "$0" # print the contents of this file to the log
echo "####################  QSUB SCRIPT END  ####################"
conda activate venvPF
ROOT_DIR='/SAN/orengolab/cath_plm/ProFam/profam'
cd $ROOT_DIR
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR

# 1) Load conda & activate the env
source /SAN/orengolab/cath_plm/localcolabfold/conda/etc/profile.d/conda.sh
conda activate /SAN/orengolab/cath_plm/localcolabfold/colabfold-conda

# 2) Make sure runtime/libs & caches are correct for this cluster
unset PYTHONPATH LD_PRELOAD
export XDG_CACHE_HOME=/SAN/orengolab/cath_plm/.cache
export PIP_CACHE_DIR=/SAN/orengolab/cath_plm/.cache/pip
export TMPDIR=/SAN/orengolab/cath_plm/tmp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}"

# 3) Run
colabfold_batch --help   # or: python -m colabfold.batch --help


LINENUM=$SGE_TASK_ID
# FASTADIR=/SAN/orengolab/cath_plm/ProFam/sampling_results/funfam_foldseek_gen0_combined
FASTADIR=/SAN/orengolab/cath_plm/ProFam/sampling_results/poet/poet_foldseek_combined_val_test_single_colabfold_fastas_seq_sim_lt_0p5

# Batch selection: choose $BATCHSIZE FASTA entries for this job index ($LINENUM)
# across $NUMTASKS jobs, with no overlap and no misses (contiguous blocks).
BATCHSIZE=13
NUMTASKS=10
LIST_FILE="$FASTADIR/fasta_file_list.txt"
TOTAL_LINES=$(wc -l < "$LIST_FILE")
START_LINE=$(( (LINENUM - 1) * BATCHSIZE + 1 ))
END_LINE=$(( START_LINE + BATCHSIZE - 1 ))

runstart="Run started at: `date`"

if [ "$START_LINE" -gt "$TOTAL_LINES" ]; then
  echo "[INFO] No FASTAs assigned to job index $LINENUM (start $START_LINE > total $TOTAL_LINES). Exiting."
  echo
  echo ${runstart}
  echo "Run ended at: `date`"
  echo
  exit 0
fi

if [ "$END_LINE" -gt "$TOTAL_LINES" ]; then
  END_LINE=$TOTAL_LINES
fi

echo "[INFO] Job $LINENUM processing FASTAs (lines $START_LINE-$END_LINE of $TOTAL_LINES)."
# ALPHAFOLD COMMAND (modify as necessary):

sed -n "${START_LINE},${END_LINE}p" "$LIST_FILE" | while IFS= read -r REL_FASTA_PATH; do
  FASTAPATH="${FASTADIR}/${REL_FASTA_PATH}"
  echo "$FASTAPATH"
  BASENAME="$(basename "$REL_FASTA_PATH")"
  BASENAME_NOEXT="${BASENAME%%.*}"
  OUTPUT_DIR="../sampling_results/poet/poet_colabfold_outputs_seq_sim_lt_0p5/${BASENAME_NOEXT}"
  echo "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  # Skip if expected output PDB already exists
  TARGET_PDB_GLOB="$OUTPUT_DIR/*_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
  if compgen -G "$TARGET_PDB_GLOB" > /dev/null; then
    echo "[INFO] Found existing output, skipping: $TARGET_PDB_GLOB"
    continue
  fi
  A3M_MATCH=$(compgen -G "$OUTPUT_DIR/*.a3m" | head -n 1)
  if [ -n "$A3M_MATCH" ]; then
    echo "[INFO] Found *.a3m; using as input: $A3M_MATCH"
    INPUT_PATH="$A3M_MATCH"
  else
    INPUT_PATH="$FASTAPATH"
  fi
  colabfold_batch "$INPUT_PATH" "$OUTPUT_DIR" --num-models 1
done
echo
echo ${runstart}
echo "Run ended at: `date`"
echo
