#!/bin/bash

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a10|a100|a100_80)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N CFnoEnsPoet
#$ -P cath
#$ -o /SAN/orengolab/cath_plm/ProFam/qsub_logs/
#$ -wd /SAN/orengolab/cath_plm/ProFam/profam
#$ -t 1-8
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
FASTADIR=/SAN/orengolab/cath_plm/ProFam/sampling_results/foldseek_combined_val_test_poet_exact_prompts_no_ensemble_2025_10_01/median_only

# Batch selection: choose $BATCHSIZE FASTA entries for this job index ($LINENUM)
# across $NUMTASKS jobs, with no overlap and no misses (contiguous blocks).
BATCHSIZE=16
NUMTASKS=8
LIST_FILE="$FASTADIR/fasta_file_list.txt"
# If the FASTA list file does not exist, create it from files in FASTADIR
if [ ! -f "$LIST_FILE" ]; then
  echo "[INFO] FASTA list file not found; generating: $LIST_FILE"
  if [ ! -d "$FASTADIR" ]; then
    echo "[ERROR] FASTADIR does not exist: $FASTADIR"
    exit 1
  fi
  (
    cd "$FASTADIR" && \
    ls | grep ".fasta" > "fasta_file_list.txt"
  )
fi
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
  OUTPUT_DIR="../sampling_results/colabfold_outputs/foldseek_combined_val_test_poet_exact_prompts_no_ensemble_2025_10_01/${BASENAME_NOEXT}"
  echo "$OUTPUT_DIR"
  mkdir -p "$OUTPUT_DIR"
  # Skip if expected output PDB already exists
  TARGET_PDB="gen0_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb"
  if [ -f "$OUTPUT_DIR/$TARGET_PDB" ]; then
    echo "[INFO] Found existing output, skipping: $OUTPUT_DIR/$TARGET_PDB"
    continue
  fi
  A3M_FILE="$OUTPUT_DIR/gen0.a3m"
  if [ -f "$A3M_FILE" ]; then
    echo "[INFO] Found gen0.a3m; using as input: $A3M_FILE"
    INPUT_PATH="$A3M_FILE"
  else
    INPUT_PATH="$FASTAPATH"
  fi
  colabfold_batch "$INPUT_PATH" "$OUTPUT_DIR" --num-models 1
done
echo
echo ${runstart}
echo "Run ended at: `date`"
echo
