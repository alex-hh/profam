#!/bin/bash

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a10|a100|a100_80)
#$ -l h_rt=47:55:30
#$ -S /bin/bash
#$ -N ECmultiSample
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
python ${ROOT_DIR}/scripts/sample_sequences_from_checkpoint_model.py \
--glob "../data/ec/ec_validation_dataset/alignments/*aln.filtered.fasta" \
--save_dir ../sampling_results/profam_ec_multi_seq_synthetic_msas \
--sampler ensemble \
--num_samples 50 \
--num_variants 8 \
--task_index $(($SGE_TASK_ID - 1)) \
--num_tasks 10

date