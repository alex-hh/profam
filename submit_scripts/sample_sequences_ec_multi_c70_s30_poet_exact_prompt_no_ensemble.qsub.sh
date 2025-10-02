#!/bin/bash

#$ -l tmem=127G
#$ -l gpu=true
#$ -l gpu_type=(a40|a10|a100|a100_80)
#$ -l h_rt=23:55:30
#$ -S /bin/bash
#$ -N PoetECnoEns
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
python ${ROOT_DIR}/scripts/sample_sequences_from_checkpoint_model.py \
--glob "../data/ec/ec_validation_dataset_clustered_c70_pid_30/poet_exact_prompts_ec_clustered_c70_pid_30/*.fasta" \
--save_dir "../sampling_results/ec_multi_c70_s30_poet_exact_prompts_no_ensemble_2025_10_01" \
--sampler single \
--num_samples 100 \
--max_tokens 8192 \
--task_index $(($SGE_TASK_ID - 1)) \
--seed 42 \
--num_tasks 8 \
--max_sequence_length_multiplier 2 \
--disable_repeat_guard


date