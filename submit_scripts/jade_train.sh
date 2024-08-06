#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:8
module load cuda/12.4
module load python/3.8.6

source ~/pfenv/bin/activate

cd ~/ProFam/profam

HYDRA_FULL_ERROR=1 python src/train.py experiment=main_pfam