#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5

# set name of job
#SBATCH --job-name=job123

#SBATCH --partition=long
# set number of GPUs
#SBATCH --gres=gpu:1
#SBATCH --output=/jmain02/home/J2AD021/dxt03/axh06-dxt03/ProFam/profam/slurm_logs/slurm_%j.out


module load cuda/12.4
module load python/3.8.6

source ~/pfenv/bin/activate

cd ~/ProFam/profam

# TODO: copy data to local scratch space
WANDB_MODE="offline" HYDRA_FULL_ERROR=1 python src/train.py experiment=main_pfam trainer=gpu logger=stdout float32_matmul_precision=null trainer.precision=32 data.num_workers=5
