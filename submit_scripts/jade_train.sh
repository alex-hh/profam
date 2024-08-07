#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:8
#SBATCH --partition=long
#SBATCH --cpus-per-task=5
#SBATCH --ntasks-per-node=8
#SBATCH --output=/jmain02/home/J2AD021/dxt03/axh06-dxt03/ProFam/profam/slurm_logs/slurm_%j.out


module load cuda/12.4
module load python/3.8.6

source ~/pfenv/bin/activate

cd ~/ProFam/profam

WANDB_MODE="offline" HYDRA_FULL_ERROR=1 python src/train.py experiment=main_pfam trainer=ddp trainer.devices=8 +logger.wandb.mode="offline" float32_matmul_precision=null data.num_workers=20