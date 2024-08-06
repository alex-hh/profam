#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=10:00:00

# set name of job
#SBATCH --job-name=job123

# set number of GPUs
#SBATCH --gres=gpu:8
#SBATCH --output=/jmain02/home/J2AD021/dxt03/axh06-dxt03/ProFam/profam/slurm_logs/slurm_%j.out
#SBATCH --error=/jmain02/home/J2AD021/dxt03/axh06-dxt03/ProFam/profam/slurm_logs/slurm_%j.err


module load cuda/12.4
module load python/3.8.6

source ~/pfenv/bin/activate

cd ~/ProFam/profam

WANDB_MODE="offline" HYDRA_FULL_ERROR=1 python src/train.py experiment=main_pfam trainer=ddp trainer.devices=8