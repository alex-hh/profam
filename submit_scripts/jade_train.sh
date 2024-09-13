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

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

mkdir /raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data
scp -r ~/ProFam/data/pfam /raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data/
scp -r ~/ProFam/data/ProteinGym /raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data/
WANDB_MODE="offline" HYDRA_FULL_ERROR=1 srun python src/train.py +environment=jade paths.data_dir=/raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data $@
# https://github.com/acherstyx/hydra-torchrun-launcher
# WANDB_MODE="offline" HYDRA_FULL_ERROR=1 python src/train.py -m hydra/launcher=torchrun hydra.launcher.nproc_per_node=8 experiment=main_pfam +environment=jade data.data_dir=/raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data
