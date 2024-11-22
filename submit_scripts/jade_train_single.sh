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

echo "Command line arguments" $@

module load cuda/12.4
module load python/3.8.6

source ~/pfenv/bin/activate

cd ~/ProFam/profam

# TODO: copy data to local scratch space
# mkdir /raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data÷
export HF_HUB_OFFLINE=1
scp -r ~/ProFam/data /raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/
WANDB_MODE="offline" HYDRA_FULL_ERROR=1 python src/train.py +environment=jade_single data.data_dir=/raid/local_scratch/$SLURM_JOB_USER/$SLURM_JOB_ID/data "$@"
