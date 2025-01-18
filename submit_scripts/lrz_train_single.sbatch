#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5

# set name of job
#SBATCH --job-name=profam

# set number of GPUs
#SBATCH --gres=gpu:1

#SBATCH -o file_to_redirect_the_std_output.out
#SBATCH -e file_to_redirect_the_std_err.err

srun \
	--container-mounts=/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0001/ga32bav2:/dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0001/ga32bav2 \
       	--container-image='/dss/dsshome1/09/ga32bav2/profam_pytorch.sqsh' \
	bash -c "cd /dss/dssfs04/lwp-dss-0002/pr63ci/pr63ci-dss-0001/ga32bav2/profam/profam/ ; source pfenv/bin/activate ; python src/train.py experiment=train_single_seq data.pack_to_max_tokens=64000 data.dataset_builders.afdb_s50.preprocessor.cfg.max_tokens_per_example=320 data.dataset_builders.afdb_s50.preprocessor.cfg.max_tokens_per_example=320 data.num_workers=5 trainer=gpu trainer.accelerator=gpu trainer.devices=1"
