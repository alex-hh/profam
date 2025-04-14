# shuffles openfold parquets in chunks by iterative shuffle, split, random recombine, shuffle, repeat
# number of files after each stage:
# 3866 -> 25k -> 2 -> 2048

python data_creation_scripts/shuffling/split_shuffle.py \
../data/openfold/uniclust30_clustered \
../data/openfold/uniclust30_clustered_shuffled_stage1_simple \
--task_index 0 \
--num_tasks 1 \
--chunks_per_file 2 && \

python data_creation_scripts/shuffling/allocate_sub_files.py \
../data/openfold/uniclust30_clustered_shuffled_stage1_simple \
../data/openfold/uniclust30_clustered_shuffled_stage1_simple/allocation.json \
2 && \

python data_creation_scripts/shuffling/recombine.py \
../data/openfold/uniclust30_clustered_shuffled_stage1_simple/allocation.json \
../data/openfold/uniclust30_clustered_shuffled_from_2_files \
--task_index 0 \
--num_tasks 1 && \


python data_creation_scripts/shuffling/split_shuffle.py \
../data/openfold/uniclust30_clustered_shuffled_from_2_files \
../data/openfold/uniclust30_clustered_shuffled_final \
--task_index 0 \
--num_tasks 1 \
--chunks_per_file 1024 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_from_2_files

