# shuffles openfold parquets in chunks by iterative shuffle, split, random recombine, shuffle, repeat
# number of files after each stage:
# 3866 -> 25k -> 10 -> 10_000 -> 10 -> 2048

python data_creation_scripts/shuffling/split_shuffle.py \
../data/openfold/uniclust30_clustered \
../data/openfold/uniclust30_clustered_shuffled_stage1 \
--task_index 0 \
--num_tasks 1 \
--chunks_per_file 6 && \

python data_creation_scripts/shuffling/allocate_sub_files.py \
../data/openfold/uniclust30_clustered_shuffled_stage1 \
../data/openfold/uniclust30_clustered_shuffled_stage1/allocation.json \
10 && \

python data_creation_scripts/shuffling/recombine.py \
../data/openfold/uniclust30_clustered_shuffled_stage1/allocation.json \
../data/openfold/uniclust30_clustered_shuffled_stage2 \
--task_index 0 \
--num_tasks 1 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_stage1 && \

python data_creation_scripts/shuffling/split_shuffle.py \
../data/openfold/uniclust30_clustered_shuffled_stage2 \
../data/openfold/uniclust30_clustered_shuffled_stage3 \
--task_index 0 \
--num_tasks 1 \
--chunks_per_file 1000 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_stage2 && \

python data_creation_scripts/shuffling/allocate_sub_files.py \
../data/openfold/uniclust30_clustered_shuffled_stage3 \
../data/openfold/uniclust30_clustered_shuffled_stage3/allocation.json \
10 && \

python data_creation_scripts/shuffling/recombine.py \
../data/openfold/uniclust30_clustered_shuffled_stage3/allocation.json \
../data/openfold/uniclust30_clustered_shuffled_stage4 \
--task_index 0 \
--num_tasks 1 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_stage3 && \

python data_creation_scripts/shuffling/split_shuffle.py \
../data/openfold/uniclust30_clustered_shuffled_stage4 \
../data/openfold/uniclust30_clustered_shuffled_stage5 \
--task_index 0 \
--num_tasks 1 \
--chunks_per_file 1000 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_stage4 && \

python data_creation_scripts/shuffling/allocate_sub_files.py \
../data/openfold/uniclust30_clustered_shuffled_stage5 \
../data/openfold/uniclust30_clustered_shuffled_stage5/allocation.json \
2048 && \

python data_creation_scripts/shuffling/recombine.py \
../data/openfold/uniclust30_clustered_shuffled_stage5/allocation.json \
../data/openfold/uniclust30_clustered_shuffled_stage6 \
--task_index 0 \
--num_tasks 1 && \

rm -rf ../data/openfold/uniclust30_clustered_shuffled_stage5




