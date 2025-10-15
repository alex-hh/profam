import glob
import os
import sys


def is_idx_file(path):
    if path.endswith(".idx.npy") or path.endswith(".idx.info"):
        return True
    return False


base_dirs = [
    "../data/funfams/s50_text/train_test_split_v2",
    "../data/openfold/uniclust30_clustered_shuffled_final_text/train_test_split_v2",
    "../data/uniref/uniref90_text_shuffled/train_test_split_v2",
    "../data/foldseek/foldseek_s50_seq_only_text/train_test_split_v2",
]  # sys.argv[1]
for base_dir in base_dirs:
    assert os.path.exists(base_dir)
    assert os.path.isdir(base_dir)

    # Walk through all subdirectories and delete .idx files
    patterns = ["**/*.idx.npy", "**/*.idx.info"]
    deleted_count = 0
    for pattern in patterns:
        for file_path in glob.iglob(os.path.join(base_dir, pattern), recursive=True):
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}", file=sys.stderr)

    print(f"Total files deleted: {deleted_count}")
