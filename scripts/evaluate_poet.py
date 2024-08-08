import argparse
import glob
import os
from hydra import compose, initialize_config_dir
from src.constants import BASEDIR
from datasets import load_dataset


def main(args):
    with initialize_config_dir(os.path.join(BASEDIR, "configs/data/dataset")):
        cfg = compose(config_name=args.dataset)  # for example
    if cfg.data_path_pattern is not None:
        # replace hf path resolution with manual glob, to allow repetition
        # https://github.com/huggingface/datasets/blob/98fdc9e78e6d057ca66e58a37f49d6618aab8130/src/datasets/data_files.py#L323
        data_files = glob.glob(os.path.join(data_dir, cfg.data_path_pattern))
    else:
        assert cfg.data_path_file is not None
        with open(os.path.join(data_dir, cfg.data_path_file), "r") as f:
            data_files = [
                os.path.join(data_dir, data_file) for data_file in f.read().splitlines()
            ]

    assert isinstance(data_files, list)
    data_files = data_files * cfg.file_repeats
    print(
        f"Loading {cfg.name} dataset from {len(data_files)} files, "
        f"({cfg.file_repeats} repeats), "
        f"{os.path.join(data_dir, cfg.data_path_pattern)}"
    )
    if cfg.is_parquet:
        dataset = load_dataset(
            path="parquet",
            data_files=data_files,
            split="train",
            streaming=True,
            ignore_verifications=True,
        )
    else:
        dataset = load_dataset(
            "text",
            data_files=data_files,
            split="train",
            streaming=True,
            sample_by="document",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    main(args)
