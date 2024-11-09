"""We can upload parquet files to the hub directly."""
import argparse
from huggingface_hub import HfApi


def main(args):
    api = HfApi()
    path_in_repo = f"{args.config_name}/{args.split_name}"
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
        path_in_repo=path_in_repo,
        repo_type="dataset",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("data_dir", type=str)
    # determine repo organization
    # https://huggingface.co/docs/datasets/en/repository_structure
    # makes sense to have a folder for each config and split (config_name/split_name)
    # Your data files may also be placed into different directories named train, test, and validation where each directory contains the data files for that split:
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="train")
    args = parser.parse_args()
    main(args)
