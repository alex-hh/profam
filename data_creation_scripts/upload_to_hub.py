"""We can upload parquet files to the hub directly.

Use hf_transfer: this is a Rust-based library meant to speed up uploads on machines with very high bandwidth. To use hf_transfer:
Specify the hf_transfer extra when installing huggingface_hub (i.e., pip install huggingface_hub[hf_transfer]).
Set HF_HUB_ENABLE_HF_TRANSFER=1 as an environment variable.
"""
import argparse
from huggingface_hub import HfApi


def main(args):
    # https://huggingface.co/docs/huggingface_hub/en/guides/upload#upload-a-folder-by-chunks
    api = HfApi()
    path_in_repo = f"{args.config_name}/{args.split_name}"
    # TODO: use upload_large_folder (requires correct local directory structure since path_in_repo not supported)
    api.upload_large_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    # api.upload_folder(
    #     folder_path=args.data_dir,
    #     repo_id=args.repo_id,
    #     path_in_repo=path_in_repo,
    #     repo_type="dataset",
    #     multi_commits=True,
    # )


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
