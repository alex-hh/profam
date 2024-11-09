"""We can upload parquet files to the hub directly."""
import argparse
from huggingface_hub import HfApi


def main(args):
    api = HfApi()
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
