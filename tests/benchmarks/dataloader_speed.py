"""
Relevant issues:
 - shuffling an iterable dataset causes slowdown: https://github.com/huggingface/datasets/issues/7102
 - transforms remove formatting: https://github.com/huggingface/datasets/issues/6833, https://github.com/huggingface/datasets/issues/5864
 - batched iteration and feature types are essential for performance: https://github.com/huggingface/datasets/issues/5841
 - making batched iteration compatible with pytorch data loader: https://github.com/huggingface/datasets/pull/7054
    -> we want to do something along these lines for sequence packing.

TODO: configure this script somehow to allow benchmarking different configurations.
"""
import argparse
import cProfile
import io
import os
import pstats
import time

import hydra
from hydra import compose, initialize_config_dir

from src.constants import BASEDIR


def main(max_iters: int, loader_type: str, data_folder: str):
    pr = cProfile.Profile()
    pr.enable()

    with initialize_config_dir(os.path.join(BASEDIR, "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=foldseek_inverse_folding",
                "data=foldseek_interleaved",
                "data.num_workers=0",
                f"data.dataset_cfgs.foldseek.holdout_data_files=null",
                f"data.dataset_cfgs.foldseek.data_path_pattern={data_folder}/*.parquet",
                f"paths.root_dir={BASEDIR}",
            ],
        )
        tokenizer_cfg = compose(
            config_name="tokenizer/profam",
        )

    # TODO: look into batched iteration: https://github.com/huggingface/datasets/blob/3.0.1/src/datasets/iterable_dataset.py#L2844
    tokenizer = hydra.utils.instantiate(tokenizer_cfg.tokenizer)
    dm = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer, _convert_="partial")
    dm.setup()
    if loader_type == "loader":
        print("Loading from dataloader")
        train_loader = dm.train_dataloader()
    else:
        print("Loading from dataset directly")
        train_loader = dm.train_dataset

    t_0 = time.time()
    for ix, batch in enumerate(train_loader):
        if ix * dm.batch_size >= max_iters:
            break
    t_1 = time.time()
    print(f"Total iteration time: {t_1 - t_0:.4f} seconds")

    pr.disable()  # Stop profiling
    pr.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())

    # Optionally, save profiling results to a file
    ps.dump_stats(
        os.path.join(
            BASEDIR, "tests", "benchmarks", f"{data_folder}_dataloader_profile.prof"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--loader_type", choices=["loader", "dataset"])
    parser.add_argument("--data_folder", type=str, default="foldseek_struct")
    args = parser.parse_args()
    main(args.max_iters, args.loader_type, args.data_folder)
