import argparse
import cProfile
import io
import os
import pstats
import time

import hydra
from hydra import compose, initialize_config_dir

from src.constants import BASEDIR


def main(max_iters: int):
    pr = cProfile.Profile()
    pr.enable()

    with initialize_config_dir(os.path.join(BASEDIR, "configs")):
        cfg = compose(
            config_name="train",
            overrides=[
                "experiment=foldseek_inverse_folding",
                "data=foldseek_interleaved",
                "data.num_workers=0",
                f"paths.root_dir={BASEDIR}",
            ],
        )
        tokenizer_cfg = compose(
            config_name="tokenizer/profam",
        )

    tokenizer = hydra.utils.instantiate(tokenizer_cfg.tokenizer)
    dm = hydra.utils.instantiate(cfg.data, tokenizer=tokenizer, _convert_="partial")
    dm.setup()
    train_loader = dm.train_dataloader()

    t_prev = time.time()
    for ix, batch in enumerate(train_loader):
        if ix >= max_iters:
            break
        print(ix, time.time() - t_prev)
        t_prev = time.time()

    pr.disable()  # Stop profiling
    pr.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    print(s.getvalue())

    # Optionally, save profiling results to a file
    ps.dump_stats(
        os.path.join(BASEDIR, "tests", "benchmarks", "dataloader_profile.prof")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("max_iters", type=int)
    args = parser.parse_args()
    main(args.max_iters)
