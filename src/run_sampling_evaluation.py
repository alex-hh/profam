import gc
import os

import hydra
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf

from src.constants import BASEDIR
from src.evaluators.base import SamplingEvaluator

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


# if this config path doesnt work, use basedir
@hydra.main(
    version_base="1.3",
    config_path=os.path.join(BASEDIR, "configs"),
    config_name="sampling_eval",
)
def run(cfg: DictConfig) -> None:
    """Things that might not fit straightforwardly in generation validation:
    * perplexity
    * effective sample size

    Run a single model + validation combination. To run multiple validations, use multirun.
    """
    print(OmegaConf.to_yaml(cfg))  # useful for spotting any possible errors

    evaluators = hydra.utils.instantiate(cfg.evaluators)
    if isinstance(evaluators, dict) or isinstance(evaluators, DictConfig):
        evaluator_names = list(evaluators.keys())
        evaluators = list(evaluators.values())
    else:
        assert isinstance(evaluators, SamplingEvaluator)
        evaluator_names = [evaluators.name]
        evaluators = [evaluators]

    print(
        f"Running validation {evaluator_names} on generator {cfg.sampler.name}"
        f" on data from pipeline {cfg.pipeline.pipeline_id}"
    )

    pipeline = hydra.utils.instantiate(cfg.pipeline)
    sampler = hydra.utils.instantiate(cfg.sampler, _convert_="partial")
    # TODO: save sampler config to results directory and verify that it matches on rerun.
    pipeline.run(
        sampler,
        evaluators=evaluators,
        rerun_sampler=cfg.rerun_sampler,
        rerun_evaluator=cfg.rerun_evaluator,
        device=cfg.device,
    )
    # for hydra multirun memory management issues:
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
