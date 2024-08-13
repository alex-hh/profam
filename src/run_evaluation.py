import gc
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.constants import BASEDIR


# if this config path doesnt work, use basedir
@hydra.main(
    version_base="1.3",
    config_path=os.path.join(BASEDIR, "configs"),
    config_name="benchmark",
)
def run(cfg: DictConfig) -> None:
    """Things that might not fit straightforwardly in generation validation:
    * perplexity
    * effective sample size

    Run a single model + validation combination. To run multiple validations, use multirun.
    """
    print(OmegaConf.to_yaml(cfg))  # useful for spotting any possible errors
    print(
        f"Running validation {cfg.validation.name} on generator {cfg.generator.name}"
        f" on data from pipeline {cfg.pipeline.pipeline_id}"
    )
    # get fsspec filesystem based on environment variables
    fs = get_fs()
    # instantiate evaluator
    evaluator = hydra.utils.instantiate(cfg.evaluator)
    pipeline = hydra.utils.instantiate(cfg.pipeline, evaluator=evaluator)
    model = hydra.utils.instantiate(cfg.pretrained_model)
    pipeline.run(
        model, rerun_model=cfg.rerun_model, rerun_evaluator=cfg.rerun_evaluator
    )
    # for hydra multirun memory management issues:
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
