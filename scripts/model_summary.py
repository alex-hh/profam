from typing import Optional

import hydra
import rootutils
from lightning import LightningModule
from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import summarize
from omegaconf import DictConfig

from src.utils import RankedLogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Entry point for model summarisation.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    log.info(f"Instantiating model <{cfg.model._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    model: LightningModule = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
    model_summary = summarize(model)
    summary_data = model_summary._get_summary_data()
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size
    total_training_modes = model_summary.total_training_modes
    RichModelSummary.summarize(
        summary_data,
        total_parameters,
        trainable_parameters,
        model_size,
        total_training_modes,
    )


if __name__ == "__main__":
    main()
