import os
from argparse import Namespace
from typing import Any, Dict, Mapping, Optional, Union

import hydra
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override

from src.constants import BASEDIR


# TODO: use logging
class StdOutLogger(Logger):
    def __init__(self):
        self._experiment = DummyExperiment()

    @rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, float], step: Optional[int] = None
    ) -> None:
        for k, v in metrics.items():
            print(f"{k}: {v}", flush=True)

    @property
    def experiment(self) -> DummyExperiment:
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        pass

    @property
    @override
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    @override
    def version(self) -> str:
        """Return the experiment version."""
        return ""


class WandbLogger(WandbLogger):
    # TODO: extended to optionally log hydra config file and git hash
    def __init__(
        self, log_hydra_config_file: bool = False, log_git_hash: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.log_hydra_config = log_hydra_config_file
        self.log_git_hash = log_git_hash

    @rank_zero_only
    def log_hyperparameters(self, hparams, **kwargs):
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        hydra_cfg["runtime"]["output_dir"]
        if self.log_hydra_config_file:
            import wandb

            artifact = wandb.Artifact("hydra_outputs", type="config")
            hydra_dir = os.path.join(hydra_cfg["runtime"]["output_dir"], ".hydra")
            artifact.add_file(os.path.join(hydra_dir, "config.yaml"))
            artifact.add_file(os.path.join(hydra_dir, "hydra.yaml"))
            artifact.add_file(os.path.join(hydra_dir, "overrides.yaml"))
            self.experiment.log_artifact(artifact)

        if self.log_git_hash:
            with open(os.path.join(BASEDIR, "commit_hash.txt"), "r") as f:
                commit_hash = f.read().strip()
            hparams["git_hash"] = commit_hash
        super().log_hyperparameters(hparams, **kwargs)
