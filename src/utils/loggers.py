import os
import subprocess
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
        self.log_hydra_config_file = log_hydra_config_file
        self.log_git_hash = log_git_hash

    @rank_zero_only
    def log_hyperparams(self, hparams, **kwargs):
        hparams["wandb_run_name"] = self.experiment.name
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        hydra_cfg["runtime"]["output_dir"]
        if self.log_hydra_config_file:
            import wandb

            artifact = wandb.Artifact("hydra_outputs", type="config")
            hydra_dir = os.path.join(hydra_cfg["runtime"]["output_dir"], ".hydra")
            artifact.add_dir(hydra_dir)
            self.experiment.log_artifact(artifact)

        if self.log_git_hash:
            hash_file = os.path.join(BASEDIR, "commit_hash.txt")
            if os.path.isfile(hash_file):
                with open(hash_file, "r") as f:
                    commit_hash = f.read().strip()
                hparams["git_hash"] = commit_hash
            else:
                try:
                    commit_hash = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=BASEDIR,
                        )
                        .decode("utf-8")
                        .strip()
                    )
                    # TODO: why write to file? should it better if it updates dynamically from git?
                    # with open(hash_file, "w") as f:
                    #     f.write(commit_hash)
                    hparams["git_hash"] = commit_hash
                except subprocess.CalledProcessError:
                    # raise FileNotFoundError(
                    print(
                        f"WARNING:"
                        f"Could not get git hash. Please ensure you are in a git repository and run:\n"
                        f"git rev-parse HEAD > commit_hash.txt"
                    )
        # hparams is just wandb_run_name and git_hash - why?
        super().log_hyperparams(hparams, **kwargs)
