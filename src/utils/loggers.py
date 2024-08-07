from typing import Any, Dict, Mapping, Optional, Union
from typing_extensions import override
from argparse import Namespace
from lightning.fabric.loggers.logger import _DummyExperiment as DummyExperiment
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


# TODO: use logging
class StdOutLogger(Logger):

    def __init__(self):
        self._experiment = DummyExperiment()

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            print(f'{k}: {v}')

    @property
    def experiment(self) -> DummyExperiment:
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        print(params)

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