import os
import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def seed_all(seed: Optional[int] = None, deterministic: bool = False) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducibility.

    Parameters
    ----------
    seed : Optional[int]
        Seed value. If None, no-op.
    deterministic : bool, default=False
        If True, configure PyTorch for deterministic algorithms where possible.
    """
    if seed is None:
        return
    import os
    import random

    import numpy as np

    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    try:
        np.random.seed(int(seed))
    except Exception:
        pass

    if torch is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            if deterministic:
                try:
                    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass


def np_random(seed: Optional[int] = None) -> Any:
    """Returns a numpy random number generator with a given seed.

    :param seed: The seed value for the random number generator.
    :return: A numpy random number generator.
    """
    if seed is not None:
        rnd = np.random.default_rng(seed)
    else:
        # to maintain control by global seed
        rnd = np.random
    return rnd


def maybe_print(*args, verbose=False, **kwargs):
    if verbose:
        print(*args)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def nested_getattr(obj, attr_path, default=None):
    """
    Retrieve a nested attribute value from an object given a dot-separated path.

    Parameters:
    - obj: The object from which to retrieve the attribute.
    - attr_path: A string representing the dot-separated path to the nested attribute.
    - default: The default value to return if the attribute is not found.

    Returns:
    The value of the nested attribute or the default value if not found.
    """
    attributes = attr_path.split(".")
    try:
        for attr in attributes:
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return default


def get_config_from_cpt_path(cpt_path: str) -> DictConfig:
    cpt_dir = os.path.dirname(cpt_path)
    config_path = os.path.join(cpt_dir, "../.hydra/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return DictConfig(config)
