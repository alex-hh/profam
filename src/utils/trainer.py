import datetime
import math
from typing import Optional

import torch
from lightning import Trainer
from lightning.pytorch.strategies import DDPStrategy


class ProFamTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_tokens_per_batch=None,
        batch_size=None,
        tokens_per_document=None,
        timeout: Optional[int] = None,
        # n.b. val_check_interval uses BatchProgresss. This is a local counter.
        val_check_interval_divide_by_world_size: bool = True,
        **kwargs
    ):
        """
        timeout: timeout in seconds if using ddp strategy.
        target_tokens_per_batch: target number of tokens per batch.
        """
        devices = kwargs.get("devices", "auto")
        if devices == "auto":
            assert torch.cuda.is_available()
            devices = torch.cuda.device_count()
            print("Setting CUDA devices to", devices)
        if target_tokens_per_batch is not None:
            assert (
                "accumulate_grad_batches" not in kwargs
            ), "accumulate_grad_batches should not be set when target_tokens_per_batch is set"
            assert (
                tokens_per_document is not None
            ), "tokens_per_document must be set when target_tokens_per_batch is set"
            kwargs["accumulate_grad_batches"] = math.ceil(
                target_tokens_per_batch / (tokens_per_document * batch_size * devices)
            )
            print(
                "Setting accumulate_grad_batches to", kwargs["accumulate_grad_batches"]
            )
        if timeout is not None:
            assert kwargs.get("strategy", "auto") == "ddp"
            # default is 1800 seconds
            kwargs["strategy"] = DDPStrategy(
                timeout=datetime.timedelta(seconds=timeout)
            )
        if (
            val_check_interval_divide_by_world_size
            and kwargs.get("val_check_interval", 1.0) != 1.0
            and kwargs.get("val_check_interval", 1.0) is not None
        ):
            val_check_interval = kwargs.get("val_check_interval", 1.0)
            kwargs["val_check_interval"] = val_check_interval // devices
            print("Setting val_check_interval to", kwargs["val_check_interval"])
        super().__init__(*args, **kwargs)
