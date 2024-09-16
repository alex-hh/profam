import math

import torch
from lightning import Trainer


class ProFamTrainer(Trainer):
    def __init__(
        self,
        *args,
        target_tokens_per_batch=None,
        batch_size=None,
        tokens_per_document=None,
        # n.b. val_check_interval uses BatchProgresss. This is a local counter.
        val_check_interval_divide_by_world_size: bool = True,
        **kwargs
    ):
        devices = kwargs.get("devices", "auto")
        if devices == "auto":
            assert torch.cuda.is_available()
            devices = torch.cuda.device_count()
        if target_tokens_per_batch is not None:
            assert (
                "accumulate_grad_batches" not in kwargs
            ), "accumulate_grad_batches should not be set when target_tokens_per_batch is set"
            kwargs["accumulate_grad_batches"] = math.ceil(
                target_tokens_per_batch / (tokens_per_document * batch_size * devices)
            )
            print(
                "Setting accumulate_grad_batches to", kwargs["accumulate_grad_batches"]
            )
        if val_check_interval_divide_by_world_size:
            val_check_interval = kwargs.get("val_check_interval", 1)
            kwargs["val_check_interval"] = val_check_interval // devices
            print("Setting val_check_interval to", kwargs["val_check_interval"])
        super().__init__(*args, **kwargs)
