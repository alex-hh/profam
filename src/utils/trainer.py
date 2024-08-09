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
        **kwargs
    ):
        if target_tokens_per_batch is not None:
            devices = kwargs.get("devices", "auto")
            if devices == "auto":
                assert torch.cuda.is_available()
                devices = torch.cuda.device_count()

            assert (
                "accumulate_grad_batches" not in kwargs
            ), "accumulate_grad_batches should not be set when target_tokens_per_batch is set"
            kwargs["accumulate_grad_batches"] = math.ceil(
                target_tokens_per_batch / (tokens_per_document * batch_size * devices)
            )
            print(
                "Setting accumulate_grad_batches to", kwargs["accumulate_grad_batches"]
            )
        super().__init__(*args, **kwargs)
