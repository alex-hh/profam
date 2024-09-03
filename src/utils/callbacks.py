import time

from lightning.pytorch.callbacks import Callback, ThroughputMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ShuffleCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
        trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)


class EpochTimerCallback(Callback):
    """Needs to be a callback rather than module hooks becaues callbacks are always
    called first, so e.g. printcallback on_train_epoch_end wont have access to time
    from on_train_epoch_end unless we log it here.
    # https://github.com/Lightning-AI/pytorch-lightning/blob/1551a16b94f5234a4a78801098f64d0732ef5cb5/src/lightning/pytorch/loops/fit_loop.py#L375
    """

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0_epoch = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        self._t1_epoch = time.time()
        pl_module.log(
            "train/epoch_time",
            self._t1_epoch - self._t0_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_t0_epoch = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        self._val_t1_epoch = time.time()
        pl_module.log(
            "val/epoch_time",
            self._val_t1_epoch - self._val_t0_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )


class PrintCallback(Callback):
    def __init__(self, print_freq=1):
        self.print_freq = print_freq

    def on_train_epoch_end(self, trainer, pl_module):
        if self.print_freq > 0 and (
            (pl_module.current_epoch + 1) % self.print_freq == 0
        ):
            metrics = trainer.callback_metrics
            metrics_msg = "\t".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            log.info(f"Epoch {pl_module.current_epoch}, metrics:\t{metrics_msg}")


class TokenThroughputMonitor(ThroughputMonitor):
    """Modified to compute samples / tokens sizes and skip validation throughput (for now.)

    The length_fn is used to compute items_per_sec (effectively tokens per second)
    """

    def __init__(self, run_on_validation: bool = False):
        super().__init__(
            batch_size_fn=lambda x: x["input_ids"].shape[0],
            length_fn=lambda x: x["input_ids"].shape[1] * x["input_ids"].shape[0],
        )
        self.run_on_validation = run_on_validation

    @override
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module):
        if self.run_on_validation:
            super().on_validation_start(trainer, pl_module)

    @override
    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.run_on_validation:
            super().on_validation_end(trainer, pl_module)

    @override
    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, *args, **kwargs
    ):
        if self.run_on_validation:
            super().on_validation_batch_end(
                trainer, pl_module, outputs, batch, *args, **kwargs
            )
