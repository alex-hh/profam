from lightning.pytorch.callbacks import Callback, ThroughputMonitor
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from typing_extensions import override


class ShuffleCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
        trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)


class TokenThroughputMonitor(ThroughputMonitor):
    """Modified to compute samples / tokens sizes and skip validation throughput (for now.)"""

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


class PrintCallback(Callback):
    def __init__(self, print_freq=1):
        self.print_freq = print_freq

    def on_train_epoch_end(self, trainer, pl_module):
        if self.print_freq > 0 and (
            (pl_module.current_epoch + 1) % self.print_freq == 0
        ):
            metrics = trainer.callback_metrics
            metrics_msg = "\t".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            print(
                f"Epoch {pl_module.current_epoch}, metrics:\t{metrics_msg}", flush=True
            )
