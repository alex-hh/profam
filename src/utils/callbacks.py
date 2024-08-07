from lightning.pytorch.callbacks import Callback, ThroughputMonitor


class ShuffleCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
        trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)


class TokenThroughputMonitor(ThroughputMonitor):
    def __init__(self):
        super().__init__(
            batch_size_fn=lambda x: x["input_ids"].shape[0],
            length_fn=lambda x: x["input_ids"].shape[1] * x["input_ids"].shape[0],
        )
