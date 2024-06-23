from lightning.pytorch.callbacks import Callback


class ShuffleCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
        trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)
