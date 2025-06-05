import time
from typing import Any, Dict, Optional

import lightning as L
import torch
from datasets import IterableDataset
from lightning.fabric.utilities.throughput import get_available_flops
from lightning.pytorch.callbacks import Callback, ThroughputMonitor
from lightning.pytorch.callbacks.throughput_monitor import _plugin_to_compute_dtype
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from typing_extensions import override

from src.utils import RankedLogger
from src.utils.throughput import Throughput

log = RankedLogger(__name__, rank_zero_only=True)


class ShuffleCallback(Callback):
    # TODO: check this works with interleaved datasets
    # https://huggingface.co/docs/datasets/en/stream#reshuffle
    def on_train_epoch_start(self, trainer, pl_module):
        # https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.to_iterable_dataset
        if isinstance(trainer.train_dataloader.dataset, IterableDataset):
            trainer.train_dataloader.dataset.set_epoch(trainer.current_epoch)
        # Also set epoch on the sampler if it supports shuffling per epoch
        sampler = trainer.train_dataloader.sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(trainer.current_epoch)


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
            sync_dist=True,
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
            sync_dist=False,
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


# if getting a bug like this, upgrade lightning:
# You set `Trainer(accumulate_grad_batches=31, log_every_n_steps=10)` but these are not divisible and thus will not log anything.
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
        self._samples: Dict[RunningStage, int] = {}
        self._non_padding_lengths: Dict[RunningStage, int] = {}
        self._proteins: Dict[RunningStage, int] = {}

    @override
    def setup(
        self, trainer: "Trainer", pl_module: "LightningModule", stage: str
    ) -> None:
        dtype = _plugin_to_compute_dtype(trainer.precision_plugin)
        self.available_flops = get_available_flops(trainer.strategy.root_device, dtype)

        if stage == TrainerFn.FITTING and trainer.enable_validation:
            # `fit` includes validation inside
            throughput = Throughput(
                available_flops=self.available_flops,
                world_size=trainer.world_size,
                **self.kwargs,
            )
            self._throughputs[RunningStage.VALIDATING] = throughput

        throughput = Throughput(
            available_flops=self.available_flops,
            world_size=trainer.world_size,
            **self.kwargs,
        )
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage] = throughput

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

    def _start(self, trainer: "Trainer") -> None:
        stage = trainer.state.stage
        assert stage is not None
        self._throughputs[stage].reset()
        self._lengths[stage] = 0
        self._t0s[stage] = time.perf_counter()
        self._samples[stage] = 0
        self._non_padding_lengths[stage] = 0
        self._proteins[stage] = 0

    def _compute(self, trainer: "Trainer", iter_num: Optional[int] = None) -> None:
        # modified to add 'throughput' as a prefix
        if not trainer._logger_connector.should_update_logs:
            return
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]
        metrics = throughput.compute()
        # prefix with the stage to avoid collisions
        metrics = {
            f"throughput/{stage.value}{throughput.separator}{k}": v
            for k, v in metrics.items()
        }
        trainer._logger_connector.log_metrics(metrics, step=iter_num)  # type: ignore[arg-type]

    @torch.inference_mode()  # in case `length_fn` or `batch_size_fn` computes grads
    def _update(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        batch: Any,
        iter_num: int,
    ) -> None:
        stage = trainer.state.stage
        assert stage is not None
        throughput = self._throughputs[stage]

        if trainer.strategy.root_device.type == "cuda":
            # required or else perf_counter() won't be correct
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._t0s[stage]
        if self.length_fn is not None:
            self._lengths[stage] += self.length_fn(batch)

        if hasattr(pl_module, "tokenizer"):
            padding_mask = (
                batch["input_ids"] != pl_module.tokenizer.pad_token_id
            ).float()
            self._non_padding_lengths[stage] += padding_mask.sum().item()
            self._proteins[stage] += (
                (batch["input_ids"] == pl_module.tokenizer.sep_token_id).sum().item()
            )

        self._samples[stage] += self.batch_size_fn(batch)

        if hasattr(pl_module, "flops_per_batch"):
            flops_per_batch = pl_module.flops_per_batch
        else:
            # rank_zero_warn(
            #     "When using the `ThroughputMonitor`, you need to define a `flops_per_batch` attribute or property"
            #     f" in {type(pl_module).__name__} to compute the FLOPs."
            # )
            flops_per_batch = None

        throughput.update(
            time=elapsed,
            batches=iter_num,
            # this assumes that all iterations used the same batch size
            samples=self._samples[stage],
            lengths=None if self.length_fn is None else self._lengths[stage],
            non_padding_lengths=self._non_padding_lengths[stage],
            proteins=self._proteins[stage],
            flops=flops_per_batch,
        )


class SampleCounter(Callback):
    """
    Tracks the total number of samples seen during training.

    This callback maintains a counter of samples processed across all dataloaders,
    which persists through checkpoint saves and loads. The counter works with
    distributed training across multiple devices and nodes.
    """

    def __init__(self):
        super().__init__()
        self.samples_seen = 0
        self.dataset_sample_counts = {}

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update the sample count after each batch is processed"""
        if isinstance(batch["batch_size"], torch.Tensor):
            batch_size = batch["batch_size"].item()
        else:
            batch_size = batch["batch_size"]

        # In distributed setting, use Lightning’s strategy to reduce across all ranks
        if trainer.world_size > 1:
            batch_size_tensor = torch.tensor(batch_size, device=pl_module.device)
            # Use the trainer strategy’s reduce method to sum across ranks
            batch_size_tensor = trainer.strategy.reduce(
                batch_size_tensor, reduce_op="SUM"
            )
            batch_size = batch_size_tensor.item()

        self.samples_seen += batch_size

        pl_module.samples_seen = self.samples_seen

        # # Log dataset sample counts
        # ds_name = batch["ds_name"].text
        # for ds in ds_name:
        #     self.dataset_sample_counts[ds] = self.dataset_sample_counts.get(ds, 0) + 1

        pl_module.log(
            "train/total_samples_seen",
            self.samples_seen,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            rank_zero_only=True,
        )
        # rank_zero_info(f"Total samples seen: {self.samples_seen}")

        # # Log dataset sample counts
        # pl_module.log_dict(
        #     {
        #         f"train/{k}_times_sampled": v
        #         for k, v in self.dataset_sample_counts.items()
        #     },
        #     on_step=True,
        #     on_epoch=False,
        #     sync_dist=False,  # Ensure counts are synchronized across devices
        #     rank_zero_only=True,  # Allow all ranks to log
        # )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "samples_seen": self.samples_seen,
            "dataset_sample_counts": self.dataset_sample_counts,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.samples_seen = state_dict.get("samples_seen", 0)
        self.dataset_sample_counts = state_dict.get("dataset_sample_counts", {})

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        trainer.samples_seen = self.samples_seen
