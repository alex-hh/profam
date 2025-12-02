import hashlib
import time
from typing import Any, Dict, Optional, Union

import lightning as L
import lightning as pl
import torch
import torch.distributed as dist
from datasets import IterableDataset
from lightning.fabric.utilities.throughput import get_available_flops
from lightning.pytorch.callbacks import Callback, ThroughputMonitor
from lightning.pytorch.callbacks.throughput_monitor import _plugin_to_compute_dtype
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn
from omegaconf import DictConfig
from typing_extensions import override

from src.utils import RankedLogger
from src.utils.throughput import Throughput

log = RankedLogger(__name__, rank_zero_only=True)


class ShuffleCallback(Callback):
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
        self._samples: Union[Dict[RunningStage, int], DictConfig] = {}
        self._non_padding_lengths: Union[Dict[RunningStage, int], DictConfig] = {}
        self._proteins: Union[Dict[RunningStage, int], DictConfig] = {}

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
            self._proteins[stage] += max(
                (batch["input_ids"] == pl_module.tokenizer.sep_token_id).sum().item(), 1
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

        # In distributed setting, use Lightning's strategy to reduce across all ranks
        if trainer.world_size > 1:
            batch_size_tensor = torch.tensor(batch_size, device=pl_module.device)
            # Use the trainer strategy's reduce method to sum across ranks
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


class CountUniqueBatches(Callback):
    """
    Checks for repeated batches during training
    1) checks if the same samples are occuring in the packed batch together
    2) checks how often individual samples are seen during training
    """

    def __init__(self):
        super().__init__()
        self.samples_seen = 0
        self.dataset_sample_counts = {}
        self.identifier_sample_counts = {}
        self.batch_identifier_counts = {}

    def _merge_counts(
        self, trainer: L.Trainer, local_dict: Dict[str, int]
    ) -> Dict[str, int]:
        """Gather and sum counts across all ranks to get global counts."""
        if trainer.world_size > 1 and dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(trainer.world_size)]
            dist.all_gather_object(gathered, local_dict)
            merged = {}
            for d in gathered:
                for key, val in d.items():
                    merged[key] = merged.get(key, 0) + val
            return merged
        return local_dict.copy()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        identifier_strings = batch["identifier"].text[0].split("$")
        ds_strings = batch["ds_name"].text[0].split("$")
        ids_w_ds = [f"{id}{ds}" for id, ds in zip(identifier_strings, ds_strings)]
        # Hash the batch identifier for determinism and brevity
        raw_identifier = "".join(sorted(ids_w_ds))
        batch_id_hash = hashlib.md5(raw_identifier.encode("utf-8")).hexdigest()

        # Update batch identifier counts using hashed identifier
        self.batch_identifier_counts[batch_id_hash] = (
            self.batch_identifier_counts.get(batch_id_hash, 0) + 1
        )

        # Update sample counts per identifier
        for id in ids_w_ds:
            self.identifier_sample_counts[id] = (
                self.identifier_sample_counts.get(id, 0) + 1
            )

        # Update dataset sample counts
        for ds in ds_strings:
            self.dataset_sample_counts[ds] = self.dataset_sample_counts.get(ds, 0) + 1

        # Update total samples seen
        num_samples = len(ids_w_ds)
        self.samples_seen += num_samples

        # Sync local dictionaries into global counts
        global_batch_counts = self._merge_counts(trainer, self.batch_identifier_counts)
        global_identifier_counts = self._merge_counts(
            trainer, self.identifier_sample_counts
        )
        global_dataset_counts = self._merge_counts(trainer, self.dataset_sample_counts)

        # Compute metrics for logging each step
        batch_counts = list(global_batch_counts.values())
        total_batches = sum(batch_counts)
        unique_batches = len(batch_counts)
        repeated_batches = total_batches - unique_batches
        max_batch_repetition = max(batch_counts) if batch_counts else 0
        min_batch_repetition = min(batch_counts) if batch_counts else 0
        mean_batch_repetition = (
            total_batches / unique_batches if unique_batches else 0.0
        )

        sample_counts = list(global_identifier_counts.values())
        total_samples = sum(sample_counts)
        unique_samples = len(sample_counts)
        repeated_samples = total_samples - unique_samples
        max_sample_repetition = max(sample_counts) if sample_counts else 0
        min_sample_repetition = min(sample_counts) if sample_counts else 0
        mean_sample_repetition = (
            total_samples / unique_samples if unique_samples else 0.0
        )

        # Prepare metrics dict
        metrics = {
            "train_batch_monitoring/total_batches": total_batches,
            "train_batch_monitoring/unique_batches": unique_batches,
            "train_batch_monitoring/repeated_batches": repeated_batches,
            "train_batch_monitoring/max_batch_repetition": max_batch_repetition,
            "train_batch_monitoring/min_batch_repetition": min_batch_repetition,
            "train_batch_monitoring/mean_batch_repetition": mean_batch_repetition,
            "train_batch_monitoring/total_samples": total_samples,
            "train_batch_monitoring/unique_samples": unique_samples,
            "train_batch_monitoring/repeated_samples": repeated_samples,
            "train_batch_monitoring/max_sample_repetition": max_sample_repetition,
            "train_batch_monitoring/min_sample_repetition": min_sample_repetition,
            "train_batch_monitoring/mean_sample_repetition": mean_sample_repetition,
        }
        # Add per-dataset counts
        for ds, count in global_dataset_counts.items():
            metrics[f"train_batch_monitoring/{ds}_sample_count"] = count

        # Log metrics (already globally aggregated) only on rank zero
        pl_module.log_dict(
            metrics, on_step=True, on_epoch=False, sync_dist=False, rank_zero_only=True
        )


class StepGradientAccumulationScheduler(Callback):
    r"""Change gradient accumulation factor according to a step-based schedule.

    Args:
        scheduling: A dictionary where keys are global step numbers (non-negative integers)
                    and values are the accumulation factors (positive integers) to apply
                    starting from that step. Example: {0: 2, 1000: 4, 5000: 8}
                    If step 0 is not specified, it defaults to an accumulation factor of 1
                    until the first specified step.
    implementation based on:
    https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/callbacks/gradient_accumulation_scheduler.html#GradientAccumulationScheduler
    """

    def __init__(self, scheduling: Union[dict[int, int], DictConfig]):
        super().__init__()
        assert scheduling is not None
        if not isinstance(scheduling, Union[dict, DictConfig]) or not scheduling:
            raise MisconfigurationException(
                "`scheduling` must be a non-empty dictionary."
            )

        if any(not isinstance(step, int) or step < 0 for step in scheduling.keys()):
            raise MisconfigurationException(
                f"Scheduler steps must be non-negative integers. Got {list(scheduling.keys())}."
            )

        if any(
            not isinstance(factor, int) or factor < 1 for factor in scheduling.values()
        ):
            raise MisconfigurationException(
                f"Accumulation factors must be positive integers. Got {list(scheduling.values())}."
            )

        self.scheduling = scheduling.copy()

        if 0 not in self.scheduling:
            # If scheduling is not empty and min key > 0, or if scheduling is empty (which is caught above)
            # This ensures that if user defines e.g. {100:2}, it implies {0:1, 100:2}
            # min() is safe here due to prior checks (non-empty, keys >=0)
            if min(self.scheduling.keys()) > 0:
                self.scheduling[0] = 1

        self.sorted_steps = sorted(self.scheduling.keys())
        # Ensure sorted_steps is not empty, which should be guaranteed if scheduling[0] is added.
        if not self.sorted_steps:
            raise MisconfigurationException(
                "`scheduling` must define at least one step, typically step 0."
            )

    def _get_accumulate_grad_batches(self, current_global_step: int) -> int:
        accumulate_grad_batches = (
            1  # Default, should be overridden by self.scheduling[0]
        )
        for step_threshold in reversed(self.sorted_steps):
            if current_global_step >= step_threshold:
                accumulate_grad_batches = self.scheduling[step_threshold]
                break
        return accumulate_grad_batches

    def _is_method_overridden(
        self, method_name: str, pl_module: "pl.LightningModule"
    ) -> bool:
        """Simple check to see if a method has been overridden from the base class."""
        try:
            # Check if the method exists and is not the same as the base class method
            base_method = getattr(pl.LightningModule, method_name, None)
            module_method = getattr(pl_module, method_name, None)

            if base_method is None or module_method is None:
                return False

            # If the methods are different objects, it's likely overridden
            return base_method != module_method
        except Exception:
            # If anything goes wrong, just return False to be safe
            return False

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not pl_module.automatic_optimization:
            raise MisconfigurationException(
                "Automatic gradient accumulation and the `StepGradientAccumulationScheduler` is not supported for "
                "manual optimization. Please remove the callback or switch to automatic optimization."
            )

        # Check if optimizer methods are overridden (optional warning)
        overridden_optimizer_step = self._is_method_overridden(
            "optimizer_step", pl_module
        )
        overridden_optimizer_zero_grad = self._is_method_overridden(
            "optimizer_zero_grad", pl_module
        )
        is_any_factor_greater_than_one = any(v > 1 for v in self.scheduling.values())

        if (
            overridden_optimizer_step or overridden_optimizer_zero_grad
        ) and is_any_factor_greater_than_one:
            rank_zero_warn(
                "When using `StepGradientAccumulationScheduler` with factors > 1 and overriding "
                "`LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch "
                "(rather, they are called on every optimization step)."
            )

        # Using MisconfigurationException for consistency with PTL error types
        from lightning.pytorch.strategies import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy):
            rank_zero_warn(  # Changed to warn, as some DeepSpeed versions might handle this. User should verify.
                f"The `{type(trainer.strategy).__name__}` might not support `accumulate_grad_batches` changing "
                "dynamically via this callback. Please verify DeepSpeed's accumulation settings."
            )

        if trainer.accumulate_grad_batches != 1:
            raise MisconfigurationException(
                f"You are using the `StepGradientAccumulationScheduler` callback, but `trainer.accumulate_grad_batches` "
                f"is {trainer.accumulate_grad_batches} (expected 1). "
                "Please ensure `accumulate_grad_batches` is set to 1 in the Trainer when using this scheduler."
            )

        # Set initial accumulation factor based on step 0 schedule
        trainer.accumulate_grad_batches = self._get_accumulate_grad_batches(0)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        trainer.accumulate_grad_batches = self._get_accumulate_grad_batches(
            trainer.global_step
        )
