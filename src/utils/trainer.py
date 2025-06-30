import datetime
import math
from typing import List, Optional

import torch
from lightning import Callback, Trainer
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

# Import the callback from the callbacks module
from src.utils.callbacks import StepGradientAccumulationScheduler


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
        **kwargs,
    ):
        """
        timeout: timeout in seconds if using ddp strategy.
        target_tokens_per_batch: target number of tokens per batch.
        """

        # Determine num_effective_devices for calculations
        _devices_arg = kwargs.get("devices", "auto")
        num_effective_devices = 1  # Default (e.g., CPU, or if parsing fails)
        if isinstance(_devices_arg, int):
            if _devices_arg == -1:  # All available CUDA devices
                num_effective_devices = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 1
                )
            elif _devices_arg > 0:
                num_effective_devices = _devices_arg
        elif isinstance(_devices_arg, (list, tuple)):
            num_effective_devices = len(_devices_arg) if len(_devices_arg) > 0 else 1
        elif isinstance(_devices_arg, str):
            if _devices_arg == "auto":
                if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                    num_effective_devices = torch.cuda.device_count()
            elif _devices_arg.lower() in ["cpu", "mps"]:  # Common PTL device strings
                num_effective_devices = 1
            # Other string device identifiers (e.g., "cuda:0") typically refer to 1 device.

        if num_effective_devices == 0:  # Safety for division
            num_effective_devices = 1

        # Check for StepGradientAccumulationScheduler
        step_scheduler_active = False

        try:
            step_scheduler_active = any(
                [
                    isinstance(cb, StepGradientAccumulationScheduler)
                    for cb in kwargs.get("callbacks")
                ]
            )
        except:
            step_scheduler_active = False

        if step_scheduler_active:
            if target_tokens_per_batch is not None:
                raise ValueError(
                    "Cannot use `target_tokens_per_batch` when `StepGradientAccumulationScheduler` is active. "
                    "Configure accumulation scheduling via the callback (including for step 0)."
                )
            if (
                "accumulate_grad_batches" in kwargs
                and kwargs["accumulate_grad_batches"] != 1
            ):
                rank_zero_warn(
                    f"Trainer's `accumulate_grad_batches` was set to {kwargs['accumulate_grad_batches']} but "
                    "`StepGradientAccumulationScheduler` is active. Forcing to 1. "
                    "The scheduler will control accumulation."
                )
            kwargs[
                "accumulate_grad_batches"
            ] = 1  # Ensure PTL Trainer starts with 1 for scheduler to work correctly

        elif (
            target_tokens_per_batch is not None
        ):  # Original logic if scheduler is NOT active
            if (
                "accumulate_grad_batches" in kwargs
                and kwargs["accumulate_grad_batches"] is not None
            ):
                raise ValueError(
                    "Both `target_tokens_per_batch` and `accumulate_grad_batches` were specified. Please choose one."
                )
            if tokens_per_document is None or batch_size is None:
                raise ValueError(
                    "`tokens_per_document` and `batch_size` must be set when `target_tokens_per_batch` is used."
                )

            calculated_accumulate_grad_batches = math.ceil(
                target_tokens_per_batch
                / (tokens_per_document * batch_size * num_effective_devices)
            )
            kwargs["accumulate_grad_batches"] = calculated_accumulate_grad_batches
            print(
                f"Setting accumulate_grad_batches to {calculated_accumulate_grad_batches} "
                f"(target_tokens_per_batch={target_tokens_per_batch}, tokens_per_document={tokens_per_document}, "
                f"batch_size={batch_size}, num_effective_devices={num_effective_devices})"
            )

        if timeout is not None:
            # Check if strategy is explicitly DDP or auto (which might resolve to DDP)
            # This check is pre-trainer-instantiation, so trainer.strategy is not available
            current_strategy = kwargs.get("strategy", "auto")
            is_ddp_likely = False
            if isinstance(current_strategy, DDPStrategy):
                is_ddp_likely = True
            elif (
                isinstance(current_strategy, str) and current_strategy.lower() == "ddp"
            ):
                is_ddp_likely = True

            if (
                not is_ddp_likely and current_strategy != "auto"
            ):  # If auto, let PTL decide. If specific and not DDP, warn.
                rank_zero_warn(
                    f"Timeout is specified, but strategy is '{current_strategy}'. Timeout is typically for DDP."
                )

            # This will overwrite if user passed a DDPStrategy object without timeout.
            # If user passed DDPStrategy WITH timeout, this might conflict.
            # PTL usually merges this well.
            kwargs["strategy"] = DDPStrategy(
                timeout=datetime.timedelta(seconds=timeout)
            )

        if (
            val_check_interval_divide_by_world_size
            and kwargs.get("val_check_interval", 1.0) != 1.0  # type: ignore
            and kwargs.get("val_check_interval") is not None  # Ensure it's not None
        ):
            val_check_interval = kwargs.get("val_check_interval", 1.0)
            # Ensure val_check_interval is float or int for division
            if isinstance(val_check_interval, (int, float)):
                # Ensure num_effective_devices is at least 1 for division
                effective_divisor = max(1, num_effective_devices)
                new_val_check_interval = math.floor(
                    val_check_interval / effective_divisor
                )  # Use floor to get int
                if (
                    new_val_check_interval < 1 and val_check_interval > 0
                ):  # Don't let it become 0 if original > 0
                    new_val_check_interval = 1  # At least 1
                    rank_zero_warn(
                        f"Calculated val_check_interval ({val_check_interval} // {effective_divisor}) resulted in < 1. Setting to 1."
                    )
                kwargs["val_check_interval"] = new_val_check_interval
                print(
                    f"Dividing val_check_interval by num_effective_devices ({effective_divisor}). New val_check_interval: {kwargs['val_check_interval']}"
                )
            else:
                rank_zero_warn(
                    f"val_check_interval is not a number ({val_check_interval}), cannot divide by world size."
                )

        super().__init__(*args, **kwargs)
