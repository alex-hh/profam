import logging
from typing import Optional

import lightning as L
import torch
from omegaconf import DictConfig


def setup_profiler(
    cfg: DictConfig, log: logging.Logger
) -> Optional[L.pytorch.profilers.base.Profiler]:
    profiler_name = cfg.name
    if profiler_name is None:
        return None

    if profiler_name not in [None, "simple", "advanced", "pytorch"]:
        raise ValueError(
            f"Profiler {profiler_name} not recognized. Choose from [None, simple, advanced, pytorch]"
        )

    # build profiler's kwargs
    profiler_cfg = cfg.get(profiler_name, {})
    if cfg.log_tensorboard:
        on_trace_ready = torch.profiler.tensorboard_trace_handler(
            profiler_cfg.get("dirpath", "./")
        )
        schedule = torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=1,
        )
    else:
        on_trace_ready = None
        schedule = None

    profiler_cls = getattr(L.pytorch.profilers, profiler_cfg["_target_"])
    profiler_kwargs = {
        "on_trace_ready": on_trace_ready,
        "schedule": schedule,
    }
    profiler_kwargs.update(profiler_cfg)

    profiler = profiler_cls(**profiler_kwargs)
    log.info(f"Created profiler {profiler}")
    log.info(f"Profiler {profiler_name} kwargs = {profiler_kwargs}")

    return profiler


def save_profiler(
    profiler: L.pytorch.profilers.base.Profiler, stage: str, log: logging.Logger
) -> None:
    """Save profiler data if needed."""
    if profiler is not None:
        log.info("\nSaving unsaved profiler data if needed...")
        profiler.teardown(stage)
        # profiler.save_report()
