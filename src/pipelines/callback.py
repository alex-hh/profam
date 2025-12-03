import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig

from src.evaluators.base import SamplingEvaluator
from src.models.inference import ProFamSampler, PromptBuilder
from src.pipelines.pipeline import GenerationsEvaluatorPipeline
from src.utils.loggers import WandbLogger


class SamplingEvaluationPipelineCallback(Callback):
    def __init__(
        self,
        pipeline: GenerationsEvaluatorPipeline,
        evaluators: Union[
            List[SamplingEvaluator], Dict[str, SamplingEvaluator], SamplingEvaluator
        ],
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
        match_representative_length: bool = False,
    ):
        self.pipeline = pipeline
        assert (
            not self.pipeline.save_results_to_file
        ), "Pipeline should not save to file during callback"
        self.evaluators = evaluators
        self.sampling_kwargs = sampling_kwargs or {}
        self.prompt_builder = prompt_builder
        self.match_representative_length = match_representative_length
        if isinstance(self.evaluators, Dict) or isinstance(self.evaluators, DictConfig):
            self.evaluators = list(self.evaluators.values())
        if not isinstance(self.evaluators, List):
            assert isinstance(self.evaluators, SamplingEvaluator)
            self.evaluators: List[SamplingEvaluator] = [self.evaluators]

    def _log_sequences_to_wandb(self, trainer, evaluator_results, non_numeric_cols):
        """Log sequences to WandB if WandB logger is configured.
        Args:
            trainer: The PyTorch Lightning trainer
            evaluator_results: DataFrame containing evaluation results
            non_numeric_cols: List of non-numeric column names
        """
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                for col in non_numeric_cols:
                    logger.experiment.log(
                        {
                            f"{self.pipeline.pipeline_id}/{col}": evaluator_results[col]
                            .iloc[0]
                            .split("|")[:5]
                        }
                    )
                break

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, model):
        # run on val epoch end rather than train to stay in sync with other validation metrics
        if trainer.sanity_checking:
            return

        sampler = ProFamSampler(
            "profam_sampler",
            model,
            prompt_builder=self.prompt_builder,
            sampling_kwargs=self.sampling_kwargs,
            match_representative_length=self.match_representative_length,
        )
        # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#rank-zero-only
        # Q: how does logging work across ranks? if i log only from rank 0, what happens?
        all_metrics = defaultdict(list)
        t0 = time.time()
        # self.pipeline.reset()  # clear stale results (rerun sampler and rerun evaluator should suffice anyway but no harm)
        results_dfs = self.pipeline.run(
            sampler,
            self.evaluators,
            verbose=False,
            rerun_evaluator=True,
            rerun_sampler=True,
            device=model.device,
            disable_tqdm=True,
        )

        all_metrics = {}
        for evaluator_name, evaluator_results in results_dfs.items():
            numeric_cols = evaluator_results.select_dtypes(include=np.number).columns
            non_numeric_cols = evaluator_results.columns.difference(numeric_cols)
            mean_results = evaluator_results[numeric_cols].mean().to_dict()
            t1 = time.time()
            all_metrics.update(
                {
                    f"{self.pipeline.pipeline_id}/{evaluator_name}/{k}": v
                    for k, v in mean_results.items()
                }
            )
            all_metrics[f"{self.pipeline.pipeline_id}/{evaluator_name}/time"] = t1 - t0
            # Log sequences if using WandB
            if len(non_numeric_cols) > 0:
                self._log_sequences_to_wandb(
                    trainer, evaluator_results, non_numeric_cols
                )
        model.log_dict(all_metrics, on_epoch=True, rank_zero_only=True, sync_dist=True)

    @rank_zero_only
    def on_test_start(self, trainer, model):
        self.on_validation_epoch_end(trainer, model)
