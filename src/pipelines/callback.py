import time
from collections import defaultdict
from typing import Dict, Optional

from lightning.pytorch.callbacks import Callback

from src.data.preprocessing import BasePreprocessorConfig
from src.evaluators.base import SamplingEvaluator
from src.models.inference import ProFamSampler, PromptBuilder
from src.pipelines.pipeline import GenerationsEvaluatorPipeline


class SamplingEvaluationPipelineCallback(Callback):
    def __init__(
        self,
        pipeline: GenerationsEvaluatorPipeline,
        evaluator: SamplingEvaluator,
        preprocessor: BasePreprocessorConfig,
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        sampling_kwargs: Optional[Dict] = None,
    ):
        self.pipeline = pipeline
        assert (
            not self.pipeline.save_to_file
        ), "Pipeline should not save to file during callback"
        self.evaluator = evaluator
        self.sampling_kwargs = sampling_kwargs or {}
        self.preprocessor = preprocessor
        assert self.preprocessor is not None
        self.max_tokens = max_tokens
        self.seed = seed

    def on_validation_epoch_end(self, trainer, model):
        # run on val epoch end rather than train to stay in sync with other validation metrics
        if trainer.sanity_checking:
            return
        device = model.device
        sampler = ProFamSampler(
            "profam_sampler",
            model,
            prompt_builder=PromptBuilder(
                self.preprocessor,
                max_tokens=self.max_tokens,
                seed=self.seed,
            ),
            sampling_kwargs=self.sampling_kwargs,
        )
        if trainer.is_global_zero:
            # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#rank-zero-only
            # Q: how does logging work across ranks? if i log only from rank 0, what happens?
            all_metrics = defaultdict(list)
            t0 = time.time()
            results_df = self.pipeline.run(
                sampler,
                self.evaluator,
                verbose=False,
            )
            sampler.to(device)
            mean_results = results_df.mean().to_dict()
            t1 = time.time()
            all_metrics = {
                f"{self.evaluator.name}/{k}": v for k, v in mean_results.items()
            }
            all_metrics[f"{self.evaluator.name}/time"] = t1 - t0
            model.log_dict(all_metrics, on_epoch=True, rank_zero_only=True)
