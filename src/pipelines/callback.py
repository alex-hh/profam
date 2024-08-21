from collections import defaultdict
from typing import Dict, Optional

from lightning.pytorch.callbacks import Callback


class SamplingEvaluationPipelineCallback(Callback):
    def __init__(
        self, pipeline, evaluator, num_samples, sampling_kwargs: Optional[Dict] = None
    ):
        self.pipeline = pipeline
        assert (
            not self.pipeline.save_to_file
        ), "Pipeline should not save to file during callback"
        self.evaluator = evaluator
        self.num_samples = num_samples
        self.sampling_kwargs = sampling_kwargs or {}

    def on_val_epoch_end(self, trainer, model):
        # run on val epoch end rather than train to stay in sync with other validation metrics
        if trainer.is_global_zero:
            # https://lightning.ai/docs/pytorch/stable/visualize/logging_advanced.html#rank-zero-only
            # Q: how does logging work across ranks? if i log only from rank 0, what happens?
            all_metrics = defaultdict(list)
            results_df = self.pipeline.run(
                model,
                "profam_model",
                self.evaluator,
                sampling_kwargs=self.sampling_kwargs,
            )
            mean_results = results_df.mean().to_dict()
            all_metrics = {
                f"{self.evaluator_name}/{k}": v for k, v in mean_results.items()
            }
            trainer.log_dict(all_metrics, on_epoch=True, rank_zero_only=True)
