from typing import Dict, List, Optional

import pandas as pd
from scipy.stats import spearmanr

from src.data.objects import ProteinDocument


class BaseEvaluator:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("should be implemented on child class")


class SamplingEvaluator(BaseEvaluator):
    def __init__(
        self,
        name: str,
        num_samples: Optional[int] = None,
    ):
        super().__init__(name)
        self.num_samples = num_samples

    def evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        num_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        if num_samples is not None and len(samples) != num_samples:
            assert len(samples) >= num_samples, f"Need at least {num_samples} samples"
            samples = samples[:num_samples]  # assuming samples are unsorted

        return self._evaluate_samples(
            prompt, protein_document, samples, output_dir=output_dir, device=device
        )

    def _evaluate_samples(
        self,
        prompt: ProteinDocument,
        protein_document: ProteinDocument,
        samples: List[str],
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        raise NotImplementedError("should be implemented on child class")

    def __call__(
        self,
        sampler,
        protein_document: ProteinDocument,
        num_samples: int,
        device: Optional[str] = None,
    ):
        sampler.to(device)
        samples, prompt = sampler.sample_seqs(protein_document, num_samples)
        return self.evaluate_samples(prompt, samples, device=device)


class ScoringEvaluator(BaseEvaluator):
    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)

    def evaluate_scored_completions(
        self,
        prompt: ProteinDocument,
        scored_mutants_df: pd.DataFrame,
        device: Optional[str] = None,
    ):
        raise NotImplementedError("should be implemented on child class")

    def __call__(
        self,
        scorer,
        protein_document: ProteinDocument,
        completions_df: pd.DataFrame,
        device: Optional[str] = None,
    ):
        scorer.to(device)
        scored_completions_df, prompt = scorer.score_completions(
            protein_document, completions_df
        )
        return self.evaluate_scored_completions(
            scored_completions_df=scored_completions_df, device=device
        )


class FitnessPredictionEvaluator(ScoringEvaluator):
    # TODO: add other metrics e.g. auc, ndcg
    def evaluate_scored_completions(
        self,
        prompt: ProteinDocument,
        scored_completions_df: pd.DataFrame,
        device: Optional[str] = None,
    ):
        spearman_corr, _ = spearmanr(
            scored_completions_df["DMS_score"], scored_completions_df["predicted_score"]
        )
        if "DMS_score_bin" in scored_completions_df.columns:
            pass
        return {"spearman": spearman_corr}
