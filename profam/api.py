"""Public Python API for ProFam."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from profam.checkpoint import load_model
from profam.data.objects import ProteinDocument
from profam.data.processors.preprocessing import (
    AlignedProteinPreprocessingConfig,
    ProteinDocumentPreprocessor,
)
from profam.models.inference import (
    EnsemblePromptBuilder,
    ProFamEnsembleSampler,
    ProFamSampler,
    PromptBuilder,
)
from profam.models.llama import LlamaLitModule
from profam.scoring import score_variants_ensemble
from profam.utils.utils import seed_all


@dataclass
class GenerationResult:
    """Result of a sequence generation call."""

    sequences: List[str]
    scores: List[float]


@dataclass
class ScoringResult:
    """Result of a sequence scoring call."""

    sequences: List[str]
    scores: np.ndarray
    residue_scores: Optional[List[np.ndarray]] = field(default=None)


def _build_protein_document(sequences: list[str]) -> ProteinDocument:
    """Build a ProteinDocument from a plain list of amino acid strings."""
    accessions = [f"seq_{i}" for i in range(len(sequences))]
    rep = accessions[0] if accessions else "representative"
    return ProteinDocument(
        sequences=sequences,
        accessions=accessions,
        identifier="api_input",
        representative_accession=rep,
    )


class ProFam:
    """High-level interface to the ProFam protein family language model.

    Parameters
    ----------
    checkpoint:
        Path to a ``.ckpt`` file, or *None* to use the default ProFam-1 checkpoint.
    device:
        Target device.  *None* auto-detects CUDA/CPU.
    dtype:
        One of ``"float32"``, ``"float16"``, ``"bfloat16"``.
    attn_implementation:
        ``"sdpa"``, ``"flash_attention_2"``, or ``"eager"``.
    auto_download:
        Automatically download the default checkpoint if it is missing.

    Example
    -------
    >>> from profam import ProFam
    >>> model = ProFam()
    >>> result = model.generate(prompt=["ACDEFGHIKLMNPQRSTVWY"], num_samples=5)
    >>> print(result.sequences)
    """

    def __init__(
        self,
        checkpoint: str | os.PathLike | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        auto_download: bool = True,
    ):
        kwargs = dict(
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            auto_download=auto_download,
        )
        if checkpoint is not None:
            kwargs["checkpoint"] = checkpoint

        self._model: LlamaLitModule = load_model(**kwargs)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: list[str],
        num_samples: int = 10,
        max_tokens: int = 8192,
        max_generated_length: int | None = None,
        max_sequence_length_multiplier: float = 1.2,
        temperature: float | None = None,
        top_p: float = 0.95,
        sampler: str = "single",
        num_prompts_in_ensemble: int = 8,
        reduction: str = "mean_probs",
        minimum_sequence_length_proportion: float = 0.5,
        minimum_sequence_identity: float | None = None,
        maximum_retries: int = 5,
        repeat_guard: bool = True,
        continuous_sampling: bool = False,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate novel protein sequences conditioned on *prompt*.

        Parameters
        ----------
        prompt:
            Family context sequences (plain amino-acid strings).
        num_samples:
            Number of sequences to generate.
        max_tokens:
            Token budget for the prompt+generation (max 8192).
        max_generated_length:
            Hard cap on generated sequence length.  *None* infers from prompt.
        max_sequence_length_multiplier:
            Caps generated length to this factor times the longest prompt seq.
        temperature:
            Sampling temperature.  *None* uses model default.
        top_p:
            Nucleus sampling probability mass.
        sampler:
            ``"single"`` or ``"ensemble"``.
        num_prompts_in_ensemble:
            Number of prompt sub-samples when using ensemble sampler.
        reduction:
            ``"mean_probs"`` or ``"sum_log_probs"`` (ensemble only).
        minimum_sequence_length_proportion:
            Discard sequences shorter than this fraction of the shortest prompt.
        minimum_sequence_identity:
            Discard sequences below this identity threshold.  *None* disables.
        maximum_retries:
            Retry limit when generated sequences are filtered out.
        repeat_guard:
            Abort and retry sequences with excessive amino-acid repeats.
        continuous_sampling:
            Ignore ``[SEP]`` EOS and generate until token budget.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        GenerationResult
        """
        if max_tokens > 8192:
            raise ValueError(
                "max_tokens must be <= 8192: model was trained up to 8192 tokens."
            )

        if seed is not None:
            seed_all(seed)

        doc_token = "[RAW]"
        pool = _build_protein_document(prompt)

        longest_prompt_len = int(max(pool.sequence_lengths))
        default_cap = int(longest_prompt_len * float(max_sequence_length_multiplier))
        if max_generated_length is None:
            max_gen_len = default_cap
        else:
            max_gen_len = min(int(max_generated_length), default_cap)
        if continuous_sampling:
            max_gen_len = None

        cfg = AlignedProteinPreprocessingConfig(
            document_token=doc_token,
            defer_sampling=True if sampler == "ensemble" else False,
            padding="do_not_pad",
            shuffle_proteins_in_document=True,
            keep_insertions=True,
            to_upper=True,
            keep_gaps=False,
            use_msa_pos=False,
            max_tokens_per_example=None if sampler == "ensemble" else max_tokens,
        )
        preprocessor = ProteinDocumentPreprocessor(cfg=cfg)

        if sampler == "ensemble":
            builder = EnsemblePromptBuilder(
                preprocessor=preprocessor, shuffle=True, seed=seed
            )
            sampler_obj = ProFamEnsembleSampler(
                name="ensemble_sampler",
                model=self._model,
                prompt_builder=builder,
                document_token=doc_token,
                reduction=reduction,
                temperature=temperature,
                top_p=top_p,
                add_final_sep=True,
            )
            sampler_obj.to(self._model.device)
            sequences, scores, _ = sampler_obj.sample_seqs_ensemble(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                num_prompts_in_ensemble=min(
                    num_prompts_in_ensemble, len(pool.sequences)
                ),
                max_generated_length=max_gen_len,
                continuous_sampling=continuous_sampling,
                minimum_sequence_length_proportion=minimum_sequence_length_proportion,
                minimum_sequence_identity=minimum_sequence_identity,
                maximum_retries=maximum_retries,
                repeat_guard=repeat_guard,
            )
        else:
            if not continuous_sampling and max_gen_len is not None:
                cfg.max_tokens_per_example = max_tokens - max_gen_len
            builder = PromptBuilder(
                preprocessor=preprocessor, prompt_is_aligned=True, seed=seed
            )
            sampling_kwargs: dict = {}
            if top_p is not None:
                sampling_kwargs["top_p"] = top_p
            if temperature is not None:
                sampling_kwargs["temperature"] = temperature
            sampler_obj = ProFamSampler(
                name="single_sampler",
                model=self._model,
                prompt_builder=builder,
                document_token=doc_token,
                sampling_kwargs=sampling_kwargs if sampling_kwargs else None,
                add_final_sep=True,
            )
            sampler_obj.to(self._model.device)
            sequences, scores, _ = sampler_obj.sample_seqs(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                max_generated_length=max_gen_len,
                continuous_sampling=continuous_sampling,
                minimum_sequence_length_proportion=minimum_sequence_length_proportion,
                minimum_sequence_identity=minimum_sequence_identity,
                maximum_retries=maximum_retries,
                repeat_guard=repeat_guard,
            )

        return GenerationResult(sequences=sequences, scores=scores)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        sequences: list[str],
        prompt: list[str] | None = None,
        ensemble_size: int = 3,
        max_tokens: int = 8192,
        scoring_max_tokens: int = 64000,
        use_diversity_weights: bool = True,
        diversity_theta: float = 0.2,
        per_residue: bool = False,
        seed: int = 42,
    ) -> ScoringResult:
        """Score candidate sequences, optionally conditioned on *prompt*.

        Parameters
        ----------
        sequences:
            Candidate amino-acid strings to score.
        prompt:
            Conditioning (family context) sequences.  *None* scores unconditionally.
        ensemble_size:
            Number of prompt sub-samples for ensemble scoring.
        max_tokens:
            Token budget for prompt+completion.
        scoring_max_tokens:
            Token budget used to set dynamic batch size.
        use_diversity_weights:
            Weight conditioning sequences by homology diversity.
        diversity_theta:
            Theta for homology neighbor definition.
        per_residue:
            If *True*, also return per-residue log-likelihoods.
        seed:
            Random seed.

        Returns
        -------
        ScoringResult
        """
        seed_all(seed)

        model = self._model

        # Tokenize candidate sequences
        comp_tok = model.tokenizer.encode_completions(
            sequences,
            bos_token=model.tokenizer.sep_token,
            eos_token=model.tokenizer.sep_token,
        )
        completion_ids = (
            torch.as_tensor(comp_tok["input_ids"], dtype=torch.long)
            .unsqueeze(0)
            .to(model.device)
        )

        tokenized_conditioning: list[list[int]] = []
        weights = None

        if prompt is not None and len(prompt) > 0:
            tokenized_conditioning = [
                model.tokenizer(
                    seq.upper().replace("-", "").replace(".", ""),
                    add_special_tokens=False,
                )["input_ids"]
                for seq in prompt
            ]

            if use_diversity_weights and len(prompt) > 1:
                from profam.data.msa_subsampling import (
                    compute_homology_weights,
                    encode_msa_sequences_to_uint8,
                )

                aligned = [seq.upper() for seq in prompt]
                encoded = encode_msa_sequences_to_uint8(aligned)
                _GAP_TOKEN_IDX = 20
                _, weights = compute_homology_weights(
                    ungapped_msa=encoded,
                    theta=diversity_theta,
                    gap_token=_GAP_TOKEN_IDX,
                    gap_token_mask=255,
                )

        with torch.no_grad():
            if len(tokenized_conditioning) > 0:
                lls = score_variants_ensemble(
                    model=model,
                    completion_ids=completion_ids,
                    tokenized_conditioning_sequences=tokenized_conditioning,
                    ensemble_size=ensemble_size,
                    scoring_max_tokens=scoring_max_tokens,
                    start_tokens=[47, 63],
                    max_tokens_override=max_tokens,
                    weights=weights,
                )
            else:
                lls = model._score_seqs_no_context(
                    completion_ids,
                    batch_size=max(
                        int(scoring_max_tokens) // completion_ids.shape[-1], 1
                    ),
                )

        residue_scores = None
        if per_residue:
            residue_scores = self._compute_residue_scores(
                completion_ids, tokenized_conditioning, max_tokens
            )

        return ScoringResult(
            sequences=sequences,
            scores=np.asarray(lls),
            residue_scores=residue_scores,
        )

    def _compute_residue_scores(
        self,
        completion_ids: torch.Tensor,
        tokenized_conditioning: list[list[int]],
        max_tokens: int,
    ) -> list[np.ndarray]:
        """Compute per-residue log-likelihoods for each sequence."""
        from profam.models.utils import log_likelihood_from_outputs

        model = self._model

        # Build a single prompt (first conditioning sequence or none)
        if len(tokenized_conditioning) > 0:
            start_tokens = [47, 63]
            prompt_ids_list = list(start_tokens)
            prompt_ids_list.extend(tokenized_conditioning[0])
            input_ids = torch.tensor(
                prompt_ids_list, dtype=torch.long, device=model.device
            ).unsqueeze(0)
        else:
            input_ids = None

        residue_scores: list[np.ndarray] = []
        if completion_ids.dim() == 3:
            completion_ids = completion_ids.squeeze(0)

        with torch.no_grad():
            for i in range(completion_ids.shape[0]):
                comp = completion_ids[i : i + 1]
                if input_ids is not None:
                    full_ids = torch.cat([input_ids, comp], dim=-1)
                    start_ix = input_ids.shape[-1]
                else:
                    full_ids = comp
                    start_ix = 1

                outputs = model.model(input_ids=full_ids, use_cache=False)
                labels = torch.where(
                    full_ids == model.tokenizer.pad_token_id,
                    -100,
                    full_ids.clone(),
                )
                ll = log_likelihood_from_outputs(outputs, labels, start_ix=start_ix)
                # ll shape: (1, L) — per-position log-likelihoods
                ll_np = ll[0].cpu().float().numpy()
                # Trim padding positions
                shift_labels = labels[..., start_ix + 1 :]
                mask = (shift_labels[0] != -100).cpu().numpy()
                residue_scores.append(ll_np[mask])

        return residue_scores
