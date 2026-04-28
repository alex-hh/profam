"""Public Python API for ProFam."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


@dataclass
class ConditioningPrompt:
    """A single family-context prompt actually fed to the model.

    The ``"single"`` sampler produces exactly one
    :class:`ConditioningPrompt` (the truncated prompt after
    token-budget and preprocessor transforms). The ``"ensemble"``
    sampler produces one :class:`ConditioningPrompt` per ensemble
    member.

    Attributes
    ----------
    sequences:
        The amino-acid sequences that made it into this prompt, in the
        order they were concatenated.
    accessions:
        Parallel list of accessions. If the caller supplied
        ``prompt_accessions`` to :meth:`ProFam.generate`, these carry
        the originals through (permutation included). Otherwise they
        are auto-generated placeholders like ``"seq_0"``.
    """

    sequences: List[str]
    accessions: List[str]


@dataclass
class GenerationResult:
    """Result of a sequence generation call.

    Attributes
    ----------
    sequences:
        The generated amino-acid sequences (as plain strings, with
        special tokens stripped), in the order they were produced.
    scores:
        One score per generated sequence, aligned with ``sequences``.
        Each score is the **mean per-token log-probability**
        of the sampled completion under the model, i.e.

        .. math::
           s = \\frac{1}{T} \\sum_{t=1}^{T} \\log p(x_t \\mid x_{<t}, \\text{prompt})

        where :math:`x_t` is the token actually sampled at step
        :math:`t`, :math:`p(\\cdot \\mid \\cdot)` is the model's
        next-token distribution at that step (after temperature
        scaling and bad-token masking, but *before* nucleus
        truncation), and :math:`T` is the number of tokens in the
        completion up to and including the terminal ``[SEP]`` token
        (if one was emitted). For the ensemble sampler, the aggregated
        distribution across prompt sub-samples is used in place of
        :math:`p`.

        Scores are :math:`\\le 0`; values closer to zero indicate the
        sampler was, on average, more confident in the tokens it drew,
        so they are a reasonable proxy for "typicality" under the
        model but are **not** comparable across sequences of very
        different length or against log-likelihoods produced by
        :meth:`ProFam.score`.
    """

    sequences: List[str]
    scores: List[float]
    conditioning_prompts: Optional[List[ConditioningPrompt]] = None


@dataclass
class ScoringResult:
    """Result of a sequence scoring call.

    Attributes
    ----------
    sequences:
        The candidate amino-acid sequences that were scored, in input
        order.
    scores:
        One score per candidate, as a ``(N,)`` float array aligned
        with ``sequences``. Each score is the **mean per-token
        log-likelihood** of the candidate under the model, averaged
        over an ensemble of prompt sub-samples:

        .. math::
           s_i = \\frac{1}{K} \\sum_{k=1}^{K}
                 \\frac{1}{L_i} \\sum_{t=1}^{L_i}
                 \\log p(x^{(i)}_t \\mid x^{(i)}_{<t}, \\text{prompt}_k)

        where :math:`x^{(i)}` is the :math:`i`-th candidate sequence
        (with ``[SEP]`` BOS/EOS wrapping), :math:`L_i` is its number
        of non-padding completion tokens, :math:`K` is
        ``ensemble_size``, and each :math:`\\text{prompt}_k` is a
        distinct random sub-sample drawn from the conditioning
        sequences (optionally weighted by homology diversity weights).
        Padding positions are masked out of both the sum and
        :math:`L_i`.

        When ``prompt`` is ``None`` no conditioning is used, the
        ensemble collapses to :math:`K = 1`, and the score reduces to
        the mean unconditional per-token log-likelihood.

        Scores are :math:`\\le 0`; higher (less negative) values mean
        the model assigns higher likelihood per residue to the
        candidate. Because the score is length-normalised it can be
        compared across sequences of different lengths, but absolute
        values still depend on the prompt and on ``ensemble_size``.
    residue_scores:
        Only populated when ``per_residue=True`` was passed to
        :meth:`ProFam.score`. A list with one entry per candidate
        sequence; each entry is a 1-D ``np.ndarray`` of shape
        ``(L_i,)`` giving the per-residue natural log-likelihood
        :math:`\\log p(x^{(i)}_t \\mid x^{(i)}_{<t}, \\text{prompt})`
        at every amino-acid position (padding stripped). Unlike
        ``scores``, these are computed from a **single** prompt (the
        first conditioning sequence, or no prompt if none was given)
        rather than averaged over the ensemble, so
        ``residue_scores[i].mean()`` will not exactly equal
        ``scores[i]``.
    """

    sequences: List[str]
    scores: np.ndarray
    residue_scores: Optional[List[np.ndarray]] = field(default=None)


_GAP_PATTERN = re.compile(r"[-.]")
_INSERTION_PATTERN = re.compile(r"[a-z.]")


def _strip_gaps(s: str) -> str:
    return _GAP_PATTERN.sub("", s)


def _strip_insertions(s: str) -> str:
    return _INSERTION_PATTERN.sub("", s)


def _resolve_prompt(
    prompt: "str | os.PathLike | list[str] | None",
    use_diversity_weights: bool,
) -> Tuple[List[str], Optional[List[str]], Optional[str]]:
    """Load conditioning and aligned views from a user-supplied prompt.

    Returns ``(conditioning, aligned_or_None, source_path_or_None)`` where:

    * ``conditioning`` — sequences with insertions kept and gaps removed,
      tokenized as the model prompt.
    * ``aligned_or_None`` — equal-length aligned sequences (insertions
      stripped, gaps preserved) used to compute homology diversity
      weights, or ``None`` when the input is not aligned.
    * ``source_path_or_None`` — the original file path when ``prompt``
      was a path, used as a cache key for diversity weights.

    Raises
    ------
    ValueError
        If ``use_diversity_weights=True`` but the input is not an
        aligned MSA (sequences differ in length after stripping
        a2m/a3m insertions).
    TypeError
        If ``prompt`` is not a path, list of strings, or ``None``.
    """
    if prompt is None:
        return [], None, None

    source_path: Optional[str] = None
    if isinstance(prompt, (str, os.PathLike)):
        source_path = os.fspath(prompt)
        _, aligned = read_fasta(
            prompt, keep_insertions=False, keep_gaps=True, to_upper=True
        )
        _, conditioning = read_fasta(
            prompt, keep_insertions=True, keep_gaps=False, to_upper=True
        )
    elif isinstance(prompt, list):
        aligned = [_strip_insertions(s).upper() for s in prompt]
        conditioning = [_strip_gaps(s).upper() for s in prompt]
    else:
        raise TypeError(
            "prompt must be a path, list[str], or None; "
            f"got {type(prompt).__name__}."
        )

    is_aligned = len(aligned) > 0 and len({len(s) for s in aligned}) == 1

    if use_diversity_weights and not is_aligned:
        raise ValueError(
            "use_diversity_weights=True requires aligned conditioning "
            "sequences (equal-length sequences after stripping a2m/a3m "
            "insertions), but the supplied prompt is not an aligned MSA. "
            "Either provide an aligned MSA, or pass "
            "use_diversity_weights=False to score without weighting."
        )

    return conditioning, (aligned if is_aligned else None), source_path


def _compute_diversity_weights(
    aligned: List[str],
    source_path: Optional[str],
    diversity_theta: float,
    cache_weights: bool,
    recompute_cached_weights: bool,
) -> np.ndarray:
    """Compute (or load from disk cache) homology diversity weights.

    When ``cache_weights=True`` a ``source_path`` must be available so
    the cache file can be keyed to the source MSA; an in-memory list
    has no such path and therefore cannot be cached.
    """
    from profam.data.msa_subsampling import (
        compute_homology_sequence_weights_with_cache,
        compute_homology_weights,
        encode_msa_sequences_to_uint8,
    )

    if cache_weights:
        if source_path is None:
            raise ValueError(
                "cache_weights=True requires a prompt supplied as a file "
                "path; diversity weights cannot be cached for an "
                "in-memory sequence list. Either set cache_weights=False "
                "or pass the path to the MSA file as the prompt."
            )
        return compute_homology_sequence_weights_with_cache(
            msa_file=source_path,
            sequences=aligned,
            theta=diversity_theta,
            force_recalc=recompute_cached_weights,
        )

    encoded = encode_msa_sequences_to_uint8(aligned)
    _GAP_TOKEN_IDX = 20
    _, weights = compute_homology_weights(
        ungapped_msa=encoded,
        theta=diversity_theta,
        gap_token=_GAP_TOKEN_IDX,
        gap_token_mask=255,
    )
    return weights


def _build_protein_document(
    sequences: list[str],
    accessions: Optional[list[str]] = None,
    identifier: str = "api_input",
) -> ProteinDocument:
    """Build a ProteinDocument from a plain list of amino acid strings.

    If ``accessions`` is omitted, placeholders ``seq_0``, ``seq_1`` …
    are generated. If provided, its length must match ``sequences``.
    """
    if accessions is None:
        accessions = [f"seq_{i}" for i in range(len(sequences))]
    elif len(accessions) != len(sequences):
        raise ValueError(
            f"accessions length {len(accessions)} does not match "
            f"sequences length {len(sequences)}"
        )
    rep = accessions[0] if accessions else "representative"
    return ProteinDocument(
        sequences=sequences,
        accessions=list(accessions),
        identifier=identifier,
        representative_accession=rep,
    )


def _prompt_doc_to_conditioning(doc: ProteinDocument) -> "ConditioningPrompt":
    """Convert a :class:`ProteinDocument` prompt to a :class:`ConditioningPrompt`."""
    seqs = list(doc.sequences)
    if getattr(doc, "accessions", None):
        accs = list(doc.accessions)
    else:
        accs = [f"prompt_{i}" for i in range(len(seqs))]
    return ConditioningPrompt(sequences=seqs, accessions=accs)


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
    >>> from profam import ProFam                                         # doctest: +SKIP
    >>> model = ProFam()                                                  # doctest: +SKIP
    >>> result = model.generate(prompt=["ACDEFGHIKLMNPQRSTVWY"], num_samples=5)  # doctest: +SKIP
    >>> print(result.sequences)                                           # doctest: +SKIP
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
        seed: int | None = None,
        prompt_accessions: list[str] | None = None,
    ) -> GenerationResult:
        """Generate novel protein sequences conditioned on *prompt*.

        Parameters
        ----------
        prompt:
            Family context sequences (plain amino-acid strings).
        prompt_accessions:
            Optional parallel list of accessions for ``prompt``; when
            supplied they are preserved through the sampler and surfaced
            on :attr:`GenerationResult.conditioning_prompts`. Defaults to
            placeholders ``seq_0``, ``seq_1``, …
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
        seed:
            Random seed for reproducibility.

        Returns
        -------
        GenerationResult
            ``result.sequences`` contains the decoded amino-acid
            strings; ``result.scores`` contains the mean per-token
            natural log-probability of each sampled completion under
            the model (see :class:`GenerationResult` for the exact
            definition). ``result.conditioning_prompts`` exposes the
            family-context prompts actually passed to the model — a
            single-element list in ``sampler="single"`` mode, and one
            :class:`ConditioningPrompt` per ensemble member in
            ``sampler="ensemble"`` mode.
        """
        if max_tokens > 8192:
            raise ValueError(
                "max_tokens must be <= 8192: model was trained up to 8192 tokens."
            )

        if seed is not None:
            seed_all(seed)

        doc_token = "[RAW]"
        pool = _build_protein_document(prompt, accessions=prompt_accessions)

        longest_prompt_len = int(max(pool.sequence_lengths))
        default_cap = int(longest_prompt_len * float(max_sequence_length_multiplier))
        if max_generated_length is None:
            max_gen_len = default_cap
        else:
            max_gen_len = min(int(max_generated_length), default_cap)

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
            sequences, scores, prompt_docs = sampler_obj.sample_seqs_ensemble(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                num_prompts_in_ensemble=min(
                    num_prompts_in_ensemble, len(pool.sequences)
                ),
                max_generated_length=max_gen_len,
                minimum_sequence_length_proportion=minimum_sequence_length_proportion,
                minimum_sequence_identity=minimum_sequence_identity,
                maximum_retries=maximum_retries,
                repeat_guard=repeat_guard,
            )
            conditioning_prompts = [_prompt_doc_to_conditioning(d) for d in prompt_docs]
        else:
            if max_gen_len is not None:
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
            sequences, scores, prompt_doc = sampler_obj.sample_seqs(
                protein_document=pool,
                num_samples=num_samples,
                max_tokens=max_tokens,
                max_generated_length=max_gen_len,
                minimum_sequence_length_proportion=minimum_sequence_length_proportion,
                minimum_sequence_identity=minimum_sequence_identity,
                maximum_retries=maximum_retries,
                repeat_guard=repeat_guard,
            )
            conditioning_prompts = [_prompt_doc_to_conditioning(prompt_doc)]

        return GenerationResult(
            sequences=sequences,
            scores=scores,
            conditioning_prompts=conditioning_prompts,
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(
        self,
        sequences: list[str],
        prompt: "str | os.PathLike | list[str] | None" = None,
        ensemble_size: int = 3,
        max_tokens: int = 8192,
        scoring_max_tokens: int = 64000,
        use_diversity_weights: bool = True,
        diversity_theta: float = 0.2,
        cache_weights: bool = False,
        recompute_cached_weights: bool = False,
        per_residue: bool = False,
        seed: int = 42,
    ) -> ScoringResult:
        """Score candidate sequences, optionally conditioned on *prompt*.

        Parameters
        ----------
        sequences:
            Candidate amino-acid strings to score.
        prompt:
            Conditioning family context. Either a path to a FASTA / a2m /
            a3m file, an in-memory ``list[str]`` of sequences, or
            ``None`` to score unconditionally. Whether the input is an
            aligned MSA is inferred: if every sequence has the same
            length after stripping a2m/a3m insertions, an aligned view
            is derived (insertions stripped, gaps preserved) for
            homology diversity weights, and a parallel unaligned view
            (insertions kept, gaps removed) is tokenized as the model
            prompt. Otherwise only the unaligned view is built and
            diversity weights are unavailable.
        ensemble_size:
            Number of prompt sub-samples for ensemble scoring.
        max_tokens:
            Token budget for prompt+completion.
        scoring_max_tokens:
            Token budget used to set dynamic batch size: reduce if you get
            OOM while scoring.
        use_diversity_weights:
            Weight conditioning sequences by homology diversity. Requires
            an aligned prompt; raises ``ValueError`` if the prompt is
            not an aligned MSA.
        diversity_theta:
            Theta for homology neighbor definition.
        cache_weights:
            If *True*, cache the computed diversity weights on disk next
            to the source MSA file (``<basename>_weights.npz``).
            Requires ``prompt`` to be a file path; raises ``ValueError``
            when the prompt is an in-memory ``list[str]``.
        recompute_cached_weights:
            Only meaningful when ``cache_weights=True``; ignores any
            existing cache file and recomputes the weights.
        per_residue:
            If *True*, also return per-residue log-likelihoods.
        seed:
            Random seed. Drives both the global RNG (via ``seed_all``)
            and the ensemble prompt sub-sampler in
            :func:`profam.scoring.score_variants_ensemble`, so different
            values draw different conditioning subsets.

        Returns
        -------
        ScoringResult
            ``result.scores[i]`` is the mean per-token natural
            log-likelihood of ``sequences[i]`` under the model,
            averaged over ``ensemble_size`` prompt sub-samples drawn
            from ``prompt`` (optionally weighted by homology diversity
            weights). When ``per_residue=True``,
            ``result.residue_scores[i]`` additionally contains the
            per-residue log-likelihoods computed from a single prompt.
            See :class:`ScoringResult` for the exact definition.
        """
        seed_all(seed)

        model = self._model

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

        conditioning, aligned, source_path = _resolve_prompt(
            prompt, use_diversity_weights
        )

        weights = None
        if use_diversity_weights and aligned is not None and len(aligned) > 1:
            weights = _compute_diversity_weights(
                aligned=aligned,
                source_path=source_path,
                diversity_theta=diversity_theta,
                cache_weights=cache_weights,
                recompute_cached_weights=recompute_cached_weights,
            )

        tokenized_conditioning = [
            model.tokenizer(
                _strip_gaps(seq).upper(),
                add_special_tokens=False,
            )["input_ids"]
            for seq in conditioning
        ]

        residue_scores: "list[np.ndarray] | None" = None
        with torch.no_grad():
            if len(tokenized_conditioning) > 0:
                if per_residue:
                    lls, residue_scores = score_variants_ensemble(
                        model=model,
                        completion_ids=completion_ids,
                        tokenized_conditioning_sequences=tokenized_conditioning,
                        ensemble_size=ensemble_size,
                        scoring_max_tokens=scoring_max_tokens,
                        start_tokens=[47, 63],
                        max_tokens_override=max_tokens,
                        weights=weights,
                        seed=seed,
                        return_per_residue=True,
                    )
                else:
                    lls = score_variants_ensemble(
                        model=model,
                        completion_ids=completion_ids,
                        tokenized_conditioning_sequences=tokenized_conditioning,
                        ensemble_size=ensemble_size,
                        scoring_max_tokens=scoring_max_tokens,
                        start_tokens=[47, 63],
                        max_tokens_override=max_tokens,
                        weights=weights,
                        seed=seed,
                    )
            else:
                no_context_batch_size = max(
                    int(scoring_max_tokens) // completion_ids.shape[-1], 1
                )
                if per_residue:
                    residue_scores = model.score_seqs(
                        None,
                        completion_ids,
                        batch_size=no_context_batch_size,
                        return_per_residue=True,
                    )
                    lls = np.array(
                        [float(r.mean()) if r.size > 0 else 0.0 for r in residue_scores]
                    )
                else:
                    lls = model._score_seqs_no_context(
                        completion_ids,
                        batch_size=no_context_batch_size,
                    )

        return ScoringResult(
            sequences=sequences,
            scores=np.asarray(lls),
            residue_scores=residue_scores,
        )
