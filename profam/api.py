"""Public Python API for ProFam."""

from __future__ import annotations

import os
import re
import warnings
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
from profam.sequence.fasta import read_fasta
from profam.utils.utils import seed_all


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
        :class:`FamilyPrompt` (optionally weighted by homology
        diversity weights). Padding positions are masked out of both
        the sum and :math:`L_i`.

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


@dataclass
class FamilyPrompt:
    """A protein-family context with two co-registered views.

    Scoring needs two different representations of the family:

    * ``conditioning`` — variable-length, insertion residues kept, gaps
      removed. This is what gets tokenized and fed to the model as the
      prompt. The model was trained to see insertions, so they should be
      preserved here.
    * ``aligned`` — equal-length MSA columns, insertion residues stripped,
      gap characters (``-``) preserved. This is the only view in which
      column-wise Hamming similarity (and therefore homology-based
      diversity weights) is meaningful.

    The two lists must be parallel: ``conditioning[i]`` and ``aligned[i]``
    refer to the same family member. ``aligned`` may be ``None`` to signal
    that no aligned view is available, in which case diversity weights
    cannot be computed.

    Prefer the ``from_aligned`` / ``from_unaligned`` constructors over
    building one directly.

    Attributes
    ----------
    conditioning:
        Unaligned sequences (insertions kept, gaps removed) used as the
        tokenized model prompt.
    aligned:
        Equal-length aligned sequences used for homology diversity
        weights. ``None`` when no aligned view is available.
    source_path:
        If this prompt was built from a file, the path is retained so
        downstream consumers (e.g. ``ProFam.score(cache_weights=True)``)
        can use it as a cache key. ``None`` when the prompt was built
        from an in-memory list.
    """

    conditioning: List[str]
    aligned: Optional[List[str]] = None
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.aligned is not None:
            if len(self.aligned) != len(self.conditioning):
                raise ValueError(
                    "FamilyPrompt.aligned and FamilyPrompt.conditioning must be "
                    f"parallel lists of equal length; got "
                    f"len(aligned)={len(self.aligned)}, "
                    f"len(conditioning)={len(self.conditioning)}."
                )
            if len(self.aligned) > 0:
                lengths = {len(s) for s in self.aligned}
                if len(lengths) != 1:
                    raise ValueError(
                        "FamilyPrompt.aligned must contain equal-length "
                        f"sequences (aligned MSA columns); got lengths {sorted(lengths)}."
                    )

    def __len__(self) -> int:
        return len(self.conditioning)

    @property
    def has_alignment(self) -> bool:
        """Whether an aligned view is available (required for diversity weights)."""
        return self.aligned is not None and len(self.aligned) > 0

    @classmethod
    def from_aligned(
        cls, source: "str | os.PathLike | list[str]"
    ) -> "FamilyPrompt":
        """Build a ``FamilyPrompt`` from an aligned MSA.

        Parameters
        ----------
        source:
            Either a path to an aligned sequence file (FASTA, a2m, or a3m),
            or an in-memory ``list[str]`` of aligned sequences. Lowercase
            letters and ``.`` are treated as per-sequence insertions
            (a2m/a3m convention) and stripped when deriving the aligned
            view used for diversity weights; ``-`` and ``.`` are stripped
            when deriving the insertions-kept conditioning view used as
            the model prompt.

        Raises
        ------
        ValueError
            If, after stripping insertions, the sequences are not all the
            same length (i.e. ``source`` is not a valid aligned MSA).
        """
        source_path: Optional[str] = None
        if isinstance(source, (str, os.PathLike)):
            source_path = os.fspath(source)
            _, conditioning = read_fasta(
                source, keep_insertions=True, keep_gaps=False, to_upper=True
            )
            _, aligned = read_fasta(
                source, keep_insertions=False, keep_gaps=True, to_upper=True
            )
            source_desc = f"file {source_path!r}"
        elif isinstance(source, list):
            aligned = [_strip_insertions(s).upper() for s in source]
            conditioning = [_strip_gaps(s).upper() for s in source]
            source_desc = "the supplied list"
        else:
            raise TypeError(
                "FamilyPrompt.from_aligned expected a path or list[str]; "
                f"got {type(source).__name__}."
            )

        if aligned:
            lengths = {len(s) for s in aligned}
            if len(lengths) != 1:
                raise ValueError(
                    f"Cannot build an aligned FamilyPrompt from {source_desc}: "
                    "after stripping insertions (lowercase letters and '.') "
                    f"sequences have different lengths {sorted(lengths)}; this "
                    "is not a valid aligned MSA. Use "
                    "FamilyPrompt.from_unaligned(...) for unaligned sequences."
                )
        return cls(conditioning=conditioning, aligned=aligned, source_path=source_path)

    @classmethod
    def from_unaligned(
        cls, source: "str | os.PathLike | list[str]"
    ) -> "FamilyPrompt":
        """Build a ``FamilyPrompt`` from unaligned sequences.

        Parameters
        ----------
        source:
            Either a path to an unaligned FASTA file, or an in-memory
            ``list[str]`` of unaligned sequences. Gap-like characters
            (``-`` and ``.``) are stripped defensively; remaining residues
            are uppercased and used as the conditioning view.

        Notes
        -----
        No aligned view is produced, so diversity weights cannot be
        computed from the resulting prompt and
        ``use_diversity_weights=True`` will be downgraded with a warning
        when it is passed to ``ProFam.score``.
        """
        source_path: Optional[str] = None
        if isinstance(source, (str, os.PathLike)):
            source_path = os.fspath(source)
            _, sequences = read_fasta(
                source, keep_insertions=True, keep_gaps=False, to_upper=True
            )
            conditioning = sequences
        elif isinstance(source, list):
            conditioning = [_strip_gaps(s).upper() for s in source]
        else:
            raise TypeError(
                "FamilyPrompt.from_unaligned expected a path or list[str]; "
                f"got {type(source).__name__}."
            )
        return cls(
            conditioning=conditioning, aligned=None, source_path=source_path
        )


_GAP_PATTERN = re.compile(r"[-.]")
_INSERTION_PATTERN = re.compile(r"[a-z.]")


def _strip_gaps(s: str) -> str:
    return _GAP_PATTERN.sub("", s)


def _strip_insertions(s: str) -> str:
    return _INSERTION_PATTERN.sub("", s)


def _compute_diversity_weights(
    family_prompt: "FamilyPrompt",
    diversity_theta: float,
    cache_weights: bool,
    recompute_cached_weights: bool,
) -> np.ndarray:
    """Compute (or load from disk cache) homology diversity weights.

    When ``cache_weights=True`` the prompt must expose a ``source_path``
    so the cache file can be keyed to the source MSA; a list-backed
    prompt has no such path and therefore cannot be cached.
    """
    from profam.data.msa_subsampling import (
        compute_homology_sequence_weights_with_cache,
        compute_homology_weights,
        encode_msa_sequences_to_uint8,
    )

    if cache_weights:
        if family_prompt.source_path is None:
            raise ValueError(
                "cache_weights=True requires a FamilyPrompt constructed from "
                "a file path; diversity weights cannot be cached for a "
                "FamilyPrompt built from an in-memory sequence list. Either "
                "set cache_weights=False or use "
                "FamilyPrompt.from_aligned(<path>)."
            )
        return compute_homology_sequence_weights_with_cache(
            msa_file=family_prompt.source_path,
            sequences=family_prompt.aligned,
            theta=diversity_theta,
            force_recalc=recompute_cached_weights,
        )

    encoded = encode_msa_sequences_to_uint8(family_prompt.aligned)
    _GAP_TOKEN_IDX = 20
    _, weights = compute_homology_weights(
        ungapped_msa=encoded,
        theta=diversity_theta,
        gap_token=_GAP_TOKEN_IDX,
        gap_token_mask=255,
    )
    return weights


def _coerce_family_prompt(
    prompt: "FamilyPrompt | list[str] | None",
    use_diversity_weights: bool,
) -> "FamilyPrompt | None":
    """Normalise a user-supplied prompt into a :class:`FamilyPrompt`.

    Plain ``list[str]`` inputs are accepted for backwards compatibility:

    * If ``use_diversity_weights`` is *True* and every sequence has the
      same length, the list is treated as an aligned MSA (and a
      deprecation warning is emitted suggesting ``FamilyPrompt``).
    * Otherwise the list is treated as unaligned sequences; if the user
      asked for diversity weights, a warning is emitted explaining that
      they will not be applied.
    """
    if prompt is None:
        return None
    if isinstance(prompt, FamilyPrompt):
        return prompt
    if not isinstance(prompt, list):
        raise TypeError(
            "prompt must be a FamilyPrompt, a list[str], or None; "
            f"got {type(prompt).__name__}."
        )
    if len(prompt) == 0:
        return FamilyPrompt(conditioning=[], aligned=None)

    lengths = {len(s) for s in prompt}
    looks_aligned = len(lengths) == 1

    if use_diversity_weights and looks_aligned and len(prompt) > 1:
        warnings.warn(
            "Passing a list[str] to ProFam.score(prompt=...) is deprecated; "
            "build a FamilyPrompt with FamilyPrompt.from_aligned(...) or "
            "FamilyPrompt.from_unaligned(...) so the aligned and "
            "conditioning views are explicit. The list was interpreted as "
            "an aligned MSA because every sequence has the same length.",
            DeprecationWarning,
            stacklevel=3,
        )
        return FamilyPrompt.from_aligned(prompt)

    if use_diversity_weights and not looks_aligned:
        warnings.warn(
            "use_diversity_weights=True was requested, but the prompt "
            "list[str] contains sequences of different lengths and so "
            "cannot be an aligned MSA. Diversity weights will be "
            "disabled. Pass a FamilyPrompt built via "
            "FamilyPrompt.from_aligned(...) to provide an aligned view.",
            RuntimeWarning,
            stacklevel=3,
        )

    return FamilyPrompt.from_unaligned(prompt)


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
            ``result.sequences`` contains the decoded amino-acid
            strings; ``result.scores`` contains the mean per-token
            natural log-probability of each sampled completion under
            the model (see :class:`GenerationResult` for the exact
            definition).
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
        prompt: "FamilyPrompt | list[str] | None" = None,
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
            Conditioning family context. Prefer a :class:`FamilyPrompt`,
            which carries both an aligned view (used for diversity
            weights) and an unaligned view with insertions kept (used as
            the model prompt). A plain ``list[str]`` is accepted for
            backwards compatibility and is treated as unaligned
            sequences; in that case ``use_diversity_weights=True`` is
            downgraded to ``False`` with a warning unless every sequence
            already has the same length (in which case the list is
            assumed to be an aligned MSA). Pass ``None`` to score
            unconditionally.
        ensemble_size:
            Number of prompt sub-samples for ensemble scoring.
        max_tokens:
            Token budget for prompt+completion.
        scoring_max_tokens:
            Token budget used to set dynamic batch size.
        use_diversity_weights:
            Weight conditioning sequences by homology diversity. Requires
            an aligned view (i.e. a :class:`FamilyPrompt` built with
            ``from_aligned(...)``, or a ``list[str]`` of equal-length
            sequences).
        diversity_theta:
            Theta for homology neighbor definition.
        cache_weights:
            If *True*, cache the computed diversity weights on disk next to
            the source MSA file (``<basename>_weights.npz``). Requires the
            ``prompt`` to be a :class:`FamilyPrompt` whose ``source_path``
            is set — i.e. one built via ``FamilyPrompt.from_aligned(<path>)``.
            Raises ``ValueError`` if *True* and no source path is available
            (for example, when the prompt was built from an in-memory
            ``list[str]``).
        recompute_cached_weights:
            Only meaningful when ``cache_weights=True``; ignores any
            existing cache file and recomputes the weights.
        per_residue:
            If *True*, also return per-residue log-likelihoods.
        seed:
            Random seed.

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

        family_prompt = _coerce_family_prompt(prompt, use_diversity_weights)

        tokenized_conditioning: list[list[int]] = []
        weights = None

        if family_prompt is not None and len(family_prompt) > 0:
            tokenized_conditioning = [
                model.tokenizer(
                    _strip_gaps(seq).upper(),
                    add_special_tokens=False,
                )["input_ids"]
                for seq in family_prompt.conditioning
            ]

            if (
                use_diversity_weights
                and len(family_prompt) > 1
                and family_prompt.has_alignment
            ):
                weights = _compute_diversity_weights(
                    family_prompt=family_prompt,
                    diversity_theta=diversity_theta,
                    cache_weights=cache_weights,
                    recompute_cached_weights=recompute_cached_weights,
                )
            elif use_diversity_weights and not family_prompt.has_alignment:
                warnings.warn(
                    "use_diversity_weights=True was requested but the prompt "
                    "has no aligned view; diversity weights will not be "
                    "applied. Pass a FamilyPrompt built via "
                    "FamilyPrompt.from_aligned(...) to enable them.",
                    RuntimeWarning,
                    stacklevel=2,
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
