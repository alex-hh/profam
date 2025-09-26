import copy
import functools
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.preprocessing import (
    ProteinDocumentPreprocessor,
    default_transforms,
)
from src.data.tokenizers import ProFamTokenizer
from src.models.base import BaseFamilyLitModule
from src.constants import aa_letters, aa_letters_lower

def has_too_many_repeats(seq: str, repeat_length:int = 9, repeat_count:int = 9) -> bool:
    """
    heuristic to detect failed sampling by checking for 
    repeated segments.
    """
    if len(seq) < repeat_length * repeat_count:
        return False

    substring = seq[-repeat_length:]
    # find all occurrences of the substring
    if seq.count(substring) >= repeat_count:
        return True
    return False


def _mmseqs_best_identity(prompt_sequences: List[str], query_sequences: List[str], threads: int = 1) -> List[float]:
    """Compute best identity per query sequence against prompt sequences using mmseqs easy-search.

    Returns a list of floats in [0,1], one per query, representing the maximum percent identity
    (converted to fraction) across all target prompt sequences. Queries with no hits get 0.0.
    """
    if len(query_sequences) == 0:
        return []
    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is None:
        raise RuntimeError("mmseqs binary not found in PATH; required for identity filtering")

    with tempfile.TemporaryDirectory() as tmpdir:
        target_fa = os.path.join(tmpdir, "targets.fasta")
        query_fa = os.path.join(tmpdir, "queries.fasta")
        result_tsv = os.path.join(tmpdir, "res.tsv")
        # Write FASTAs with unique IDs
        with open(target_fa, "w") as ft:
            for i, seq in enumerate(prompt_sequences):
                ft.write(f">t{i}\n{seq}\n")
        with open(query_fa, "w") as fq:
            for i, seq in enumerate(query_sequences):
                fq.write(f">q{i}\n{seq}\n")

        # Run mmseqs easy-search; outputs TSV
        # Fields: query,target,pident (we only need these)
        cmd = [
            mmseqs_bin,
            "easy-search",
            query_fa,
            target_fa,
            result_tsv,
            tmpdir,
            "--threads",
            str(int(threads)),
            "--format-output",
            "query,target,pident",
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"mmseqs easy-search failed: {e}")

        # Parse results; mmseqs may output multiple hits per query -> take max pident
        best: Dict[str, float] = {}
        if os.path.exists(result_tsv):
            with open(result_tsv, "r") as fr:
                for line in fr:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    qid, _tid, pident = parts[0], parts[1], parts[2]
                    try:
                        pid = float(pident) / 100.0
                    except ValueError:
                        continue
                    prev = best.get(qid, 0.0)
                    if pid > prev:
                        best[qid] = pid
        # Map back to query order
        out: List[float] = []
        for i in range(len(query_sequences)):
            out.append(float(best.get(f"q{i}", 0.0)))
        return out


class PromptBuilder:
    def __init__(
        self,
        preprocessor: ProteinDocumentPreprocessor,
        seed: Optional[int] = None,
        prompt_is_aligned: bool = False,
    ):
        self.preprocessor = preprocessor
        assert preprocessor is not None
        self.seed = seed
        self.prompt_is_aligned = prompt_is_aligned

    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        proteins = self.preprocessor.apply_transforms(proteins, tokenizer, rng=rng)
        return proteins


class FewShotInterleavedInverseFoldingPromptBuilder(PromptBuilder):
    pass


class InterleavedInverseFoldingPromptBuilder(PromptBuilder):
    """Prompt builder for interleaved inverse folding tasks.

    Instead of finishing with a sep, we finish with a structure sequence sep
    We also know the ground truth sequence: the representative sequence in the protein
    document: ProteinDocument.representative.sequence
    """

    def __init__(
        self,
        preprocessor: ProteinDocumentPreprocessor,  # n.b. only preprocessing cfg and transform fns actually matter
        seed: Optional[int] = None,
        prompt_is_aligned: bool = False,
    ):
        super().__init__(preprocessor, seed, prompt_is_aligned=prompt_is_aligned)
        assert self.preprocessor.interleave_structure_sequence

    # we need to exclude token space for length seed*2 from preprocessing
    # TODO: write tests for this
    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        proteins = proteins.clone()
        representative = proteins.representative

        # We want to interleave the structure with an empty sequence
        # for now a hack to do this is to replace the sequence with an empty sequence
        representative_doc = ProteinDocument.from_proteins(
            [representative], representative_accession=representative.accession
        )
        _preprocessor_single_protein_documents = (
            self.preprocessor.single_protein_documents
        )
        self.preprocessor.single_protein_documents = True
        rng = np.random.default_rng(self.seed) if self.seed is not None else None
        representative_doc = self.preprocessor.apply_transforms(
            representative_doc, tokenizer, rng=rng
        )
        self.preprocessor.single_protein_documents = (
            _preprocessor_single_protein_documents
        )

        representative_doc = representative_doc.slice_arrays(
            [slice(0, len(representative.sequence) + 1)]
        )
        return representative_doc


class ProFamSampler:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        document_token: str = "[RAW]",
        sampling_kwargs: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        match_representative_length: bool = False,
        dtype: Optional[torch.dtype] = None,
        add_final_sep: bool = False,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        assert prompt_builder is not None
        self.sampling_kwargs = sampling_kwargs
        self.checkpoint_path = checkpoint_path
        self.document_token = document_token
        self.match_representative_length = match_representative_length
        self.dtype = dtype or torch.float32
        self.add_final_sep = add_final_sep

        if hasattr(self.model, "dtype") and self.model.dtype is None:
            self.model.dtype = self.dtype
        if self.checkpoint_path is not None:
            print(
                f"Initialising ProFam sampler, loading checkpoint {self.checkpoint_path}"
            )
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.model.device
            )["state_dict"]
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)

    def sample_seqs(
        self,
        protein_document: ProteinDocument,
        num_samples: int,
        max_tokens: int,
        document_is_prompt=False,
        max_generated_length: int = None,
        continuous_sampling: bool = False,
        minimum_sequence_length_proportion: Optional[float] = None,
        minimum_sequence_identity: Optional[float] = None,
        maximum_retries: int = 5,
        repeat_guard: bool = True,
        repeat_length: int = 9,
        repeat_count: int = 9,
        repeat_guard_max_restarts: int = 3,
    ):
        assert not (repeat_guard and continuous_sampling), "Repeat guard and continuous sampling are not supported together"
        sampling_kwargs = copy.deepcopy(self.sampling_kwargs or {})
        if self.match_representative_length:
            sampling_kwargs["fixed_length"] = len(
                protein_document.representative.sequence
            )

        if document_is_prompt:
            raise NotImplementedError("We need to infer original sequence length...")
        else:
            prompt = self.prompt_builder(protein_document, self.model.tokenizer)
        encoded = self.model.tokenizer.encode(
            prompt,
            document_token=self.document_token,
            padding="longest",
            add_final_sep=self.add_final_sep,  # add sep token prior to this point
        )
        # Convert numpy arrays to torch tensors if needed
        for key in encoded:
            if isinstance(encoded[key], np.ndarray):
                encoded[key] = torch.from_numpy(encoded[key])

        # Prepare filter thresholds
        prompt_sequences = list(prompt.sequences)
        min_prompt_len = min((len(s) for s in prompt_sequences), default=0)

        # Identity/length filtering handled in mmseqs batch per round below

        accepted_sequences: List[str] = []
        accepted_scores: List[float] = []
        max_sequence_identities: List[float] = []
        target = int(num_samples)
        rounds = 0
        with torch.no_grad():
            if not continuous_sampling:
                while len(accepted_sequences) < target and rounds <= int(maximum_retries):
                    need = target - len(accepted_sequences)
                    tokens, scores = self.model._sample_seqs(
                        encoded["input_ids"].unsqueeze(0).to(self.model.device),
                        max_tokens=max_tokens,
                        max_generated_length=max_generated_length,
                        num_samples=need,
                        input_residue_index=encoded["residue_index"].unsqueeze(0).to(self.model.device),
                        input_coords=encoded["coords"].unsqueeze(0).to(self.model.device).to(self.dtype) if self.model.embed_coords else None,
                        continuous_sampling=False,
                        repeat_guard=repeat_guard,
                        repeat_length=repeat_length,
                        repeat_count=repeat_count,
                        repeat_guard_max_restarts=repeat_guard_max_restarts,
                        **sampling_kwargs,
                    )
                    batch_seqs = self.model.tokenizer.decode_tokens(tokens)
                    # Length pre-filter
                    len_ok_mask = [True] * len(batch_seqs)
                    if minimum_sequence_length_proportion is not None:
                        min_len = int(min_prompt_len * float(minimum_sequence_length_proportion))
                        len_ok_mask = [len(s) >= min_len for s in batch_seqs]
                    # Identity via mmseqs on those that passed length
                    idents: List[float] = [0.0] * len(batch_seqs)
                    idx_map: List[int] = [i for i, ok in enumerate(len_ok_mask) if ok]
                    if len(idx_map) > 0 and (minimum_sequence_identity is not None):
                        queries = [batch_seqs[i] for i in idx_map]
                        id_vals = _mmseqs_best_identity(prompt_sequences, queries, threads=1)
                        for j, i in enumerate(idx_map):
                            idents[i] = id_vals[j]
                    # Accept
                    for i, seq in enumerate(batch_seqs):
                        passes_len = len_ok_mask[i]
                        passes_id = True if minimum_sequence_identity is None else (idents[i] >= float(minimum_sequence_identity))
                        if passes_len and passes_id:
                            accepted_sequences.append(seq)
                            accepted_scores.append(float(scores[i] if isinstance(scores, list) else scores))
                            max_sequence_identities.append(idents[i])
                            if len(accepted_sequences) >= target:
                                break
                    rounds += 1

                # Fallback: generate remaining without filters
                if len(accepted_sequences) < target:
                    need = target - len(accepted_sequences)
                    tokens, scores = self.model._sample_seqs(
                        encoded["input_ids"].unsqueeze(0).to(self.model.device),
                        max_tokens=max_tokens,
                        max_generated_length=max_generated_length,
                        num_samples=need,
                        input_residue_index=encoded["residue_index"].unsqueeze(0).to(self.model.device),
                        input_coords=encoded["coords"].unsqueeze(0).to(self.model.device).to(self.dtype) if self.model.embed_coords else None,
                        continuous_sampling=False,
                        repeat_guard=False,
                        **sampling_kwargs,
                    )
                    batch_seqs = self.model.tokenizer.decode_tokens(tokens)
                    for i, seq in enumerate(batch_seqs):
                        accepted_sequences.append(seq)
                        accepted_scores.append(float(scores[i] if isinstance(scores, list) else scores))
                        max_sequence_identities.append(idents[i])
                        if len(accepted_sequences) >= target:
                            break
            else:
                # Continuous mode: one-by-one, but batch identity per attempt when multiple segments emitted
                while len(accepted_sequences) < target and rounds <= int(maximum_retries):
                    need = target - len(accepted_sequences)
                    tokens, scores = self.model._sample_seqs(
                        encoded["input_ids"].unsqueeze(0).to(self.model.device),
                        max_tokens=max_tokens,
                        max_generated_length=max_generated_length,
                        num_samples=1,
                        input_residue_index=encoded["residue_index"].unsqueeze(0).to(self.model.device),
                        input_coords=encoded["coords"].unsqueeze(0).to(self.model.device).to(self.dtype) if self.model.embed_coords else None,
                        continuous_sampling=True,
                        repeat_guard=False,
                        repeat_guard_max_restarts=0,
                        **sampling_kwargs,
                    )
                    sep_id = self.model.tokenizer.sep_token_id
                    row = tokens[0]
                    seg_texts: List[str] = []
                    seg_scores: List[float] = []
                    if not (row == self.model.tokenizer.pad_token_id).all():
                        sep_positions = (row == sep_id).nonzero(as_tuple=False).flatten().tolist()
                        prev = 0
                        score_ix = 0
                        for sep_pos in sep_positions:
                            seg = row[prev:sep_pos]
                            if seg.numel() > 0:
                                text = self.model.tokenizer.decode(seg.tolist(), skip_special_tokens=True).replace(" ", "")
                                if text:
                                    seg_texts.append(text)
                                    val = float(scores[score_ix]) if isinstance(scores, list) and score_ix < len(scores) else float("nan")
                                    seg_scores.append(val)
                                    score_ix += 1
                            prev = sep_pos + 1
                    # Length pre-filter
                    keep_idxs = list(range(len(seg_texts)))
                    if minimum_sequence_length_proportion is not None:
                        min_len = int(min_prompt_len * float(minimum_sequence_length_proportion))
                        keep_idxs = [i for i in keep_idxs if len(seg_texts[i]) >= min_len]
                    # Identity batch via mmseqs
                    if len(keep_idxs) > 0 and (minimum_sequence_identity is not None):
                        id_vals = _mmseqs_best_identity(prompt_sequences, [seg_texts[i] for i in keep_idxs], threads=1)
                        keep_idxs = [i for i, idv in zip(keep_idxs, id_vals) if idv >= float(minimum_sequence_identity)]
                    for i in keep_idxs:
                        accepted_sequences.append(seg_texts[i])
                        accepted_scores.append(seg_scores[i])
                        if len(accepted_sequences) >= target:
                            break
                    rounds += 1

                if len(accepted_sequences) < target:
                    # Fallback: generate remaining without filters
                    need = target - len(accepted_sequences)
                    tokens, scores = self.model._sample_seqs(
                        encoded["input_ids"].unsqueeze(0).to(self.model.device),
                        max_tokens=max_tokens,
                        max_generated_length=max_generated_length,
                        num_samples=need,
                        input_residue_index=encoded["residue_index"].unsqueeze(0).to(self.model.device),
                        input_coords=encoded["coords"].unsqueeze(0).to(self.model.device).to(self.dtype) if self.model.embed_coords else None,
                        continuous_sampling=False,
                        **sampling_kwargs,
                    )
                    batch_seqs = self.model.tokenizer.decode_tokens(tokens)
                    for i, seq in enumerate(batch_seqs):
                        accepted_sequences.append(seq)
                        accepted_scores.append(float(scores[i] if isinstance(scores, list) else scores))
                        if len(accepted_sequences) >= target:
                            break

        return accepted_sequences, accepted_scores, prompt

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str,
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
        name_suffix: str = "",
        dtype: Optional[torch.dtype] = None,
    ):
        # automatically load checkpoint path and, if possible, wandb run name
        raise NotImplementedError("Not implemented yet")
        return cls(
            model=model,
            prompt_builder=prompt_builder,
            sampling_kwargs=sampling_kwargs,
            checkpoint_path=checkpoint_dir,
            dtype=dtype,
        )


class EnsemblePromptBuilder:
    """
    Build multiple prompt variants (subsamples of an MSA/family) that each
    fit under a token budget. Assumes no coords/residue index embedding.
    """

    def __init__(
        self,
        preprocessor: ProteinDocumentPreprocessor,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def _estimate_prompt_tokens(self, sequences: List[str], tokenizer: ProFamTokenizer) -> int:
        extra_tokens_per_protein = 1  # sep token per sequence
        return tokenizer.num_start_tokens + sum(len(s) + extra_tokens_per_protein for s in sequences)

    def _choose_indices_under_budget(
        self,
        proteins: ProteinDocument,
        tokenizer: ProFamTokenizer,
        max_tokens: int,
    ) -> List[int]:
        n = len(proteins.sequences)
        order = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(order)
        chosen: List[int] = []
        total = tokenizer.num_start_tokens
        for i in order:
            prospective = total + len(proteins.sequences[i]) + 1
            if prospective <= max_tokens:
                chosen.append(int(i))
                total = prospective
            else:
                # Try to ensure at least one sequence fits
                if len(chosen) == 0 and (tokenizer.num_start_tokens + len(proteins.sequences[i]) + 1) <= max_tokens:
                    chosen.append(int(i))
                break
        if len(chosen) == 0:
            raise ValueError("No sequences fit within the provided max_tokens")
        return chosen

    def build_variants(
        self,
        proteins: ProteinDocument,
        tokenizer: ProFamTokenizer,
        num_prompts_in_ensemble: int,
        max_tokens: int,
        sample_context_length: bool = True,
        use_clustering: bool = True,
    ) -> List[ProteinDocument]:
        # Helper to cluster sequences using mmseqs easy-cluster
        def _cluster_with_mmseqs(seqs: List[str], min_seq_id: float = 0.3, coverage: float = 0.7, threads: int = 1) -> Dict[int, int]:
            mapping: Dict[int, int] = {}
            mmseqs_bin = shutil.which("mmseqs")
            if mmseqs_bin is None:
                return mapping  # empty indicates no clustering available
            with tempfile.TemporaryDirectory() as tmpdir:
                fasta_path = os.path.join(tmpdir, "input.fasta")
                with open(fasta_path, "w") as f:
                    for i, s in enumerate(seqs):
                        f.write(f">s{i}\n{s}\n")
                out_prefix = os.path.join(tmpdir, "cluster")
                cmd = [
                    mmseqs_bin, "easy-cluster",
                    fasta_path,
                    out_prefix,
                    out_prefix,
                    "--min-seq-id", str(float(min_seq_id)),
                    "-c", str(float(coverage)),
                    "--threads", str(int(threads)),
                    "--remove-tmp-files", "1",
                    "--cluster-mode", "1",
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    return {}
                cluster_tsv = f"{out_prefix}_cluster.tsv"
                if not os.path.exists(cluster_tsv):
                    return {}
                rep_to_cid: Dict[str, int] = {}
                next_cid = 0
                with open(cluster_tsv, "r") as fr:
                    for line in fr:
                        parts = line.strip().split("\t")
                        if len(parts) < 2:
                            continue
                        rep, mem = parts[0], parts[1]
                        if rep not in rep_to_cid:
                            rep_to_cid[rep] = next_cid
                            next_cid += 1
                        cid = rep_to_cid[rep]
                        if mem.startswith("s"):
                            try:
                                idx = int(mem[1:])
                            except Exception:
                                continue
                            mapping[idx] = cid
                        # also map representative itself if present as rep id style "sX"
                        if rep.startswith("s"):
                            try:
                                idx_r = int(rep[1:])
                                mapping.setdefault(idx_r, cid)
                            except Exception:
                                pass
                # Any sequences not present in mapping -> singleton clusters
                for i in range(len(seqs)):
                    mapping.setdefault(i, next_cid)
                    if mapping[i] == next_cid:
                        next_cid += 1
                return mapping

        # Prepare (normalize) once; do not sample-to-max here
        prepared = self.preprocessor.apply_transforms(proteins, tokenizer, rng=self.rng)
        variants: List[ProteinDocument] = []
        max_context_tokens = max_tokens - int(np.max(prepared.sequence_lengths) * 1.2)
        max_context_tokens = max(max_context_tokens, 0)

        if not use_clustering:
            for _ in range(num_prompts_in_ensemble):
                if sample_context_length:
                    low = int(np.max(prepared.sequence_lengths))
                    high = int(max_context_tokens + 1)
                    if high <= low:
                        this_context_tokens = low
                    else:
                        this_context_tokens = int(self.rng.integers(low, high))
                    this_context_tokens = max(int(this_context_tokens), int(max_context_tokens // 2))
                else:
                    this_context_tokens = max_context_tokens
                idxs = self._choose_indices_under_budget(prepared, tokenizer, this_context_tokens)
                perm = np.array(idxs)
                if self.shuffle:
                    self.rng.shuffle(perm)
                variants.append(prepared[perm.tolist()])
            return variants

        # Simplified selection prioritizing equal max length across variants
        seqs = list(prepared.sequences)
        lengths = np.array([len(s) for s in seqs], dtype=int)
        idx_to_cluster: Dict[int, int] = _cluster_with_mmseqs(seqs, min_seq_id=0.3, coverage=0.7, threads=1) if use_clustering else {}

        # Choose a shared per-variant token budget
        if sample_context_length:
            low = int(np.max(prepared.sequence_lengths))
            high = int(max_context_tokens + 1)
            if high <= low:
                this_context_tokens = low
            else:
                this_context_tokens = int(self.rng.integers(low, high))
            this_context_tokens = max(int(this_context_tokens), int(max_context_tokens // 2))
        else:
            this_context_tokens = max_context_tokens

        # Target the anchor length near the 90th percentile but within budget
        target_len = int(np.percentile(lengths, 90)) if len(lengths) > 0 else 0
        target_len = min(target_len, max(1, this_context_tokens - 1))

        # Pick anchors: closest to target_len, prefer different clusters if available
        order = np.argsort(np.abs(lengths - target_len)).tolist()
        used_anchor_clusters: set = set()
        anchors: List[int] = []
        for i in order:
            if len(anchors) >= num_prompts_in_ensemble:
                break
            cid = idx_to_cluster.get(int(i), None)
            if len(idx_to_cluster) > 0:
                if cid in used_anchor_clusters:
                    continue
                used_anchor_clusters.add(cid)
            anchors.append(int(i))
        # If not enough unique-cluster anchors, fill remaining by closeness
        if len(anchors) < num_prompts_in_ensemble:
            for i in order:
                if len(anchors) >= num_prompts_in_ensemble:
                    break
                if int(i) not in anchors:
                    anchors.append(int(i))

        # Build each variant starting from its anchor, then greedily fill under budget
        for k in range(num_prompts_in_ensemble):
            total = int(tokenizer.num_start_tokens)
            chosen: List[int] = []
            used_clusters: set = set()

            anchor_idx = anchors[min(k, len(anchors) - 1)] if len(anchors) > 0 else None
            if anchor_idx is not None:
                anchor_tokens = int(lengths[anchor_idx] + 1)
                if total + anchor_tokens <= this_context_tokens:
                    chosen.append(int(anchor_idx))
                    total += anchor_tokens
                    if len(idx_to_cluster) > 0:
                        used_clusters.add(idx_to_cluster.get(int(anchor_idx)))

            # Greedy fill: prioritize unseen clusters, shortest-first to pack more
            pool = np.argsort(lengths).tolist()
            # First pass: distinct clusters
            if len(idx_to_cluster) > 0:
                for i in pool:
                    if i in chosen:
                        continue
                    cid = idx_to_cluster.get(int(i))
                    if cid in used_clusters:
                        continue
                    tokens_i = int(lengths[i] + 1)
                    if total + tokens_i > this_context_tokens:
                        continue
                    chosen.append(int(i))
                    used_clusters.add(cid)
                    total += tokens_i
            # Second pass: allow repeats if budget remains
            for i in pool:
                if total >= this_context_tokens:
                    break
                if i in chosen:
                    continue
                tokens_i = int(lengths[i] + 1)
                if total + tokens_i > this_context_tokens:
                    continue
                chosen.append(int(i))
                total += tokens_i

            perm = np.array(chosen)
            if self.shuffle:
                self.rng.shuffle(perm)
            variants.append(prepared[perm.tolist()])

        return variants


class EnsembleDecoder:
    """
    Step-wise ensemble decoding: average next-token distributions across prompt variants.
    Enforces no coords/residue index/sequence index embeddings.
    """

    def __init__(
        self,
        model: BaseFamilyLitModule,
        tokenizer: ProFamTokenizer,
        reduction: str = "mean_probs",
        sample_gaps: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        # Enforce constraints
        if getattr(model, "embed_coords", False):
            raise ValueError("embed_coords must be False for EnsembleDecoder")
        if getattr(model, "embed_sequence_index", False):
            raise ValueError("embed_sequence_index must be False for EnsembleDecoder")
        if getattr(model, "embed_residue_index", False):
            raise ValueError("embed_residue_index must be False for EnsembleDecoder")

        self.model = model
        self.tokenizer = tokenizer
        self.reduction = reduction
        self.sample_gaps = sample_gaps
        self.temperature = temperature
        self.top_p = top_p

        if self.top_p is not None:
            if not (0.0 < float(self.top_p) <= 1.0):
                raise ValueError("top_p must be in the interval (0, 1]")

        self._bad_token_ids = self._compute_bad_token_ids()
        self._eos_id = tokenizer.sep_token_id
        self._pad_id = tokenizer.pad_token_id

    def _compute_bad_token_ids(self) -> List[int]:
        bad_aas = ["X", "x", "B", "J", "O", "U", "Z"] + aa_letters_lower
        if not self.sample_gaps:
            bad_aas.append("-")

        bad_ids = [tid for tid in self.tokenizer.all_special_ids if tid != self.model.tokenizer.sep_token_id]
        bad_ids += [self.tokenizer.convert_tokens_to_ids(tok) for tok in bad_aas]
        # unique and valid
        bad_ids = [int(i) for i in set(bad_ids) if i is not None and i >= 0]
        return bad_ids

    def _mask_bad_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        if len(self._bad_token_ids) == 0:
            return logits
        logits[..., self._bad_token_ids] = float("-inf")
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: List[torch.Tensor],
        max_generated_length: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        continuous_sampling: bool = False,
        # Repeat-guard parameters
        repeat_guard: bool = True,
        repeat_length: int = 9,
        repeat_count: int = 9,
        repeat_guard_max_restarts: int = 3,
    ) -> torch.Tensor:
        device = self.model.device
        # Ensure tensors are on device
        input_ids = [ids.to(device) for ids in input_ids]
        eos = self.model.tokenizer.sep_token_id if eos_token_id is None else eos_token_id
        assert not (repeat_guard and continuous_sampling), "Repeat guard and continuous sampling are not supported together"
        # Helper to prepare per-variant state: independent KV caches and lengths (no padding)
        B = len(input_ids)
        def _init_states() -> List[Dict]:
            variant_states_local: List[Dict] = []
            for b in range(B):
                ids_b_full = input_ids[b]
                eff_len = int(ids_b_full.shape[-1])
                ids_b = ids_b_full.unsqueeze(0)  # (1, L_b)
                am_b = torch.ones((1, eff_len), dtype=torch.long, device=device)
                pos_b = am_b.long().cumsum(dim=-1) - 1

                outputs_b = self.model.model(
                    input_ids=ids_b,
                    attention_mask=am_b,
                    position_ids=pos_b,
                    use_cache=True,
                )
                pkv_b = outputs_b.past_key_values
                logits_last_b = outputs_b.logits[0, eff_len - 1, :]  # (V)
                variant_states_local.append(
                    {
                        "length": eff_len,
                        "past": pkv_b,
                        "logits": logits_last_b,
                    }
                )
            return variant_states_local

        variant_states: List[Dict] = _init_states()

        completions: List[int] = []
        total_logp: float = 0.0
        token_count: int = 0
        segment_scores: List[float] = []
        seg_total: float = 0.0
        seg_count: int = 0
        step = 0
        restarts_done: int = 0
        guard_active: bool = bool(repeat_guard)
        while True:
            # Compute per-variant next-token distributions
            probs_list: List[torch.Tensor] = []
            for state in variant_states:
                logits_v = state["logits"].unsqueeze(0)  # (1, V)
                logits_v = self._mask_bad_tokens(logits_v)
                if self.temperature is not None and self.temperature > 0:
                    logits_v = logits_v / float(self.temperature)
                probs_v = F.softmax(logits_v, dim=-1)[0]  # (V)
                probs_list.append(probs_v)

            # Aggregate across variants
            if self.reduction == "sum_log_probs":
                log_probs_stack = torch.stack([torch.log(p + 1e-12) for p in probs_list], dim=0)  # (B, V)
                agg = torch.mean(log_probs_stack, dim=0)
                agg_probs = F.softmax(agg, dim=-1)
            else:
                agg_probs = torch.mean(torch.stack(probs_list, dim=0), dim=0)

            # Sample next token (optionally using nucleus sampling)
            if self.top_p is not None and 0.0 < float(self.top_p) < 1.0:
                # Nucleus sampling over aggregated distribution
                sorted_probs, sorted_idx = torch.sort(agg_probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                keep = cumsum <= float(self.top_p)
                keep[..., 0] = True # Ensure at least the highest-prob token is included
                candidate_probs = sorted_probs[keep]
                candidate_indices = sorted_idx[keep]
                candidate_probs = candidate_probs / candidate_probs.sum()
                sampled_local = torch.multinomial(candidate_probs, num_samples=1).squeeze(-1)
                token_id = int(candidate_indices[sampled_local].item())
                token_prob = float(candidate_probs[sampled_local].item())
            else:
                sampled = torch.multinomial(agg_probs, num_samples=1).squeeze(-1)
                token_id = int(sampled.item())
                token_prob = float(agg_probs[token_id].item())
            completions.append(token_id)

            # Accumulate per-step log prob (include SEP)
            token_logp = float(torch.log(torch.tensor(token_prob + 1e-12)).item())
            if not continuous_sampling:
                total_logp += token_logp
                token_count += 1
            else:
                seg_total += token_logp
                seg_count += 1
                if token_id == eos:
                    # end segment; record mean log-prob for this segment
                    segment_scores.append(seg_total / max(seg_count, 1))
                    seg_total = 0.0
                    seg_count = 0

            # Repeat-guard: check for excessive repeats and optionally restart rollout
            if guard_active:
                # Decode current completion as AA string
                try:
                    seq_so_far = self.model.tokenizer.decode(completions, skip_special_tokens=True).replace(" ", "")
                except Exception:
                    seq_so_far = ""
                if seq_so_far and has_too_many_repeats(seq_so_far, repeat_length=repeat_length, repeat_count=repeat_count):
                    restarts_done += 1
                    if restarts_done <= int(repeat_guard_max_restarts):
                        # Reset rollout state and start again from scratch
                        completions = []
                        total_logp = 0.0
                        token_count = 0
                        segment_scores = []
                        seg_total = 0.0
                        seg_count = 0
                        step = 0
                        variant_states = _init_states()
                        # Continue with fresh rollout
                        continue
                    else:
                        # Exhausted restarts; disable guard and continue sampling to completion
                        guard_active = False

            # Termination
            step += 1
            if (not continuous_sampling) and token_id == eos:
                break
            if max_generated_length is not None and step >= max_generated_length:
                break

            # Feed sampled token to each variant independently, advancing its cache
            for vi in range(B):
                state = variant_states[vi]
                curr_len = int(state["length"])
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)  # (1,1)
                am_next = torch.ones((1, curr_len + 1), dtype=torch.long, device=device)
                pos_next = torch.tensor([[curr_len]], dtype=torch.long, device=device)
                outputs_next = self.model.model(
                    input_ids=step_ids,
                    past_key_values=state["past"],
                    attention_mask=am_next,
                    position_ids=pos_next,
                    use_cache=True,
                )
                # Update state
                state["past"] = outputs_next.past_key_values
                state["length"] = curr_len + 1
                state["logits"] = outputs_next.logits[0, -1, :]

        if len(completions) == 0:
            gen = torch.empty((0,), dtype=torch.long, device=device)
        else:
            if (not continuous_sampling) and completions[-1] == eos:
                completions = completions[:-1]
            gen = torch.tensor(completions, dtype=torch.long, device=device)
        if not continuous_sampling:
            mean_logp = total_logp / max(token_count, 1)
            return gen, mean_logp
        else:
            return gen, segment_scores


class ProFamEnsembleSampler:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: EnsemblePromptBuilder,
        document_token: str = "[RAW]",
        checkpoint_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        reduction: str = "mean_probs",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        add_final_sep: bool = False,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        self.document_token = document_token
        self.checkpoint_path = checkpoint_path
        self.dtype = dtype or torch.float32
        self.reduction = reduction
        self.temperature = temperature
        self.top_p = top_p
        self.add_final_sep = add_final_sep

        if getattr(self.model, "embed_coords", False):
            raise ValueError("embed_coords must be False for ProFamEnsembleSampler")
        if getattr(self.model, "embed_sequence_index", False):
            raise ValueError("embed_sequence_index must be False for ProFamEnsembleSampler")
        if getattr(self.model, "embed_residue_index", False):
            raise ValueError("embed_residue_index must be False for ProFamEnsembleSampler")

        if hasattr(self.model, "dtype") and self.model.dtype is None:
            self.model.dtype = self.dtype
        if self.checkpoint_path is not None:
            print(
                f"Initialising ProFamEnsembleSampler, loading checkpoint {self.checkpoint_path}"
            )
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.model.device
            )["state_dict"]
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)

    def _encode_variants(self, variants: List[ProteinDocument]) -> List[torch.Tensor]:
        # Encode each variant without padding and return list of tensors
        encoded_list: List[torch.Tensor] = []
        for v in variants:
            enc = self.model.tokenizer.encode(
                v,
                document_token=self.document_token,
                padding="do_not_pad",
                add_final_sep=self.add_final_sep,
            )
            input_ids = torch.as_tensor(enc["input_ids"], dtype=torch.long)
            encoded_list.append(input_ids)
        return encoded_list

    def sample_seqs_ensemble(
        self,
        protein_document: ProteinDocument,
        num_samples: int,
        max_tokens: int,
        num_prompts_in_ensemble: int,
        max_generated_length: Optional[int] = None,
        continuous_sampling: bool = False,
        minimum_sequence_length_proportion: Optional[float] = None,
        minimum_sequence_identity: Optional[float] = None,
        maximum_retries: int = 5,
        # Repeat-guard parameters
        repeat_guard: bool = True,
        repeat_length: int = 9,
        repeat_count: int = 9,
        repeat_guard_max_restarts: int = 3,
    ) -> Tuple[List[str], List[float], List[ProteinDocument]]:
        assert not (repeat_guard and continuous_sampling), "Repeat guard and continuous sampling are not supported together"
        # Build prompt variants
        variants = self.prompt_builder.build_variants(
            protein_document,
            self.model.tokenizer,
            num_prompts_in_ensemble=num_prompts_in_ensemble,
            max_tokens=max_tokens,
        )
        # Also prepare prompt sequences (post-transforms) for filtering thresholds
        prepared = self.prompt_builder.preprocessor.apply_transforms(protein_document, self.model.tokenizer, rng=self.prompt_builder.rng)
        prompt_sequences = list(prepared.sequences)
        min_prompt_len = min((len(s) for s in prompt_sequences), default=0)
        # Encode prompts without padding
        prompt_ids_list = self._encode_variants(variants)

        decoder = EnsembleDecoder(
            model=self.model,
            tokenizer=self.model.tokenizer,
            reduction=self.reduction,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Identity/length filtering handled in mmseqs batch per round below

        sequences: List[str] = []
        scores_out: List[float] = []
        target = int(num_samples)
        rounds = 0
        if not continuous_sampling:
            while len(sequences) < target and rounds <= int(maximum_retries):
                need = target - len(sequences)
                cand_seqs: List[str] = []
                cand_scores: List[float] = []
                for _ in range(need):
                    gen_ids, gen_scores = decoder.generate(
                        input_ids=prompt_ids_list,
                        max_generated_length=max_generated_length,
                        eos_token_id=self.model.tokenizer.sep_token_id,
                        continuous_sampling=False,
                        repeat_guard=repeat_guard,
                        repeat_length=repeat_length,
                        repeat_count=repeat_count,
                        repeat_guard_max_restarts=repeat_guard_max_restarts,
                    )
                    if isinstance(gen_ids, torch.Tensor) and gen_ids.numel() == 0:
                        continue
                    seq = self.model.tokenizer.decode_tokens(gen_ids.unsqueeze(0))[0]
                    cand_seqs.append(seq)
                    cand_scores.append(float(gen_scores if isinstance(gen_scores, (int, float)) else gen_scores))
                # Length filter
                len_ok = [True] * len(cand_seqs)
                if minimum_sequence_length_proportion is not None:
                    min_len = int(min_prompt_len * float(minimum_sequence_length_proportion))
                    len_ok = [len(s) >= min_len for s in cand_seqs]
                # Identity batch via mmseqs
                idents: List[float] = [0.0] * len(cand_seqs)
                idx_map = [i for i, ok in enumerate(len_ok) if ok]
                if len(idx_map) > 0 and (minimum_sequence_identity is not None):
                    id_vals = _mmseqs_best_identity(prompt_sequences, [cand_seqs[i] for i in idx_map], threads=1)
                    for j, i in enumerate(idx_map):
                        idents[i] = id_vals[j]
                # Accept
                for i, seq in enumerate(cand_seqs):
                    passes_len = len_ok[i]
                    passes_id = True if minimum_sequence_identity is None else (idents[i] >= float(minimum_sequence_identity))
                    if passes_len and passes_id:
                        sequences.append(seq)
                        scores_out.append(cand_scores[i])
                        if len(sequences) >= target:
                            break
                rounds += 1

            # Fallback: fill remaining with last-round generations ignoring filters
            if len(sequences) < target:
                need = target - len(sequences)
                for _ in range(need):
                    gen_ids, gen_scores = decoder.generate(
                        input_ids=prompt_ids_list,
                        max_generated_length=max_generated_length,
                        eos_token_id=self.model.tokenizer.sep_token_id,
                        continuous_sampling=False,
                        repeat_guard=False,
                    )
                    if isinstance(gen_ids, torch.Tensor) and gen_ids.numel() == 0:
                        sequences.append("")
                        scores_out.append(float("nan"))
                    else:
                        seq = self.model.tokenizer.decode_tokens(gen_ids.unsqueeze(0))[0]
                        sequences.append(seq)
                        scores_out.append(float(gen_scores if isinstance(gen_scores, (int, float)) else gen_scores))
        else:
            # Continuous mode: generate one at a time; batch identity per attempt across segments
            while len(sequences) < target and rounds <= int(maximum_retries):
                gen_ids, gen_scores = decoder.generate(
                    input_ids=prompt_ids_list,
                    max_generated_length=max_generated_length,
                    eos_token_id=self.model.tokenizer.sep_token_id,
                    continuous_sampling=True,
                    repeat_guard=False, # no repeat guard in continuous mode because repititions across sequences are likely to occur
                    repeat_guard_max_restarts=0,
                )
                if isinstance(gen_ids, torch.Tensor) and gen_ids.numel() == 0:
                    rounds += 1
                    continue
                sep_id = self.model.tokenizer.sep_token_id
                row = gen_ids
                sep_positions = (row == sep_id).nonzero(as_tuple=False).flatten().tolist()
                prev = 0
                segs: List[str] = []
                seg_scores: List[float] = []
                seg_idx = 0
                for sep_pos in sep_positions:
                    seg = row[prev:sep_pos]
                    if seg.numel() > 0:
                        text = self.model.tokenizer.decode(seg.tolist(), skip_special_tokens=True).replace(" ", "")
                        if text:
                            segs.append(text)
                            if isinstance(gen_scores, list) and seg_idx < len(gen_scores):
                                seg_scores.append(float(gen_scores[seg_idx]))
                                seg_idx += 1
                            else:
                                seg_scores.append(float("nan"))
                    prev = sep_pos + 1
                # Length filter
                keep = list(range(len(segs)))
                if minimum_sequence_length_proportion is not None:
                    min_len = int(min_prompt_len * float(minimum_sequence_length_proportion))
                    keep = [i for i in keep if len(segs[i]) >= min_len]
                # Identity via mmseqs in batch
                if len(keep) > 0 and (minimum_sequence_identity is not None):
                    id_vals = _mmseqs_best_identity(prompt_sequences, [segs[i] for i in keep], threads=1)
                    keep = [i for i, idv in zip(keep, id_vals) if idv >= float(minimum_sequence_identity)]
                for i in keep:
                    sequences.append(segs[i])
                    scores_out.append(seg_scores[i])
                    if len(sequences) >= target:
                        break
                rounds += 1
            if len(sequences) < target:
                # Fallback: generate remaining single sequences without filters
                need = target - len(sequences)
                for _ in range(need):
                    gen_ids, gen_scores = decoder.generate(
                        input_ids=prompt_ids_list,
                        max_generated_length=max_generated_length,
                        eos_token_id=self.model.tokenizer.sep_token_id,
                        continuous_sampling=False,
                        repeat_guard=False,
                    )
                    if isinstance(gen_ids, torch.Tensor) and gen_ids.numel() == 0:
                        sequences.append("")
                        scores_out.append(float("nan"))
                    else:
                        seq = self.model.tokenizer.decode_tokens(gen_ids.unsqueeze(0))[0]
                        sequences.append(seq)
                        scores_out.append(float(gen_scores if isinstance(gen_scores, (int, float)) else gen_scores))

        return sequences, scores_out, variants
