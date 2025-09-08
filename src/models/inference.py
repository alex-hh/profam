import copy
import functools
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
        proteins = self.preprocessor.apply_transforms(proteins, tokenizer)
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
        representative_doc = self.preprocessor.apply_transforms(
            representative_doc, tokenizer
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
    ):
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
            add_final_sep=False,  # add sep token prior to this point
        )
        # Convert numpy arrays to torch tensors if needed
        for key in encoded:
            if isinstance(encoded[key], np.ndarray):
                encoded[key] = torch.from_numpy(encoded[key])

        with torch.no_grad():  # prob unnecessary
            tokens = self.model._sample_seqs(
                encoded["input_ids"].unsqueeze(0).to(self.model.device),
                max_tokens=max_tokens,
                max_generated_length=max_generated_length,
                num_samples=num_samples,
                input_residue_index=encoded["residue_index"]
                .unsqueeze(0)
                .to(self.model.device),
                input_coords=encoded["coords"]
                .unsqueeze(0)
                .to(self.model.device)
                .to(self.dtype)
                if self.model.embed_coords
                else None,  # n.b. preprocessing will produce coords for every input even when missing - careful about this
                **sampling_kwargs,
            )
            sequences = self.model.tokenizer.decode_tokens(tokens)
        return sequences, prompt

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
        num_variants: int,
        max_tokens: int,
    ) -> List[ProteinDocument]:
        # Prepare (normalize) once; do not sample-to-max here
        prepared = self.preprocessor.apply_transforms(proteins, tokenizer)
        variants: List[ProteinDocument] = []
        for _ in range(num_variants):
            idxs = self._choose_indices_under_budget(prepared, tokenizer, max_tokens)
            # Preserve order within chosen set but randomly permute for diversity
            perm = np.array(idxs)
            if self.shuffle:
                self.rng.shuffle(perm)
            variant_doc = prepared[perm.tolist()]
            variants.append(variant_doc)
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
    ) -> torch.Tensor:
        device = self.model.device
        # Ensure tensors are on device
        input_ids = [ids.to(device) for ids in input_ids]
        eos = self.model.tokenizer.sep_token_id if eos_token_id is None else eos_token_id

        # Prepare per-variant state: independent KV caches and lengths (no padding)
        B = len(input_ids)
        variant_states: List[Dict] = []

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
            variant_states.append(
                {
                    "length": eff_len,
                    "past": pkv_b,
                    "logits": logits_last_b,
                }
            )

        completions: List[int] = []
        step = 0
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

            sampled = torch.multinomial(agg_probs, num_samples=1).squeeze(-1)
            token_id = int(sampled.item())
            completions.append(token_id)

            # Termination
            step += 1
            if token_id == eos:
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
            return torch.empty((0,), dtype=torch.long, device=device)
        if completions[-1] == eos:
            completions = completions[:-1]
        return torch.tensor(completions, dtype=torch.long, device=device)


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
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        self.document_token = document_token
        self.checkpoint_path = checkpoint_path
        self.dtype = dtype or torch.float32
        self.reduction = reduction
        self.temperature = temperature

        # Enforce embedding constraints eagerly
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
                add_final_sep=True,
            )
            input_ids = torch.as_tensor(enc["input_ids"], dtype=torch.long)
            encoded_list.append(input_ids)
        return encoded_list

    def sample_seqs_ensemble(
        self,
        protein_document: ProteinDocument,
        num_samples: int,
        max_tokens: int,
        num_variants: int,
        max_generated_length: Optional[int] = None,
    ) -> Tuple[List[str], List[ProteinDocument]]:
        # Build prompt variants
        variants = self.prompt_builder.build_variants(
            protein_document,
            self.model.tokenizer,
            num_variants=num_variants,
            max_tokens=max_tokens,
        )
        # Encode prompts without padding
        prompt_ids_list = self._encode_variants(variants)

        decoder = EnsembleDecoder(
            model=self.model,
            tokenizer=self.model.tokenizer,
            reduction=self.reduction,
            temperature=self.temperature,
        )

        sequences: List[str] = []
        for _ in range(num_samples):
            gen_ids = decoder.generate(
                input_ids=prompt_ids_list,
                max_generated_length=max_generated_length,
                eos_token_id=self.model.tokenizer.sep_token_id,
            )
            if gen_ids.numel() == 0:
                sequences.append("")
            else:
                seq = self.model.tokenizer.decode_tokens(gen_ids.unsqueeze(0))[0]
                sequences.append(seq)

        return sequences, variants
