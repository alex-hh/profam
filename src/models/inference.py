import copy
from typing import Dict, Optional

import torch

from src.constants import RESIDUE_LEVEL_FEATURES
from src.data.objects import ProteinDocument
from src.data.preprocessing import (
    BasePreprocessor,
    default_transforms,
    preprocess_protein_sequences,
)
from src.models.base import BaseFamilyLitModule
from src.utils.tokenizers import ProFamTokenizer


class PromptBuilder:
    def __init__(
        self,
        preprocessor: BasePreprocessor,
        max_tokens: int,
        seed: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        assert preprocessor is not None
        self.seed = seed
        self.max_tokens = max_tokens

    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        max_length = max(len(seq) for seq in proteins.sequences)
        transform_fns = default_transforms(
            max_tokens=self.max_tokens - max_length, shuffle=True, seed=self.seed
        ) + (self.preprocessor.transform_fns or [])
        proteins = preprocess_protein_sequences(
            proteins, self.preprocessor.cfg, tokenizer, transform_fns=transform_fns
        )
        # TODO: maybe just return the prompt, tokenize elsewhere.
        encoded = tokenizer.encode(
            proteins,
            document_token=self.preprocessor.cfg.document_token,
            padding="longest",
            add_final_sep=True,
        )
        return proteins, encoded


class InterleavedInverseFoldingPromptBuilder(PromptBuilder):
    """Prompt builder for interleaved inverse folding tasks.

    Instead of finishing with a sep, we finish with a structure sequence sep
    We also know the ground truth sequence: the representative sequence in the protein
    document: ProteinDocument.representative.sequence
    """

    def __init__(
        self,
        preprocessor: BasePreprocessor,  # n.b. only preprocessing cfg and transform fns actually matter
        max_tokens: int,
        seed: Optional[int] = None,
        representative_only: bool = False,
    ):
        super().__init__(preprocessor, max_tokens, seed)
        self.representative_only = representative_only
        assert self.preprocessor.interleave_structure_sequence

    # we need to exclude token space for length seed*2 from preprocessing
    # TODO: write tests for this
    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        proteins = proteins.clone()
        representative = proteins.pop_representative()
        if not self.representative_only:
            raise NotImplementedError("Not implemented yet")
            max_tokens = self.max_tokens - len(representative)
            transform_fns = default_transforms(
                max_tokens=max_tokens, shuffle=False, seed=self.seed
            ) + (self.preprocessor.transform_fns or [])
            proteins = preprocess_protein_sequences(
                proteins, self.preprocessor.cfg, tokenizer, transform_fns=transform_fns
            )
            example = tokenizer.encode(
                proteins,
                document_token=self.preprocessor.cfg.document_token,
                padding="longest",
                add_final_sep=False,
            )

        # TODO: tokenize representative
        representative_doc = ProteinDocument.from_proteins([representative])
        transform_fns = default_transforms(max_tokens=None, shuffle=False) + (
            self.preprocessor.transform_fns or []
        )
        representative_doc = preprocess_protein_sequences(
            representative_doc,
            self.preprocessor.cfg,
            tokenizer,
            transform_fns=transform_fns,
        )
        representative_example = tokenizer.encode(
            representative_doc,
            document_token=self.preprocessor.cfg.document_token,
            padding="longest",
            add_final_sep=False,
        )
        seq_start = (
            torch.argwhere(
                representative_example["input_ids"] == tokenizer.seq_struct_sep_token_id
            ).min()
            + 1
        )
        assert (
            representative_example["input_ids"][:seq_start][-1]
            == tokenizer.seq_struct_sep_token_id
        ), f"seq_start: {seq_start}, input_ids: {representative_example['input_ids'][:seq_start]}"
        # TODO: maybe just return the prompt, tokenize elsewhere. Difficult thing is slicing out target sequence.
        for feat in representative_example.keys():
            if feat in RESIDUE_LEVEL_FEATURES:
                representative_example[feat] = representative_example[feat][:seq_start]
                if not self.representative_only:
                    raise NotImplementedError("Not implemented yet")
                    # representative_example["feat"] = torch.cat(
                    #     (example[feat], representative_example[feat]), dim=1
                    # )
        return representative_doc, representative_example


class ProFamSampler:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        match_representative_length: bool = False,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        assert prompt_builder is not None
        self.sampling_kwargs = sampling_kwargs
        self.checkpoint_path = checkpoint_path
        self.match_representative_length = match_representative_length
        if self.checkpoint_path is not None:
            print(
                f"Initialising ProFam sampler, loading checkpoint {self.checkpoint_path}"
            )
            checkpoint = torch.load(
                self.checkpoint_path, map_location=self.model.device
            )["state_dict"]
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def sample_seqs(self, protein_document: ProteinDocument, num_samples: int):
        prompt, encoded = self.prompt_builder(protein_document, self.model.tokenizer)
        sampling_kwargs = copy.deepcopy(self.sampling_kwargs or {})
        if self.match_representative_length:
            sampling_kwargs["fixed_length"] = len(
                protein_document.representative.sequence
            )
        with torch.no_grad():  # prob unnecessary
            tokens = self.model._sample_seqs(
                encoded["input_ids"].unsqueeze(0).to(self.model.device),
                num_samples=num_samples,
                input_seq_pos=encoded["seq_pos"].unsqueeze(0).to(self.model.device),
                input_coords=encoded["coords"]
                .unsqueeze(0)
                .to(self.model.device)
                .float()
                if self.model.embed_coords
                else None,  # n.b. preprocessing will produce coords for every input even when missing - careful about this
                **sampling_kwargs,
            )
            return self.model.tokenizer.decode_tokens(tokens), prompt

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str,
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
        name_suffix: str = "",
    ):
        # automatically load checkpoint path and, if possible, wandb run name
        raise NotImplementedError("Not implemented yet")
        return cls(
            model=model,
            prompt_builder=prompt_builder,
            sampling_kwargs=sampling_kwargs,
            checkpoint_path=checkpoint_dir,
        )
