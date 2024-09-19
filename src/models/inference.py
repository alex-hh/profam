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
        preprocessor: BasePreprocessor,  # n.b. only preprocessing cfg and transform fns actually matter
        max_tokens: int,
        seed: Optional[int] = None,
    ):
        super().__init__(preprocessor, max_tokens, seed)
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
        transform_fns = default_transforms(max_tokens=None, shuffle=False) + (
            self.preprocessor.transform_fns or []
        )
        representative_doc = preprocess_protein_sequences(
            representative_doc,
            self.preprocessor.cfg,
            tokenizer,
            transform_fns=transform_fns,
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
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        assert prompt_builder is not None
        self.sampling_kwargs = sampling_kwargs
        self.checkpoint_path = checkpoint_path
        self.document_token = document_token
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

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)

    def sample_seqs(
        self,
        protein_document: ProteinDocument,
        num_samples: int,
        document_is_prompt=False,
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
            add_final_sep=False,
        )

        with torch.no_grad():  # prob unnecessary
            tokens = self.model._sample_seqs(
                encoded["input_ids"].unsqueeze(0).to(self.model.device),
                max_tokens=self.prompt_builder.max_tokens,
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
