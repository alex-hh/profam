from typing import Optional

import torch

from src.data import transforms
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
            proteins, self.preprocessor.cfg, tokenizer
        )
        proteins = transforms.apply_transforms(
            transform_fns, proteins, tokenizer, max_tokens=self.max_tokens - max_length
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
        )
        representative_doc = transforms.apply_transforms(
            transform_fns, representative_doc, tokenizer, max_tokens=None
        )
        representative_doc = representative_doc.slice_arrays(
            [slice(0, len(representative.sequence) + 1)]
        )
        return representative_doc


class ProFamInferenceModel:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        document_token: str = "[RAW]",
        checkpoint_path: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        assert prompt_builder is not None
        self.checkpoint_path = checkpoint_path
        self.document_token = document_token
        if self.checkpoint_path is not None:
            print(
                f"Initialising ProFam inference model, loading checkpoint {self.checkpoint_path}"
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

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: str,
        prompt_builder: PromptBuilder,
        name_suffix: str = "",
        **kwargs,
    ):
        # automatically load checkpoint path and, if possible, wandb run name
        raise NotImplementedError("Not implemented yet")
        return cls(
            model=model,
            prompt_builder=prompt_builder,
            sampling_kwargs=sampling_kwargs,
            checkpoint_path=checkpoint_dir,
        )
