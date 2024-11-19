import copy
import functools
from typing import Dict, Optional

import torch

from src.data.objects import ProteinDocument
from src.data.processors import transforms
from src.data.processors.preprocessing import (
    ProteinDocumentPreprocessor,
    default_transforms,
)
from src.data.tokenizers import ProFamTokenizer
from src.models.base import BaseFamilyLitModule


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
                input_residue_index=encoded["residue_index"]
                .unsqueeze(0)
                .to(self.model.device),
                input_coords=encoded["coords"]
                .unsqueeze(0)
                .to(self.model.device)
                .float()
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
    ):
        # automatically load checkpoint path and, if possible, wandb run name
        raise NotImplementedError("Not implemented yet")
        return cls(
            model=model,
            prompt_builder=prompt_builder,
            sampling_kwargs=sampling_kwargs,
            checkpoint_path=checkpoint_dir,
        )
