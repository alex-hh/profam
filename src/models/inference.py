from typing import Dict, List, Optional

import torch

from src.data.objects import Protein, ProteinDocument
from src.data.preprocessing import (
    BasePreprocessorConfig,
    preprocess_protein_sequences,
    subsample_and_tokenize_protein_data,
)
from src.models.base import BaseFamilyLitModule
from src.utils.tokenizers import ProFamTokenizer


class PromptBuilder:
    def __init__(
        self,
        preprocessor: BasePreprocessorConfig,
        max_tokens: int,
        seed: Optional[int] = None,
    ):
        self.preprocessor = preprocessor
        self.seed = seed
        self.max_tokens = max_tokens

    def __call__(self, proteins: ProteinDocument, tokenizer: ProFamTokenizer):
        proteins = preprocess_protein_sequences(proteins, self.preprocessor, tokenizer)
        max_length = max(len(seq) for seq in proteins.sequences)
        batch = subsample_and_tokenize_protein_data(
            proteins,
            cfg=self.preprocessor,
            tokenizer=tokenizer,
            shuffle=True,
            seed=self.seed,
            padding="longest",
            max_tokens=self.max_tokens - max_length,
        )  # a dictionary
        return batch


class ProFamSampler:
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        sampling_kwargs: Optional[Dict] = None,
    ):
        self.name = name
        self.model = model
        self.prompt_builder = prompt_builder
        self.sampling_kwargs = sampling_kwargs


# TODO: return type should be List[Protein]?
class ProFamSequenceSampler(ProFamSampler):
    def sample(self, protein_document: ProteinDocument, num_samples: int) -> List[str]:
        prompt = self.prompt_builder.build_prompt(
            protein_document, self.model.tokenizer
        )
        tokens = self.model._sample_seqs(
            prompt["input_ids"].unsqueeze(0).to(self.model.device),
            num_samples=num_samples,
            input_seq_pos=prompt["seq_pos"].unsqueeze(0).to(self.model.device),
            **self.sampling_kwargs
        )
        return self.model.tokenizer.decode_tokens(tokens)


class ProFusionStructureSampler(ProFamSampler):
    def sample(
        self, protein_document: ProteinDocument, num_samples: int, length: int
    ) -> torch.Tensor:
        prompt = self.prompt_builder.build_prompt(
            protein_document, self.model.tokenizer
        )
        # TODO: do we need completion seq pos? currently yes
        backbone_coords = self.model._sample_coords(
            input_ids=prompt["input_ids"].unsqueeze(0).to(self.model.device),
            num_samples=num_samples,
            input_coords=prompt["coords"].unsqueeze(0).to(self.model.device),
            length=length,
            input_seq_pos=prompt["seq_pos"].unsqueeze(0).to(self.model.device),
            **self.sampling_kwargs
        )
        return backbone_coords


class ProFusionJointSampler(ProFamSampler):
    pass
