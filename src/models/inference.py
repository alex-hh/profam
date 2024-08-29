from typing import Dict, Optional

import torch

from src.data.objects import ProteinDocument
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

    def to(self, device):
        self.model.to(device)

    def sample_seqs(self, protein_document: ProteinDocument, num_samples: int):
        prompt = self.prompt_builder(protein_document, self.model.tokenizer)
        with torch.no_grad():  # prob unnecessary
            tokens = self.model._sample_seqs(
                prompt["input_ids"].unsqueeze(0).to(self.model.device),
                num_samples=num_samples,
                input_seq_pos=prompt["seq_pos"].unsqueeze(0).to(self.model.device),
                **self.sampling_kwargs
            )
            return self.model.tokenizer.decode_tokens(tokens)
