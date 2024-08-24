from typing import Dict, Optional

from src.data.objects import ProteinDocument
from src.data.preprocessing import (
    BasePreprocessorConfig,
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
        self.interleave_structure_sequence = preprocessor.get(
            "interleave_structure_tokens", False
        )

    def __call__(self, protein_document: ProteinDocument, tokenizer: ProFamTokenizer):
        if self.interleave_structure_sequence:
            max_tokens = max_tokens // 2  # TODO: account for sep
        return subsample_and_tokenize_protein_data(
            protein_document.sequences,
            cfg=self.preprocessor,
            tokenizer=tokenizer,
            coords=protein_document.backbone_coords,
            plddts=protein_document.plddts,
            structure_tokens=protein_document.structure_tokens,
            max_tokens=max_tokens,
            shuffle=True,
            seed=self.seed,
            interleave_structure_tokens=self.interleave_structure_sequence,
        )  # a dictionary


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

    def sample_seqs(self, protein_document: ProteinDocument, num_samples: int):
        prompt = self.prompt_builder.build_prompt(
            protein_document, self.model.tokenizer
        )
        return self.model._sample_seqs(
            prompt["input_ids"].unsqueeze(0).to(self.model.device),
            num_samples=num_samples,
            input_seq_pos=prompt["seq_pos"].unsqueeze(0).to(self.model.device),
            **self.sampling_kwargs
        )
