import copy
from typing import Dict, Optional

import torch

from src.data.objects import ProteinDocument
from src.evaluators.models import SamplingModelForEvaluation
from src.models.base import BaseFamilyLitModule
from src.models.inference import ProFamInferenceModel, PromptBuilder


class ProFamSampler(ProFamInferenceModel, SamplingModelForEvaluation):
    def __init__(
        self,
        name: str,
        model: BaseFamilyLitModule,
        prompt_builder: PromptBuilder,
        document_token: str = "[RAW]",
        checkpoint_path: Optional[str] = None,
        sampling_kwargs: Optional[Dict] = None,
        match_representative_length: bool = False,
    ):
        super().__init__(
            name=name,
            model=model,
            prompt_builder=prompt_builder,
            document_token=document_token,
            checkpoint_path=checkpoint_path,
        )
        self.sampling_kwargs = sampling_kwargs
        self.match_representative_length = match_representative_length

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
            sequences = self.model.tokenizer.decode_tokens(tokens)
        return sequences, prompt
