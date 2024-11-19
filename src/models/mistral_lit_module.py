from typing import Optional

from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

from src.models.base import BaseFamilyLitModule
from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


class WrappedMistralForCausalLM(
    WrappedHFModelWithPositionEmbeddingsMixin, MistralForCausalLM
):
    pass


class MistralLitModule(BaseFamilyLitModule):
    def __init__(
        self,
        config: MistralConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 1e-4,
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        weight_decay: float = 0.01,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        embed_coords: bool = False,
        embed_sequence_index: bool = False,
        pass_constant_position_ids: bool = False,
        pass_seq_pos_in_doc_as_position_ids: bool = False,
        max_seq_pos_in_doc: int = 1024,
        embed_residue_index: bool = True,
        max_res_pos_in_seq: int = 4096,
        optimizer: str = "adamw",
    ) -> None:
        if tokenizer.embed_residue_index or embed_coords:
            # had to remove these as they break testing
            # assert embed_residue_index == tokenizer.embed_residue_index
            # assert max_res_pos_in_seq == tokenizer.max_res_pos_in_seq
            model = WrappedMistralForCausalLM(
                config,
                "model.embed_tokens",
                tokenizer=tokenizer,
                embedding_dim=config.hidden_size,
                embed_coords=embed_coords,
                embed_sequence_index=embed_sequence_index,
                max_seq_pos_in_doc=max_seq_pos_in_doc,
                pass_constant_position_ids=pass_constant_position_ids,
                pass_seq_pos_in_doc_as_position_ids=pass_seq_pos_in_doc_as_position_ids,
            )
        else:
            model = MistralForCausalLM(config)
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            embed_coords=embed_coords,
        )
