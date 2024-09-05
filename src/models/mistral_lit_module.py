from typing import Optional

from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

from src.models.base import BaseFamilyLitModule


class MistralLitModule(BaseFamilyLitModule):
    model_class = MistralForCausalLM

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
        pass_constant_position_ids_for_global_index: bool = False,
        pass_sequence_position_ids_for_global_index: bool = False,
        max_sequence_index: int = 1024,
    ) -> None:
        super().__init__(
            config,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
            use_kv_cache_for_scoring=use_kv_cache_for_scoring,
            embed_coords=embed_coords,
            embed_sequence_index=embed_sequence_index,
            max_sequence_index=max_sequence_index,
            pass_constant_position_ids_for_global_index=pass_constant_position_ids_for_global_index,
            pass_sequence_position_ids_for_global_index=pass_sequence_position_ids_for_global_index,
        )
