from typing import Optional

from transformers import PreTrainedTokenizerFast, MistralConfig, MistralForCausalLM

from src.models.base import BaseFamilyLitModule
from src.models.wrapper import TransformerWithSequencePositionEmbeddings


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
        use_seq_pos: bool = False,
        max_seq_pos: int = 2048,
    ) -> None:
        model = MistralForCausalLM(config)
        if (
            use_seq_pos
        ):
            model = TransformerWithSequencePositionEmbeddings(
                model,
                model.embed_tokens,
                embedding_dim=config.hidden_size,
                use_seq_pos=use_seq_pos,
                max_seq_pos=max_seq_pos,
            )
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
            use_seq_pos=use_seq_pos,
        )
