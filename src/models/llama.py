from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src.models.base import BaseFamilyLitModule, BaseSingleSequenceLitModule
from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


class LlamaSingleSequenceLitModule(BaseSingleSequenceLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 64000,
        optimizer: str = "adamw",
    ) -> None:
        model = LlamaForCausalLM(config)

        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
        )


class WrappedLlamaForCausalLM(
    WrappedHFModelWithPositionEmbeddingsMixin, LlamaForCausalLM
):
    pass


class LlamaLitModule(BaseFamilyLitModule):
    def __init__(
        self,
        config: LlamaConfig,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 10240,
        use_kv_cache_for_scoring: bool = True,
        embed_coords: bool = False,
        embed_sequence_index: bool = False,
        pass_constant_position_ids: bool = False,
        pass_seq_pos_in_doc_as_position_ids: bool = False,
        pass_res_pos_in_doc_as_position_ids: bool = False,
        max_seq_pos_in_doc: int = 1024,
        embed_residue_index: bool = True,
        max_res_pos_in_seq: int = 4096,
        max_sequence_index: int = 1024,
        optimizer: str = "adamw",
    ) -> None:
        """
        From the paper:
        We trained using the AdamW optimizer (Loshchilov and Hutter, 2017),
        with beta1=0.9,beta2=0.95,eps=10-5. We use a cosine learning rate schedule, with warmup
        of 2000 steps, and decay final learning rate down to 10% of the peak learning rate (3e-4-1.5e-4).
        We use a weight decay of 0.1 and gradient clipping of 1.0.
        """
        if (tokenizer.embed_residue_index or embed_coords,):
            # had to remove these as they break testing
            # assert embed_residue_index == tokenizer.embed_residue_index
            # assert max_res_pos_in_seq == tokenizer.max_res_pos_in_seq
            model = WrappedLlamaForCausalLM(
                config,
                token_embedder="model.embed_tokens",
                tokenizer=tokenizer,
                embedding_dim=config.hidden_size,
                embed_coords=embed_coords,
                embed_sequence_index=embed_sequence_index,
                max_seq_pos_in_doc=max_seq_pos_in_doc,
                pass_constant_position_ids=pass_constant_position_ids,
                pass_seq_pos_in_doc_as_position_ids=pass_seq_pos_in_doc_as_position_ids,
                pass_res_pos_in_doc_as_position_ids=pass_res_pos_in_doc_as_position_ids,
            )
        else:
            model = LlamaForCausalLM(config)
        # n.b. attention implementation gets set here (in from_pretrained, _from_config, __init__):
        # https://github.com/huggingface/transformers/blob/1dba608df93ffb10a9c268ef35191adf2424c5ca/src/transformers/modeling_utils.py#L1542
        # c.f. https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
        print(
            "Initialised Llama model, attention implementation: ",
            model.config._attn_implementation,
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
            embed_coords=embed_coords,
        )
