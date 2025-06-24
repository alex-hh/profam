from typing import Optional

import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src.models.base import BaseFamilyLitModule
from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


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
        pass_res_pos_in_seq_as_position_ids: bool = False,
        pass_res_pos_in_doc_as_position_ids: bool = True,
        max_seq_pos_in_doc: int = 1024,
        embed_residue_index: bool = True,
        max_res_pos_in_seq: int = 4096,
        max_sequence_index: int = 1024,
        optimizer: str = "adamw",
        override_optimizer_on_load: bool = False,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0,
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
                pass_res_pos_in_seq_as_position_ids=pass_res_pos_in_seq_as_position_ids,
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
            override_optimizer_on_load=override_optimizer_on_load,
        )

        # Dynamically inject focal loss if requested.
        if use_focal_loss:
            # Ensure a sensible default if the YAML passes null.
            gamma = focal_gamma if focal_gamma is not None else 2.0

            ignore_index = self.ignore_index  # from BaseLitModule

            def _focal_loss(
                logits,
                labels,
                vocab_size: int,
                num_items_in_batch: int = None,
                ignore_index: int = ignore_index,
                **kwargs,
            ):
                """Token-level focal loss for causal language modelling.

                Mirrors the default `ForCausalLMLoss` implementation but applies the
                focal scaling factor (\(1-p_t)^{\gamma}\)).
                """
                # Upcast for numerical stability
                logits_ = logits.float()
                labels_ = labels.to(logits_.device)

                # Shift tokens for next-token prediction
                shift_logits = logits_[..., :-1, :].contiguous()
                shift_labels = labels_[..., 1:].contiguous()

                # Flatten
                shift_logits = shift_logits.view(-1, vocab_size)
                shift_labels = shift_labels.view(-1)

                # Compute per-token CE loss (no reduction)
                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=ignore_index,
                    reduction="none",
                )

                # Filter out ignored positions
                valid_mask = shift_labels != ignore_index
                if valid_mask.any():
                    ce_loss = ce_loss[valid_mask]
                    pt = torch.exp(-ce_loss)  # (N,)
                    focal = ((1 - pt) ** gamma) * ce_loss
                else:
                    focal = ce_loss  # all ignored -> zero length

                if num_items_in_batch is not None and num_items_in_batch > 0:
                    return focal.sum() / num_items_in_batch
                else:
                    return focal.mean()

            # Attach to the underlying HF model so its `forward` picks it up.
            self.model.loss_function = _focal_loss
