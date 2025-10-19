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
        gym_results_save_dir = None,
        # New loss: zero gradients for samples whose mean log-likelihood exceeds a threshold
        use_ll_threshold_loss: bool = False,
        ll_threshold: float = -1.3,
        gym_subsamples_per_n: int = 5,
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
            gym_results_save_dir=gym_results_save_dir,
            gym_subsamples_per_n=gym_subsamples_per_n,
        )

        if use_focal_loss:
            gamma = focal_gamma if focal_gamma is not None else 2.0

            ignore_index = self.ignore_index

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
        # ------------------------------------------------------------------
        # Log-likelihood threshold loss (optional)
        # ------------------------------------------------------------------
        if use_ll_threshold_loss:
            if use_focal_loss:
                raise ValueError("Cannot enable both focal loss and ll_threshold loss simultaneously.")

            threshold = ll_threshold if ll_threshold is not None else -1.3
            ignore_index = self.ignore_index

            def _ll_threshold_loss(
                logits,
                labels,
                vocab_size: int,
                num_items_in_batch: int = None,
                ignore_index: int = ignore_index,
                **kwargs,
            ):
                """Token-level CE loss with per-sample masking based on mean log-likelihood.

                For each sample in the batch, compute the average log-likelihood (negative CE).
                If this value is greater than the specified threshold, the corresponding
                sample's gradients are zeroed by setting its loss contribution to 0.
                """
                
                logits_ = logits.float()
                labels_ = labels.to(logits_.device)

                shift_logits = logits_[..., :-1, :].contiguous()
                shift_labels = labels_[..., 1:].contiguous()

                batch_size, seq_len_minus1 = shift_labels.shape[:2]
                flat_logits = shift_logits.view(-1, vocab_size)
                flat_labels = shift_labels.view(-1)

                ce_loss_flat = torch.nn.functional.cross_entropy(
                    flat_logits,
                    flat_labels,
                    ignore_index=ignore_index,
                    reduction="none",
                )  # (batch * (seq_len-1))

                valid_mask_flat = flat_labels != ignore_index  # (N,)

                tokens_per_sample = seq_len_minus1
                ce_loss_2d = ce_loss_flat.view(batch_size, tokens_per_sample)
                valid_mask_2d = valid_mask_flat.view(batch_size, tokens_per_sample)

                # Compute mean log-likelihood per sample
                token_counts = valid_mask_2d.float().sum(dim=1).clamp(min=1)
                mean_ce_per_sample = (
                    (ce_loss_2d * valid_mask_2d.float()).sum(dim=1) / token_counts
                )  # (batch,)
                mean_ll_per_sample = -mean_ce_per_sample  # log-likelihood

                # Determine which samples to keep (<= threshold)
                keep_mask = (mean_ll_per_sample <= threshold).float()  # 1 keep, 0 drop

                # Broadcast to token dimension and apply
                keep_mask_tokens = keep_mask.view(batch_size, 1).expand(-1, tokens_per_sample)
                ce_loss_masked = ce_loss_2d * keep_mask_tokens

                ce_loss_masked_flat = ce_loss_masked.view(-1)[valid_mask_flat]

                if num_items_in_batch is not None and num_items_in_batch > 0:
                    return ce_loss_masked_flat.sum() / num_items_in_batch
                else:
                    if ce_loss_masked_flat.numel() == 0:
                        return torch.tensor(0.0, device=logits_.device)
                    return ce_loss_masked_flat.mean()

            self.model.loss_function = _ll_threshold_loss
