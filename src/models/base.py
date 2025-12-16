import copy
import math
import os
import random
import time
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from lightning import LightningModule
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from transformers import PreTrainedTokenizerFast, StoppingCriteriaList
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

from src.constants import BASEDIR, aa_letters, aa_letters_lower
from src.data.objects import StringObject
from src.data.tokenizers import ProFamTokenizer
from src.models import metrics
from src.models.utils import InputAwareDynamicCache, log_likelihood_from_outputs
from src.utils import RankedLogger
from src.utils.sampling_utils import RepeatStoppingCriteria, has_too_many_repeats

log = RankedLogger(__name__, rank_zero_only=True)


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


def load_checkpoint(checkpoint_dir, **kwargs):
    config_dir = os.path.join(BASEDIR, checkpoint_dir, ".hydra")
    cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    log.info(OmegaConf.to_yaml(cfg.model))
    # TODO: check callback config
    checkpoint_path = os.path.join(BASEDIR, checkpoint_dir, "checkpoints/last.ckpt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )["state_dict"]
    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


class BaseFamilyLitModule(LightningModule):
    def __init__(
        self,
        model,
        tokenizer: ProFamTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        eps: float = 1e-5,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        num_decay_steps: Optional[int] = None,
        scoring_max_tokens: int = 32_000,
        use_kv_cache_for_scoring: bool = True,
        override_optimizer_on_load: bool = False,
        ignore_index: int = -100,
        pass_res_pos_in_doc_as_position_ids: bool = True,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(logger=False, ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_decay_steps = num_decay_steps
        self.scheduler_name = scheduler_name
        self.scoring_max_tokens = scoring_max_tokens
        self.override_optimizer_on_load = override_optimizer_on_load
        self.ignore_index = ignore_index
        self.pass_res_pos_in_doc_as_position_ids = pass_res_pos_in_doc_as_position_ids
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._train_dataset_sample_counts = defaultdict(int)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        # TODO: verify that different model implementations interpret
        # past key values in same way wrt e.g. position ids.
        if not (input_ids[:, 0] == self.tokenizer.bos_token_id).all():
            raise ValueError("Documents must start with a bos token")
            # note that when sampling we don't end up here, rather we call:
            # BaseLitModule.model.generate()
            # similarly, when using score_seqs (eg. protein_gym) we go via:
            # BaseLitModule.model.forward()
            # in general we assume that if you call BaseLitModule.forward()
            # you are not using KV cache.

        if labels is not None:
            labels[labels == self.tokenizer.bos_token_id] = self.ignore_index

        position_ids = self.get_position_ids_for_model_forward(
            input_ids, past_key_values
        )

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            **kwargs,
        )

    def compute_res_pos_in_doc(self, input_ids):
        """Needs to start at 0 for compatibility with sequence packing:
        https://github.com/huggingface/transformers/blob/70b07d97cf2c5f61fff55700b65528a1b6845cd2/src/transformers/modeling_flash_attention_utils.py#L133
        """
        assert (
            input_ids.shape[0] == 1
        ), "Since we are typically packing sequences, we assume batch size is 1"
        counter = torch.arange(input_ids.shape[1], device=input_ids.device)
        document_indices = (
            torch.cumsum(input_ids[0] == self.tokenizer.bos_token_id, 0) - 1
        )
        assert (
            document_indices >= 0
        ).all(), "Negative document indices encountered: check that bos token is first token in each document"
        doc_starts = (
            torch.argwhere(input_ids[0] == self.tokenizer.bos_token_id)
        ).flatten()
        offsets = counter[doc_starts][document_indices]
        position_ids = (counter - offsets).unsqueeze(0)
        return position_ids

    def get_position_ids_for_model_forward(self, input_ids, past_key_values):
        position_ids = None
        if past_key_values is not None:
            assert (
                input_ids == self.tokenizer.bos_token_id
            ).sum() <= 1, "Sequence packing not supported with past_key_values"
            position_ids = None
        elif self.pass_res_pos_in_doc_as_position_ids:
            position_ids = self.compute_res_pos_in_doc(input_ids)
        return position_ids

    def on_train_batch_start(self, batch, batch_idx: int):
        self._t0 = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        # TODO: handle ddp.
        self._t1 = time.time()
        self.log(
            "train/batch_time",
            self._t1 - self._t0,
            on_step=True,
            prog_bar=True,
        )

    def on_before_optimizer_step(self, optimizer):
        # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
        self.log(
            "train/grad_norm",
            calc_grad_norm(self.model.parameters()),
            on_step=True,
            prog_bar=True,
        )
        self.log("train/lr", optimizer.param_groups[0]["lr"])

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # uncomment for debugging ddp (train.py +experiment=ddp_test)
        # print(f"Rank: {self.trainer.global_rank}", batch["identifier"].text, flush=True)

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "train", log_global=True)
        self.log(
            "train/n_seqs",
            (batch["input_ids"] == self.tokenizer.sep_token_id)
            .float()
            .sum(axis=1)
            .mean()
            .item(),
            on_step=True,
            prog_bar=True,
            on_epoch=False,
        )
        self.log(
            "train/accumulate_grad_batches",
            self.trainer.accumulate_grad_batches,
            on_step=True,
            on_epoch=False,
        )
        self.log_train_dataset_sample_counts(batch)
        return loss

    def log_train_dataset_sample_counts(self, batch: Dict[str, Any]) -> None:
        """Keep and log a running count of *samples* seen per dataset name during training.

        Handles:
        - **Sequence packing**: `batch["ds_name"].text` is a length-1 list where the single string
          concatenates per-sample dataset names with "$" delimiters.
        - **No packing**: `batch["ds_name"].text` is a list of dataset-name strings, one per sample.

        Logs only in training (caller responsibility) and only logs dataset(s) updated this step.
        """
        if "ds_name" not in batch or batch["ds_name"] is None:
            return

        ds_name_obj = batch["ds_name"]
        # Prefer the project's StringObject convention, but be permissive.
        if hasattr(ds_name_obj, "text"):
            texts = ds_name_obj.text
        else:
            texts = ds_name_obj

        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        ds_names: List[str] = []
        for t in texts_list:
            if t is None:
                continue
            if "$" in t:
                ds_names.extend([x for x in t.split("$") if x])
            else:
                ds_names.append(t)

        if len(ds_names) == 0:
            return

        updated_totals: Dict[str, int] = {}
        for name in ds_names:
            self._train_dataset_sample_counts[name] += 1
            updated_totals[name] = self._train_dataset_sample_counts[name]

        # Log updated totals this step. Use tensors so Lightning can handle device placement.
        for name, total in updated_totals.items():
            self.log(
                f"train/dataset_samples_seen/{name}",
                torch.tensor(int(total), device=self.device),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                reduce_fx="sum",
            )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            print("validation step:", batch["DMS_id"].text[0])
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = outputs.loss
        self.log_metrics(
            batch,
            outputs,
            "val",
            log_global=dataloader_idx == 0,
        )
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "test", log_global=dataloader_idx == 0)
        return loss

    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading, optionally overriding optimizer and scheduler states.

        If override_optimizer_on_load is True, we'll remove the optimizer and
        lr_scheduler states from the checkpoint, forcing Lightning to create new ones
        based on the current config hyperparameters.
        """
        if self.override_optimizer_on_load:
            if "optimizer_states" in checkpoint:
                log.info(
                    "Overriding optimizer state from checkpoint with current config values"
                )
                del checkpoint["optimizer_states"]

            if "lr_schedulers" in checkpoint:
                log.info(
                    "Overriding lr scheduler state from checkpoint with current config values"
                )
                del checkpoint["lr_schedulers"]

            # Set a flag to tell Lightning not to expect optimizer states
            checkpoint["optimizer_states"] = []
            checkpoint["lr_schedulers"] = []

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer_name = self.hparams.get("optimizer", "adamw")
        log.info(f"Using optimizer {optimizer_name}")
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.95),
                eps=self.eps,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        optim_dict = {"optimizer": optimizer}
        if self.scheduler_name is not None:
            if self.scheduler_name == "cosine_with_min_lr":
                scheduler = get_scheduler(
                    self.scheduler_name,
                    optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                    scheduler_specific_kwargs={"min_lr_rate": 0.1},
                )
            elif self.scheduler_name == "warmup_stable_decay":
                if self.num_decay_steps is None:
                    raise ValueError(
                        "num_decay_steps is required for warmup_stable_decay scheduler"
                    )

                num_warmup_steps = self.num_warmup_steps
                num_decay_steps = self.num_decay_steps
                num_training_steps = self.num_training_steps
                num_decay_start_step = num_training_steps - num_decay_steps
                min_lr_ratio = 0.1

                def lr_lambda(current_step: int):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    elif current_step < num_decay_start_step:
                        return 1.0
                    else:
                        progress = min(
                            1.0,
                            float(current_step - num_decay_start_step)
                            / float(max(1, num_decay_steps)),
                        )
                        return (
                            max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                            * (1.0 - min_lr_ratio)
                            + min_lr_ratio
                        )

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = get_scheduler(
                    self.scheduler_name,
                    optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                )
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return optim_dict

    def trim_eval_batch(self, seqs_ids):
        """
        trim to first padding token in mini-batch
        (if batch-size is 1: avoid padding entirely)
        """
        pad_tok = self.tokenizer.vocab["[PAD]"]
        mask = seqs_ids != pad_tok
        indices = torch.arange(seqs_ids.shape[-1], device=seqs_ids.device).expand(
            seqs_ids.shape
        )
        # Set indices with padding to 0
        indices = torch.where(mask, indices, torch.tensor(0, device=seqs_ids.device))
        max_non_pad_index_per_seq = torch.max(indices, dim=-1).values
        return seqs_ids[..., : max_non_pad_index_per_seq.max() + 1]

    def _score_seqs_kv_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        assert (
            input_ids[0, 0] == self.tokenizer.vocab["[start-of-document]"]
            and input_ids[0, 1] > 19
        ), "First two tokens should be special start-of-doc and document type"
        if completion_ids[0, 0, 0] == self.tokenizer.sep_token_id:
            assert (
                input_ids[0, -1] != self.tokenizer.sep_token_id
            ), "Double sep token in input and completion"
        outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        L = completion_ids.shape[-1]

        for batch_start in tqdm.tqdm(
            range(0, completion_ids.shape[1], batch_size), disable=not verbose
        ):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            # fmt: off
            this_input_ids = completion_ids[
                :, batch_start: batch_start + batch_size
            ].reshape(-1, L)  # b_mut, L
            # fmt: on
            # remove unnecessary padding:
            this_input_ids = self.trim_eval_batch(this_input_ids)
            L_mini_batch = this_input_ids.shape[-1]

            actual_batch_size = this_input_ids.shape[0]
            cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
            cache.batch_repeat_interleave(actual_batch_size)  # careful: returns None!
            # fmt: off
            outputs = self.model(
                input_ids=this_input_ids,
                past_key_values=cache,
                use_cache=True,
            )
            # fmt: on
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            # start_ix is 0 as this is likelihood for first AA (pos 1)
            log_likelihood = log_likelihood_from_outputs(outputs, labels, start_ix=0)

            # mask padded positions in before computing the mean.
            shift_labels = labels[..., 1:].to(
                log_likelihood.device
            )  # aligns with start_ix=0
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean)  # b_mut

        lls = torch.cat(all_lls).cpu().float().numpy()
        return lls

    def _score_seqs_no_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        if batch_size > 1:
            raise NotImplementedError(
                "Mutant batch size > 1 not yet supported for mutant scoring"
            )
        all_lls = []
        likelihood_start_ix = input_ids.shape[1]
        for completion_ix in tqdm.tqdm(
            range(completion_ids.shape[1]), disable=not verbose
        ):
            this_input_ids = torch.cat(
                [input_ids, completion_ids[:, completion_ix]],
                dim=1,
            )
            # remove unnecessary padding:
            this_input_ids = self.trim_eval_batch(this_input_ids)
            L_mini_batch = this_input_ids.shape[-1]  # beware: includes prompt too
            # https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/data/data_collator.py#L823
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            assert (
                this_input_ids[..., likelihood_start_ix] not in self.tokenizer.aa_tokens
            ), "Likelihood start ix is an AA token - likelihood cannot be computed for this position"

            outputs = self.model(input_ids=this_input_ids, use_cache=False)
            # TODO: maybe relabel start_ix - a bit confusing
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=likelihood_start_ix
            )  # 1, L
            shift_labels = labels[..., likelihood_start_ix + 1 :].to(
                log_likelihood.device
            )
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean.item())
        lls = np.array(all_lls)
        return lls

    def _score_seqs_no_context(
        self,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
        start_tokens: list[int] = [47, 63],
    ):
        if len(completion_ids.shape) == 3:
            completion_ids = completion_ids.squeeze(0)
        if (completion_ids[:, 0] == self.tokenizer.sep_token_id).any():
            assert (
                completion_ids[:, 0] == self.tokenizer.sep_token_id
            ).all(), "Some sequences have sep token at start but not all"
            completion_ids = completion_ids[:, 1:]
        if (completion_ids[:, 0] != start_tokens[0]).any():
            start_tokens_tensor = (
                torch.tensor(start_tokens, device=completion_ids.device)
                .unsqueeze(0)
                .repeat(completion_ids.shape[0], 1)
            )
            completion_ids = torch.cat([start_tokens_tensor, completion_ids], dim=-1)
        all_lls = []
        for completion_ix in tqdm.tqdm(
            range(0, completion_ids.shape[0], batch_size), disable=not verbose
        ):
            this_input_ids = completion_ids[completion_ix : completion_ix + batch_size]
            outputs = self.model(input_ids=this_input_ids, use_cache=False)
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=1
            )  # 1, L
            shift_labels = labels[..., 2:].to(
                log_likelihood.device
            )  # aligns with start_ix=1
            mask = shift_labels != -100
            denom = mask.sum(dim=-1).clamp(min=1)
            ll_mean = (log_likelihood * mask).sum(dim=-1) / denom
            all_lls.append(ll_mean)

        lls = torch.cat(all_lls).cpu().float().numpy()
        return lls

    def score_seqs(
        self,
        input_ids,
        completion_ids,
        use_cache: bool = True,
        batch_size: int = 1,
    ):
        if input_ids is not None:
            assert (
                input_ids.shape[0] == 1
            ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
            assert (
                input_ids.ndim == 2 and completion_ids.ndim == 3
            ), f"input ids shape {input_ids.shape}, completion ids shape {completion_ids.shape}"  # b, L; b, n, L
            if use_cache:
                return self._score_seqs_kv_cache(
                    input_ids,
                    completion_ids,
                    batch_size=batch_size,
                )
            else:
                return self._score_seqs_no_cache(
                    input_ids,
                    completion_ids,
                    batch_size=batch_size,
                )
        else:
            return self._score_seqs_no_context(
                completion_ids,
                batch_size=batch_size,
            )

    def _sample_seqs(
        self,
        input_ids,
        num_samples,
        max_tokens: int,
        max_generated_length: Optional[int] = None,
        max_total_length: Optional[
            int
        ] = None,  # maximum length of inputs plus completions
        fixed_length: Optional[int] = None,
        greedy: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        sample_gaps: bool = False,
        structure_tokens: bool = False,
        continuous_sampling: bool = False,
        repeat_guard: bool = True,
        repeat_length: int = 9,  # if last repeat_length chars appear repeat_count times, seq is aborted
        repeat_count: int = 9,
        max_retries: int = 3,
    ):
        """
        Conditionally independent sequence generation: sequences are generated independently of each other
        given the prompt. Once sep token is generated, the sequence is considered complete.
        (i.e. we don't generate a sequence of sequences directly).
        """
        # TODO: pass attention mask, pad_token_id to avoid the following warning:
        # The attention mask and the pad token id were not set. As a consequence, you may
        # observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        # TODO: add min length kwarg
        if max_total_length is None:
            max_total_length = max_tokens
        if max_generated_length is not None:
            assert max_generated_length <= max_total_length
        generation_kwargs = {}
        sep_token_id = self.tokenizer.sep_token_id
        if fixed_length is not None:
            if max_total_length is not None:
                assert input_ids.shape[1] + fixed_length <= max_total_length
            generation_kwargs["min_new_tokens"] = fixed_length
            generation_kwargs["max_new_tokens"] = fixed_length
            generation_kwargs["eos_token_id"] = None
        elif max_generated_length is not None:
            generation_kwargs["min_new_tokens"] = 3
            generation_kwargs["max_new_tokens"] = max_generated_length
            generation_kwargs["eos_token_id"] = (
                None if continuous_sampling else sep_token_id
            )
        else:
            generation_kwargs["min_new_tokens"] = 3  # for esmfold
            generation_kwargs["eos_token_id"] = (
                None if continuous_sampling else sep_token_id
            )
            generation_kwargs["max_length"] = max_total_length
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if top_p is not None:
            # nucleus sampling; ensure valid range
            if not (0.0 < float(top_p) <= 1.0):
                raise ValueError("top_p must be in the interval (0, 1]")
            generation_kwargs["top_p"] = float(top_p)
        bad_aas = [
            "X",
            "x",
            "B",
            "J",
            "O",
            "U",
            "Z",
        ]
        if not sample_gaps:
            bad_aas.append("-")
        if structure_tokens:
            bad_aas = bad_aas + aa_letters
        else:
            bad_aas = bad_aas + aa_letters_lower

        # each 'word' is treated as a list of tokens
        # TODO: write test for this with random model.
        generation_kwargs["bad_words_ids"] = [
            [tok_id]
            for tok_id in self.tokenizer.all_special_ids
            if tok_id != self.tokenizer.eos_token_id
        ]
        generation_kwargs["bad_words_ids"] += [
            [self.tokenizer.convert_tokens_to_ids(bad_aa)] for bad_aa in bad_aas
        ]

        assert (
            input_ids.shape[0] == 1 and input_ids.ndim == 2
        ), "Only batch size 1 is supported for sampling; batch dim must be present"

        all_outputs: List[torch.Tensor] = []
        all_scores: List[float] = []
        # Always generate exactly one sequence at a time
        for batch_start in tqdm.tqdm(range(num_samples), "Generating sequences"):
            remaining = 1
            attempt = 0
            batch_collected: List[torch.Tensor] = []
            batch_scores: List[float] = []
            while remaining > 0:
                # Build stopping criteria that knows prompt length (non-continuous only)
                stopping = None
                if not continuous_sampling and repeat_guard:
                    prompt_len = input_ids.shape[1]
                    stopping = StoppingCriteriaList(
                        [
                            RepeatStoppingCriteria(
                                self.tokenizer,
                                repeat_length=repeat_length,
                                repeat_count=repeat_count,
                                prompt_length=prompt_len,
                            )
                        ]
                    )
                gen_out = self.model.generate(
                    input_ids=input_ids,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=not greedy,
                    temperature=temperature,
                    stopping_criteria=stopping,
                    **generation_kwargs,
                )
                seqs_full = gen_out.sequences  # (remaining, L_total)
                scores_list = gen_out.scores  # List[T] of (remaining, V)
                # Slice off prompt
                prompt_len = input_ids.shape[1]
                seqs = seqs_full[:, prompt_len:]

                # Evaluate which are acceptable vs need retry
                failed_indices: List[int] = []
                for i in range(seqs.shape[0]):
                    row = seqs[i]
                    # find last non-pad token index
                    pad_id = self.tokenizer.pad_token_id
                    valid_len = int((row != pad_id).sum().item())
                    last_tok = (
                        int(row[valid_len - 1].item()) if valid_len > 0 else pad_id
                    )
                    text = self.tokenizer.decode(
                        row[:valid_len].tolist(), skip_special_tokens=True
                    ).replace(" ", "")
                    ends_with_sep = last_tok == self.tokenizer.sep_token_id
                    is_repeaty = has_too_many_repeats(
                        text, repeat_length=repeat_length, repeat_count=repeat_count
                    )
                    if (not ends_with_sep) or (
                        is_repeaty and (not continuous_sampling)
                    ):
                        failed_indices.append(i)
                    else:
                        # accept and score
                        batch_collected.append(row.unsqueeze(0))
                        # compute mean logp up to SEP if present
                        total_logp = 0.0
                        count = 0
                        finished_non_cont = False
                        T = len(scores_list)
                        for t in range(T):
                            token_id = (
                                int(seqs[i, t].item()) if t < seqs.shape[1] else pad_id
                            )
                            lp = F.log_softmax(scores_list[t], dim=-1)[
                                i, token_id
                            ].item()
                            if not continuous_sampling:
                                if finished_non_cont:
                                    continue
                                total_logp += float(lp)
                                count += 1
                                if token_id == self.tokenizer.sep_token_id:
                                    finished_non_cont = True
                            else:
                                raise ValueError(
                                    "Continuous sampling is not supported for base model"
                                )
                        batch_scores.append(total_logp / max(count, 1))

                if len(failed_indices) == 0:
                    remaining = 0
                else:
                    attempt += 1
                    if attempt > max_retries:
                        # accept remaining failed ones as-is (score them) to avoid infinite loop
                        for i in failed_indices:
                            row = seqs[i]
                            batch_collected.append(row.unsqueeze(0))
                            total_logp = 0.0
                            count = 0
                            T = len(scores_list)
                            for t in range(T):
                                token_id = (
                                    int(seqs[i, t].item())
                                    if t < seqs.shape[1]
                                    else pad_id
                                )
                                lp = F.log_softmax(scores_list[t], dim=-1)[
                                    i, token_id
                                ].item()
                                total_logp += float(lp)
                                count += 1
                            batch_scores.append(total_logp / max(count, 1))
                        remaining = 0
                    else:
                        remaining = len(failed_indices)

            # Commit collected from this batch
            if len(batch_collected) > 0:
                all_outputs.append(torch.cat(batch_collected, dim=0))
                all_scores.extend(batch_scores)

        max_output_length = max([o.shape[1] for o in all_outputs])
        padded_outputs = torch.full(
            (num_samples, max_output_length), self.tokenizer.pad_token_id
        )
        start_ix = 0
        for o in all_outputs:
            padded_outputs[start_ix : start_ix + o.shape[0], : o.shape[1]] = o
            start_ix += o.shape[0]

        return padded_outputs, all_scores

    @torch.no_grad()
    def log_metrics(self, batch, outputs, step_name, log_global: bool = True):
        # N.B. actually val logging is a bit different because of this ds name thing
        loss = outputs.loss
        n_tokens = batch["input_ids"].shape[-1]
        if step_name == "train":
            ds_names = None
        else:
            ds_names = batch["ds_name"].text
        dataset_accuracies = metrics.accuracy_from_outputs(
            batch["input_ids"],
            outputs,
            batch["labels"],
            ignore_index=self.ignore_index,
            dataset_names=ds_names,  # a list of dataset names (StringObject.text)
            ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                ["-", "X", "x", "[start-of-document]"]
                + aa_letters_lower
                + self.tokenizer.all_special_tokens
            ),
            sep_token_id=self.tokenizer.sep_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            calc_full_no_context_accuracies=True,
        )

        global_metrics = {
            "loss": loss,
            "ppl": torch.exp(loss),
            "aa_accuracy": dataset_accuracies.pop("global"),
            "aa_accuracy_first_sequence": dataset_accuracies.pop("first_sequence"),
            "aa_accuracy_last_sequence": dataset_accuracies.pop("last_sequence"),
            "n_tokens_in_batch": n_tokens,
        }

        if log_global:
            self.log_dict(
                {f"{step_name}/{k}": v for k, v in global_metrics.items()},
                on_step=step_name == "train",
                on_epoch=step_name != "train",
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=step_name != "train",
            )

        # n.b. this assumes a batch only contains a single dataset - only true during val!
        # assert all([ds_name == batch["ds_name"][0] for ds_name in batch["ds_name"]])
        assert isinstance(batch["ds_name"], StringObject)

        is_single_dataset_batch = len(set(batch["ds_name"].text)) == 1
        for ds_name in set(batch["ds_name"].text):
            if ds_name not in dataset_accuracies:
                continue
            ds_metrics = {
                f"{step_name}/{ds_name}/aa_accuracy": dataset_accuracies[ds_name],
                f"{step_name}/{ds_name}/aa_accuracy_first_sequence": dataset_accuracies[
                    ds_name + "_first_sequence"
                ],
                f"{step_name}/{ds_name}/aa_accuracy_last_sequence": dataset_accuracies[
                    ds_name + "_last_sequence"
                ],
            }
            if is_single_dataset_batch:
                # global metrics are dataset specific
                ds_metrics[f"{step_name}/{ds_name}/loss"] = loss
            self.log_dict(
                ds_metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=False,
                sync_dist=step_name != "train",  # Q: what happens if sync_dist is False
            )
        add_dataloader_idx = step_name != "train"
        seq_len_stats = metrics.sequence_lengths(
            batch["labels"], self.tokenizer.sep_token_id
        )
        sep_tokens_in_batch = (
            (batch["labels"] == self.tokenizer.sep_token_id).sum().item()
        )
        start_of_doc_tokens_in_batch = (
            (batch["input_ids"] == self.tokenizer.bos_token_id).sum().item()
        )
        for reduce_fx in ["min", "max", "mean"]:
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_seq_len_in_batch",
                value=seq_len_stats[f"{reduce_fx}_seq_length"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_sep_tokens_in_batch",
                value=sep_tokens_in_batch,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )
            self.log(
                name=f"{step_name}/token_stats/{reduce_fx}_start_of_doc_tokens_in_batch",
                value=start_of_doc_tokens_in_batch,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                reduce_fx=reduce_fx,
                add_dataloader_idx=add_dataloader_idx,
            )

    def validation_step_proteingym(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Assumes that batch contains the following:

        input_ids: the prompt (i.e. MSA)
        completion_ids: the completions (i.e. mutated sequences)

        on caching: it seems like, if we modify what is passed to attention forward, existing cache
        might just work. currently model/sampling loop probably passes just the next token.
        """
        assert batch["DMS_scores"].ndim == 2  # b, n
        L = batch["completion_ids"].shape[-1]
        L_prompt = batch["input_ids"].shape[-1]
        lls = self.score_seqs(
            batch["input_ids"],
            batch["completion_ids"],
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=max(self.scoring_max_tokens // (L + L_prompt), 1)
            if self.use_kv_cache_for_scoring
            else 1,
        )
        if lls.min() == lls.max():
            spearman_corr = 0
        else:
            spearman_corr, _ = spearmanr(
                lls.astype(np.float32),
                batch["DMS_scores"][0].to(torch.float32).cpu().numpy(),
            )
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman",
            spearman_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "gym/log_likelihood",
            lls.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def on_train_epoch_end(self):
        # Commenting out as may cause deadlock in DDP
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19604
        log.info("Train epoch end %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
