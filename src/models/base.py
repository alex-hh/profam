import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, List

import hydra
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from lightning import LightningModule
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from transformers import PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler
import random
import warnings
import copy

from src.constants import BASEDIR, aa_letters, aa_letters_lower
from src.data.objects import StringObject
from src.data.tokenizers import ProFamTokenizer
from src.models import metrics
from src.models.utils import InputAwareDynamicCache, log_likelihood_from_outputs
from src.utils import RankedLogger

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


class BaseLitModule(LightningModule):
    """Assumes signature of CausalLM: e.g. labels is a kwarg"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        eps: float = 1e-5,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 32000,
        optimizer: str = "adamw",
        override_optimizer_on_load: bool = False,  # if True overwrite lr params from checkpoint w config params
        ignore_index: int = -100,
        gym_results_save_dir = "proteingym_variants"
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(logger=False, ignore=["model"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scheduler_name = scheduler_name
        self.scoring_max_tokens = scoring_max_tokens
        self.override_optimizer_on_load = override_optimizer_on_load
        self.ignore_index = ignore_index
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.gym_results_save_dir = gym_results_save_dir
        os.makedirs(gym_results_save_dir, exist_ok=True)
        print("proteinGym results saved in", self.gym_results_save_dir)


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
        elif optimizer_name == "lion":
            from bitsandbytes.optim import Lion

            optimizer = Lion(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99),
            )
        elif optimizer_name == "lion8bit":
            from bitsandbytes.optim import Lion8bit

            optimizer = Lion8bit(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.99),
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

    # def compute_consumed_samples(self, steps_since_resume=0):
    #     app_state = AppState()

    #     if self.cfg.get('rampup_batch_size', None):
    #         current_global_batch_size = get_current_global_batch_size() if get_current_global_batch_size() else 1
    #         consumed_samples = self.prev_consumed_samples + self.if_first_step * current_global_batch_size
    #     else:
    #         consumed_samples = (
    #             self.init_consumed_samples
    #             + steps_since_resume
    #             * app_state.data_parallel_size
    #             * self.cfg.micro_batch_size
    #             * get_num_microbatches()
    #         )
    #     return int(consumed_samples)

    # def _compute_consumed_samples_after_training_step(self):
    #     # Add +1 to account for the current batch, which is not counted yet in `trainer.global_step`.
    #     if not hasattr(self, 'init_global_step'):
    #         self.init_global_step = 0  # in case this method is called before training starts.
    #     return self.compute_consumed_samples(self.trainer.global_step + 1 - self.init_global_step)

    # def _extract_consumed_samples_from_ckpt(self, ckpt_path):
    #     try:
    #         init_consumed_samples = int(float(re.findall(r"consumed_samples\=([0-9]+.[0-9]+)", ckpt_path)[0]))
    #     except (ValueError, TypeError, IndexError):
    #         logging.warning("Cannot parse the checkpoint file to get the consumed samples. assume it is zero.")
    #         init_consumed_samples = 0

    #     return init_consumed_samples

    def get_forward_kwargs(self, batch):
        return {}

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
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

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
        has_3di = False

        global_metrics = {
            "loss": loss,
            "ppl": torch.exp(loss),
            "aa_accuracy": dataset_accuracies.pop("global"),
            "aa_accuracy_first_sequence": dataset_accuracies.pop("first_sequence"),
            "aa_accuracy_last_sequence": dataset_accuracies.pop("last_sequence"),
            "n_tokens_in_batch": n_tokens,
        }
        if "coords" in batch:
            global_metrics["has_coords_frac"] = metrics.has_coords_frac(**batch)
            if "plddts" in batch:
                global_metrics.update(metrics.plddt_metrics(**batch))
            is_interleaved = (
                batch["input_ids"] == self.tokenizer.seq_struct_sep_token_id
            ).any()
            if is_interleaved:
                # accuracy where coordinates are available
                aa_has_coords_mask = batch["interleaved_coords_mask"].any((-1, -2))
                has_coords_dataset_accuracies = metrics.accuracy_from_outputs(
                    batch["input_ids"],
                    outputs,
                    batch["labels"],
                    ignore_index=self.ignore_index,
                    dataset_names=batch[
                        "ds_name"
                    ].text,  # a list of dataset names (StringObject.text)
                    ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                        ["-", "X", "x"]
                        + aa_letters_lower
                        + self.tokenizer.all_special_tokens
                    ),
                    sep_token_id=self.tokenizer.sep_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    calc_full_no_context_accuracies=True,
                    mask=(aa_has_coords_mask & batch["aa_mask"]),
                )
                global_metrics[
                    "has_coords_aa_accuracy"
                ] = has_coords_dataset_accuracies.pop("global")
                global_metrics[
                    "has_coords_aa_accuracy_first_sequence"
                ] = has_coords_dataset_accuracies.pop("first_sequence")
                global_metrics[
                    "has_coords_aa_accuracy_last_sequence"
                ] = has_coords_dataset_accuracies.pop("last_sequence")
                global_metrics["aa_has_coords_frac"] = (
                    aa_has_coords_mask & batch["aa_mask"]
                ).float().sum() / batch["aa_mask"].float().sum()
            global_metrics["aa_count"] = batch["aa_mask"].float().sum()

        if has_3di:
            dataset_accuracies_3di = metrics.accuracy_from_outputs(
                batch["input_ids"],
                outputs,
                batch["labels"],
                ignore_index=self.ignore_index,
                dataset_names=batch["ds_name"].text,
                ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                    ["-", "X", "x"] + aa_letters + self.tokenizer.all_special_tokens
                ),
                sep_token_id=self.tokenizer.sep_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                calc_full_no_context_accuracies=True,
            )
            global_metrics["3di_accuracy"] = dataset_accuracies_3di.pop("global")
            global_metrics["3di_accuracy_first_sequence"] = dataset_accuracies_3di.pop(
                "first_sequence"
            )
            global_metrics["3di_accuracy_last_sequence"] = dataset_accuracies_3di.pop(
                "last_sequence"
            )

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
            # TODO: coords frac for each dataset
            if is_single_dataset_batch:
                # global metrics are dataset specific
                ds_metrics[f"{step_name}/{ds_name}/loss"] = loss
            if has_3di:
                ds_metrics[
                    f"{step_name}/{ds_name}/3di_accuracy"
                ] = dataset_accuracies_3di[ds_name]
                ds_metrics[
                    f"{step_name}/{ds_name}/3di_accuracy_first_sequence"
                ] = dataset_accuracies_3di[ds_name + "_first_sequence"]
                ds_metrics[
                    f"{step_name}/{ds_name}/3di_accuracy_last_sequence"
                ] = dataset_accuracies_3di[ds_name + "_last_sequence"]
            if "coords" in batch and is_interleaved:
                ds_metrics[
                    f"{step_name}/{ds_name}/has_coords_aa_accuracy"
                ] = has_coords_dataset_accuracies[ds_name]
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
            (batch["labels"] == self.tokenizer.bos_token_id).sum().item()
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

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        forward_kwargs = self.get_forward_kwargs(batch)
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            **forward_kwargs,
        )
        loss = outputs.loss
        self.log_metrics(batch, outputs, "train", log_global=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
        self.log(
            "train/grad_norm",
            calc_grad_norm(self.model.parameters()),
            on_step=True,

            prog_bar=True,
        )
        self.log("train/lr", optimizer.param_groups[0]["lr"])

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        print("Entering validation step")
        if "DMS_scores" in batch:
            print("validation step:", batch["DMS_id"].text[0])
            outputs = self.validation_step_proteingym(batch)
            return outputs
        elif "family_labels" in batch:
            outputs = self.validation_step_family_classification(batch)
            return outputs
        else:
            forward_kwargs = self.get_forward_kwargs(batch)
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                **forward_kwargs,
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
            forward_kwargs = self.get_forward_kwargs(batch)
            outputs = self(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                **forward_kwargs,
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


class BaseFamilyLitModule(BaseLitModule):
    def __init__(
        self,
        model,
        tokenizer: ProFamTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 32_000,
        use_kv_cache_for_scoring: bool = True,
        embed_coords: bool = False,
        override_optimizer_on_load: bool = False,
        max_tokens: int = 8192,
        gym_subsamples_per_n: int = 100,
        gym_results_save_dir = "proteingym_variants"
    ):
        super().__init__(
            model,
            tokenizer,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_name=scheduler_name,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scoring_max_tokens=scoring_max_tokens,
            optimizer="adamw",
            override_optimizer_on_load=override_optimizer_on_load,
            gym_results_save_dir=gym_results_save_dir
        )
        self.scoring_max_tokens = scoring_max_tokens
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.embed_residue_index = self.tokenizer.embed_residue_index
        self.max_res_pos_in_seq = self.tokenizer.max_res_pos_in_seq
        self.embed_coords = embed_coords
        self.embed_sequence_index = self.model.embed_sequence_index
        # NEW FOR EVALUATING PROTEIN GYM OFFLINE ONLY-------------------------
        self.max_tokens = max_tokens
        self.gym_subsamples_per_n = gym_subsamples_per_n
        # ---------------------------------------------------------------------

    def get_forward_kwargs(self, batch):
        forward_kwargs = {}
        if self.embed_coords:
            assert batch["coords"] is not None
            forward_kwargs["coords"] = batch["coords"]
        if self.embed_residue_index:
            assert batch["residue_index"] is not None
            forward_kwargs["residue_index"] = batch["residue_index"]
        return forward_kwargs

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
        input_residue_index: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        completion_residue_index: Optional[torch.LongTensor] = None,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        assert input_ids[0,0] == self.tokenizer.vocab['[start-of-document]'] and input_ids[0,1] > 19, "First two tokens should be special start-of-doc and document type"
        if completion_ids[0,0,0] == self.tokenizer.sep_token_id:
            assert input_ids[0, -1] != self.tokenizer.sep_token_id, "Double sep token in input and completion"
        forward_kwargs = self.get_forward_kwargs(
            {"residue_index": input_residue_index, "coords": coords}
        )
        outputs = self.model(input_ids=input_ids, use_cache=True, **forward_kwargs)
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
            this_input_ids = self.trim_eval_batch(this_input_ids)  # todo trim strct etc
            L_mini_batch = this_input_ids.shape[-1]
            forward_kwargs = {}
            if self.embed_residue_index:
                # fmt: off
                this_res_ix = completion_residue_index[
                    :, batch_start: batch_start + batch_size, :L_mini_batch
                               ].reshape(-1, L_mini_batch)
                # fmt: on
                forward_kwargs["residue_index"] = this_res_ix
            if self.embed_coords:
                assert coords is not None
                raise NotImplementedError("Coords not yet supported for mutant scoring")

            actual_batch_size = this_input_ids.shape[0]
            cache = InputAwareDynamicCache.from_legacy_cache(past_key_values)
            cache.batch_repeat_interleave(actual_batch_size)  # careful: returns None!
            # fmt: off
            outputs = self.model(
                input_ids=this_input_ids,
                past_key_values=cache,
                use_cache=True,
                **forward_kwargs,
            )
            # fmt: on
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            # start_ix is 0 as this is likelihood for first AA (pos 1)
            log_likelihood = log_likelihood_from_outputs(outputs, labels, start_ix=0)

            all_lls.append(log_likelihood.mean(-1))  # b_mut

        lls = torch.cat(all_lls).cpu().float().numpy()
        return lls

    def _score_seqs_no_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        input_residue_index: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        completion_residue_index: Optional[torch.LongTensor] = None,
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
            forward_kwargs = {}
            # https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/data/data_collator.py#L823
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            assert (
                this_input_ids[..., likelihood_start_ix] not in self.tokenizer.aa_tokens
            ), "Likelihood start ix is an AA token - likelihood cannot be computed for this position"
            if self.embed_residue_index:
                this_res_ix = torch.cat(
                    [input_residue_index, completion_residue_index[:, completion_ix]],
                    dim=1,
                )[..., :L_mini_batch]
                forward_kwargs["residue_index"] = this_res_ix
            if self.embed_coords:
                assert coords is not None
                raise NotImplementedError("Coords not yet supported for mutant scoring")
            outputs = self.model(
                input_ids=this_input_ids, use_cache=False, **forward_kwargs
            )
            # TODO: maybe relabel start_ix - a bit confusing
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=likelihood_start_ix
            )  # 1, L

            all_lls.append(log_likelihood.mean(-1).item())
        lls = np.array(all_lls)
        return lls

    def _score_seqs_no_context(
        self,
        completion_ids,
        batch_size: int = 1,
        verbose: bool = False,
        start_tokens: list[int] = [47, 63]
        
    ):
        if len(completion_ids.shape) == 3:
            completion_ids = completion_ids.squeeze(0)
        if (completion_ids[:,0] == self.tokenizer.sep_token_id).any():
            assert (completion_ids[:,0] == self.tokenizer.sep_token_id).all(), "Some sequences have sep token at start but not all"
            completion_ids = completion_ids[:, 1:]
        if (completion_ids[:,0] != start_tokens[0]).any():
            start_tokens_tensor = torch.tensor(start_tokens, device=completion_ids.device).unsqueeze(0).repeat(completion_ids.shape[0], 1)
            completion_ids = torch.cat([start_tokens_tensor, completion_ids], dim=-1)
        all_lls = []
        for completion_ix in tqdm.tqdm(
            range(0, completion_ids.shape[0], batch_size), disable=not verbose
        ):
            this_input_ids = completion_ids[completion_ix:completion_ix+batch_size]
            forward_kwargs = {}
            outputs = self.model(
                input_ids=this_input_ids, use_cache=False, **forward_kwargs
            )
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=1
            )  # 1, L
            all_lls.append(log_likelihood.mean(-1))

        lls = torch.cat(all_lls).cpu().float().numpy()
        return lls

    # TODO: make this part of a mixin so that it can be reused across models
    # c.f. GenerationsMixin
    def score_seqs(
        self,
        input_ids,
        completion_ids,
        use_cache: bool = True,
        batch_size: int = 1,
        coords: Optional[torch.FloatTensor] = None,
        input_residue_index: Optional[torch.LongTensor] = None,
        completion_residue_index: Optional[torch.LongTensor] = None,
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
                    coords=coords,
                    input_residue_index=input_residue_index,
                    completion_residue_index=completion_residue_index,
                )
            else:
                return self._score_seqs_no_cache(
                    input_ids,
                    completion_ids,
                    batch_size=batch_size,
                    coords=coords,
                    input_residue_index=input_residue_index,
                    completion_residue_index=completion_residue_index,
                )
        else:
            assert input_residue_index is None and completion_residue_index is None
            if coords is not None:
                raise NotImplementedError("Coords not yet supported for mutant scoring")
            return self._score_seqs_no_context(
                completion_ids,
                batch_size=batch_size,
            )

    def _sample_seqs(
        self,
        input_ids,
        num_samples,
        max_tokens: int,
        batch_size: int = 1,
        max_generated_length: Optional[int] = None,
        max_total_length: Optional[
            int
        ] = None,  # maximum length of inputs plus completions
        input_residue_index: Optional[torch.LongTensor] = None,
        input_coords: Optional[torch.FloatTensor] = None,
        include_prompt_in_output: bool = False,
        fixed_length: Optional[int] = None,
        greedy: bool = False,
        temperature: Optional[float] = None,
        sample_gaps: bool = False,
        structure_tokens: bool = False,
    ):
        """
        Conditionally independent sequence generation: sequences are generated independently of each other
        given the prompt. Once sep token is generated, the sequence is considered complete.
        (i.e. we don't generate a sequence of sequences directly).
        """
        # TODO: pass attention mask, pad_token_id to avoid the following warning:
        # The attention mask and the pad token id were not set. As a consequence, you may
        # observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
        # TODO: add temperature kwarg
        # TODO: add min length kwarg
        # TODO: check whether model spontaneously adds the SEP token
        if max_total_length is None:
            if self.embed_residue_index:
                max_total_length = min(
                    max_tokens,
                    input_ids.shape[1] + self.tokenizer.max_res_pos_in_seq,
                )
            else:
                max_total_length = max_tokens
        if max_generated_length is not None:
            assert max_generated_length <= max_total_length
        generation_kwargs = {}
        if fixed_length is not None:
            if max_total_length is not None:
                assert input_ids.shape[1] + fixed_length <= max_total_length
            generation_kwargs["min_new_tokens"] = fixed_length
            generation_kwargs["max_new_tokens"] = fixed_length
            generation_kwargs["eos_token_id"] = None
        elif max_generated_length is not None:
            generation_kwargs["min_new_tokens"] = 3
            generation_kwargs["max_new_tokens"] = max_generated_length
            generation_kwargs["eos_token_id"] = self.tokenizer.sep_token_id
        else:
            generation_kwargs["min_new_tokens"] = 3  # for esmfold
            generation_kwargs["eos_token_id"] = self.tokenizer.sep_token_id
            generation_kwargs["max_length"] = max_total_length
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
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

        assert input_residue_index.shape == input_ids.shape
        all_outputs = []
        for batch_start in range(0, num_samples, batch_size):
            num_return_sequences = min(batch_size, num_samples - batch_start)
            # TODO: understand how this gets reshaped...within prepare inputs for generation it already is expanded
            forward_kwargs = self.get_forward_kwargs(
                {
                    "residue_index": input_residue_index,
                    "coords": input_coords,
                }
            )
            # TemperatureLogitsWarper
            # TODO: migrate to model.sample
            # N.B. we need to be careful about generationconfig -- in particular eos token id
            # if we want to generate multiple sequences in a single family: we either need to restore eos token id
            # or we just do a batched generation like we do here. latter is more explicit.
            # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1908
            outputs = self.model.generate(
                input_ids=input_ids,
                num_return_sequences=num_return_sequences,
                return_dict_in_generate=False,
                do_sample=not greedy,
                temperature=temperature,
                # https://huggingface.co/docs/transformers/en/generation_strategies
                **generation_kwargs,
                **forward_kwargs,
            )
            if not include_prompt_in_output:
                outputs = outputs[:, input_ids.shape[1] :]
            all_outputs.append(outputs)

        max_output_length = max([o.shape[1] for o in all_outputs])
        # TODO: poss just return a list instead of the padded tensor
        # TODO: does padding include eos (sep)? seems no?
        padded_outputs = torch.full(
            (num_samples, max_output_length), self.tokenizer.pad_token_id
        )
        start_ix = 0
        for o in all_outputs:
            padded_outputs[start_ix : start_ix + o.shape[0], : o.shape[1]] = o
            start_ix += o.shape[0]

        return padded_outputs



    # ---------------------------------------------------------------------
    # Context subsampling helpers (adapted from eval_ckpt_model_on_gym_multi_prompt)
    # ---------------------------------------------------------------------
    @staticmethod
    def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Deep-clone a ProteinGym batch so that in-place slicing does not affect the original."""
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.clone()
            elif isinstance(v, list):
                out[k] = v.copy()
            else:
                out[k] = copy.deepcopy(v)
        return out

    def _generate_context_variants(
        self,
        batch: Dict[str, torch.Tensor],
        sep_tok_id: int,
        start_tokens: list[int] = [47, 63]
    ):
        """Create context-truncated variants (random non-contiguous sampling).

        For each *n* up to the maximum number of sequences that can fit under the
        `max_tokens`, generate `gym_subsamples_per_n` variants by *randomly*
        selecting *n* unique sequences **without replacement**.  Sequence order
        within the prompt is shuffled (randomised).  We attempt several random
        draws until a set whose combined token count fits under the limit is
        found; if none is found we fall back to the *contiguous* subset with the
        smallest token count (still ≤ limit).  This ensures diversity while
        guaranteeing progress.
        """

        if batch["input_ids"] is None:
            raise ValueError("input_ids must be present for context subsampling")

        input_ids = batch["input_ids"]
        device = input_ids.device

        # Compute start & end token indices for every context sequence
        seq_ends = (input_ids[0] == sep_tok_id).nonzero(as_tuple=True)[0].cpu()
        total_seqs = len(seq_ends)
        seq_starts = torch.zeros_like(seq_ends)
        seq_starts[1:] = seq_ends[:-1] + 1

        # Pre-compute token lengths for each sequence so we can sum quickly
        seq_lengths = (seq_ends - seq_starts + 1).tolist()  # python ints

        prefix_token_counts = seq_ends + 1  # inclusive counts (0-based index + 1)
        max_n_under_limit = int((prefix_token_counts <= self.max_tokens).sum())

        variants = []
        rng = random.Random()
        n_forward_search = min(40, max_n_under_limit)
        n_log_samples = n_forward_search
        while True:
            n_vals = [int(s) for s in np.logspace(0, np.log10(max_n_under_limit), n_log_samples)] + [max_n_under_limit + 1]
            n_vals = list(set(n_vals))
            if len(n_vals) >= n_forward_search:
                break
            n_log_samples += 1
        n_vals.sort()
        for n in n_vals:
            for rep in range(self.gym_subsamples_per_n):
                # Random permutation of all sequence indices
                perm = list(range(total_seqs))
                rng.shuffle(perm)

                chosen_seq_idxs = []
                slices = []
                token_count = 0

                # Greedily add sequences in shuffled order until we hit n or limit
                for idx in perm:
                    length = seq_lengths[idx]
                    if len(chosen_seq_idxs) == n:
                        break
                    if token_count + length > self.max_tokens:
                        break
                    chosen_seq_idxs.append(idx)
                    start_tok = seq_starts[idx].item()
                    end_tok = seq_ends[idx].item()
                    slices.append((start_tok, end_tok))
                    token_count += length

                if len(chosen_seq_idxs) == 0:
                    # If even the shortest sequence exceeds the limit (extremely unlikely), skip variant
                    continue

                # Construct variant prompt
                var_batch = self._clone_batch(batch)

                def _concat_slices(tensor):
                    parts = [tensor[..., s : e + 1] for s, e in slices]
                    return torch.cat(parts, dim=-1)

                var_batch["input_ids"] = _concat_slices(var_batch["input_ids"]).clone()
                # remove final sep and add start tokens
                var_batch["input_ids"] = torch.cat([torch.tensor(start_tokens, device=var_batch["input_ids"].device).unsqueeze(0), var_batch["input_ids"]], dim=-1)
                if var_batch["input_ids"][0, -1] == self.tokenizer.sep_token_id:
                    var_batch["input_ids"] = var_batch["input_ids"][:, :-1]
                if "residue_index" in var_batch and var_batch["residue_index"] is not None:
                    var_batch["residue_index"] = _concat_slices(var_batch["residue_index"]).clone()

                variants.append(
                    (
                        var_batch,
                        {
                            "variant_type": "random_greedy",
                            "target_n": n,
                            "n_seqs": len(chosen_seq_idxs),
                            "n_tokens": token_count,
                            "replicate": rep,
                            "seq_indices": chosen_seq_idxs,
                        },
                    )
                )
        return variants

    @staticmethod
    def _compute_spearman(lls: np.ndarray, dms_scores: np.ndarray):
        if lls.min() == lls.max():
            return 0.0
        return float(spearmanr(lls.astype(np.float32), dms_scores.astype(np.float32))[0])

    def _save_variant_scatter_plot(
        self,
        n_seqs_list: list[int],
        variant_lls: list[np.ndarray],
        dms_id: str,
    ):
        """Create and save scatter plot and histogram of variant evaluation results.

        The scatter and histogram are drawn on separate subplots in the same
        figure.  The number of histogram bins is determined as
        ``max(1, int(0.2 * n_observations))`` where *n_observations* is the
        number of data points.
        """
        import warnings
        try:
            # Calculate per-variant mean log-likelihoods
            mean_lls_per_variant = [float(ll.mean()) for ll in variant_lls]

            n_observations = len(n_seqs_list)
            n_bins = max(1, int(0.2 * n_observations))

            # Create figure with two distinct subplots (side-by-side)
            fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

            # Scatter plot: mean log-likelihood vs number of sequences
            scatter = ax_scatter.scatter(
                n_seqs_list,
                mean_lls_per_variant,
                c=list(range(n_observations)),
                cmap="viridis",
                alpha=0.3,
            )
            ax_scatter.set_xlabel("Number of sequences sampled")
            ax_scatter.set_ylabel("Mean log likelihood")
            ax_scatter.set_title(f"Context variants for {dms_id}")


            ax_hist.hist(
                n_seqs_list,
                bins=n_bins,
                color="grey",
                alpha=0.7,
            )
            ax_hist.set_xlabel("Number of sequences sampled")
            ax_hist.set_ylabel("Count")
            ax_hist.set_title("Histogram of sampled sequences")

            # Colour bar associated with the scatter
            cbar = fig.colorbar(scatter, ax=ax_scatter)
            cbar.set_label("Variant index (earlier → later)")

            scatter_path = os.path.join(
                self.variant_csv_dir,
                f"batch_{dms_id}_scatter.png",
            )
            fig.tight_layout()
            fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to create scatter plot: {e}")



    def _evaluate_and_save_variants_v3(
        self,
        batch: Dict[str, torch.Tensor],
        start_tokens: list[int] = [47, 63],
        min_target_likelihood: float = -2.5,
        max_target_likelihood: float = -0.9,
        starting_target_n: int = 25,
    ):
        """
        Simplified adaptive variant evaluation.

        Generates diverse random subsamples of the context prompt whose
        mean log-likelihood lies roughly inside the user-specified band
        `[min_target_likelihood, max_target_likelihood]`.

        The procedure is:
        1.  Binary-search over *contiguous* prefixes of the prompt to find an
            `n_opt` (number of context sequences) that lands inside the band.
            If the monotonicity assumption breaks we fall back to a random `n`.
        2.  Draw `self.gym_subsamples_per_n` random variants that each contain
            exactly `n_opt` sequences (shuffled order, no replacement) without
            exceeding `self.max_tokens`.
        3.  Evaluate *all* completion sequences for every variant, saving both
            summary metrics (CSV) and the full likelihood matrix (`.npz`).
        4.  Ensemble the per-sequence likelihoods across variants and return
            the aggregate metrics.
        """
        random.seed(42)
        rng = random.Random()
        optimal_likelihood = min_target_likelihood + (max_target_likelihood - min_target_likelihood) / 2
        # ------------------------------------------------------------------ #
        # Canonicalise the prompt                                            #
        # ------------------------------------------------------------------ #
        sep_tok_id = self.tokenizer.sep_token_id
        start_tokens_tensor = torch.tensor(start_tokens, device=batch["input_ids"].device).unsqueeze(0)

        # Strip leading start tokens if present
        if (batch["input_ids"][0, : start_tokens_tensor.shape[1]] == start_tokens_tensor).all():
            batch["input_ids"] = batch["input_ids"][:, start_tokens_tensor.shape[1] :]

        # Ensure trailing SEP token
        if batch["input_ids"][0, -1] != sep_tok_id:
            batch["input_ids"] = torch.cat(
                [
                    batch["input_ids"],
                    torch.tensor([sep_tok_id], device=batch["input_ids"].device).unsqueeze(0),
                ],
                dim=-1,
            )

        # ------------------------------------------------------------------ #
        # Prompt statistics                                                  #
        # ------------------------------------------------------------------ #
        seq_ends = (batch["input_ids"][0] == sep_tok_id).nonzero(as_tuple=True)[0].cpu()
        total_seqs = len(seq_ends)
        seq_starts = torch.zeros_like(seq_ends)
        seq_starts[1:] = seq_ends[:-1] + 1
        seq_lengths = (seq_ends - seq_starts + 1).tolist()  # python ints

        def calculate_entropy_per_prompt(lls_array):
            exp_log = np.exp(lls_array)
            prob_denominator = np.sum(exp_log, axis=1)
            seq_probs = exp_log / prob_denominator.reshape(lls_array.shape[0], 1)
            per_prompt_entropies = -np.sum(seq_probs * np.log(seq_probs), axis=1)
            return per_prompt_entropies

        def _make_truncated_batch(idxs):
            """Deep-clone *batch* keeping only the sequences at *idxs*.

            If *idxs* is empty (i.e. no context sequences are selected) we return a
            batch with *input_ids* and *residue_index* set to **None** so that
            downstream code takes the no-context scoring path.
            """
            new_batch = self._clone_batch(batch)
            # Handle the no-context case early ---------------------------------
            if len(idxs) == 0:
                new_batch["input_ids"] = None
                if "residue_index" in new_batch:
                    new_batch["residue_index"] = None
                return new_batch
            # ------------------------------------------------------------------
            def _concat_slices(tensor):
                parts = [tensor[..., seq_starts[i] : seq_ends[i] + 1] for i in idxs]
                concat = torch.cat(parts, dim=-1)
                concat = torch.cat([start_tokens_tensor, concat], dim=-1)
                if concat[0, -1] == sep_tok_id:
                    concat = concat[:, :-1] # remove the final sep token as this is in the completions
                return concat

            new_batch["input_ids"] = _concat_slices(new_batch["input_ids"]).clone()
            if "residue_index" in new_batch and new_batch["residue_index"] is not None:
                new_batch["residue_index"] = _concat_slices(new_batch["residue_index"]).clone()
            return new_batch
        completion_length = batch["completion_ids"].shape[-1]
        max_context_tokens = (self.max_tokens - completion_length) - 5 # 5 is a buffer.
        avg_seq_len = sum(seq_lengths) / len(seq_lengths)
        max_n_by_tokens = max(0, min(int(max_context_tokens // avg_seq_len) + 2, total_seqs))

        # ------------------------------------------------------------------ #
        # Quick evaluator for contiguous prefixes                            #
        # ------------------------------------------------------------------ #
        @torch.no_grad()
        def _eval_prefix(n):
            n = max(0, min(n, total_seqs))
            if n == 0:
                vb = self._clone_batch(batch)
                vb["input_ids"] = None
                vb["residue_index"] = None
                L_prompt = 0
            else:
                selected_idxs = rng.sample(range(total_seqs), n)
                vb = _make_truncated_batch(selected_idxs)
                L_prompt = vb["input_ids"].shape[-1]
            vb_device = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in vb.items()}
            # Use at most 100 completions for speed
            comp_ids = vb_device["completion_ids"][:, : min(100, vb_device["completion_ids"].shape[1]), :]
            L = comp_ids.shape[-1]
            
            lls = self.score_seqs(
                vb_device["input_ids"],
                comp_ids,
                input_residue_index=vb_device.get("residue_index", None),
                completion_residue_index=vb_device.get("completion_residue_index", None),
                use_cache=self.use_kv_cache_for_scoring,
                batch_size=max((self.scoring_max_tokens) // (L + L_prompt), 1)
                if self.use_kv_cache_for_scoring
                else 1,
            )
            return float(lls.mean())

        # ------------------------------------------------------------------ #
        # Forward logspace search                                            #
        # ------------------------------------------------------------------ #
        n_forward_search = min(30, max_n_by_tokens)
        n_log_samples = n_forward_search
        while True:
            n_vals = [0] + [int(s) for s in np.logspace(0, np.log10(max_n_by_tokens), n_log_samples)]
            n_vals = list(set(n_vals))
            if len(n_vals) >= n_forward_search:
                break
            n_log_samples += 1
        n_vals.sort()
        n_seqs_list = []
        ll_list = []
        found = False
        vals_in_range = []
        for n_curr in n_vals:
            ll_curr = _eval_prefix(n_curr)
            n_seqs_list.append(n_curr)
            ll_list.append(ll_curr)
            if min_target_likelihood <= ll_curr <= max_target_likelihood:
                n_opt = n_curr
                found = True
                vals_in_range.append((n_curr, abs(n_curr - starting_target_n)))
        if found:
            vals_in_range.sort(key=lambda x: x[1])
            n_opt = vals_in_range[0][0]
        else:
            # ------------------------------------------------------------------ #
            # Binary search for n_opt                                            #
            # ------------------------------------------------------------------ #
            low_n, high_n = 0, max_n_by_tokens
            n_opt = random.randint(0, max_n_by_tokens)  # fallback
            found = False

            for _ in range(12):  # enough for the search space
                n_curr = (low_n + high_n) // 2
                ll_curr = _eval_prefix(n_curr)
                n_seqs_list.append(n_curr)
                ll_list.append(ll_curr)
                if min_target_likelihood <= ll_curr <= max_target_likelihood:
                    n_opt = n_curr
                    found = True
                    break
                if ll_curr < min_target_likelihood:
                    low_n = n_curr + 1
                else:
                    high_n = n_curr - 1
                if low_n > high_n:
                    break
            if not found:
                n_opt = random.randint(0, max_n_by_tokens)

        corr_coeff = np.corrcoef(n_seqs_list, ll_list)[0, 1]
        if corr_coeff < 0:
            random_strategy = True
        else:
            random_strategy = False
        if not found and random_strategy:
            n_opt = random.randint(1, max_n_by_tokens)
        elif not found and not random_strategy:
            if min(ll_list) >= max_target_likelihood:
                n_opt = 0
            elif max(ll_list) <= min_target_likelihood:
                n_opt = max_n_by_tokens
            else:
                raise ValueError("Should not happen")
        # ------------------------------------------------------------------ #
        # Generate random variants                                           #
        spearman_list = []
        variants = []
        dms_scores_np = batch["DMS_scores"][0].float().cpu().numpy()
        rows, variant_lls = [], []
        n_seqs_list = []
        tok_cnt_list = []
        min_cov_list = []
        self.variant_csv_dir = os.path.join(self.gym_results_save_dir, self.timestamp)
        os.makedirs(self.variant_csv_dir, exist_ok=True)
        csv_path = os.path.join(self.variant_csv_dir, f"batch_{batch['DMS_id'].text[0]}_v3.csv")
        token_count_attempts = 100
        fail_count = 0
        for rep in range(self.gym_subsamples_per_n):
            while n_opt >= 0:
                idxs = rng.sample(range(total_seqs), n_opt)
                rng.shuffle(idxs)
                tok_cnt = sum(seq_lengths[i] for i in idxs)
                # Gracefully handle the empty *idxs* case
                shortest_seq_len = min(seq_lengths[i] for i in idxs) if idxs else 0
                if tok_cnt + completion_length <= self.max_tokens:
                    var_batch = _make_truncated_batch(idxs)
                    min_cov = (shortest_seq_len / batch['completion_ids'].shape[-1]) if shortest_seq_len > 0 else 0.0
                    meta = {
                        "variant_idx": rep,
                        "replicate": rep,
                        "n_seqs": n_opt,
                        "n_tokens": tok_cnt,
                        "seq_indices": idxs,
                        "min_completion_coverage": min_cov,
                    }
                    n_seqs_list.append(n_opt)
                    tok_cnt_list.append(tok_cnt)
                    min_cov_list.append(min_cov)
                    variants.append((var_batch, meta))
                    var_batch_device = {
                        k: v.to(self.device) if torch.is_tensor(v) else v for k, v in var_batch.items()
                    }
                    L = var_batch_device["completion_ids"].shape[-1]
                    L_prompt = (
                        var_batch_device["input_ids"].shape[-1]
                        if var_batch_device["input_ids"] is not None
                        else 0
                    )
                    lls = self.score_seqs(
                        var_batch_device["input_ids"],
                        var_batch_device["completion_ids"],
                        input_residue_index=var_batch_device.get("residue_index", None),
                        completion_residue_index=var_batch_device.get("completion_residue_index", None),
                        use_cache=self.use_kv_cache_for_scoring,
                        batch_size=max((self.scoring_max_tokens) // (L + L_prompt), 1)
                        if self.use_kv_cache_for_scoring
                        else 1,
                    )
                    mean_ll = float(lls.mean())
                    variant_lls.append(lls)
                    spearman_list.append(float(self._compute_spearman(lls, dms_scores_np)))
                    rows.append({**meta, "mean_log_likelihood": mean_ll, "spearman": float(self._compute_spearman(lls, dms_scores_np)), "DMS_id": batch["DMS_id"].text[0]})
                    # update n_opt
                    if random_strategy:
                        if len(vals_in_range) > 1:
                            min_n_in_range = min(vals_in_range, key=lambda x: x[0])[0]
                            max_n_in_range = max(vals_in_range, key=lambda x: x[0])[0]
                            n_opt = random.choice(range(min_n_in_range, max_n_in_range + 1))
                        else:
                            n_opt = random.randint(1, max_n_by_tokens + 1)
                    else:
                        p_random_in_range = 0.1
                        if len(vals_in_range) > 1 and random.random() < p_random_in_range:
                            min_n_in_range = min(vals_in_range, key=lambda x: x[0])[0]
                            max_n_in_range = max(vals_in_range, key=lambda x: x[0])[0]
                            n_opt = random.choice(range(min_n_in_range, max_n_in_range + 1))
                        else:
                            if mean_ll > optimal_likelihood:
                                n_opt = max(1, n_opt -1)
                            else:
                                n_opt = min(n_opt + 1, total_seqs)
                    fail_count = 0
                    break
                else:
                    fail_count += 1
                    if fail_count > token_count_attempts:
                        n_opt -= 1
                        fail_count = 0
            if n_opt < 0:
                fail_count = 0
                if len(vals_in_range) > 1:
                    min_n_in_range = min(vals_in_range, key=lambda x: x[0])[0]
                    max_n_in_range = max(vals_in_range, key=lambda x: x[0])[0]
                    n_opt = random.choice(range(min_n_in_range, max_n_in_range + 1))
                else:
                    n_opt = random.randint(0, max_n_by_tokens + 1)
        lls_array = np.stack(variant_lls, axis=0)
        entropy_per_prompt = calculate_entropy_per_prompt(lls_array)
        if getattr(self, "global_rank", 0) == 0:
            mean_per_forward_pass = lls_array.mean(axis=1)
            sorted_indices_ll = np.argsort(-mean_per_forward_pass)
            sorted_indices_entropy = np.argsort(entropy_per_prompt)
            for top_pct in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                top_k = max(1, int(top_pct * len(sorted_indices_ll)))
                top_k_ll_mean_ll = lls_array[sorted_indices_ll[:top_k]].mean(axis=0)
                top_k_entropy_mean_ll = lls_array[sorted_indices_entropy[:top_k]].mean(axis=0)
                top_k_ll_spearman = self._compute_spearman(top_k_ll_mean_ll, dms_scores_np)
                top_k_entropy_spearman = self._compute_spearman(top_k_entropy_mean_ll, dms_scores_np)
                self.log(
                    f"gym/top_{top_pct}_ll_spearman",
                    top_k_ll_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    f"gym/bottom_{top_pct}_entropy_spearman",
                    top_k_entropy_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
        mean_lls = lls_array.mean(axis=0)
        ensemble_spearman = self._compute_spearman(mean_lls, dms_scores_np)
        ensemble_log_ll = float(mean_lls.mean())
        if getattr(self, "global_rank", 0) == 0:
            lls_save_path = os.path.join(self.variant_csv_dir, f"batch_{batch['DMS_id'].text[0]}_v3_lls.npz")
            try:
                np.savez_compressed(
                    lls_save_path,
                    lls=lls_array.astype(np.float32),
                    n_prompt_seqs=n_seqs_list,
                    tok_cnt_list=tok_cnt_list,
                    min_cov_list=min_cov_list,
                    entropy_per_prompt=entropy_per_prompt.astype(np.float32),
                    dms_scores = dms_scores_np.astype(np.float32),
                )
            except Exception as e:
                warnings.warn(f"Could not save likelihoods to {lls_save_path}: {e}")
        mean_spearman = np.mean(spearman_list)
        if getattr(self, "global_rank", 0) == 0:
            self.log("gym/mean_spearman_v3", mean_spearman, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_spearman_v3", ensemble_spearman, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_log_ll_v3", ensemble_log_ll, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/entropy_and_ll_spearman_correlation", self._compute_spearman(mean_per_forward_pass, entropy_per_prompt), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return ensemble_log_ll, ensemble_spearman



    def _evaluate_and_save_variants_v4(
        self,
        batch: Dict[str, torch.Tensor],
        start_tokens: list[int] = [47, 63],
        min_target_likelihood: float = -2.5,
        max_target_likelihood: float = -0.9,
        n_opt_range_extension: int = 4,
    ):
        """
        Simplified adaptive variant evaluation.

        Generates diverse random subsamples of the context prompt whose
        mean log-likelihood lies roughly inside the user-specified band
        `[min_target_likelihood, max_target_likelihood]`.

        """
        random.seed(42)
        rng = random.Random()
        # ------------------------------------------------------------------ #
        # Canonicalise the prompt                                            #
        # ------------------------------------------------------------------ #
        sep_tok_id = self.tokenizer.sep_token_id
        start_tokens_tensor = torch.tensor(start_tokens, device=batch["input_ids"].device).unsqueeze(0)

        # Strip leading start tokens if present
        if (batch["input_ids"][0, : start_tokens_tensor.shape[1]] == start_tokens_tensor).all():
            batch["input_ids"] = batch["input_ids"][:, start_tokens_tensor.shape[1] :]

        # Ensure trailing SEP token
        if batch["input_ids"][0, -1] != sep_tok_id:
            batch["input_ids"] = torch.cat(
                [
                    batch["input_ids"],
                    torch.tensor([sep_tok_id], device=batch["input_ids"].device).unsqueeze(0),
                ],
                dim=-1,
            )

        # ------------------------------------------------------------------ #
        # Prompt statistics                                                  #
        # ------------------------------------------------------------------ #
        seq_ends = (batch["input_ids"][0] == sep_tok_id).nonzero(as_tuple=True)[0].cpu()
        total_seqs = len(seq_ends)
        seq_starts = torch.zeros_like(seq_ends)
        seq_starts[1:] = seq_ends[:-1] + 1
        seq_lengths = (seq_ends - seq_starts + 1).tolist()  # python ints

        def calculate_entropy_per_prompt(lls_array):
            exp_log = np.exp(lls_array)
            prob_denominator = np.sum(exp_log, axis=1)
            seq_probs = exp_log / prob_denominator.reshape(lls_array.shape[0], 1)
            per_prompt_entropies = -np.sum(seq_probs * np.log(seq_probs), axis=1)
            return per_prompt_entropies

        def _make_truncated_batch(idxs):
            """Deep-clone *batch* keeping only the sequences at *idxs*."""
            new_batch = self._clone_batch(batch)

            def _concat_slices(tensor):
                parts = [tensor[..., seq_starts[i] : seq_ends[i] + 1] for i in idxs]
                concat = torch.cat(parts, dim=-1)
                concat = torch.cat([start_tokens_tensor, concat], dim=-1)
                if concat[0, -1] == sep_tok_id:
                    concat = concat[:, :-1] # remove the final sep token as this is in the completions
                return concat

            new_batch["input_ids"] = _concat_slices(new_batch["input_ids"]).clone()
            if "residue_index" in new_batch and new_batch["residue_index"] is not None:
                new_batch["residue_index"] = _concat_slices(new_batch["residue_index"]).clone()
            # Also slice optional per-sequence metadata to keep consistency with v5
            if "sequence_similarities" in new_batch and new_batch["sequence_similarities"] is not None:
                new_batch["sequence_similarities"] = new_batch["sequence_similarities"][0, idxs].clone()
            if "coverages" in new_batch and new_batch["coverages"] is not None:
                new_batch["coverages"] = new_batch["coverages"][0, idxs].clone()
            return new_batch
        completion_length = batch["completion_ids"].shape[-1]
        max_context_tokens = (self.max_tokens - completion_length) - 5 # 5 is a buffer.
        avg_seq_len = sum(seq_lengths) / len(seq_lengths)
        max_n_by_tokens = max(0, min(int(max_context_tokens // avg_seq_len) + 2, total_seqs))


        @torch.no_grad()
        def _eval_prefix(n):
            n = max(0, min(n, total_seqs))
            if n == 0:
                vb = self._clone_batch(batch)
                vb["input_ids"] = None
                vb["residue_index"] = None
                L_prompt = 0
            else:
                selected_idxs = rng.sample(range(total_seqs), n)
                vb = _make_truncated_batch(selected_idxs)
                L_prompt = vb["input_ids"].shape[-1]
            vb_device = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in vb.items()}
            # Use at most 100 completions for speed
            comp_ids = vb_device["completion_ids"][:, : min(100, vb_device["completion_ids"].shape[1]), :]
            L = comp_ids.shape[-1]
            
            lls = self.score_seqs(
                vb_device["input_ids"],
                comp_ids,
                input_residue_index=vb_device.get("residue_index", None),
                completion_residue_index=vb_device.get("completion_residue_index", None),
                use_cache=self.use_kv_cache_for_scoring,
                batch_size=max((self.scoring_max_tokens) // (L + L_prompt), 1)
                if self.use_kv_cache_for_scoring
                else 1,
            )
            return float(lls.mean())

        # ------------------------------------------------------------------ #
        # Forward logspace search                                            #
        # ------------------------------------------------------------------ #
        n_forward_search = min(30, max_n_by_tokens)
        n_log_samples = n_forward_search

        # select the initial search space:
        while True:
            n_vals = [0] + [int(s) for s in np.logspace(0, np.log10(max_n_by_tokens), n_log_samples)]
            n_vals = list(set(n_vals))
            if len(n_vals) >= n_forward_search:
                break
            n_log_samples += 1
        n_vals.sort()
        n_seqs_list = []
        ll_list = []

        # find range of n_opt values that are in the target likelihood range:
        vals_in_range = []
        for n_curr in n_vals:
            ll_curr = _eval_prefix(n_curr)
            n_seqs_list.append(n_curr)
            ll_list.append(ll_curr)
            if min_target_likelihood <= ll_curr <= max_target_likelihood:
                n_opt = n_curr
                vals_in_range.append(n_curr)
        if len(vals_in_range) > 0:
            vals_in_range = np.arange(
                max(0, min(vals_in_range) - n_opt_range_extension), 
                min(max(vals_in_range) + n_opt_range_extension, max_n_by_tokens + 1) + 1
            )
            n_opt = random.choice(vals_in_range)
            if 0 not in vals_in_range:
                vals_in_range.append(0)
        else:
            vals_in_range = list(range(max_n_by_tokens + 2))
            n_opt = random.choice(vals_in_range)

        # compute likelihoods for each n_opt value in the range:
        spearman_list = []
        variants = []
        dms_scores_np = batch["DMS_scores"][0].float().cpu().numpy()
        rows, variant_lls = [], []
        n_seqs_list = []
        tok_cnt_list = []
        min_cov_list = []
        # Additional metrics to mirror v5
        min_length_ratio_list = []
        min_sequence_similarity_list, mean_sequence_similarity_list, max_sequence_similarity_list = [], [], []
        min_coverage_list, mean_coverage_list, max_coverage_list = [], [], []
        self.variant_csv_dir = os.path.join(self.gym_results_save_dir, self.timestamp)
        os.makedirs(self.variant_csv_dir, exist_ok=True)
        csv_path = os.path.join(self.variant_csv_dir, f"batch_{batch['DMS_id'].text[0]}_v4.csv")
        
        token_count_attempts = 100
        if completion_length + 2 > self.max_tokens:
            n_opt = 0
            repeats = 1
        else:
            repeats = self.gym_subsamples_per_n
        for rep in range(repeats):
            fail_count = 0
            while True:
                if n_opt == 0 and 0 in n_seqs_list:
                    n_opt = random.choice(vals_in_range)
                idxs = rng.sample(range(total_seqs), n_opt)
                rng.shuffle(idxs)
                tok_cnt = sum(seq_lengths[i] for i in idxs)
                if tok_cnt + completion_length <= self.max_tokens:
                    fail_count = 0
                    break
                else:
                    fail_count += 1
                    if fail_count > token_count_attempts:
                        n_opt = max(0, n_opt - 1)
                        fail_count = 0
            
            if n_opt == 0:
                # No context sequences selected; use empty prompt
                idxs = []
                tok_cnt = 0
                shortest_seq_len = 0
                var_batch = self._clone_batch(batch)
                var_batch["input_ids"] = None
                var_batch["residue_index"] = None
                min_completion_coverage = 0
                min_length_ratio = 0
                min_sequence_similarity = 0
                mean_sequence_similarity = 0
                max_sequence_similarity = 0
                min_coverage = 0
                mean_coverage = 0
                max_coverage = 0
            else:
                shortest_seq_len = min(seq_lengths[i] for i in idxs)
                var_batch = _make_truncated_batch(idxs)
                min_completion_coverage = shortest_seq_len / batch["completion_ids"].shape[-1] if batch["completion_ids"].shape[-1] > 0 else 0
                # Additional metrics consistent with v5
                min_length_ratio = min_completion_coverage
                seq_sims = var_batch.get("sequence_similarities", None)
                covs = var_batch.get("coverages", None)
                if seq_sims is not None:
                    min_sequence_similarity = seq_sims.min().item()
                    mean_sequence_similarity = seq_sims.mean().item()
                    max_sequence_similarity = seq_sims.max().item()
                else:
                    min_sequence_similarity = 0
                    mean_sequence_similarity = 0
                    max_sequence_similarity = 0
                if covs is not None:
                    min_coverage = covs.min().item()
                    mean_coverage = covs.mean().item()
                    max_coverage = covs.max().item()
                else:
                    min_coverage = 0
                    mean_coverage = 0
                    max_coverage = 0
            meta = {
                "variant_idx": rep,
                "replicate": rep,
                "n_seqs": n_opt,
                "n_tokens": tok_cnt,
                "seq_indices": idxs,
                "min_completion_coverage": min_completion_coverage,
                "min_length_ratio": min_length_ratio,
                "min_sequence_similarity": min_sequence_similarity,
                "mean_sequence_similarity": mean_sequence_similarity,
                "max_sequence_similarity": max_sequence_similarity,
                "min_coverage": min_coverage,
                "mean_coverage": mean_coverage,
                "max_coverage": max_coverage,
            }
            n_seqs_list.append(n_opt)
            tok_cnt_list.append(tok_cnt)
            min_cov_list.append(min_completion_coverage)
            # Track additional lists for NPZ logging
            min_length_ratio_list.append(min_length_ratio)
            min_sequence_similarity_list.append(min_sequence_similarity)
            mean_sequence_similarity_list.append(mean_sequence_similarity)
            max_sequence_similarity_list.append(max_sequence_similarity)
            min_coverage_list.append(min_coverage)
            mean_coverage_list.append(mean_coverage)
            max_coverage_list.append(max_coverage)
            variants.append((var_batch, meta))
            var_batch_device = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in var_batch.items()}
            L = var_batch_device["completion_ids"].shape[-1]
            L_prompt = 0 if var_batch_device["input_ids"] is None else var_batch_device["input_ids"].shape[-1]
            lls = self.score_seqs(
                var_batch_device["input_ids"],
                var_batch_device["completion_ids"],
                input_residue_index=var_batch_device.get("residue_index", None),
                completion_residue_index=var_batch_device.get("completion_residue_index", None),
                use_cache=self.use_kv_cache_for_scoring,
                batch_size=max((self.scoring_max_tokens) // (L + L_prompt), 1)
                if self.use_kv_cache_for_scoring
                else 1,
            )
            mean_ll = float(lls.mean())
            variant_lls.append(lls)
            spearman_list.append(float(self._compute_spearman(lls, dms_scores_np)))
            rows.append({**meta, "mean_log_likelihood": mean_ll, "spearman": float(self._compute_spearman(lls, dms_scores_np)), "DMS_id": batch["DMS_id"].text[0]})
            n_opt = random.choice(vals_in_range)


        lls_array = np.stack(variant_lls, axis=0)
        entropy_per_prompt = calculate_entropy_per_prompt(lls_array)
        if getattr(self, "global_rank", 0) == 0:
            mean_per_forward_pass = lls_array.mean(axis=1)
            sorted_indices_ll = np.argsort(-mean_per_forward_pass)
            sorted_indices_entropy = np.argsort(entropy_per_prompt)
            for top_pct in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                top_k = max(1, int(top_pct * len(sorted_indices_ll)))
                top_k_ll_mean_ll = lls_array[sorted_indices_ll[:top_k]].mean(axis=0)
                top_k_entropy_mean_ll = lls_array[sorted_indices_entropy[:top_k]].mean(axis=0)
                top_k_ll_spearman = self._compute_spearman(top_k_ll_mean_ll, dms_scores_np)
                top_k_entropy_spearman = self._compute_spearman(top_k_entropy_mean_ll, dms_scores_np)
                self.log(
                    f"gym/top_{top_pct}_ll_spearman",
                    top_k_ll_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    f"gym/bottom_{top_pct}_entropy_spearman",
                    top_k_entropy_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
        mean_lls = lls_array.mean(axis=0)
        ensemble_spearman = self._compute_spearman(mean_lls, dms_scores_np)
        ensemble_log_ll = float(mean_lls.mean())
        if getattr(self, "global_rank", 0) == 0:
            # Save CSV summary and NPZ payload consistent with v5 naming
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            lls_save_path = os.path.join(self.variant_csv_dir, f"batch_{batch['DMS_id'].text[0]}_v4_lls.npz")
            try:
                np.savez_compressed(
                    lls_save_path,
                    lls=lls_array.astype(np.float32),
                    n_prompt_seqs=n_seqs_list,
                    tok_cnt_list=tok_cnt_list,
                    min_cov_list=min_cov_list,
                    min_length_ratio_list=min_length_ratio_list,
                    min_sequence_similarity_list=np.asarray(min_sequence_similarity_list, dtype=np.float32),
                    mean_sequence_similarity_list=np.asarray(mean_sequence_similarity_list, dtype=np.float32),
                    max_sequence_similarity_list=np.asarray(max_sequence_similarity_list, dtype=np.float32),
                    min_coverage_list=np.asarray(min_coverage_list, dtype=np.float32),
                    mean_coverage_list=np.asarray(mean_coverage_list, dtype=np.float32),
                    max_coverage_list=np.asarray(max_coverage_list, dtype=np.float32),
                    entropy_per_prompt=entropy_per_prompt.astype(np.float32),
                    dms_scores = dms_scores_np.astype(np.float32),
                )
            except Exception as e:
                warnings.warn(f"Could not save likelihoods to {lls_save_path}: {e}")
            # Save diagnostic scatter similar to v5
            self._save_variant_scatter_plot(
                n_seqs_list,
                variant_lls,
                batch['DMS_id'].text[0],
            )
        mean_spearman = np.mean(spearman_list)
        if getattr(self, "global_rank", 0) == 0:
            self.log("gym/mean_spearman_v4", mean_spearman, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_spearman_v4", ensemble_spearman, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_log_ll_v4", ensemble_log_ll, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/entropy_and_ll_spearman_correlation", self._compute_spearman(mean_per_forward_pass, entropy_per_prompt), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return ensemble_log_ll, ensemble_spearman

    def _evaluate_and_save_variants_v5(
        self,
        batch: Dict[str, torch.Tensor],
        start_tokens: list[int] = [47, 63],
        min_target_likelihood: float = -2.0,
        max_target_likelihood: float = -0.8,
        n_opt_range_extension: int = 4,
    ):
        """
        Variant evaluation with Bayesian optimization.

        Uses Bayesian optimization to pick the number of context sequences
        (``n_opt``) whose mean log-likelihood lies inside the user-specified
        band `[min_target_likelihood, max_target_likelihood]`.

        The optimisation domain is the integer range `[0, max_sequences]`.
        After optimisation we draw ``self.gym_subsamples_per_n`` random
        variants that each contain a number of sequences sampled from the
        optimised neighbourhood and evaluate all completions.
        """
        import warnings
        warnings.filterwarnings(
            "ignore",
            message=r"The objective has been evaluated at point .* before",
            category=UserWarning,
        )
        try:
            from skopt.space import Integer
            from skopt import Optimizer
        except ImportError as e:
            raise ImportError("scikit-optimize is required for _evaluate_and_save_variants_v5. Please install via `pip install scikit-optimize`.") from e

        random.seed(42)
        rng = random.Random()

        # ------------------------------------------------------------------ #
        # Canonicalise the prompt                                            #
        # ------------------------------------------------------------------ #
        sep_tok_id = self.tokenizer.sep_token_id
        start_tokens_tensor = torch.tensor(start_tokens, device=batch["input_ids"].device).unsqueeze(0)

        # Strip leading start tokens if present
        if (batch["input_ids"][0, : start_tokens_tensor.shape[1]] == start_tokens_tensor).all():
            batch["input_ids"] = batch["input_ids"][:, start_tokens_tensor.shape[1] :]

        # Ensure trailing SEP token
        if batch["input_ids"][0, -1] != sep_tok_id:
            batch["input_ids"] = torch.cat(
                [
                    batch["input_ids"],
                    torch.tensor([sep_tok_id], device=batch["input_ids"].device).unsqueeze(0),
                ],
                dim=-1,
            )

        # ------------------------------------------------------------------ #
        # Prompt statistics                                                  #
        # ------------------------------------------------------------------ #
        seq_ends = (batch["input_ids"][0] == sep_tok_id).nonzero(as_tuple=True)[0].cpu()
        total_seqs = len(seq_ends)
        seq_starts = torch.zeros_like(seq_ends)
        seq_starts[1:] = seq_ends[:-1] + 1
        seq_lengths = (seq_ends - seq_starts + 1).tolist()

        completion_length = batch["completion_ids"].shape[-1]
        max_context_tokens = (self.max_tokens - completion_length) - 5  # buffer
        avg_seq_len = sum(seq_lengths) / len(seq_lengths)
        max_n_by_tokens = max(0, min(int(max_context_tokens // avg_seq_len) + 2, total_seqs))


        def _distance_to_band(ll_val: float) -> float:
            if min_target_likelihood <= ll_val <= max_target_likelihood:
                return 0.0
            if ll_val < min_target_likelihood:
                return min_target_likelihood - ll_val
            return ll_val - max_target_likelihood



        # ------------------------------------------------------------------ #
        # Helper for entropy                                                 #
        # ------------------------------------------------------------------ #
        def calculate_entropy_per_prompt(lls_array):
            exp_log = np.exp(lls_array)
            prob_denominator = np.sum(exp_log, axis=1)
            seq_probs = exp_log / prob_denominator.reshape(lls_array.shape[0], 1)
            return -np.sum(seq_probs * np.log(seq_probs), axis=1)

        def _make_truncated_batch(idxs):
            new_batch = self._clone_batch(batch)

            def _concat_slices(tensor):
                parts = [tensor[..., seq_starts[i] : seq_ends[i] + 1] for i in idxs]
                concat = torch.cat(parts, dim=-1)
                concat = torch.cat([start_tokens_tensor, concat], dim=-1)
                if concat[0, -1] == sep_tok_id:
                    concat = concat[:, :-1]
                return concat

            new_batch["input_ids"] = _concat_slices(new_batch["input_ids"]).clone()
            if "residue_index" in new_batch and new_batch["residue_index"] is not None:
                new_batch["residue_index"] = _concat_slices(new_batch["residue_index"]).clone()
            if "sequence_similarities" in new_batch and new_batch["sequence_similarities"] is not None:
                new_batch["sequence_similarities"] = new_batch["sequence_similarities"][0, idxs].clone()
            if "coverages" in new_batch and new_batch["coverages"] is not None:
                new_batch["coverages"] = new_batch["coverages"][0, idxs].clone()
            return new_batch

        # ------------------------------------------------------------------ #
        # Variant generation and scoring                                     #
        # ------------------------------------------------------------------ #
        spearman_list, variant_lls = [], []
        n_seqs_list, tok_cnt_list, min_length_ratio_list, rows = [], [], [], []
        min_sequence_similarity_list, mean_sequence_similarity_list, max_sequence_similarity_list = [], [], []
        min_coverage_list, mean_coverage_list, max_coverage_list = [], [], []
        dms_scores_np = batch["DMS_scores"][0].float().cpu().numpy()

        self.variant_csv_dir = os.path.join(self.gym_results_save_dir, self.timestamp)
        os.makedirs(self.variant_csv_dir, exist_ok=True)
        csv_path = os.path.join(self.variant_csv_dir, f"batch_{batch['DMS_id'].text[0]}_v5.csv")

        token_count_attempts = 100
        # ------------------------------------------------------------------ #
        # Bayesian optimisation loop using skopt. Each iteration corresponds
        # 1-to-1 with a context-variant evaluation that we will keep for the
        # final ensemble metrics. No secondary sampling phase.
        # ------------------------------------------------------------------ #
        opt = Optimizer(
            [Integer(0, max_n_by_tokens)], 
            n_initial_points=int(self.gym_subsamples_per_n * 0.2),
            # acq_func="gp_hedge",
            acq_func="EI",
            acq_func_kwargs={"xi": 0.05}, # larger xi = more exploration
            avoid_duplicates=False,
            random_state=0
            )
        # Track whether the zero-context prompt (n=0) has already been evaluated.
        zero_evaluated = False
        zero_objective: Optional[float] = None
        for rep in range(self.gym_subsamples_per_n):
            # Keep asking until we get a suggestion other than 0 if we've already
            # evaluated n=0. When 0 is suggested again, immediately feed the
            # cached objective back to the optimiser so it can update its model
            # without us wasting an evaluation.
            ask_retries = 0
            while True:
                suggested_n = int(opt.ask()[0])
                if suggested_n == 0 and zero_objective is not None:
                    opt.tell([0], zero_objective)
                    ask_retries += 1
                    if ask_retries > 5:
                        # Fall back to a random non-zero point to guarantee progress
                        if max_n_by_tokens >= 1:
                            suggested_n = rng.randint(1, max_n_by_tokens)
                        break
                    continue
                break
            chosen_n = max(0, min(suggested_n, total_seqs))
            # If n == 0 has already been evaluated, pick a different n
            if chosen_n == 0 and zero_evaluated:
                if max_n_by_tokens >= 1:
                    chosen_n = random.randint(1, max_n_by_tokens)
                # If max_n_by_tokens is 0 we keep chosen_n = 0 (nothing else possible)
            if n_opt_range_extension > 0:
                max_subtract = min(chosen_n - 1, n_opt_range_extension)
                max_add = min(max_n_by_tokens - chosen_n, n_opt_range_extension)
                chosen_n = max(0, min(chosen_n + rng.randint(-max_subtract, max_add), max_n_by_tokens))

            fail_count = 0
            while True:
                if completion_length + 2 > self.max_tokens:
                    chosen_n = 0
                    break
                idxs = rng.sample(range(total_seqs), chosen_n)
                rng.shuffle(idxs)
                tok_cnt = sum(seq_lengths[i] for i in idxs)
                if tok_cnt + completion_length <= self.max_tokens:
                    break
                fail_count += 1
                if fail_count > token_count_attempts:
                    chosen_n = max(1, chosen_n - 1)
                    fail_count = 0

            if chosen_n == 0:
                idxs, tok_cnt, shortest_seq_len = [], 0, 0
                var_batch = self._clone_batch(batch)
                var_batch["input_ids"] = None
                var_batch["residue_index"] = None
                min_length_ratio = 0
            else:
                shortest_seq_len = min(seq_lengths[i] for i in idxs)
                var_batch = _make_truncated_batch(idxs)
                min_length_ratio = shortest_seq_len / batch["completion_ids"].shape[-1] if batch["completion_ids"].shape[-1] > 0 else 0
                min_sequence_similarity = var_batch["sequence_similarities"].min().item() if var_batch["sequence_similarities"] is not None else 0
                mean_sequence_similarity = var_batch["sequence_similarities"].mean().item() if var_batch["sequence_similarities"] is not None else 0
                max_sequence_similarity = var_batch["sequence_similarities"].max().item() if var_batch["sequence_similarities"] is not None else 0
                min_coverage = var_batch["coverages"].min().item() if var_batch["coverages"] is not None else 0
                mean_coverage = var_batch["coverages"].mean().item() if var_batch["coverages"] is not None else 0
                max_coverage = var_batch["coverages"].max().item() if var_batch["coverages"] is not None else 0

            
            meta = {
                "variant_idx": rep,
                "replicate": rep,
                "n_seqs": chosen_n,
                "n_tokens": tok_cnt,
                "seq_indices": idxs,
                "min_length_ratio": min_length_ratio,
                "min_sequence_similarity": min_sequence_similarity,
                "mean_sequence_similarity": mean_sequence_similarity,
                "max_sequence_similarity": max_sequence_similarity,
                "min_coverage": min_coverage,
                "mean_coverage": mean_coverage,
                "max_coverage": max_coverage,
            }
            n_seqs_list.append(chosen_n)
            tok_cnt_list.append(tok_cnt)
            min_length_ratio_list.append(min_length_ratio)
            min_sequence_similarity_list.append(min_sequence_similarity)
            mean_sequence_similarity_list.append(mean_sequence_similarity)
            max_sequence_similarity_list.append(max_sequence_similarity)
            min_coverage_list.append(min_coverage)
            mean_coverage_list.append(mean_coverage)
            max_coverage_list.append(max_coverage)
            var_batch_device = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in var_batch.items()}
            L = var_batch_device["completion_ids"].shape[-1]
            L_prompt = 0 if var_batch_device["input_ids"] is None else var_batch_device["input_ids"].shape[-1]
            lls = self.score_seqs(
                var_batch_device["input_ids"],
                var_batch_device["completion_ids"],
                input_residue_index=var_batch_device.get("residue_index", None),
                completion_residue_index=var_batch_device.get("completion_residue_index", None),
                use_cache=self.use_kv_cache_for_scoring,
                batch_size=max((self.scoring_max_tokens) // (L + L_prompt), 1) if self.use_kv_cache_for_scoring else 1,
            )
            variant_lls.append(lls)
            spearman = float(self._compute_spearman(lls, dms_scores_np))
            spearman_list.append(spearman)
            mean_ll_val = float(lls.mean())
            # Mark and cache the zero-context objective value
            if chosen_n == 0:
                zero_evaluated = True
                zero_objective = _distance_to_band(mean_ll_val)
            rows.append({**meta, "mean_log_likelihood": mean_ll_val, "spearman": spearman, "DMS_id": batch["DMS_id"].text[0]})
            # Update Bayesian optimizer with the observed objective value
            opt.tell([chosen_n], _distance_to_band(mean_ll_val))

        lls_array = np.stack(variant_lls, axis=0)
        entropy_per_prompt = calculate_entropy_per_prompt(lls_array)
        if getattr(self, "global_rank", 0) == 0:
            mean_per_forward_pass = lls_array.mean(axis=1)
            sorted_indices_ll = np.argsort(-mean_per_forward_pass)
            sorted_indices_entropy = np.argsort(entropy_per_prompt)
            for top_pct in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                top_k = max(1, int(top_pct * len(sorted_indices_ll)))
                top_k_ll_mean_ll = lls_array[sorted_indices_ll[:top_k]].mean(axis=0)
                top_k_entropy_mean_ll = lls_array[sorted_indices_entropy[:top_k]].mean(axis=0)
                top_k_ll_spearman = self._compute_spearman(top_k_ll_mean_ll, dms_scores_np)
                top_k_entropy_spearman = self._compute_spearman(top_k_entropy_mean_ll, dms_scores_np)
                self.log(
                    f"gym/top_{top_pct}_ll_spearman",
                    top_k_ll_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )
                self.log(
                    f"gym/bottom_{top_pct}_entropy_spearman",
                    top_k_entropy_spearman,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                    batch_size=1,
                )

        if getattr(self, "global_rank", 0) == 0:
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            lls_save_path = csv_path.replace(".csv", ".npz")
            np.savez_compressed(
                lls_save_path,
                lls=lls_array.astype(np.float32),
                n_prompt_seqs=n_seqs_list,
                tok_cnt_list=tok_cnt_list,
                min_length_ratio_list=min_length_ratio_list,
                min_sequence_similarity_list=min_sequence_similarity_list,
                mean_sequence_similarity_list=mean_sequence_similarity_list,
                max_sequence_similarity_list=max_sequence_similarity_list,
                min_coverage_list=min_coverage_list,
                mean_coverage_list=mean_coverage_list,
                max_coverage_list=max_coverage_list,
                entropy_per_prompt=entropy_per_prompt.astype(np.float32),
                dms_scores=dms_scores_np.astype(np.float32),
            )

            self._save_variant_scatter_plot(
                n_seqs_list,
                variant_lls,
                batch['DMS_id'].text[0],
            )

            self.log("gym/mean_spearman_v5", np.mean(spearman_list), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_spearman_v5", float(self._compute_spearman(lls_array.mean(axis=0), dms_scores_np)), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("gym/ensemble_log_ll_v5", float(lls_array.mean()), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return float(lls_array.mean()), float(self._compute_spearman(lls_array.mean(axis=0), dms_scores_np))


    def validation_step_proteingym(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate ProteinGym batch with multiple randomly-subsampled contexts."""
        if batch_idx is None:
            batch_idx = -1  # fallback when Lightning doesn't supply the index

        ensemble_log_ll, ensemble_spearman = self._evaluate_and_save_variants_v4(
            batch
        )

        # Log aggregate metrics so that Lightning tracks them across batches
        self.log(
            "gym/spearman",
            ensemble_spearman,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "gym/log_likelihood",
            ensemble_log_ll,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return torch.tensor(ensemble_spearman, device=self.device, dtype=torch.float32)

    def validation_step_family_classification(
        self, batch: Dict[str, torch.Tensor], task: str = "classification"
    ) -> torch.Tensor:
        """
        Val step for family classification task.

        Assumes that batch contains the following:
        input_ids: the prompt (i.e. MSA)
        completion_ids: the completions (i.e. mutated seqs / seqs to be classified)
        """
        assert (
            batch["family_labels"].ndim == 2
            and batch["input_ids"].ndim == 2
            and batch["input_ids"].shape[0] == 1
            and batch["completion_ids"].ndim == 3
        )
        L = batch["completion_ids"].shape[-1]
        L_prompt = batch["input_ids"].shape[-1]
        lls = self.score_seqs(
            batch["input_ids"],
            batch["completion_ids"],
            input_residue_index=batch.get("residue_index", None),
            completion_residue_index=batch.get("completion_residue_index", None),
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=1,
            # (self.scoring_max_tokens - L_prompt) // L
            # if self.use_kv_cache_for_scoring
            # else 1,
        )
        target_vals = batch["family_labels"][0].cpu().numpy()
        # TODO: maybe specify which family is classified in metric

        precision, recall, thresholds = precision_recall_curve(target_vals, lls)
        metric = auc(recall, precision)
        self.log(
            f"val/{batch.get('ds_name').text[0]}_auprc_classification",
            metric,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        au_roc = roc_auc_score(target_vals, lls)
        self.log(
            f"val/{batch.get('ds_name').text[0]}_auroc_classification",
            au_roc,
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=False,
        )
        k_vals = [k for k in [1, 2, 5, 10] if k < len(target_vals)]
        for top_k in k_vals:
            top_k_acc = len(
                set(np.argsort(lls)[::-1][:top_k]).intersection(
                    set(np.where(target_vals)[0])
                )
            ) / min(top_k, sum(target_vals))
            self.log(
                f"val/{batch.get('ds_name').text[0]}_top_{top_k}_acc_classification",
                top_k_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False,
            )
        if batch["ds_name"].text[0] in ["pfam_fam_class"]:
            # only do this for evals where the eval seqs remain the same across
            # batches and we consider the likelihood of each eval seq conditioned
            # on different family 'prompts'
            self.update_family_likelihoods(batch, lls)
        return torch.tensor(metric, device=self.device, dtype=torch.float32)

    def update_family_likelihoods(self, batch, lls):
        """
        each batch evaluates the ll of all test seqs
        conditioned on a single family. This means
        we can re-use the KV cache across all seqs.
        For the multi-class objective we need to store
        the likelihood of each seq conditioned on each
        family. lls from each batch are stored here
        """
        if not hasattr(self, "family_likelihoods"):
            self.family_likelihoods = {}
            self.batch_counter = 0
        val_ds_name = batch["ds_name"].text[0]
        if val_ds_name not in self.family_likelihoods:
            self.family_likelihoods[val_ds_name] = {}
        for eval_seq_ix, bin_label in enumerate(
            batch["family_labels"][0].cpu().numpy()
        ):
            ll = lls[eval_seq_ix]
            if eval_seq_ix not in self.family_likelihoods[val_ds_name]:
                self.family_likelihoods[val_ds_name][eval_seq_ix] = {}
            if bin_label == 1:
                if (
                    1 in self.family_likelihoods[val_ds_name][eval_seq_ix]
                ):  # 1 fam per seq
                    warnings.warn("Multiple families assigned for eval seq")
                self.family_likelihoods[val_ds_name][eval_seq_ix][1] = ll
            else:
                if 0 not in self.family_likelihoods[val_ds_name][eval_seq_ix]:
                    self.family_likelihoods[val_ds_name][eval_seq_ix][0] = []
                self.family_likelihoods[val_ds_name][eval_seq_ix][0].append(ll)
        self.batch_counter += 1
        if self.trainer.sanity_checking:
            self.family_likelihoods = {}
            self.batch_counter = 0

    def on_validation_epoch_end(self):
        """
        Likelihood scores are accumulated across batches
        at end of epoch multi-class metrics can be calcd
        """
        super().on_validation_epoch_end()
        if self.trainer.sanity_checking:
            return
        if hasattr(self, "family_likelihoods"):
            ce_scores = []
            acc_scores = []
            for val_name in self.family_likelihoods:
                for eval_seq, lls in self.family_likelihoods[val_name].items():
                    # softmax likelihoods to get probability over families
                    labels = np.array([1] + [0] * len(lls[0]))
                    if 1 in lls:
                        lls_arr = np.array([lls[1]] + lls[0])
                        self.log(
                            f"val/{val_name}_mean_ll_across_fam_prompts",
                            lls_arr.mean(),
                            on_step=False,
                            add_dataloader_idx=False,
                        )
                        self.log(
                            f"val/{val_name}_variance_ll_across_fam_prompts",
                            np.var(lls_arr),
                            on_step=False,
                            add_dataloader_idx=False,
                        )
                        lls_arr = lls_arr - lls_arr.max()
                        probs = np.exp(lls_arr) / np.exp(lls_arr).sum()
                        # calculate cross entropy
                        ce = -np.log(probs[labels == 1]).mean()
                        ce_scores.append(ce)
                        if np.argmax(probs) == 0:
                            acc_scores.append(1)
                        else:
                            acc_scores.append(0)
                    else:
                        warnings.warn(f"Warning: Eval seq has no positive family")

                self.log(
                    f"val/{val_name}_multiclass_cr_ent",
                    sum(ce_scores) / len(ce_scores),
                    on_step=False,
                    add_dataloader_idx=False,
                )

                self.log(
                    f"val/{val_name}_multiclass_acc",
                    sum(acc_scores) / len(acc_scores),
                    on_step=False,
                    add_dataloader_idx=False,
                )
        self.family_likelihoods = {}
        self.batch_counter = 0

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # uncomment for debugging ddp (train.py +experiment=ddp_test)
        # print(f"Rank: {self.trainer.global_rank}", batch["identifier"].text, flush=True)
        forward_kwargs = self.get_forward_kwargs(batch)
        # TODO: write a wrapper to compute loss / metrics if we have 3di tokens?
        # one option would be to write our own versions of classes llike llamaforcausallm

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            **forward_kwargs,
        )
        loss = outputs.loss
        # TODO: handle ds-level metrics for train batches which can include multiple datasets
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
        return loss

    def on_train_epoch_end(self):
        # Commenting out as may cause deadlock in DDP
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19604
        log.info("Train epoch end %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
