import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import tqdm
from lightning import LightningModule
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from transformers import PreTrainedTokenizerFast
from transformers.optimization import get_scheduler

from src.models.utils import (
    UpdatedDynamicCache,
    accuracy_from_outputs,
    log_likelihood_from_outputs,
)


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


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
        scoring_max_tokens: int = 10240,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.save_hyperparameters(logger=False)
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scheduler_name = scheduler_name
        self.scoring_max_tokens = scoring_max_tokens

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=self.eps,
        )
        optim_dict = {"optimizer": optimizer}
        if self.scheduler_name is not None:
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
        self._t1 = time.time()
        self.log(
            "train/batch_time",
            self._t1 - self._t0,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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
        # labels have -100 at padding positions due to collater
        accuracy = accuracy_from_outputs(outputs, batch["labels"], ignore_index=-100)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        # https://huggingface.co/docs/transformers/perplexity
        # n.b. this might be biased for batch size > 1 (averaging over all docs before exp rather than other way round
        with torch.no_grad():
            self.log(
                "train/ppl",
                torch.exp(loss),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "train/n_seqs",
                (batch["input_ids"] == self.tokenizer.sep_token_id)
                .float()
                .sum(axis=1)
                .mean()
                .item(),
                on_step=True,
                on_epoch=False,
            )
        return loss

    def on_before_optimizer_step(self, optimizer):
        # https://github.com/Lightning-AI/pytorch-lightning/issues/1462
        self.log(
            "grad_norm",
            calc_grad_norm(self.model.parameters()),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("lr", optimizer.param_groups[0]["lr"])

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
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
        accuracy = accuracy_from_outputs(outputs, batch["labels"], ignore_index=-100)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False)
        # n.b. this might be biased for batch size > 1
        self.log(
            "val/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
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
        accuracy = accuracy_from_outputs(outputs, batch["labels"], ignore_index=-100)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # n.b. this might be biased for batch size > 1
        self.log(
            "test/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss


class BaseSingleSequenceLitModule(BaseLitModule):

    # TODO: make this part of a mixin so that it can be reused across models
    # c.f. GenerationsMixin
    def score_seqs(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
    ):
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
        assert input_ids.ndim == 2 and completion_ids.ndim == 3  # b, L; b, n, L
        L = completion_ids.shape[-1]
        all_lls = []
        for batch_start in range(0, completion_ids.shape[1], batch_size):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            input_ids = completion_ids[
                :, batch_start : batch_start + batch_size
            ].reshape(
                -1, L
            )  # b_mut, L
            outputs = self.model(input_ids=input_ids)
            labels = torch.where(
                input_ids == self.tokenizer.pad_token_id, -100, input_ids.clone()
            )
            log_likelihood = log_likelihood_from_outputs(outputs, labels, start_ix=0)
            all_lls.append(log_likelihood.mean(-1))  # b_mut

        lls = torch.cat(all_lls).cpu().numpy()
        return lls

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
        lls = self.score_seqs(
            batch["input_ids"],
            batch["completion_ids"],
            batch_size=self.scoring_max_tokens // L,
        )
        spearman_corr, _ = spearmanr(lls, batch["DMS_scores"][0].cpu().numpy())
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman", spearman_corr, on_step=False, on_epoch=True, prog_bar=True
        )


class BaseFamilyLitModule(BaseLitModule):
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerFast,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        use_seq_pos: bool = False,
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
        )
        self.scoring_max_tokens = scoring_max_tokens
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.dataset_sample_counts = {}
        self.doc_hash_counts = {}
        self.use_seq_pos = use_seq_pos

    def get_forward_kwargs(self, batch):
        return {"seq_pos": batch.get("seq_pos", None)} if self.use_seq_pos else {}

    def _score_seqs_kv_cache(
        self,
        input_ids,
        completion_ids,
        seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        forward_kwargs = {"seq_pos": seq_pos} if self.use_seq_pos else {}
        outputs = self.model(input_ids=input_ids, use_cache=True, **forward_kwargs)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        L = completion_ids.shape[-1]
        for batch_start in tqdm.tqdm(
            range(0, completion_ids.shape[1], batch_size), disable=not verbose
        ):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            this_input_ids = completion_ids[
                :, batch_start : batch_start + batch_size
            ].reshape(
                -1, L
            )  # b_mut, L
            forward_kwargs = {}
            if self.use_seq_pos:
                this_seq_pos = completion_seq_pos[
                    :, batch_start : batch_start + batch_size
                ].reshape(
                    -1, L
                )  # TODO: does cache affect seq pos in any way? doesnt seem like it should
                forward_kwargs["seq_pos"] = this_seq_pos
            actual_batch_size = this_input_ids.shape[0]
            cache = UpdatedDynamicCache.from_legacy_cache(past_key_values)

            outputs = self.model(
                input_ids=this_input_ids,
                past_key_values=cache.batch_repeat_interleave(actual_batch_size),
                use_cache=True,
                **forward_kwargs,
            )
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            log_likelihood = log_likelihood_from_outputs(outputs, labels, start_ix=0)
            all_lls.append(log_likelihood.mean(-1))  # b_mut

        lls = torch.cat(all_lls).cpu().numpy()
        return lls

    def _score_seqs_no_cache(
        self,
        input_ids,
        completion_ids,
        batch_size: int = 1,
        seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        if batch_size > 1:
            raise NotImplementedError(
                "Mutant batch size > 1 not yet supported for mutant scoring"
            )
        all_lls = []
        completion_start_pos = input_ids.shape[1] + 1  # skip the SEP token
        for completion_ix in tqdm.tqdm(
            range(completion_ids.shape[1]), disable=not verbose
        ):
            this_input_ids = torch.cat(
                [input_ids, completion_ids[:, completion_ix]],
                dim=1,
            )
            forward_kwargs = {}
            # https://github.com/huggingface/transformers/blob/048f599f3506e57e0a595b455d9d2834c8d45023/src/transformers/data/data_collator.py#L823
            labels = torch.where(
                this_input_ids == self.tokenizer.pad_token_id,
                -100,
                this_input_ids.clone(),
            )
            assert (
                this_input_ids[..., completion_start_pos - 1]
                == self.tokenizer.sep_token_id
            )  # SEP token
            if self.use_seq_pos:
                this_seq_pos = torch.cat(
                    [seq_pos, completion_seq_pos[:, completion_ix]],
                    dim=1,
                )
                forward_kwargs["seq_pos"] = this_seq_pos
            outputs = self.model(input_ids=this_input_ids, **forward_kwargs)
            # TODO: maybe relabel start_ix - a bit confusing
            log_likelihood = log_likelihood_from_outputs(
                outputs, labels, start_ix=completion_start_pos - 1
            )  # 1, L
            all_lls.append(log_likelihood.mean(-1).item())
        lls = np.array(all_lls)
        return lls

    # TODO: make this part of a mixin so that it can be reused across models
    # c.f. GenerationsMixin
    def score_seqs(
        self,
        input_ids,
        completion_ids,
        use_cache: bool = True,
        batch_size: int = 1,
        input_seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
    ):
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
        assert input_ids.ndim == 2 and completion_ids.ndim == 3  # b, L; b, n, L
        if use_cache:
            return self._score_seqs_kv_cache(
                input_ids,
                completion_ids,
                batch_size=batch_size,
                seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
            )
        else:
            return self._score_seqs_no_cache(
                input_ids,
                completion_ids,
                batch_size=batch_size,
                seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
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
            input_seq_pos=batch.get("seq_pos", None),
            completion_seq_pos=batch.get("completion_seq_pos", None),
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=(self.scoring_max_tokens - L_prompt) // L
            if self.use_kv_cache_for_scoring
            else 1,
        )
        spearman_corr, _ = spearmanr(lls, batch["DMS_scores"][0].cpu().numpy())
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman", spearman_corr, on_step=False, on_epoch=True, prog_bar=False
        )

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
            input_seq_pos=batch.get("seq_pos", None),
            completion_seq_pos=batch.get("completion_seq_pos", None),
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=(self.scoring_max_tokens - L_prompt) // L
            if self.use_kv_cache_for_scoring
            else 1,
        )
        target_vals = batch["family_labels"][0].cpu().numpy()
        # TODO: maybe specify which family is classified in metric

        precision, recall, thresholds = precision_recall_curve(target_vals, lls)
        metric = auc(recall, precision)
        self.log(
            "auprc_fam_classification",
            metric,
            on_step=False,
            on_epoch=True,
        )
        au_roc = roc_auc_score(target_vals, lls)
        self.log(
            "auroc_fam_classification",
            au_roc,
            on_step=False,
            on_epoch=True,
        )
        return torch.tensor(metric, device=self.device)

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
        # labels have -100 at padding positions due to collater
        accuracy = accuracy_from_outputs(outputs, batch["labels"], ignore_index=-100)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        # https://huggingface.co/docs/transformers/perplexity
        # n.b. this might be biased for batch size > 1 (averaging over all docs before exp rather than other way round
        with torch.no_grad():
            self.log(
                "train/ppl",
                torch.exp(loss),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "train/n_seqs",
                (batch["input_ids"] == self.tokenizer.sep_token_id)
                .float()
                .sum(axis=1)
                .mean()
                .item(),
                on_step=True,
                on_epoch=False,
            )
            self.log_ds_sample_counts(batch)

            # TODO: verify that on_epoch skips missing batches
            if "ds_name" in batch:
                per_dataset_accuracies = accuracy_from_outputs(
                    outputs,
                    batch["input_ids"],
                    dataset_names=batch["ds_name"].text,
                )
                self.log_dict(
                    {
                        f"train/{k}_acc": v.item()
                        for k, v in per_dataset_accuracies.items()
                    },
                    on_step=False,
                    on_epoch=True,
                )

            if "doc_hash" in batch:
                for i, (dataset, doc_hash) in enumerate(
                    zip(batch["ds_name"].text, batch["doc_hash"].text)
                ):
                    self.doc_hash_counts[dataset] = self.doc_hash_counts.get(
                        dataset, {}
                    )
                    self.doc_hash_counts[dataset][doc_hash] = (
                        self.doc_hash_counts[dataset].get(doc_hash, 0) + 1
                    )
                self.log_dict(
                    {
                        f"{k}_max_sampled_doc": max(v.values())
                        for k, v in self.doc_hash_counts.items()
                    },
                    on_step=False,
                    on_epoch=True,
                )
            if "total_num_sequences" in batch:
                self.log(
                    "train/total_num_sequences",
                    batch["total_num_sequences"].float().mean(),
                )
        return loss

    def log_ds_sample_counts(self, batch):
        sd_name = batch["ds_name"].text
        for ds in sd_name:
            self.dataset_sample_counts[ds] = self.dataset_sample_counts.get(ds, 0) + 1

        self.log_dict(
            {
                f"train/{k}_times_sampled": v
                for k, v in self.dataset_sample_counts.items()
            },
            on_step=True,
            on_epoch=False,
        )
