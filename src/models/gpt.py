import json
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from lightning import LightningModule
from scipy.stats import spearmanr
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

from src.models.utils import (
    UpdatedDynamicCache,
    accuracy_from_outputs,
    calc_grad_norm,
    log_likelihood_from_outputs,
)

with open("scripts/vocab.json", "r") as jf:
    vocab = json.load(jf)


class GPT2SingleSequenceLitModule(LightningModule):
    def __init__(
        self,
        config: GPT2Config,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 64000,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = GPT2LMHeadModel(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scoring_max_tokens = scoring_max_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
        outputs = self(
            batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
        )
        loss = outputs.loss
        logits = outputs.logits
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        # https://huggingface.co/docs/transformers/perplexity
        # n.b. this might be biased for batch size > 1 (averaging over all docs before exp rather than other way round)
        self.log(
            "train/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
        )
        with torch.no_grad():
            self.log(
                "train/n_seqs",
                (batch["input_ids"] == vocab["[SEP]"])
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
        self.log(
            "lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def score_mutants(self, input_ids, completion_ids, batch_size: int = 1):
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
        assert input_ids.ndim == 2 and completion_ids.ndim == 3  # b, L; b, n, L
        print(completion_ids.shape)
        L = completion_ids.shape[-1]
        all_lls = []
        for batch_start in range(0, completion_ids.shape[1], batch_size):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            input_ids = completion_ids[
                :, batch_start : batch_start + batch_size
            ].reshape(
                -1, L
            )  # b_mut, L
            actual_batch_size = input_ids.shape[0]
            outputs = self.model(input_ids)
            log_likelihood = log_likelihood_from_outputs(outputs, input_ids, start_ix=0)
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
        lls = self.score_mutants(
            batch["input_ids"],
            batch["completion_ids"],
            batch_size=self.scoring_max_tokens // L,
        )
        spearman_corr, _ = spearmanr(lls, batch["DMS_scores"][0].cpu().numpy())
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman", spearman_corr, on_step=False, on_epoch=True, prog_bar=True
        )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
        loss = outputs.loss
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
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
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
        loss = outputs.loss
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # n.b. this might be biased for batch size > 1
        self.log(
            "test/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
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


class GPT2LitModule(LightningModule):
    def __init__(
        self,
        config: GPT2Config,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 8000,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = GPT2LMHeadModel(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.scoring_max_tokens = scoring_max_tokens

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=False,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
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
        outputs = self(
            batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
        )
        loss = outputs.loss
        logits = outputs.logits
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True
        )
        # https://huggingface.co/docs/transformers/perplexity
        # n.b. this might be biased for batch size > 1 (averaging over all docs before exp rather than other way round)
        self.log(
            "train/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
        )
        with torch.no_grad():
            self.log(
                "train/n_seqs",
                (batch["input_ids"] == vocab["[SEP]"])
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
        self.log(
            "lr",
            optimizer.param_groups[0]["lr"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def _score_mutants_kv_cache(self, input_ids, completion_ids, batch_size: int = 1):
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        outputs = self.model(input_ids, use_cache=True)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        L = completion_ids.shape[-1]
        for batch_start in range(0, completion_ids.shape[1], batch_size):
            # TODO: for batch_size > 1, we need to expand out the cache - c.f. generate
            input_ids = completion_ids[
                :, batch_start : batch_start + batch_size
            ].reshape(
                -1, L
            )  # b_mut, L
            actual_batch_size = input_ids.shape[0]
            cache = UpdatedDynamicCache.from_legacy_cache(past_key_values)
            outputs = self.model(
                input_ids,
                past_key_values=cache.batch_repeat_interleave(actual_batch_size),
                use_cache=True,
            )
            log_likelihood = log_likelihood_from_outputs(outputs, input_ids, start_ix=0)
            all_lls.append(log_likelihood.mean(-1))  # b_mut

        lls = torch.cat(all_lls).cpu().numpy()
        return lls

    def _score_mutants_no_cache(self, input_ids, completion_ids, batch_size: int = 1):
        # input_ids is b, L; completion_ids is b, n, L
        if batch_size > 1:
            raise NotImplementedError(
                "Mutant batch size > 1 not yet supported for mutant scoring"
            )
        all_lls = []
        completion_start_pos = input_ids.shape[1] + 1  # skip the SEP token
        for completion_ix in range(completion_ids.shape[1]):
            input_ids = torch.cat(
                [input_ids, completion_ids[:, completion_ix]],
                dim=1,
            )
            assert (
                input_ids[..., completion_start_pos - 1] == vocab["[SEP]"]
            )  # SEP token
            outputs = self.model(input_ids)
            # TODO: maybe relabel start_ix - a bit confusing
            log_likelihood = log_likelihood_from_outputs(
                outputs, input_ids, start_ix=completion_start_pos - 1
            )  # 1, L
            all_lls.append(log_likelihood.mean(-1).item())
        lls = np.array(all_lls)
        return lls

    # TODO: make this part of a mixin so that it can be reused across models
    # c.f. GenerationsMixin
    def score_mutants(
        self, input_ids, completion_ids, use_cache: bool = True, batch_size: int = 1
    ):
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"
        assert input_ids.ndim == 2 and completion_ids.ndim == 3  # b, L; b, n, L
        if use_cache:
            return self._score_mutants_kv_cache(
                input_ids, completion_ids, batch_size=batch_size
            )
        else:
            return self._score_mutants_no_cache(
                input_ids, completion_ids, batch_size=batch_size
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
        lls = self.score_mutants(
            batch["input_ids"],
            batch["completion_ids"],
            batch_size=self.scoring_max_tokens // L,
        )
        spearman_corr, _ = spearmanr(lls, batch["DMS_scores"][0].cpu().numpy())
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman", spearman_corr, on_step=False, on_epoch=True, prog_bar=True
        )

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # we check whether we are in proteingym loader by looking at keys in batch
        if "DMS_scores" in batch:
            outputs = self.validation_step_proteingym(batch)
            return outputs
        else:
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
        loss = outputs.loss
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
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
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
        loss = outputs.loss
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # n.b. this might be biased for batch size > 1
        self.log(
            "test/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False
        )
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        optim_dict = {"optimizer": optimizer}
        if self.scheduler_name is not None:
            scheduler = get_scheduler(
                self.scheduler_name,
                optimizer,
                num_warmup_steps=self.num_warmup_steps,
                num_training_steps=self.num_training_steps,
            )
            # https://lightning.ai/docs/pytorch/stable/common/optimization.html#automatic-optimization
            optim_dict["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }
        return optim_dict
