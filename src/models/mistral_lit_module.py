import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningModule
from scipy.stats import spearmanr
from torchmetrics import MeanMetric
from transformers import MistralConfig, MistralForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

# Initialize the tokenizer
with open("scripts/vocab.json", "r") as jf:
    vocab = json.load(jf)


def calc_grad_norm(params):
    grad_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]
        ),
        2,
    )

    return grad_norm


class UpdatedDynamicCache(DynamicCache):
    """A DynamicCache that allows for batched key-value caching.
    Manually implements latest version of DynamicCache from transformers.
    (once this is released we can remove this class)
    """

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(
                repeats, dim=0
            )


def accuracy_from_outputs(model_outputs, input_ids, start_ix=0, ignore_index=-100):
    """Compute the accuracy of the target sequence given the model outputs.

    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.
        ignore_index: Token index to ignore when computing accuracy.
            (this will get added automatically by the data collator as padding)

    Returns:
        The accuracy of the target sequence.
    """
    logits = model_outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:-1, :].contiguous()  # b, L, V
    shift_labels = input_ids[..., start_ix + 1 :].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    non_padding_mask = shift_labels != ignore_index
    # TODO: we might also want to ignore gaps...
    accuracy = (shift_logits.argmax(-1) == shift_labels).float()
    accuracy = (accuracy * non_padding_mask).sum() / non_padding_mask.sum()
    return accuracy


def log_likelihood_from_outputs(model_outputs, input_ids, start_ix=0, flatten=False):
    """Compute the negative log likelihood of the target sequence given the model outputs.

    Args:
        model_outputs: The model outputs from the forward pass.
        input_ids: The input sequence.

    Returns:
        The negative log likelihood of the target sequence.
    """
    logits = model_outputs.logits
    # https://github.com/huggingface/transformers/blob/4a6024921fa142f28e8d0034ae28693713b3bfd0/src/transformers/models/mistral/modeling_mistral.py#L1210

    # Shift so that tokens < n predict n
    shift_logits = logits[..., start_ix:-1, :].contiguous()  # b, L, V
    shift_labels = input_ids[..., start_ix + 1 :].contiguous()  # b, L
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # TODO: handle possible padding?

    if flatten:
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        log_likelihood = -loss_fct(shift_logits, shift_labels)
    else:
        log_likelihood = -loss_fct(shift_logits.permute(0, 2, 1), shift_labels)

    return log_likelihood


class MistralLitModule(LightningModule):
    def __init__(
        self,
        config: MistralConfig,
        lr: float = 1e-4,
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        weight_decay: float = 0.01,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = MistralForCausalLM(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        # TODO: add a max tokens in batch kwarg for scoring purposes.
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
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

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["input_ids"])
        loss = outputs.loss
        logits = outputs.logits
        accuracy = accuracy_from_outputs(outputs, batch["input_ids"], ignore_index=-100)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False
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
        self.log("lr", optimizer.param_groups[0]["lr"])

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
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=self.scoring_max_tokens // L
            if self.use_kv_cache_for_scoring
            else 1,
        )
        spearman_corr, _ = spearmanr(lls, batch["DMS_scores"][0].cpu().numpy())
        # TODO: log the specific landscape name
        self.log(
            "gym/spearman", spearman_corr, on_step=False, on_epoch=True, prog_bar=False
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
            optim_dict["lr_scheduler"] = scheduler
        return optim_dict
