import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from lightning import LightningModule
from scipy.stats import spearmanr
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.optimization import get_scheduler

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


class GPT2SingleFamilyLitModule(LightningModule):
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

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["input_ids"])
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

    def score_mutants(self, input_ids, completion_ids, batch_size: int = 1):
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
