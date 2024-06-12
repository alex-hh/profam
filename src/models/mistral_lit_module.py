import json
from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning import LightningModule
from scipy.stats import spearmanr
from torchmetrics import MeanMetric
from transformers import MistralConfig, MistralForCausalLM

# Initialize the tokenizer
with open("scripts/vocab.json", "r") as jf:
    vocab = json.load(jf)


class MistralLitModule(LightningModule):
    def __init__(self, config: MistralConfig, compile: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = MistralForCausalLM(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["input_ids"])
        loss = outputs.loss
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        with torch.no_grad():
            self.log(
                "train/n_seqs",
                (batch["input_ids"] == vocab["[SEP]"])
                .float()
                .sum(axis=1)
                .mean()
                .item(),  #  TODO: remove hardcoded SEP token
                on_step=True,
                on_epoch=False,
            )
        return loss

    def validation_step_proteingym(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Assumes that batch contains the following:

        input_ids: the prompt (i.e. MSA)
        completion_ids: the completions (i.e. mutated sequences)

        on caching: it seems like, if we modify what is passed to attention forward, existing cache
        might just work. currently model/sampling loop probably passes just the next token.
        """
        # https://huggingface.co/docs/transformers/v4.41.3/en/llm_tutorial_optimization#2-flash-attention
        # v1: no cache
        # we concatenate completion ids to input ids in the same way we would at generation time
        # its important that a start of sequence token is present in the completion ids
        # https://github.com/huggingface/transformers/blob/c39aaea97224cac70b0125f58a47ed74d637c4ac/src/transformers/generation/utils.py#L2630
        # TODO: handle completion batch size > 1
        assert (
            batch["input_ids"].shape[0] == 1
        ), "Only batch size 1 is supported for proteingym evaluation"
        # N.B. batch size being one means attention_mask isn't needed
        all_nlls = []
        assert (
            batch["input_ids"].ndim == 2
            and batch["completion_ids"].ndim == 3
            and batch["DMS_scores"].ndim == 2
        )  # b, L; b, n, L
        completion_start_ix = batch["input_ids"].shape[1] + 1  # skip the SEP token
        for completion_ix in range(batch["completion_ids"].shape[1]):
            input_ids = torch.cat(
                [
                    batch["input_ids"],
                    batch["completion_ids"][
                        :, completion_ix
                    ],  # completion_ids have sep token at ix 0
                ],
                dim=1,
            )
            assert (
                input_ids[..., completion_start_ix - 1] == vocab["[SEP]"]
            )  #  SEP token
            outputs = self.model(input_ids)
            logits = outputs.logits
            # https://github.com/huggingface/transformers/blob/4a6024921fa142f28e8d0034ae28693713b3bfd0/src/transformers/models/mistral/modeling_mistral.py#L1210

            # Shift so that tokens < n predict n
            shift_logits = logits[..., completion_start_ix - 1 : -1, :].contiguous()
            shift_labels = input_ids[..., completion_start_ix:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            nll = -loss_fct(shift_logits, shift_labels).mean(
                -1
            )  # mean is invariant to seq len
            all_nlls.append(nll.item())

        nlls = np.array(all_nlls)
        spearman_corr, _ = spearmanr(nlls, batch["DMS_scores"][0].cpu().numpy())
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
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
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
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, weight_decay=0.01)
        return {"optimizer": optimizer}
