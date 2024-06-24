import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import wandb
from lightning import LightningModule
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC, BinaryPrecisionRecallCurve
from transformers import MistralConfig, MistralForCausalLM
from transformers.cache_utils import DynamicCache

# Initialize the tokenizer
with open("scripts/vocab.json", "r") as jf:
    vocab = json.load(jf)


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


def accuracy_from_outputs(model_outputs,
                          input_ids,
                          start_ix=0,
                          ignore_index=-100,
                          dataset_names=None,
                          ):
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
    if dataset_names is not None:
        ds_accuracies = {}
        for ds_name in set(dataset_names):
            in_dataset_mask = np.array(dataset_names) == ds_name
            ds_accuracies[ds_name] = (
                (
                    accuracy[in_dataset_mask] *
                    non_padding_mask[in_dataset_mask]
                ).sum() /
                non_padding_mask[in_dataset_mask].sum()
            )
        return ds_accuracies
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
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = MistralForCausalLM(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        # TODO: add a max tokens in batch kwarg for scoring purposes.
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.scoring_max_tokens = scoring_max_tokens
        self.dataset_sample_counts = {}
        self.doc_hash_counts = {}

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
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train/batch_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/perplexity
            # n.b. this might be biased for batch size > 1 (averaging over all docs before exp rather than other way round)
            self.log(
                "train/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
            )
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
            self.log_ds_sample_counts(batch)
            if "ds_name" in batch:
                per_dataset_accuracies = accuracy_from_outputs(
                    outputs,
                    batch["input_ids"],
                    dataset_names=batch["ds_name"].text,
                )
                self.log_dict(
                    {f"train/{k}_acc": v.item() for k, v in per_dataset_accuracies.items()},
                    on_step=True,
                    on_epoch=False
                )

            if "doc_hash" in batch:
                for i, (dataset, doc_hash) in enumerate(
                    zip(batch["ds_name"].text, batch["doc_hash"].text)
                ):
                    self.doc_hash_counts[dataset] = self.doc_hash_counts.get(dataset, {})
                    self.doc_hash_counts[dataset][doc_hash] = (
                        self.doc_hash_counts[dataset].get(doc_hash, 0) + 1
                    )
                self.log_dict(
                    {f"{k}_max_sampled_doc": max(v.values()) for k, v in self.doc_hash_counts.items()},
                    on_step=True,
                    on_epoch=False,
                )
        return loss

    def _score_seqs_kv_cache(self, input_ids, completion_ids, batch_size: int = 1):
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

    def _score_seqs_no_cache(self, input_ids, completion_ids, batch_size: int = 1):
        # input_ids is b, L; completion_ids is b, n, L
        if batch_size > 1:
            raise NotImplementedError(
                "Target batch size > 1 not yet supported for target scoring"
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
    def score_seqs(
        self, input_ids, completion_ids, use_cache: bool = True, batch_size: int = 1
    ):
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for target scoring; batch dim must be present"
        assert (input_ids.ndim == 2, completion_ids.ndim == 3)  # b, L; b, n, L
        if use_cache:
            return self._score_seqs_kv_cache(
                input_ids, completion_ids, batch_size=batch_size
            )
        else:
            return self._score_seqs_no_cache(
                input_ids, completion_ids, batch_size=batch_size
            )

    def validation_step_likelihood_scoring(
        self, batch: Dict[str, torch.Tensor], task: str = "classification"
    ) -> torch.Tensor:
        """
        Val step for proteinGym and family classification tasks.

        Assumes that batch contains the following:
        input_ids: the prompt (i.e. MSA)
        completion_ids: the completions (i.e. mutated seqs / seqs to be classified)
        """
        if "DMS_scores" in batch:
            target = "DMS_scores"
        elif "family_labels" in batch:
            target = "family_labels"
        assert (
            batch[target].ndim == 2
            and batch["input_ids"].ndim == 2
            and batch["input_ids"].shape[0] == 1
            and batch["completion_ids"].ndim == 3
        )
        L = batch["completion_ids"].shape[-1]
        lls = self.score_seqs(
            batch["input_ids"],
            batch["completion_ids"],
            use_cache=self.use_kv_cache_for_scoring,
            batch_size=self.scoring_max_tokens // L
            if self.use_kv_cache_for_scoring
            else 1,
        )
        target_vals = batch[target][0].cpu().numpy()
        if target == "DMS_scores":
            metric, _ = spearmanr(lls, target_vals)
            metric_name = "spearman_proteinGym"
        elif target == "family_labels":
            precision, recall, thresholds = precision_recall_curve(target_vals, lls)
            metric = auc(recall, precision)
            metric_name = "auprc_fam_classification"
            au_roc = roc_auc_score(target_vals, lls)
            self.log(
                "auroc_fam_classification",
                au_roc,
                on_step=False,
                on_epoch=True,
            )
        self.log(
            metric_name,
            metric,
            on_step=False,
            on_epoch=True,
        )
        return torch.tensor(metric, device=self.device)

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        if "DMS_scores" in batch or "family_labels" in batch:
            return self.validation_step_likelihood_scoring(batch)
        else:
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
            loss = outputs.loss
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            # n.b. this might be biased for batch size > 1
            self.log(
                "val/ppl", torch.exp(loss), on_step=False, on_epoch=True, prog_bar=False
            )
            return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        if "DMS_scores" in batch or "family_labels" in batch:
            metric = self.validation_step_likelihood_scoring(batch)
            return metric
        else:
            outputs = self(
                batch["input_ids"], batch["attention_mask"], labels=batch["input_ids"]
            )
            loss = outputs.loss
            self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            # n.b. this might be biased for batch size > 1
            self.log(
                "test/ppl",
                torch.exp(loss),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, weight_decay=0.01)
        return {"optimizer": optimizer}

    def log_ds_sample_counts(self, batch):
        sd_name = batch["ds_name"].text
        for ds in sd_name:
            self.dataset_sample_counts[ds] = self.dataset_sample_counts.get(ds, 0) + 1
        for k, v in self.dataset_sample_counts.items():
            self.log(f"train/{k}_times_sampled", v, on_step=True, on_epoch=False)

    def log_per_dataset_accuracy(self, batch, outputs):
        pass