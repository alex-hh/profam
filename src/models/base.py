import os
import time
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import torch
import tqdm
from lightning import LightningModule
from omegaconf import OmegaConf
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from transformers import PreTrainedTokenizerFast
from transformers.optimization import get_scheduler

from src.constants import BASEDIR
from src.models.utils import (
    UpdatedDynamicCache,
    accuracy_from_outputs,
    log_likelihood_from_outputs,
)
from src.utils.tokenizers import ProFamTokenizer


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
    if "tokenizer" in cfg:
        old_config = False
        tokenizer = hydra.utils.instantiate(cfg.tokenizer)
    else:
        # old config
        old_config = True
        from src.utils.tokenizers import ProFamTokenizer

        tokenizer = ProFamTokenizer(
            tokenizer_file=cfg.data.tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            bos_token="[start-of-document]",
            add_special_tokens=True,
            add_final_sep=True,
            add_bos_token=True,
            add_document_type_token=True,
            use_seq_pos=cfg.data.use_seq_pos,
            max_seq_pos=cfg.data.max_seq_pos,
            max_tokens=cfg.data.max_tokens,
        )
        del cfg.model.use_seq_pos
        del cfg.model.max_seq_pos

    print(OmegaConf.to_yaml(cfg.model))
    # TODO: check callback config
    checkpoint_path = os.path.join(BASEDIR, checkpoint_dir, "checkpoints/last.ckpt")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )["state_dict"]
    if old_config and tokenizer.use_seq_pos:
        # TODO: we'll have to convert keys and change model class if using an old-style checkpoint.
        checkpoint = {
            k.replace("model.model.", "model."): v for k, v in checkpoint.items()
        }

    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)
    model.load_state_dict(checkpoint)
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
        accuracy = accuracy_from_outputs(
            outputs,
            batch["labels"],
            ignore_index=-100,
            ignore_token_ids=[self.tokenizer.convert_tokens_to_ids("-")],
        )
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
        ds_name = (
            batch["ds_name"][0]
            if isinstance(batch["ds_name"], list)
            else batch["ds_name"].text[0]
        )
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
        accuracy = accuracy_from_outputs(
            outputs,
            batch["labels"],
            ignore_index=-100,
            ignore_token_ids=[self.tokenizer.convert_tokens_to_ids("-")],
        )
        self.log(
            f"val/{ds_name}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        if dataloader_idx == 0:
            # log the loss again with generic name for the sake of model checkpointing
            self.log(
                f"val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                add_dataloader_idx=True,
            )

        self.log(
            f"val/{ds_name}/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        # n.b. this might be biased for batch size > 1
        self.log(
            f"val/{ds_name}/ppl",
            torch.exp(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        return loss

    def test_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        ds_name = batch["ds_name"].text[0]
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
        accuracy = accuracy_from_outputs(
            outputs,
            batch["labels"],
            ignore_index=-100,
            ignore_token_ids=[self.tokenizer.convert_tokens_to_ids("-")],
        )
        self.log(
            f"test/{ds_name}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        # n.b. this might be biased for batch size > 1
        self.log(
            f"test/{ds_name}/ppl",
            torch.exp(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            add_dataloader_idx=False,
        )
        self.log(
            f"test/{ds_name}/accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
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
            "gym/spearman",
            spearman_corr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )


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
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
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
        self.use_seq_pos = self.tokenizer.use_seq_pos
        self.max_seq_pos = self.tokenizer.max_seq_pos

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
            cache.batch_repeat_interleave(actual_batch_size)

            outputs = self.model(
                input_ids=this_input_ids,
                past_key_values=cache,
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

    def _sample_seqs(
        self,
        input_ids,
        num_samples,
        batch_size: int = 1,
        max_length: int = 8192,  # maximum length of inputs plus completions
        input_seq_pos: Optional[torch.LongTensor] = None,
        include_prompt_in_output: bool = False,
        fixed_length: Optional[int] = None,
        greedy: bool = False,
        temperature: Optional[float] = None,
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
        print("Sampling seqs batch size", batch_size)
        generation_kwargs = {}
        if fixed_length is not None:
            if max_length is not None:
                assert input_ids.shape[1] + fixed_length <= max_length
            generation_kwargs["min_new_tokens"] = fixed_length
            generation_kwargs["max_new_tokens"] = fixed_length
            generation_kwargs["eos_token_id"] = None
        else:
            generation_kwargs["eos_token_id"] = self.tokenizer.sep_token_id
            generation_kwargs["max_length"] = max_length
        generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        assert (
            input_ids.shape[0] == 1
        ), "Only batch size 1 is supported for mutant scoring; batch dim must be present"

        assert input_ids.ndim == 2  # b, L
        assert (input_ids[:, -1] == self.tokenizer.sep_token_id).all()
        all_outputs = []
        for batch_start in range(0, num_samples, batch_size):
            num_return_sequences = min(batch_size, num_samples - batch_start)
            forward_kwargs = (
                {"seq_pos": input_seq_pos.expand(num_return_sequences, -1)}
                if self.use_seq_pos
                else {}
            )
            # TemperatureLogitsWarper
            # TODO: migrate to model.sample
            # N.B. we need to be careful about generationconfig -- in particular eos token id
            # if we want to generate multiple sequences in a single family: we either need to restore eos token id
            # or we just do a batched generation like we do here. latter is more explicit.
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
        for i, o in enumerate(all_outputs):
            padded_outputs[i, : o.shape[1]] = o
        return padded_outputs

    def sample_seqs(
        self,
        sequence_prompt: List[str],
        num_samples,
        position_indices: Optional[List[int]] = None,
        batch_size: int = 1,
        include_prompt_in_output: bool = False,
        greedy: bool = False,
        fixed_length: Optional[int] = None,  # makes sense especially for MSA generation
        temperature: Optional[float] = None,
    ):
        # TODO: encode sequence prompt and get sequence pos if necessary.
        tokenized = self.tokenizer.encode_sequences(
            sequence_prompt, positions=position_indices
        )
        if "seq_pos" in tokenized.data:
            seq_pos = tokenized.data["seq_pos"].unsqueeze(0)
        else:
            seq_pos = None
        encoded = self._sample_seqs(
            tokenized.input_ids.unsqueeze(0),
            num_samples,
            input_seq_pos=seq_pos,
            batch_size=batch_size,
            include_prompt_in_output=include_prompt_in_output,
            greedy=greedy,
            fixed_length=fixed_length,
            temperature=temperature,
        )
        # print("samples shape", encoded.shape)
        return self.tokenizer.decode_tokens(encoded)

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
            "val/auprc_fam_classification",
            metric,
            on_step=False,
            on_epoch=True,
        )
        au_roc = roc_auc_score(target_vals, lls)
        self.log(
            "val/auroc_fam_classification",
            au_roc,
            on_step=False,
            on_epoch=True,
        )
        k_vals = [k for k in [1, 2, 5, 10] if k < len(target_vals)]
        for top_k in k_vals:
            top_k_acc = len(
                set(np.argsort(lls)[::-1][:top_k]).intersection(
                    set(np.where(target_vals)[0])
                )
            ) / min(top_k, sum(target_vals))
            self.log(
                f"val/top_{top_k}_acc_fam_classification",
                top_k_acc,
                on_step=False,
                on_epoch=True,
            )
        return torch.tensor(metric, device=self.device, dtype=torch.float32)

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
        accuracy = accuracy_from_outputs(
            outputs,
            batch["labels"],
            ignore_index=-100,
            ignore_token_ids=[self.tokenizer.convert_tokens_to_ids("-")],
        )
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
            if "ds_name" in batch:
                per_dataset_accuracies = accuracy_from_outputs(
                    outputs,
                    batch["input_ids"],
                    dataset_names=batch["ds_name"].text,
                    ignore_token_ids=[self.tokenizer.convert_tokens_to_ids("-")],
                )
                self.log_dict(
                    {
                        f"train/{k}_acc": v.item()
                        for k, v in per_dataset_accuracies.items()
                    },
                    on_step=True,
                    on_epoch=False,
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
                    on_step=True,
                    on_epoch=False,
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
