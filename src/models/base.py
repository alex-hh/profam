import os
import time
from typing import Any, Dict, Optional

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
from transformers.cache_utils import DynamicCache
from transformers.optimization import get_scheduler

from src.constants import BASEDIR, aa_letters
from src.data.objects import StringObject
from src.models import metrics
from src.models.utils import log_likelihood_from_outputs
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
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    print(OmegaConf.to_yaml(cfg.model))
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
        scoring_max_tokens: int = 10240,
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

    @torch.no_grad()
    def log_metrics(self, batch, outputs, step_name, log_global: bool = True):
        # N.B. actually val logging is a bit different because of this ds name thing
        loss = outputs.loss

        dataset_accuracies = metrics.accuracy_from_outputs(
            outputs,
            batch["labels"],
            ignore_index=-100,
            dataset_names=batch[
                "ds_name"
            ].text,  # a list of dataset names (StringObject.text)
            ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                ["-", "X", "x"]
                + [aa.lower() for aa in aa_letters]
                + self.tokenizer.all_special_tokens
            ),
            mask=None,
        )
        has_3di = torch.isin(
            batch["input_ids"],
            torch.tensor(
                self.tokenizer.convert_tokens_to_ids([aa.lower() for aa in aa_letters])
            ).to(batch["input_ids"]),
        ).any()

        global_metrics = {
            "loss": loss,
            "ppl": torch.exp(loss),
            "aa_accuracy": dataset_accuracies.pop("global"),
        }
        if "coords" in batch:
            global_metrics["has_coords_frac"] = metrics.has_coords_frac(**batch)
            if "plddts" in batch:
                global_metrics.update(metrics.plddt_metrics(**batch))
            is_interleaved = (
                batch["input_ids"] == self.tokenizer.seq_struct_sep_token_id
            ).any()
            if is_interleaved:
                aa_has_coords_mask = batch["interleaved_coords_mask"].any((-1, -2))
                has_coords_dataset_accuracies = metrics.accuracy_from_outputs(
                    outputs,
                    batch["labels"],
                    ignore_index=-100,
                    dataset_names=batch[
                        "ds_name"
                    ].text,  # a list of dataset names (StringObject.text)
                    ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                        ["-", "X", "x"]
                        + [aa.lower() for aa in aa_letters]
                        + self.tokenizer.all_special_tokens
                    ),
                    mask=(aa_has_coords_mask & batch["aa_mask"]),
                )
                global_metrics[
                    "has_coords_aa_accuracy"
                ] = has_coords_dataset_accuracies.pop("global")
                global_metrics["aa_has_coords_frac"] = (
                    aa_has_coords_mask & batch["aa_mask"]
                ).float().sum() / batch["aa_mask"].float().sum()
            global_metrics["aa_count"] = batch["aa_mask"].float().sum()

        if has_3di:
            dataset_accuracies_3di = metrics.accuracy_from_outputs(
                outputs,
                batch["labels"],
                ignore_index=-100,
                dataset_names=batch["ds_name"].text,
                ignore_token_ids=self.tokenizer.convert_tokens_to_ids(
                    ["-", "X", "x"] + aa_letters + self.tokenizer.all_special_tokens
                ),
            )
            global_metrics["3di_accuracy"] = dataset_accuracies_3di.pop("global")

        if log_global:
            self.log_dict(
                {f"{step_name}/{k}": v for k, v in global_metrics.items()},
                on_step=step_name == "train",
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )

        # n.b. this assumes a batch only contains a single dataset - only true during val!
        # assert all([ds_name == batch["ds_name"][0] for ds_name in batch["ds_name"]])
        assert isinstance(batch["ds_name"], StringObject)
        is_single_dataset_batch = len(set(batch["ds_name"].text)) == 1
        for ds_name in set(batch["ds_name"].text):
            ds_metrics = {
                f"{step_name}/{ds_name}/aa_accuracy": dataset_accuracies[ds_name]
            }
            # TODO: coords frac for each dataset
            if is_single_dataset_batch:
                # global metrics are dataset specific
                ds_metrics[f"{step_name}/{ds_name}/loss"] = loss
            if has_3di:
                ds_metrics[
                    f"{step_name}/{ds_name}/3di_accuracy"
                ] = dataset_accuracies_3di[ds_name]
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
        assert (
            input_ids.ndim == 2 and completion_ids.ndim == 3
        ), f"input ids shape {input_ids.shape}, completion ids shape {completion_ids.shape}"  # b, L; b, n, L
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
        embed_coords: bool = False,
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
        self.doc_id_counts = {}
        self.use_seq_pos = self.tokenizer.use_seq_pos
        self.max_seq_pos = self.tokenizer.max_seq_pos
        self.embed_coords = embed_coords
        if self.use_seq_pos:
            self.embed_sequence_index = self.model.embed_sequence_index
        else:
            self.embed_sequence_index = False

    def get_forward_kwargs(self, batch):
        forward_kwargs = {}
        if self.embed_coords:
            assert batch["coords"] is not None
            forward_kwargs["coords"] = batch["coords"]
        if self.use_seq_pos:
            assert batch["seq_pos"] is not None
            forward_kwargs["seq_pos"] = batch["seq_pos"]
        return forward_kwargs

    def _score_seqs_kv_cache(
        self,
        input_ids,
        completion_ids,
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        # input_ids is b, L; completion_ids is b, n, L
        # https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
        # https://github.com/huggingface/transformers/blob/b7672826cad31e30319487af876e608d8af7d37b/src/transformers/generation/utils.py#L1879
        # https://github.com/huggingface/transformers/blob/67a4ef89d4ddbfd7d61e479359a1b609e5ee9843/src/transformers/models/mistral/modeling_mistral.py#L1233
        all_lls = []
        forward_kwargs = self.get_forward_kwargs({"seq_pos": seq_pos, "coords": coords})
        outputs = self.model(input_ids=input_ids, use_cache=True, **forward_kwargs)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        L = completion_ids.shape[-1]

        if self.embed_sequence_index:
            prompt_sequence_index = self.model.compute_sequence_index(input_ids)
            assert (input_ids[:, -1] == input_ids[0, -1]).all()
            if input_ids[0, -1] == self.tokenizer.sep_token_id:
                start_sequence_index = prompt_sequence_index[:, -1] + 1
            else:
                # maybe completion ids starts with sep token, in which case sequence index
                # will automatically be incremented in model forward
                start_sequence_index = prompt_sequence_index[:, -1]

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
            if self.embed_coords:
                assert coords is not None
                raise NotImplementedError("Coords not yet supported for mutant scoring")
            if self.embed_sequence_index:
                forward_kwargs["start_sequence_index"] = start_sequence_index

            actual_batch_size = this_input_ids.shape[0]
            cache = DynamicCache.from_legacy_cache(past_key_values)
            cache.batch_repeat_interleave(actual_batch_size)  # careful: returns None!

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
        coords: Optional[torch.FloatTensor] = None,
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
            if self.embed_coords:
                assert coords is not None
                raise NotImplementedError("Coords not yet supported for mutant scoring")
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
        coords: Optional[torch.FloatTensor] = None,
        input_seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
    ):
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
                seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
            )
        else:
            return self._score_seqs_no_cache(
                input_ids,
                completion_ids,
                batch_size=batch_size,
                coords=coords,
                seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
            )

    def _sample_seqs(
        self,
        input_ids,
        num_samples,
        batch_size: int = 1,
        max_generated_length: Optional[int] = None,
        max_total_length: Optional[
            int
        ] = None,  # maximum length of inputs plus completions
        input_seq_pos: Optional[torch.LongTensor] = None,
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
            if self.use_seq_pos:
                max_total_length = min(
                    self.tokenizer.max_tokens,
                    input_ids.shape[1] + self.tokenizer.max_seq_pos,
                )
            else:
                max_total_length = self.tokenizer.max_tokens
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
        bad_aas = ["X", "x"]
        if not sample_gaps:
            bad_aas.append("-")
        if structure_tokens:
            bad_aas = bad_aas + aa_letters
        else:
            bad_aas = bad_aas + [aa.lower() for aa in aa_letters]

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

        # why is ending in sep token necessary? may not be...
        assert input_ids[:, -1].item() in [
            self.tokenizer.sep_token_id,
            self.tokenizer.seq_struct_sep_token_id,
        ]
        assert input_seq_pos.shape == input_ids.shape
        all_outputs = []
        for batch_start in range(0, num_samples, batch_size):
            num_return_sequences = min(batch_size, num_samples - batch_start)
            # TODO: understand how this gets reshaped...within prepare inputs for generation it already is expanded
            forward_kwargs = self.get_forward_kwargs(
                {"seq_pos": input_seq_pos, "coords": input_coords}
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
        start_ix = 0
        for o in all_outputs:
            padded_outputs[start_ix : start_ix + o.shape[0], : o.shape[1]] = o
            start_ix += o.shape[0]

        return padded_outputs

    # def sample_seqs(
    #     self,
    #     sequence_prompt: List[str],
    #     num_samples,
    #     position_indices: Optional[List[int]] = None,
    #     batch_size: int = 1,
    #     include_prompt_in_output: bool = False,
    #     greedy: bool = False,
    #     fixed_length: Optional[int] = None,  # makes sense especially for MSA generation
    #     temperature: Optional[float] = None,
    #     document_token: str = "[RAW]",
    # ):
    # # TODO: encode sequence prompt and get sequence pos if necessary.
    # tokenized = self.tokenizer.encode_sequences(
    #     sequence_prompt, positions=position_indices, document_token=document_token
    # )
    # if "seq_pos" in tokenized.data:
    #     seq_pos = tokenized.data["seq_pos"].unsqueeze(0).to(self.device)
    # else:
    #     seq_pos = None
    # encoded = self._sample_seqs(
    #     tokenized.input_ids.unsqueeze(0).to(self.device),
    #     num_samples,
    #     input_seq_pos=seq_pos,
    #     batch_size=batch_size,
    #     include_prompt_in_output=include_prompt_in_output,
    #     greedy=greedy,
    #     fixed_length=fixed_length,
    #     temperature=temperature,
    #     sample_gaps=document_token == "[MSA]",
    # )
    # return self.tokenizer.decode_tokens(encoded)

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
            batch_size=1,
            # batch_size=(self.scoring_max_tokens - L_prompt) // L
            # if self.use_kv_cache_for_scoring
            # else 1,
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
        self.log_ds_sample_counts(batch)
        return loss

    def log_ds_sample_counts(self, batch):
        """Log statistics about dataset usage.

        N.B. in distributed setting, these will be device-specific.
        """
        ds_name = batch["ds_name"].text
        for ds in ds_name:
            self.dataset_sample_counts[ds] = self.dataset_sample_counts.get(ds, 0) + 1

        self.log_dict(
            {
                f"train/{k}_times_sampled": v
                for k, v in self.dataset_sample_counts.items()
            },
            on_step=True,
            on_epoch=False,
        )
        if "identifier" in batch:
            for i, (dataset, doc_id) in enumerate(
                zip(batch["ds_name"].text, batch["identifier"].text)
            ):
                self.doc_id_counts[dataset] = self.doc_id_counts.get(dataset, {})
                self.doc_id_counts[dataset][doc_id] = (
                    self.doc_id_counts[dataset].get(doc_id, 0) + 1
                )
            self.log_dict(
                {
                    f"{k}_max_sampled_doc": max(v.values())
                    for k, v in self.doc_id_counts.items()
                },
                on_step=False,
                on_epoch=True,
            )
