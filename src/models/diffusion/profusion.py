from typing import Dict, List, Optional

import torch

from src.constants import BACKBONE_ATOMS
from src.models.base import BaseFamilyLitModule
from src.models.diffusion.diffusion_wrapper import ProFusionCoordsDiffusion
from src.models.diffusion.model_wrapper import WrappedHFProFusionModel
from src.models.utils import accuracy_from_outputs
from src.utils.tokenizers import ProFamTokenizer


class ProFusionLitModule(BaseFamilyLitModule):
    """N.B. we don't actually need to change ANYTHING in the model.
    The forward pass and the LM loss computation remain ENTIRELY valid.
    We just add an additional diffusion loss term, calculated from the
    outputs of the same forward pass.

    x0 as diffusion targets; xt as diffusion inputs (fed to inputs_embeds).

    n.b. we might want, alphafold style, to use a different network to actually
    run diffusion, conditioned on some input embeddings (which would then need
    to exclude any noisy coordinates), but include any fixed coordinates.

    Embed noisy coordinates as well as fixed coordinates.
    For slightly improved efficiency we could include fixed coordinates in the 'prompt' -
    but this requires handling arbitrary orders.
    """

    def __init__(
        self,
        model: WrappedHFProFusionModel,
        diffusion: ProFusionCoordsDiffusion,
        tokenizer: ProFamTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        diffusion_loss_weight: float = 1.0,
        atom_names: List[str] = ["N", "CA", "C", "O"],  # which backbone atoms to model
        bidirectional_within_sequence_attention: bool = False,
        use_explicit_causal_mask: bool = False,
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
        self.diffusion = diffusion
        self.diffusion_loss_weight = diffusion_loss_weight
        # self.diffusion_head = nn.Linear(model.config.hidden_size, len(atom_names) * 3)
        self.scoring_max_tokens = scoring_max_tokens
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.dataset_sample_counts = {}
        self.doc_id_counts = {}
        self.use_seq_pos = self.tokenizer.use_seq_pos
        self.max_seq_pos = self.tokenizer.max_seq_pos
        self.atom_names = atom_names
        self.bidirectional_within_sequence_attention = (
            bidirectional_within_sequence_attention
        )
        self.use_explicit_causal_mask = use_explicit_causal_mask
        assert (
            not self.use_explicit_causal_mask
            and self.bidirectional_within_sequence_attention
        )

    @property
    def num_atoms(self):
        return len(self.atom_names)

    def get_coords(self, all_coords):
        return all_coords[..., [BACKBONE_ATOMS.index(a) for a in self.atom_names], :]

    def build_coords(self, coords):
        # fills in missing atoms with nans
        assert coords.ndim == 4
        all_coords = torch.full(
            (coords.shape[0], coords.shape[1], len(BACKBONE_ATOMS), 3),
        )
        for i, atom_name in enumerate(self.atom_names):
            idx = BACKBONE_ATOMS.index(atom_name)
            all_coords[:, :, idx, :] = coords[:, :, i, :]
        return all_coords

    def make_causal_attention_mask(self, input_ids):
        # for checking that custom 4d masks are working - compare to not passing explicit mask at train / test / generation time
        min_dtype = torch.finfo(
            torch.float32
        ).min  # TODO: possibly infer dtype rather than hardcoding
        bsz, L = input_ids.shape
        causal_mask = torch.full(
            (L, L), min_dtype, dtype=torch.float32, device=self.device
        )
        # TODO: one possible issue with custom mask is the strange hack they have for fully masked rows...
        # TODO: find link for this
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None].expand(
            bsz, -1, -1, -1
        )  # b, h, L, L
        return causal_mask

    def make_sequence_bidirectional_attention_mask(self, input_ids):
        """N.B. explicit masks are not currently compatible with flash attention."""
        min_dtype = torch.finfo(
            torch.float32
        ).min  # TODO: possibly infer dtype rather than hardcoding
        bsz, L = input_ids.shape
        causal_mask = torch.full(
            (L, L), min_dtype, dtype=torch.float32, device=self.device
        )
        # TODO: one possible issue with custom mask is the strange hack they have for fully masked rows...
        # TODO: find link for this
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None].expand(
            bsz, -1, -1, -1
        )  # b, h, L, L
        assert not (
            input_ids == self.tokenizer.seq_struct_sep_token_id
        ).any()  # not handled for now
        # we allow attention to all positions in the current sequence.
        # we dont attent to the next sep token, which marks the start of the next sequence
        # so we might want to think carefully about this
        sequence_index = torch.cumsum(
            (input_ids == self.tokenizer.sep_token_id).float(), dim=-1
        )  # b, l
        same_sequence_mask = (
            sequence_index[:, None, :] == sequence_index[:, :, None]
        )  # b, l, l
        causal_mask = causal_mask.masked_fill(
            same_sequence_mask[:, None], 0.0
        )  # allow attention within sequence, while retaining causal attention between sequences
        return causal_mask

    def get_forward_kwargs(self, batch):
        forward_kwargs = (
            {"seq_pos": batch.get("seq_pos", None)} if self.use_seq_pos else {}
        )
        if self.bidirectional_within_sequence_attention:
            forward_kwargs[
                "attention_mask"
            ] = self.make_sequence_bidirectional_attention_mask(batch["input_ids"])
        elif self.use_explicit_causal_mask:
            forward_kwargs["attention_mask"] = self.make_causal_attention_mask(
                batch["input_ids"]
            )
        if "xt" in batch:
            # training time
            forward_kwargs["coords"] = batch["xt"]
            forward_kwargs["timestep"] = batch["timestep"]
        else:
            forward_kwargs["coords"] = batch["coords"]
            forward_kwargs["timestep"] = torch.zeros(
                (batch["input_ids"].shape[0], batch["input_ids"].shape[1]),
                device=self.device,
            ).long()
        return forward_kwargs

    def _sample_coords(
        self,
        input_ids,
        num_samples,
        length: int,  # to generate
        batch_size: int = 1,
        input_seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
        use_cache: bool = True,
        input_coords: Optional[torch.Tensor] = None,
    ):
        assert input_ids.shape[0] == 1 and input_ids.ndim == 2
        if input_seq_pos is not None:
            assert input_seq_pos.shape == input_ids.shape
        if input_coords is not None:
            raise NotImplementedError(
                "input_coords not yet supported"
            )  # would need to pass to forward_kwargs

        assert (input_ids[:, -1] == self.tokenizer.sep_token_id).all()
        forward_kwargs = self.get_forward_kwargs(batch={"seq_pos": input_seq_pos})
        forward_kwargs["input_ids"] = input_ids
        sampled_coords = self.diffusion.sample_coords(
            model=self.model,
            tokenizer=self.tokenizer,
            num_samples=num_samples,
            length=length,
            batch_size=batch_size,
            forward_kwargs=forward_kwargs,
            completion_seq_pos=completion_seq_pos,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            token_id_for_completion=token_id_for_completion,
            use_cache=use_cache,
        )
        return self.build_coords(sampled_coords)

    # for sequence scoring
    def get_forward_kwargs_for_kv_cache(
        self,
        completion_seq_pos: Optional[torch.LongTensor],
        coords: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # if coords and timestep are not provided, we can pass defaults
        forward_kwargs = super().get_forward_kwargs_for_kv_cache(
            completion_seq_pos, **kwargs
        )
        bsz, L = completion_seq_pos.shape
        if coords is None:
            forward_kwargs["coords"] = torch.zeros(
                bsz, L, self.num_atoms, 3, device=self.device
            )
        if timestep is None:
            forward_kwargs["timestep"] = torch.zeros(bsz, L, device=self.device).long()
        return forward_kwargs

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
        assert (
            input_ids.ndim == 2 and completion_ids.ndim == 3
        ), f"input ids shape {input_ids.shape}, completion ids shape {completion_ids.shape}"  # b, L; b, n, L
        if use_cache:
            return self._score_seqs_kv_cache(
                input_ids,
                completion_ids,
                batch_size=batch_size,
                seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
            )
        else:
            raise NotImplementedError("only kv cache version implemented for profusion")

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """If we just use fixed coords at positions for which we dont care about diffusion,
        that should prevent those positions doing much to the diffusion loss.

        We could explicitly mask such positions in the diffusion loss: this
        would probably be a sensible idea: we need a coordinates mask anyway.

        AF3:
        We apply a weighted aligned MSE loss to the denoised structure output from the Diffusion
        Module. We first perform a rigid alignment of the ground truth to the denoised structure

        then compute a weighted mse.

        they also compute a bond loss, which is mse on bond lengths, and an lddt loss.
        """
        self.diffusion.prepare_batch(batch)
        forward_kwargs = self.get_forward_kwargs(batch, is_train=True)
        # TODO: write a wrapper to compute loss / metrics if we have 3di tokens?
        # one option would be to write our own versions of classes llike llamaforcausallm

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
            **forward_kwargs,
        )
        diffusion_loss = self.diffusion.compute_loss(outputs, batch)
        loss = outputs.loss + self.diffusion_loss_weight * diffusion_loss
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

            # TODO: verify that on_epoch skips missing batches
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
                    on_step=False,
                    on_epoch=True,
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
