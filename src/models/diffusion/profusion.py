from typing import Dict, List, Optional

import torch
from torch import nn

from src.constants import BACKBONE_ATOMS
from src.models.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.diffusion.resample import UniformSampler
from src.models.diffusion.wrapper import WrappedHFProFusionModel
from src.models.utils import UpdatedDynamicCache, accuracy_from_outputs
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
    """

    def __init__(
        self,
        model: WrappedHFProFusionModel,
        diffusion: GaussianDiffusion,
        tokenizer: ProFamTokenizer,
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        scheduler_name: Optional[str] = None,
        num_warmup_steps: int = 1000,
        num_training_steps: Optional[int] = None,
        scoring_max_tokens: int = 8000,
        use_kv_cache_for_scoring: bool = True,
        diffusion_loss_weight: float = 1.0,
        diffusion_loss_prob: float = 1.0,
        atom_names: List[str] = ["N", "CA", "C", "O"],  # which backbone atoms to model
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
        self.diffusion_loss_prob = diffusion_loss_prob
        self.diffusion_head = nn.Linear(model.config.hidden_size, len(atom_names) * 3)
        self.schedule_sampler = UniformSampler(diffusion)
        self.scoring_max_tokens = scoring_max_tokens
        self.use_kv_cache_for_scoring = use_kv_cache_for_scoring
        self.dataset_sample_counts = {}
        self.doc_id_counts = {}
        self.use_seq_pos = self.tokenizer.use_seq_pos
        self.max_seq_pos = self.tokenizer.max_seq_pos
        self.atom_names = atom_names

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

    def get_forward_kwargs(self, batch, is_train: bool = False):
        forward_kwargs = (
            {"seq_pos": batch.get("seq_pos", None)} if self.use_seq_pos else {}
        )
        if is_train:
            # coords and timestep are created in the training step
            return forward_kwargs
        else:
            forward_kwargs["coords"] = batch["coords"]
            forward_kwargs["timestep"] = torch.zeros(
                (batch["input_ids"].shape[0], batch["input_ids"].shape[1]),
                device=self.device,
            ).long()
            return forward_kwargs

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

    def _sample_coords_kv_cache(
        self,
        input_ids,
        num_samples,
        input_coords,
        length: int,  # to generate
        batch_size: int = 1,
        input_seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
    ):
        """N.B. whereas autoregressive sampling extends the sequence by a variable amount,
        Profusion sampling requires a pre-specified length.

        Sep token should be included in inputs.
        """
        assert input_ids.shape[0] == 1 and input_ids.ndim == 2
        if input_seq_pos is not None:
            assert input_seq_pos.shape == input_ids.shape
        assert input_coords.shape[:2] == input_ids.shape
        input_L = input_ids.shape[-1]
        assert (input_ids[:, -1] == self.tokenizer.sep_token_id).all()
        all_outputs = []
        if self.use_seq_pos:
            assert input_seq_pos is not None
        forward_kwargs = self.get_forward_kwargs(batch={"seq_pos": input_seq_pos})
        outputs = self.model(input_ids=input_ids, use_cache=True, **forward_kwargs)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        token_id_for_completion = (
            token_id_for_completion or self.tokenizer.mask_token_id
        )
        completions = self.tokenizer.encode_completions(
            ["[MASK]" * length], bos_token="", eos_token=""
        )
        if completion_seq_pos is None:
            completion_seq_pos = completions["seq_pos"]
            completion_ids = completions["input_ids"]
            assert completion_ids.shape == completion_seq_pos.shape
            assert completion_ids.shape[-1] == length
            assert (completion_ids == self.tokenizer.mask_token_id).all()
        else:
            raise NotImplementedError("completion seq pos must be None currently")

        for batch_start in range(0, num_samples, batch_size):
            num_samples_this_iter = min(batch_size, num_samples - batch_start)
            coords_shape = (
                num_samples_this_iter,
                length,
                self.num_atoms,
                3,
            )
            cache = UpdatedDynamicCache.from_legacy_cache(past_key_values)
            cache.batch_repeat_interleave(num_samples_this_iter)
            # c.f. p_sample_loop_progressive
            def model_forward_wrapper(x, t, **kwargs):
                # TODO: we need to handle the fact that the diffusion bit has
                # a different shape (it's a slice of the full model).
                # TODO: we can also exploit kv caching here: then we won't need cooncatenation
                # TODO: handle rescaling, rotation, etc.
                timestep = t.unsqueeze(-1).expand(
                    num_samples_this_iter, length
                )  # already scaled
                outputs = self.model(
                    coords=x,
                    input_ids=completion_ids,
                    timestep=timestep,
                    output_hidden_states=True,
                    seq_pos=completion_seq_pos,
                    past_key_values=cache,
                    use_cache=True,
                    **kwargs,
                )
                emb = outputs.hidden_states[-1]
                eps = self.diffusion_head(emb).view(-1, length, self.num_atoms, 3)
                return eps

            all_outputs.append(
                self.diffusion.p_sample_loop(
                    model_forward_wrapper,
                    coords_shape,
                    noise=noise,
                    device=self.device,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                )
            )
        return self.build_coords(torch.cat(all_outputs, dim=0))

    def _sample_coords_no_cache(
        self,
        input_ids,
        num_samples,
        input_coords,
        length: int,
        batch_size: int = 1,
        input_seq_pos: Optional[torch.LongTensor] = None,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
    ):
        assert input_ids.shape[0] == 1 and input_ids.ndim == 2
        if input_seq_pos is not None:
            assert input_seq_pos.shape == input_ids.shape
        assert input_coords.shape[:2] == input_ids.shape
        input_L = input_ids.shape[-1]
        all_outputs = []
        token_id_for_completion = (
            token_id_for_completion or self.tokenizer.mask_token_id
        )
        for batch_start in range(0, num_samples, batch_size):
            num_samples_this_iter = min(batch_size, num_samples - batch_start)
            coords_shape = (
                num_samples_this_iter,
                length,
                self.num_atoms,
                3,
            )
            # TODO: figure out the appropriate extension of input_seq_pos
            assert (
                completion_seq_pos is not None and completion_seq_pos.shape[1] == length
            )
            batch_seq_pos = torch.cat(
                [
                    input_seq_pos.expand(num_samples_this_iter, -1),
                    completion_seq_pos.expand(num_samples_this_iter, -1),
                ]
            )
            batch_input_coords = input_coords.expand(num_samples_this_iter, -1, -1, -1)
            batch_input_ids = input_ids.expand(num_samples_this_iter, -1)
            # model_kwargs = {"seq_pos": seq_pos} if self.use_seq_pos else {}
            # c.f. p_sample_loop_progressive
            def model_forward_wrapper(x, t, **kwargs):
                # TODO: we need to handle the fact that the diffusion bit has
                # a different shape (it's a slice of the full model).
                # TODO: we can also exploit kv caching here: then we won't need cooncatenation
                # TODO: handle rescaling, rotation, etc.
                timestep = t.unsqueeze(-1).expand(
                    num_samples_this_iter, length + input_L
                )  # already scaled
                outputs = self.model(
                    coords=torch.cat([batch_input_coords, x], dim=1),
                    input_ids=torch.cat(
                        [
                            batch_input_ids,
                            torch.full(
                                (num_samples_this_iter, length),
                                token_id_for_completion,
                                device=self.device,
                            ).long(),
                        ],
                        dim=1,
                    ),
                    timestep=timestep,
                    output_hidden_states=True,
                    seq_pos=batch_seq_pos,
                    **kwargs,
                )
                emb = outputs.hidden_states[-1]
                eps = self.diffusion_head(emb)[:, input_L:]
                return eps

            all_outputs.append(
                self.diffusion.p_sample_loop(
                    model_forward_wrapper,
                    coords_shape,
                    noise=noise,
                    device=self.device,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                )
            )
        return self.build_coords(torch.cat(all_outputs, dim=0))

    def _sample_coords(
        self,
        input_ids,
        num_samples,
        input_coords,
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
    ):
        """N.B. whereas autoregressive sampling extends the sequence by a variable amount,
        Profusion sampling requires a pre-specified length.
        """
        if use_cache:
            return self._sample_coords_kv_cache(
                input_ids,
                num_samples,
                input_coords,
                length,
                batch_size=batch_size,
                input_seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                token_id_for_completion=token_id_for_completion,
            )
        else:
            return self._sample_coords_no_cache(
                input_ids,
                num_samples,
                input_coords,
                length,
                batch_size=batch_size,
                input_seq_pos=input_seq_pos,
                completion_seq_pos=completion_seq_pos,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                token_id_for_completion=token_id_for_completion,
            )

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
        forward_kwargs = self.get_forward_kwargs(batch, is_train=True)
        # TODO: write a wrapper to compute loss / metrics if we have 3di tokens?
        # one option would be to write our own versions of classes llike llamaforcausallm
        coin_flip = torch.rand(1).item()
        bsz, L = batch["input_ids"].shape
        coords = batch["coords"]
        coords_mask = batch["coords_mask"]
        if coin_flip < self.diffusion_loss_prob:
            noise = torch.zeros_like(batch["x0"])
            timestep = torch.zeros((bsz, L), device=self.device).long()
            xt = coords
            coords_mask = torch.zeros_like(coords_mask)  # just affects loss
        else:
            noise = torch.randn_like(batch["x0"])
            # n.b. we can ignore weights since equal to 1
            t, _ = self.schedule_sampler.sample(
                bsz, self.device
            )  # weights is second retval, 1s for now
            assert t.shape == (bsz,)
            timestep = self._scale_timesteps(t).unsqueeze(-1).expand(bsz, L)
            xt = self.diffusion.q_sample(coords, t, noise=noise)
            xt = torch.where(coords_mask, xt, coords)

        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
            timestep=timestep,
            coords=xt,
            **forward_kwargs,
        )
        emb = outputs.hidden_states[-1]  # hidden states is a tuple
        noise_pred = self.diffusion_head(emb)
        diffusion_loss = (
            nn.MSELoss(reduction="none")(noise_pred, noise) * coords_mask.float()
        ).sum() / coords_mask.sum()
        loss = outputs.loss
        if coin_flip < self.diffusion_loss_prob:
            loss = loss + self.diffusion_loss_weight * diffusion_loss
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
