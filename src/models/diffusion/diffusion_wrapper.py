from typing import Dict, Optional

import torch
from torch import nn

from src.models.diffusion.gaussian_diffusion import GaussianDiffusion
from src.models.diffusion.resample import UniformSampler
from src.models.diffusion.superimposition import rigid_align
from src.models.utils import UpdatedDynamicCache


class ProFusionCoordsDiffusion(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_atoms,
        device,
        diffusion: GaussianDiffusion,
        diffusion_loss_prob: float = 1.0,
    ):
        self.diffusion_loss_prob = diffusion_loss_prob
        self.diffusion = diffusion
        self.diffusion_head = nn.Linear(emb_dim, num_atoms * 3)
        self.schedule_sampler = UniformSampler(diffusion)
        self.device = device

    def prepare_batch(self, batch):
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

        batch["xt"] = xt
        batch["timestep"] = timestep
        batch["noise"] = noise
        batch["coords_mask"] = coords_mask  # updated
        batch["coin_flip"] = coin_flip

    def compute_loss(self, outputs, batch):
        """Run a model forward pass, and compute the diffusion loss. Return model outputs and diffusion loss."""
        emb = outputs.hidden_states[-1]  # hidden states is a tuple
        noise_pred = self.diffusion_head(emb)
        # ah - in AF the loss is not on the epsilon but on the denoised structure
        # diffusion_loss = (
        #     nn.MSELoss(reduction="none")(noise_pred, noise) * coords_mask.float()
        # ).sum() / coords_mask.sum()

        # TODO: check all inputs - is scale timestep correct for example? what shape should noise_pred be?
        # t = batch["timestep"][:, 0]
        x0_pred = self.diffusion._predict_xstart_from_eps(batch["xt"], t, noise_pred)
        # TODO: these need to be flattened - although actually I think rigid_align can handle this
        x0_pred_gt_aligned = rigid_align(batch["coords"], x0_pred)
        if batch["coin_flip"] < self.diffusion_loss_prob:
            diffusion_loss = (
                nn.MSELoss(reduction="none")(x0_pred_gt_aligned, batch["coords"])
                * batch["coords_mask"].float()
            ).sum() / batch["coords_mask"].sum()
        else:
            diffusion_loss = torch.tensor(0.0, device=self.device)
        return diffusion_loss

    def _sample_coords_kv_cache(
        self,
        model,
        tokenizer,
        num_samples,
        length: int,  # to generate
        batch_size: int = 1,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
        forward_kwargs: Optional[Dict] = None,
    ):
        """N.B. whereas autoregressive sampling extends the sequence by a variable amount,
        Profusion sampling requires a pre-specified length.

        Sep token should be included in inputs.
        """

        all_outputs = []
        # TODO: handle this
        forward_kwargs = forward_kwargs or {}
        forward_kwargs["use_cache"] = True
        outputs = model(**forward_kwargs)
        past_key_values = (
            outputs.past_key_values
        )  # just a tuple of tensors - doesn't get extended
        token_id_for_completion = token_id_for_completion or tokenizer.mask_token_id
        completions = tokenizer.encode_completions(
            ["[MASK]" * length], bos_token="", eos_token=""
        )
        if completion_seq_pos is None:
            completion_seq_pos = completions["seq_pos"]
            completion_ids = completions["input_ids"]
            assert completion_ids.shape == completion_seq_pos.shape
            assert completion_ids.shape[-1] == length
            assert (completion_ids == tokenizer.mask_token_id).all()
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
                outputs = model(
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
        return torch.cat(all_outputs, dim=0)

    def _sample_coords_no_cache(
        self,
        model,
        tokenizer,
        num_samples,
        length: int,
        batch_size: int = 1,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
        forward_kwargs: Optional[Dict] = None,
    ):
        all_outputs = []
        token_id_for_completion = (
            token_id_for_completion or self.tokenizer.mask_token_id
        )
        raise NotImplementedError("Need to handle concatenation of forward kwargs")
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
        return torch.cat(all_outputs, dim=0)

    def _sample_coords(
        self,
        model,
        tokenizer,
        num_samples,
        length: int,  # to generate
        batch_size: int = 1,
        completion_seq_pos: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        denoised_fn=None,
        cond_fn=None,
        token_id_for_completion=None,
        use_cache: bool = True,
        forward_kwargs: Optional[Dict] = None,
    ):
        """N.B. whereas autoregressive sampling extends the sequence by a variable amount,
        Profusion sampling requires a pre-specified length.
        """
        if use_cache:
            return self._sample_coords_kv_cache(
                model=model,
                tokenizer=tokenizer,
                num_samples=num_samples,
                length=length,
                batch_size=batch_size,
                completion_seq_pos=completion_seq_pos,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                token_id_for_completion=token_id_for_completion,
                forward_kwargs=forward_kwargs,
            )
        else:
            return self._sample_coords_no_cache(
                model=model,
                tokenizer=tokenizer,
                num_samples=num_samples,
                length=length,
                batch_size=batch_size,
                completion_seq_pos=completion_seq_pos,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                token_id_for_completion=token_id_for_completion,
                forward_kwargs=forward_kwargs,
            )
