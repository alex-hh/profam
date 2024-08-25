from typing import List, Optional

import torch
from torch import nn

from src.models.wrapper import WrappedHFModelWithPositionEmbeddingsMixin


class WrappedHFProFusionModel(WrappedHFModelWithPositionEmbeddingsMixin):
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)
    N.B. timestep conditioning in diffusion transformer is performed via adaptive layer norm ...

    this is also used in alphafold 3
    'we initialise the activations from the single embedding, use a variant of Adaptive Layernorm'
    '[27] for the single conditioning and logit biasing for the pair conditioning.'

    Transfusion:
    'We add an embedding of the timestep t to every patch vector before the linear layer'
    """

    # This is a mixin for models that require seq pos input during generation
    # using the mixin allows the use of standard generation code
    def __init__(
        self,
        config,
        token_embedder: str,
        embedding_dim: int,
        use_seq_pos: bool = False,
        max_seq_pos: int = 2048,
        require_seq_pos: bool = True,
        num_atoms: int = 1,
        num_timesteps: int = 1000,
    ):
        super().__init__(
            config,
            token_embedder=token_embedder,
            embedding_dim=embedding_dim,
            use_seq_pos=use_seq_pos,
            max_seq_pos=max_seq_pos,
            require_seq_pos=require_seq_pos,
        )
        self.coords_embedding = nn.Linear(num_atoms * 3, embedding_dim, bias=False)
        self.timestep_embedding = nn.Embedding(num_timesteps, embedding_dim)

    # This needs to be the instantiation target if using seq pos... or wrapped hf model needs to handle properly
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # to be used during sequence generation
        # need to extend coords, timestep by an additional zero coords, timestep...
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if input_ids.shape[-1] != kwargs["timestep"].shape[-1]:
            # we have incremented input ids but not timestep
            assert inputs["input_ids"].shape[-1] == 1
            # just provide a single new value
            assert kwargs["timestep"].ndim == 2  # b, l
            bsz = kwargs["timestep"].shape[0]
            inputs["timestep"] = torch.full(
                (
                    bsz,
                    1,
                ),
                0,
            ).to(kwargs["timestep"])
            assert kwargs["coords"].ndim == 4  # b, l, n, 3
            inputs["coords"] = torch.full(
                (
                    bsz,
                    1,
                )
                + kwargs["coords"].shape[-2:],
                0,
            ).to(kwargs["coords"])
        else:
            inputs["coords"] = kwargs["coords"]
            inputs["timestep"] = kwargs["timestep"]

        return inputs

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = super().embed_inputs(
            input_ids, seq_pos=seq_pos
        )  # position and token embeddings
        if coords is not None:
            coords_embeds = self.coords_embedding(coords)
            inputs_embeds += coords_embeds
        if timestep is not None:
            timestep_embeds = self.timestep_embedding(timestep)
            inputs_embeds += timestep_embeds

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_pos: Optional[torch.LongTensor] = None,  # added this line for PFLM
        coords: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # q. what is position_ids? oh its for position embedding
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # e.g. labels
    ):
        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"
        inputs_embeds = self.embed_inputs(
            input_ids, seq_pos=seq_pos, coords=coords, timestep=timestep
        )
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
