from typing import List, Optional

import torch
from torch import nn

from src.utils.utils import nested_getattr


class WrappedHFModelWithPositionEmbeddingsMixin:
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)

    args:
        embed_sequence_index: if True, embed index of sequence within sequence of sequences (TODO: rename)
        pass_constant_position_ids_for_global_index: if True, pass constant position ids to model (for e.g. inbuilt ROPE embeddings)
        pass_sequence_position_ids_for_global_index: if True, pass sequence position ids to model
    """

    # This is a mixin for models that require seq pos input during generation
    # using the mixin allows the use of standard generation code
    def __init__(
        self,
        config,
        token_embedder: str,
        embedding_dim: int,
        sep_token_id: int,
        use_seq_pos: bool = False,
        max_seq_pos: int = 2048,
        require_seq_pos: bool = True,
        embed_coords: bool = False,
        embed_sequence_index: bool = False,
        pass_constant_position_ids_for_global_index: bool = False,
        pass_sequence_position_ids_for_global_index: bool = False,
        max_sequence_index: int = 1024,
    ):
        super().__init__(config)
        self.use_seq_pos = use_seq_pos
        # TODO: avoid re-tracking - does this happen automatically?
        self.token_embedder = nested_getattr(
            self, token_embedder
        )  # TODO: use self.embed_tokens or sthg
        self.require_seq_pos = require_seq_pos
        self.max_seq_pos = max_seq_pos
        self.embed_coords = embed_coords
        self.num_atoms = 4
        self.embed_sequence_index = embed_sequence_index
        self.max_sequence_index = max_sequence_index
        self.sep_token_id = sep_token_id
        self.pass_constant_position_ids_for_global_index = (
            pass_constant_position_ids_for_global_index
        )
        self.pass_sequence_position_ids_for_global_index = (
            pass_sequence_position_ids_for_global_index
        )
        if self.embed_coords:
            self.coords_embedding = nn.Linear(
                self.num_atoms * 3, embedding_dim, bias=False
            )
        if self.use_seq_pos:
            self.seq_pos_embedding = nn.Embedding(self.max_seq_pos, embedding_dim)
        if self.embed_sequence_index:
            self.sequence_index_embedding = nn.Embedding(
                self.max_sequence_index, embedding_dim
            )

    # This needs to be the instantiation target if using seq pos... or wrapped hf model needs to handle properly
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # we're going to assume that the prompt ends with a separator token
        assert "seq_pos" in kwargs
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if input_ids.shape[-1] != kwargs["seq_pos"].shape[-1]:
            # we have incremented input ids but not seq pos
            increment = input_ids.shape[-1] - kwargs["seq_pos"].shape[-1]
            assert (
                inputs["input_ids"].shape[-1] == 1
            )  # we have sliced out the last token
            # https://github.com/huggingface/transformers/blob/cf32ee1753c9747b877113a309c2aa989f6d006c/src/transformers/models/llama/modeling_llama.py#L1236
            # just automatically increment the seq pos: this corresponds to never generating insertions in case of msas.
            # we need to slice out all previously considered sequence positions - at what point does this occur for input ids?
            # we need to be very careful about caching - inputs embeds is hopefully not being passed to the outer model? but if it is we
            # need to handle that. and we also need to ensure that slicing of seq pos is consistent with slicing of input ids.
            # frustratingly slicing of input ids is done on the super prepare inputs for generation
            prev_seq_pos = kwargs["seq_pos"][:, -1:]
            seq_pos = prev_seq_pos + increment
            inputs["seq_pos"] = seq_pos
            assert kwargs["coords"].ndim == 4  # b, l, n, 3
            bsz = prev_seq_pos.shape[0]
            if self.embed_coords:
                inputs["coords"] = torch.full(
                    (
                        bsz,
                        1,
                    )
                    + kwargs["coords"].shape[-2:],
                    0.0,
                ).to(kwargs["coords"])
        else:
            inputs["seq_pos"] = kwargs["seq_pos"]
            if self.embed_coords:
                inputs["coords"] = kwargs["coords"]
        return inputs

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
    ):
        # n.b. we need to be careful about what happens when caching.
        # I think in that case input_ids should just be the continuation
        # and inputs_embeds should also.
        # different models will have different token embedders.
        # we assume (which is case for e.g. gpt2 and mistral)
        # that the model will itself add its own position embeddings to inputs_embeds
        assert input_ids.ndim == 2
        # gpt2 code
        # elif input_ids is not None:
        #     self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        #     input_shape = input_ids.size()
        #     input_ids = input_ids.view(-1, input_shape[-1])
        #     batch_size = input_ids.shape[0]

        # in this case model's position ids will be inferred from inputs_embeds
        inputs_embeds = self.token_embedder(input_ids)
        if self.use_seq_pos:
            if self.require_seq_pos:
                assert seq_pos is not None
            if seq_pos is not None:
                pos_embeds = self.seq_pos_embedding(seq_pos)
                inputs_embeds = inputs_embeds + pos_embeds

        # TODO: might want to embed coords mask to allow for masked coords
        if self.embed_coords:
            coords_embeds = self.coords_embedding(coords)
            inputs_embeds += coords_embeds

        if self.embed_sequence_index:
            sequence_index = torch.cumsum(
                (input_ids == self.sep_token_id).float(), dim=-1
            )
            inputs_embeds += self.sequence_index_embedding(sequence_index)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        seq_pos: Optional[torch.LongTensor] = None,  # added this line for PFLM
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        coords: Optional[torch.FloatTensor] = None,
        **kwargs,  # e.g. labels
    ):
        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"
        inputs_embeds = self.embed_inputs(input_ids, seq_pos=seq_pos, coords=coords)
        # TODO: test these; make sure they get called during generation for example.
        if self.pass_constant_position_ids_for_global_index:
            assert position_ids is None
            position_ids = torch.full_like(input_ids, 10).long()
        if self.pass_sequence_position_ids_for_global_index:
            assert position_ids is None
            assert seq_pos is not None
            position_ids = seq_pos
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
