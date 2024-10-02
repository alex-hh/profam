from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from transformers.utils import ModelOutput

from src.utils.tokenizers import ProFamTokenizer
from src.utils.utils import nested_getattr


def assert_only_padding_after_eos(input_ids, eos_token_id, padding_token_id):
    # as long as we pad after sep, it doesn't matter what seq_pos is associated with sep
    sep_counts = (input_ids == eos_token_id).cumsum(dim=-1)
    assert sep_counts.max() <= 1
    should_pad = sep_counts.cumsum(-1) > 1
    assert (input_ids[should_pad] == padding_token_id).all()


# TODO: try to modularise...
class WrappedHFModelWithPositionEmbeddingsMixin:
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    have position_ids argument in .forward() method
    use modeling_attn_mask_utils.py::_prepare_4d_attention_mask() function for 4d mask generation


    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)

    args:
        use_seq_pos: embed position of amino acid within sequence (TODO: standardise variable naming)
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
        tokenizer: ProFamTokenizer,
        require_seq_pos: bool = True,
        embed_coords: bool = False,
        start_seq_pos: int = 2,
        embed_sequence_index: bool = False,
        pass_constant_position_ids_for_global_index: bool = False,
        pass_sequence_position_ids_for_global_index: bool = False,
        max_sequence_index: int = 1024,
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.use_seq_pos = tokenizer.use_seq_pos
        self.start_seq_pos = start_seq_pos  # TODO: double-check this is consistent
        # TODO: avoid re-tracking - does this happen automatically?
        self.token_embedder = nested_getattr(
            self, token_embedder
        )  # TODO: use self.embed_tokens or sthg
        self.require_seq_pos = require_seq_pos
        self.tokenizer = tokenizer
        self.embed_coords = embed_coords
        self.start_seq_pos = start_seq_pos
        self.num_atoms = 4
        self.embed_sequence_index = embed_sequence_index
        self.max_sequence_index = max_sequence_index
        self.pass_constant_position_ids_for_global_index = (
            pass_constant_position_ids_for_global_index
        )
        self.pass_sequence_position_ids_for_global_index = (
            pass_sequence_position_ids_for_global_index
        )
        if self.embed_coords:
            self.coords_embedding = nn.Linear(
                self.num_atoms * 3,
                embedding_dim,
                bias=False,
            )
        if self.tokenizer.use_seq_pos:
            self.seq_pos_embedding = nn.Embedding(
                self.tokenizer.max_seq_pos, embedding_dim
            )
        if self.embed_sequence_index:
            self.sequence_index_embedding = nn.Embedding(
                self.max_sequence_index,
                embedding_dim,
            )

    def update_seq_pos_for_generation(self, input_ids, prompt_seq_pos):
        # n.b. generate automatically adds pad token to the end of finished sequences.
        # so if we want to support generation only of single sequences, we can just not worry about
        # effect of sep token on incrementation of seq pos.
        prompt_length = prompt_seq_pos.shape[-1]
        if input_ids.shape[-1] != prompt_length:
            generated_tokens = input_ids[:, prompt_length:]
            # basically we are saying that eos_token_id in generation config must be sep_token_id
            assert_only_padding_after_eos(
                generated_tokens,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            )

            # we have incremented input ids but not seq pos
            increment = input_ids.shape[-1] - prompt_length

            # https://github.com/huggingface/transformers/blob/cf32ee1753c9747b877113a309c2aa989f6d006c/src/transformers/models/llama/modeling_llama.py#L1236
            # just automatically increment the seq pos: this corresponds to never generating insertions in case of msas.

            input_final_seq_pos = prompt_seq_pos[:, -1:]
            if (input_final_seq_pos[:, -1] == 0).any():  # handles sep cases
                assert input_ids[0, prompt_length - 1].item() in [
                    self.tokenizer.sep_token_id,
                    self.tokenizer.seq_struct_sep_token_id,
                ], f"{input_ids[0, prompt_length-1]} {increment}"
                assert (input_final_seq_pos[:, -1] == 0).all()
                # we are starting new sequences
                seq_pos = torch.full_like(
                    input_final_seq_pos, self.start_seq_pos + increment - 1
                )
                # seq_pos corresponds to position of previously generated token in the sequence
                # when increment is 1, seq_pos is self.start_seq_pos
            else:
                if increment == 1:
                    print(
                        f"Warning: not sampling a new sequence, check inputs if this is desired behaviour "
                        f"({prompt_seq_pos}, {input_ids})"
                    )
                seq_pos = input_final_seq_pos + increment
        else:
            seq_pos = prompt_seq_pos

        return seq_pos

    # This needs to be the instantiation target if using seq pos... or wrapped hf model needs to handle properly
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        seq_pos=None,
        cache_position=None,
        coords=None,
        start_sequence_index=0,
        **kwargs,
    ):
        """Build inputs dictionary for next step in generation,
        given full input_ids (prompt + generated tokens), and other full length model inputs.

        Main function is to slice out unprocessed inputs using cache_position.

        Inputs are then incremented in _update_model_kwargs_for_generation.
        For some inputs, we currently have to update here (seq_pos, sequence_index) -
        https://github.com/huggingface/transformers/issues/33548

        n.b. we need to be aware of main steps of generation pipeline (self.generate)
        1. null cache gets created (setting past_key_values in model_kwargs) - unless creation is required from start
        https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L1854
        2. get_initial_cache_position:
        https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L2969
        n.b. cache_position is a very misleading name for cache-aware position index
        initially, cache position is just arange(len(input_ids))
        Then loop:
        3. prepare_inputs_for_generation: slice out input ids if using cache.
            in first iteration this does nothing, since cache_position shape == len(input_ids)
            in subsequent iterations, cache_position is index of newly generated token(s)
            relative to updated input_ids (i.e. prompt + generated tokens). so input_ids[cache_position]
            selects just the newly generated token(s) from the last sampling iteration to feed to the model
        4. compute logits for next position in sequence, predict a new token and update input ids
        5. update_model_kwargs_for_generation: update cache, attention_mask, cache_position.
        """
        # TODO: consider putting this in update_model_kwargs_for_generation - definitely yes!.

        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # in sample loop 'input_ids' gets incremented with generated tokens

        assert input_ids.ndim == 2

        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )  # slices out prompt and uses cache typically.

        # input_ids is prompt + generated tokens
        # kwargs["seq_pos"] is prompt only
        # inputs["input_ids"] is last generated token - so far not passed through model:
        # this is sliced from input_ids and added to inputs dict in base class prepare_inputs_for_generation
        if self.use_seq_pos:
            inputs["seq_pos"] = self.update_seq_pos_for_generation(input_ids, seq_pos)

        if self.embed_sequence_index:
            # model will automatically do compute_sequence_index on new tokens
            # so we just need to tell it the sequence index of the new tokens
            # suppose input_ids[:, -1] is sep token. then compute_sequence_index here will assign it
            # to previous sequence, and in forward will just pass through start_sequence_index
            full_sequence_index = self.compute_sequence_index(
                input_ids, start_sequence_index=start_sequence_index
            )
            inputs["start_sequence_index"] = full_sequence_index[:, -1]

        if self.embed_coords:
            # updated in _update_model_kwargs_for_generation
            assert input_ids.shape[-1] == coords.shape[1]
            inputs["coords"] = (
                coords[:, cache_position] if past_key_values is not None else coords
            )

        return inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ):
        """Update model kwargs for next step in generation, given model outputs and current model kwargs.

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        next_tokens = ....
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if new token is sep, then new seq pos should be incremented.
        if prev token is sep, then new seq pos should be 0 and new sequence index should be incremented.
        """
        # update past_key_values using model output, token_type_ids, attention_mask, cache_position
        # TODO: check whether attention_mask update assumes 2d?
        super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if "coords" in model_kwargs:
            bsz, _, n_atoms, _ = model_kwargs["coords"].shape
            model_kwargs["coords"] = torch.cat(
                [
                    model_kwargs["coords"],
                    torch.zeros(bsz, num_new_tokens, n_atoms, 3).to(
                        model_kwargs["coords"]
                    ),
                ],
                dim=1,
            )

        return model_kwargs

    def compute_sequence_index(self, input_ids, start_sequence_index=0):
        # TODO: test - if input_ids is just sep token we return start_sequence_index
        # cat means sep token gets index of PREVIOUS sequence
        return start_sequence_index + torch.cat(
            (
                torch.full_like(input_ids[..., :1], 0),
                torch.cumsum(
                    (input_ids == self.tokenizer.sep_token_id).float(), dim=-1
                ).long()[..., :-1],
            ),
            dim=-1,
        )

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
        coords: Optional[torch.FloatTensor] = None,
        start_sequence_index: int = 0,
    ):
        # we assume (which is case for e.g. gpt2 and mistral)
        # that the model will itself add its own position embeddings to inputs_embeds
        assert input_ids.ndim == 2

        # in this case model's position ids will be inferred from inputs_embeds
        inputs_embeds = self.token_embedder(input_ids)
        if self.tokenizer.use_seq_pos:
            if self.require_seq_pos:
                assert seq_pos is not None
            if seq_pos is not None:
                pos_embeds = self.seq_pos_embedding(seq_pos)
                inputs_embeds = inputs_embeds + pos_embeds

        # TODO: might want to embed coords mask to allow for masked coords
        if self.embed_coords:
            assert coords.ndim == 4, coords.shape  # b, l, n, 3
            coords_embeds = self.coords_embedding(coords.flatten(start_dim=-2))
            inputs_embeds += coords_embeds

        if self.embed_sequence_index:
            sequence_index = self.compute_sequence_index(
                input_ids, start_sequence_index=start_sequence_index
            )
            inputs_embeds += self.sequence_index_embedding(sequence_index)

        return inputs_embeds

    def get_position_ids_for_model_forward(self, input_ids, seq_pos, position_ids):
        # TODO: test these; make sure they get called during generation for example.
        if self.pass_constant_position_ids_for_global_index:
            assert position_ids is None
            position_ids = torch.full_like(input_ids, 10).long()
        if self.pass_sequence_position_ids_for_global_index:
            assert position_ids is None
            assert seq_pos is not None
            position_ids = seq_pos
        return position_ids

    def compute_start_sequence_index(self, past_key_values):
        if past_key_values is None:
            return 0
        else:
            raise NotImplementedError("Compute from cached input ids")

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
        start_sequence_index: Optional[
            Union[torch.Tensor, int]
        ] = None,  # index of sequence within document. modify when using cache.
        **kwargs,  # e.g. labels
    ):
        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"

        if self.embed_sequence_index and past_key_values is not None:
            assert (
                start_sequence_index is not None
            ), "Must pass start_sequence_index if using sequence index embeddings with cache"
        elif start_sequence_index is None:
            start_sequence_index = 0

        inputs_embeds = self.embed_inputs(
            input_ids,
            seq_pos=seq_pos,
            coords=coords,
            start_sequence_index=start_sequence_index,
        )
        position_ids = self.get_position_ids_for_model_forward(
            input_ids, seq_pos, position_ids
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
