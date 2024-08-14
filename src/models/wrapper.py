from typing import Callable, List, Optional

import torch
from torch import nn
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel


# TODO: unify with below?
class WrappedHFModel(PreTrainedModel):
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # main place this gets called is in sample loop:
        # https://github.com/huggingface/transformers/blob/e7f4ace0929600606424efd4cd91947bd567d323/src/transformers/generation/utils.py#L2413
        # we're going to assume that the prompt ends with a separator token
        assert "seq_pos" in kwargs
        inputs = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if input_ids.shape[-1] != kwargs["seq_pos"].shape[-1]:
            # we have incremented input ids but not seq pos
            assert input_ids.shape[-1] == kwargs["seq_pos"].shape[-1] + 1
            # just automatically increment the seq pos: this corresponds to never generating insertions in case of msas.
            prev_seq_pos = kwargs["seq_pos"][:, -1]
            seq_pos = torch.cat([prev_seq_pos, prev_seq_pos + 1], dim=-1)
            inputs["seq_pos"] = seq_pos
        return inputs


class TransformerWithSequencePositionEmbeddings(nn.Module):
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        token_embedder: Callable,
        embedding_dim: int,
        use_seq_pos: bool = False,
        max_seq_pos: int = 2048,
        require_seq_pos: bool = True,
    ):
        super().__init__()
        self.model = WrappedHFModel(model)
        self.token_embedder = token_embedder
        self.use_seq_pos = use_seq_pos
        self.require_seq_pos = require_seq_pos
        self.max_seq_pos = max_seq_pos
        if self.use_seq_pos:
            self.seq_pos_embedding = nn.Embedding(self.max_seq_pos, embedding_dim)

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
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
        **kwargs,  # e.g. labels
    ):
        assert (
            inputs_embeds is None
        ), "Do not pass pre-computed embeddings to this class"
        inputs_embeds = self.embed_inputs(input_ids, seq_pos=seq_pos)
        return self.model(
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
