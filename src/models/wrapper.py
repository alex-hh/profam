from typing import List, Optional

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel


class TransformerWithSequencePositionEmbeddings(nn.Module):
    """Wrap a pre-trained model to add sequence-relative position embeddings.

    (Optionally other embeddings, e.g. structure embeddings, could be added in similar way.)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        use_seq_pos: bool = False,
        max_seq_pos: int = 2048,
    ):
        super().__init__()
        self.model = model
        self.use_seq_pos = use_seq_pos
        self.max_seq_pos = max_seq_pos
        if self.use_seq_pos:
            # TODO: do all models have embed_tokens embedding layer? presumably yes...
            self.seq_pos_embedding = nn.Embedding(
                self.max_seq_pos, self.model.embed_tokens.embedding_dim
            )

    def embed_inputs(
        self,
        input_ids: Optional[torch.LongTensor],
        seq_pos: Optional[torch.LongTensor] = None,
    ):
        # n.b. we need to be careful about what happens when caching.
        # I think in that case input_ids should just be the continuation
        # and inputs_embeds should also.
        inputs_embeds = self.embed_tokens(input_ids)
        if self.use_seq_pos:
            assert seq_pos is not None
            pos_embeds = self.seq_pos_embedding(seq_pos)
        return inputs_embeds + pos_embeds

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
        inputs_embeds = self.embed_inputs(input_ids, inputs_embeds)
        return self.model(
            input_ids=None,
            attention_mask=attention_mask,
            seq_pos=seq_pos,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
