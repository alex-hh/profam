import os

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.constants import BASEDIR
from src.data.objects import ProteinDocument


def test_compute_sequence_index(default_model, profam_tokenizer):
    sequences = ["ARC", "MKLLL", "MKPP"]
    document = ProteinDocument(sequences=sequences)
    tokenized = profam_tokenizer.encode(document)
    sequence_indices = default_model.model.compute_sequence_index(
        tokenized.input_ids[None]
    )
    expected_sequence_indices = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    )
    assert (sequence_indices == expected_sequence_indices).all()


@pytest.fixture()
def model_seq_index(profam_tokenizer):
    with initialize_config_dir(os.path.join(BASEDIR, "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["model.embed_sequence_index=True"],
        )
    return hydra.utils.instantiate(cfg.model, tokenizer=profam_tokenizer)


def test_prepare_inputs_for_generation(model_seq_index, profam_tokenizer):
    # n.b. we need to be aware of main steps of generation pipeline (self.generate)
    # Question: what does use_cache effect in generation? Seems like cache gets created regardless
    # 1. null cache gets created (setting past_key_values in model_kwargs) - unless creation is required from start
    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L1854
    # 2. get_initial_cache_position:
    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L2969
    # n.b. cache_position is a very misleading name for cache-aware position index
    # Then loop:
    # 3. prepare_inputs_for_generation: slice out input ids if using cache. 'slice input_ids' through cache_position - what does this mean.
    # 4. predict a new token and update input ids
    # 5. update_model_kwargs_for_generation: update cache, attention mask, cache positions

    model_seq_index.eval()  # required for cache to be activated (even with use_cache True)
    sequences = ["ARC", "MKLL", "M"]
    # imagine we are generating a new sequence after the second prompt sequence
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences), add_final_sep=False
    )
    input_seq_pos = tokenized.seq_pos[None, :-1]
    input_ids = tokenized.input_ids[None, :-1]

    model_kwargs = {
        "seq_pos": input_seq_pos,
        "use_cache": True,
        "past_key_values": None,
    }
    # c.f. sample:
    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L2969C9-L2969C81
    model_kwargs = model_seq_index.model._get_initial_cache_position(
        input_ids, model_kwargs
    )
    # cache position is [0,1,2,3,4,5,6,7,8,9,10]

    with torch.no_grad():
        outputs = model_seq_index.model(input_ids=input_ids, **model_kwargs)

    # in the loop at this point we sample a token from the logits and cat to input ids
    # here we imagine that that token is M (third sequence)
    # given this sampled token, we need to update cache, cache_position
    # updated cache position is index of input ids to pass to model for next token
    # i.e. input_ids[:, cache_position] should be the input ids to pass to model
    model_kwargs = model_seq_index.model._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=model_seq_index.model.config.is_encoder_decoder,
    )

    print(tokenized.input_ids[None], model_kwargs["cache_position"])
    # what happens to produce input ids
    print(tokenized.input_ids[None, model_kwargs["cache_position"]])

    # cache position is [11]
    # we have to pass past_key_values for default prepare_inputs_for_generation to slice out added input ids
    inputs = model_seq_index.model.prepare_inputs_for_generation(
        tokenized.input_ids[None],
        **model_kwargs,
    )
    print(inputs["input_ids"], inputs["start_sequence_index"], inputs["seq_pos"])
    assert (inputs["start_sequence_index"] == 2).all()
    assert (inputs["seq_pos"] == 2).all()

    # TODO: imagine we have already generated a bit; imagine we are starting from bos
