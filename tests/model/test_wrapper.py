import numpy as np
import torch

from src.data.objects import ProteinDocument
from src.data.processors.batch_transforms import pack_examples

# def test_generate():
#     from transformers import LlamaForCausalLM, LlamaConfig
#     config = LlamaConfig(hidden_dim=64, num_attention_heads=4, num_hidden_layers=1, vocab_size=20, intermediate_size=128)
#     model = LlamaForCausalLM(config)

#     model.eval()
#     input_ids = torch.ones(1, 10).long()
#     attention_mask = torch.zeros((
#         1, 1, 10, 10
#     ))
#     # attention_mask = torch.ones(1, 10)
#     model.generate(
#         input_ids=input_ids,
#         max_new_tokens=2,
#         do_sample=True,
#         num_return_sequences=2,
#         attention_mask=attention_mask,
#     )


def test_compute_sequence_index(test_model, profam_tokenizer):
    sequences = ["ARC", "MKLLL", "MKPP"]
    document = ProteinDocument(sequences=sequences)
    tokenized = profam_tokenizer.encode(document)
    sequence_indices = test_model.model.compute_sequence_index(
        torch.from_numpy(tokenized.input_ids[None])
    )
    expected_sequence_indices = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    )
    assert (sequence_indices.numpy() == expected_sequence_indices).all()


def test_compute_res_pos_in_doc(test_model, profam_tokenizer):
    sequences = ["ARC", "MKLL", "MK"]
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences, original_size=len(sequences)),
        add_final_sep=True,
    )
    # n.b. packing happens after tokenization
    examples = [tokenized.data, tokenized.data]
    packed_examples = pack_examples(examples)
    position_ids = test_model.model.compute_res_pos_in_doc(
        torch.from_numpy(packed_examples["input_ids"][None, :])
    )
    expected_position_ids = np.array(
        list(range(sum(len(s) for s in sequences) + len(sequences) + 2)) * 2
    )
    assert (
        position_ids[0].numpy() == expected_position_ids
    ).all(), f"Expected {expected_position_ids}, got {position_ids[0].numpy()}"


def test_packed_compute_sequence_index(test_model, profam_tokenizer):
    sequences = ["ARC", "MKLLL", "MKPP"]
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences, original_size=len(sequences)),
        add_final_sep=True,
    )
    # n.b. packing happens after tokenization
    examples = [tokenized.data, tokenized.data]
    packed_examples = pack_examples(examples)
    sequence_indices = test_model.model.compute_sequence_index(
        torch.from_numpy(packed_examples["input_ids"][None, :])
    )
    expected_sequence_indices = np.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2] * 2]
    )
    assert (
        sequence_indices.numpy() == expected_sequence_indices
    ).all(), f"Expected {expected_sequence_indices}, got {sequence_indices.numpy()}"


def test_prepare_inputs_for_generation(model_seq_index, profam_tokenizer):
    # n.b. we need to be aware of main steps of generation pipeline (self.generate)
    # 1. null cache gets created (setting past_key_values in model_kwargs) - unless creation is required from start
    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L1854
    # 2. get_initial_cache_position:
    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L2969
    # n.b. cache_position is a very misleading name for cache-aware position index
    # initially, cache position is just arange(len(input_ids))
    # Then loop:
    # 3. prepare_inputs_for_generation: slice out input ids if using cache.
    #    in first iteration this does nothing, since cache_position shape == len(input_ids)
    #    in subsequent iterations, cache_position is index of newly generated token(s)
    #    relative to updated input_ids (i.e. prompt + generated tokens). so input_ids[cache_position]
    #    selects just the newly generated token(s) from the last sampling iteration to feed to the model
    # 4. compute logits for next position in sequence, predict a new token and update input ids
    # 5. update_model_kwargs_for_generation: update cache, attention_mask, cache_position.

    model_seq_index.eval()  # required for cache to be activated (even with use_cache True)
    sequences = ["ARC", "MKLL", "MK"]
    # imagine we are generating a new sequence after the second prompt sequence
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences), add_final_sep=False
    )
    input_ids = torch.from_numpy(tokenized.input_ids[None, :-2])

    model_kwargs = {
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
        torch.from_numpy(tokenized.input_ids[None, :-1]),
        **model_kwargs,
    )

    # TODO: add next step.

    with torch.no_grad():
        outputs = model_seq_index.model(**inputs)

    model_kwargs = model_seq_index.model._update_model_kwargs_for_generation(
        outputs,
        model_kwargs,
        is_encoder_decoder=model_seq_index.model.config.is_encoder_decoder,
    )
    inputs = model_seq_index.model.prepare_inputs_for_generation(
        torch.from_numpy(tokenized.input_ids[None]),
        **model_kwargs,
    )


def test_compute_sequence_index(model_seq_index, profam_tokenizer):
    """Sequence index gets computed based on sep tokens.

    TODO: add test for multiple documents.
    TODO: add test for compute sequence index with cache
    """
    model_seq_index.eval()  # required for cache to be activated (even with use_cache True)
    sequences = ["ARC", "MKLL", "MK"]
    # imagine we are generating a new sequence after the second prompt sequence
    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences), add_final_sep=False
    )
    input_ids = torch.from_numpy(tokenized.input_ids[None, :-2])

    model_kwargs = {
        "use_cache": True,
        "past_key_values": None,
    }

    with torch.no_grad():
        outputs = model_seq_index.model(input_ids=input_ids, **model_kwargs)

    sequence_index = model_seq_index.model.compute_sequence_index(
        torch.from_numpy(tokenized.input_ids[None])
    )
    extra_start_tokens_per_doc = 2
    expected_sequence_index = torch.tensor(
        [0] * (3 + extra_start_tokens_per_doc + 1) + [1] * (4 + 1) + [2] * 2
    )
    assert (sequence_index == expected_sequence_index).all()

    # TODO: add edge cases - e.g. final sep token; final bos token.
    start_sequence_index = model_seq_index.model.compute_start_sequence_index(
        outputs["past_key_values"]
    )
    assert start_sequence_index.item() == 2

    tokenized = profam_tokenizer.encode(
        ProteinDocument(sequences=sequences), add_final_sep=True
    )
    input_ids = torch.from_numpy(tokenized.input_ids[None, :])

    model_kwargs = {
        "use_cache": True,
        "past_key_values": None,
    }

    with torch.no_grad():
        outputs = model_seq_index.model(input_ids=input_ids, **model_kwargs)

    start_sequence_index = model_seq_index.model.compute_start_sequence_index(
        outputs["past_key_values"]
    )
    assert start_sequence_index.item() == 3
