"""
Refs
https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization
"""
import torch
from torch.utils.data import DataLoader
from transformers import MistralConfig, PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache

from src.data.proteingym import load_gym_dataset
from src.models.mistral_lit_module import MistralLitModule, log_likelihood_from_outputs

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="src/data/components/profam_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[start-of-document]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    add_special_tokens=True,
)
config = MistralConfig(
    vocab_size=tokenizer.vocab_size,
    num_hidden_layers=2,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=4,
    pad_token_id=25,
    bos_token_id=26,
    eos_token_id=27,
)
gym_dataset = load_gym_dataset(
    dms_ids=["BLAT_ECOLX_Jacquier_2013", "DLG4_RAT_McLaughlin_2012"],
    tokenizer=tokenizer,
    max_mutated_sequences=5,
)
batch_size = 1
gym_loader = DataLoader(gym_dataset, batch_size=batch_size, num_workers=0, shuffle=False)
batch = next(iter(gym_loader))
print(
    batch["completion_ids"].shape, batch["input_ids"].shape, batch["DMS_scores"].shape
)
model = MistralLitModule(config)
model.eval()


# first run un-cached forward pass for comparison
input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
print("Prompt shape", batch["input_ids"].shape)
completion_start_ix = batch["input_ids"].shape[1] + 1  # skip the SEP token
assert input_ids[..., completion_start_ix - 1] == tokenizer.sep_token_id
past_key_values = None
with torch.no_grad():
    outputs = model(input_ids, use_cache=False)
    logits_v1 = outputs.logits
    log_likelihood_v1 = log_likelihood_from_outputs(
        outputs, input_ids, start_ix=completion_start_ix - 1
    )

# next run forward pass, caching the kv states
# input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
past_key_values = None
with torch.no_grad():
    outputs = model(batch["input_ids"], past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values

assert len(past_key_values) == config.num_hidden_layers
assert len(past_key_values[0]) == 2  # tuple (k, v)
assert past_key_values[0][0].shape == (batch_size,
                                       config.num_key_value_heads,
                                       batch["input_ids"].shape[-1],
                                       config.hidden_size // config.num_attention_heads)
# Is this also neceessary at train time?

# Note on past_key_values: Two formats are allowed:

# a Cache instance;
# Tuple of tuple(torch.FloatTensor) of length config.n_layers, with each tuple having 2
# tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)). This
# is also known as the legacy cache format.
# The model will output the same cache format that is fed as input. If no past_key_values are passed, the legacy cache format will be returned.
# run forward pass for completion using cached kv states

# results are not identical
# this might be expected as discussed here:
# https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535
# the claim would be that for the first logit, the results are very close...
# we might then see more variation for subsequent logits (not considered in the issue)
# due to accumulation of errors

# bug if use_cache = False

# position ids
# if use_cache and not isinstance(past_key_values, Cache):
#     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
#     return_legacy_cache = True

# if cache_position is None:
#     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
#     cache_position = torch.arange(
#         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
#     )

# if position_ids is None:
#     position_ids = cache_position.unsqueeze(0)

# THIS IS JUST WHAT HAPPENS UNDER THE HOOD IN FORWARD TO COMPUTE POSITION IDS
cache = DynamicCache.from_legacy_cache(past_key_values)
past_seen_tokens = cache.get_seq_length()
# inputs_embeds = self.embed_tokens(input_ids)
cache_position = torch.arange(
    past_seen_tokens,
    past_seen_tokens + batch["completion_ids"][:, 0].shape[1],
    device=input_ids.device,
)
# print("Completion ids", batch["completion_ids"][:, 0])
print("Cache positions (position ids)", cache_position.unsqueeze(0))
with torch.no_grad():
    outputs = model(
        batch["completion_ids"][:, 0], past_key_values=past_key_values, use_cache=True
    )
    logits_v2 = outputs.logits
    log_likelihood_v2 = log_likelihood_from_outputs(
        outputs, batch["completion_ids"][:, 0]
    )


# check that if we re-run, cache hasn't been updated (not even possible: its a tuple of tensors)
with torch.no_grad():
    outputs = model(
        batch["completion_ids"][:, 0], past_key_values=past_key_values, use_cache=True
    )
    logits_v3 = outputs.logits
    log_likelihood_v3 = log_likelihood_from_outputs(
        outputs, batch["completion_ids"][:, 0]
    )


with torch.no_grad():
    outputs = model(input_ids)
    log_likelihood_v4 = log_likelihood_from_outputs(
        outputs, input_ids, start_ix=completion_start_ix - 1
    )

assert torch.isclose(log_likelihood_v1, log_likelihood_v2).all()
assert torch.isclose(log_likelihood_v3, log_likelihood_v2).all()

print((logits_v1[:, completion_start_ix - 1] - logits_v2[:, 0]).abs().max())
