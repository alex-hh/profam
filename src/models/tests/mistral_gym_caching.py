import torch
from torch.utils.data import DataLoader
from transformers import MistralConfig, PreTrainedTokenizerFast

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
gym_loader = DataLoader(gym_dataset, batch_size=1, num_workers=0, shuffle=False)
batch = next(iter(gym_loader))
print(
    batch["completion_ids"].shape, batch["input_ids"].shape, batch["DMS_scores"].shape
)
model = MistralLitModule(config)
model.eval()


# first run un-cached forward pass for comparison
input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
completion_start_ix = batch["input_ids"].shape[1] + 1  # skip the SEP token
assert input_ids[..., completion_start_ix - 1] == tokenizer.sep_token_id  # SEP token
past_key_values = None
with torch.no_grad():
    outputs = model(input_ids)
    log_likelihood = log_likelihood_from_outputs(
        outputs, input_ids, start_ix=completion_start_ix - 1
    )

print(log_likelihood, log_likelihood.shape)

# next run forward pass, caching the kv states
input_ids = torch.cat([batch["input_ids"], batch["completion_ids"][:, 0]], dim=1)
past_key_values = None
with torch.no_grad():
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values

# run forward pass for completion using cached kv states
with torch.no_grad():
    outputs = model(batch["completion_ids"][:, 0], past_key_values=past_key_values)
    log_likelihood = log_likelihood_from_outputs(outputs, input_ids)

print(log_likelihood, log_likelihood.shape)
