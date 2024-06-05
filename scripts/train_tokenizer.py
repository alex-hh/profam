# train_tokenizer.py

from tokenizers import Tokenizer, decoders, models, trainers
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Initialize the tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Define tokens
tokens = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "[PAD]",
    "[start-of-document]",
    "[end-of-document]",
    "[SEP]",
    "[UNK]",
    "[MASK]",
]
tokenizer.add_tokens(tokens)
tokenizer.model.unk_token = "[UNK]"

# Train the tokenizer
trainer = trainers.BpeTrainer(
    special_tokens=tokens, vocab_size=25
)  # Set vocab_size to control the number of tokens
# Replace the empty iterator with actual data
data = ["ARNDC QEGHIL KMFPST WYV", "ARNDC"]
tokenizer.train_from_iterator(data, trainer=trainer)

# Save the tokenizer
tokenizer.save("src/data/components/profam_tokenizer.json")

# Load the tokenizer with transformers
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="src/data/components/profam_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[start-of-document]",
    eos_token="[end-of-document]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Test the tokenizer
example_sequence = [
    "[start-of-document] ARNDC QEGHIL KMFPST WYV [end-of-document] [PAD] [UNK] [SEP]",
    "[start-of-document] ARNDC",
]
tokens = fast_tokenizer.batch_encode_plus(
    example_sequence,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=25,  # Set a max length for truncation
)
ids = tokens["input_ids"]

# Assert tests
assert len(ids) == 2, "The number of encoded sequences should be 2."
assert (
    ids[0][0] == fast_tokenizer.cls_token_id
), "The first token of the first sequence should be the CLS token."
assert (
    ids[1][0] == fast_tokenizer.cls_token_id
), "The first token of the second sequence should be the CLS token."
assert ids[1][1] == fast_tokenizer.convert_tokens_to_ids(
    "A"
), "The second token of the second sequence should be the 'A' token."
assert (
    ids[0][-1] == fast_tokenizer.sep_token_id
), "The last token of the first sequence should be the SEP token."


print("Tokens:", tokens)
print("Token IDs:", ids)

# Save the tokenizer in the Hugging Face format
fast_tokenizer.save_pretrained("src/data/components/profam_tokenizer")
