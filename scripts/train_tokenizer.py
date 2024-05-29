# train_tokenizer.py

from tokenizers import Tokenizer, models, decoders, trainers
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

# Initialize the tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# Define tokens
tokens = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
    "[PAD]", "[start-of-document]", "[end-of-document]", 
    "[SEP]", "[UNK]", "[MASK]"
]
tokenizer.add_tokens(tokens)
tokenizer.model.unk_token = "[UNK]"

# Train the tokenizer
trainer = trainers.BpeTrainer(special_tokens=tokens)
tokenizer.train_from_iterator([], trainer=trainer)

# Save the tokenizer
tokenizer.save("src/data/components/profam_tokenizer.json")

# Load the tokenizer with transformers
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="profam_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[start-of-document]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)

# Test the tokenizer
example_sequence = ["ARNDC [start-of-document] QEGHIL KMFPST WYV [end-of-document] [PAD] [UNK] [SEP]", "[start-of-document]ARNDC"]
tokens = fast_tokenizer.batch_encode_plus(example_sequence, return_tensors="pt")
ids = tokens["input_ids"]

# Assert tests
assert len(ids) == 2, "The number of encoded sequences should be 2."
assert ids[0][0] == fast_tokenizer.cls_token_id, "The first token of the first sequence should be the CLS token."
assert ids[1][0] == fast_tokenizer.cls_token_id, "The first token of the second sequence should be the CLS token."
assert ids[1][1] == fast_tokenizer.token_to_id("A"), "The second token of the second sequence should be the 'A' token."
assert ids[0][-1] == fast_tokenizer.sep_token_id, "The last token of the first sequence should be the SEP token."
assert ids[1][-1] == fast_tokenizer.sep_token_id, "The last token of the second sequence should be the SEP token."

print("Tokens:", tokens)
print("Token IDs:", ids)

# Save the tokenizer in the Hugging Face format
fast_tokenizer.save_pretrained("profam")
