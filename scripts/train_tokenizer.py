"""Overview of tokenisation pipeline:

https://huggingface.co/learn/nlp-course/chapter6/4?fw=pt
"""
import json

from tokenizers import Tokenizer, decoders, models, processors
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# Initialize the tokenizer
with open("scripts/vocab.json", "r") as jf:
    vocab = json.load(jf)
print(vocab)
tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))

# TODO: check whether we need a normalizer or similar to handle newline?
# seems to be treated as just a whitespace, which is fine
# The WhitespaceSplit pre-tokenizer splits on whitespace only (inc newline?). This produces a set
# of 'words' which the tokenizer will act on
tokenizer.pre_tokenizer = WhitespaceSplit()
# https://discuss.huggingface.co/t/regular-tokens-vs-special-tokens/6187
# special tokens are things that shouldnt be split
unassigned_special_tokens = [f"[SP{i}]" for i in range(1, 11)]
special_tokens = [
    "[PAD]",
    "[start-of-document]",
    "[end-of-document]",
    "[SEP]",
    "[MASK]",
    "[UNK]",
    "[RAW]",
    "[MSA]",
    "[RAW-WITH-MSA-POS]",
] + unassigned_special_tokens
tokenizer.add_special_tokens(special_tokens)
# N.B. if all special tokens aren't assigned we have tokenization issues...

# we can't include unk tokens in inputs explicitly; they have to be inferred.
# i.e. the inputs to encode should not contain any [unk]
print(
    tokenizer.encode(
        "[start-of-document] ARNDC QEGHIL KMFPST WYV [end-of-document] [PAD] rnd [SEP]\n[SEP]"
    ).tokens
)

# adding post-processor:
# https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt
# template procesing only supports a single sequence, so it makes sense to just do our own
# explicit addition of sep and bos eos tokens
# tokenizer.post_processor = processors.TemplateProcessing(
#     single=f"[CLS]:0 $A:0 [SEP]:0",
# )

# Save the tokenizer
tokenizer.save("src/data/components/profam_tokenizer.json")

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="src/data/components/profam_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[start-of-document]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    # Add them here to ensure they are skipped when decoding with skip_special_tokens is set to True
    additional_special_tokens=unassigned_special_tokens,
)

# Test the tokenizer
example_sequence_1 = (
    "[start-of-document] ARNDC [SEP] QEGHIL [SEP] KMFPST [SEP] WYV [SEP]"
)
example_sequence_2 = (
    "[start-of-document]ARNDC[SEP]QEGHIL[SEP]KMFPST[SEP]WYV[SEP]"  # also fine
)
example_sequence_3 = "[start-of-document] arndc [SEP] ARNDC [SEP] QEGHIL [SEP] KMFPST [SEP] WYV [SEP]"  # also fine
print("Example sequence 1 encoding", tokenizer.encode(example_sequence_1).tokens)
print("Example sequence 2 encoding", tokenizer.encode(example_sequence_2).tokens)
print("Example sequence 3 encoding", tokenizer.encode(example_sequence_3).tokens)
tokens = fast_tokenizer.batch_encode_plus(
    [example_sequence_1, example_sequence_2, example_sequence_3],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=25,  # Set a max length for truncation
)
ids = tokens["input_ids"]
print(ids)

# Assert tests
assert len(ids) == 3, "The number of encoded sequences should be 2."
assert (
    ids[0][0] == fast_tokenizer.bos_token_id
), "The first token of the first sequence should be the CLS token."
assert (
    ids[1][0] == fast_tokenizer.bos_token_id
), "The first token of the second sequence should be the CLS token."
assert ids[1][1] == fast_tokenizer.convert_tokens_to_ids(
    "A"
), "The second token of the second sequence should be the 'A' token."
assert (
    ids[0][-1] == fast_tokenizer.sep_token_id
), "The last token of the first sequence should be the EOS token."
assert (ids[0] == fast_tokenizer.sep_token_id).sum() == 4, "Expected 3 sep tokens"
assert (ids[1] == fast_tokenizer.sep_token_id).sum() == 4, "Expected 3 sep tokens"


print("Tokens:", tokens)
print("Token IDs:", ids)

# # Save the tokenizer in the Hugging Face format
# fast_tokenizer.save_pretrained("src/data/components/profam_tokenizer")
