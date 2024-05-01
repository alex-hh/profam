"""
See https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/language_modeling.ipynb
"""

import bisect
import itertools
import random

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    MistralConfig,
    MistralForCausalLM,
    Trainer,
    TrainingArguments,
)

# we don't actually want to read the whole thing into memory...so we 'stream'
data = load_dataset(
    "text",
    data_files=[
        "data/example_data/interpro/protein-matching-IPR000117.fasta",
        "data/example_data/interpro/protein-matching-IPR000105.fasta",
    ],
    sample_by="document",
    streaming=True,  # create an iterable dataset https://huggingface.co/docs/datasets/main/en/stream
)
train_data = data["train"]


tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
tokenizer.model_max_length = 1e8


def remove_labels(example):
    # n.b. it would probably be better to do this on the file level, or at file loading level, or on the fly
    example["text"] = "\n".join(
        [line for line in example["text"].split("\n") if not line.startswith(">")]
    )
    return example


def subsample(example, max_length=5000, sequence_separator="\n"):
    sequences = [
        line for line in example["text"].split("\n") if not line.startswith(">")
    ]
    random.shuffle(sequences)
    cumulative_lengths = list(
        itertools.accumulate([len(s) + 1 for s in sequences])
    )  # +1 for separator
    insertion_point = bisect.bisect_left(cumulative_lengths, max_length)
    example["text"] = sequence_separator.join(sequences[:insertion_point])
    return example


# TODO replace sequence separator with something actually in the vocabulary
def tokenize(example):
    out = tokenizer(example["text"])
    return out


# Q. if we pass this to trainer, will the map happen differently each epoch?
train_data = train_data.map(subsample)
train_data = train_data.map(tokenize, remove_columns="text")


collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
config = MistralConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=8,
    num_key_value_heads=2,
    # TODO
    # pad_token_id=,
    # bos_token_id=,
    # eos_token_id=,
)
model = MistralForCausalLM(config)

training_args = TrainingArguments(
    output_dir="profam-examples",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    max_steps=2,  # needed because iterabledataset has no len
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=collator,
)

print("training")
trainer.train()
