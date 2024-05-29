"""
See https://colab.research.google.com/github/huggingface/notebooks/blob/main/transformers_doc/en/pytorch/language_modeling.ipynb
"""

import bisect
import itertools
import random
import pandas as pd
from configs.config_mixral import train_example as config

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


class FastaDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
        preprocessed_text = self.preprocess_text(text)
        return {'text': preprocessed_text}


    def preprocess_text(self, text):
        sequences = text.split(">")[1:]  # Split the text by '>' and remove the first empty element
        processed_sequences = []
        for seq in sequences:
            lines = seq.strip().split("\n")
            header = lines[0]
            sequence = "".join(lines[1:])
            sequence = sequence.replace("-", "")  # Remove deletions indicated by '-'
            processed_sequences.append(sequence)
        return "\n".join(processed_sequences)

def preprocess_text(sample, max_tokens=5000):
    text = sample["text"]
    sequences = text.split(">")[1:]  # Split the text by '>' and remove the first empty element
    processed_sequences = []

    for seq in sequences:
        lines = seq.strip().split("\n")
        header = lines[0]
        sequence = "".join(lines[1:])
        sequence = sequence.replace("-", "")  # Remove deletions indicated by '-'
        processed_sequences.append(sequence)

    sampled_sequences = [processed_sequences[0]]
    token_count = len(processed_sequences[0]) + 3
    shuffled_sequences = processed_sequences[1:]
    random.shuffle(shuffled_sequences)
    for seq in shuffled_sequences:
        if token_count + len(seq) + 1 > max_tokens:
            break
        sampled_sequences.append(seq)
        token_count += len(seq) + 1
    sample['text'] = "\n".join(sampled_sequences)
    return sample

# def load_data(filepaths):
#   # Use newline character as the special sequence separator
#
#     dataset = load_dataset("text", data_files=filepaths, sample_by="document", streaming=True)
#     dataset = dataset.map(lambda example: {"text": preprocess_text(example["text"])})
#     return dataset


def load_model(config, tokenizer):
    mistral_config = MistralConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config['hidden_size'],
        intermediate_size=config['intermediate_size'],
        num_hidden_layers=config['num_hidden_layers'],
        num_attention_heads=config['num_attention_heads'],
        num_key_value_heads=config['num_key_value_heads'],
        # TODO
        # pad_token_id=,
        # bos_token_id=,
        # eos_token_id=,
    )
    model = MistralForCausalLM(mistral_config)
    return model

def tokenize(example):
    return tokenizer(example["text"])


if __name__ == "__main__":
    filepaths = [
        "data/example_data/openfold/A0A0B7J0C7/a3m/uniclust30.a3m",
        "data/example_data/openfold/A0A0G3BHQ8/a3m/uniclust30.a3m",
        "data/example_data/openfold/A0A0P6DUH5/a3m/uniclust30.a3m",
        # Add more filepaths as needed
    ]
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    tokenizer.model_max_length = 1e8
    def tokenize(sample):
        return tokenizer(sample["text"])

    model = load_model(config['model'], tokenizer)
    # dataset = FastaDataset(filepaths)
    # dataset = dataset.map(lambda x: tokenizer(x['text']), remove_columns="text")
    paths_pattern = "data/example_data/openfold/*/a3m/uniclust30.a3m"
    dataset = load_dataset("text", data_files=filepaths, sample_by="document", streaming=True)
    dataset = dataset['train']
    processed_dataset = dataset.map(preprocess_text)
    tokenized_dataset = processed_dataset.map(tokenize, remove_columns="text")
    training_args = TrainingArguments(
        output_dir="openfold-examples",
        evaluation_strategy="no",
        report_to="none",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        max_steps=200,  # seemingly needed because iterabledataset has no len
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
