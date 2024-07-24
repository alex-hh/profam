"""
create evaluation datasets
for family classification
which contains:
- input_ids (prompt tokens)
- completion_ids (tokens of seqs to predict)
- family_labels (binary in-family or not)

for pfam we evluate likelihood of sequence
conditioning on prompts from different families
likelihoods for different prompts are logits
in a multi-class classification problem

to benefit from kv-caching we evaluate
all seqs against one msa at a time
"""

import glob
import random
from functools import partial

import pandas as pd
from torch import stack, arange
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data import fasta
from src.data import utils as data_utils
from src.data.proteingym import tokenize, tokenize_completions, tokenize_msa

def load_pfam_classification_dataset(
    tokenizer,
    max_tokens=10000,
    use_seq_pos=False,
    max_seq_pos: int = 1024,
    seed=42,
    num_workers=4,
):
    pfam_dir = '../data/pfam/pfam_eval_splits/clustered_split_fastas_debug'
    combined_eval_seqs = []
    eval_labels = []
    msa_paths = sorted(glob.glob(f"{pfam_dir}/*_train.fasta"))
    eval_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_test.fasta"))
    for eval_path in eval_seq_paths:
        eval_name = eval_path.split('/')[-1].split('_')[0]
        eval_names, eval_seqs = fasta.read_fasta(eval_path)
        assert len(set(eval_names)) == 1
        assert eval_name in eval_names
        combined_eval_seqs.extend(eval_seqs)
        eval_labels.extend(eval_names)
    max_tokens_for_msa = max_tokens - max([len(s) for s in combined_eval_seqs]) - 2
    data_rows = []
    for msa_path in msa_paths:
        msa_name = msa_path.split('/')[-1].split('_')[0]
        msa_names, msa_seqs = fasta.read_fasta(msa_path)
        assert len(set(msa_names)) == 1
        assert msa_name in msa_names
        assert set(msa_seqs).intersection(set(combined_eval_seqs)) == set()
        # msa_seqs = data_utils.sample_to_max_tokens(
        #     msa_seqs, seed=seed, max_tokens=max_tokens_for_msa
        # )
        data_row = {
            "MSA": msa_seqs,
            "completion_seqs": combined_eval_seqs,
            "family_labels": [1 if s == msa_name else 0 for s in eval_labels]
        }
        data_rows.append(data_row)
    dataset = Dataset.from_pandas(pd.DataFrame(data_rows))
    dataset = dataset.map(
        partial(
            tokenize,
            tokenizer=tokenizer,
            use_seq_pos=use_seq_pos,
            max_seq_pos=max_seq_pos
        ),
        batched=False,
        remove_columns=["MSA", "completion_seqs"]
    )
    columns = ["input_ids", "completion_ids", "family_labels"]
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset


# def tokenize_eval_seqs(tokenizer, eval_seqs, use_seq_pos, max_seq_pos):
#     # Tokenize eval_seqs once
#     tokenized_eval_seqs = tokenizer(
#         eval_seqs,
#         padding=True,
#         truncation=True,
#         max_length=max_seq_pos if use_seq_pos else None,
#         return_tensors="pt"
#     )
#     return tokenized_eval_seqs


def process_msa(msa_path, eval_names, tokenized_eval_seqs,
                tokenizer, use_seq_pos, max_seq_pos, completion_seq_pos):
    msa_name = msa_path['msa_paths'].split('/')[-1].split('_')[0]
    msa_names, msa_seqs = fasta.read_fasta(msa_path['msa_paths'])
    assert len(set(msa_names)) == 1
    assert msa_name in msa_names

    sample = {
        "MSA": msa_seqs,
        "completion_ids": tokenized_eval_seqs,
    }
    if use_seq_pos:
        sample["completion_seq_pos"] = completion_seq_pos

    sample = tokenize(
        sample,
        tokenizer=tokenizer,
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
        mutant_bos_token="sep"
    )

    sample["family_labels"] = [1 if s == msa_name else 0 for s in eval_names]
    return sample




def load_pfam_classification_dataset_optimized(
    tokenizer,
    max_tokens=10000,
    use_seq_pos=False,
    max_seq_pos: int = 1024,
    seed=42,
    num_workers=4,
    max_eval_per_fam=4,
):
    pfam_dir = '../data/pfam/pfam_eval_splits/clustered_split_fastas_debug'
    combined_eval_seqs = []
    eval_labels = []

    eval_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_test.fasta"))
    for eval_path in eval_seq_paths:
        eval_name = eval_path.split('/')[-1].split('_')[0]
        eval_names, eval_seqs = fasta.read_fasta(eval_path)
        assert len(set(eval_names)) == 1
        assert eval_name in eval_names
        combined_eval_seqs.extend(eval_seqs[:max_eval_per_fam])
        eval_labels.extend(eval_names[:max_eval_per_fam])

    # Tokenize eval sequences once
    tok_eval_seqs = tokenize_completions(
        sample={"completion_seqs": combined_eval_seqs},
        tokenizer=tokenizer, bos_token="sep"
    )
    if use_seq_pos:
        n_seqs, longest = tok_eval_seqs["completion_ids"].shape
        completion_seq_pos = arange(-1, longest-1).repeat(n_seqs, 1)
        assert completion_seq_pos.shape == tok_eval_seqs["completion_ids"].shape
        completion_seq_pos[:, 0] = 0
        completion_seq_pos.clamp_(max=max_seq_pos)
    tokenized_eval_seqs = tok_eval_seqs["completion_ids"]
    # tokenized_eval_seqs = tokenize_eval_seqs(tokenizer, combined_eval_seqs, use_seq_pos, max_seq_pos)
    msa_paths = sorted(glob.glob(f"{pfam_dir}/*_train.fasta"))

    process_func = partial(
        process_msa,
        eval_names=eval_labels,
        tokenized_eval_seqs=tokenized_eval_seqs,
        tokenizer=tokenizer,
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
        completion_seq_pos=completion_seq_pos,
    )

    # Create dataset and apply processing
    dataset = Dataset.from_dict({"msa_paths": msa_paths})
    dataset = dataset.map(
        process_func,
        remove_columns=["msa_paths"],
        batched=False,
        # batch_size=10,  # Adjust this based on your memory constraints
        num_proc=num_workers,
    )

    columns = ["input_ids", "completion_ids", "family_labels"]
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset

if __name__ == '__main__':
    load_pfam_classification_dataset_optimized()
