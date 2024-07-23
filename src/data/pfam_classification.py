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
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.data import fasta
from src.data import utils as data_utils
from src.data.proteingym import tokenize

def load_pfam_classification_dataset(
    tokenizer,
    max_tokens=10000,
    use_seq_pos=False,
    max_seq_pos: int = 1024,
    seed=42
):
    pfam_dir = '../data/pfam/pfam_eval_splits/clustered_split_fastas'
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

if __name__ == '__main__':
    load_pfam_classification_dataset()
