"""
For a given type of family
we create a dataset with a target family
and a mix of decoys (members of other families)
and non-decoys (members of the target family)
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


def load_classifier_dataset(
    fasta_file_pattern,
    tokenizer: PreTrainedTokenizerFast,
    max_tokens=10000,
    num_decoys_per_target=5,  # todo remove this once we have fixed the data
    use_seq_pos=False,
    max_seq_pos: int = 1024,
    keep_gaps=True,
    to_upper=False,
    keep_insertions=True,
    max_eval_seqs=100,
    seed=42,
):
    paths = sorted(glob.glob(fasta_file_pattern))
    random.seed(seed)
    dataset_list = []

    for target_family_path in paths:
        # Load sequences from the target family
        _, seqs = fasta.read_fasta(target_family_path)
        target_family_seqs = seqs.copy()
        n_targ_seqs = min(len(target_family_seqs) // 2, max_eval_seqs)
        target_seqs = random.sample(target_family_seqs, n_targ_seqs)
        remaining_seqs = list(set(target_family_seqs) - set(target_seqs))
        decoy_seqs = []
        for decoy_family_path in paths:
            if decoy_family_path == target_family_path:
                continue
            # Load sequences from the decoy family
            _, seqs = fasta.read_fasta(decoy_family_path)
            # Sample sequences for the decoys
            decoy_seqs.extend(random.sample(seqs, num_decoys_per_target))
        completion_seqs = target_seqs + decoy_seqs
        # save space for completions
        max_tokens_for_msa = max_tokens - max([len(s) for s in completion_seqs]) - 2
        msa_seqs = data_utils.sample_to_max_tokens(
            remaining_seqs, seed=seed, max_tokens=max_tokens_for_msa
        )

        family_labels = [1] * len(target_seqs) + [0] * len(decoy_seqs)

        # Create a dictionary for the current target family
        family_dict = {
            "MSA": msa_seqs,
            "completion_seqs": completion_seqs,
            "family_labels": family_labels,
        }
        dataset_list.append(family_dict)

    # Create a dataset from the list of dictionaries
    dataset = Dataset.from_pandas(pd.DataFrame(dataset_list))
    dataset = dataset.map(  #  todo 20 lines almost identical to src/data/proteingym.py
        partial(
            tokenize,
            tokenizer=tokenizer,
            mutant_bos_token="sep",  # todo check this
            use_seq_pos=use_seq_pos,
            max_seq_pos=max_seq_pos,
        ),
        batched=False,
        remove_columns=["MSA", "completion_seqs"],
    )
    columns = ["input_ids", "completion_ids", "family_labels"]
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset
