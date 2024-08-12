"""
create evaluation datasets
for family classification
which contains:
- input_ids (prompt tokens)
- completion_ids (tokens of seqs to predict)
- family_labels (binary in-family or not)

for pfam we evaluate likelihood of sequence
conditioning on prompts from different families
likelihoods for different prompts are logits
in a multi-class classification problem

to benefit from kv-caching we evaluate
all seqs against one msa at a time
"""

import glob
from functools import partial

from datasets import Dataset
from torch import arange

from src.data import fasta
from src.data.utils import tokenize, tokenize_completions
import src.data.utils as data_utils




def tokenize_pfam(
    msa_path,
    eval_names,
    tokenized_eval_seqs,
    tokenizer,
    max_msa_tokens,
    use_seq_pos,
    max_seq_pos,
    completion_seq_pos,
    keep_insertions,
    to_upper,
    keep_gaps,
    seed=42,
):
    """"""
    msa_name = msa_path["msa_paths"].split("/")[-1].split("_")[0]
    msa_names, msa_seqs = fasta.read_fasta(
        msa_path["msa_paths"],
        keep_insertions=keep_insertions,
        to_upper=to_upper,
        keep_gaps=keep_gaps,
    )
    assert len(set(msa_names)) == 1
    assert msa_name in msa_names

    data_utils.sample_to_max_tokens(
        msa_seqs,
        seed=seed,
        keep_first=True,
        drop_first=False,
        max_tokens=max_msa_tokens,
    )

    sample = {
        "MSA": msa_seqs,
        "completion_ids": tokenized_eval_seqs,
        "family_id": msa_name,
    }
    if use_seq_pos:
        sample["completion_seq_pos"] = completion_seq_pos

    sample = tokenize(
        sample,
        tokenizer=tokenizer,
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
        mutant_bos_token="sep",
    )

    sample["family_labels"] = [1 if s == msa_name else 0 for s in eval_names]
    sample["ds_name"] = "pfam"
    return sample


def load_pfam_classification_dataset(
    tokenizer,
    keep_insertions,
    to_upper,
    keep_gaps,
    pfam_dir="../data/pfam/pfam_eval_splits/clustered_split_fastas/debug_subsample",
    max_tokens=10000,
    use_seq_pos=False,
    max_seq_pos: int = 1024,
    seed=42,
    num_workers=4,
    max_eval_per_fam=4,
):
    combined_eval_seqs = []
    eval_labels = []

    eval_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_test.fasta"))
    max_eval_len = 0
    for eval_path in eval_seq_paths:
        eval_name = eval_path.split("/")[-1].split("_")[0]
        eval_names, eval_seqs = fasta.read_fasta(eval_path)
        assert len(set(eval_names)) == 1
        assert eval_name in eval_names
        combined_eval_seqs.extend(eval_seqs[:max_eval_per_fam])
        eval_labels.extend(eval_names[:max_eval_per_fam])
        # todo: current form always counts gaps towards limit
        max_eval_len = max(max_eval_len, max(map(len, eval_seqs)))

    # Tokenize eval sequences once and re-use
    tok_eval_seqs = tokenize_completions(
        sample={"completion_seqs": combined_eval_seqs},
        tokenizer=tokenizer,
        bos_token="sep",
    )
    max_msa_tokens = max_tokens - max_eval_len - 2
    if use_seq_pos:
        n_seqs, longest = tok_eval_seqs["completion_ids"].shape
        completion_seq_pos = arange(-1, longest - 1).repeat(n_seqs, 1)
        assert completion_seq_pos.shape == tok_eval_seqs["completion_ids"].shape
        completion_seq_pos[:, 0] = 0
        completion_seq_pos.clamp_(max=max_seq_pos)
    tokenized_eval_seqs = tok_eval_seqs["completion_ids"]
    # tokenized_eval_seqs = tokenize_eval_seqs(tokenizer, combined_eval_seqs, use_seq_pos, max_seq_pos)
    msa_paths = sorted(glob.glob(f"{pfam_dir}/*_train.fasta"))

    process_func = partial(
        tokenize_pfam,
        eval_names=eval_labels,
        tokenized_eval_seqs=tokenized_eval_seqs,
        tokenizer=tokenizer,
        max_msa_tokens=max_msa_tokens,
        use_seq_pos=use_seq_pos,
        max_seq_pos=max_seq_pos,
        completion_seq_pos=completion_seq_pos,
        keep_insertions=keep_insertions,
        to_upper=to_upper,
        keep_gaps=keep_gaps,
        seed=seed,
    )

    # Create dataset and apply processing
    dataset = Dataset.from_dict({"msa_paths": msa_paths})
    dataset = dataset.map(
        process_func,
        remove_columns=["msa_paths"],
        batched=False,
        # batch_size=10,  # Adjust this based on your memory constraints
        num_proc=num_workers or None,
    )

    columns = ["input_ids", "completion_ids", "family_labels", "ds_name"]
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset
