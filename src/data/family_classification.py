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

from src.data import fasta
from src.data import utils as data_utils
from src.data.proteingym import tokenize
from src.utils.tokenizers import ProFamTokenizer
from src.data.utils import tokenize

family_columns = [
    "input_ids",
    "completion_ids",
    "family_labels",
    "ds_name",
    "family_id",
    "eval_fam_ids",
]


def family_dataset_from_dict_list(dataset_list, tokenizer):
    dataset = Dataset.from_pandas(pd.DataFrame(dataset_list))
    dataset = dataset.map(  #  todo 20 lines almost identical to src/data/proteingym.py
        partial(
            tokenize,
            tokenizer=tokenizer,
            mutant_bos_token="sep",  # todo check this
            document_token="[RAW]",
        ),
        batched=False,
        remove_columns=["MSA", "completion_seqs"],
    )
    columns = family_columns
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset


def load_classifier_dataset(
    fasta_file_pattern,
    tokenizer: ProFamTokenizer,
    max_tokens=10000,
    max_seqs_to_predict=10,
    num_decoys_per_target=5,
    seed=42,
):
    paths = sorted(glob.glob(fasta_file_pattern))
    random.seed(seed)
    dataset_list = []

    for target_family_path in paths:
        # Load sequences from the target family
        _, seqs = fasta.read_fasta(target_family_path)
        target_fam_id = (
            target_family_path.split("/")[-1].replace(".fasta", "").replace(".fa", "")
        )
        target_family_seqs = seqs.copy()
        n_targ_seqs = min(len(target_family_seqs) // 2, max_seqs_to_predict)
        n_targ_seqs = min(n_targ_seqs, len(target_family_seqs))
        target_seqs = random.sample(target_family_seqs, n_targ_seqs)
        remaining_seqs = list(set(target_family_seqs) - set(target_seqs))
        decoy_seqs = []
        target_fam_ids = [target_fam_id] * len(target_seqs)
        decoy_fam_ids = []
        for decoy_family_path in paths:
            fam_id = (
                decoy_family_path.split("/")[-1]
                .replace(".fasta", "")
                .replace(".fa", "")
            )
            if decoy_family_path == target_family_path:
                continue
            # Load sequences from the decoy family
            _, seqs = fasta.read_fasta(decoy_family_path)
            n_samples = min(len(seqs), num_decoys_per_target)
            # Sample sequences for the decoys
            decoy_seqs.extend(random.sample(seqs, n_samples))
            decoy_fam_ids.extend([fam_id] * n_samples)
        completion_seqs = target_seqs + decoy_seqs
        # save space for completions
        max_tokens_for_prompt = max_tokens - max([len(s) for s in completion_seqs]) - 2
        msa_seqs = data_utils.sample_to_max_tokens(
            remaining_seqs, seed=seed, max_tokens=max_tokens_for_prompt
        )
        assert (
            len(msa_seqs) > 0
        ), f"No msa seqs sampled for family classification, check max tokens {max_tokens_for_prompt}"

        family_labels = [1] * len(target_seqs) + [0] * len(decoy_seqs)

        # Create a dictionary for the current target family
        family_dict = {
            "MSA": msa_seqs,
            "completion_seqs": completion_seqs,
            "family_labels": family_labels,
            "ds_name": "ec_class",
            "family_id": target_fam_id,
            "eval_fam_ids": "|".join(target_fam_ids + decoy_fam_ids),
        }
        dataset_list.append(family_dict)

    dataset = family_dataset_from_dict_list(dataset_list, tokenizer)
    return dataset


def get_prompt_from_ec_num(
    ec_num: str,
    fasta_dir: str,
    exclusion_ids: list[str],
):
    ec_fasta_path = f"{fasta_dir}/{ec_num.replace('.', '_')}.fasta"
    ids, seqs = fasta.read_fasta(
        ec_fasta_path,
        keep_insertions=True,
        keep_gaps=False,  # currently no gaps in fasta: if this changes beware eval seqs are not aligned
        to_upper=False,
    )
    ids = [id.split("|")[1] for id in ids]
    assert set(ids).intersection(
        set(exclusion_ids)
    )  # at least one of the eval seqs should be in the fam
    assert set(ids) - set(
        exclusion_ids
    )  # at least one of the eval seqs should NOT be in the fam
    keep_idxs = [i for i, id in enumerate(ids) if id not in exclusion_ids]
    ids = [ids[i] for i in keep_idxs]
    seqs = [seqs[i] for i in keep_idxs]
    return ids, seqs


def load_ec_cluster_classifier_dataset(
    tokenizer: ProFamTokenizer,
    fasta_dir: str = "../data/ec/ec_fastas",
    val_df_path: str = "data/val/ec_val_clustered_seqs_w_different_ec_nums.csv",
    max_tokens=10000,
    seed=42,
):
    """
    classifies sequences as member of family
    or not where the eval sequences are from
    the same seq-similarity cluster.
    iterate through each cluster and create an
    eval sample for each EC number in the cluster
    """
    val_df = pd.read_csv(val_df_path)
    dataset_list = []
    for c_id in val_df.val_cluster_id.unique():
        c = val_df[val_df.val_cluster_id == c_id]
        completion_seqs = c.Sequence.values
        eval_ids = c.Entry.values
        eval_ecs = c["EC number"].values
        for ec_num in eval_ecs:
            ids, prompt_seqs = get_prompt_from_ec_num(
                ec_num, fasta_dir=fasta_dir, exclusion_ids=eval_ids
            )
            max_tokens_for_prompt = (
                max_tokens - max([len(s) for s in completion_seqs]) - 2
            )
            prompt_seqs = data_utils.sample_to_max_tokens(
                prompt_seqs, seed=seed, max_tokens=max_tokens_for_prompt
            )
            labels = [1 if eval_ecs[i] == ec_num else 0 for i in range(len(eval_ecs))]
            family_dict = {
                "MSA": prompt_seqs,
                "completion_seqs": completion_seqs,
                "family_labels": labels,
                "family_id": ec_num,
                "eval_fam_ids": "|".join(eval_ecs),
                "eval_cluster_level": c.val_cluster_level.values.min(),  # eval seqs share this level of similarity
                "eval_sim_min_max": c.val_cluster_min_max.values.min(),  # min & max sim with any other EC sequence
                "ds_name": "ec_cluster_class",
            }
            dataset_list.append(family_dict)
    dataset = family_dataset_from_dict_list(
        dataset_list,
        tokenizer,
    )
    return dataset


if __name__ == "__main__":
    load_ec_cluster_classifier_dataset()
