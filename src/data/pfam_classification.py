"""
create evaluation datasets
for family classification
which contains:
- input_ids (prompt tokens)
- completion_ids (tokens of seqs to predict)
- family_labels (binary in-family or not)
- sequence_positions (if True)
- eval_fam_ids, str of all eval family ids:
            PF00167.18|PF00112.12|PF... this is not necessary but is used for debugging
- family_id, str of the family id for the prompt


for pfam we evaluate likelihood of sequence
conditioning on prompts from different families
likelihoods for different prompts are logits
in a multi-class classification problem

to benefit from kv-caching we evaluate
all seqs against one msa at a time
"""

import glob
import warnings
from functools import partial
from typing import Dict, List, Optional

import torch
from datasets import Dataset

from src.data.fasta import convert_sequence_with_positions, read_fasta
from src.data.objects import ProteinDocument
from src.data.preprocessing import FastaPreprocessorConfig, preprocess_protein_sequences
from src.utils.tokenizers import ProFamTokenizer


def tokenize_eval_seqs(
    proteins: ProteinDocument,
    tokenizer: ProFamTokenizer,
):
    eval_seq_ids = [
        tokenizer(tokenizer.sep_token + s + tokenizer.sep_token, return_tensors="pt")
        for s in proteins.sequences
    ]
    return eval_seq_ids


def tokenize_pfam_prompt(proteins: ProteinDocument, tokenizer: ProFamTokenizer):
    return tokenizer(tokenizer.sep_token.join(proteins.sequences), return_tensors="pt")


def prep_pfam_sample(
    msa_path: str,
    eval_names: List[str],
    tokenized_eval_seqs: List[List[int]],
    completion_seq_pos: List[List[int]],
    cfg: FastaPreprocessorConfig,
    tokenizer: ProFamTokenizer,
    seed=42,
):
    msa_name = msa_path["msa_paths"].split("/")[-1].split("_")[0]
    msa_names, msa_seqs = read_fasta(
        msa_path["msa_paths"],
        keep_insertions=cfg.keep_insertions,
        to_upper=cfg.to_upper,
        keep_gaps=cfg.keep_gaps,
    )

    msa_proteins = ProteinDocument(sequences=msa_seqs)
    msa_proteins = preprocess_protein_sequences(
        msa_proteins,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    msa_tokenized = tokenize_pfam_prompt(msa_proteins)

    sample = {
        "input_ids": msa_tokenized["input_ids"],
        "seq_pos": msa_proteins.positions,  #  todo this is wrong
        "completion_ids": tokenized_eval_seqs,
        "family_id": msa_name,
    }
    if cfg.use_seq_pos:
        sample["seq_pos"] = msa_proteins["seq_pos"]
        sample["completion_seq_pos"] = completion_seq_pos
    eval_fam_names = [n.split("_")[-1] for n in eval_names]
    sample["family_labels"] = torch.tensor(
        [1 if s == msa_name else 0 for s in eval_fam_names]
    )
    assert sample["family_labels"].sum() > 0  # at least one eval seq is in the fam
    sample["ds_name"] = "pfam"
    sample["eval_fam_ids"] = "|".join(eval_fam_names)
    return sample


def load_pfam_classification_dataset(
    tokenizer: ProFamTokenizer,
    keep_insertions: bool,
    to_upper: bool,
    keep_gaps: bool,
    pfam_dir: str,
    max_tokens: int = 10000,
    use_seq_pos: bool = False,
    max_seq_pos: int = 1024,
    use_msa_pos: bool = False,
    seed: int = 42,
    num_workers: int = 4,
    max_eval_per_fam: int = 4,
    document_token: str = "[MSA]",
):
    eval_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_test.fasta"))
    prompt_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_train.fasta"))
    assert len(eval_seq_paths) == len(prompt_seq_paths)

    train_names = set([p.split("/")[-1].split("_train.")[0] for p in prompt_seq_paths])
    test_names = set([p.split("/")[-1].split("_test.")[0] for p in eval_seq_paths])
    assert train_names == test_names, "Pfam prompt and completion files do not match."

    seq_load_func = partial(
        read_fasta,
        keep_insertions=True,
        keep_gaps=True,
        to_upper=False,
    )

    combined_eval_seqs = []
    eval_labels = []

    for eval_path in eval_seq_paths:
        eval_names, eval_seqs = seq_load_func(eval_path)
        combined_eval_seqs.extend(eval_seqs[:max_eval_per_fam])
        eval_labels.extend(eval_names[:max_eval_per_fam])

    eval_proteins = ProteinDocument(
        sequences=combined_eval_seqs,
        accessions=eval_labels,
        identifier=None,  # what should go here?
    )

    cfg = FastaPreprocessorConfig(
        preprocessor="",
        keep_insertions=keep_insertions,
        to_upper=to_upper,
        keep_gaps=keep_gaps,
        document_token=document_token,
        truncate_after_n_sequences=None,
        use_msa_pos=use_msa_pos,  # for msa sequences, if true, position index will be relative to alignment cols
        transforms=None,
        keep_columns=None,
        return_all_fields=False,
        allow_unk=False,
    )

    # add seq_pos and transform raw seqs:
    eval_proteins = preprocess_protein_sequences(
        eval_proteins,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    eval_proteins.positions = [[0] + pos + [0] for pos in eval_proteins.positions]
    tokenized_eval_seqs = tokenize_eval_seqs(eval_proteins, tokenizer)
    longest_eval_seq = max(
        [len(single_seq.input_ids[0]) for single_seq in tokenized_eval_seqs]
    )
    max_msa_tokens = max_tokens - longest_eval_seq - 2
    assert all(
        [
            single_seq.input_ids[0][0] == tokenizer.sep_token_id
            for single_seq in tokenized_eval_seqs
        ]
    )
    assert all(
        [
            single_seq.input_ids[0][-1] == tokenizer.sep_token_id
            for single_seq in tokenized_eval_seqs
        ]
    )
    process_func = partial(
        prep_pfam_sample,
        eval_names=eval_labels,
        tokenized_eval_seqs=tokenized_eval_seqs,
        tokenizer=tokenizer,
        completion_seq_pos=eval_proteins.positions,
        cfg=cfg,
        seed=seed,
    )

    dataset = Dataset.from_dict({"msa_paths": prompt_seq_paths})
    dataset = dataset.map(
        process_func,
        remove_columns=["msa_paths"],
        batched=False,
        num_proc=num_workers,
    )

    columns = [
        "input_ids",
        "completion_ids",
        "family_labels",
        "family_id",
        "ds_name",
        "eval_fam_ids",
    ]
    if use_seq_pos:
        columns += ["seq_pos", "completion_seq_pos"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset
