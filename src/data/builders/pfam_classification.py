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
import copy
import glob
import os
from functools import partial
from typing import List

import torch
from datasets import Dataset

from src.data.objects import ProteinDocument
from src.data.processors import PreprocessingConfig, preprocess_protein_sequences
from src.data.tokenizers import ProFamTokenizer
from src.sequence import fasta


def pfam_sample_from_msa_path(
    msa_path: str,
    tokenized_eval_seqs,
    eval_names: List[str],
    tokenizer: ProFamTokenizer,
    max_tokens: int,
    cfg: PreprocessingConfig,
):
    raise NotImplementedError("Was originally using FastaPreprocessorConfig")
    msa_cfg = copy.deepcopy(cfg)
    msa_cfg.add_final_sep = False
    with open(msa_path["msa_paths"], "r") as file:
        fasta_file_contents = file.read()
    example = {
        "text": fasta_file_contents,
    }
    tokenized_msa = preprocess_fasta_data(
        example,
        msa_cfg,
        tokenizer,
        max_tokens,
        shuffle=True,
    )
    tokenized_msa["completion_ids"] = tokenized_eval_seqs.input_ids
    if tokenizer.embed_residue_index:
        tokenized_msa["completion_residue_index"] = tokenized_eval_seqs.residue_index
    fam_id = msa_path["msa_paths"].split("/")[-1].split("_")[0]
    tokenized_msa["family_id"] = fam_id
    eval_fam_names = [n.split("_")[-1] for n in eval_names]
    tokenized_msa["family_labels"] = torch.tensor(
        [1 if s == fam_id else 0 for s in eval_fam_names]
    )
    assert tokenized_msa["family_labels"].sum() > 0, f"no eval seq for {fam_id}"
    tokenized_msa["ds_name"] = "pfam_fam_class"
    tokenized_msa["eval_fam_ids"] = "|".join(eval_fam_names)
    return tokenized_msa


def load_pfam_classification_dataset(
    tokenizer: ProFamTokenizer,
    keep_insertions: bool,
    to_upper: bool,
    keep_gaps: bool,
    pfam_dir: str,
    max_tokens: int = 10000,
    num_workers: int = 4,
    max_eval_per_fam: int = 4,
    use_msa_pos: bool = True,
):
    raise NotImplementedError("need updating")
    if not os.path.exists(pfam_dir):
        zip_path = f"{pfam_dir}/../../pfam_val_test_fastas.zip"
        raise FileNotFoundError(
            f"Decompress the following file to get the Pfam fastas: {zip_path}"
        )  # todo decompress automatically
    eval_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_test.fasta"))
    prompt_seq_paths = sorted(glob.glob(f"{pfam_dir}/*_train.fasta"))
    assert len(eval_seq_paths) == len(prompt_seq_paths)

    train_names = set([p.split("/")[-1].split("_train.")[0] for p in prompt_seq_paths])
    test_names = set([p.split("/")[-1].split("_test.")[0] for p in eval_seq_paths])
    assert train_names == test_names, "Pfam prompt and completion files do not match."

    seq_load_func = partial(
        fasta.read_fasta,
        keep_insertions=True,
        keep_gaps=True,
        to_upper=False,
    )

    combined_eval_seqs = []
    eval_labels = []
    # load eval seqs from all the eval families
    for eval_path in eval_seq_paths:
        eval_names, eval_seqs = seq_load_func(eval_path)
        combined_eval_seqs.extend(eval_seqs[:max_eval_per_fam])
        eval_labels.extend(eval_names[:max_eval_per_fam])
    # sort by length for efficient batching
    combined_eval_seqs = sorted(combined_eval_seqs, key=lambda x: len(x))
    eval_proteins = ProteinDocument(
        sequences=combined_eval_seqs,
        accessions=eval_labels,
        identifier=None,  # what should go here?
    )

    if keep_gaps:
        document_token = "[MSA]"
    else:
        if use_msa_pos:
            document_token = "[RAW-WITH-MSA-POS]"
        else:
            document_token = "[RAW]"
    cfg = FastaPreprocessorConfig(
        keep_insertions=keep_insertions,
        to_upper=to_upper,
        keep_gaps=keep_gaps,
        document_token=document_token,
        truncate_after_n_sequences=None,
        use_msa_pos=use_msa_pos,  # for msa sequences, if true, position index will be relative to alignment cols
        transforms=None,
        keep_columns=None,
        allow_unk=False,
    )

    # add res_pos and transform raw seqs:
    eval_proteins = preprocess_protein_sequences(
        eval_proteins,
        cfg=cfg,
        tokenizer=tokenizer,
    )

    tokenized_eval_seqs = tokenizer.encode_completions(
        sequences=eval_proteins.sequences,
        residue_positions=eval_proteins.residue_positions,
        bos_token=tokenizer.sep_token,
        eos_token=tokenizer.sep_token,
    )

    longest_eval_seq = tokenized_eval_seqs.input_ids.shape[-1]
    max_msa_tokens = max_tokens - longest_eval_seq - 2
    process_func = partial(
        pfam_sample_from_msa_path,
        tokenized_eval_seqs=tokenized_eval_seqs,
        eval_names=eval_labels,
        tokenizer=tokenizer,
        max_tokens=max_msa_tokens,
        cfg=cfg,
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
    if tokenizer.embed_residue_index:
        columns += ["residue_index", "completion_residue_index"]

    dataset.set_format(
        type="torch",
        columns=columns,
    )
    return dataset
