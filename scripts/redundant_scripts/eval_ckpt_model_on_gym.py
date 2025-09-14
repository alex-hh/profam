"""
Created by Jude Wells 2025-04-02
Takes a model checkpoint and evaluates it on the ProteinGym dataset.

"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.builders.proteingym import ProteinGymDataset
from src.models.llama import LlamaLitModule
from src.utils import rich_utils
from src.utils.utils import get_config_from_cpt_path

foldseek_dms_ids = [
    "A0A1I9GEU1_NEIME_Kennouche_2019",
    "ADRB2_HUMAN_Jones_2020",
    "AMFR_HUMAN_Tsuboyama_2023_4G3O",
    "BBC1_YEAST_Tsuboyama_2023_1TG0",
    "CASP7_HUMAN_Roychowdhury_2020",
    "CD19_HUMAN_Klesmith_2019_FMC_singles",
    "DLG4_RAT_McLaughlin_2012",
    "GCN4_YEAST_Staller_2018",
    "KCNE1_HUMAN_Muhammad_2023_expression",
    "KCNE1_HUMAN_Muhammad_2023_function",
    "MBD11_ARATH_Tsuboyama_2023_6ACV",
    "ODP2_GEOSE_Tsuboyama_2023_1W4G",
    "PABP_YEAST_Melamed_2013",
    "PHOT_CHLRE_Chen_2023",
    "PRKN_HUMAN_Clausen_2023",
    "RD23A_HUMAN_Tsuboyama_2023_1IFY",
    "SCN5A_HUMAN_Glazer_2019",
    "SPG2_STRSG_Tsuboyama_2023_5UBS",
    "SQSTM_MOUSE_Tsuboyama_2023_2RRU",
    "SRBS1_HUMAN_Tsuboyama_2023_2O2W",
    "THO1_YEAST_Tsuboyama_2023_2WQG",
    "TRPC_THEMA_Chan_2017",
    "VKOR1_HUMAN_Chiasson_2020_abundance",
    "VKOR1_HUMAN_Chiasson_2020_activity",
    "YAP1_HUMAN_Araya_2012",
    "YNZC_BACSU_Tsuboyama_2023_2JVD",
]


def get_alignment_metrics(query_seq, input_seq):
    """
    Calculate alignment metrics between query sequence and input sequence.

    Args:
        query_seq (str): The query sequence to align
        input_seq (str): The input sequence to align against

    Returns:
        dict: Dictionary containing:
            - sequence_identity: Percentage of identical residues in the alignment
            - completion_coverage: Percentage of query sequence covered by the alignment
            - alignment_coverage: Percentage of input sequence covered by the alignment
    """
    # Perform global alignment
    alignments = pairwise2.align.globalxx(query_seq, input_seq)
    if not alignments:
        return {
            "sequence_identity": 0.0,
            "completion_coverage": 0.0,
            "alignment_coverage": 0.0,
        }

    # Get the best alignment
    alignment = alignments[0]
    aligned_query, aligned_input = alignment[0], alignment[1]

    # Calculate sequence identity (percentage of identical residues)
    matches = sum(1 for a, b in zip(aligned_query, aligned_input) if a == b)
    total_aligned = len(aligned_query)
    sequence_identity = matches / total_aligned if total_aligned > 0 else 0.0

    # Calculate completion coverage (percentage of query sequence covered)
    query_gaps = aligned_query.count("-")
    completion_coverage = (
        (len(query_seq) - query_gaps) / len(query_seq) if len(query_seq) > 0 else 0.0
    )

    # Calculate alignment coverage (percentage of input sequence covered)
    input_gaps = aligned_input.count("-")
    alignment_coverage = (
        (len(input_seq) - input_gaps) / len(input_seq) if len(input_seq) > 0 else 0.0
    )

    return {
        "sequence_identity": sequence_identity,
        "completion_coverage": completion_coverage,
        "alignment_coverage": alignment_coverage,
    }


def get_alignment_statistics(
    tokenizer, input_ids, completion_ids, n_comparisons: int = 100
):
    """
    Calculate alignment statistics between completion sequences and input sequences.

    Args:
        tokenizer: The tokenizer used to decode sequences
        input_ids: Input sequence IDs
        completion_ids: Completion sequence IDs
        n_comparisons: Number of random comparisons to make

    Returns:
        dict: Dictionary containing alignment statistics
    """
    remove_tokens = ["[start-of-document]", "[RAW]", "[MSA]", "[RAW-WITH-MSA-POS]"]
    input_msa = tokenizer.decode(input_ids[0])
    completion_sequences = [
        tokenizer.decode(completion_ids[0, i]) for i in range(completion_ids.shape[1])
    ]
    completion_sequences = [
        seq.replace("[SEP]", "").replace(" ", "") for seq in completion_sequences
    ]

    for tok in remove_tokens:
        input_msa = input_msa.replace(tok, "")
    input_msa = input_msa.replace(" ", "")
    input_sequences = input_msa.split("[SEP]")

    # shuffle the input sequences
    random.shuffle(input_sequences)
    random.shuffle(completion_sequences)

    alignment_metrics = []
    for i in range(n_comparisons):
        query_seq = completion_sequences[i % len(completion_sequences)]
        alignment_metrics.append(
            get_alignment_metrics(query_seq, input_sequences[i % len(input_sequences)])
        )

    # Calculate means of each metric
    mean_sequence_identity = np.mean(
        [m["sequence_identity"] for m in alignment_metrics]
    )
    mean_completion_coverage = np.mean(
        [m["completion_coverage"] for m in alignment_metrics]
    )
    mean_alignment_coverage = np.mean(
        [m["alignment_coverage"] for m in alignment_metrics]
    )
    max_sequence_identity = np.max(
        [m["sequence_identity"] for m in alignment_metrics]
    )
    # Print means of each computed metric
    print(f"Mean sequence identity: {mean_sequence_identity}")
    print(f"Max sequence identity: {max_sequence_identity}")
    print(f"Mean completion coverage: {mean_completion_coverage}")
    print(f"Mean alignment coverage: {mean_alignment_coverage}")


    return {
        "mean_sequence_identity": mean_sequence_identity,
        "max_sequence_identity": max_sequence_identity,
        "mean_completion_coverage": mean_completion_coverage,
        "mean_alignment_coverage": mean_alignment_coverage,
        "n_completion_sequences": len(completion_sequences),
        "n_completion_tokens": sum(len(seq) for seq in completion_sequences),
    }


def sample_from_model(model: LlamaLitModule, n_tokens: int = 1000):
    """
    Samples n_tokens from the model with no conditioning or context.

    Args:
        model: The loaded LlamaLitModule model
        n_tokens: Number of tokens to generate

    Returns:
        The generated text as a string
    """
    for i in range(20):
        try:
            # Create a minimal input with just the beginning of document token
            input_ids = torch.tensor(
                [[model.tokenizer.bos_token_id]], device=model.device
            )

            # Create a minimal residue index tensor (required by the model)
            # The model expects residue_index to be present and start at position 2
            residue_index = torch.tensor([[2]], device=model.device)

            # Set up generation parameters
            generation_kwargs = {
                "max_new_tokens": n_tokens,
                "do_sample": True,
                "temperature": 0.5,
                "pad_token_id": model.tokenizer.pad_token_id,
                "eos_token_id": None,  # Don't stop at any particular token
            }

            # Generate unconditionally
            with torch.no_grad():
                # Use the model's generate method directly
                outputs = model.model.generate(input_ids=input_ids, **generation_kwargs)

            # Decode the generated tokens
            generated_text = model.tokenizer.decode(
                outputs[0], skip_special_tokens=False
            )

            # Print the first 100 characters of the generated text
            print(f"\n\n\n{generated_text}\n\n\n")

        except Exception as e:
            print(f"Error during sampling: {e}")
            raise


def build_protein_gym_dataloader(
    config: DictConfig,
    dms_ids: Optional[List[str]] = None,
    max_context_seqs: Optional[int] = None,
    max_context_tokens: int = 16_000,
    use_foldseek_msa: bool = False,
) -> DataLoader:
    print(
        f"Building ProteinGym dataloader with max_context_tokens={max_context_tokens} and max_context_seqs={max_context_seqs}"
    )
    dataset_builder = ProteinGymDataset(
        name="protein_gym",
        dms_ids=dms_ids,
        seed=42,
        max_mutated_sequences=None,
        mutant_bos_token="sep" if max_context_seqs != 0 else None,
        keep_gaps=False,
        use_filtered_msa=False,
        extra_tokens_per_document=2,
        use_msa_pos=False,
        num_proc=None,
        max_tokens_per_example=max_context_tokens,
        max_context_seqs=max_context_seqs,
        use_foldseek_msa=use_foldseek_msa,
    )
    dataset = dataset_builder.load(
        data_dir=config.paths.data_dir,
        world_size=1,
        verbose=False,
    )
    dataset = dataset_builder.process(
        dataset,
        tokenizer=model.tokenizer,
        feature_names=config.data.feature_names,
        pack_to_max_tokens=max_context_tokens,
    )
    return DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)


def modify_batch_for_single_seq_scoring(
    batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    start_toks = batch["input_ids"][:, :2]
    new_batch = {}
    if "ds_name" in batch:
        new_batch["ds_name"] = batch["ds_name"]
    if "DMS_scores" in batch:
        new_batch["DMS_scores"] = batch["DMS_scores"]
    if "completion_residue_index" in batch:
        new_batch["completion_residue_index"] = batch["completion_residue_index"]
    new_completion_ids = batch["completion_ids"][:, :, 1:]  # remove the sep token

    # Check dimensions and reshape if needed
    if len(start_toks.shape) != len(new_completion_ids.shape):
        # If new_completion_ids is 3D and start_toks is 2D, we need to reshape start_toks
        if len(new_completion_ids.shape) == 3 and len(start_toks.shape) == 2:
            start_toks = start_toks.expand(
                1, new_completion_ids.shape[1], start_toks.shape[-1]
            )

    new_batch["completion_ids"] = torch.cat([start_toks, new_completion_ids], dim=-1)
    new_batch["input_ids"] = None
    return new_batch


def apply_shuffle_in_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    permutation = torch.randperm(batch["completion_ids"].shape[1])
    batch["completion_ids"] = batch["completion_ids"][:, permutation]
    batch["DMS_scores"] = batch["DMS_scores"][:, permutation]
    if "input_ids" in batch and batch["input_ids"] is not None:

        batch["input_ids"] = batch["input_ids"][:, permutation]
    return batch


def limit_number_of_prompt_sequences(
    batch: Dict[str, torch.Tensor], sep_tok_id: int, n_seqs: int = 1
) -> Dict[str, torch.Tensor]:

    assert batch["input_ids"].shape[0] == 1, "Batch size must be 1"
    _, sequence_ends = torch.where(batch["input_ids"] == sep_tok_id)
    n_seqs = min(n_seqs, sequence_ends.shape[0])
    cut_off_index = sequence_ends[n_seqs - 1]
    batch["input_ids"] = batch["input_ids"][:, :cut_off_index]
    assert all(batch["input_ids"][:, -1] < 20), "Last token should be a residue"
    if "residue_index" in batch:
        batch["residue_index"] = batch["residue_index"][:, :cut_off_index]
    return batch


# -------------------------------------------------------------
# Utility for capturing metrics logged inside `validation_step_proteingym`
# -------------------------------------------------------------
# The ProteinGym validation step inside the LightningModule logs metrics via
# `self.log` but does not return them.  Here we monkey‑patch the module's
# `log` method for the duration of the call so that we can intercept the
# values of interest (mean log‑likelihood and Spearman correlation) without
# modifying the original library code.


def capture_gym_metrics(model: "LlamaLitModule", batch: Dict[str, torch.Tensor]):
    captured_metrics: Dict[str, float] = {}
    original_log_fn = model.log

    def patched_log(name, value, *args, **kwargs):  # type: ignore[override]
        # Intercept the metrics we need
        if name in {"gym/spearman", "gym/log_likelihood"}:
            # Convert tensors to Python scalars for easier serialization
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            captured_metrics[name] = float(value)
        if callable(original_log_fn):
            original_log_fn(name, value, *args, **kwargs)

    try:
        # Patch
        model.log = patched_log  # type: ignore[assignment]
        # Run the validation step – it will now call our patched logger
        model.validation_step_proteingym(batch)
    finally:
        # Always restore the original logger
        model.log = original_log_fn  # type: ignore[assignment]

    spearman = captured_metrics.get("gym/spearman")
    log_likelihood = captured_metrics.get("gym/log_likelihood")

    return spearman, log_likelihood


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train_openfold_raw/runs/no_start_of_doc_2025_04_13/checkpoints/last.ckpt"
        # default="logs/train_single_seq_ur90_1bn/runs/bubba_2025-04-02/checkpoints/last.ckpt"
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Enable shuffling of the ProteinGym dataset",
    )
    parser.add_argument(
        "--use_foldseek_msa",
        action="store_true",
        default=False,
        help="Whether to use foldseek MSA files",
    )
    parser.add_argument(
        "--limit_n_seqs",
        type=int,
        default=None,
        help="Number of prompt sequences to evaluate on",
    )
    parser.add_argument(
        "--use_dms_ids",
        action="store_true",
        default=False,
        help="Whether to use config list of DMS IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fseekS50_ur90_model_on_normal_gym_context",
        help="Directory to save the CSV file",
    )
    parser.add_argument(
        "--max_context_seqs",
        type=int,
        default=None,
        help="Maximum number of context sequences to use",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=None,
        help="Maximum length of completion sequences to use",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config_from_cpt_path(args.ckpt_path)
    model = LlamaLitModule.load_from_checkpoint(args.ckpt_path)
    model.scoring_max_tokens = 1
    model.use_kv_cache_for_scoring = (
        False  # seems to be a memory leak issue with kv cache
    )
    model.eval()
    dtype = torch.bfloat16
    model.to(device, dtype=dtype)
    # sample_from_model(model)
    if args.use_dms_ids:
        dms_ids = foldseek_dms_ids  # config.constants.gym_val_assay_list
    else:
        dms_ids = None

    dataloader = build_protein_gym_dataloader(
        config,
        dms_ids,
        args.max_context_seqs,
        max_context_tokens=7500,
        use_foldseek_msa=args.use_foldseek_msa,
    )
    # rich_utils.print_config_tree(config, resolve=True, save_to_file=False)
    print(config)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, f"model_checkpoint.txt"), "w") as f:
        f.write(args.ckpt_path)

    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(args.output_dir, f"gym_evaluation_{timestamp}.csv")

    # Define CSV headers
    csv_headers = [
        "batch_idx",
        "dataset_name",
        "mean_sequence_identity",
        "mean_completion_coverage",
        "mean_alignment_coverage",
        "n_completion_sequences",
        "n_completion_tokens",
        "log_likelihood",
        "spearman_correlation",
    ]
    results = []
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n\nProcessing batch {batch_idx}")
        if args.max_completion_length is not None:
            if batch["completion_ids"].shape[-1] > args.max_completion_length:
                print(
                    f"Skipping batch {batch_idx} because completion length is too long"
                )
                continue
        # Get alignment statistics
        alignment_stats = get_alignment_statistics(
            model.tokenizer, batch["input_ids"], batch["completion_ids"]
        )

        # Prepare batch for model evaluation
        batch_for_model = {}
        for k,v in batch.items():
            if isinstance(v, list):
                batch_for_model[k] = v
            else:
                batch_for_model[k] = v.to(device)
        if args.shuffle:
            batch_for_model = apply_shuffle_in_batch(batch_for_model)
        if args.limit_n_seqs is not None:
            batch_for_model = limit_number_of_prompt_sequences(
                batch_for_model, model.tokenizer.sep_token_id, args.limit_n_seqs
            )

        spearman, log_likelihood = capture_gym_metrics(model, batch_for_model)
        row_data = {
            "batch_idx": batch_idx,
            "dataset_name": batch.get("ds_name", "unknown"),
            "mean_sequence_identity": alignment_stats["mean_sequence_identity"],
            "mean_completion_coverage": alignment_stats["mean_completion_coverage"],
            "mean_alignment_coverage": alignment_stats["mean_alignment_coverage"],
            "n_completion_sequences": alignment_stats["n_completion_sequences"],
            "n_completion_tokens": alignment_stats["n_completion_tokens"],
            "log_likelihood": log_likelihood,
            "spearman_correlation": spearman,
        }

        results.append(row_data)
        print(
            f"Batch {batch_idx} - Log Likelihood: {row_data['log_likelihood']:.4f}, Spearman: {row_data['spearman_correlation']:.4f}"
        )
        df = pd.DataFrame(results)
        df.to_csv(csv_filename, index=False)
        print(f"Mean spearman: {df['spearman_correlation'].mean()}")
        print(f"Mean log likelihood: {df['log_likelihood'].mean()}")
    print(f"\nEvaluation complete. Results saved to {csv_filename}")
