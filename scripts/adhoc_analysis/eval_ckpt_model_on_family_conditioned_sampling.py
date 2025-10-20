"""
Created by Jude Wells 2025-04-30

Takes a family and adds mutations with random probability
checks the TM score and other metrics.

For each sequence apply mutations with:
with the following probabilities (on each position):
2%
5%
7%
10%
15%
20%
30%
40%
50%
60%
70%
80%
90%
100%
Each mutation has:
90% chance of substitution
5% chance of deletion
5% chance of insertion


"""

import argparse
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.objects import ProteinDocument
from src.data.processors import transforms as proc_transforms
from src.data.processors.preprocessing import (
    AlignedProteinPreprocessingConfig,
    ProteinDocumentPreprocessor,
)
from src.evaluators.esmfold import ESMFoldSamplingEvaluator
from src.models.inference import ProFamSampler, PromptBuilder
from src.models.llama import LlamaLitModule
from src.sequence import fasta
from src.utils.utils import get_config_from_cpt_path, seed_all



def build_prompt_builder(max_tokens: int = 20_000) -> PromptBuilder:
    """Create a PromptBuilder mimicking the raw_from_msa_add_final_sep preprocessor.

    Args:
        max_tokens: Maximum number of tokens allowed in the prompt.

    Returns:
        PromptBuilder instance ready for use with ProFamSampler.
    """
    cfg = AlignedProteinPreprocessingConfig(
        # Equivalent settings to configs/preprocessor/cfg/raw_from_msa.yaml
        document_token="[RAW]",
        keep_insertions=True,
        keep_gaps=False,
        to_upper=True,
        use_msa_pos=False,
        max_tokens_per_example=max_tokens,
        shuffle_proteins_in_document=True,
        padding="do_not_pad",
    )

    # We replicate the behaviour of raw_from_msa_add_final_sep which first applies
    # default preprocessing and finally appends an explicit [SEP] token.
    transform_fns = [
        # Add final separator token so completions always start with BOS/SEP.
        proc_transforms.add_final_sep,
    ]

    preprocessor = ProteinDocumentPreprocessor(cfg=cfg, transform_fns=transform_fns)
    return PromptBuilder(preprocessor=preprocessor)


def save_sequences_fasta(seqs: List[str], output_path: str):
    names = [f"seq{i}" for i in range(len(seqs))]
    fasta.output_fasta(names, seqs, output_path)


def compute_seq_similarity_stats(
    sample_seqs: List[str], prompt_seqs: List[str], threads: int = 1
):
    """Compute minimum, maximum **and mean** sequence identity between generated
    samples and prompt sequences.

    The function performs an all-vs-all BLASTP (query = samples, subject =
    prompt).  For every query-subject pair we take the number of identical
    matches *nident* reported by BLAST and divide by the *maximum* of the two
    full-length sequences (``max(query_len, subject_len)``).  This yields an
    identity figure that penalises N/C-terminal overhangs (i.e. residues that
    are **not** part of the aligned region) instead of merely considering the
    aligned block.

    The statistics across **all** pairs (missing alignments count as identity
    0.0) are returned.

    Parameters
    ----------
    sample_seqs : List[str]
        List of generated/sample sequences (queries).
    prompt_seqs : List[str]
        List of sequences from the prompt (subjects).
    threads : int, default=1
        Number of CPU threads provided to BLASTP.

    Returns
    -------
    Tuple[float, float, float]
        (*min_identity*, *max_identity*, *mean_identity*) in fractional units,
        e.g. 0.85 = 85 % identity.
    """
    import os
    import subprocess
    import tempfile
    from collections import defaultdict

    if not sample_seqs or not prompt_seqs:
        return float("nan"), float("nan"), float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        query_fa = os.path.join(tmpdir, "queries.fa")
        subject_fa = os.path.join(tmpdir, "subjects.fa")
        blast_out = os.path.join(tmpdir, "blast.tsv")

        # Write FASTA files ----------------------------------------------------------
        with open(query_fa, "w") as qfh:
            for i, seq in enumerate(sample_seqs):
                qfh.write(f">query_{i}\n{seq}\n")

        with open(subject_fa, "w") as sfh:
            for j, seq in enumerate(prompt_seqs):
                sfh.write(f">subject_{j}\n{seq}\n")

        # Build BLASTP command -------------------------------------------------------
        outfmt_cols = "6 qseqid sseqid nident qlen slen"
        cmd = [
            "blastp",
            "-query",
            query_fa,
            "-subject",
            subject_fa,
            "-outfmt",
            outfmt_cols,
            "-max_hsps",
            "1",
            "-seg",
            "no",
            "-num_threads",
            str(threads),
            "-evalue",
            "1000",
        ]

        # Execute BLASTP -------------------------------------------------------------
        with open(blast_out, "w") as bout:
            subprocess.run(cmd, check=True, stdout=bout)

        # Parse results --------------------------------------------------------------
        # identity_map[(qidx, sidx)] = max_identity
        identity_map = defaultdict(float)
        with open(blast_out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                qseqid, sseqid, nident_str, qlen_str, slen_str = line.strip().split(
                    "\t"
                )
                nident = int(nident_str)
                qlen = int(qlen_str)
                slen = int(slen_str)
                denom = max(qlen, slen) if max(qlen, slen) > 0 else 1
                identity = nident / denom
                # Extract numeric indices from labels
                qidx = int(qseqid.split("_")[1])
                sidx = int(sseqid.split("_")[1])
                key = (qidx, sidx)
                identity_map[key] = max(identity_map[key], identity)

        # Fill in missing pairs with 0.0 ---------------------------------------------
        identities: List[float] = []
        for qi in range(len(sample_seqs)):
            for si in range(len(prompt_seqs)):
                identities.append(identity_map.get((qi, si), 0.0))

        if not identities:
            return 0.0, 0.0, 0.0
        return min(identities), max(identities), float(np.mean(identities))


# Chance per position to apply mutation according to specified probabilities
MUTATION_PROBABILITIES = [
    0.02,
    0.05,
    0.07,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard amino acids


def mutate_sequence(seq: str, mutation_rate: float) -> str:
    """Return a mutated copy of *seq*.

    Each position is mutated with *mutation_rate* probability. A mutation is
    chosen according to: 90 % substitution, 5 % deletion, 5 % insertion.

    Substitutions are chosen uniformly at random from the 19 amino-acids that
    differ from the wild-type residue. Insertions are single random residues
    inserted *before* the current position, after which the original residue is
    kept. Deletions simply drop the residue.
    """
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if random.random() < mutation_rate:
            r = random.random()
            if r < 0.90:  # substitution
                wt = seq[i]
                choices = [aa for aa in AMINO_ACIDS if aa != wt]
                out.append(random.choice(choices))
                i += 1  # consume original residue
            elif r < 0.95:  # deletion
                i += 1  # skip residue (delete)
            else:  # insertion
                out.append(random.choice(AMINO_ACIDS))  # inserted residue
                # keep original residue afterwards
                out.append(seq[i])
                i += 1
        else:
            out.append(seq[i])
            i += 1
    return "".join(out)




def main(args):
    # Seed RNGs
    seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    dtype = torch.bfloat16

    print(f"Loading checkpoint from {args.ckpt_path}")
    config = get_config_from_cpt_path(args.ckpt_path)
    print("Loaded model config:\n", OmegaConf.to_yaml(config))

    model: LlamaLitModule = LlamaLitModule.load_from_checkpoint(args.ckpt_path)
    model.eval()
    model.to(device, dtype=dtype)

    # Disable KV cache during scoring to avoid potential leak (see Gym script).
    model.scoring_max_tokens = 1
    model.use_kv_cache_for_scoring = False

    prompt_builder = build_prompt_builder(max_tokens=args.max_tokens)

    sampling_kwargs: Dict = {
        "temperature": args.temperature,
        "greedy": False,
    }

    sampler = ProFamSampler(
        name="profam_sampler",
        model=model,
        prompt_builder=prompt_builder,
        sampling_kwargs=sampling_kwargs,
    )

    evaluator = ESMFoldSamplingEvaluator(
        name="esmfold",
        save_structures=True,
        prompt_plddt=True,
        half_precision=args.half_precision,
    )

    df = pd.read_parquet(args.parquet_path)
    print(f"Loaded parquet containing {len(df)} families from {args.parquet_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    results: List[Dict] = []

    for idx, row in df.iterrows():
        fam_id = row.get("fam_id", row.get("family_id", f"family_{idx}"))
        print(f"\nProcessing family {fam_id} ({idx+1}/{len(df)})")
        sequences: List[str] = list(row["sequences"])
        sequences = [s.replace("-", "").replace(".", "") for s in sequences]
        if len(sequences) > 1:
            sequences = sequences[:1]
        max_prompt_length = max([len(seq) for seq in sequences])
        if max_prompt_length > args.max_tokens:
            print(
                f"Skipping family {fam_id} because max prompt length ({max_prompt_length}) is greater than max tokens ({args.max_tokens})"
            )
            continue
        accessions: Optional[List[str]] = (
            list(row["accessions"]) if "accessions" in row else None
        )
        protein_doc = ProteinDocument(
            sequences=sequences,
            accessions=accessions,
            identifier=fam_id,
        )


        gen_seqs, prompt = sampler.sample_seqs(
            protein_document=protein_doc,
            num_samples=args.num_generations,
            max_tokens=args.max_tokens,
            max_generated_length=args.max_generated_length,
        )

        mutated_seqs: List[str] = []
        for _ in range(args.num_random_mutations):
            # Choose mutation rate uniformly from predefined set
            m_rate = random.choice(MUTATION_PROBABILITIES)
            # For now mutate the first/representative sequence
            mutated_seqs.append(mutate_sequence(sequences[0], m_rate))

        fam_output_dir = os.path.join(args.output_dir, fam_id)
        os.makedirs(fam_output_dir, exist_ok=True)

        # Save generated and mutated sequences to FASTA
        save_sequences_fasta(gen_seqs, os.path.join(fam_output_dir, "samples.fa"))
        save_sequences_fasta(
            mutated_seqs, os.path.join(fam_output_dir, "mutated_samples.fa")
        )

        # Evaluate generated sequences & save structures (PDBs)
        metrics = evaluator.evaluate_samples(
            prompt=prompt,
            protein_document=protein_doc,
            samples=gen_seqs,
            output_dir=fam_output_dir,
            device=device,
        )

        mut_metrics = evaluator.evaluate_samples(
            prompt=prompt,
            protein_document=protein_doc,
            samples=mutated_seqs,
            output_dir=os.path.join(fam_output_dir, "mutated"),
            device=device,
        )


        metrics["mean_sample_length"] = np.mean([len(seq) for seq in gen_seqs])
        metrics["max_sample_length"] = np.max([len(seq) for seq in gen_seqs])
        metrics["min_sample_length"] = np.min([len(seq) for seq in gen_seqs])

        metrics["mut_mean_sample_length"] = np.mean([len(seq) for seq in mutated_seqs])
        metrics["mut_max_sample_length"] = np.max([len(seq) for seq in mutated_seqs])
        metrics["mut_min_sample_length"] = np.min([len(seq) for seq in mutated_seqs])

        metrics["mean_prompt_length"] = np.mean([len(seq) for seq in sequences])
        metrics["max_prompt_length"] = max_prompt_length
        metrics["min_prompt_length"] = np.min([len(seq) for seq in sequences])
        metrics["n_seqs_in_prompt"] = len(sequences)

        for k, v in mut_metrics.items():
            metrics[f"mut_{k}"] = v


        try:
            min_id, max_id, mean_id = compute_seq_similarity_stats(
                sample_seqs=gen_seqs,
                prompt_seqs=sequences,
                threads=args.blast_threads,
            )
            metrics["min_sample_prompt_identity"] = min_id
            metrics["max_sample_prompt_identity"] = max_id
            metrics["mean_sample_prompt_identity"] = mean_id

            mut_min_id, mut_max_id, mut_mean_id = compute_seq_similarity_stats(
                sample_seqs=mutated_seqs,
                prompt_seqs=sequences,
                threads=args.blast_threads,
            )
            metrics["mut_min_sample_prompt_identity"] = mut_min_id
            metrics["mut_max_sample_prompt_identity"] = mut_max_id
            metrics["mut_mean_sample_prompt_identity"] = mut_mean_id
        except Exception as e:
            print(
                f"Warning: BLASTP similarity computation failed for family {fam_id}: {e}"
            )
            metrics["min_sample_prompt_identity"] = np.nan
            metrics["max_sample_prompt_identity"] = np.nan
            metrics["mean_sample_prompt_identity"] = np.nan
            metrics["mut_min_sample_prompt_identity"] = np.nan
            metrics["mut_max_sample_prompt_identity"] = np.nan
            metrics["mut_mean_sample_prompt_identity"] = np.nan

        # prefix mut_metrics already included

        metrics_row = {"family_id": fam_id, **metrics}
        results.append(metrics_row)

        print(
            "  Metrics: "
            + ", ".join(
                [
                    f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in metrics.items()
                ]
            )
        )

        # Proactively clear CUDA cache after each family to manage memory.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        timestamp = datetime.now().strftime("%Y%m%d_%H")
        metrics_csv = os.path.join(
            args.output_dir, f"family_sampling_metrics_{timestamp}.csv"
        )
        pd.DataFrame(results).to_csv(metrics_csv, index=False)
        print(f"\nFinished. Metrics saved to {metrics_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint with family-conditioned sampling and ESMFold structure prediction."
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="logs/train_openfold_raw/runs/2025-04-15_12-40-50-263209/checkpoints/last.ckpt",
        help="Path to checkpoint .ckpt file",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="../data/funfams/s50_parquets/train_val_test_split/val/val_000.parquet",
        help="Parquet file containing family data (sequences, accessions, fam_id)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/family_sampling_results",
        help="Directory to store outputs (PDBs, metrics)",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=3,
        help="Number of sequences to generate per family",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20_000,
        help="Maximum tokens for prompt + generations during sampling",
    )
    parser.add_argument(
        "--max_generated_length",
        type=int,
        default=350,
        help="Maximum length (in tokens) of generated sequence",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.5, help="Sampling temperature"
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run ESMFold in half precision (saves memory, requires Ampere+) ",
    )
    parser.add_argument(
        "--blast_threads",
        type=int,
        default=4,
        help="Number of CPU threads for BLASTP when computing sequence similarities",
    )
    parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Load model with bfloat16 precision (saves GPU memory)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force computation on CPU even if CUDA is available",
    )
    parser.add_argument(
        "--num_random_mutations",
        type=int,
        default=5,
        help="Number of random mutations to generate for baseline control",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )

    parsed_args = parser.parse_args()
    main(parsed_args)
