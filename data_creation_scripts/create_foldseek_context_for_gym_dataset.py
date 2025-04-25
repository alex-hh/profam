"""
Created by Jude Wells 2025-04-25


We want to test if poor family model performance on ProteinGym
is caused by change in distribution between training families
and the gym MSAs

To test if this is the problem we create a new dataset for protein
gym where the evolutionary context is taken from the FoldSeek families.

Build a parquet file for protein Gym each row contains one DMS experiment

Columns are:
DMS_id
completion_sequences
context_sequences
target_sequence
DMS_scores

"""
import glob
import os
import time
from typing import Iterable, List, Dict, Union

import numpy as np
import pandas as pd
import requests


def convert_uniprot_ids_to_uniprot_accessions(uniprot_ids):
    """Convert a collection of UniProt identifiers to primary UniProt accessions.

    The *ProteinGym* dataset stores protein identifiers either as
    *entry names* (e.g. ``TP53_HUMAN``) **or** already as accessions
    (e.g. ``P04637``).  Down-stream processing relies on having the
    canonical accession, so this helper resolves any entry names to
    their corresponding primary accession via the UniProt REST API
    (``https://rest.uniprot.org``).

    Parameters
    ----------
    uniprot_ids : Iterable[str]
        A list/series/iterable containing UniProt entry names or
        accessions.

    Returns
    -------
    List[str]
        A list of the same length as *uniprot_ids* where every element
        is the resolved accession (or *None* if the lookup failed).
    """

    def _looks_like_accession(uid: str) -> bool:
        """Heuristic check if *uid* already appears to be an accession."""
        # Typical accessions have length 6 (e.g. P04637) or 10 (e.g. A0A023GPI8)
        return len(uid) in (6, 10) and uid[0].isalpha() and uid[-1].isdigit()

    if isinstance(uniprot_ids, (pd.Series, list, tuple, set, np.ndarray)):
        ids_iterable: Iterable[str] = uniprot_ids
    else:
        raise TypeError("uniprot_ids should be a pandas Series, list, tuple, or set of strings")

    resolved_accessions: List[Union[str, None]] = []
    cache: Dict[str, Union[str, None]] = {}

    for uid in ids_iterable:
        # Re-use previous look-ups to avoid redundant API calls
        if uid in cache:
            resolved_accessions.append(cache[uid])
            continue

        if _looks_like_accession(uid):
            accession: Union[str, None] = uid  # Already an accession
        else:
            # Query UniProt REST API for this entry name
            url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    accession = r.json().get("primaryAccession")
                else:
                    accession = None
            except requests.RequestException:
                accession = None

            # Be gentle with the public UniProt server – small sleep to avoid
            # triggering rate limits when large batches are processed.
            time.sleep(0.1)

        cache[uid] = accession
        resolved_accessions.append(accession)

    return resolved_accessions

def get_all_uniprot_accessions_from_protein_gym_dataset(gym_df):
    uniprot_ids = gym_df["UniProt_ID"].unique()
    return convert_uniprot_ids_to_uniprot_accessions(uniprot_ids)

def _align_foldseek_family_with_gym_target_sequence_mafft_deprecated(*args, **kwargs):
    """DEPRECATED – kept for reference only. Use the BLASTP-based implementation
    defined below (`align_foldseek_family_with_gym_target_sequence`)."""
    raise NotImplementedError("MAFFT-based alignment has been deprecated. Use BLASTP implementation.")

def align_foldseek_family_with_gym_target_sequence(foldseek_family_seq, gym_target_seq, threads: int = 1,
                                                   identity_threshold: float = 0.15,
                                                   coverage_threshold: float = 0.70,
                                                   measure_time: bool = True):
    """Align *each* FoldSeek family sequence individually to the ProteinGym
    target using the *BLAST+* `blastp` command-line tool.  In contrast to the
    MAFFT-based approach (which performs a multiple alignment), this function
    executes pair-wise alignments in a *single* BLAST call where the target is
    provided via the `-subject` argument and the family sequences are the
    *queries*.

    Sequences that meet the same **15 % identity** and **70 % coverage**
    thresholds (coverage measured on the *target*) are kept, and we trim the
    query to the reported alignment coordinates (i.e. we remove N/C-terminal
    overhangs that are not aligned).

    Parameters
    ----------
    foldseek_family_seq : List[str] | np.ndarray
        Sequences of the FoldSeek family.
    gym_target_seq : str
        Target sequence of the DMS experiment.
    threads : int, default=1
        Number of CPU threads for BLAST.
    identity_threshold : float, default=0.15
        Minimum sequence identity to **retain** a sequence.
    coverage_threshold : float, default=0.70
        Minimum target coverage to **retain** a sequence.
    measure_time : bool, default=True
        If *True*, the runtime of this function is returned alongside the
        sequences.

    Returns
    -------
    List[str] | Tuple[List[str], float]
        Trimmed sequences that pass the thresholds.  If *measure_time* is
        *True* the return value is a *(sequences, seconds)* tuple, otherwise
        only the sequence list is returned.
    """

    import tempfile
    import subprocess
    import time

    # Early exit for empty families -------------------------------------------------
    if foldseek_family_seq is None or len(foldseek_family_seq) == 0:
        return ([], [], 0.0) if measure_time else ([], [])

    start_t = time.perf_counter()

    # Ensure we work with a list of str ------------------------------------------------
    if isinstance(foldseek_family_seq, np.ndarray):
        queries = foldseek_family_seq.tolist()
    else:
        queries = list(foldseek_family_seq)

    with tempfile.TemporaryDirectory() as tmpdir:
        query_fa = os.path.join(tmpdir, "queries.fa")
        subject_fa = os.path.join(tmpdir, "target.fa")
        blast_out = os.path.join(tmpdir, "blast.tsv")

        # Write fasta files ----------------------------------------------------------
        with open(subject_fa, "w") as fh:
            fh.write(">target\n" + gym_target_seq + "\n")

        with open(query_fa, "w") as fh:
            for i, seq in enumerate(queries):
                fh.write(f">seq_{i}\n{seq}\n")

        # Build BLAST command ---------------------------------------------------------
        # We ask for a custom tabular output (outfmt 6) that includes the fields
        # we need:   qseqid nident length pident qlen slen qstart qend sstart send
        outfmt_cols = "6 qseqid nident length pident qlen slen qstart qend sstart send qseq sseq"

        cmd = [
            "blastp",
            "-query", query_fa,
            "-subject", subject_fa,
            "-outfmt", outfmt_cols,
            "-max_hsps", "1",        # best HSP per query-subject pair
            "-max_target_seqs", "1",  # we only have one subject
            "-num_threads", str(threads),
            "-evalue", "0.05",         # essentially no e-value cutoff
        ]

        with open(blast_out, "w") as bout:
            subprocess.run(cmd, check=True, stdout=bout)

        # Parse BLAST tabular output --------------------------------------------------
        kept_sequences: List[str] = []  # ungapped, trimmed sequences
        aligned_strings: List[str] = []  # alignment strings (same length as target) using '.' for gaps
        # Map query label -> original sequence for fast lookup
        label_to_seq = {f"seq_{i}": seq for i, seq in enumerate(queries)}

        with open(blast_out) as fh:
            for line in fh:
                if not line.strip():
                    continue
                (qseqid, nident, alen, pident, qlen, slen,
                 qstart, qend, sstart, send, qseq_aln, sseq_aln) = line.strip().split("\t")

                nident = int(nident)
                alen = int(alen)
                pident = float(pident)
                qstart = int(qstart)
                qend = int(qend)
                slen = int(slen)

                identity = nident / alen if alen else 0.0
                coverage = alen / slen if slen else 0.0

                if identity < identity_threshold or coverage < coverage_threshold:
                    continue

                # Extract trimmed (aligned) region from the original query.
                # qstart/qend are inclusive 1-based coordinates.
                start_idx = min(qstart, qend) - 1  # python 0-based
                end_idx = max(qstart, qend)        # slice end is exclusive
                full_seq = label_to_seq[qseqid]
                trimmed_seq = full_seq[start_idx:end_idx]

                kept_sequences.append(trimmed_seq)

                # Build alignment string aligned to the *full* target length
                target_len = len(gym_target_seq)
                ali_arr = ['.'] * target_len

                # subject start position (1-based from BLAST) may be > end if alignment is reverse, handle both
                subj_pos = min(int(sstart), int(send)) - 1  # 0-based index into target

                # Iterate through aligned block
                for s_char, q_char in zip(sseq_aln, qseq_aln):
                    if s_char == '-':
                        # insertion w.r.t. subject – skip (do not advance subject position)
                        continue
                    # subject has residue; map onto current subject position
                    if q_char == '-':
                        ali_arr[subj_pos] = '.'  # deletion in query
                    else:
                        ali_arr[subj_pos] = q_char
                    subj_pos += 1

                aligned_strings.append(''.join(ali_arr))

    elapsed = time.perf_counter() - start_t
    if measure_time:
        return kept_sequences, aligned_strings, elapsed
    else:
        return kept_sequences, aligned_strings

if __name__ == "__main__":
    foldseek_parquet_pattern = "../data/foldseek_s50_seq_only/train_val_test_split/*/*.parquet"
    gym_df_path = "../data/ProteinGym/DMS_substitutions.csv"
    completion_seq_dir = "../data/ProteinGym/DMS_ProteinGym_substitutions"
    gym_df = pd.read_csv(gym_df_path)

    # ------------------------------------------------------------------
    # Resolve UniProt identifiers to canonical accessions and augment the
    # ProteinGym dataframe with a new column. Persist the augmented file
    # so that downstream scripts do not need to resolve again.
    # ------------------------------------------------------------------
    if "UniProt_Accession" not in gym_df.columns:   
        gym_df["UniProt_Accession"] = pd.Series(
            convert_uniprot_ids_to_uniprot_accessions(gym_df["UniProt_ID"])
        )
        gym_df.to_csv(gym_df_path, index=False)

    # For matching with Pfam data we only need the unique set of accessions
    all_gym_accessions = set(gym_df["UniProt_Accession"].dropna().unique())

    foldseek_parquets = glob.glob(foldseek_parquet_pattern)
    matched_foldseek_rows = {}
    new_gym_df_rows = []
    for foldseek_parquet in foldseek_parquets:
        df = pd.read_parquet(foldseek_parquet)
        for i, row in df.iterrows():
            accessions = row.accessions
            matched_accessions = set(accessions).intersection(all_gym_accessions)
            if matched_accessions:
                matched_foldseek_rows[row["fam_id"]] = row
                for accession in matched_accessions:
                    gym_matched = gym_df[gym_df.UniProt_Accession == accession]
                    for _, gym_row in gym_matched.iterrows():
                        print(f">{gym_row.DMS_id}\n{gym_row.target_seq}")
                        print(f">{row.fam_id}\n{row.sequences[0]}\n\n")
                        dms_filepath = os.path.join(completion_seq_dir, gym_row.DMS_filename)
                        dms_data = pd.read_csv(dms_filepath)
                        new_row = {
                            "DMS_id": gym_row.DMS_id,
                            "target_seq": gym_row.target_seq,
                            "gym_accession": accession,
                            "foldseek_fam_id": row["fam_id"],
                            "completion_sequences": dms_data.mutated_sequence.values,
                            "target_sequence": gym_row.target_seq,
                            "DMS_scores": dms_data.DMS_score.values
                        }
                        # Align FoldSeek family against the ProteinGym target using BLASTP
                        aligned_foldseek_sequences, aligned_strings, aligned_time = align_foldseek_family_with_gym_target_sequence(
                            row.sequences, gym_row.target_seq, threads=8, measure_time=True
                        )

                        # Store ungapped aligned sequences as context
                        new_row["context_sequences"] = aligned_foldseek_sequences

                        # ------------------------------------------------------------------
                        # Save MSA (.a2m) file comprising target + aligned sequences
                        # ------------------------------------------------------------------
                        msa_dir = os.path.join("..", "data", "ProteinGym", "foldseek_s50_DMS_msa_files")
                        os.makedirs(msa_dir, exist_ok=True)
                        msa_path = os.path.join(msa_dir, f"{gym_row.DMS_id}.a2m")

                        with open(msa_path, "w") as msa_f:
                            # First write the target sequence (reference) – no gaps
                            msa_f.write(f">{gym_row.DMS_id}_target\n{gym_row.target_seq}\n")

                            # Write each aligned family member – use '.' for gaps
                            for idx, ali_seq in enumerate(aligned_strings):
                                msa_f.write(f">seq_{idx}\n{ali_seq}\n")

                        # Persist path for downstream reference if needed
                        new_row["msa_file"] = msa_path
                        new_gym_df_rows.append(new_row)
