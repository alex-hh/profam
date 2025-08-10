"""
We note that sometimes the first sequence in the proteinGym MSAs is not really the WT of the DMS assay
so we now include the WT sequence from the DMS assay in the proteinGym MSAs
and recompute the alignment and then calculate the coverage and sequence similarity
for EACH sequence in the proteinGym MSAs.

and then we do the same for the PoET MSAs.

Index(['DMS_id', 'DMS_filename', 'UniProt_ID', 'taxon', 'source_organism',
       'target_seq', 'seq_len', 'includes_multiple_mutants',
       'DMS_total_number_mutants', 'DMS_number_single_mutants',
       'DMS_number_multiple_mutants', 'DMS_binarization_cutoff',
       'DMS_binarization_method', 'first_author', 'title', 'year', 'jo',
       'region_mutated', 'molecule_name', 'selection_assay', 'selection_type',
       'MSA_filename', 'MSA_start', 'MSA_end', 'MSA_len', 'MSA_bitscore',
       'MSA_theta', 'MSA_num_seqs', 'MSA_perc_cov', 'MSA_num_cov', 'MSA_N_eff',
       'MSA_Neff_L', 'MSA_Neff_L_category', 'MSA_num_significant',
       'MSA_num_significant_L', 'raw_DMS_filename', 'raw_DMS_phenotype_name',
       'raw_DMS_directionality', 'raw_DMS_mutant_column', 'weight_file_name',
       'pdb_file', 'ProteinGym_version', 'raw_mut_offset',
       'coarse_selection_type'],
      dtype='object')
"""
import glob
import os

import numpy as np
import pandas as pd

from scripts.adhoc_analysis.generate_cluster_alignments_and_logos import run_alignment_with_mafft, create_logo_from_fasta, write_fasta
from src.data.builders.proteingym import load_msa_for_row
from src.sequence import fasta


def compute_sequence_similarity(seq1: str, seq2: str) -> float:
    """Compute sequence similarity between two sequences."""
    seq_len = max(len(seq1.replace('-', '')), len(seq2.replace('-', '')))
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b and a != "-")
    return matches / seq_len if seq_len > 0 else 0.0

def compute_coverage(wt_seq, aligned_seq):
    len_wt = len(wt_seq.replace("-", ""))
    len_aligned = len(aligned_seq.replace("-", ""))
    return len_aligned / len_wt


def compute_coverage_and_similarity_for_all_seqs(wt_seq, aligned_seqs):
    """
    Compute coverage and sequence similarity for each sequence in the MSA against the WT.
    
    Args:
        wt_seq: The wild-type sequence (without gaps)
        aligned_seqs: List of aligned sequences (with gaps)
    
    Returns:
        tuple: (sequence_similarities, coverages) as numpy arrays
    """
    seq_sims = []
    coverages = []
    
    for aligned_seq in aligned_seqs:
        seq_sim = compute_sequence_similarity(wt_seq, aligned_seq)
        coverage = compute_coverage(wt_seq, aligned_seq)
        seq_sims.append(seq_sim)
        coverages.append(coverage)
    
    return np.array(seq_sims), np.array(coverages)


def convert_mutant_to_wt(mutant_seq, mutation_string):
    muts = mutation_string.split(":")
    for mut in muts:
        wt_aa = mut[0]
        mut_aa = mut[-1]
        position = int(mut[1:-1])
        assert mutant_seq[position-1] == mut_aa, "Mutation string does not match mutant sequence"
        mutant_seq = mutant_seq[:position-1] + wt_aa + mutant_seq[position:]
    return mutant_seq

def check_target_sequence_consistency(gym_df):
    csv_pattern = "../data/ProteinGym/DMS_ProteinGym_substitutions/*.csv"
    fail_counter = 0
    success_counter = 0
    for csv_path in glob.glob(csv_pattern):
        df = pd.read_csv(csv_path)
        dms_id = os.path.basename(csv_path).split(".")[0]
        row = gym_df[gym_df.DMS_id == dms_id]
        target_seq = row.target_seq.iloc[0]
        reconstructed_seq = convert_mutant_to_wt(
            df.iloc[0].mutated_sequence,
            df.iloc[0].mutant
        )
        if reconstructed_seq != target_seq:
            print(f"Target sequence mismatch for {csv_path} row {idx}")
            print(row['target_seq'], row['DMS_target_seq'])
            fail_counter += 1
        else:
            success_counter += 1
    print(f"Failed to process {fail_counter} DMSs")
    print(f"Successfully processed {success_counter} DMSs")


def save_coverage_similarity_data(msa_path, wt_seq, seq_sims, coverages):
    """
    Save coverage and sequence similarity data for each sequence in the MSA as a .npz file.
    
    Args:
        msa_path: Path to the original MSA file
        wt_seq: The wild-type sequence
        aligned_seqs: List of aligned sequences
        seq_sims: Array of sequence similarities
        coverages: Array of coverages
    """
    # Create output path with .npz extension
    base_name = os.path.splitext(msa_path)[0]
    npz_path = f"{base_name}.npz"
    
    # Save the data
    np.savez(
        npz_path,
        wt_sequence=wt_seq,
        sequence_similarities=seq_sims,
        coverages=coverages,
        msa_path=msa_path
    )
    print(f"Saved coverage and similarity data to {npz_path}")


if __name__ == "__main__":
    # gym_msa_pattern = "../data/ProteinGym/DMS_msa_files/*hhfilter.a3m"
    # gym_msa_pattern = "../data/ProteinGym/DMS_msa_files/*.a3m"
    gym_msa_pattern = "../data/ProteinGym/DMS_msa_files/*.a2m"
    poet_gym_msa_pattern = "../data/ProteinGym/PoET_DMS_msa_files/DMS_substitutions/*.a3m"
    gym_csv_path = "../data/ProteinGym/DMS_substitutions.csv"
    df = pd.read_csv(gym_csv_path)
    # check_target_sequence_consistency(df)
    df['MSA_filename'] = df['MSA_filename'].apply(lambda x: x.split(".")[0])
    
    # for path_pattern in [gym_msa_pattern, poet_gym_msa_pattern]:
    for path_pattern in [gym_msa_pattern]:
        fail_counter = 0
        results_rows = []
        for msa_path in glob.glob(path_pattern):
            # check if .npz file exists and skip if so
            base_name = os.path.splitext(msa_path)[0]
            npz_path = f"{base_name}.npz"
            if os.path.exists(npz_path):
                continue
            if "hhfilter" not in path_pattern and "hhfilter" in msa_path:
                continue
            if "PoET" in msa_path:
                row  = df[df.DMS_id == os.path.basename(msa_path).split(".")[0]]
                row = row.iloc[0]
                csv_save_path = "poet_gym_msa_meta_analysis_results.csv"
            else:
                row = df[df.MSA_filename== os.path.basename(msa_path).split(".")[0].replace("_reformat_hhfilter", "").replace("_reformat", "")]
                if len(row) == 0:
                    print(f"No row found for {msa_path}")
                    fail_counter += 1
                    results_rows.append({"msa_path": msa_path})
                    continue
                row = row.iloc[0]
                if "hhfilter" in gym_msa_pattern:
                    csv_save_path = "gym_msa_meta_analysis_results.csv"
                else:
                    csv_save_path = "unfiltered_gym_msa_meta_analysis_results.csv"
            row['MSA_filename'] = msa_path
            new_row = {
                "DMS_id": row['DMS_id'],
                "target_seq": row['target_seq'],
            }
            _, seqs = fasta.read_fasta(
                msa_path,
                keep_insertions=False,
                to_upper=False,
                keep_gaps=True,
            )
            _, seqs_w_insertions = fasta.read_fasta(
                msa_path,
                keep_insertions=True,
                to_upper=True,
                keep_gaps=False,
            )
            new_row["first_msa_seq"] = seqs[0]
            new_row["msa_path"] = msa_path
            # assert no lower case in seqs
            assert all(seq.isupper() for seq in seqs), "msa contains lower case sequences"
            
            wt_seq = row['target_seq']
            
            if wt_seq not in seqs and wt_seq not in seqs_w_insertions:
                # WT sequence not in MSA, need to redo alignment
                print(f"WT sequence not found in MSA for {row['DMS_id']}, redoing alignment...")
                
                # Add WT sequence to the beginning of the sequences
                updated_seqs = [wt_seq] + seqs
                updated_accessions = [f"WT_{row['DMS_id']}"] + [f"seq_{i}" for i in range(len(seqs))]
                
                # Create temporary FASTA file for alignment
                temp_fasta_path = msa_path.replace(".a3m", "_temp.fasta")
                write_fasta(updated_seqs, updated_accessions, temp_fasta_path)
                
                # Run alignment
                dirpath = os.path.dirname(msa_path)
                filename = os.path.basename(msa_path)
                new_dirpath = os.path.join(dirpath, "wt_updated_msa_files")
                new_msa_path = os.path.join(new_dirpath, filename)
                os.makedirs(new_dirpath, exist_ok=True)
                run_alignment_with_mafft(temp_fasta_path, new_msa_path, threads=1)
                
                # Read the updated alignment
                _, updated_seqs = fasta.read_fasta(
                    new_msa_path,
                    keep_insertions=False,
                    to_upper=False,
                    keep_gaps=True,
                )
                
                # Clean up temporary file
                os.remove(temp_fasta_path)
                
                new_row["wt_matches_msa"] = 0
                new_row["alignment_updated"] = True
                new_row["updated_msa_path"] = new_msa_path
                
                # Calculate coverage and similarity for all sequences
                seq_sims, coverages = compute_coverage_and_similarity_for_all_seqs(wt_seq, updated_seqs[1:])
                
                # Save the data as .npz file
                save_coverage_similarity_data(msa_path, wt_seq, seq_sims, coverages)
                
                new_row["mean_seq_sim"] = np.mean(seq_sims)
                new_row["mean_coverage"] = np.mean(coverages)
                new_row["n_seqs_in_msa"] = len(updated_seqs)
                
            else:
                if wt_seq != seqs[0] and wt_seq == seqs_w_insertions[0]:
                    # if the original file had insertions on the WT we need to use the version without insertions
                    wt_seq = seqs[0]
                # WT sequence is in MSA, use original alignment
                new_row["wt_matches_msa"] = 1
                new_row["alignment_updated"] = False
                
                # Calculate coverage and similarity for all sequences
                seq_sims, coverages = compute_coverage_and_similarity_for_all_seqs(wt_seq, seqs)
                
                # Save the data as .npz file
                save_coverage_similarity_data(msa_path, wt_seq, seq_sims, coverages)
                
                new_row["mean_seq_sim"] = np.mean(seq_sims)
                new_row["min_seq_sim"] = min(seq_sims)
                new_row["max_seq_sim"] = max(seq_sims)
                new_row["mean_coverage"] = np.mean(coverages)
                new_row["min_coverage"] = min(coverages)
                new_row["n_seqs_in_msa"] = len(seqs)
            
            print("\n", row['DMS_id'])
            print("mean seq sim", new_row["mean_seq_sim"], "mean coverage", new_row["mean_coverage"])
            results_rows.append(new_row)
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(csv_save_path, index=False)
        print(f"Failed to process {fail_counter} MSAs")




