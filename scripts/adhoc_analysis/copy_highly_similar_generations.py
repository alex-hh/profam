import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import shutil
import glob
from Bio import SeqIO
from src.data.objects import Protein
from src.utils.evaluation_utils import pairwise_sequence_identity

profam_csv = "../sampling_results/colabfold_outputs/profam_structural_evaluation.csv"
poet_csv = "../sampling_results/colabfold_outputs/poet_structural_eval/poet_structural_evaluation.csv"
output_dir = "../sampling_results/inspect_similar_seq_dissimilar_structs"
os.makedirs(output_dir, exist_ok=True)
profam_df = pd.read_csv(profam_csv)
poet_df = pd.read_csv(poet_csv)

gt_pdb_pattern = "../data/val_test_v2_pdbs/foldseek/*.pdb"
gt_pdbs = glob.glob(gt_pdb_pattern)

def get_pdb_paths_from_fasta_path(fasta_path, gt_pdbs_list):
    prompt_records = list(SeqIO.parse(fasta_path, "fasta"))
    prompt_ids = [record.id for record in prompt_records]
    pdb_paths = [p for p in gt_pdbs_list if any(pid in p for pid in prompt_ids)]
    return pdb_paths

def copy_generated_and_best_prompt(row, root_output_dir):
    try:
        if row["seq_identity_max"] <= 0.95 or not (row["tm_max"] < 0.7 or row["lddt_max"] < 0.8):
            return
    except Exception:
        return

    generated_pdb_path = row.get("generated_pdb", None)
    prompt_fasta_path = row.get("prompt_fasta", None)
    if generated_pdb_path is None or prompt_fasta_path is None:
        return
    si = round(row["seq_identity_max"], 2)
    tm = round(row["tm_max"], 2)
    lddt = round(row["lddt_max"], 2)
    new_dir_name = f"{root_output_dir}/{row['generated_id']}_si_{si}_tm_{tm}_lddt_{lddt}"
    os.makedirs(new_dir_name, exist_ok=True)

    # Copy generated PDB
    if isinstance(generated_pdb_path, str) and os.path.exists(generated_pdb_path):
        try:
            shutil.copy(generated_pdb_path, f"{new_dir_name}/generated.pdb")
        except Exception:
            pass

    # Identify prompt sequence with max identity and copy its PDB
    try:
        gen_prot = Protein.from_pdb(generated_pdb_path, bfactor_is_plddt=True)
        gen_seq = gen_prot.sequence if gen_prot is not None else ""
    except Exception:
        gen_seq = ""

    try:
        prompt_records = list(SeqIO.parse(prompt_fasta_path, "fasta"))
    except Exception:
        prompt_records = []

    best_record = None
    best_identity = -1.0
    if gen_seq and len(prompt_records) > 0:
        for rec in prompt_records:
            try:
                identity = pairwise_sequence_identity(gen_seq, str(rec.seq).replace("-", ""))
            except Exception:
                identity = float("nan")
            if not np.isnan(identity) and identity > best_identity:
                best_identity = identity
                best_record = rec

    if best_record is not None:
        candidate_prompt_pdbs = get_pdb_paths_from_fasta_path(prompt_fasta_path, gt_pdbs)
        # Prefer PDB that contains the exact best_record id in its path
        matching = [p for p in candidate_prompt_pdbs if best_record.id in p]
        prompt_pdb_to_copy = matching[0] if len(matching) > 0 else (candidate_prompt_pdbs[0] if len(candidate_prompt_pdbs) > 0 else None)
        if prompt_pdb_to_copy is not None and os.path.exists(prompt_pdb_to_copy):
            try:
                shutil.copy(prompt_pdb_to_copy, f"{new_dir_name}/prompt_max_seq_identity.pdb")
            except Exception:
                pass

for i,row in profam_df.iterrows():
    copy_generated_and_best_prompt(row, output_dir)

for i,row in poet_df.iterrows():
    copy_generated_and_best_prompt(row, output_dir)