"""
csv_path = "../casp16/targetlist_extracted_pdb.csv"

Target,Type,Res,Oligo.State,Entry Date, Server Exp.,Human Exp.,QA Exp.,Cancellation Date,Description,pdb_id
T1201,All groups,210,A2,2024-05-01,2024-05-04,2024-05-15,2024-05-18,-,Q9GZX9 8bwd,8bwd
H1202,All groups,190,A2B2,2024-05-02,2024-05-05,2024-05-16,2024-05-19,-,Q9GZX9_plus_P43026 8bwl,8bwl

head ../casp16/casp16.T1.seq.txt
>T1201 Q9GZX9, Human, 210 residues|
ETGCNKALCASDVSKCLIQELCQCRPGEGNCSCCKECMLCLGALWDECCDCVGMCNPRNYSDTPPTSKSTVEELHEPIPSLFRALTEGDTQLNWNIVSFPVAEELSHHENLVSFLETVNQPHHQNVSVPSNNVHAPYSSDKEHMCTVVYFDDCMSIHQCKISCESMGASKYRWFHNACCECIGPECIDYGSKTVKCMNCMFGTKHHHHHH
>T1206 Porcine astrovirus 4 capsid spike, Porcine astrovirus 4, 237 residues|
MGGPTTDPVQIYSPSLFGEPALYGSTATIGQRVPVAAVCMQAVGGAQKVYTYSLRELLDPVFVQNGNIIDITVIDLPTYPIYQKDGS
"""

import pandas as pd
import os
import glob
import urllib.request
from typing import Dict, Tuple, Optional

from Bio import pairwise2
from Bio.PDB import PDBParser, PDBIO, Select, PPBuilder
from Bio.PDB.Polypeptide import is_aa

def filter_pdb_csv_by_fasta(csv_path, fasta_path):
    with open(fasta_path, "r") as f:
        lines = f.readlines()
    casp_ids = []
    for line in lines:
        if line.startswith(">"):
            casp_id = line[1:].split()[0].strip()
            casp_ids.append(casp_id)
    df = pd.read_csv(csv_path)
    df = df[df["Target"].isin(casp_ids)]
    return df

def parse_fasta_to_dict(fasta_path: str) -> Dict[str, str]:
    """
    Parse FASTA file where headers begin with the CASP Target ID (e.g. ">T1201 ...")
    and return a mapping from Target ID to sequence string (concatenated across lines).
    """
    target_to_seq: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_seq: list[str] = []
    with open(fasta_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    target_to_seq[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0].strip()
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        target_to_seq[current_id] = "".join(current_seq)
    return target_to_seq


def download_pdb(pdb_id: str, save_dir: str) -> str:
    """
    Download a PDB file from RCSB by PDB ID and save to save_dir. Returns local path.
    """
    os.makedirs(save_dir, exist_ok=True)
    pdb_id_up = pdb_id.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id_up}.pdb"
    out_path = os.path.join(save_dir, f"{pdb_id_up}.pdb")
    if not os.path.exists(out_path):
        urllib.request.urlretrieve(url, out_path)
    return out_path


def extract_chain_sequences(pdb_path: str) -> Dict[str, str]:
    """
    Extract amino-acid sequences for each chain in the PDB using backbone traces.
    Returns mapping chain_id -> sequence.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(id=os.path.basename(pdb_path), file=pdb_path)
    model = structure[0]
    ppb = PPBuilder()
    chain_to_seq: Dict[str, str] = {}
    for chain in model:
        peptides = ppb.build_peptides(chain)
        if len(peptides) == 0:
            continue
        # Join contiguous peptide segments per chain; gaps are removed for comparison
        seq = "".join([pep.get_sequence().__str__() for pep in peptides])
        chain_to_seq[chain.id] = seq
    return chain_to_seq


def alignment_stats(target: str, chain: str) -> Tuple[float, float]:
    """
    Compute coverage and identity between target and chain sequences via global alignment.
    - coverage: aligned non-gap positions in target / len(target)
    - identity: identical matches / aligned non-gap positions
    Returns (coverage, identity).
    """
    if len(target) == 0:
        return 0.0, 0.0
    # Global alignment with simple scores to prioritize identity
    aligns = pairwise2.align.globalms(target, chain, 2, -1, -10, -0.5, one_alignment_only=True)
    if not aligns:
        return 0.0, 0.0
    a_target, a_chain, score, start, end = aligns[0]
    aligned_non_gap_in_target = 0
    identical_matches = 0
    for t_char, c_char in zip(a_target, a_chain):
        if t_char != "-":
            if c_char != "-":
                aligned_non_gap_in_target += 1
                if t_char == c_char:
                    identical_matches += 1
    coverage = aligned_non_gap_in_target / max(1, len(target))
    identity = (identical_matches / max(1, aligned_non_gap_in_target)) if aligned_non_gap_in_target > 0 else 0.0
    return coverage, identity


def select_best_chain(target_seq: str, chain_to_seq: Dict[str, str]) -> Tuple[str, float, float]:
    """
    For a target sequence, pick the chain with highest identity (tie-break by coverage).
    Returns (best_chain_id, coverage, identity).
    """
    best_chain = None
    best_cov = -1.0
    best_id = -1.0
    for chain_id, chain_seq in chain_to_seq.items():
        cov, ident = alignment_stats(target_seq, chain_seq)
        if ident > best_id or (ident == best_id and cov > best_cov):
            best_chain = chain_id
            best_cov = cov
            best_id = ident
    if best_chain is None:
        raise RuntimeError("No chains with valid sequences were found in the PDB.")
    return best_chain, best_cov, best_id


class ChainOnlyAminoAcidSelect(Select):
    def __init__(self, target_chain_id: str):
        super().__init__()
        self.target_chain_id = target_chain_id

    def accept_chain(self, chain):
        return 1 if chain.id == self.target_chain_id else 0

    def accept_residue(self, residue):
        # Accept only standard amino acids
        return 1 if is_aa(residue, standard=True) else 0

    def accept_atom(self, atom):
        # Accept only ATOM records (Bio.PDBIO handles HETATM via residue selection)
        return 1


def save_chain_only_pdb(src_pdb: str, dst_pdb: str, chain_id: str) -> None:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(id=os.path.basename(src_pdb), file=src_pdb)
    io = PDBIO()
    io.set_structure(structure)
    os.makedirs(os.path.dirname(dst_pdb), exist_ok=True)
    io.save(dst_pdb, select=ChainOnlyAminoAcidSelect(chain_id))


def save_target_fasta(target_id: str, sequence: str, save_dir: str) -> str:
    """
    Save the provided sequence to a FASTA file named {target_id}.fasta in save_dir.
    Returns the path to the written file.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{target_id}.fasta")
    with open(out_path, "w") as f:
        f.write(f">{target_id}\n")
        f.write(sequence + "\n")
    return out_path

if __name__ == "__main__":
    csv_path = "../casp16/targetlist_extracted_pdb.csv"
    fasta_path = "../casp16/casp16.T1.seq.txt"
    pdb_save_dir = "../casp16/pdbs"
    target_fasta_dir = "../casp16/target_fastas"
    df = filter_pdb_csv_by_fasta(csv_path, fasta_path)
    targets_to_sequences = parse_fasta_to_dict(fasta_path)

    os.makedirs(pdb_save_dir, exist_ok=True)

    for _, row in df.iterrows():
        target_id = row["Target"].strip()
        pdb_id = str(row["pdb_id"]).strip()
        if target_id not in targets_to_sequences:
            raise ValueError(f"Target {target_id} not found in FASTA file")
        target_seq = targets_to_sequences[target_id]

        # 1) Download PDB
        try:
            pdb_path = download_pdb(pdb_id, pdb_save_dir)
        except Exception as e:
            print(f"Error downloading PDB {pdb_id}: {e}")
            continue

        # 2) Extract chain sequences and align
        chain_to_seq = extract_chain_sequences(pdb_path)
        if len(chain_to_seq) == 0:
            raise RuntimeError(f"No chain sequences could be extracted for {pdb_id}")
        best_chain, cov, ident = select_best_chain(target_seq, chain_to_seq)

        # 3) Validate thresholds
        if cov < 0.6 or ident < 0.95:
            print(
                f"No chain in {pdb_id} matches {target_id} at required thresholds: "
                f"best_chain={best_chain}, coverage={cov:.3f}, identity={ident:.3f}"
            )
            continue

        # 4) Save cleaned chain-only PDB (prefix with target_id)
        out_pdb = os.path.join(pdb_save_dir, f"{target_id}_{pdb_id.strip().upper()}_chain_{best_chain}.pdb")
        save_chain_only_pdb(pdb_path, out_pdb, best_chain)
        # Also save the target sequence as FASTA for matched single-chain cases
        out_fasta = save_target_fasta(target_id, target_seq, target_fasta_dir)
        print(
            f"Saved cleaned chain PDB for {target_id} from {pdb_id} -> {out_pdb} "
            f"and FASTA -> {out_fasta} (coverage={cov:.3f}, identity={ident:.3f})"
        )