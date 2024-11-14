import pandas as pd
import random
import os
import requests
import subprocess
import re
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import PPBuilder
from Bio import pairwise2

def main():
    parquet_dir = '/Users/judewells/Documents/dataScienceProgramming/cath_plm/data/pfam/shuffled_parquets'  # Update this path
    out_dir = "../visualise_families/pfam"
    os.makedirs(out_dir, exist_ok=True)
    df = load_random_parquet_file(parquet_dir)
    random_row = pick_random_row(df)
    sampled_accessions, uniprot_ids = sample_uniprot_ids(random_row)
    if not sampled_accessions:
        return

    accession_sequence_map = get_sequences_for_accessions(random_row, sampled_accessions)
    sequences = [accession_sequence_map[acc] for acc in sampled_accessions]

    pdb_paths = []
    for uid in uniprot_ids:
        pdb_path = download_alphafold_structure(uid)
        if pdb_path:
            pdb_paths.append(pdb_path)
        else:
            print(f"Failed to download structure for {uid}")
            return

    trimmed_pdb_paths = []
    for pdb_path, sequence, acc in zip(pdb_paths, sequences, sampled_accessions):
        uid = extract_uniprot_id(acc)
        trimmed_pdb_path = os.path.join('structures', f'{uid}_trimmed.pdb')
        trim_pdb(pdb_path, sequence, trimmed_pdb_path)
        trimmed_pdb_paths.append(trimmed_pdb_path)

    output_image = f'{out_dir}/{"_".join([a.replace("/", "_") for a in sampled_accessions])}.png'
    superpose_structures(trimmed_pdb_paths[0], trimmed_pdb_paths[1], output_image)
    print(f"Image saved as {output_image}")

def load_random_parquet_file(parquet_dir):
    """Loads a random parquet file from the specified directory."""
    parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    random_parquet_file = random.choice(parquet_files)
    df = pd.read_parquet(random_parquet_file)
    return df

def pick_random_row(df):
    """Selects a random row from the dataframe."""
    random_row = df.sample(n=1)
    return random_row

def extract_uniprot_id(accession_string):
    """Extracts the UniProt ID from the accession string."""
    # The UniProt accession is the first part before any '_'
    # For example: 'G0L7H1_ZOBGA/176-335' -> 'G0L7H1'
    uniprot_id = accession_string.split('_')[0]
    return uniprot_id

def sample_uniprot_ids(random_row):
    """Samples two UniProt IDs from the 'accessions' column."""
    accessions = random_row['accessions'].iloc[0]
    if len(accessions) >= 2:
        sampled_accessions = random.sample(list(accessions), 2)
        uniprot_ids = [extract_uniprot_id(acc) for acc in sampled_accessions]
        return sampled_accessions, uniprot_ids
    else:
        print("Not enough accessions in this row.")
        return None, None

def get_sequences_for_accessions(random_row, sampled_accessions):
    """Retrieves sequences associated with the sampled accessions."""
    sequences = random_row['sequences'].iloc[0]
    accession_sequence_map = dict(zip(random_row['accessions'].iloc[0], sequences))
    accession_sequence_subset = {acc: accession_sequence_map[acc] for acc in sampled_accessions}
    return accession_sequence_subset

def download_alphafold_structure(uniprot_id, output_dir='structures'):
    """Downloads the AlphaFold predicted structure for a given UniProt ID."""
    "https://alphafold.ebi.ac.uk/files/AF-A0A086FAV4-F1-model_v4.pdb"
    url = f'https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb'
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(output_dir, exist_ok=True)
        pdb_path = os.path.join(output_dir, f'{uniprot_id}.pdb')
        with open(pdb_path, 'wb') as f:
            f.write(response.content)
        return pdb_path
    else:
        print(f'Failed to download structure for {uniprot_id}')
        return None

def process_sequence(sequence):
    """Processes sequences to remove gaps and convert to uppercase."""
    # Remove dots, dashes, and any non-letter characters
    sequence = re.sub(r'[.\-]', '', sequence)
    sequence = re.sub(r'[^A-Za-z]', '', sequence)
    sequence = sequence.upper()
    return sequence

def trim_pdb(pdb_path, target_sequence_raw, output_path):
    """Trims the PDB structure to match the target sequence."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    ppb = PPBuilder()

    # Build the sequence from the PDB
    pdb_sequence = ''
    residue_list = []
    for pp in ppb.build_peptides(structure):
        pdb_sequence += str(pp.get_sequence())
        residue_list.extend(pp)

    # Process sequences
    target_sequence = process_sequence(target_sequence_raw)

    # Align the PDB sequence with the target sequence
    alignments = pairwise2.align.globalxx(pdb_sequence, target_sequence)
    best_alignment = alignments[0]
    pdb_aligned = best_alignment.seqA
    target_aligned = best_alignment.seqB

    # Determine which residues to keep
    positions_to_keep = []
    pdb_index = 0
    for i in range(len(pdb_aligned)):
        if pdb_aligned[i] != '-':
            if target_aligned[i] != '-':
                positions_to_keep.append(pdb_index)
            pdb_index += 1

    # Select residues to keep
    class ResidueSelect(Select):
        def accept_residue(self, residue):
            try:
                idx = residue_list.index(residue)
                return idx in positions_to_keep
            except ValueError:
                return False

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, ResidueSelect())

def superpose_structures(pdb1_path, pdb2_path, output_image):
    """Creates a PyMOL script to superpose two structures and saves the image."""
    pymol_script = f"""
load {pdb1_path}, structure1
load {pdb2_path}, structure2
super structure1, structure2
bg_color white
hide everything
show cartoon
color cyan, structure1
color magenta, structure2
png {output_image}, ray=1
quit
"""
    with open('pymol_script.pml', 'w') as f:
        f.write(pymol_script)
    subprocess.run(['pymol', '-cq', 'pymol_script.pml'])

if __name__ == '__main__':
    while True:
        main()
