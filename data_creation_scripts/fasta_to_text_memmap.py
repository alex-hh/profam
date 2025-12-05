"""
Created by Jude Wells 2025-12-04
input is a directory of fasta files
each fasta file contains a collection of sequences
representing one protein family.
output is a text memmap file:
.mapping and .sequences files
"""

import glob
import os
import argparse
from Bio import SeqIO
from tqdm import tqdm

def process_files(input_dir, output_prefix):
    # Find all fasta files
    # Only look for fasta files in the directory
    fasta_files = sorted(glob.glob(os.path.join(input_dir, "*.fasta")))
    if not fasta_files:
        print(f"No fasta files found in {input_dir}")
        return

    # Open output files
    sequences_filename = f"{output_prefix}.sequences"
    mapping_filename = f"{output_prefix}.mapping"

    # We need to make sure directories exist
    os.makedirs(os.path.dirname(sequences_filename) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(mapping_filename) or '.', exist_ok=True)

    seq_idx = 0
    
    # We will accumulate mapping entries and write them at the end 
    # to ensure we don't write a trailing newline
    mapping_entries = []

    # Use a buffer or just write directly. 
    # Since we need to write mapping at the end or carefully manage newlines,
    # and mapping size is proportional to number of families (not sequences), it fits in memory.
    
    print(f"Writing sequences to {sequences_filename}")
    with open(sequences_filename, "w") as seq_out:
        for fasta_file in tqdm(fasta_files, desc="Processing families"):
            # Family name logic: name of file, remove suffix after first dot
            basename = os.path.basename(fasta_file)
            family_name = basename.split('.')[0]
            
            family_indices = []
            
            # Read sequences
            # Use SeqIO if available, or manual parsing if simple enough.
            # Using SeqIO for robustness with multi-line fasta etc.
            for record in SeqIO.parse(fasta_file, "fasta"):
                # Header
                header = record.description
                # Sequence: remove gaps
                sequence = str(record.seq).replace("-", "")
                
                # Write to sequences file
                # Format:
                # >Header
                # Sequence
                seq_out.write(f">{header}\n")
                seq_out.write(f"{sequence}\n")
                
                family_indices.append(str(seq_idx))
                seq_idx += 1
            
            if family_indices:
                # Construct mapping entry
                # >FamilyID
                # filename:indices
                entry_line1 = f">{family_name}"
                # The filename in the mapping should be the basename of the sequences file
                sequences_basename = os.path.basename(sequences_filename)
                entry_line2 = f"{sequences_basename}:{','.join(family_indices)}"
                mapping_entries.append((entry_line1, entry_line2))

    print(f"Writing mapping to {mapping_filename}")
    with open(mapping_filename, "w") as map_out:
        for i, (l1, l2) in enumerate(mapping_entries):
            map_out.write(f"{l1}\n{l2}")
            if i < len(mapping_entries) - 1:
                map_out.write("\n")
                
    print(f"Processed {len(fasta_files)} families into {sequences_filename} and {mapping_filename}")
    print(f"Total sequences: {seq_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA files to ProFam text memmap format")
    parser.add_argument("--input_dir", type=str, default="../data/ProFam_atlas/foldseek_s50_hold_out_validation_128", help="Directory containing input FASTA files")
    parser.add_argument("--output_prefix", type=str, default="val_hold_out_128", help="Prefix for output .sequences and .mapping files")
    
    args = parser.parse_args()
    process_files(args.input_dir, args.output_prefix)
