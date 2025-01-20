import os

import logomaker
from Bio import AlignIO

all_df = logomaker.list_color_schemes()


def calculate_coverage(alignment):
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    total_gaps = sum(seq.seq.count('-') for seq in alignment)
    coverage = (alignment_length * num_sequences - total_gaps) / (alignment_length * num_sequences)
    return coverage


# Specify the directory containing the a3m files
directory = '/Users/judewells/Documents/dataScienceProgramming/cath_plm/data/ProteinGym/processed_DMS_msa_files'
logo_dir = '/Users/judewells/Documents/dataScienceProgramming/cath_plm/data/ProteinGym/logos'
os.makedirs(logo_dir, exist_ok=True)

# Iterate through each a3m file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.a2m'):
        file_path = os.path.join(directory, filename)

        # Read the a3m file using AlignIO
        alignment = AlignIO.read(file_path, 'fasta')

        # Calculate coverage and similarity
        coverage = calculate_coverage(alignment)

        print(f"File: {filename}")
        print(f"Coverage: {coverage:.2f}")

        # Convert alignment to a list of sequences
        sequences = [str(record.seq) for record in alignment]

        # Create sequence logo
        counts_matrix = logomaker.alignment_to_matrix(sequences)
        logo = logomaker.Logo(counts_matrix, color_scheme="weblogo_protein", width=0.8, figsize=(60, 2.5))
        logo_filename = os.path.splitext(filename)[0] + '_logo.png'
        logo.fig.savefig(os.path.join(logo_dir, logo_filename))

        print(f"Sequence logo saved as {logo_filename}")
        print("---")