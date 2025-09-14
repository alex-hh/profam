"""
Takes a path to a multi-sequence fasta file and output directory 
and splits the fasta file into individual sequence fasta files.
plus a file called fasta_file_list.txt
which has one filename per line
"""
from Bio import SeqIO
import os

def split_multi_fasta_to_individual(fasta_path: str, output_dir: str) -> None:
    """Split a multi-FASTA into per-identifier FASTA files using Biopython.

    Each output filename is exactly the FASTA identifier (record.id, token up to
    first whitespace) with a .fasta extension. A file named
    fasta_file_list.txt is written in the output directory containing one
    filename per line in input order.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_filenames = []
    seen_identifiers = set()

    records = list(SeqIO.parse(fasta_path, "fasta"))
    for record in records:
        identifier = record.id
        if identifier in seen_identifiers:
            raise ValueError(f"Duplicate FASTA identifier encountered: {identifier}")
        seen_identifiers.add(identifier)

        filename = f"{identifier}.fasta"
        out_path = os.path.join(output_dir, filename)

        with open(out_path, "w") as out_f:
            out_f.write(f">{record.description}\n")
            out_f.write(str(record.seq) + "\n")

        output_filenames.append(filename)

    list_path = os.path.join(output_dir, "fasta_file_list.txt")
    with open(list_path, "w") as list_f:
        for name in output_filenames:
            list_f.write(name + "\n")

if __name__ == "__main__":
    fasta_path  = "/mnt/disk2/cath_plm/data/bfvd_logan.fasta"
    output_dir = "/mnt/disk2/cath_plm/data/viral_bfvd_logan_individual"
    split_multi_fasta_to_individual(fasta_path, output_dir)