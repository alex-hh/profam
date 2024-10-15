import csv
import json
import os.path
from collections import defaultdict
"""
Download file from pfam ftp site:
https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/
Pfam-A.regions.uniprot.tsv.gz

This file reformats it into a json:
pfam_family_id: [uniprot_acc1, uniprot_acc2, ...]
"""

input_file = '../data/pfam/Pfam-A.regions.uniprot.tsv'
output_file = '../data/pfam/pfam_uniprot_mappings.json'
def convert_pfam_uniprot_to_json(input_file=input_file, output_file=output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file {input_file} not found"
            "Download file from pfam ftp site: "
            "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/" 
            "Pfam-A.regions.uniprot.tsv.gz"
        )
    pfam_uniprot_dict = defaultdict(list)
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for i, row in enumerate(reader):
            if len(row) >= 5:
                uniprot_acc = row[0]
                pfam_acc = row[4]
                pfam_uniprot_dict[pfam_acc].append(uniprot_acc)
            if (i + 1) % 1_000_000 == 0:
                print(f"Processed {i + 1} lines")

    with open(output_file, 'w') as out:
        json.dump(pfam_uniprot_dict, out, indent=2)


# Usage

convert_pfam_uniprot_to_json(input_file, output_file)

print(f"Conversion complete. JSON file saved as {output_file}")