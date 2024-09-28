import csv
import json
from collections import defaultdict


def convert_pfam_uniprot_to_json(input_file, output_file):
    pfam_uniprot_dict = defaultdict(list)

    # Process the input file line by line
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

    # Write the JSON file
    with open(output_file, 'w') as out:
        json.dump(pfam_uniprot_dict, out, indent=2)


# Usage
input_file = '../data/pfam/Pfam-A.regions.uniprot.tsv'
output_file = '../data/pfam/pfam_uniprot_mappings.json'
convert_pfam_uniprot_to_json(input_file, output_file)

print(f"Conversion complete. JSON file saved as {output_file}")