import requests
import pandas as pd
import csv
from io import StringIO

def get_uniprot_accession(identifier):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={identifier}&format=tsv&fields=accession"
    response = requests.get(url)
    if response.status_code == 200:
        tsv_reader = csv.reader(StringIO(response.text), delimiter='\t')
        next(tsv_reader)  # Skip header
        for row in tsv_reader:
            if row:
                return row[0]
    return "Not found"

def process_identifiers(input_file, output_file):
    df = pd.read_csv(input_file)
    with open(output_file, 'w') as outfile:
        outfile.write("fam_id\taccession\tsequence_name\tuniprot_accession\n")
        for i, row in df.iterrows():
            uniprot_accession = get_uniprot_accession(row.sequence_name.split("/")[0])
            outfile.write(f"{row.fam_id}\t{row.accession}\t{row.sequence_name}\t{uniprot_accession}\t{row['split']}\n")

if __name__ == "__main__":
    input_file = "../data/pfam/pfam_eval_splits/pfam_val_test_accessions.csv"
    output_file = "../data/pfam/pfam_eval_splits/pfam_val_test_accessions_uniprot.csv"
    process_identifiers(input_file, output_file)
    print(f"UniProt accessions have been written to {output_file}")