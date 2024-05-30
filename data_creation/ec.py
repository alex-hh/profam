import requests
import re

def download_sequence(uniprot_id):
    base_url = "https://www.uniprot.org/uniprot/"
    response = requests.get(base_url + uniprot_id + ".fasta")
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_uniprot_ids(file_content):
    pattern = r"((?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\w*)"
    uniprot_ids = re.findall(pattern, file_content)
    return set(uniprot_ids)

def generate_fasta_file(uniprot_ids, output_file):
    with open(output_file, "w") as file:
        for uniprot_id in uniprot_ids:
            sequence = download_sequence(uniprot_id)
            if sequence:
                file.write(sequence)

if __name__ == "__main__":
    # Read the file content
    ec_dir = "data/example_data/expasy_ec"
    with open(f"{ec_dir}/1.1.1.1.txt", "r") as file:
        file_content = file.read()

    # Extract UniProt IDs from the file
    uniprot_ids = extract_uniprot_ids(file_content)

    # Generate the FASTA file
    output_file = f"{ec_dir}/1.1.1.1.fasta"
    generate_fasta_file(uniprot_ids, output_file)

    print("FASTA file generated successfully.")