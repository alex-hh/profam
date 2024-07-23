import requests
import re
import os


def download_sequence(uniprot_id):
    base_url = "https://www.uniprot.org/uniprot/"
    response = requests.get(base_url + uniprot_id + ".fasta")
    if response.status_code == 200:
        return response.text
    else:
        return None


def extract_uniprot_ids(subclass_content):
    pattern = r"((?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\w*)"
    uniprot_ids = re.findall(pattern, subclass_content)
    return set(uniprot_ids)


def generate_fasta_file(uniprot_ids, output_file):
    with open(output_file, "w") as file:
        for uniprot_id in uniprot_ids:
            sequence = download_sequence(uniprot_id)
            if sequence:
                file.write(sequence)


def parse_expasy_file(file_path, output_dir):
    with open(file_path, "r") as file:
        content = file.read()

    subclass_pattern = r"ID   ([\d.]+).*?//.*?"
    subclass_matches = re.findall(subclass_pattern, content, re.DOTALL)
    print(f"Found {len(subclass_matches)} subclasses")
    skip = True
    last_complete = '6_5_1_1'
    for ix, subclass_match in enumerate(subclass_matches):
        subclass_id = subclass_match.replace(".", "_")
        if subclass_id == last_complete:
            skip = False
        if skip:
            continue
        output_file = os.path.join(output_dir, f"{subclass_id}.fasta")
        if (ix + 1) % 100 == 0:
            print(f"Completed {ix + 1} subclasses")
        if os.path.exists(output_file):
            continue
        subclass_content = re.search(rf"ID   {re.escape(subclass_match)}.*?//", content, re.DOTALL).group()

        uniprot_ids = extract_uniprot_ids(subclass_content)

        if uniprot_ids:
            os.makedirs(output_dir, exist_ok=True)

            generate_fasta_file(uniprot_ids, output_file)
            print(f"FASTA file generated for EC subclass {subclass_match}: {output_file}")
        else:
            print(f"No UniProt IDs found for EC subclass {subclass_match}")

def create_train_test_split():
    """
    Pick some EC classes at different levels
    of the hierarchy to create a train-test split
    cluster sequences to make the classification
    more challenging
    """
    pass

if __name__ == "__main__":
    # Parse the expasy_enzyme_commission_all_class.dat file
    file_path = "../data/ec/expasy_enzyme_commission_all_class.dat"
    output_dir = "../data/ec/ec_fastas"
    parse_expasy_file(file_path, output_dir)