import gzip
import logging
import os
import csv
import sys
import urllib.request
from collections import defaultdict
from tqdm import tqdm
import argparse
import math

"""
This script processes the Gene Ontology Annotation (GOA) database to create a mapping of
Molecular Function (MF) GO terms to UniProt IDs. It performs the following tasks:

1. Downloads the GOA UniProt file (goa_uniprot_all.gaf.gz, ~19GB) from EBI
2. Parses the file to extract MF GO terms and their associated UniProt IDs
3. Calculates Information Content (IC) for each GO term
4. Filters and processes the data to create a mapping of GO terms to UniProt IDs
5. Outputs the processed data to a compressed TSV file

The script focuses on GO terms that 'enable' molecular functions and requires
approximately 36GB of RAM to run efficiently due to the large dataset size.
"""

UNIPROT_ID_IDX = 1
QUALIFIER_IDX = 3
GO_TERM_IDX = 4
ASPECT_IDX = 8
DB_OBJECT_TYPE_IDX = 11 
INPUT_URL = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz'
GOA_FILE = 'data/GO_MF/goa_uniprot_all.gaf.gz'
GO_OBO_URL = 'http://purl.obolibrary.org/obo/go.obo'
GO_OBO_FILE = 'data/GO_MF/go.obo'
OUTPUT_FILE = 'data/GO_MF/mf_to_uniprot_mapping_ic.tsv.gz'
#MAX_UNIPROT_IDS = 100000  # 100k seems most reasonable
MIN_UNIPROT_IDS = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(file_path):
    """Ensure that the directory for the given file path exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def download_file(url, output_path):
    """Downloads a file from the given URL to the specified output path using urllib."""
    ensure_dir(output_path)
    if not os.path.exists(output_path):
        try:
            logging.info(f"Downloading file from {url}")
            urllib.request.urlretrieve(url, output_path)
            logging.info(f"Downloaded file to {output_path}")
        except Exception as e:
            logging.error(f"Error downloading file: {e}")
            sys.exit(1)
    else:
        logging.info(f"File already exists at {output_path}")

class GOTerm:
    def __init__(self, id):
        self.id = id
        self.parents = set()
        self.children = set()
        self.namespace = None
        self.annotation_count = 0
        self.is_obsolete = False

def parse_go_obo(file_path):
    """Parse the GO OBO file and return a dictionary of MF GO terms."""
    go_terms = {}
    current_term = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '[Term]':
                current_term = None
            elif line.startswith('id: '):
                go_id = line.split(': ')[1]
                current_term = GOTerm(go_id)
            elif current_term is not None:
                if line.startswith('namespace: '):
                    current_term.namespace = line.split(': ')[1]
                    if current_term.namespace == 'molecular_function':
                        go_terms[current_term.id] = current_term
                    else:
                        current_term = None
                elif line.startswith('is_a: ') and current_term is not None:
                    parent_id = line.split(' ')[1]
                    current_term.parents.add(parent_id)
                    if parent_id in go_terms:
                        go_terms[parent_id].children.add(current_term.id)
                elif line == 'is_obsolete: true':  # Add this block
                    current_term.is_obsolete = True

    return go_terms

def count_annotations(go_terms, go_to_uniprot):
    """Count annotations for each GO term, including children."""
    for go_id, uniprot_ids in go_to_uniprot.items():
        if go_id in go_terms:
            term = go_terms[go_id]
            term.annotation_count += len(uniprot_ids)
            
            # Propagate counts to parents
            stack = list(term.parents)
            while stack:
                parent_id = stack.pop()
                if parent_id in go_terms:
                    parent = go_terms[parent_id]
                    parent.annotation_count += len(uniprot_ids)
                    stack.extend(parent.parents)

def calculate_ic(go_terms):
    """Calculate the Information Content for each GO term."""
    namespace_totals = defaultdict(int)
    for term in go_terms.values():
        namespace_totals[term.namespace] += term.annotation_count

    ic_values = {}
    for go_id, term in go_terms.items():
        if term.namespace and term.annotation_count > 0:
            p_t = term.annotation_count / namespace_totals[term.namespace]
            ic_values[go_id] = -math.log(p_t)
        else:
            ic_values[go_id] = 0

    return ic_values

def download_go_obo():
    """Downloads the GO OBO file."""
    ensure_dir(GO_OBO_FILE)
    if not os.path.exists(GO_OBO_FILE):
        try:
            logging.info(f"Downloading GO OBO file from {GO_OBO_URL}")
            urllib.request.urlretrieve(GO_OBO_URL, GO_OBO_FILE)
            logging.info(f"Downloaded GO OBO file to {GO_OBO_FILE}")
        except Exception as e:
            logging.error(f"Error downloading GO OBO file: {e}")
            sys.exit(1)
    else:
        logging.info(f"GO OBO file already exists at {GO_OBO_FILE}")

def process_goa_file(input_file, output_file):
    """Processes the GOA file, filters entries, and creates GO term to UniProt ID mapping."""
    go_to_uniprot = defaultdict(set)
    
    total_lines = 1_192_000_000  # 1.192 billion lines
    
    with gzip.open(input_file, 'rt') as infile:
        for line in tqdm(infile, total=total_lines, unit='line', desc="Processing GOA file"):
            if line.startswith('!'):
                continue
            columns = line.strip().split('\t')
            if (columns[DB_OBJECT_TYPE_IDX] == 'protein' and 
                columns[ASPECT_IDX] == 'F' and 
                columns[QUALIFIER_IDX] == 'enables'):
                go_term = columns[GO_TERM_IDX]
                uniprot_id = columns[UNIPROT_ID_IDX]
                go_to_uniprot[go_term].add(uniprot_id)
    
    logging.info(f"Filtering GO terms based on UniProt ID count (min: {MIN_UNIPROT_IDS})")
    filtered_go_to_uniprot = {
        go_term: uniprot_ids
        for go_term, uniprot_ids in go_to_uniprot.items()
        if MIN_UNIPROT_IDS <= len(uniprot_ids)
    }
    
    # Parse GO OBO file
    logging.info("Parsing GO OBO file")
    go_terms = parse_go_obo(GO_OBO_FILE)
    
    # Count annotations
    logging.info("Counting annotations")
    count_annotations(go_terms, filtered_go_to_uniprot)
    
    # Calculate IC
    logging.info("Calculating Information Content")
    ic_values = calculate_ic(go_terms)
    
    logging.info(f"Writing non-obsolete GO documents to {output_file}")
    ensure_dir(output_file)
    with gzip.open(output_file, 'wt', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for go_term, uniprot_ids in filtered_go_to_uniprot.items():
            if go_term in go_terms and not go_terms[go_term].is_obsolete:  # Add this condition
                ic = ic_values.get(go_term, 0)
                writer.writerow([go_term, ic, ','.join(uniprot_ids)])

def main():
    """Main function to process the GOA file and create the mapping."""
    parser = argparse.ArgumentParser(description="Process GOA UniProt file and create GO term to UniProt ID mapping.")
    parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Specify the output file path")
    args = parser.parse_args()

    logging.info("Starting download and processing of GOA UniProt file.")
    download_file(INPUT_URL, GOA_FILE)
    
    logging.info("Downloading GO OBO file.")
    download_go_obo()
    
    logging.info("Processing GOA UniProt file to create GO term to UniProt ID mapping.")
    process_goa_file(GOA_FILE, args.output)
    
    logging.info(f"Processing complete. Output written to {args.output}")

if __name__ == "__main__":
    main()