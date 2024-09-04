import gzip
import logging
import os
import csv
import sys
import urllib.request
from collections import defaultdict
from tqdm import tqdm

# download and process go uniprot file : https://www.ebi.ac.uk/GOA/downloads.html
# goa_uniprot_all.gaf.gz : 19GB file contains all GO annotations for proteins in UniProtKB
# this script requires ~36 GB of RAM and maps GO terms to UniProt IDs that 'enable' a molecular function (MF)

UNIPROT_ID_IDX = 1
QUALIFIER_IDX = 3
GO_TERM_IDX = 4
ASPECT_IDX = 8
DB_OBJECT_TYPE_IDX = 11
INPUT_URL = 'ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz'
GOA_FILE = 'data/GO_MF/goa_uniprot_all.gaf.gz'
OUTPUT_FILE = 'data/GO_MF/mf_to_uniprot_100k_mapping.tsv.gz'
MAX_UNIPROT_IDS = 100000  # 100k seems most reasonable
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
    
    logging.info(f"Filtering GO terms based on UniProt ID count (min: {MIN_UNIPROT_IDS}, max: {MAX_UNIPROT_IDS})")
    filtered_go_to_uniprot = {
        go_term: uniprot_ids
        for go_term, uniprot_ids in go_to_uniprot.items()
        if MIN_UNIPROT_IDS <= len(uniprot_ids) <= MAX_UNIPROT_IDS
    }
    
    logging.info(f"Writing {len(filtered_go_to_uniprot)} GO documents to {output_file}")
    ensure_dir(output_file)
    with gzip.open(output_file, 'wt', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for go_term, uniprot_ids in filtered_go_to_uniprot.items():
            writer.writerow([go_term, ','.join(uniprot_ids)])

def main():
    """Main function to process the GOA file and create the mapping."""
    logging.info("Starting download and processing of GOA UniProt file.")
    download_file(INPUT_URL, GOA_FILE)
    
    logging.info("Processing GOA UniProt file to create GO term to UniProt ID mapping.")
    process_goa_file(GOA_FILE, OUTPUT_FILE)
    
    logging.info(f"Processing complete. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()