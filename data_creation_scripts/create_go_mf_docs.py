import gzip
import logging
import os
import csv
import sys
import urllib.request
from collections import defaultdict
from tqdm import tqdm
import argparse
from goatools import obo_parser
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.semantic import TermCounts, get_info_content

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
OUTPUT_FILE = 'data/GO_MF/mf_to_uniprot_mapping_ic.tsv.gz'
#MAX_UNIPROT_IDS = 100000  # 100k seems most reasonable
MIN_UNIPROT_IDS = 2

GO_OBO_URL = 'http://purl.obolibrary.org/obo/go.obo'
GO_OBO_FILE = 'data/GO_MF/go.obo'

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

def calculate_ic(go_to_uniprot):
    """Calculates Information Content for GO terms using GOATOOLS."""
    logging.info("Calculating Information Content for GO terms")
    
    # Load the GO OBO file
    go_obo = obo_parser.GODag(GO_OBO_FILE)
    
    # Create a Gene2GoReader-like object from our data
    class CustomGene2GoReader:
        def __init__(self, go_to_uniprot):
            self.go_to_uniprot = go_to_uniprot
        
        def associations(self):
            for go_id, uniprot_ids in self.go_to_uniprot.items():
                for uniprot_id in uniprot_ids:
                    yield (uniprot_id, go_id)
    
    gene2go = CustomGene2GoReader(go_to_uniprot)
    
    # Calculate termcounts
    termcounts = TermCounts(go_obo, gene2go.associations())
    
    # Calculate IC for each GO term
    go_to_ic = {}
    for go_id in go_to_uniprot.keys():
        if go_id in go_obo:
            go_to_ic[go_id] = get_info_content(go_id, termcounts)
        else:
            logging.warning(f"GO term {go_id} not found in the OBO file")
            go_to_ic[go_id] = 0
    
    return go_to_ic

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
    
    # Calculate Information Content for each GO term using GOATOOLS
    go_to_ic = calculate_ic(filtered_go_to_uniprot)
    
    logging.info(f"Writing {len(filtered_go_to_uniprot)} GO documents to {output_file}")
    ensure_dir(output_file)
    with gzip.open(output_file, 'wt', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for go_term, uniprot_ids in filtered_go_to_uniprot.items():
            ic = go_to_ic.get(go_term, 0)
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