import gzip
import csv
import random
import logging
import requests
import os
from PIL import Image
from io import BytesIO
import sys

# Add this line near the top of the file, after the imports
csv.field_size_limit(sys.maxsize)

INPUT_FILE = "data/GO_MF/mf_to_uniprot_100k_mapping.tsv.gz"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_go_terms(file_path):
    """Reads the TSV file and returns a dictionary of GO terms with their UniProt counts."""
    go_terms = {}
    with gzip.open(file_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            go_term = row[0]
            uniprot_ids = row[1].split(',')
            go_terms[go_term] = len(uniprot_ids)
    return go_terms

def filter_go_terms(go_terms, target_counts, tolerance=500):
    """Filters GO terms based on target UniProt counts with a given tolerance."""
    filtered_terms = {count: [] for count in target_counts}
    for go_term, count in go_terms.items():
        for target in target_counts:
            if target - tolerance <= count <= target + tolerance:
                filtered_terms[target].append(go_term)
    return filtered_terms

def sample_go_terms(filtered_terms, sample_size=2):
    """Samples a specified number of GO terms from each filtered list."""
    sampled_terms = {}
    for count, terms in filtered_terms.items():
        if len(terms) >= sample_size:
            sampled_terms[count] = random.sample(terms, sample_size)
        else:
            sampled_terms[count] = terms
    return sampled_terms

def get_ancestor_chart(go_term):
    """Fetches the ancestor chart image for a GO term from QuickGO API."""
    chart_url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_term}/chart?format=png"
    response = requests.get(chart_url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        logging.warning(f"Failed to fetch ancestor chart for GO term: {go_term}")
        return None

def get_go_term_info(go_term, uniprot_count):
    """Queries the QuickGO API for GO term information and ancestor chart."""
    base_url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
    response = requests.get(f"{base_url}{go_term}")
    if response.status_code == 200:
        data = response.json()
        term_info = data['results'][0] if data['results'] else None
        
        if term_info:
            chart_image = get_ancestor_chart(go_term)
            return {
                'name': term_info['name'],
                'definition': term_info['definition']['text'],
                'parents': [p['id'] for p in term_info.get('parents', [])],
                'children': [c['id'] for c in term_info.get('children', [])],
                'ancestor_chart': chart_image,
                'uniprot_count': uniprot_count
            }
    
    logging.warning(f"Failed to fetch info for GO term: {go_term}")
    return None

def main():
    """Main function to read GO terms, filter, sample, and process the results."""
    logging.info("Reading GO terms from the file.")
    go_terms = read_go_terms(INPUT_FILE)
    
    target_counts = [10, 100, 1000,10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    logging.info("Filtering GO terms based on target counts.")
    filtered_terms = filter_go_terms(go_terms, target_counts)
    
    logging.info("Sampling GO terms.")
    sampled_terms = sample_go_terms(filtered_terms)
    
    logging.info("Fetching GO term information and ancestor charts from QuickGO API.")
    
    os.makedirs("data/GO_MF/ancestor_charts", exist_ok=True)
    
    for count, terms in sampled_terms.items():
        # Create a subfolder for each UniProt count
        count_folder = f"data/GO_MF/ancestor_charts/{count}_uniprot_ids"
        os.makedirs(count_folder, exist_ok=True)
        
        for term in terms:
            term_info = get_go_term_info(term, count)
            if term_info:
                if term_info['ancestor_chart']:
                    chart_filename = f"{count_folder}/{term}_ancestor_chart.png"
                    term_info['ancestor_chart'].save(chart_filename)
                    logging.info(f"Saved ancestor chart for {term} to {chart_filename}")
            else:
                logging.warning(f"No information found for GO term: {term}")

if __name__ == "__main__":
    main()