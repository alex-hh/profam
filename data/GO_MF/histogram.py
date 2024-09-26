import gzip
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
import numpy as np
import argparse

# Add this import at the top of your file
from sys import maxsize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add this line before the read_output_file function
csv.field_size_limit(maxsize)

def read_output_file(file_path):
    """Reads the output TSV file and returns a list of the number of UniProt IDs per GO term."""
    uniprot_counts = []
    with gzip.open(file_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            uniprot_ids = row[1].split(',')
            uniprot_counts.append(len(uniprot_ids))
    return uniprot_counts

def calculate_statistics(uniprot_counts):
    """Calculates and returns basic statistics for the UniProt counts."""
    stats = {
        'mean': np.mean(uniprot_counts),
        'median': np.median(uniprot_counts),
        'std_dev': np.std(uniprot_counts),
        'unique_go_terms': len(uniprot_counts),
        'max': np.max(uniprot_counts),
        'min': np.min(uniprot_counts)
    }
    return stats

def visualize_distribution(uniprot_counts, stats, output_file):
    """Visualizes the distribution of the number of UniProt IDs per GO term and saves the plot."""
    df = pd.DataFrame({'UniProt IDs': uniprot_counts})
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='UniProt IDs', color='skyblue', edgecolor='black', bins=200)
    
    plt.title('Distribution of Number of UniProt IDs per GO Document', fontsize=16)
    plt.xlabel('Number of UniProt IDs per GO Document', fontsize=12)
    plt.ylabel('# of GO Documents', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(labelsize=10)
    
    # Create an improved stats box
    stats_text = (
        f"Statistics Summary:\n\n"
        f"Total GO Documents: {stats['unique_go_terms']:,}\n"
        f"Mean UniProt IDs: {stats['mean']:,.0f}\n"
        f"Median UniProt IDs: {stats['median']:,.0f}\n"
        f"Max UniProt IDs: {stats['max']:,}\n"
        f"Min UniProt IDs: {stats['min']:,}"
    )
    plt.text(0.9, 0.9, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
             fontsize=12, fontweight='bold', family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    logging.info(f"Plot saved as {output_file}")

def main():
    """Main function to read the output file and visualize the distribution."""
    parser = argparse.ArgumentParser(description="Visualize distribution of UniProt IDs per GO term.")
    parser.add_argument("-i", "--input", required=True, help="Input file path (gzipped TSV)")
    parser.add_argument("-o", "--output", required=True, help="Output file path for the plot")
    args = parser.parse_args()

    logging.info(f"Reading input file: {args.input}")
    uniprot_counts = read_output_file(args.input)
    stats = calculate_statistics(uniprot_counts)
    visualize_distribution(uniprot_counts, stats, args.output)
    logging.info("Visualization complete.")

if __name__ == "__main__":
    main()