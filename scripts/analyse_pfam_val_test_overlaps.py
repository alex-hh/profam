import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_overlap_counts(dataset_name, base_overlap_counts_dir):
    """
    Load and merge JSON files containing overlap counts for a given dataset.
    """
    data_dir = os.path.join(base_overlap_counts_dir, dataset_name)
    counts_dict = {}

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist.")
        return counts_dict

    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    logger.info(f"Found {len(json_files)} JSON files for dataset {dataset_name}.")

    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            counts_dict.update(data)

    logger.info(f"Loaded overlap counts for {len(counts_dict)} families in dataset {dataset_name}.")
    return counts_dict

def compute_statistics(counts_dict):
    """
    Compute required statistics and prepare data for histograms.
    """
    stats_list = []

    for fam_id, pfam_counts in counts_dict.items():
        num_pfam_families = len(pfam_counts)
        counts = list(pfam_counts.values())
        total_hits = sum(counts)
        average_hits_per_pfam_family = np.mean(counts) if counts else 0
        max_hits = max(counts) if counts else 0

        stats_list.append({
            'fam_id': fam_id,
            'num_pfam_families': num_pfam_families,
            'total_hits': total_hits,
            'average_hits_per_pfam_family': average_hits_per_pfam_family,
            'max_hits': max_hits
        })

    df_stats = pd.DataFrame(stats_list)
    return df_stats

def generate_histograms(df_stats, dataset_name, output_dir):
    """
    Generate histograms and save them to the output directory.
    """
    histograms = {
        'Number of PFAM families per training family': {
            'data': df_stats['num_pfam_families'],
            'filename': f'{dataset_name}_num_pfam_families_histogram.png',
            'xlabel': 'Number of PFAM families',
            'ylabel': 'Number of training families',
            'title': f'Distribution of PFAM families per training family for {dataset_name}'
        },
        'Total number of hits per training family': {
            'data': df_stats['total_hits'],
            'filename': f'{dataset_name}_total_hits_histogram.png',
            'xlabel': 'Total number of hits',
            'ylabel': 'Number of training families',
            'title': f'Distribution of total hits per training family for {dataset_name}'
        },
        'Average number of hits per PFAM family': {
            'data': df_stats['average_hits_per_pfam_family'],
            'filename': f'{dataset_name}_average_hits_histogram.png',
            'xlabel': 'Average number of hits per PFAM family',
            'ylabel': 'Number of training families',
            'title': f'Distribution of average hits per PFAM family for {dataset_name}'
        }
    }

    for desc, hist_info in histograms.items():
        plt.figure(figsize=(10, 6))
        plt.hist(hist_info['data'], bins=50, edgecolor='black')
        plt.xlabel(hist_info['xlabel'])
        plt.ylabel(hist_info['ylabel'])
        plt.title(hist_info['title'])
        plt.tight_layout()

        save_path = os.path.join(output_dir, hist_info['filename'])
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved histogram: {save_path}")

def main():
    base_overlap_counts_dir = "data/val_test/overlap_counts"
    output_dir = "../out/pfam_val_test_overlap_analysis"
    os.makedirs(output_dir, exist_ok=True)

    dataset_names = ['foldseek', 'ted', 'ec', 'funfam', 'go_mf']

    overall_stats = []

    for dataset_name in dataset_names:
        logger.info(f"Processing dataset: {dataset_name}")

        counts_dict = load_overlap_counts(dataset_name, base_overlap_counts_dir)
        if not counts_dict:
            logger.warning(f"No data to process for dataset {dataset_name}.")
            continue

        df_stats = compute_statistics(counts_dict)

        # Compute per-dataset statistics
        num_families_with_overlap = len(df_stats)
        avg_num_pfam_families = df_stats['num_pfam_families'].mean()
        avg_max_hits = df_stats['max_hits'].mean()

        overall_stats.append({
            'dataset_name': dataset_name,
            'num_families_with_overlap': num_families_with_overlap,
            'avg_num_pfam_families_per_family': avg_num_pfam_families,
            'avg_max_hits_per_family': avg_max_hits
        })

        # Generate histograms
        generate_histograms(df_stats, dataset_name, output_dir)

        # Save per-family statistics to CSV
        stats_csv_path = os.path.join(output_dir, f'{dataset_name}_per_family_statistics.csv')
        df_stats.to_csv(stats_csv_path, index=False)
        logger.info(f"Saved per-family statistics to {stats_csv_path}")

    # Save overall statistics to CSV
    df_overall_stats = pd.DataFrame(overall_stats)
    overall_stats_csv_path = os.path.join(output_dir, 'overall_dataset_statistics.csv')
    df_overall_stats.to_csv(overall_stats_csv_path, index=False)
    logger.info(f"Saved overall statistics to {overall_stats_csv_path}")

if __name__ == "__main__":
    main()