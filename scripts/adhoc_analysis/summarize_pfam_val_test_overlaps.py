import glob
import os

import pandas as pd


def summarize_dataset_statistics(input_dir, output_file):
    # Get all CSV files with the pattern '*_per_family_statistics.csv'
    csv_files = glob.glob(os.path.join(input_dir, "*_per_family_statistics.csv"))

    summary_data = []

    for csv_file in csv_files:
        dataset_name = os.path.basename(csv_file).split("_per_family_statistics.csv")[0]
        df = pd.read_csv(csv_file)

        summary = {
            "dataset": dataset_name,
            "num_families": len(df),
            "avg_num_pfam_families": df["num_pfam_families"].mean(),
            "median_num_pfam_families": df["num_pfam_families"].median(),
            "max_num_pfam_families": df["num_pfam_families"].max(),
            "avg_total_hits": df["total_hits"].mean(),
            "median_total_hits": df["total_hits"].median(),
            "max_total_hits": df["total_hits"].max(),
            "avg_hits_per_pfam_family": df["average_hits_per_pfam_family"].mean(),
            "median_hits_per_pfam_family": df["average_hits_per_pfam_family"].median(),
            "max_hits_per_pfam_family": df["average_hits_per_pfam_family"].max(),
            "avg_max_hits": df["max_hits"].mean(),
            "median_max_hits": df["max_hits"].median(),
            "max_max_hits": df["max_hits"].max(),
        }

        summary_data.append(summary)

    # Create a DataFrame from the summary data and save it to a CSV file
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary statistics saved to {output_file}")


if __name__ == "__main__":
    input_directory = "../out/pfam_val_test_overlap_analysis"
    output_file = os.path.join(input_directory, "dataset_summary_statistics.csv")

    summarize_dataset_statistics(input_directory, output_file)
