import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

"""
1) aggregate all cluster counts into a single dataframe
2) check how many parquet files have cluster counts > 0
3) check how many parquet files have cluster counts > 0 for all families

Analysing only those parquet files that have cluster counts:
count the total nuber of unique clusters in each level of sequence identity
how this data in a line plot
save the figure as a png file
"""



if __name__ == "__main__":
    cluster_count_dir = "../data/ted/s100_parquets/cluster_counts"
    cluster_count_files = glob.glob(os.path.join(cluster_count_dir, "*.csv"))
    
    # Read all CSV files into a single DataFrame
    all_counts = pd.concat([pd.read_csv(f) for f in cluster_count_files])
    print(f"Total number of rows: {len(all_counts)}")
    # Analysis 2: Files with any cluster counts >0
    has_clusters_mask = all_counts.filter(like='count_').gt(0).any(axis=1)
    files_with_clusters = all_counts[has_clusters_mask]['parquet_file'].nunique()
    print(f"Files with any clusters: {files_with_clusters}/{len(all_counts['parquet_file'].unique())}")
    # Analysis 3: Files where ALL families have clusters
    all_families_have_clusters = all_counts.groupby('parquet_file').apply(
        lambda g: g.filter(like='count_').gt(0).all(axis=1).all()
    )
    files_with_all_clusters = all_families_have_clusters.sum()
    print(f"Files where ALL families have clusters: {files_with_all_clusters}")
    
    # Analysis 4: Total unique clusters per identity level (using files with clusters)
    filtered_counts = all_counts[has_clusters_mask]
    identity_levels = sorted(["_".join(col.split('_')[2:]) for col in filtered_counts.columns if col.startswith('count_')], 
                           key=int, reverse=True)
    
    # Create plot data
    plot_data = []
    for level in identity_levels:
        identity_level = int(level) if len(level)==2 else int(level) * 10
        total_clusters = filtered_counts[f'count_0_{level}'].sum()
        print(f"Identity level: {identity_level}, Total clusters: {total_clusters}")
        plot_data.append((identity_level, total_clusters))
    
    # Create and save plot
    plt.figure(figsize=(10, 6))
    levels, counts = zip(*sorted(plot_data))
    plt.plot(levels, counts, marker='o')
    plt.xlabel('Sequence Identity Threshold (%)')
    plt.ylabel('Total Clusters')
    plt.title('Cluster Counts by Sequence Identity Level')
    plt.grid(True)
    plt.savefig('cluster_counts_by_identity.png')
    print("Saved cluster_counts_by_identity.png")
