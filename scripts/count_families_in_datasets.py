import os
import pandas as pd
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_data_dir = "../data"
    results = []

    # Foldseek dataset
    logger.info("Processing Foldseek dataset")
    foldseek_file = os.path.join(base_data_dir, "afdb/1-AFDBClusters-entryId_repId_taxId.tsv")
    df_foldseek = pd.read_csv(foldseek_file, sep='\t', header=None, names=["id", "clust", "tax"])
    num_families = df_foldseek['clust'].nunique()
    results.append({'Dataset': 'foldseek', 'NumFamilies': num_families})
    logger.info(f"Foldseek number of families: {num_families}")

    # TED dataset
    logger.info("Processing TED dataset")
    ted_dir = os.path.join(base_data_dir, "ted/ted_s50_by_sfam")
    ted_files = [f for f in os.listdir(ted_dir) if f.endswith('.fasta')]
    fam_ids = set()
    for filename in ted_files:
        fam_id = ".".join(filename.split(".")[:-1])
        fam_ids.add(fam_id)
    num_families = len(fam_ids)
    results.append({'Dataset': 'TED', 'NumFamilies': num_families})
    logger.info(f"TED number of families: {num_families}")

    # EC dataset
    logger.info("Processing EC dataset")
    ec_dir = os.path.join(base_data_dir, "ec/ec_fastas")
    ec_files = [f for f in os.listdir(ec_dir) if f.endswith('.fasta')]
    fam_ids = set()
    for filename in ec_files:
        fam_id = ".".join(filename.split(".")[:-1])
        fam_ids.add(fam_id)
    num_families = len(fam_ids)
    results.append({'Dataset': 'EC', 'NumFamilies': num_families})
    logger.info(f"EC number of families: {num_families}")

    # FunFam dataset
    logger.info("Processing FunFam dataset")
    funfam_dir = os.path.join(base_data_dir, "funfams/parquets")
    parquet_files = [f for f in os.listdir(funfam_dir) if f.endswith('.parquet')]
    fam_ids = set()
    for filename in parquet_files:
        df = pd.read_parquet(os.path.join(funfam_dir, filename))
        fam_ids.update(df['fam_id'].unique())
    num_families = len(fam_ids)
    results.append({'Dataset': 'funfam', 'NumFamilies': num_families})
    logger.info(f"FunFam number of families: {num_families}")

    # GO_MF dataset
    logger.info("Processing GO_MF dataset")
    go_mf_dir = os.path.join(base_data_dir, "GO_MF/mfparquets")
    parquet_files = [f for f in os.listdir(go_mf_dir) if f.endswith('.parquet')]
    fam_ids = set()
    for filename in parquet_files:
        df = pd.read_parquet(os.path.join(go_mf_dir, filename))
        fam_ids.update(df['fam_id'].unique())
    num_families = len(fam_ids)
    results.append({'Dataset': 'GO_MF', 'NumFamilies': num_families})
    logger.info(f"GO_MF number of families: {num_families}")

    # Pfam dataset
    logger.info("Processing Pfam dataset")
    pfam_dir = os.path.join(base_data_dir, "pfam/combined_parquets")
    parquet_files = [f for f in os.listdir(pfam_dir) if f.endswith('.parquet')]
    fam_ids = set()
    for filename in parquet_files:
        df = pd.read_parquet(os.path.join(pfam_dir, filename))
        fam_ids.update(df['fam_id'].unique())
    num_families = len(fam_ids)
    results.append({'Dataset': 'Pfam', 'NumFamilies': num_families})
    logger.info(f"Pfam number of families: {num_families}")

    # Save results to CSV
    output_file = os.path.join(base_data_dir, "n_families_per_dataset.csv")
    logger.info(f"Saving results to {output_file}")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()