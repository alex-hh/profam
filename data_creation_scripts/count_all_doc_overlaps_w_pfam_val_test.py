import os
import json
import pandas as pd
from collections import defaultdict

class BaseOverlapCounter:
    def __init__(self):
        self.pfam_val_test = self.load_pfam_val_test()

    def load_pfam_val_test(self):
        pfam_val_test_csv = "data/val_test/pfam/pfam_val_test_accessions_w_unip_accs.csv"
        df = pd.read_csv(pfam_val_test_csv)
        print(f"Loaded pfam val test csv with {len(df)} rows")
        fam_to_up = defaultdict(set)
        for i, row in df.iterrows():
            fam_to_up[row['fam_id']].add(row['Entry'])
        return fam_to_up

    def count_overlaps(self):
        fam_id_up_ids = self.get_fam_id_up_ids()
        overlap_counts = {}
        total_fams = len(fam_id_up_ids)
        print(f"founds {total_fams} families")
        for i, (fam_id, up_ids) in enumerate(fam_id_up_ids.items()):
            if i % (total_fams // 20) == 0:
                print(f"Processed {i+1}/{total_fams} families")
            for pfam_fam, test_ids in self.pfam_val_test.items():
                intersection = set(up_ids).intersection(test_ids)
                if len(intersection) > 0:
                    if fam_id not in overlap_counts:
                        overlap_counts[fam_id] = defaultdict(int)
                    overlap_counts[fam_id][pfam_fam] = len(intersection)
        return overlap_counts

class FastaOverlapCounter(BaseOverlapCounter):
    def __init__(self, data_dir):
        super().__init__()
        self.fasta_dir = data_dir
    
    def get_fam_id_from_docpath(self, doc_path):
        return doc_path.split("/")[-1].split(".")[0]

    def up_id_from_line(self, fasta_line):
        return fasta_line.split("|")[1]

    def get_up_ids_from_fasta_lines(self, fasta_lines):
        up_ids = []
        for line in fasta_lines:
            if line.startswith(">"):
                up_ids.append(self.up_id_from_line(line))
        return up_ids

    def get_fam_id_up_ids(self):
        fam_id_up_ids = {}
        print("Processing fasta files in", self.fasta_dir)
        fasta_files = [f for f in os.listdir(self.fasta_dir) if f.endswith(".fasta")]
        print(f"Found {len(fasta_files)} fasta files")
        for filename in fasta_files:
            fam_id = self.get_fam_id_from_docpath(filename)
            with open(os.path.join(self.fasta_dir, filename), 'r') as f:
                fasta_lines = f.readlines()
            up_ids = self.get_up_ids_from_fasta_lines(fasta_lines)
            fam_id_up_ids[fam_id] = up_ids
        return fam_id_up_ids

class ECOverlapCounter(FastaOverlapCounter):
    def __init__(self, data_dir="../data/ec/ec_fastas"):
        super().__init__(data_dir)

class TEDOverlapCounter(FastaOverlapCounter):
    def __init__(self, data_dir="../data/ted/ted_s50_by_sfam"):
        super().__init__(data_dir)

    def up_id_from_line(self, fasta_line):
        return fasta_line.split("-")[1]

class FoldseekOverlapCounter(BaseOverlapCounter):
    def __init__(self, foldseek_cluster_index_file):
        super().__init__()
        self.foldseek_cluster_index_file = foldseek_cluster_index_file
    
    def get_fam_id_up_ids(self):
        json_path = self.foldseek_cluster_index_file.replace(".tsv", ".json")
        if not os.path.exists(json_path):
            id_clust_tax = pd.read_csv(
                    self.foldseek_cluster_index_file,
                    sep="\t",
                    header=None,
                    names=["id", "clust", "tax"]
                    )
            clust_to_up_ids = defaultdict(list)
            for _, row in id_clust_tax.iterrows():
                clust_to_up_ids[row["clust"]].append(row["id"])
            with open(json_path, "w") as json_file:
                json.dump(clust_to_up_ids, json_file, indent=4)
        else:
            with open(json_path) as json_file:
                clust_to_up_ids = json.load(json_file)
        return dict(clust_to_up_ids)

class ParquetOverlapCounter(BaseOverlapCounter):
    def __init__(self, data_dir, fam_id_col, up_id_col):
        super().__init__()
        self.parquet_dir = data_dir
        self.fam_id_col = fam_id_col
        self.up_id_col = up_id_col

    def get_fam_id_up_ids(self):
        fam_id_up_ids = {}
        for filename in os.listdir(self.parquet_dir):
            if filename.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(self.parquet_dir, filename))
                for i, row in df.iterrows():
                    fam_id = row[self.fam_id_col]
                    up_ids = row[self.up_id_col].apply(lambda x: x.split("/"[0]))
                    if fam_id not in fam_id_up_ids:
                        fam_id_up_ids[fam_id] = up_ids
                    else:
                        fam_id_up_ids[fam_id].extend(up_ids)
        return fam_id_up_ids


def process_dataset(counter_class, **kwargs):
    counter = counter_class(**kwargs)
    print("counter initialised for", counter_class.__name__ )
    overlap_counts = counter.count_overlaps()
    return overlap_counts

if __name__ == "__main__":
    base_data_dir = "../data"
    save_dir = "data/val_test/"
    os.makedirs(save_dir, exist_ok=True)
    # Process Foldseek dataset
    save_path = os.path.join(save_dir, "foldseek_pfam_overlap_counts.json")
    if not os.path.exists(save_path):
        print("Processing Foldseek dataset")
        foldseek_cluster_index_file = "../visualise_families/1-AFDBClusters-entryId_repId_taxId.tsv"
        if not os.path.exists(foldseek_cluster_index_file):
            foldseek_cluster_index_file = os.path.join(
                base_data_dir,
                "afdb",
                "1-AFDBClusters-entryId_repId_taxId.tsv",
            )
        foldseek_counts = process_dataset(
            FoldseekOverlapCounter,
            foldseek_cluster_index_file=foldseek_cluster_index_file
        )
        with open(save_path, 'w') as f:
            json.dump(foldseek_counts, f, indent=2)
        print("Foldseek counts saved to", save_path)
    else:
        print("Foldseek counts already exist at", save_path)
    
    # Process TED dataset
    save_path = os.path.join(save_dir, "ted_pfam_overlap_counts.json")
    if not os.path.exists(save_path):
        ted_counts = process_dataset(TEDOverlapCounter)
        with open(save_path, 'w') as f:
            json.dump(ted_counts, f, indent=2)
        print("TED counts saved to", save_path)
    else:
        print("TED counts already exist at", save_path)

    # Process EC dataset
    save_path = os.path.join(save_dir, "ec_pfam_overlap_counts.json")
    if not os.path.exists(save_path):
        ec_counts = process_dataset(ECOverlapCounter)
        with open(save_path, 'w') as f:
            json.dump(ec_counts, f, indent=2)
        print("EC counts saved to", save_path)
    else:
        print("EC counts already exist at", save_path)

    # Process Parquet datasets

    # Process funfams
    save_path = os.path.join(base_data_dir, "funfam_pfam_overlap_counts.json")
    if not os.path.exists(save_path):
        funfam_counts = process_dataset(
            counter_class=ParquetOverlapCounter,
            data_dir=os.path.join(base_data_dir, "funfams/parquets"),
            fam_id_col="fam_id",
            up_id_col="accessions"
        )

        with open(save_path, 'w') as f:
            json.dump(funfam_counts, f, indent=2)
        print("Funfam counts saved to", save_path)
    else:
        print("Funfam counts already exist at", save_path)
