"""
map which pfam families are in each parquet file
calculate the size in bytes of each family
create a shuffled list of all families
chunk the families into parquets that have
roughly equal size in bytes.
save the new shuffled and resized parquets.
"""
import glob
import json
import os
import pickle
import sys

import pandas as pd

from src.sequence.fasta import read_fasta_lines


def create_parquet_map(indir: str, mapping_path: str, limit_mb=250):
    """
    shuffles rows across all parquet files
    creates a map for the new parquet files
    where each one can contain rows from many parquets
    splits the parquets to ensure roughly equal size.
    """
    df_rows = []
    assert os.path.exists(indir)
    parquet_paths = glob.glob(f"{indir}/*.parquet")
    assert len(parquet_paths) > 0
    preshuffled_parquet_map = os.path.join(os.path.dirname(mapping_path), "preshuffled_parquet_map.csv")
    if os.path.exists(preshuffled_parquet_map):
        print(f"loading preshuffled parquet map from {preshuffled_parquet_map}")
        df = pd.read_csv(preshuffled_parquet_map)
    else:
        for parquet_file in glob.glob(f"{indir}/*.parquet"):
            df = pd.read_parquet(parquet_file)
            for i, row in df.iterrows():
                fam_id = row["pfam_acc"]
                size_mb = sys.getsizeof(row["text"]) / 1024 / 1024
                fname = os.path.basename(parquet_file)
                df_rows.append(
                    {"pfam_acc": fam_id, "size_mb": size_mb, "parquet_file": fname}
                )
        df = pd.DataFrame(df_rows)
        total_rows = df.shape[0]
        print(f"total rows from all parquets: {total_rows}")
        df.to_csv(preshuffled_parquet_map, index=False)
    df = df.sample(frac=1).reset_index(drop=True)
    parq_ix = 0
    current_size = 0
    new_mapping = {parq_ix: {}}
    for i, row in df.iterrows():
        fam_id = row["pfam_acc"]
        size_mb = row["size_mb"]
        parquet_file = row["parquet_file"]
        if parquet_file not in new_mapping[parq_ix]:
            new_mapping[parq_ix][parquet_file] = []
        new_mapping[parq_ix][parquet_file].append(fam_id)
        current_size += size_mb
        if current_size > limit_mb:
            print(f"parquet {parq_ix} size: {current_size}")
            parq_ix += 1
            current_size = 0
            new_mapping[parq_ix] = {}
    print(f"saved new mapping to {mapping_path}")
    with open(mapping_path, "w") as f:
        json.dump(new_mapping, f, indent=2)


def reformat_df(df, max_mb_per_entry, name, outdir):
    new_rows = []
    for i, row in df.iterrows():
        fam_id = row["pfam_acc"]
        fasta_iterator = read_fasta_lines(
            row["text"].split("\n"),
            keep_gaps=True,
            keep_insertions=True,
            to_upper=False,
        )

        names, seqs = [], []
        seqs_mb = 0
        sub_fam_counter = 0
        for name, seq in fasta_iterator:
            names.append(name)
            seqs.append(seq)
            seqs_mb += sys.getsizeof(seq) / 1024 / 1024
            if seqs_mb > max_mb_per_entry:
                save_path = (
                    f"{outdir}/{name.split('_')[0]}_{fam_id}_{sub_fam_counter}.parquet"
                )
                # save to parquet and reset
                df = pd.DataFrame(
                    [
                        {
                            "fam_id": fam_id,
                            "accessions": names,
                            "sequences": seqs,
                        }
                    ]
                )
                df.to_parquet(save_path)
                names, seqs = [], []
                seqs_mb = 0
                sub_fam_counter += 1
        new_rows.append(
            {
                "fam_id": fam_id,
                "accessions": names,
                "sequences": seqs,
            }
        )
    return pd.DataFrame(new_rows)

def shuffle_pfam_parquets(indir, outdir, limit_mb):
    mapping_path = f"{outdir}/new_pfam_parquet_map.json"
    if not os.path.exists(mapping_path):
        mapping = create_parquet_map(indir, mapping_path, limit_mb=limit_mb)
    else:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    for k, v in mapping.items():
        new_dom_path = f"{outdir}/Domain_{str(k).zfill(3)}.parquet"
        new_fam_path = f"{outdir}/Family_{str(k).zfill(3)}.parquet"
        if os.path.exists(new_dom_path) and os.path.exists(new_fam_path):
            continue
        entries_fam = []
        entries_dom = []
        for old_parq_path in v.keys():
            old_df = pd.read_parquet(f"{indir}/{old_parq_path}")
            new_df = old_df[old_df["pfam_acc"].isin(v[old_parq_path])]
            new_df = reformat_df(
                new_df, max_mb_per_entry=limit_mb, name=old_parq_path, outdir=outdir
            )
            if new_df.empty:
                continue
            if old_parq_path.startswith("Domain"):
                entries_dom.append(new_df)
            elif old_parq_path.startswith("Family"):
                entries_fam.append(new_df)
            else:
                raise ValueError(f"unknown parquet type: {old_parq_path}")

        if len(entries_fam):
            df_fam = pd.concat(entries_fam)
            df_fam = df_fam.sample(frac=1).reset_index(drop=True)
            df_fam.to_parquet(new_fam_path, index=False)

        if len(entries_dom):
            df_dom = pd.concat(entries_dom)
            df_dom = df_dom.sample(frac=1).reset_index(drop=True)
            df_dom.to_parquet(new_dom_path, index=False)

if __name__ == "__main__":
    outdir = "../data/pfam/shuffled_parquets"
    os.makedirs(outdir, exist_ok=True)
    indir = "../data/pfam/combined_parquets"
    limit_mb = 250
    shuffle_pfam_parquets(
        indir=indir,
        outdir=outdir,
        limit_mb=limit_mb
    )

