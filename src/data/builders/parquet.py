from typing import List, Optional

import pandas as pd

from src.data.objects import Protein
from src.data.processors import backbone_coords_from_example


def build_representative_df(
    cluster_df, has_structure: bool = False, document_id_col="fam_id"
):
    """Assumes that the document id is the accession of the representative."""
    records = []
    for _, row in cluster_df.iterrows():
        rep_index = list(row["accessions"]).index(row[document_id_col])
        rep_dict = {
            "sequence": row["sequences"][rep_index],
            "accession": row["accessions"][rep_index],
            document_id_col: row[document_id_col],
            "length": len(row["sequences"][rep_index]),
        }
        if has_structure:
            rep_dict["plddt"] = (row["plddts"][rep_index],)
            rep_dict["N"] = (row["N"][rep_index],)
            rep_dict["CA"] = (row["CA"][rep_index],)
            rep_dict["C"] = (row["C"][rep_index],)
            rep_dict["O"] = (row["O"][rep_index],)
            rep_dict["mean_plddt"] = row["plddts"][rep_index].mean()
        records.append(rep_dict)
    return pd.DataFrame(records)


def export_protein_from_cluster_df(cluster_df, cluster_id, accession):
    row = cluster_df[(cluster_df["fam_id"] == cluster_id)].iloc[0]
    rep_index = list(row["accessions"]).index(accession)
    backbone_coords, _ = backbone_coords_from_example(row)[rep_index]
    protein = Protein(
        sequence=row["sequences"][rep_index],
        accession=row["accessions"][rep_index],
        plddt=row["plddts"][rep_index],
        backbone_coords=backbone_coords,
    )
    return protein


def export_pdb_from_cluster_df(cluster_df, cluster_id, accession, filepath):
    protein = export_protein_from_cluster_df(cluster_df, cluster_id, accession)
    protein.to_pdb_file(filepath)
