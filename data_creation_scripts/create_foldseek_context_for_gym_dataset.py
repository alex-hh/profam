"""
Created by Jude Wells 2025-04-24

PROBLEM WITH PFAM IS THAT FAMILIES OFTEN ONLY CONTAIN SHORT ALIGNED
FRAGMENTS OF THE FULL CHAIN SEQUENCE WHICH IS USED IN PROTEINGYM
INDIVIDUAL SEQUENCE FRAGMENTS IN PFAM CAN VARY IN LENGTH BY 10X

We want to test if poor family model performance on ProteinGym
is caused by change in distribution between training families
and the gym MSAs

To test if this is the problem we create a new dataset for protein
gym where the evolutionary context is taken from the Pfam MSA.

Build a parquet file for protein Gym each row contains one DMS experiment

Columns are:
DMS_id
completion_sequences
context_sequences
target_sequence
DMS_scores

Pfam parquets are currently structured liks so:
>>> df = pd.read_parquet("train_Domain_022.parquet")
>>> df.shape
(25, 6)
>>> df.columns
Index(['fam_id', 'accessions', 'sequences', 'pfam_version',
       'family_uniprot_accessions', 'matched_accessions'],
      dtype='object')
>>> df.iloc[0]
fam_id                                                                 PF13914
accessions                   [A0A091VM24_NIPNI/134-266, A0A093P3D8_PYGAD/71...
sequences                    [................................................
pfam_version                                                                11
family_uniprot_accessions                                                   []
matched_accessions           [None, None, None, None, None, None, None, Non...
Name: 0, dtype: object

"""
import glob
import time
from typing import Iterable, List, Dict, Union

import numpy as np
import pandas as pd
import requests


def convert_uniprot_ids_to_uniprot_accessions(uniprot_ids):
    """Convert a collection of UniProt identifiers to primary UniProt accessions.

    The *ProteinGym* dataset stores protein identifiers either as
    *entry names* (e.g. ``TP53_HUMAN``) **or** already as accessions
    (e.g. ``P04637``).  Down-stream processing relies on having the
    canonical accession, so this helper resolves any entry names to
    their corresponding primary accession via the UniProt REST API
    (``https://rest.uniprot.org``).

    Parameters
    ----------
    uniprot_ids : Iterable[str]
        A list/series/iterable containing UniProt entry names or
        accessions.

    Returns
    -------
    List[str]
        A list of the same length as *uniprot_ids* where every element
        is the resolved accession (or *None* if the lookup failed).
    """

    def _looks_like_accession(uid: str) -> bool:
        """Heuristic check if *uid* already appears to be an accession."""
        # Typical accessions have length 6 (e.g. P04637) or 10 (e.g. A0A023GPI8)
        return len(uid) in (6, 10) and uid[0].isalpha() and uid[-1].isdigit()

    if isinstance(uniprot_ids, (pd.Series, list, tuple, set, np.ndarray)):
        ids_iterable: Iterable[str] = uniprot_ids
    else:
        raise TypeError("uniprot_ids should be a pandas Series, list, tuple, or set of strings")

    resolved_accessions: List[Union[str, None]] = []
    cache: Dict[str, Union[str, None]] = {}

    for uid in ids_iterable:
        # Re-use previous look-ups to avoid redundant API calls
        if uid in cache:
            resolved_accessions.append(cache[uid])
            continue

        if _looks_like_accession(uid):
            accession: Union[str, None] = uid  # Already an accession
        else:
            # Query UniProt REST API for this entry name
            url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    accession = r.json().get("primaryAccession")
                else:
                    accession = None
            except requests.RequestException:
                accession = None

            # Be gentle with the public UniProt server – small sleep to avoid
            # triggering rate limits when large batches are processed.
            time.sleep(0.1)

        cache[uid] = accession
        resolved_accessions.append(accession)

    return resolved_accessions

def get_all_uniprot_accessions_from_protein_gym_dataset(gym_df):
    uniprot_ids = gym_df["UniProt_ID"].unique()
    return convert_uniprot_ids_to_uniprot_accessions(uniprot_ids)


if __name__ == "__main__":
    pfam_parquet_pattern = "../data/pfam/train_test_split_parquets/train/*.parquet"
    gym_df_path = "../data/ProteinGym/DMS_substitutions.csv"
    gym_df = pd.read_csv(gym_df_path)

    # ------------------------------------------------------------------
    # Resolve UniProt identifiers to canonical accessions and augment the
    # ProteinGym dataframe with a new column. Persist the augmented file
    # so that downstream scripts do not need to resolve again.
    # ------------------------------------------------------------------
    gym_df["UniProt_Accession"] = pd.Series(
        convert_uniprot_ids_to_uniprot_accessions(gym_df["UniProt_ID"])
    )

    # Save the enriched dataframe – overwrite or write to a new file as desired
    gym_df.to_csv(gym_df_path, index=False)

    # For matching with Pfam data we only need the unique set of accessions
    all_gym_accessions = set(gym_df["UniProt_Accession"].dropna().unique())

    pfam_parquets = glob.glob(pfam_parquet_pattern)
    matched_pfam_rows = {}
    new_gym_df_rows = []
    for pfam_parquet in pfam_parquets:
        df = pd.read_parquet(pfam_parquet)
        for i, row in df.iterrows():
            accessions = [a.split("_")[0] for a in row["accessions"]]
            matched_accessions = set(accessions).intersection(all_gym_accessions)
            if matched_accessions:
                matched_pfam_rows[row["fam_id"]] = row
                for accession in matched_accessions:
                    gym_matched = gym_df[gym_df.UniProt_Accession == accession]
                    for _, gym_row in gym_matched.iterrows():
