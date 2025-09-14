import os
import glob
import json
import time
import urllib.request
import urllib.error
import shutil



ALPHAFOLD_FILE_TEMPLATE_V4 = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
ALPHAFOLD_FILE_TEMPLATE_V3 = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v3.pdb"
ALPHAFOLD_API_PREDICTION = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"


def _download_to_path(url: str, dest_path: str, timeout: int = 30, max_retries: int = 3, retry_backoff_sec: float = 1.5) -> bool:
    """Download URL to dest_path. Returns True on success, False otherwise."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status != 200:
                    last_error = urllib.error.HTTPError(url, response.status, "Unexpected status", hdrs=None, fp=None)
                    raise last_error
                with open(dest_path, "wb") as out_f:
                    shutil.copyfileobj(response, out_f)
            return True
        except urllib.error.HTTPError as e:
            # 404s shouldn't be retried
            if e.code == 404:
                return False
            last_error = e
        except Exception as e:
            last_error = e

        if attempt < max_retries:
            time.sleep(retry_backoff_sec * attempt)

    if last_error is not None:
        print(f"Failed to download {url} -> {dest_path}: {last_error}")
    return False


def _fetch_pdb_url_from_api(uniprot_id: str) -> str | None:
    api_url = ALPHAFOLD_API_PREDICTION.format(uniprot_id=uniprot_id)
    try:
        with urllib.request.urlopen(api_url, timeout=20) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    if isinstance(payload, list) and len(payload) > 0:
        # Prefer first entry
        first = payload[0]
        pdb_url = first.get("pdbUrl")
        if isinstance(pdb_url, str) and pdb_url.startswith("http"):
            return pdb_url
    return None


def fetch_alphafold_pdb(uniprot_id: str, pdb_path: str) -> bool:
    """Fetch AlphaFold PDB for a UniProt ID if missing. Returns True if file exists or was downloaded."""
    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
        return True

    # Try direct file URLs (v4 then v3)
    v4_url = ALPHAFOLD_FILE_TEMPLATE_V4.format(uniprot_id=uniprot_id)
    if _download_to_path(v4_url, pdb_path):
        return True

    v3_url = ALPHAFOLD_FILE_TEMPLATE_V3.format(uniprot_id=uniprot_id)
    if _download_to_path(v3_url, pdb_path):
        return True

    # Fallback to API discovery
    pdb_url = _fetch_pdb_url_from_api(uniprot_id)
    if pdb_url and _download_to_path(pdb_url, pdb_path):
        return True

    return False

def get_all_uniprot_ids(fasta_path):
    up_ids = set()
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                up_ids.add(line[1:].strip())
    return up_ids

if __name__ == "__main__":
    fasta_paths = glob.glob("../data/val_test_v2_fastas/foldseek/*/*.fasta")
    total = len(fasta_paths)
    missing = 0
    downloaded = 0

    for idx, fasta_path in enumerate(fasta_paths, start=1):
        rep_id = os.path.basename(fasta_path).split(".")[0]
        up_ids = get_all_uniprot_ids(fasta_path)
        if rep_id not in up_ids:
            up_ids.add(rep_id)
        for uniprot_id in up_ids:
            pdb_path = f"../data/val_test_v2_pdbs/foldseek/{uniprot_id}.pdb"
            if not os.path.exists(pdb_path):
                print(f"[{idx}/{total}] Fetching PDB for {uniprot_id}...")
                if fetch_alphafold_pdb(uniprot_id, pdb_path):
                    downloaded += 1
                    print(f"  Saved -> {pdb_path}")
                    time.sleep(1)
                else:
                    print(f"  Not found in AlphaFold DB: {uniprot_id}")
                    missing += 1
            else:
                print(f"[{idx}/{total}] Exists: {pdb_path}")

        print(f"Done. FASTAs: {total}, missing PDBs: {missing}, downloaded: {downloaded}")