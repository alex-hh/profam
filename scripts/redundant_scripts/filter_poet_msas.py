import numpy as np
import glob
import os
from MSA_Pairformer.dataset import MSA


def write_filtered_fasta(msa_obj, final_indices, final_seqs_arr, name, out_dir="filtered_msas_poet"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_filtered.fasta")
    with open(out_path, "w") as fh:
        for i, orig_idx in enumerate(final_indices):
            header_id = msa_obj.ids_l[orig_idx]
            seq_str = "".join(final_seqs_arr[i])
            fh.write(f">{header_id}\n")
            fh.write(f"{seq_str}\n")
    return out_path


def compute_identity_to_query(msa_arr):
    # msa_arr: np.ndarray [depth, length] of characters
    if msa_arr.shape[0] == 0:
        return np.array([])
    query = msa_arr[0]
    non_gap_mask = (query != '-')
    identities = []
    for i in range(msa_arr.shape[0]):
        seq = msa_arr[i]
        both_non_gap = non_gap_mask & (seq != '-')
        denom = both_non_gap.sum()
        if denom == 0:
            identities.append(0.0)
        else:
            identities.append(((seq[both_non_gap] == query[both_non_gap]).sum()) / denom)
    return np.array(identities)


def prefilter_by_identity(msa_arr, min_identity=0.15):
    identities = compute_identity_to_query(msa_arr)
    # Always keep the query (index 0)
    keep_mask = identities >= min_identity
    keep_mask[0] = True
    kept_indices = np.where(keep_mask)[0]
    return msa_arr[kept_indices], kept_indices


def mean_non_gap_length(msa_arr):
    if msa_arr.shape[0] == 0:
        return 0.0
    non_gap_counts = (msa_arr != '-').sum(axis=1)
    return float(non_gap_counts.mean())


def select_with_hhfilter_and_relax(msa_obj, prefiltered_msa, target_depth, hhfilter_binary):
    # Progressive relaxation of hhfilter parameters
    param_grid = [

        dict(seq_id=90, cov=70, qid=15, qsc=-30.0),
        dict(seq_id=90, cov=65, qid=15, qsc=-30.0),
        dict(seq_id=90, cov=60, qid=15, qsc=-30.0),
        dict(seq_id=90, cov=55, qid=15, qsc=-30.0),

        dict(seq_id=95, cov=60, qid=10, qsc=-40.0),
        dict(seq_id=95, cov=55, qid=10, qsc=-40.0),

        dict(seq_id=90, cov=50, qid=15, qsc=-45.0),
        dict(seq_id=90, cov=45, qid=15, qsc=-45.0),

        dict(seq_id=95, cov=50, qid=10, qsc=-40.0),
        dict(seq_id=95, cov=45, qid=10, qsc=-40.0),

        dict(seq_id=90, cov=40, qid=15, qsc=-50.0),
        dict(seq_id=90, cov=35, qid=15, qsc=-50.0),

        dict(seq_id=95, cov=40, qid=10, qsc=-40.0),
        dict(seq_id=95, cov=35, qid=10, qsc=-40.0),

        dict(seq_id=90, cov=30, qid=15, qsc=-55.0),
        dict(seq_id=90, cov=25, qid=15, qsc=-55.0),

        dict(seq_id=90, cov=30, qid=15, qsc=-65.0),
        dict(seq_id=90, cov=25, qid=15, qsc=-65.0),

        dict(seq_id=90, cov=15, qid=15, qsc=-85.0),
        dict(seq_id=90, cov=10, qid=15, qsc=-95.0),

        dict(seq_id=95, cov=15, qid=5,  qsc=-95.0),
        dict(seq_id=95, cov=10, qid=0, qsc=-100.0),
        dict(seq_id=95, cov=7, qid=0, qsc=-100.0),
        dict(seq_id=95, cov=5, qid=0, qsc=-100.0),

    ]

    best_msa = None
    best_indices = None

    if msa_obj.n_diverse_seqs <= target_depth:
        param_grid = param_grid[-1:]

    for params in param_grid:
        filtered_msa, kept_idx = msa_obj.hhfilter_select(
            prefiltered_msa,
            M="a3m",
            seq_id=params["seq_id"],
            diff=int(target_depth),
            cov=params["cov"],
            qid=params["qid"],
            qsc=params["qsc"],
            binary=hhfilter_binary,
        )

        # Track the best (largest) in case we never meet target
        if best_msa is None or filtered_msa.shape[0] > best_msa.shape[0]:
            best_msa, best_indices = filtered_msa, kept_idx

        if filtered_msa.shape[0] >= target_depth:
            print(params)
            return filtered_msa, kept_idx


    # Otherwise, return the best we achieved
    print(params)
    return best_msa, best_indices



TARGET_NON_GAP_TOKENS = 800000


# msa_file_pattern = "/mnt/disk2/cath_plm/data/ProteinGym/DMS_msa_files/*reformat.a3m"
msa_file_pattern = "/mnt/disk2/cath_plm/data/ProteinGym/PoET_DMS_msa_files/DMS_substitutions/*.a3m"
hhfilter_binary = "/mnt/disk2/msa_pairformer/hhsuite/hhfilter"

for msa_file in glob.glob(msa_file_pattern):
    name = msa_file.split("/")[-1].split(".")[0]
    total_length = 4096

    # Build MSA object without applying selection yet
    np.random.seed(42)
    msa_obj = MSA(
        msa_file_path=msa_file,
        max_seqs=10**9,  # large cap; we'll control depth ourselves
        max_length=total_length,
        max_tokens=10**12,
        diverse_select_method="none",
        hhfilter_kwargs={"binary": hhfilter_binary}
    )

    # Work on cropped MSA
    cropped_msa = msa_obj.random_crop  # [depth, length] array of characters

    # 1) Pre-filter by identity >= 15% to query
    prefiltered_msa, prefilter_indices = prefilter_by_identity(cropped_msa, min_identity=0.15)

    if prefiltered_msa.shape[0] == 0:
        print(f"{name}: No sequences passed identity prefilter; skipping.")
        continue

    # 2) Compute dynamic target depth to reach ~800k non-gap tokens
    mean_len = mean_non_gap_length(prefiltered_msa)
    if mean_len <= 0:
        print(f"{name}: Mean non-gap length is zero; skipping.")
        continue
    target_depth = int(max(1, round(TARGET_NON_GAP_TOKENS / mean_len)))

    # 3) Apply hhfilter with relaxation; then greedy fallback if needed
    filtered_msa, kept_idx_prefilter = select_with_hhfilter_and_relax(
        msa_obj, prefiltered_msa, target_depth, hhfilter_binary
    )

    # Map kept indices back to original indices of msa_obj.ids_l
    final_indices = [int(prefilter_indices[i]) for i in kept_idx_prefilter]

    out_fasta = write_filtered_fasta(
        msa_obj=msa_obj,
        final_indices=final_indices,
        final_seqs_arr=filtered_msa,
        name=name,
        out_dir="filtered_msas_poet",
    )
    print(f"Wrote filtered FASTA: {out_fasta}")
