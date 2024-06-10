"""
Compute the similarity and coverage
between the query sequence and each
matched sequence in the MSA.
Append the results to the identifier
row in the MSA fasta file
"""

import os
import re
import json
import argparse
import glob
import numpy as np
# import matplotlib.pyplot as plt

def add_coverage_seq_identity_to_msa(msa_path, debug=False):
    with open(msa_path, "r") as f:
        lines = f.readlines()
    up_ids = [l.split("|")[1] for l in lines if l.startswith(">")]
    seq_lines = [np.frombuffer(re.sub(r'[a-z]', '', l.strip()).encode(), dtype='S1') for l in lines if not l.startswith(">")]
    max_length = max(arr.size for arr in seq_lines)
    aligned_arrays = np.full((len(seq_lines), max_length), b'-', dtype='S1')
    for i, arr in enumerate(seq_lines):
        aligned_arrays[i, :arr.size] = arr
    non_gap_positions = (aligned_arrays != b'-').sum(axis=1)
    coverage = (non_gap_positions / non_gap_positions[0])
    seq_identity = (aligned_arrays[0] == aligned_arrays).mean(axis=1)
    # if debug:
        # plt.imshow(aligned_arrays[0] == aligned_arrays, interpolation='none')
        # plt.title(f'seqID: {round(seq_identity.mean(), 3)}, \ncove: {round(coverage.mean(), 3)}')
        # plt.tight_layout()
        # plt.show()
        #
        # plt.imshow((aligned_arrays != b'-'))
        # plt.tight_layout()
        # plt.show()
    assert len(up_ids) == len(seq_identity) == len(coverage)
    results = {}
    for idx, up_id in enumerate(up_ids):
        results[up_id] = {
            "id": round(seq_identity[idx], 2),
            "cov": round(coverage[idx], 2)
        }
    return results



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_index", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=1)
    args = parser.parse_args()
    msa_dir = "/SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_filtered"
    json_save_dir = '/SAN/orengolab/cath_plm/ProFam/data/openfold/coverage_identity_jsons'
    if not os.path.exists(msa_dir):
        debug=True
        msa_dir = "data/example_data/openfold/uniclust30_filtered"
        json_save_dir = 'data/example_data/openfold/coverage_identity_jsons'
    msa_paths = sorted(list(glob.glob(f'{msa_dir}/*/a3m/uniclust30.a3m')))
    print(f'Found {len(msa_paths)} MSA files')
    batch_size = len(msa_paths) // args.num_tasks + 1
    msa_paths = msa_paths[args.task_index*batch_size:(args.task_index+1)*batch_size]
    print(f'Processing {len(msa_paths)} MSA files, task index: {args.task_index}')
    os.makedirs(json_save_dir, exist_ok=True)
    results_dict = {}
    with open(f"{json_save_dir}/task_{str(args.task_index).zfill(2)}.json", "w") as f:
        for msa_path in msa_paths:
            try:
                uniprot_id = msa_path.split("/")[-3]
                result = add_coverage_seq_identity_to_msa(msa_path, debug=debug)
                assert uniprot_id not in results_dict
                results_dict[uniprot_id] = result
            except Exception as e:
                print(f"Error processing {msa_path}")
                print(e)
        json.dump(results_dict, f, indent=4)

