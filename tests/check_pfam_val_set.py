import glob
import os

import pandas as pd

from src.data.fasta import read_fasta_lines

fasta_dir = (
    "/mnt/disk2/cath_plm/data/pfam/pfam_eval_splits/clustered_split_fastas_debug"
)

prompts = sorted(glob.glob(f"{fasta_dir}/*_train.fasta"))
completions = sorted(glob.glob(f"{fasta_dir}/*_test.fasta"))
assert len(prompts) == len(completions)

all_c = []
all_p = []

for p, c in zip(prompts, completions):
    assert p.split("/")[-1].split("_")[0] == c.split("/")[-1].split("_")[0]
    with open(p, "r") as f:
        prompt = f.read()
    with open(c, "r") as f:
        completion = f.read()

    prompt = prompt.split("\n")
    completion = completion.split("\n")

    p_names = []
    p_seqs = []

    prompt = read_fasta_lines(
        prompt,
        keep_gaps=True,
        keep_insertions=True,
        to_upper=True,
    )

    for n, s in prompt:
        p_names.append(n)
        p_seqs.append(s)

    comps = read_fasta_lines(
        completion,
        keep_gaps=True,
        keep_insertions=True,
        to_upper=True,
    )

    c_names = []
    c_seqs = []
    for n, s in comps:
        c_names.append(n)
        c_seqs.append(s)
    all_p.extend(p_names)
    all_c.extend(c_names[:4])

print(f"n prompts seqs: {len(all_p)}")
print(f"n completions seqs: {len(all_c)}")
print(f"n eval files: {len(completions)}")
print(f"n prompt files: {len(prompts)}")
assert set(p_seqs).intersection(set(c_seqs)) == set()


bp = 1
