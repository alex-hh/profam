import glob
import os
import re
import shutil

try:
    from src.sequence.fasta import fasta_generator, output_fasta  # type: ignore
except Exception:
    fasta_generator = None  # type: ignore
    output_fasta = None  # type: ignore


# /generated_sequences_funfams_val_3_90_1170_40-FF-000004_rep_seq_id999/3.90.1170.40-FF-000004_rep_seq.id999_samples20_seed42.fasta

pattern = "/mnt/disk2/ProteinGym_PoET_benchmark/ProteinGym/proteingym/baselines/PoET/data/generated_sequences_*/*_samples20_seed42.fasta"


def _fallback_fasta_generator(filepath):
    desc = None
    seq = None
    with open(filepath, "r") as fin:
        for line in fin:
            if line and line[0] == ">":
                if seq is not None:
                    assert isinstance(desc, str)
                    yield desc, seq
                desc = line.strip()[1:]
                seq = ""
            else:
                assert isinstance(seq, (str, type(None)))
                if seq is None:
                    # Skip until we see a header
                    continue
                seq += line.strip()
        if seq is not None and desc is not None:
            yield desc, seq


def _get_ith_sequence(filepath, index_one_based):
    gen = (
        fasta_generator(filepath)
        if callable(fasta_generator)
        else _fallback_fasta_generator(filepath)
    )
    for i, (h, s) in enumerate(gen, start=1):
        if i == index_one_based:
            return h, s
    return None


# Track created single-sequence FASTAs per output directory to write list files
created_single_by_dir = {}
single_sequence_directory = "../sampling_results/poet_funfam_foldseek_gen10_combined"
os.makedirs(single_sequence_directory, exist_ok=True)
for fasta in glob.glob(pattern):
    if "funfams" in fasta:
        ds = "funfams"
    elif "foldseek" in fasta:
        ds = "foldseek"
    else:
        raise ValueError(f"Unknown dataset in {fasta}")
    if "_val_" in fasta:
        split = "val"
    elif "_test_" in fasta:
        split = "test"
    else:
        raise ValueError(f"Unknown split in {fasta}")

    new_dir = f"../sampling_results/{ds}_{split}/poet/"
    os.makedirs(new_dir, exist_ok=True)

    # Copy original multi-sequence FASTA into target directory
    copied_name = os.path.basename(fasta).replace(
        "_samples20_seed42.fasta", "_generated.fasta"
    )
    new_path = f"{new_dir}/{copied_name}"
    shutil.copy(fasta, new_path)

    # Create single-sequence FASTA using the 10th sequence (index 10)
    orig_stem = os.path.splitext(os.path.basename(fasta))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", orig_stem)
    single_name = f"{ds}_{split}_{safe_stem}_gen10.fasta"
    single_path = os.path.abspath(os.path.join(single_sequence_directory, single_name))

    tenth = _get_ith_sequence(fasta, 10)
    print(f"Tenth sequence: {tenth[1]}")
    if tenth is None:
        print(f"Warning: fewer than 10 sequences in {fasta}; skipping single FASTA.")
    else:
        header, seq = tenth
        try:
            if callable(output_fasta):
                output_fasta([header], [seq], single_path)  # type: ignore
            else:
                with open(single_path, "w") as fout:
                    fout.write(f">{header}\n{seq}\n")
        except Exception as e:
            print(f"Error writing single FASTA {single_path}: {e}")
        else:
            created_single_by_dir.setdefault(single_sequence_directory, []).append(os.path.basename(single_path))

# Write fasta_file_list.txt for each output directory containing single FASTAs
for out_dir, paths in created_single_by_dir.items():
    list_path = os.path.abspath(os.path.join(single_sequence_directory, "fasta_file_list.txt"))
    try:
        with open(list_path, "w") as f:
            for p in paths:
                f.write(p + "\n")
    except Exception as e:
        print(f"Error writing list file {list_path}: {e}")
