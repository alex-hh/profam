import glob
import os
import shutil
import subprocess
import tempfile
from Bio import SeqIO


HHFILTER_BINARY = os.environ.get("HHFILTER_BINARY", "/mnt/disk2/msa_pairformer/hhsuite/hhfilter")


def _run(cmd):
    print(f"[cmd] {cmd}")
    return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def align_all_to_first_with_mafft(input_fasta, output_aln_fasta, threads=12):
    """
    Align all sequences to the first sequence in the FASTA using MAFFT, keeping
    the first sequence ungapped (anchored reference).

    Command pattern:
      mafft --thread N --keeplength --addfull others.fa first.fa > output.fasta
    """
    records = list(SeqIO.parse(input_fasta, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in {input_fasta}")

    os.makedirs(os.path.dirname(output_aln_fasta), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mafft_anchor_") as tmpdir:
        first_path = os.path.join(tmpdir, "first.fasta")
        others_path = os.path.join(tmpdir, "others.fasta")

        # Write first sequence
        with open(first_path, "w") as f_first:
            SeqIO.write([records[0]], f_first, "fasta")

        # Write remaining sequences
        if len(records) > 1:
            with open(others_path, "w") as f_others:
                SeqIO.write(records[1:], f_others, "fasta")
        else:
            # Only one sequence: just copy as alignment
            shutil.copyfile(first_path, output_aln_fasta)
            return output_aln_fasta

        cmd = (
            f"mafft --thread {int(threads)} --keeplength --addfull {others_path} {first_path} > {output_aln_fasta}"
        )
        _run(cmd)
    return output_aln_fasta


def run_hhfilter(input_path, output_path):
    """
    Run HHfilter with default parameters on an aligned FASTA (AFA).
    We explicitly set -M afa to treat the input/output as aligned FASTA.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = f"{HHFILTER_BINARY} -i {input_path} -o {output_path} -M afa"
    _run(cmd)
    return output_path


def write_one_line_fasta(record, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f">{record.id}\n")
        f.write(str(record.seq).replace("\n", "").replace("\r", "") + "\n")
    return output_path


if __name__ == "__main__":
    ec_pattern = "../data/ec/ec_fastas/*"
    output_dir = "../data/ec/ec_validation_dataset"
    single_sequences_dir = f"{output_dir}/ec_single_sequences"
    ec_paths = glob.glob(ec_pattern)
    min_seqs_for_val = 50
    threads = int(os.environ.get("MAFFT_THREADS", "12"))

    for ec_path in ec_paths:
        try:
            ec_id = os.path.basename(ec_path).split(".")[0]
            records = list(SeqIO.parse(ec_path, "fasta"))
            align_dir = os.path.join(output_dir, "alignments")
            os.makedirs(align_dir, exist_ok=True)
            msa_path = os.path.join(align_dir, f"{ec_id}_aln.fasta")
            filtered_msa_path = os.path.join(align_dir, f"{ec_id}_aln.filtered.fasta")
            single_seq_path = os.path.join(single_sequences_dir, f"{ec_id}.fasta")
            if len(records) < min_seqs_for_val:
                print(f"Skipping {ec_id}: only {len(records)} sequences (< {min_seqs_for_val})")
                for p in [msa_path, filtered_msa_path, single_seq_path]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                continue
            elif os.path.exists(filtered_msa_path):
                records = list(SeqIO.parse(filtered_msa_path, "fasta"))
                if len(records) < min_seqs_for_val:
                    print(f"Removing {ec_id}: only {len(records)} sequences (< {min_seqs_for_val})")
                    for p in [msa_path, filtered_msa_path, single_seq_path]:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except OSError:
                            pass
                    continue


            # 1) Align all sequences to the first
            if not os.path.exists(msa_path):
                try:
                    align_all_to_first_with_mafft(ec_path, msa_path, threads=threads)
                except subprocess.CalledProcessError as e:
                    print(e.stdout)
                    print(e.stderr)
                    print(f"Error aligning {ec_id}")
                    continue
                except Exception as e:
                    print(f"Error aligning {ec_id}: {e}")
                    continue

            # 2) Run HHfilter with default params on aligned FASTA
            if not os.path.exists(filtered_msa_path):
                try:
                    run_hhfilter(msa_path, filtered_msa_path)
                except subprocess.CalledProcessError as e:
                    print(e.stdout)
                    print(e.stderr)
                    print(f"Error running hhfilter for {ec_id}")
                    # If hhfilter fails, discard outputs for this EC
                    if os.path.exists(msa_path):
                        os.remove(msa_path)
                    if os.path.exists(filtered_msa_path):
                        os.remove(filtered_msa_path)
                    continue
                except Exception as e:
                    print(f"Error running hhfilter for {ec_id}: {e}")
                    if os.path.exists(msa_path):
                        os.remove(msa_path)
                    if os.path.exists(filtered_msa_path):
                        os.remove(filtered_msa_path)
                    continue

            # 3) Count sequences remaining after filtering
            try:
                remaining = list(SeqIO.parse(filtered_msa_path, "fasta"))
                num_remaining = len(remaining)
            except Exception as e:
                print(f"Error reading filtered MSA for {ec_id}: {e}")
                num_remaining = 0

            # 4) If below threshold, discard this EC (delete files)
            if num_remaining < min_seqs_for_val:
                print(
                    f"Discarding {ec_id}: {num_remaining} sequences after hhfilter (< {min_seqs_for_val}). Deleting files."
                )
                for p in [msa_path, filtered_msa_path]:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except OSError:
                        pass
                # Also remove single-seq file if created previously
                single_seq_path = os.path.join(single_sequences_dir, f"{ec_id}.fasta")
                if os.path.exists(single_seq_path):
                    try:
                        os.remove(single_seq_path)
                    except OSError:
                        pass
                continue

            # 5) Keep: write a one-line FASTA containing only the first sequence
            try:
                write_one_line_fasta(records[0], single_seq_path)
                print(
                    f"Kept {ec_id}: {num_remaining} sequences after hhfilter. Wrote first sequence to {single_seq_path}"
                )
            except Exception as e:
                print(f"Error writing single-seq FASTA for {ec_id}: {e}")
                # Do not discard; alignment already validated
                continue
        except Exception as e:
            print(f"Unexpected error for {ec_path}: {e}")



