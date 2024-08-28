import os
import argparse
import pyarrow as pa
import pyarrow.parquet as pq

"""
First run submit_scripts/grep_pfam_breaks.qsub.sh
to generate pfam_end_grepper.txt (containing line-numbers)
where each family ends.

Each job processes a certain number of families
and saves them in a parquet file.

Subsequently run:
scripts/shuffle_pfam_parquets.py
data_creation_scripts/pfam/deduplicate_pfam.py
data_creation_scripts/pfam/create_pfam_eval_fastas.py
data_creation_scripts/pfam/train_test_split_pfam_parquets.py
"""

def read_end_lines(grepper_file):
    with open(grepper_file, 'r') as f:
        return [int(line.split(':')[0]) for line in f]

def process_pfam_batch(pfam_file, start_line, end_lines, output_dir):
    current_msa = []
    current_type = None
    msa_id = None
    pfam_acc = None
    fasta_buffer = {'Family': [], 'Domain': []}
    msa_count = {'Family': 0, 'Domain': 0}
    current_chunk_size = 0

    def save_to_parquet(msa_type):
        if fasta_buffer[msa_type]:
            texts, msa_ids, pfam_accs = zip(*fasta_buffer[msa_type])
            table = pa.table({
                'text': texts,
                'msa_id': msa_ids,
                'pfam_acc': pfam_accs
            })
            output_file = os.path.join(output_dir, msa_type, f"{msa_type}_{start_line}_{end_lines[-1]}_{msa_count[msa_type]}.parquet")
            pq.write_table(table, output_file)
            fasta_buffer[msa_type] = []

    def add_to_fasta_buffer(msa_type, fasta_content, msa_id, pfam_acc):
        fasta_buffer[msa_type].append((fasta_content, msa_id, pfam_acc))
        msa_count[msa_type] += 1
        if len(fasta_buffer[msa_type]) >= 100:  # Save every 100 MSAs
            save_to_parquet(msa_type)

    with open(pfam_file, 'r', encoding='latin-1', errors='replace') as f:
        # Skip to start_line
        for _ in range(start_line - 1):
            next(f)

        for line_num, line in enumerate(f, start=start_line):
            if line.startswith("# STOCKHOLM 1.0"):
                if current_msa and current_type in ['Family', 'Domain']:
                    fasta_content = ''.join(current_msa)
                    add_to_fasta_buffer(current_type, fasta_content, msa_id, pfam_acc)
                current_msa = []
                current_type = None
                msa_id = None
                pfam_acc = None
                current_chunk_size = 0
            elif line.startswith("#=GF TP"):
                current_type = line.strip().split()[-1]
            elif line.startswith("#=GF ID"):
                msa_id = line.strip().split()[-1]
            elif line.startswith("#=GF AC"):
                pfam_acc = line.strip().split()[-1]
            elif not line.startswith("#") and line.strip() and current_type in ['Family', 'Domain']:
                parts = line.strip().split()
                if len(parts) >= 2:
                    sequence = f">{parts[0]}\n{parts[1]}\n"
                    current_msa.append(sequence)
                    current_chunk_size += len(sequence.encode('utf-8'))
                    if current_chunk_size > 1_000_000_000:  # 1 billion bytes
                        fasta_content = ''.join(current_msa)
                        add_to_fasta_buffer(current_type, fasta_content, msa_id, pfam_acc)
                        current_msa = []
                        current_chunk_size = 0

            if line.strip() == "//" or line_num >= end_lines[-1]:
                if current_msa and current_type in ['Family', 'Domain']:
                    fasta_content = ''.join(current_msa)
                    add_to_fasta_buffer(current_type, fasta_content, msa_id, pfam_acc)
                if line_num >= end_lines[-1]:
                    break

    # Save any remaining MSAs
    for msa_type in ['Family', 'Domain']:
        print(f"Saving {len(fasta_buffer[msa_type])} {msa_type} MSAs")
        if fasta_buffer[msa_type]:
            save_to_parquet(msa_type)

    return msa_count

def main(args):
    grepper_file = "/SAN/orengolab/cath_plm/ProFam/data/pfam/pfam_end_grepper.txt"
    pfam_file = "/SAN/orengolab/cath_plm/ProFam/data/pfam/Pfam-A.full"
    output_dir = "/SAN/orengolab/cath_plm/ProFam/data/pfam/array_processed"

    end_lines = read_end_lines(grepper_file)
    total_msas = len(end_lines)
    msas_per_task = (total_msas // args.num_tasks) + 1
    start_index = args.task_index * msas_per_task
    end_index = (args.task_index + 1) * msas_per_task if args.task_index < args.num_tasks - 1 else total_msas
    start_line = 1 if args.task_index == 0 else end_lines[start_index - 1] + 1
    batch_end_lines = end_lines[start_index:end_index]
    print(f"processing lines {start_line} to {batch_end_lines[-1]}")
    print(f"num_msas =  {len(batch_end_lines)}")
    os.makedirs(os.path.join(output_dir, "Family"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "Domain"), exist_ok=True)
    family_file = os.path.join(output_dir, "Family", f"Family_{start_line}_{batch_end_lines[-1]}.parquet")
    domain_file = os.path.join(output_dir, "Domain", f"Domain_{start_line}_{batch_end_lines[-1]}.parquet")
    print(f"family path: {family_file}")
    print(f"domain path: {domain_file}")
    if os.path.exists(family_file) and os.path.exists(domain_file):
        print(f"Task {args.task_index}: Files already exist. Skipping.")
        return

    msa_count = process_pfam_batch(pfam_file, start_line, batch_end_lines, output_dir)

    print(f"Task {args.task_index}: Processed {msa_count['Family']} families and {msa_count['Domain']} domains.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a batch of MSAs from Pfam file")
    parser.add_argument("--task_index", type=int, required=True, help="Task index (0-based)")
    parser.add_argument("--num_tasks", type=int, required=True, help="Total number of tasks")
    args = parser.parse_args()

    main(args)
