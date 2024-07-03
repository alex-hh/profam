import os
import pyarrow as pa
import pyarrow.parquet as pq

def process_pfam_file(input_file, output_dir):
    current_msa = []
    current_type = None
    file_count = {'Family': 0, 'Domain': 0}
    msa_count = {'Family': 0, 'Domain': 0}
    fasta_buffer = {'Family': [], 'Domain': []}

    def save_to_parquet(msa_type):
        nonlocal fasta_buffer, file_count
        if fasta_buffer[msa_type]:
            table = pa.table({'text': fasta_buffer[msa_type]})
            output_file = os.path.join(output_dir, msa_type, f"{msa_type}_{file_count[msa_type]}.parquet")
            pq.write_table(table, output_file)
            fasta_buffer[msa_type] = []
            file_count[msa_type] += 1
    line_counter = 0
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith("# STOCKHOLM 1.0"):
                current_msa = []
                current_type = None
            elif line.startswith("TP "):
                current_type = line.strip().split()[-1]
            elif line.startswith("#=GF ID "):
                msa_id = line.strip().split()[-1]
            elif not line.startswith("#") and line.strip() and current_type in ['Family', 'Domain']:
                parts = line.strip().split()
                if len(parts) >= 2:
                    current_msa.append(f">{parts[0]}\n{parts[1]}\n")
            elif line.strip() == "//" and current_msa and current_type in ['Family', 'Domain']:
                fasta_content = f">{msa_id}\n" + ''.join(current_msa)
                fasta_buffer[current_type].append(fasta_content)
                msa_count[current_type] += 1

                if msa_count[current_type] % 1000 == 0:
                    save_to_parquet(current_type)
            line_counter += 1
            if line_counter % 10000 == 0:
                print(f"Procd {line_counter} of 242m")

    # Save any remaining MSAs
    for msa_type in ['Family', 'Domain']:
        if fasta_buffer[msa_type]:
            save_to_parquet(msa_type)

    print(f"Processed {msa_count['Family']} families and {msa_count['Domain']} domains.")
    print(f"Created {file_count['Family']} family parquet files and {file_count['Domain']} domain parquet files.")

# Usage
input_file = "/SAN/orengolab/cath_plm/ProFam/data/pfam/Pfam-A.full"
output_dir = "/SAN/orengolab/cath_plm/ProFam/data/pfam/proccessed_pfam_files"

# Create output directories
os.makedirs(os.path.join(output_dir, "Family"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "Domain"), exist_ok=True)

process_pfam_file(input_file, output_dir)