import os
import json
import pandas as pd
import argparse


def load_index(index_path):
    with open(index_path, 'r') as f:
        return json.load(f)


def load_task_data(task_path):
    with open(task_path, 'r') as f:
        return json.load(f)


def parse_fasta(fasta_string):
    sequences = []
    lines = fasta_string.strip().split('\n')
    current_seq = []
    current_id = ""

    for line in lines:
        if line.startswith('>'):
            if current_id:
                sequences.append((current_id, ''.join(current_seq)))
                current_seq = []
            current_id = line[1:].split()[0]
        else:
            current_seq.append(line)

    if current_id:
        sequences.append((current_id, ''.join(current_seq)))

    return sequences


def filter_msa(msa, task_data, query_id, min_coverage=0.5, min_identity=0.2):
    filtered_msa = [msa[0]]  # Always keep the query sequence
    for seq_id, seq in msa[1:]:
        short_seq_id = seq_id.split('|')[1]
        if short_seq_id in task_data[query_id]:
            cov = task_data[query_id][short_seq_id]['cov']
            identity = task_data[query_id][short_seq_id]['id']
            if cov >= min_coverage and identity >= min_identity:
                filtered_msa.append((seq_id, seq))
        else:
            print(f"coverage and identity not found for {short_seq_id}")
    return filtered_msa


def msa_to_string(msa):
    return '\n'.join(f">{seq_id}\n{seq}" for seq_id, seq in msa)


def process_msa_file(input_file, output_file, index, task_data_dir):
    df = pd.read_parquet(input_file)
    filtered_msas = []

    for _, row in df.iterrows():
        msa_text = row['text']
        msa = parse_fasta(msa_text)
        query_id = msa[0][0].split('|')[1]

        task_file = index.get(query_id)
        if task_file:
            task_data = load_task_data(os.path.join(task_data_dir, task_file))
            filtered_msa = filter_msa(msa, task_data, query_id)
            print(f"Filtered {len(msa)} sequences to {len(filtered_msa)} sequences for {query_id}")
            filtered_msas.append(msa_to_string(filtered_msa))
        else:
            print(f"Task data not found for {query_id}")

    output_df = pd.DataFrame({'text': filtered_msas})
    output_df.to_parquet(output_file)


def main():
    parser = argparse.ArgumentParser(description="Filter MSAs based on sequence similarity and coverage.")
    parser.add_argument('--task_index', type=int, required=True, help="Task index (0-269)")
    args = parser.parse_args()

    input_dir = '../data/openfold/uniclust30_filtered_parquet'
    output_dir = '../data/openfold/filtered_msas'
    index_path = '../data/openfold/coverage_identity_jsons/index.json'
    task_data_dir = '../data/openfold/coverage_identity_jsons'

    os.makedirs(output_dir, exist_ok=True)

    index = load_index(index_path)

    input_file = os.path.join(input_dir, f'{args.task_index}.parquet')
    output_file = os.path.join(output_dir, f'{args.task_index}.parquet')

    process_msa_file(input_file, output_file, index, task_data_dir)


if __name__ == "__main__":
    main()