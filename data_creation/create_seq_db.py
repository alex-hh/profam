import os
import json
import sqlite3
import argparse
import math


def get_sequence_from_profam_db(uniprot_id, cursor):
    cursor.execute('SELECT sequence FROM sequences WHERE sequence_id = ?', (uniprot_id,))
    result = cursor.fetchone()
    return result[0] if result else None


def extract_sequence_from_msa_line(msa_file_path, uniprot_id):
    with open(msa_file_path, 'r') as msa_file:
        for line in msa_file:
            if line.startswith(f">{uniprot_id}"):
                sequence = next(msa_file).strip()
                return sequence.upper().replace('-', '')
    return None


def get_identity_coverage_values(query_id, uniprot_id, index_data):
    if query_id in index_data:
        task_file = index_data[query_id]
        with open(f"/SAN/orengolab/cath_plm/ProFam/data/openfold/coverage_identity_jsons/{task_file}",
                  'r') as task_file:
            task_data = json.load(task_file)
            if query_id in task_data and uniprot_id in task_data[query_id]:
                return task_data[query_id][uniprot_id]['id'], task_data[query_id][uniprot_id]['cov']
    return None, None


def create_sequence_header(uniprot_id, is_query, full_seq_retrieved, identity, coverage):
    header = f">{uniprot_id}|{'query' if is_query else 'match'}|{'profam_db' if full_seq_retrieved else 'msa'}|"
    header += f"identity:{identity if identity else 'NONE'}|coverage:{coverage if coverage else 'NONE'}\n"
    return header


def update_index_file(index_file_path, query_id, db_file, start_offset, sequence_length):
    index_data = {}
    if os.path.exists(index_file_path):
        with open(index_file_path, 'r') as index_file:
            index_data = json.load(index_file)
    index_data[query_id] = {
        'db': db_file,
        'start': start_offset,
        'length': sequence_length
    }
    with open(index_file_path, 'w') as index_file:
        json.dump(index_data, index_file, indent=2)


def create_sequence_databases(msa_dir, profam_db_file, index_file_path, db_file_prefix, task_index, n_tasks):
    msa_ids = sorted(os.listdir(msa_dir))
    num_msas_per_task = math.ceil(len(msa_ids) / n_tasks)
    start_index = task_index * num_msas_per_task
    end_index = min((task_index + 1) * num_msas_per_task, len(msa_ids))

    db_file = f"{db_file_prefix}{task_index}.db"
    conn = sqlite3.connect(profam_db_file)
    cursor = conn.cursor()

    msa_counter = 0
    line_counter = 0
    query_id_is_not_msa_id = 0
    no_seq_counter = 0
    line_error_counter = 0
    msa_error_counter = 0

    with open('/SAN/orengolab/cath_plm/ProFam/data/openfold/coverage_identity_jsons/index.json', 'r') as index_file:
        id_cov_index_data = json.load(index_file)
    with open(db_file, 'a') as db:
        for msa_id in msa_ids[start_index:end_index]:
            msa_counter += 1
            try:
                start_offset = db.tell()
                msa_path = f"{msa_dir}/{msa_id}/a3m/uniclust30.a3m"
                if not os.path.exists(msa_path):
                    continue

                with open(msa_path, 'r') as msa_file:
                    query_id = None
                    for line in msa_file:
                        line_counter += 1
                        try:
                            if line.startswith(">"):
                                uniprot_id = line.split("|")[1]
                                if query_id is None:
                                    query_id = uniprot_id
                                    is_query = True
                                    if query_id != msa_id:
                                        query_id_is_not_msa_id += 1
                                else:
                                    is_query = False

                                sequence = get_sequence_from_profam_db(uniprot_id, cursor)
                                full_seq_retrieved = True
                                if sequence is None:
                                    msa_line = next(msa_file)
                                    assert not msa_line.startswith(">")
                                    sequence = msa_line.strip().upper().replace('-', '')
                                    full_seq_retrieved = False
                                    no_seq_counter += 1
                                else:
                                    next(msa_file)  # Skip the sequence line in the MSA file
                                identity, coverage = get_identity_coverage_values(msa_id, uniprot_id, id_cov_index_data)
                                sequence_header = create_sequence_header(uniprot_id, is_query, full_seq_retrieved,
                                                                         identity, coverage)
                                db.write(sequence_header)
                                db.write(sequence + '\n')
                        except Exception as e:
                            print(f"Line error {e}")
                            line_error_counter += 1
                            continue
                        if line_counter == 10:
                            print(f"MSA {msa_counter} processed",
                                  f"MSA errors: {msa_error_counter}",
                                  f"lines processed: {line_counter}",
                                  f"line errors: {line_error_counter}",
                                  f"query_id is not MSA id: {query_id_is_not_msa_id}",
                                  f"no sequence in Profam: {no_seq_counter}")
                end_offset = db.tell()
                msa_length = end_offset - start_offset
                update_index_file(index_file_path, msa_id, db_file, start_offset, msa_length)

            except Exception as e:
                print(f"MSA error {e}")
                msa_error_counter += 1
            print(f"MSA {msa_counter} processed",
                  f"MSA errors: {msa_error_counter}",
                  f"lines processed: {line_counter}",
                  f"line errors: {line_error_counter}",
                  f"query_id is not MSA id: {query_id_is_not_msa_id}",
                  f"no sequence in Profam: {no_seq_counter}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create sequence databases and index files.')
    parser.add_argument('--task_index', type=int, required=True, help='Index of the current task.')
    parser.add_argument('--n_tasks', type=int, required=True, help='Total number of tasks.')
    args = parser.parse_args()

    msa_dir = "/SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30_filtered"
    profam_db_file = "/SAN/orengolab/cath_plm/ProFam/data/profam.db"
    index_file_path = f"sequence_db_{args.task_index}.index"
    db_save_path = "/SAN/orengolab/cath_plm/ProFam/data/openfold/uniclust30db"
    os.makedirs(db_save_path, exist_ok=True)
    db_file_prefix = f"{db_save_path}/sequence_db_"

    create_sequence_databases(msa_dir, profam_db_file, index_file_path, db_file_prefix, args.task_index, args.n_tasks)