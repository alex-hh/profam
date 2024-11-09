import os
import sys
import uuid
import shutil
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def get_parquet_files_and_row_counts(input_dir):
    parquet_files = []
    row_counts = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.parquet'):
            filepath = os.path.join(input_dir, filename)
            parquet_files.append(filepath)
            parquet_file = pq.ParquetFile(filepath)
            num_rows = parquet_file.metadata.num_rows
            row_counts.append(num_rows)
    return parquet_files, row_counts

def write_buffer_to_temp_file(buffer, temp_dir):
    # Concatenate batches in buffer
    table = pa.Table.from_batches(buffer)
    # Write to temp file
    temp_file = os.path.join(temp_dir, f'temp_{uuid.uuid4()}.parquet')
    pq.write_table(table, temp_file)
    return temp_file

def shuffle_and_write_output_file(temp_files, output_file):
    # Read data from temp files
    tables = []
    for temp_file in temp_files:
        table = pq.read_table(temp_file)
        tables.append(table)
    combined_table = pa.concat_tables(tables)
    # Shuffle rows
    num_rows = combined_table.num_rows
    indices = np.random.permutation(num_rows)
    shuffled_table = combined_table.take(pa.array(indices))
    # Write to final output file
    pq.write_table(shuffled_table, output_file)

def randomize_parquet_files(input_dir, output_dir):
    temp_dir = tempfile.mkdtemp()
    try:
        # Step 1: Get list of input parquet files and their row counts
        parquet_files, row_counts = get_parquet_files_and_row_counts(input_dir)
        num_input_files = len(parquet_files)
        num_output_files = num_input_files  # Can be adjusted as needed

        # Prepare data structures for output data
        output_buffers = [[] for _ in range(num_output_files)]  # Buffers to hold data before writing
        output_temp_files = [[] for _ in range(num_output_files)]  # Lists of temp files per output file
        buffer_limit = 50000  # Adjust based on memory constraints

        # Step 2: Process each input file
        for input_file in parquet_files:
            # Read input file in batches
            parquet_file = pq.ParquetFile(input_file)
            schema = parquet_file.schema_arrow
            for batch in parquet_file.iter_batches():
                num_rows_in_batch = batch.num_rows
                # Randomly assign each row to an output file
                assignments = np.random.randint(0, num_output_files, size=num_rows_in_batch)
                # Collect rows for each output file
                for i in range(num_output_files):
                    indices = np.where(assignments == i)[0]
                    if len(indices) > 0:
                        selected_rows = batch.take(pa.array(indices))
                        output_buffers[i].append(selected_rows)
                        # Write buffer to temp file if limit is reached
                        total_rows = sum(buf.num_rows for buf in output_buffers[i])
                        if total_rows >= buffer_limit:
                            temp_file = write_buffer_to_temp_file(output_buffers[i], temp_dir)
                            output_temp_files[i].append(temp_file)
                            output_buffers[i] = []

        # Write any remaining data to temp files
        for i in range(num_output_files):
            if output_buffers[i]:
                temp_file = write_buffer_to_temp_file(output_buffers[i], temp_dir)
                output_temp_files[i].append(temp_file)
                output_buffers[i] = []

        # Step 3: Shuffle and write final output files
        for i in range(num_output_files):
            temp_files = output_temp_files[i]
            final_output_file = os.path.join(output_dir, f'randomized_{i}.parquet')
            shuffle_and_write_output_file(temp_files, final_output_file)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python randomize_parquet.py <input_dir> <output_dir>')
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    randomize_parquet_files(input_dir, output_dir)
