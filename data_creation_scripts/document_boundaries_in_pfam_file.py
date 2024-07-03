import multiprocessing as mp
import pandas as pd
from itertools import islice

def find_document_boundaries(chunk, start_line):
    boundaries = []
    current_type = None
    doc_start = None
    
    for i, line in enumerate(chunk, start=start_line):
        if line.startswith("# STOCKHOLM 1.0"):
            doc_start = i
        elif line.startswith("TP   "):
            current_type = line.strip().split()[-1]
        elif line.strip() == "//" and doc_start is not None:
            if current_type in ["Family", "Domain"] or current_type is None:
                boundaries.append((doc_start, i, current_type))
                print(start, end, current_type)
            doc_start = None
            current_type = None
    boundaries.append((doc_start, i, current_type))
    return boundaries

def process_chunk(args):
    chunk, start_line = args
    return find_document_boundaries(chunk, start_line)

def parallel_find_boundaries(filename, chunk_size=1000000, num_processes=64):
    with open(filename, 'r') as f:
        pool = mp.Pool(num_processes)
        
        results = []
        start_line = 0
        
        while True:
            chunk = list(islice(f, chunk_size))
            if not chunk:
                break
            
            results.extend(pool.apply_async(process_chunk, [(chunk, start_line)]))
            start_line += len(chunk)
        
        pool.close()
        pool.join()
        
    boundaries = []
    for result in results:
        boundaries.extend(result.get())
    
    return sorted(boundaries)

# Usage
filename = "/state/patition1/ProFam_data/pfam/Pfam-A.full"
document_boundaries = parallel_find_boundaries(filename)

# Print the results
for start, end, doc_type in document_boundaries:
    print(f"Document of type {doc_type} found from line {start} to {line} end")

df = pd.DataFrame(document_boundaries, columns=["start", "end", "type"])
df.to_csv("pfam_document_boundaries.csv", index=False)