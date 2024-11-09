import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import mmap
import json
from typing import List, Dict, Optional
import numpy as np


class FastaIndex:
    """Index for fast random access to FASTA files."""

    def __init__(self, fasta_path: str):
        self.fasta_path = Path(fasta_path)
        self.index_path = self.fasta_path.with_suffix('.idx')
        self._index: Dict[str, Dict[str, int]] = {}

        if not self.index_path.exists():
            self._build_index()
        else:
            self._load_index()

    def _build_index(self):
        """Build index of sequence positions in FASTA file."""
        index = {}
        current_pos = 0

        with open(self.fasta_path, 'rb') as f:
            for line in f:
                line = line.decode('utf-8')
                if line.startswith('>'):
                    header = line[1:].strip()
                    index[header] = {
                        'start': current_pos + len(line),
                        'header_start': current_pos
                    }
                elif line.strip() and header in index:
                    if 'length' not in index[header]:
                        index[header]['length'] = 0
                    index[header]['length'] += len(line.strip())
                current_pos += len(line)

        with open(self.index_path, 'w') as f:
            json.dump(index, f)

        self._index = index

    def _load_index(self):
        """Load pre-built index from file."""
        with open(self.index_path, 'r') as f:
            self._index = json.load(f)

    def get_sequence(self, header: str) -> str:
        """Retrieve a sequence by its header."""
        if header not in self._index:
            raise KeyError(f"Sequence {header} not found in index")

        entry = self._index[header]
        with open(self.fasta_path, 'rb') as f:
            f.seek(entry['start'])
            sequence = []
            remaining = entry['length']

            while remaining > 0:
                line = f.readline().decode('utf-8').strip()
                if not line or line.startswith('>'):
                    break
                sequence.append(line)
                remaining -= len(line)

            return ''.join(sequence)

    def __len__(self):
        return len(self._index)

    def get_headers(self) -> List[str]:
        """Get list of all sequence headers."""
        return list(self._index.keys())


class ProteinDataset(Dataset):
    """Memory-efficient protein sequence dataset."""

    def __init__(
        self,
        fasta_path: str,
        tokenizer,
        max_length: int = 512,
        return_tensors: bool = True
    ):
        self.fasta_index = FastaIndex(fasta_path)
        self.headers = self.fasta_index.get_headers()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

        # Memory map the file for faster access
        self.fasta_file = open(fasta_path, 'rb')
        self.mm = mmap.mmap(self.fasta_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.headers)

    def __getitem__(self, idx: int):
        print(idx)
        header = self.headers[idx]
        sequence = self.fasta_index.get_sequence(header)
        return {
            'header': header,
            'sequence': sequence,
        }


    def __del__(self):
        """Cleanup memory-mapped file."""
        if hasattr(self, 'mm'):
            self.mm.close()
        if hasattr(self, 'fasta_file'):
            self.fasta_file.close()

if __name__=="__main__":
    from src.utils.tokenizers import ProFamTokenizer
    import time
    path_to_fasta = "../data/uniprot/uniprot_sprot.fasta"
    tokenizer = ProFamTokenizer(tokenizer_file="src/data/components/profam_tokenizer.json")
    dataset = ProteinDataset(path_to_fasta, tokenizer=tokenizer, max_length=512, return_tensors=True)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=True)
    start = time.time()
    tokens = 0
    for batch in dataloader:
        bp=1
        tokens += sum([len(s) for s in batch['sequence']])
        print(batch)
        bp=1
    end = time.time()
    toks_per_sec = tokens / (end-start)
    print(f"Processed {tokens} tokens in {end-start:.2f} seconds ({toks_per_sec:.2f} tokens/sec)")